from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass
from typing import Any
from urllib.parse import parse_qs, urlparse

from modules.internet.web.source_ranker import RankedSource
from modules.internet.web.web_models import QueryClassification, WebEvidence, WebEvidencePack, WebPolicyDecision, WebSearchMode


@dataclass(frozen=True)
class _ConflictResult:
    has_conflict: bool
    notes: list[str]
    flags_by_url: dict[str, list[str]]


_NUM_RE = re.compile(r"\b\d+(?:[\.,]\d+)?\b")


def build_evidence_pack(
    *,
    ranked_results: list[RankedSource],
    fetched_pages: dict[str, dict[str, Any]] | None,
    decision: WebPolicyDecision,
    classification: QueryClassification,
) -> WebEvidencePack:
    pages = dict(fetched_pages or {})
    max_sources = max(1, int(decision.budget.max_sources or 1))

    deduped = _dedupe_ranked_results(ranked_results)
    selected: list[WebEvidence] = []
    for row in list(deduped):
        if len(selected) >= max_sources:
            break
        item = row.item
        page = dict(pages.get(str(item.url or "").strip()) or {})
        quality = _quality_score(row=row, page=page)
        threshold = _quality_threshold(decision.mode)
        if quality < threshold:
            continue
        text = str(page.get("text") or "").strip()
        snippet = str(item.snippet or "").strip()
        source_type = "scrape" if text else "search_snippet"
        fetched_at = str(page.get("fetched_at") or "").strip()
        published_date = str(item.published_date or page.get("published_date") or "").strip()
        freshness_tag = _freshness_tag(
            published_date=published_date,
            fetched_at=fetched_at,
            classification=classification,
        )
        key_facts = _extract_key_facts(text=text, snippet=snippet, max_items=3)
        confidence_hint = _confidence_hint(
            trust=row.trust_score,
            quality=quality,
            freshness_tag=freshness_tag,
        )
        selected.append(
            WebEvidence(
                title=str(item.title or "").strip(),
                url=str(item.url or "").strip(),
                domain=str(item.source or "").strip().lower(),
                snippet=snippet,
                text=text,
                published_date=published_date,
                fetched_at=fetched_at,
                trust_score=float(row.trust_score),
                trust_tier=str(row.trust_tier),
                source_type=source_type,
                clean_method=str(page.get("clean_method") or ("search_snippet_fallback" if not text else "")),
                quality_score=float(quality),
                confidence_hint=confidence_hint,
                freshness_tag=freshness_tag,
                key_facts=key_facts,
                conflict_flags=[],
            )
        )

    # Do not return an empty pack if we have any ranked results. Keep best source as fallback.
    if not selected and deduped:
        best = deduped[0]
        best_page = dict(pages.get(str(best.item.url or "").strip()) or {})
        selected = [
            WebEvidence(
                title=str(best.item.title or "").strip(),
                url=str(best.item.url or "").strip(),
                domain=str(best.item.source or "").strip().lower(),
                snippet=str(best.item.snippet or "").strip(),
                text=str(best_page.get("text") or "").strip(),
                published_date=str(best.item.published_date or "").strip(),
                fetched_at=str(best_page.get("fetched_at") or "").strip(),
                trust_score=float(best.trust_score),
                trust_tier=str(best.trust_tier),
                source_type="scrape" if str(best_page.get("text") or "").strip() else "search_snippet",
                clean_method=str(best_page.get("clean_method") or "search_snippet_fallback"),
                quality_score=max(0.0, float(best.quality_score)),
                confidence_hint="low",
                freshness_tag="unknown",
                key_facts=_extract_key_facts(
                    text=str(best_page.get("text") or ""),
                    snippet=str(best.item.snippet or ""),
                    max_items=2,
                ),
                conflict_flags=[],
            )
        ]

    conflict = _detect_conflicts(selected)
    if conflict.has_conflict:
        updated: list[WebEvidence] = []
        for item in list(selected):
            url = str(item.url or "").strip()
            flags = list(conflict.flags_by_url.get(url) or [])
            updated.append(
                WebEvidence(
                    title=item.title,
                    url=item.url,
                    domain=item.domain,
                    snippet=item.snippet,
                    text=item.text,
                    published_date=item.published_date,
                    fetched_at=item.fetched_at,
                    trust_score=item.trust_score,
                    trust_tier=item.trust_tier,
                    source_type=item.source_type,
                    clean_method=item.clean_method,
                    quality_score=item.quality_score,
                    confidence_hint=item.confidence_hint,
                    freshness_tag=item.freshness_tag,
                    key_facts=list(item.key_facts or []),
                    conflict_flags=flags,
                )
            )
        selected = updated

    compact_citations = _compact_citations(items=selected, classification=classification, decision=decision)
    key_facts = _collect_pack_key_facts(selected, max_items=8)
    trust_hints = _build_trust_hints(selected)
    freshness_summary = _freshness_summary(selected)
    summary = _summarize_items(selected)

    return WebEvidencePack(
        items=selected,
        compact_citations=compact_citations,
        conflicting_sources=bool(conflict.has_conflict),
        summary=summary,
        key_facts=key_facts,
        trust_hints=trust_hints,
        freshness_summary=freshness_summary,
        conflict_notes=list(conflict.notes or []),
    )


def _dedupe_ranked_results(items: list[RankedSource]) -> list[RankedSource]:
    seen_url: set[str] = set()
    seen_title_domain: set[str] = set()
    out: list[RankedSource] = []
    for row in list(items or []):
        url = _normalize_url(str(row.item.url or "").strip())
        domain = str(row.item.source or "").strip().lower()
        title = str(row.item.title or "").strip().lower()
        if not url:
            continue
        key_1 = url
        key_2 = f"{domain}|{title}"
        if key_1 in seen_url or key_2 in seen_title_domain:
            continue
        seen_url.add(key_1)
        seen_title_domain.add(key_2)
        out.append(row)
    return out


def _normalize_url(url: str) -> str:
    src = str(url or "").strip()
    if not src:
        return ""
    parsed = urlparse(src)
    query = parse_qs(parsed.query)
    for key in list(query.keys()):
        if key.lower().startswith("utm_"):
            query.pop(key, None)
    normalized_query = "&".join(
        f"{k}={v[0]}"
        for k, v in sorted(query.items(), key=lambda x: str(x[0]))
        if v
    )
    base = f"{parsed.scheme}://{parsed.netloc}{parsed.path}".rstrip("/")
    if normalized_query:
        return base + "?" + normalized_query
    return base


def _quality_score(*, row: RankedSource, page: dict[str, Any]) -> float:
    text = str(page.get("text") or "").strip()
    snippet = str(row.item.snippet or "").strip()
    lexical = max(0.0, min(1.0, float(row.item.score or 0.0)))
    trust = max(0.0, min(1.0, float(row.trust_score)))
    content_len_score = min(1.0, float(len(text) if text else len(snippet)) / 1800.0)
    return max(0.0, min(1.0, (0.38 * lexical) + (0.34 * trust) + (0.28 * content_len_score)))


def _quality_threshold(mode: WebSearchMode) -> float:
    if mode == WebSearchMode.VERIFY_ONLY:
        return 0.22
    if mode == WebSearchMode.SOFT_SEARCH:
        return 0.19
    if mode == WebSearchMode.TARGETED_SEARCH:
        return 0.16
    if mode == WebSearchMode.DEEP_SEARCH:
        return 0.13
    return 0.20


def _freshness_tag(
    *,
    published_date: str,
    fetched_at: str,
    classification: QueryClassification,
) -> str:
    day = _parse_any_day(published_date) or _parse_any_day(fetched_at)
    if day is None:
        return "unknown"
    age_days = max(0, (dt.datetime.now(dt.timezone.utc).date() - day).days)
    limit = 7 if classification.requires_freshness else 30
    if age_days <= limit:
        return "fresh"
    if age_days <= (limit * 4):
        return "recent"
    return "stale"


def _parse_any_day(value: str) -> dt.date | None:
    src = str(value or "").strip()
    if not src:
        return None
    try:
        return dt.datetime.fromisoformat(src.replace("Z", "+00:00")).date()
    except Exception:
        pass
    try:
        return dt.datetime.strptime(src, "%Y-%m-%d").date()
    except Exception:
        return None


def _extract_key_facts(*, text: str, snippet: str, max_items: int) -> list[str]:
    source = str(text or "").strip() or str(snippet or "").strip()
    if not source:
        return []
    sentences = _sentences(source)
    out: list[str] = []
    for sentence in sentences:
        s = str(sentence or "").strip()
        if not s:
            continue
        if not _looks_fact_like(s):
            continue
        clean = _trim_sentence(s, max_chars=200)
        if clean and clean not in out:
            out.append(clean)
        if len(out) >= max(1, int(max_items)):
            break
    if not out:
        first = _trim_sentence(sentences[0] if sentences else source, max_chars=180)
        if first:
            out.append(first)
    return out


def _sentences(text: str) -> list[str]:
    raw = re.split(r"(?<=[.!?])\s+", str(text or ""))
    out = [str(x or "").strip() for x in raw if str(x or "").strip()]
    if out:
        return out
    token = str(text or "").strip()
    return [token] if token else []


def _looks_fact_like(sentence: str) -> bool:
    low = str(sentence or "").strip().lower()
    if not low:
        return False
    if _NUM_RE.search(low):
        return True
    markers = ("version", "release", "price", "cost", "announce", "updated", "supports")
    return any(marker in low for marker in markers)


def _trim_sentence(text: str, *, max_chars: int) -> str:
    src = str(text or "").strip()
    if len(src) <= max(24, int(max_chars)):
        return src
    clipped = src[: max_chars].rsplit(" ", 1)[0].strip()
    return (clipped or src[: max_chars]).rstrip(".") + "."


def _confidence_hint(*, trust: float, quality: float, freshness_tag: str) -> str:
    if trust >= 0.28 and quality >= 0.48 and freshness_tag in {"fresh", "recent"}:
        return "high"
    if trust >= 0.14 and quality >= 0.30:
        return "medium"
    return "low"


def _compact_citations(
    *,
    items: list[WebEvidence],
    classification: QueryClassification,
    decision: WebPolicyDecision,
) -> list[str]:
    if not items:
        return []
    if classification.query_type == "external_factual" and decision.mode == WebSearchMode.VERIFY_ONLY:
        limit = 1
    elif classification.query_type == "mixed":
        limit = 3
    else:
        limit = 2
    out: list[str] = []
    for item in list(items)[: max(1, int(limit))]:
        stamp = str(item.published_date or item.fetched_at or "").strip()
        trust = str(item.trust_tier or "").strip()
        if stamp and trust:
            out.append(f"{item.domain} ({stamp}, {trust})")
        elif stamp:
            out.append(f"{item.domain} ({stamp})")
        else:
            out.append(str(item.domain or "").strip())
    return out


def _collect_pack_key_facts(items: list[WebEvidence], *, max_items: int) -> list[str]:
    out: list[str] = []
    for item in list(items or []):
        for fact in list(item.key_facts or []):
            value = str(fact or "").strip()
            if not value or value in out:
                continue
            out.append(value)
            if len(out) >= max(1, int(max_items)):
                return out
    return out


def _build_trust_hints(items: list[WebEvidence]) -> list[str]:
    out: list[str] = []
    for item in list(items or []):
        line = f"{item.domain}: {item.trust_tier} ({item.trust_score:.2f})"
        if line not in out:
            out.append(line)
    return out


def _freshness_summary(items: list[WebEvidence]) -> str:
    if not items:
        return ""
    counts = {"fresh": 0, "recent": 0, "stale": 0, "unknown": 0}
    for item in list(items or []):
        tag = str(item.freshness_tag or "unknown").strip().lower()
        if tag not in counts:
            tag = "unknown"
        counts[tag] += 1
    return (
        f"fresh={counts['fresh']}, "
        f"recent={counts['recent']}, "
        f"stale={counts['stale']}, "
        f"unknown={counts['unknown']}"
    )


def _summarize_items(items: list[WebEvidence]) -> str:
    if not items:
        return ""
    domains: list[str] = []
    for item in list(items or []):
        domain = str(item.domain or "").strip()
        if domain and domain not in domains:
            domains.append(domain)
    if not domains:
        return ""
    if len(domains) == 1:
        return f"web evidence from {domains[0]}"
    return f"web evidence from {', '.join(domains[:3])}"


def _detect_conflicts(items: list[WebEvidence]) -> _ConflictResult:
    by_domain: dict[str, list[float]] = {}
    flags: dict[str, list[str]] = {}
    for item in list(items or []):
        blob = " ".join(list(item.key_facts or [])) or f"{item.snippet} {item.text}"
        values = _extract_numbers(blob)
        if not values:
            continue
        domain = str(item.domain or "").strip().lower() or "unknown"
        by_domain.setdefault(domain, []).extend(values)

    if len(by_domain) < 2:
        return _ConflictResult(has_conflict=False, notes=[], flags_by_url={})

    representative: list[tuple[str, float]] = []
    for domain, values in by_domain.items():
        if not values:
            continue
        representative.append((domain, float(sum(values) / max(1, len(values)))))
    if len(representative) < 2:
        return _ConflictResult(has_conflict=False, notes=[], flags_by_url={})

    representative.sort(key=lambda x: x[1])
    low_domain, low_val = representative[0]
    high_domain, high_val = representative[-1]
    if low_val <= 0:
        return _ConflictResult(has_conflict=False, notes=[], flags_by_url={})
    spread = (high_val - low_val) / low_val
    if spread <= 0.25:
        return _ConflictResult(has_conflict=False, notes=[], flags_by_url={})

    note = f"numeric_conflict:{low_domain}={low_val:.3f} vs {high_domain}={high_val:.3f}"
    for item in list(items or []):
        if str(item.domain or "").strip().lower() in {low_domain, high_domain}:
            flags.setdefault(str(item.url or "").strip(), []).append("numeric_spread")
    return _ConflictResult(has_conflict=True, notes=[note], flags_by_url=flags)


def _extract_numbers(text: str) -> list[float]:
    out: list[float] = []
    for token in _NUM_RE.findall(str(text or "")):
        value = str(token or "").strip().replace(",", ".")
        if not value:
            continue
        try:
            num = float(value)
        except Exception:
            continue
        if num <= 0:
            continue
        out.append(num)
    return out


def format_postprocess_citation_suffix(
    *,
    text: str,
    compact_citations: list[str] | None,
    classification: QueryClassification | None,
) -> str:
    src = str(text or "").rstrip()
    if not src:
        return src
    citations = [str(x or "").strip() for x in list(compact_citations or []) if str(x or "").strip()]
    if not citations:
        return src
    low = src.lower()
    if "источник:" in low or "источники:" in low or "sources:" in low:
        return src

    if classification is not None and classification.query_type == "external_factual":
        suffix = f"Source: {citations[0]}."
    elif len(citations) == 1:
        suffix = f"Sources: {citations[0]}."
    else:
        suffix = "Sources: " + "; ".join(citations[:3]) + "."

    if src.endswith((".", "!", "?")):
        return src + "\n\n" + suffix
    return src + ".\n\n" + suffix


def make_iso_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()
