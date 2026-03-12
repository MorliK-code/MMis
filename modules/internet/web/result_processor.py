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


@dataclass(frozen=True)
class _QualityAssessment:
    score: float
    breakdown: dict[str, float]
    signals: dict[str, Any]


_NUM_RE = re.compile(r"\b\d+(?:[\.,]\d+)?\b")
_DATE_RE = re.compile(r"\b(?:\d{4}-\d{2}-\d{2}|\d{1,2}[./-]\d{1,2}[./-]\d{2,4}|20\d{2}|19\d{2})\b")
_WORD_RE = re.compile(r"[A-Za-z\u0400-\u04ff0-9_]{3,}")
_LOW_INFO_TITLE_RE = re.compile(
    r"\b(home|homepage|untitled|index|read more|click here|post|article|blog|page)\b",
    flags=re.I,
)
_SEO_PATTERN_RE = re.compile(
    r"\b(best|top\s+\d+|buy now|shop now|coupon|promo|deal|cheap|free|affiliate|sponsored|casino|bet)\b",
    flags=re.I,
)
_FACT_MARKERS = (
    "version",
    "release",
    "price",
    "cost",
    "updated",
    "supports",
    "announce",
    "official",
    "rate",
    "temperature",
    "\u0432\u0435\u0440\u0441",
    "\u0440\u0435\u043b\u0438\u0437",
    "\u0446\u0435\u043d",
    "\u043a\u0443\u0440\u0441",
    "\u043e\u0431\u043d\u043e\u0432",
    "\u0434\u0430\u0442",
)


def build_evidence_pack(
    *,
    ranked_results: list[RankedSource],
    fetched_pages: dict[str, dict[str, Any]] | None,
    decision: WebPolicyDecision,
    classification: QueryClassification,
) -> WebEvidencePack:
    pages = dict(fetched_pages or {})
    max_sources = max(1, int(decision.budget.max_sources or 1))
    threshold = _quality_threshold(decision.mode)
    candidate_rows = _prepare_candidate_rows(
        ranked_results=ranked_results,
        pages=pages,
        threshold=threshold,
        max_sources=max_sources,
        classification=classification,
    )
    selected = [state["evidence"] for state in candidate_rows if isinstance(state.get("evidence"), WebEvidence)]
    source_audit = [dict(state.get("audit") or {}) for state in candidate_rows]

    # Do not return an empty pack if we have any ranked results. Keep best source as fallback.
    if not selected and candidate_rows:
        fallback_state = candidate_rows[0]
        fallback_audit = dict(fallback_state.get("audit") or {})
        fallback_audit["selected_for_evidence"] = True
        fallback_audit["filtered_out_reason"] = ""
        fallback_audit["selection_reason"] = "fallback_best_available"
        fallback_audit["quality_score"] = max(0.0, float(fallback_state.get("quality") or 0.0))
        fallback_audit["evidence_quality_score"] = max(0.0, float(fallback_state.get("quality") or 0.0))
        selected = [
            _evidence_from_candidate(
                fallback_state,
                audit=fallback_audit,
                confidence_hint_override="low",
            )
        ]
        source_audit[0] = dict(fallback_audit)

    conflict = _detect_conflicts(selected)
    if conflict.has_conflict:
        updated: list[WebEvidence] = []
        for item in list(selected):
            url = str(item.url or "").strip()
            flags = list(conflict.flags_by_url.get(url) or [])
            audit = dict(item.audit or {})
            audit["conflict_flags"] = list(flags)
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
                    audit=audit,
                )
            )
        selected = updated
        for index, row in enumerate(list(source_audit)):
            url = str(dict(row or {}).get("url") or "").strip()
            if url:
                source_audit[index]["conflict_flags"] = list(conflict.flags_by_url.get(url) or [])

    compact_citations = _compact_citations(items=selected, classification=classification, decision=decision)
    key_facts = _collect_pack_key_facts(selected, max_items=8)
    trust_hints = _build_trust_hints(selected)
    freshness_summary = _freshness_summary(selected)
    summary = _summarize_items(selected)
    selection_summary = _build_selection_summary(candidate_rows=candidate_rows, selected=selected, threshold=threshold)

    return WebEvidencePack(
        items=selected,
        compact_citations=compact_citations,
        conflicting_sources=bool(conflict.has_conflict),
        summary=summary,
        key_facts=key_facts,
        trust_hints=trust_hints,
        freshness_summary=freshness_summary,
        conflict_notes=list(conflict.notes or []),
        source_audit=source_audit,
        selection_summary=selection_summary,
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


def _prepare_candidate_rows(
    *,
    ranked_results: list[RankedSource],
    pages: dict[str, dict[str, Any]],
    threshold: float,
    max_sources: int,
    classification: QueryClassification,
) -> list[dict[str, Any]]:
    seen_url: set[str] = set()
    seen_title_domain: set[str] = set()
    out: list[dict[str, Any]] = []

    for rank_position, row in enumerate(list(ranked_results or []), start=1):
        item = row.item
        page = dict(pages.get(str(item.url or "").strip()) or {})
        url = _normalize_url(str(item.url or "").strip())
        domain = str(item.source or "").strip().lower()
        title = str(item.title or "").strip().lower()
        key_2 = f"{domain}|{title}"
        audit = _seed_audit(row=row, rank_position=rank_position)
        audit["url"] = str(item.url or "").strip()
        audit["domain"] = domain

        if not url:
            audit["filtered_out_reason"] = "missing_url"
            out.append({"row": row, "page": page, "audit": audit, "evidence": None, "quality": 0.0})
            continue
        if url in seen_url or key_2 in seen_title_domain:
            audit["filtered_out_reason"] = "duplicate_source"
            out.append({"row": row, "page": page, "audit": audit, "evidence": None, "quality": 0.0})
            continue

        seen_url.add(url)
        seen_title_domain.add(key_2)
        candidate = _candidate_payload(row=row, page=page, classification=classification)
        quality = float(candidate["quality"])
        audit["quality_score"] = quality
        audit["evidence_quality_score"] = quality
        audit["freshness_tag"] = str(candidate["freshness_tag"])
        audit["freshness_score"] = float(audit.get("freshness_score") or 0.0)
        audit["quality_breakdown"] = dict(candidate.get("quality_breakdown") or {})
        audit["quality_signals"] = dict(candidate.get("quality_signals") or {})
        out.append({"row": row, "page": page, "audit": audit, "evidence": None, "quality": quality, **candidate})

    _apply_cross_source_quality_adjustments(out)

    selectable: list[dict[str, Any]] = []
    for state in out:
        audit = dict(state.get("audit") or {})
        if str(audit.get("filtered_out_reason") or "").strip():
            state["audit"] = audit
            continue

        quality = max(0.0, min(1.0, float(state.get("quality") or 0.0)))
        state["quality"] = quality
        audit["quality_score"] = quality
        audit["evidence_quality_score"] = quality
        state["audit"] = audit
        if quality >= threshold:
            selectable.append(state)

    selected_states: set[int] = set()
    for state in sorted(
        selectable,
        key=lambda row: (
            -float(row.get("quality") or 0.0),
            int(_as_dict(row.get("audit")).get("rank_position") or 10_000),
        ),
    )[: max_sources]:
        selected_states.add(id(state))

    for state in out:
        audit = dict(state.get("audit") or {})
        if str(audit.get("filtered_out_reason") or "").strip():
            state["audit"] = audit
            continue
        if float(state.get("quality") or 0.0) < threshold:
            audit["filtered_out_reason"] = "quality_below_threshold"
            state["audit"] = audit
            continue
        if id(state) not in selected_states:
            audit["filtered_out_reason"] = "evidence_budget_cap"
            state["audit"] = audit
            continue

        audit["selected_for_evidence"] = True
        audit["selection_reason"] = "passed_quality_threshold"
        audit["filtered_out_reason"] = ""
        state["audit"] = audit
        state["evidence"] = _evidence_from_candidate(state, audit=audit)

    return out


def _seed_audit(*, row: RankedSource, rank_position: int) -> dict[str, Any]:
    base = dict(getattr(row, "audit", {}) or dict(getattr(row.item, "raw", {}) or {}).get("v2_source_audit") or {})
    ranking_quality = float(base.get("quality_score") or getattr(row, "quality_score", 0.0) or 0.0)
    freshness_score = float(base.get("freshness_score") or getattr(row, "freshness_score", 0.0) or 0.0)
    base["trust_tier"] = str(base.get("trust_tier") or getattr(row, "trust_tier", "") or "unknown")
    base["trust_score"] = float(base.get("trust_score") or getattr(row, "trust_score", 0.0) or 0.0)
    base["title"] = str(base.get("title") or getattr(row.item, "title", "") or "")
    base["domain"] = str(base.get("domain") or getattr(row.item, "source", "") or "").strip().lower()
    base["ranking_quality_score"] = float(base.get("ranking_quality_score") or ranking_quality)
    base["quality_score"] = float(base.get("quality_score") or ranking_quality)
    base["ranking_freshness_score"] = float(base.get("freshness_score") or freshness_score)
    base["freshness_score"] = freshness_score
    base["selected_for_evidence"] = False
    base.setdefault("filtered_out_reason", "")
    base.setdefault("selection_reason", "")
    base["rank_position"] = int(rank_position)
    return base


def _candidate_payload(
    *,
    row: RankedSource,
    page: dict[str, Any],
    classification: QueryClassification,
) -> dict[str, Any]:
    item = row.item
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
    quality_assessment = _quality_score(
        row=row,
        page=page,
        classification=classification,
        key_facts=key_facts,
    )
    confidence_hint = _confidence_hint(
        trust=row.trust_score,
        quality=quality_assessment.score,
        freshness_tag=freshness_tag,
    )
    return {
        "row": row,
        "text": text,
        "snippet": snippet,
        "source_type": source_type,
        "fetched_at": fetched_at,
        "published_date": published_date,
        "freshness_tag": freshness_tag,
        "quality": quality_assessment.score,
        "quality_breakdown": dict(quality_assessment.breakdown or {}),
        "quality_signals": dict(quality_assessment.signals or {}),
        "key_facts": key_facts,
        "confidence_hint": confidence_hint,
        "clean_method": str(page.get("clean_method") or ("search_snippet_fallback" if not text else "")),
    }


def _evidence_from_candidate(
    candidate: dict[str, Any],
    *,
    audit: dict[str, Any],
    confidence_hint_override: str = "",
) -> WebEvidence:
    row = candidate["row"]
    item = row.item
    confidence_hint = str(confidence_hint_override or candidate.get("confidence_hint") or "low")
    return WebEvidence(
        title=str(item.title or "").strip(),
        url=str(item.url or "").strip(),
        domain=str(item.source or "").strip().lower(),
        snippet=str(candidate.get("snippet") or "").strip(),
        text=str(candidate.get("text") or "").strip(),
        published_date=str(candidate.get("published_date") or "").strip(),
        fetched_at=str(candidate.get("fetched_at") or "").strip(),
        trust_score=float(row.trust_score),
        trust_tier=str(row.trust_tier),
        source_type=str(candidate.get("source_type") or "search_snippet"),
        clean_method=str(candidate.get("clean_method") or ""),
        quality_score=float(candidate.get("quality") or 0.0),
        confidence_hint=confidence_hint,
        freshness_tag=str(candidate.get("freshness_tag") or "unknown"),
        key_facts=list(candidate.get("key_facts") or []),
        conflict_flags=[],
        audit=dict(audit or {}),
    )


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


def _quality_score(
    *,
    row: RankedSource,
    page: dict[str, Any],
    classification: QueryClassification,
    key_facts: list[str],
) -> _QualityAssessment:
    text = str(page.get("text") or "").strip()
    snippet = str(row.item.snippet or "").strip()
    title = str(row.item.title or "").strip()
    blob = " ".join(list(key_facts or [])) or text or snippet or title

    ranking_prior = max(0.0, min(1.0, float(row.item.score or 0.0)))
    trust_signal = max(0.0, min(1.0, float(row.trust_score or 0.0)))
    content_presence = _content_presence_score(text=text, snippet=snippet)
    content_depth = _content_depth_score(text=text, snippet=snippet)
    fact_signal, fact_stats = _fact_signal(text=text, snippet=snippet, title=title, key_facts=key_facts)
    title_quality = _title_quality_score(title=title)
    seo_penalty = _seo_penalty(title=title, snippet=snippet, text=text)
    thin_penalty = _thin_penalty(text=text, snippet=snippet, key_facts=key_facts, blob=blob)
    freshness_support = 0.08 if bool(getattr(classification, "requires_freshness", False)) and _parse_any_day(str(row.item.published_date or page.get("published_date") or "")) else 0.0

    breakdown = {
        "ranking_prior": round(0.16 * ranking_prior, 6),
        "trust_signal": round(0.14 * trust_signal, 6),
        "content_presence": round(0.20 * content_presence, 6),
        "content_depth": round(0.17 * content_depth, 6),
        "fact_signal": round(0.16 * fact_signal, 6),
        "title_quality": round(0.09 * title_quality, 6),
        "freshness_support": round(float(freshness_support), 6),
        "consensus_bonus": 0.0,
        "conflict_penalty": 0.0,
        "seo_penalty": round(-0.14 * seo_penalty, 6),
        "thin_penalty": round(-0.14 * thin_penalty, 6),
    }
    raw_score = sum(float(value) for value in breakdown.values())
    score = max(0.0, min(1.0, raw_score))
    signals = {
        "has_full_text": bool(text),
        "body_chars": int(len(text)),
        "snippet_chars": int(len(snippet)),
        "title_chars": int(len(title)),
        "key_fact_count": int(len(list(key_facts or []))),
        "numbers_count": int(fact_stats.get("numbers_count") or 0),
        "dates_count": int(fact_stats.get("dates_count") or 0),
        "fact_marker_count": int(fact_stats.get("fact_marker_count") or 0),
        "fact_like_count": int(fact_stats.get("fact_like_count") or 0),
    }
    return _QualityAssessment(score=score, breakdown=breakdown, signals=signals)


def _content_presence_score(*, text: str, snippet: str) -> float:
    body_len = len(str(text or "").strip())
    snippet_len = len(str(snippet or "").strip())
    if body_len >= 320:
        return 1.0
    if body_len >= 180:
        return 0.86
    if body_len >= 100:
        return 0.68
    if snippet_len >= 160:
        return 0.46
    if snippet_len >= 80:
        return 0.30
    if snippet_len >= 30:
        return 0.18
    return 0.08


def _content_depth_score(*, text: str, snippet: str) -> float:
    body_len = len(str(text or "").strip())
    snippet_len = len(str(snippet or "").strip())
    if body_len > 0:
        return max(0.0, min(1.0, body_len / 1800.0))
    return max(0.0, min(0.45, snippet_len / 420.0))


def _fact_signal(*, text: str, snippet: str, title: str, key_facts: list[str]) -> tuple[float, dict[str, int]]:
    blob = " ".join(list(key_facts or [])) or str(text or "").strip() or str(snippet or "").strip() or str(title or "").strip()
    low = blob.lower()
    numbers_count = len(_extract_numbers(blob))
    dates_count = len(_DATE_RE.findall(blob))
    fact_marker_count = sum(1 for token in _FACT_MARKERS if token in low)
    fact_like_count = sum(1 for sentence in _sentences(str(text or "").strip() or str(snippet or "").strip()) if _looks_fact_like(sentence))

    score = 0.0
    if fact_like_count > 0:
        score += min(0.36, 0.18 * fact_like_count)
    if key_facts and (fact_like_count > 0 or numbers_count > 0 or dates_count > 0 or fact_marker_count > 0):
        score += 0.14
    if numbers_count > 0:
        score += min(0.24, 0.08 * numbers_count)
    if dates_count > 0:
        score += min(0.18, 0.09 * dates_count)
    if fact_marker_count > 0:
        score += min(0.16, 0.06 * fact_marker_count)

    return max(0.0, min(1.0, score)), {
        "numbers_count": int(numbers_count),
        "dates_count": int(dates_count),
        "fact_marker_count": int(fact_marker_count),
        "fact_like_count": int(fact_like_count),
    }


def _title_quality_score(*, title: str) -> float:
    src = str(title or "").strip()
    if not src:
        return 0.0
    length = len(src)
    alpha_tokens = [token for token in _WORD_RE.findall(src) if not token.isdigit()]
    score = 0.2
    if 12 <= length <= 110:
        score += 0.42
    elif 6 <= length <= 140:
        score += 0.24
    if len(alpha_tokens) >= 3:
        score += 0.22
    if any(ch.isdigit() for ch in src):
        score += 0.08
    if _LOW_INFO_TITLE_RE.search(src):
        score -= 0.34
    if _SEO_PATTERN_RE.search(src):
        score -= 0.22
    if src.isupper() and length > 16:
        score -= 0.12
    return max(0.0, min(1.0, score))


def _seo_penalty(*, title: str, snippet: str, text: str) -> float:
    src = " ".join(part for part in (title, snippet) if str(part or "").strip())
    low = src.lower()
    penalty = 0.0
    if _SEO_PATTERN_RE.search(src):
        penalty += 0.64
    if src.count("|") >= 2 or src.count(" - ") >= 3:
        penalty += 0.20
    if any(token in low for token in ("read more", "click here", "learn more", "subscribe", "sign up")):
        penalty += 0.24
    if not str(text or "").strip() and len(str(snippet or "").strip()) < 90:
        penalty += 0.18
    return max(0.0, min(1.0, penalty))


def _thin_penalty(*, text: str, snippet: str, key_facts: list[str], blob: str) -> float:
    body_len = len(str(text or "").strip())
    snippet_len = len(str(snippet or "").strip())
    numbers_count = len(_extract_numbers(blob))
    penalty = 0.0
    if body_len == 0 and snippet_len < 120:
        penalty += 0.48
    elif 0 < body_len < 140:
        penalty += 0.26
    if not key_facts and numbers_count == 0 and max(body_len, snippet_len) < 180:
        penalty += 0.18
    return max(0.0, min(1.0, penalty))


def _apply_cross_source_quality_adjustments(states: list[dict[str, Any]]) -> None:
    comparable = [state for state in list(states or []) if not str(_as_dict(state.get("audit")).get("filtered_out_reason") or "").strip()]
    if len(comparable) < 2:
        return

    prepared: list[dict[str, Any]] = []
    for state in comparable:
        domain = str(_as_dict(state.get("audit")).get("domain") or "").strip().lower()
        blob = " ".join(list(state.get("key_facts") or [])) or str(state.get("text") or state.get("snippet") or "").strip()
        prepared.append(
            {
                "state": state,
                "domain": domain,
                "tokens": _semantic_tokens(blob),
                "numbers": _extract_numbers(blob),
            }
        )

    if len({str(row.get("domain") or "") for row in prepared if str(row.get("domain") or "")}) < 2:
        return

    for current in prepared:
        similarities: list[float] = []
        numeric_conflicts = 0
        independent_matches = 0
        for other in prepared:
            if other is current:
                continue
            if str(other.get("domain") or "") == str(current.get("domain") or ""):
                continue
            sim = _cross_source_similarity(
                left_tokens=set(current.get("tokens") or set()),
                right_tokens=set(other.get("tokens") or set()),
                left_numbers=list(current.get("numbers") or []),
                right_numbers=list(other.get("numbers") or []),
            )
            similarities.append(sim)
            if sim >= 0.34:
                independent_matches += 1
            if _numeric_conflict(left_numbers=list(current.get("numbers") or []), right_numbers=list(other.get("numbers") or [])):
                numeric_conflicts += 1

        audit = _as_dict(current["state"].get("audit"))
        breakdown = dict(audit.get("quality_breakdown") or {})
        quality_signals = dict(audit.get("quality_signals") or {})
        if not similarities:
            audit["quality_breakdown"] = breakdown
            audit["quality_signals"] = quality_signals
            current["state"]["audit"] = audit
            continue

        similarities.sort(reverse=True)
        consensus_score = float(sum(similarities[:2]) / max(1, min(2, len(similarities))))
        consensus_bonus = 0.0
        if consensus_score >= 0.58:
            consensus_bonus = 0.10
        elif consensus_score >= 0.40:
            consensus_bonus = 0.05
        conflict_penalty = min(0.12, 0.06 * numeric_conflicts)

        breakdown["consensus_bonus"] = round(consensus_bonus, 6)
        breakdown["conflict_penalty"] = round(-conflict_penalty, 6)
        adjusted_score = max(0.0, min(1.0, float(sum(float(v) for v in breakdown.values()))))
        current["state"]["quality"] = adjusted_score
        audit["quality_score"] = adjusted_score
        audit["evidence_quality_score"] = adjusted_score
        audit["quality_breakdown"] = breakdown
        quality_signals["consensus_score"] = round(consensus_score, 6)
        quality_signals["independent_matches"] = int(independent_matches)
        quality_signals["numeric_conflicts"] = int(numeric_conflicts)
        audit["quality_signals"] = quality_signals
        current["state"]["audit"] = audit


def _cross_source_similarity(
    *,
    left_tokens: set[str],
    right_tokens: set[str],
    left_numbers: list[float],
    right_numbers: list[float],
) -> float:
    lexical = _jaccard(left_tokens, right_tokens)
    numeric = _numeric_agreement(left_numbers=left_numbers, right_numbers=right_numbers)
    return max(0.0, min(1.0, (0.65 * lexical) + (0.35 * numeric)))


def _semantic_tokens(text: str) -> set[str]:
    out: set[str] = set()
    for token in _WORD_RE.findall(str(text or "").lower()):
        word = str(token or "").strip().lower()
        if len(word) <= 2 or word.isdigit():
            continue
        out.add(word)
    return out


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    union = left | right
    if not union:
        return 0.0
    return float(len(left & right) / max(1, len(union)))


def _numeric_agreement(*, left_numbers: list[float], right_numbers: list[float]) -> float:
    left = [float(x) for x in list(left_numbers or []) if float(x) > 0]
    right = [float(x) for x in list(right_numbers or []) if float(x) > 0]
    if not left or not right:
        return 0.5
    best = 1.0
    for a in left:
        for b in right:
            den = max(1.0, abs(a), abs(b))
            rel = abs(a - b) / den
            if rel < best:
                best = rel
    if best <= 0.08:
        return 1.0
    if best <= 0.18:
        return 0.65
    if best <= 0.30:
        return 0.35
    return 0.0


def _numeric_conflict(*, left_numbers: list[float], right_numbers: list[float]) -> bool:
    return _numeric_agreement(left_numbers=left_numbers, right_numbers=right_numbers) <= 0.1


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


def _build_selection_summary(
    *,
    candidate_rows: list[dict[str, Any]],
    selected: list[WebEvidence],
    threshold: float,
) -> dict[str, Any]:
    states = [dict(x or {}) for x in list(candidate_rows or []) if isinstance(x, dict)]
    audits = [_as_dict(state.get("audit")) for state in states]
    domains_scanned = sorted(
        {
            str(audit.get("domain") or "").strip().lower()
            for audit in audits
            if str(audit.get("domain") or "").strip()
        }
    )
    selected_audits = [audit for audit in audits if bool(audit.get("selected_for_evidence"))]
    filtered_reasons: dict[str, int] = {}
    for audit in audits:
        reason = str(audit.get("filtered_out_reason") or "").strip()
        if not reason:
            continue
        filtered_reasons[reason] = int(filtered_reasons.get(reason, 0)) + 1

    selected_qualities = [float(audit.get("quality_score") or 0.0) for audit in selected_audits]
    return {
        "candidates_total": int(len(states)),
        "sources_scanned": int(sum(1 for audit in audits if str(audit.get("url") or "").strip())),
        "unique_domains_scanned": int(len(domains_scanned)),
        "domains_scanned": list(domains_scanned),
        "selected_sources": int(len(selected_audits)),
        "selected_domains": sorted(
            {
                str(item.domain or "").strip().lower()
                for item in list(selected or [])
                if str(item.domain or "").strip()
            }
        ),
        "filtered_sources": int(sum(filtered_reasons.values())),
        "filtered_reasons": dict(filtered_reasons),
        "evidence_count": int(len(list(selected or []))),
        "quality_threshold": float(max(0.0, min(1.0, threshold))),
        "selected_avg_quality": round(
            float(sum(selected_qualities) / max(1, len(selected_qualities))) if selected_qualities else 0.0,
            6,
        ),
        "best_selected_quality": round(float(max(selected_qualities)) if selected_qualities else 0.0, 6),
        "conflicting_sources": bool(any(bool(item.conflict_flags) for item in list(selected or []))),
    }


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


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


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
