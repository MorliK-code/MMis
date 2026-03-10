from __future__ import annotations

import datetime as dt
from typing import Any

from modules.internet.search import SearchResult
from modules.internet.web.source_ranker import RankedSource
from modules.internet.web.web_models import QueryClassification, WebEvidenceItem, WebEvidencePack, WebPolicyDecision


def build_evidence_pack(
    *,
    ranked_results: list[RankedSource],
    fetched_pages: dict[str, dict[str, Any]] | None,
    decision: WebPolicyDecision,
    classification: QueryClassification,
) -> WebEvidencePack:
    pages = dict(fetched_pages or {})
    max_sources = max(1, int(decision.budget.max_sources or 1))
    deduped = _dedupe_ranked_results(ranked_results)[:max_sources]

    items: list[WebEvidenceItem] = []
    for row in deduped:
        item = row.item
        page = dict(pages.get(str(item.url or "").strip()) or {})
        text = str(page.get("text") or "").strip()
        fetched_at = str(page.get("fetched_at") or "").strip()
        source_type = "scrape" if text else "search_snippet"
        items.append(
            WebEvidenceItem(
                title=str(item.title or "").strip(),
                url=str(item.url or "").strip(),
                domain=str(item.source or "").strip().lower(),
                snippet=str(item.snippet or "").strip(),
                text=text,
                published_date=str(item.published_date or "").strip(),
                fetched_at=fetched_at,
                trust_score=float(row.trust_score),
                trust_tier=str(row.trust_tier),
                source_type=source_type,
                clean_method=str(page.get("clean_method") or ("search_snippet_fallback" if not text else "")),
            )
        )

    citations = _compact_citations(
        items=items,
        classification=classification,
        decision=decision,
    )
    summary = _summarize_items(items)
    conflicting = _has_conflicting_sources(items)
    return WebEvidencePack(
        items=items,
        compact_citations=citations,
        conflicting_sources=conflicting,
        summary=summary,
    )


def _dedupe_ranked_results(items: list[RankedSource]) -> list[RankedSource]:
    seen_url: set[str] = set()
    seen_title_domain: set[str] = set()
    out: list[RankedSource] = []
    for row in list(items or []):
        url = str(row.item.url or "").strip().lower()
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


def _compact_citations(
    *,
    items: list[WebEvidenceItem],
    classification: QueryClassification,
    decision: WebPolicyDecision,
) -> list[str]:
    if not items:
        return []
    if classification.query_type == "external_factual" and decision.mode.value in {"VERIFY_ONLY", "SOFT_SEARCH"}:
        limit = 1
    elif classification.query_type == "mixed":
        limit = 3
    else:
        limit = 2

    out: list[str] = []
    for item in list(items)[: max(1, int(limit))]:
        stamp = str(item.published_date or item.fetched_at or "").strip()
        if stamp:
            line = f"{item.domain} ({stamp})"
        else:
            line = str(item.domain or "").strip()
        if line:
            out.append(line)
    return out


def _summarize_items(items: list[WebEvidenceItem]) -> str:
    if not items:
        return ""
    domains: list[str] = []
    for item in list(items or []):
        domain = str(item.domain or "").strip()
        if not domain:
            continue
        if domain not in domains:
            domains.append(domain)
    if not domains:
        return ""
    if len(domains) == 1:
        return f"web evidence from {domains[0]}"
    return f"web evidence from {', '.join(domains[:3])}"


def _has_conflicting_sources(items: list[WebEvidenceItem]) -> bool:
    # Conservative signal: different domains with highly different numeric snippets.
    values: list[float] = []
    for item in list(items or []):
        text = f"{item.snippet} {item.text}"
        for token in str(text or "").replace(",", ".").split():
            try:
                value = float(token)
            except Exception:
                continue
            if value <= 0:
                continue
            values.append(value)
    if len(values) < 2:
        return False
    low = min(values)
    high = max(values)
    if low <= 0:
        return False
    # Avoid noisy conflicts; mark only large spread.
    return bool((high - low) / low > 0.25)


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
        suffix = f"Источник: {citations[0]}."
    elif len(citations) == 1:
        suffix = f"Sources: {citations[0]}."
    else:
        suffix = "Sources: " + "; ".join(citations[:3]) + "."

    if src.endswith((".", "!", "?")):
        return src + "\n\n" + suffix
    return src + ".\n\n" + suffix


def make_iso_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()
