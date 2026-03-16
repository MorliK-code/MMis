from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import parse_qs, urlparse

from modules.internet.web.numeric_facts import (
    candidate_group_key,
    conflict_severity_for_values,
    currency_page_type_priority,
    detect_currency_page_type,
    detect_factual_page_type,
    detect_numeric_time_scope,
    detect_requested_currency_rate_type,
    extract_numeric_candidates,
    infer_numeric_profile,
    is_current_currency_page_type,
    numeric_values,
    representative_value,
    requires_strict_numeric_evidence,
)
from modules.internet.web.source_ranker import RankedSource
from modules.internet.web.topical_relevance import assess_source_topical_relevance, is_strict_factual_profile
from modules.internet.web.web_models import QueryClassification, WebEvidence, WebEvidencePack, WebPolicyDecision, WebSearchMode


@dataclass(frozen=True)
class _ConflictResult:
    has_conflict: bool
    severity: float
    notes: list[str]
    flags_by_url: dict[str, list[str]]
    type_mismatch_only: bool = False
    true_conflict_notes: list[str] = field(default_factory=list)
    type_mismatch_notes: list[str] = field(default_factory=list)


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
    query_text: str = "",
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
        query_text=query_text,
    )
    selected = [state["evidence"] for state in candidate_rows if isinstance(state.get("evidence"), WebEvidence)]
    source_audit = [dict(state.get("audit") or {}) for state in candidate_rows]

    # Do not return an empty pack if we have any ranked results. Keep best source as fallback.
    if not selected and candidate_rows and not _suppress_fallback_best_available(
        candidate_rows=candidate_rows,
        classification=classification,
        query_text=query_text,
        threshold=threshold,
    ):
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
            audit["numeric_conflict_severity"] = float(conflict.severity)
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
                source_audit[index]["numeric_conflict_severity"] = float(conflict.severity)

    compact_citations = _compact_citations(items=selected, classification=classification, decision=decision)
    key_facts = _collect_pack_key_facts(selected, max_items=8)
    trust_hints = _build_trust_hints(selected)
    freshness_summary = _freshness_summary(selected)
    summary = _summarize_items(selected)
    selection_summary = _build_selection_summary(
        candidate_rows=candidate_rows,
        selected=selected,
        threshold=threshold,
        conflict=conflict,
    )

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
    query_text: str,
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
        candidate = _candidate_payload(row=row, page=page, classification=classification, query_text=query_text)
        quality = float(candidate["quality"])
        audit["quality_score"] = quality
        audit["evidence_quality_score"] = quality
        audit["freshness_tag"] = str(candidate["freshness_tag"])
        audit["freshness_score"] = float(audit.get("freshness_score") or 0.0)
        audit["quality_breakdown"] = dict(candidate.get("quality_breakdown") or {})
        audit["quality_signals"] = dict(candidate.get("quality_signals") or {})
        audit["numeric_profile"] = str(candidate.get("numeric_profile") or "")
        audit["numeric_candidates_selected"] = [dict(x or {}) for x in list(candidate.get("numeric_candidates_selected") or [])]
        audit["numeric_candidates_rejected"] = [dict(x or {}) for x in list(candidate.get("numeric_candidates_rejected") or [])]
        audit["currency_page_type"] = str(candidate.get("currency_page_type") or "")
        audit["currency_requested_rate_type"] = str(candidate.get("currency_requested_rate_type") or "")
        audit["factual_page_type"] = str(candidate.get("factual_page_type") or "")
        audit["direct_answer_likelihood"] = float(candidate.get("direct_answer_likelihood") or 0.0)
        audit["factual_suitability"] = float(candidate.get("factual_suitability") or 0.0)
        audit["topical_profile"] = str(candidate.get("topical_profile") or "")
        audit["topical_relevance_score"] = float(candidate.get("topical_relevance_score") or 0.0)
        audit["topical_allow_selection"] = bool(candidate.get("topical_allow_selection", True))
        audit["topical_flags"] = [str(x or "").strip() for x in list(candidate.get("topical_flags") or []) if str(x or "").strip()]
        audit["page_type"] = str(_as_dict(candidate.get("quality_signals")).get("page_type") or "")
        out.append({"row": row, "page": page, "audit": audit, "evidence": None, "quality": quality, **candidate})

    _apply_cross_source_quality_adjustments(out)
    _apply_currency_rate_type_selection(
        states=out,
        classification=classification,
        query_text=query_text,
    )

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
        topical_filter_reason = _topical_filter_reason(state=state, classification=classification, query_text=query_text)
        if topical_filter_reason:
            audit["filtered_out_reason"] = topical_filter_reason
            state["audit"] = audit
            continue
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
    query_text: str,
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
    numeric_data = extract_numeric_candidates(
        text=text,
        snippet=snippet,
        title=str(item.title or "").strip(),
        url=str(item.url or "").strip(),
        key_facts=key_facts,
        query_category=str(getattr(classification, "primary_category", "") or ""),
        query_text=query_text,
    )
    currency_page_type = detect_currency_page_type(
        url=str(item.url or "").strip(),
        title=str(item.title or "").strip(),
        snippet=snippet,
        text=text,
    )
    factual_page_type = detect_factual_page_type(
        url=str(item.url or "").strip(),
        title=str(item.title or "").strip(),
        snippet=snippet,
        text=text,
        query_category=str(getattr(classification, "primary_category", "") or ""),
        query_text=query_text,
    )
    requested_rate_type = detect_requested_currency_rate_type(query_text)
    quality_assessment = _quality_score(
        row=row,
        page=page,
        classification=classification,
        key_facts=key_facts,
        query_text=query_text,
        numeric_data=numeric_data,
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
        "numeric_profile": str(numeric_data.profile or ""),
        "numeric_candidates_selected": [entry.to_dict() for entry in list(numeric_data.selected or [])],
        "numeric_candidates_rejected": [entry.to_dict() for entry in list(numeric_data.rejected or [])],
        "currency_page_type": str(currency_page_type or ""),
        "currency_requested_rate_type": str(requested_rate_type or ""),
        "factual_page_type": str(factual_page_type or ""),
        "direct_answer_likelihood": float(quality_assessment.signals.get("direct_answer_likelihood") or 0.0),
        "factual_suitability": float(quality_assessment.signals.get("factual_suitability") or 0.0),
        "topical_profile": str(quality_assessment.signals.get("topical_profile") or ""),
        "topical_relevance_score": float(quality_assessment.signals.get("topical_score") or 0.0),
        "topical_allow_selection": bool(quality_assessment.signals.get("topical_allow_selection", True)),
        "topical_flags": list(quality_assessment.signals.get("topical_flags") or []),
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
    query_text: str,
    numeric_data,
) -> _QualityAssessment:
    text = str(page.get("text") or "").strip()
    snippet = str(row.item.snippet or "").strip()
    title = str(row.item.title or "").strip()
    blob = " ".join(list(key_facts or [])) or text or snippet or title

    ranking_prior = max(0.0, min(1.0, float(row.item.score or 0.0)))
    trust_signal = max(0.0, min(1.0, float(row.trust_score or 0.0)))
    content_presence = _content_presence_score(text=text, snippet=snippet)
    content_depth = _content_depth_score(text=text, snippet=snippet)
    fact_signal, fact_stats = _fact_signal(
        text=text,
        snippet=snippet,
        title=title,
        key_facts=key_facts,
        numeric_data=numeric_data,
    )
    numeric_selected_count = int(fact_stats.get("numeric_selected_count") or 0)
    numeric_relevance = 0.0
    if bool(getattr(numeric_data, "strict_required", False)):
        if numeric_selected_count > 0:
            numeric_relevance = min(1.0, 0.40 + (0.16 * numeric_selected_count))
        else:
            numeric_relevance = 0.0
    title_quality = _title_quality_score(title=title)
    topical = assess_source_topical_relevance(
        domain=str(row.item.source or "").strip().lower(),
        url=str(row.item.url or "").strip(),
        title=title,
        snippet=snippet,
        text=text,
        key_facts=key_facts,
        query_category=str(getattr(classification, "primary_category", "") or ""),
        query_text=query_text,
    )
    currency_page_type = detect_currency_page_type(
        url=str(row.item.url or "").strip(),
        title=title,
        snippet=snippet,
        text=text,
    )
    factual_page_type = detect_factual_page_type(
        url=str(row.item.url or "").strip(),
        title=title,
        snippet=snippet,
        text=text,
        query_category=str(getattr(classification, "primary_category", "") or ""),
        query_text=query_text,
    )
    requested_rate_type = detect_requested_currency_rate_type(query_text)
    query_time_scope = detect_numeric_time_scope(
        query_text,
        profile=str(getattr(numeric_data, "profile", "") or ""),
    )
    rate_type_fit = 0.0
    if str(getattr(numeric_data, "profile", "") or "") == "fx_rate":
        rate_type_fit = currency_page_type_priority(
            page_type=currency_page_type,
            requested_rate_type=requested_rate_type,
            query_time_scope=query_time_scope,
        )
    page_type_relevance = _page_type_relevance(
        profile=str(getattr(numeric_data, "profile", "") or ""),
        factual_page_type=factual_page_type,
        currency_page_type=currency_page_type,
        requested_rate_type=requested_rate_type,
        query_time_scope=query_time_scope,
    )
    direct_answer_likelihood = _direct_answer_likelihood(
        profile=str(getattr(numeric_data, "profile", "") or ""),
        factual_page_type=factual_page_type,
        numeric_data=numeric_data,
        title=title,
        snippet=snippet,
        text=text,
    )
    factual_suitability = _factual_suitability(
        profile=str(getattr(numeric_data, "profile", "") or ""),
        factual_page_type=factual_page_type,
        direct_answer_likelihood=direct_answer_likelihood,
        topical_score=float(topical.score),
        numeric_selected_count=numeric_selected_count,
    )
    seo_penalty = _seo_penalty(title=title, snippet=snippet, text=text)
    thin_penalty = _thin_penalty(text=text, snippet=snippet, key_facts=key_facts, blob=blob)
    freshness_support = 0.08 if bool(getattr(classification, "requires_freshness", False)) and _parse_any_day(str(row.item.published_date or page.get("published_date") or "")) else 0.0

    breakdown = {
        "ranking_prior": round(0.16 * ranking_prior, 6),
        "trust_signal": round(0.14 * trust_signal, 6),
        "content_presence": round(0.20 * content_presence, 6),
        "content_depth": round(0.17 * content_depth, 6),
        "fact_signal": round(0.16 * fact_signal, 6),
        "numeric_relevance": round(0.12 * numeric_relevance, 6),
        "rate_type_fit": round(0.10 * rate_type_fit, 6),
        "page_type_relevance": round(0.08 * page_type_relevance, 6),
        "direct_answer": round(0.10 * direct_answer_likelihood, 6),
        "factual_suitability": round(0.10 * factual_suitability, 6),
        "title_quality": round(0.09 * title_quality, 6),
        "topical_relevance": round(0.16 * float(topical.score), 6),
        "freshness_support": round(float(freshness_support), 6),
        "consensus_bonus": 0.0,
        "conflict_penalty": 0.0,
        "topical_penalty": round(-0.18 * float(topical.penalty), 6),
        "seo_penalty": round(-0.14 * seo_penalty, 6),
        "thin_penalty": round(-0.14 * thin_penalty, 6),
    }
    if bool(getattr(numeric_data, "strict_required", False)) and numeric_selected_count == 0:
        breakdown["topical_penalty"] = round(float(breakdown["topical_penalty"]) - 0.16, 6)
    raw_score = sum(float(value) for value in breakdown.values())
    score = max(0.0, min(1.0, raw_score))
    signals = {
        "has_full_text": bool(text),
        "body_chars": int(len(text)),
        "snippet_chars": int(len(snippet)),
        "title_chars": int(len(title)),
        "key_fact_count": int(len(list(key_facts or []))),
        "numbers_count": int(fact_stats.get("numbers_count") or 0),
        "numeric_selected_count": int(fact_stats.get("numeric_selected_count") or 0),
        "numeric_rejected_count": int(fact_stats.get("numeric_rejected_count") or 0),
        "numeric_profile": str(getattr(numeric_data, "profile", "") or ""),
        "numeric_target_pairs": [str(x or "").strip() for x in list(getattr(numeric_data, "target_pairs", []) or []) if str(x or "").strip()],
        "numeric_target_units": [str(x or "").strip() for x in list(getattr(numeric_data, "target_units", []) or []) if str(x or "").strip()],
        "currency_page_type": str(currency_page_type or ""),
        "factual_page_type": str(factual_page_type or ""),
        "currency_requested_rate_type": str(requested_rate_type or ""),
        "query_time_scope": str(query_time_scope or ""),
        "page_type_relevance": round(float(page_type_relevance), 6),
        "direct_answer_likelihood": round(float(direct_answer_likelihood), 6),
        "factual_suitability": round(float(factual_suitability), 6),
        "dates_count": int(fact_stats.get("dates_count") or 0),
        "fact_marker_count": int(fact_stats.get("fact_marker_count") or 0),
        "fact_like_count": int(fact_stats.get("fact_like_count") or 0),
        "topical_profile": str(topical.profile),
        "topical_score": round(float(topical.score), 6),
        "topical_allow_selection": bool(topical.allow_selection),
        "topical_flags": list(topical.flags or []),
        "page_type": str(topical.page_type),
    }
    return _QualityAssessment(score=score, breakdown=breakdown, signals=signals)


def _page_type_relevance(
    *,
    profile: str,
    factual_page_type: str,
    currency_page_type: str,
    requested_rate_type: str,
    query_time_scope: str,
) -> float:
    factual = str(factual_page_type or "").strip().lower()
    currency = str(currency_page_type or "").strip().lower()
    if profile == "fx_rate":
        return currency_page_type_priority(
            page_type=currency or factual,
            requested_rate_type=requested_rate_type,
            query_time_scope=query_time_scope,
        )
    if profile == "historical":
        if factual in {"historical_rate", "generic_reference"}:
            return 0.92
        if factual == "news_or_analysis":
            return 0.34
        if factual == "index_or_aggregate":
            return 0.22
        return 0.48 if factual else 0.18
    if profile == "weather":
        if factual == "overview_page":
            return 0.90
        if factual == "historical_rate":
            return 0.24 if query_time_scope == "historical" else 0.08
        if factual == "news_or_analysis":
            return 0.20
        return 0.46 if factual else 0.14
    if profile == "price":
        if factual in {"overview_page", "official_rate", "generic_reference"}:
            return 0.84
        if factual == "news_or_analysis":
            return 0.26
        return 0.40 if factual else 0.12
    return 0.0


def _direct_answer_likelihood(
    *,
    profile: str,
    factual_page_type: str,
    numeric_data,
    title: str,
    snippet: str,
    text: str,
) -> float:
    factual = str(factual_page_type or "").strip().lower()
    selected_count = int(len(list(getattr(numeric_data, "selected", []) or [])))
    low = " ".join(part for part in (str(title or "").strip(), str(snippet or "").strip(), str(text or "").strip()[:240]) if part).lower()
    score = 0.0
    if selected_count > 0:
        score += min(0.58, 0.24 + (0.12 * selected_count))
    if factual in {"cash_rate", "official_rate", "nbu_rate", "bank_rate", "overview_page", "historical_rate", "generic_reference"}:
        score += 0.20
    if factual == "news_or_analysis":
        score -= 0.10
    if profile == "fx_rate" and any(token in low for token in ("exchange rate", "cash rate", "official rate", "usd/uah", "eur/uah")):
        score += 0.18
    if profile == "historical" and any(token in low for token in ("date", "year", "founded", "treaty", "timeline", "history")):
        score += 0.18
    if profile == "weather" and any(token in low for token in ("temperature", "forecast", "degrees", "feels like")):
        score += 0.18
    if profile == "price" and any(token in low for token in ("price", "pricing", "cost")):
        score += 0.18
    return max(0.0, min(1.0, score))


def _factual_suitability(
    *,
    profile: str,
    factual_page_type: str,
    direct_answer_likelihood: float,
    topical_score: float,
    numeric_selected_count: int,
) -> float:
    score = (0.42 * max(0.0, min(1.0, float(topical_score)))) + (0.38 * max(0.0, min(1.0, float(direct_answer_likelihood))))
    factual = str(factual_page_type or "").strip().lower()
    if numeric_selected_count > 0:
        score += min(0.24, 0.08 + (0.05 * int(numeric_selected_count)))
    if factual in {"news_or_analysis", "index_or_aggregate"} and profile in {"fx_rate", "price"}:
        score -= 0.16
    if factual == "historical_rate" and profile in {"fx_rate", "weather"}:
        score -= 0.20
    return max(0.0, min(1.0, score))


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


def _fact_signal(*, text: str, snippet: str, title: str, key_facts: list[str], numeric_data) -> tuple[float, dict[str, int]]:
    blob = " ".join(list(key_facts or [])) or str(text or "").strip() or str(snippet or "").strip() or str(title or "").strip()
    low = blob.lower()
    numeric_selected = list(getattr(numeric_data, "selected", []) or [])
    numeric_rejected = list(getattr(numeric_data, "rejected", []) or [])
    numbers_count = len(numeric_selected)
    dates_count = len(_DATE_RE.findall(blob))
    fact_marker_count = sum(1 for token in _FACT_MARKERS if token in low)
    fact_like_count = sum(1 for sentence in _sentences(str(text or "").strip() or str(snippet or "").strip()) if _looks_fact_like(sentence))

    score = 0.0
    if fact_like_count > 0:
        score += min(0.36, 0.18 * fact_like_count)
    if key_facts and (fact_like_count > 0 or numbers_count > 0 or dates_count > 0 or fact_marker_count > 0):
        score += 0.14
    if numbers_count > 0:
        score += min(0.24, 0.10 * numbers_count)
    if dates_count > 0:
        score += min(0.18, 0.09 * dates_count)
    if fact_marker_count > 0:
        score += min(0.16, 0.06 * fact_marker_count)
    if int(len(numeric_selected)) == 0 and bool(getattr(numeric_data, "strict_required", False)):
        score *= 0.45

    return max(0.0, min(1.0, score)), {
        "numbers_count": int(numbers_count),
        "numeric_selected_count": int(len(numeric_selected)),
        "numeric_rejected_count": int(len(numeric_rejected)),
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
                "numeric_candidates": [dict(x or {}) for x in list(_as_dict(state.get("audit")).get("numeric_candidates_selected") or state.get("numeric_candidates_selected") or []) if isinstance(x, dict)],
                "numeric_profile": str(_as_dict(state.get("audit")).get("numeric_profile") or state.get("numeric_profile") or ""),
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
                left_candidates=list(current.get("numeric_candidates") or []),
                right_candidates=list(other.get("numeric_candidates") or []),
                profile=str(current.get("numeric_profile") or other.get("numeric_profile") or ""),
            )
            similarities.append(sim)
            if sim >= 0.34:
                independent_matches += 1
            if _numeric_conflict(
                left_candidates=list(current.get("numeric_candidates") or []),
                right_candidates=list(other.get("numeric_candidates") or []),
                profile=str(current.get("numeric_profile") or other.get("numeric_profile") or ""),
            ):
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


def _apply_currency_rate_type_selection(
    *,
    states: list[dict[str, Any]],
    classification: QueryClassification,
    query_text: str,
) -> None:
    query_category = str(getattr(classification, "primary_category", "") or "")
    if infer_numeric_profile(query_category=query_category, query_text=query_text) != "fx_rate":
        return

    requested_rate_type = detect_requested_currency_rate_type(query_text)
    query_time_scope = detect_numeric_time_scope(query_text, profile="fx_rate")
    type_scores: dict[str, float] = {}
    chosen_rate_type = ""
    chosen_reason = ""

    for state in list(states or []):
        audit = _as_dict(state.get("audit"))
        page_type = _currency_page_type_for_state(state)
        numeric_selected = [dict(x or {}) for x in list(audit.get("numeric_candidates_selected") or state.get("numeric_candidates_selected") or []) if isinstance(x, dict)]
        audit["currency_requested_rate_type"] = requested_rate_type
        if page_type:
            audit["currency_page_type"] = page_type
        state["audit"] = audit
        if not page_type:
            continue
        if query_time_scope == "current" and page_type == "historical_rate" and not requested_rate_type:
            if not str(audit.get("filtered_out_reason") or "").strip():
                audit["filtered_out_reason"] = "historical_rate_mismatch"
                state["audit"] = audit
            continue
        if str(audit.get("filtered_out_reason") or "").strip():
            continue
        if requested_rate_type and page_type != requested_rate_type:
            continue
        quality = float(state.get("quality") or audit.get("quality_score") or 0.0)
        type_scores[page_type] = float(type_scores.get(page_type, 0.0)) + quality + (0.18 * currency_page_type_priority(
            page_type=page_type,
            requested_rate_type=requested_rate_type,
            query_time_scope=query_time_scope,
        ))

    if requested_rate_type and type_scores.get(requested_rate_type, 0.0) > 0.0:
        chosen_rate_type = requested_rate_type
        chosen_reason = "explicit_query_rate_type"
    elif type_scores:
        chosen_rate_type = max(type_scores.items(), key=lambda item: (float(item[1]), str(item[0])))[0]
        chosen_reason = "dominant_currency_rate_type"

    for state in list(states or []):
        audit = _as_dict(state.get("audit"))
        page_type = _currency_page_type_for_state(state)
        numeric_selected = [dict(x or {}) for x in list(audit.get("numeric_candidates_selected") or state.get("numeric_candidates_selected") or []) if isinstance(x, dict)]
        audit["currency_selected_rate_type"] = chosen_rate_type
        audit["currency_rate_type_reason"] = chosen_reason
        state["audit"] = audit
        if requested_rate_type and page_type and page_type != requested_rate_type and not chosen_rate_type:
            if numeric_selected and not str(audit.get("filtered_out_reason") or "").strip():
                audit["filtered_out_reason"] = "rate_type_mismatch"
                state["audit"] = audit
            continue
        if not chosen_rate_type or not page_type:
            continue
        if str(audit.get("filtered_out_reason") or "").strip():
            continue
        if not numeric_selected and page_type != "historical_rate":
            continue
        if not _currency_page_type_matches(chosen_rate_type, page_type):
            audit["filtered_out_reason"] = "rate_type_mismatch"
            state["audit"] = audit


def _currency_page_type_for_state(state: dict[str, Any]) -> str:
    audit = _as_dict(state.get("audit"))
    if str(audit.get("currency_page_type") or "").strip():
        return str(audit.get("currency_page_type") or "").strip()
    page_type = str(_as_dict(audit.get("quality_signals")).get("currency_page_type") or "").strip()
    if page_type:
        return page_type
    row = state.get("row")
    item = getattr(row, "item", None)
    return detect_currency_page_type(
        url=str(getattr(item, "url", "") or "").strip(),
        title=str(getattr(item, "title", "") or "").strip(),
        snippet=str(state.get("snippet") or getattr(item, "snippet", "") or "").strip(),
        text=str(state.get("text") or "").strip(),
    )


def _currency_page_type_matches(selected_rate_type: str, page_type: str) -> bool:
    selected = str(selected_rate_type or "").strip().lower()
    current = str(page_type or "").strip().lower()
    if not selected or not current:
        return True
    if selected == current:
        return True
    if selected == "currency_overview" and current == "currency_page":
        return True
    return False


def _topical_filter_reason(*, state: dict[str, Any], classification: QueryClassification, query_text: str) -> str:
    query_category = str(getattr(classification, "primary_category", "") or "")
    numeric_profile = infer_numeric_profile(query_category=query_category, query_text=query_text)
    strict_profile = is_strict_factual_profile(
        query_category=query_category,
        query_text=query_text,
    )
    strict_numeric = requires_strict_numeric_evidence(
        query_category=query_category,
        query_text=query_text,
    )
    if not strict_profile and not strict_numeric:
        return ""
    audit = _as_dict(state.get("audit"))
    flags = [str(x or "").strip().lower() for x in list(audit.get("topical_flags") or state.get("topical_flags") or []) if str(x or "").strip()]
    topical_allow = bool(audit.get("topical_allow_selection", state.get("topical_allow_selection", True)))
    topical_score = float(audit.get("topical_relevance_score") or state.get("topical_relevance_score") or 0.0)
    page_type = str(audit.get("page_type") or _as_dict(audit.get("quality_signals")).get("page_type") or "").strip().lower()
    factual_page_type = str(audit.get("factual_page_type") or _as_dict(audit.get("quality_signals")).get("factual_page_type") or "").strip().lower()
    direct_answer_likelihood = float(
        audit.get("direct_answer_likelihood")
        or _as_dict(audit.get("quality_signals")).get("direct_answer_likelihood")
        or 0.0
    )
    numeric_selected = [dict(x or {}) for x in list(audit.get("numeric_candidates_selected") or state.get("numeric_candidates_selected") or [])]
    if not topical_allow or any(flag.startswith("source_type:") for flag in flags):
        return "unsupported_source_type"
    if page_type in {"video", "support", "forum"}:
        return "unsupported_source_type"
    if strict_numeric and numeric_profile in {"fx_rate", "price"} and not numeric_selected:
        return "missing_relevant_numeric"
    if strict_numeric and factual_page_type in {"news_or_analysis", "index_or_aggregate"} and direct_answer_likelihood < 0.30:
        return "weak_direct_answer"
    if topical_score < 0.24:
        return "topical_mismatch"
    return ""


def _suppress_fallback_best_available(
    *,
    candidate_rows: list[dict[str, Any]],
    classification: QueryClassification,
    query_text: str,
    threshold: float,
) -> bool:
    if not candidate_rows:
        return False
    if not is_strict_factual_profile(
        query_category=str(getattr(classification, "primary_category", "") or ""),
        query_text=query_text,
    ):
        return False
    best = dict(candidate_rows[0] or {})
    audit = _as_dict(best.get("audit"))
    reason = str(audit.get("filtered_out_reason") or "").strip().lower()
    topical_score = float(audit.get("topical_relevance_score") or best.get("topical_relevance_score") or 0.0)
    quality = float(best.get("quality") or 0.0)
    if reason in {"unsupported_source_type", "topical_mismatch", "rate_type_mismatch", "historical_rate_mismatch", "weak_direct_answer"}:
        return True
    if reason == "missing_relevant_numeric":
        return True
    if quality < max(0.24, float(threshold) + 0.03) and topical_score < 0.42:
        return True
    return False


def _cross_source_similarity(
    *,
    left_tokens: set[str],
    right_tokens: set[str],
    left_candidates: list[dict[str, Any]],
    right_candidates: list[dict[str, Any]],
    profile: str = "",
) -> float:
    lexical = _jaccard(left_tokens, right_tokens)
    numeric = _numeric_agreement(left_candidates=left_candidates, right_candidates=right_candidates, profile=profile)
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


def _numeric_agreement(*, left_candidates: list[dict[str, Any]], right_candidates: list[dict[str, Any]], profile: str = "") -> float:
    left_groups = _group_numeric_candidates(left_candidates)
    right_groups = _group_numeric_candidates(right_candidates)
    overlap = sorted(set(left_groups) & set(right_groups))
    if not overlap:
        return 0.5
    best = 1.0
    for group_key in overlap:
        left = [float(x) for x in list(left_groups.get(group_key) or []) if float(x) > 0]
        right = [float(x) for x in list(right_groups.get(group_key) or []) if float(x) > 0]
        for a in left:
            for b in right:
                severity = conflict_severity_for_values(left_value=a, right_value=b, profile=profile, group_key=group_key)
                rel = min(1.0, float(severity))
                if rel < best:
                    best = rel
    if best == 1.0:
        return 0.5
    if best <= 0.05:
        return 1.0
    if best <= 0.25:
        return 0.65
    if best <= 0.58:
        return 0.35
    return 0.0


def _numeric_conflict(*, left_candidates: list[dict[str, Any]], right_candidates: list[dict[str, Any]], profile: str = "") -> bool:
    return _numeric_agreement(left_candidates=left_candidates, right_candidates=right_candidates, profile=profile) <= 0.1


def _group_numeric_candidates(rows: list[dict[str, Any]]) -> dict[str, list[float]]:
    grouped: dict[str, list[float]] = {}
    for row in list(rows or []):
        item = dict(row or {})
        try:
            value = float(item.get("value") or 0.0)
        except Exception:
            continue
        if value <= 0:
            continue
        group_key = candidate_group_key(item)
        grouped.setdefault(group_key, []).append(value)
    return grouped


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
    conflict: _ConflictResult | None = None,
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
    selected_rate_types = sorted(
        {
            str(audit.get("currency_page_type") or "").strip()
            for audit in selected_audits
            if str(audit.get("currency_page_type") or "").strip()
        }
    )
    requested_rate_type = next(
        (
            str(audit.get("currency_requested_rate_type") or "").strip()
            for audit in audits
            if str(audit.get("currency_requested_rate_type") or "").strip()
        ),
        "",
    )
    selected_rate_type = next(
        (
            str(audit.get("currency_selected_rate_type") or "").strip()
            for audit in selected_audits + audits
            if str(audit.get("currency_selected_rate_type") or "").strip()
        ),
        "",
    )
    rate_type_reason = next(
        (
            str(audit.get("currency_rate_type_reason") or "").strip()
            for audit in selected_audits + audits
            if str(audit.get("currency_rate_type_reason") or "").strip()
        ),
        "",
    )
    primary_selected = selected_audits[0] if selected_audits else {}
    derived_type_mismatches = list(getattr(conflict, "type_mismatch_notes", []) or [])
    for audit in audits:
        reason = str(audit.get("filtered_out_reason") or "").strip().lower()
        if reason not in {"rate_type_mismatch", "historical_rate_mismatch"}:
            continue
        page_type = str(audit.get("factual_page_type") or audit.get("currency_page_type") or "").strip()
        domain = str(audit.get("domain") or "").strip().lower()
        note = f"{reason}:{domain}:{page_type}".rstrip(":")
        if note not in derived_type_mismatches:
            derived_type_mismatches.append(note)
    factual_basis = {
        "selected_result": {
            "url": str(primary_selected.get("url") or "").strip(),
            "domain": str(primary_selected.get("domain") or "").strip().lower(),
            "page_type": str(primary_selected.get("factual_page_type") or primary_selected.get("currency_page_type") or "").strip(),
            "direct_answer_likelihood": round(float(primary_selected.get("direct_answer_likelihood") or 0.0), 6),
        },
        "selected_page_types": sorted(
            {
                str(audit.get("factual_page_type") or audit.get("currency_page_type") or "").strip()
                for audit in selected_audits
                if str(audit.get("factual_page_type") or audit.get("currency_page_type") or "").strip()
            }
        ),
        "accepted_numeric_candidates": [
            {
                "domain": str(audit.get("domain") or "").strip().lower(),
                "page_type": str(audit.get("factual_page_type") or audit.get("currency_page_type") or "").strip(),
                "candidates": [dict(x or {}) for x in list(audit.get("numeric_candidates_selected") or [])[:3]],
            }
            for audit in selected_audits
            if list(audit.get("numeric_candidates_selected") or [])
        ],
        "rejected_results": [
            {
                "domain": str(audit.get("domain") or "").strip().lower(),
                "url": str(audit.get("url") or "").strip(),
                "page_type": str(audit.get("factual_page_type") or audit.get("currency_page_type") or "").strip(),
                "reason": str(audit.get("filtered_out_reason") or "").strip(),
                "rejected_numeric_reasons": sorted(
                    {
                        str(dict(row or {}).get("reason") or "").strip()
                        for row in list(audit.get("numeric_candidates_rejected") or [])[:6]
                        if str(dict(row or {}).get("reason") or "").strip()
                    }
                ),
            }
            for audit in audits
            if str(audit.get("filtered_out_reason") or "").strip()
        ][:8],
        "conflict_summary": {
            "has_true_conflict": bool(list(getattr(conflict, "true_conflict_notes", []) or [])),
            "true_conflicts": list(getattr(conflict, "true_conflict_notes", []) or [])[:6],
            "type_mismatches": list(derived_type_mismatches)[:6],
        },
    }
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
        "currency_requested_rate_type": requested_rate_type,
        "currency_selected_rate_type": selected_rate_type,
        "currency_page_types_selected": list(selected_rate_types),
        "currency_rate_type_reason": rate_type_reason,
        "primary_selected_url": str(primary_selected.get("url") or "").strip(),
        "primary_selected_domain": str(primary_selected.get("domain") or "").strip().lower(),
        "primary_selected_page_type": str(primary_selected.get("currency_page_type") or "").strip(),
        "primary_selected_factual_page_type": str(primary_selected.get("factual_page_type") or "").strip(),
        "factual_basis": factual_basis,
        "true_conflict_notes": list(getattr(conflict, "true_conflict_notes", []) or [])[:6],
        "type_mismatch_notes": list(derived_type_mismatches)[:6],
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
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = {}
    pair_page_types: dict[str, set[str]] = {}
    flags: dict[str, list[str]] = {}
    profile = ""
    for item in list(items or []):
        audit = _as_dict(getattr(item, "audit", {}) or {})
        selected = [dict(x or {}) for x in list(audit.get("numeric_candidates_selected") or []) if isinstance(x, dict)]
        if not selected:
            continue
        domain = str(item.domain or "").strip().lower() or "unknown"
        url = str(item.url or "").strip()
        profile = str(audit.get("numeric_profile") or profile or "")
        for row in list(selected):
            group_key = candidate_group_key(row)
            grouped.setdefault(group_key, {}).setdefault(domain, []).append({"url": url, **row})
            pair = str(dict(row or {}).get("pair") or "").strip().upper()
            page_type = str(dict(row or {}).get("page_type") or "").strip().lower()
            if pair and page_type:
                pair_page_types.setdefault(pair, set()).add(page_type)

    if not grouped and not pair_page_types:
        return _ConflictResult(has_conflict=False, severity=0.0, notes=[], flags_by_url={})

    notes: list[str] = []
    numeric_conflict_notes: list[str] = []
    type_mismatch_notes: list[str] = []
    max_severity = 0.0
    for group_key, by_domain in grouped.items():
        if len(by_domain) < 2:
            continue
        representatives: list[tuple[str, str, float]] = []
        for domain, rows in by_domain.items():
            rep = representative_value(rows)
            if rep is None or rep <= 0:
                continue
            url = str(dict(rows[0] or {}).get("url") or "").strip()
            representatives.append((domain, url, rep))
        if len(representatives) < 2:
            continue
        representatives.sort(key=lambda item: item[2])
        low_domain, low_url, low_val = representatives[0]
        high_domain, high_url, high_val = representatives[-1]
        severity = conflict_severity_for_values(
            left_value=low_val,
            right_value=high_val,
            profile=profile or "generic",
            group_key=group_key,
        )
        if severity < 0.55:
            continue
        max_severity = max(max_severity, float(severity))
        level = "high" if severity >= 0.85 else "medium"
        label = str(group_key or "value")
        note = (
            f"numeric_conflict_{level}:{label}:{low_domain}={low_val:.3f} "
            f"vs {high_domain}={high_val:.3f}"
        )
        if note not in notes:
            notes.append(note)
        if note not in numeric_conflict_notes:
            numeric_conflict_notes.append(note)
        flags.setdefault(low_url, []).append(f"numeric_conflict_{level}")
        flags.setdefault(high_url, []).append(f"numeric_conflict_{level}")

    for pair, page_types in sorted(pair_page_types.items()):
        if len(page_types) < 2:
            continue
        label = " vs ".join(sorted(page_types))
        note = f"rate_type_mismatch:{pair}:{label}"
        if note not in notes:
            notes.append(note)
        if note not in type_mismatch_notes:
            type_mismatch_notes.append(note)

    return _ConflictResult(
        has_conflict=bool(numeric_conflict_notes),
        severity=max(0.0, min(1.0, float(max_severity))),
        notes=notes,
        flags_by_url=flags,
        type_mismatch_only=bool(notes) and not bool(numeric_conflict_notes),
        true_conflict_notes=numeric_conflict_notes,
        type_mismatch_notes=type_mismatch_notes,
    )


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
