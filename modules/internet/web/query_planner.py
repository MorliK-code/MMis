from __future__ import annotations

import datetime as dt
import re
from typing import Iterable

from modules.internet.web.web_models import QueryClassification, WebPolicyDecision, WebQueryPlan, WebSearchMode


_WORD_RE = re.compile(r"[A-Za-zА-Яа-яЁёІіЇїЄєҐґ0-9\-\._]+")
_QUOTED_RE = re.compile(r"\"([^\"]{2,120})\"|“([^”]{2,120})”|«([^»]{2,120})»")
_STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "from",
    "this",
    "that",
    "what",
    "which",
    "latest",
    "current",
    "official",
    "docs",
    "documentation",
    "news",
    "release",
    "version",
    "now",
    "today",
}
_VERSION_MARKERS = {
    "version",
    "release",
    "changelog",
    "latest",
    "stable",
}


def build_query_plan(
    *,
    query: str,
    classification: QueryClassification,
    decision: WebPolicyDecision,
    preferred_domains: list[str] | None = None,
) -> WebQueryPlan:
    text = _strip_prefix(query)
    if not text:
        return WebQueryPlan(mode=decision.mode)

    budget_queries = max(0, int(decision.budget.max_queries))
    domains = _normalize_domains(preferred_domains)
    now_year = str(dt.datetime.now(dt.timezone.utc).year)
    category = str(classification.primary_category or "").strip().lower()
    category_hint = _category_hint(category)
    entity = _extract_primary_entity(text)
    focus_terms = _extract_focus_terms(text, entity=entity)
    focus_tail = " ".join(focus_terms[:4]).strip()

    primary_query = text
    validation_query = _join_parts(entity, focus_tail, "official documentation")
    exact_entity_query = _join_parts(f"\"{entity}\"", focus_terms[0] if focus_terms else category_hint or "official")

    release_note_queries: list[str] = []
    if _needs_release_notes(text=text, classification=classification):
        release_note_queries = _dedupe_keep_order(
            [
                _join_parts(entity, "release notes", now_year),
                _join_parts(entity, "changelog", now_year),
                _join_parts(entity, "latest stable version official"),
            ]
        )

    domain_queries = [_join_parts(primary_query, f"site:{domain}") for domain in domains[:3]]
    validation_queries = _dedupe_keep_order([validation_query, _join_parts(entity, "official source", category_hint)])
    fallback_queries = _dedupe_keep_order(
        [
            _join_parts(entity, category_hint or "documentation"),
            _join_parts(entity, "official"),
        ]
    )

    query_roles = {
        "primary": _dedupe_keep_order([primary_query]),
        "validation": validation_queries,
        "release_notes": release_note_queries,
        "exact_entity": _dedupe_keep_order([exact_entity_query]),
        "domain_constrained": _dedupe_keep_order(domain_queries),
    }

    ordered_roles = _role_order_for_mode(decision.mode)
    prioritized: list[str] = []
    for role in ordered_roles:
        prioritized.extend(query_roles.get(role) or [])
    prioritized.extend(fallback_queries)
    prioritized = _dedupe_keep_order(prioritized)

    if budget_queries <= 0:
        return WebQueryPlan(
            mode=decision.mode,
            scout_queries=[],
            focused_queries=[],
            fallback_queries=[],
            preferred_domains=domains,
            query_roles=query_roles,
        )

    bounded = prioritized[:budget_queries]
    if (
        domains
        and decision.mode in {WebSearchMode.TARGETED_SEARCH, WebSearchMode.DEEP_SEARCH}
        and not any("site:" in str(x or "").lower() for x in list(bounded))
    ):
        domain_seed = str((query_roles.get("domain_constrained") or [""])[0] or "").strip()
        if domain_seed:
            if len(bounded) < budget_queries:
                bounded.append(domain_seed)
            else:
                bounded[-1] = domain_seed
            bounded = _dedupe_keep_order(bounded)[:budget_queries]
    scout_cap, focused_cap = _phase_caps(decision.mode)
    scout = bounded[: max(1, min(scout_cap, len(bounded)))]
    remaining = bounded[len(scout) :]
    focused = remaining[: max(0, focused_cap)]
    fallback = remaining[len(focused) :]

    return WebQueryPlan(
        mode=decision.mode,
        scout_queries=scout,
        focused_queries=focused,
        fallback_queries=fallback,
        preferred_domains=domains,
        query_roles=query_roles,
    )


def _strip_prefix(text: str) -> str:
    src = str(text or "").strip()
    low = src.lower()
    for token in ("/web", "/no-web"):
        if low == token:
            return ""
        if low.startswith(token + " "):
            return src[len(token) :].strip()
    return src


def _extract_primary_entity(text: str) -> str:
    src = str(text or "").strip()
    if not src:
        return "query"

    quoted = _extract_quoted_entities(src)
    if quoted:
        return quoted[0]

    tokens = _WORD_RE.findall(src)
    if not tokens:
        return src[:64]

    strong = [x for x in tokens if any(ch.isupper() for ch in x) or any(ch.isdigit() for ch in x)]
    if strong:
        return " ".join(strong[:4]).strip()

    filtered = [t for t in tokens if t.lower() not in _STOPWORDS]
    if not filtered:
        filtered = tokens
    return " ".join(filtered[:4]).strip()


def _extract_quoted_entities(text: str) -> list[str]:
    out: list[str] = []
    for match in _QUOTED_RE.findall(str(text or "")):
        for group in list(match):
            token = str(group or "").strip()
            if token and token not in out:
                out.append(token)
    return out


def _extract_focus_terms(text: str, *, entity: str) -> list[str]:
    src = str(text or "").strip()
    entity_tokens = {x.lower() for x in _WORD_RE.findall(str(entity or ""))}
    out: list[str] = []
    for token in _WORD_RE.findall(src):
        item = str(token or "").strip().lower()
        if not item or item in _STOPWORDS or item in entity_tokens:
            continue
        if item not in out:
            out.append(item)
    return out[:10]


def _normalize_domains(domains: list[str] | None) -> list[str]:
    out: list[str] = []
    for row in list(domains or []):
        item = str(row or "").strip().lower()
        if not item:
            continue
        if item.startswith("www."):
            item = item[4:]
        if item not in out:
            out.append(item)
    return out


def _category_hint(category: str) -> str:
    value = str(category or "").strip().lower()
    mapping = {
        "version": "version",
        "price": "price",
        "news": "news",
        "refactor": "best practices",
        "architecture": "architecture",
    }
    return str(mapping.get(value, "")).strip()


def _needs_release_notes(*, text: str, classification: QueryClassification) -> bool:
    if str(classification.primary_category or "").strip().lower() in {"version", "news"}:
        return True
    if classification.requires_freshness and classification.is_external_fact_question:
        return True
    low = str(text or "").lower()
    return any(marker in low for marker in _VERSION_MARKERS)


def _role_order_for_mode(mode: WebSearchMode) -> list[str]:
    if mode == WebSearchMode.VERIFY_ONLY:
        return ["primary", "validation", "exact_entity", "domain_constrained", "release_notes"]
    if mode == WebSearchMode.SOFT_SEARCH:
        return ["primary", "exact_entity", "validation", "domain_constrained", "release_notes"]
    if mode == WebSearchMode.TARGETED_SEARCH:
        return ["primary", "validation", "exact_entity", "domain_constrained", "release_notes"]
    if mode == WebSearchMode.DEEP_SEARCH:
        return ["primary", "validation", "domain_constrained", "release_notes", "exact_entity"]
    return ["primary", "validation", "exact_entity", "release_notes", "domain_constrained"]


def _phase_caps(mode: WebSearchMode) -> tuple[int, int]:
    if mode == WebSearchMode.VERIFY_ONLY:
        return (1, 1)
    if mode == WebSearchMode.SOFT_SEARCH:
        return (1, 2)
    if mode == WebSearchMode.TARGETED_SEARCH:
        return (2, 2)
    if mode == WebSearchMode.DEEP_SEARCH:
        return (2, 3)
    return (1, 1)


def _join_parts(*parts: Iterable[str] | str) -> str:
    out: list[str] = []
    for part in parts:
        if isinstance(part, str):
            token = str(part or "").strip()
            if token:
                out.append(token)
            continue
        for row in list(part or []):
            token = str(row or "").strip()
            if token:
                out.append(token)
    return " ".join(out).strip()


def _dedupe_keep_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for row in list(items or []):
        value = str(row or "").strip()
        if not value:
            continue
        key = value.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(value)
    return out
