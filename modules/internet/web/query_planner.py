from __future__ import annotations

import datetime as dt
import re

from modules.internet.web.web_models import QueryClassification, WebPolicyDecision, WebQueryPlan, WebSearchMode


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

    domains = [str(x or "").strip().lower() for x in list(preferred_domains or []) if str(x or "").strip()]
    category = str(classification.primary_category or "").strip().lower()
    year = str(dt.datetime.now(dt.timezone.utc).year)

    scout: list[str] = [text]
    focused: list[str] = []
    fallback: list[str] = []

    entity = _extract_entity(text)
    if category in {"version"} or "version" in text.lower() or "версия" in text.lower():
        focused.append(f"{entity} latest version official")
        focused.append(f"{entity} release notes")
    elif category in {"price"} or "цена" in text.lower() or "price" in text.lower():
        focused.append(f"{entity} price official")
        focused.append(f"{entity} availability {year}")
    elif category in {"news"}:
        focused.append(f"{entity} latest news {year}")
        focused.append(f"{entity} official announcement")
    elif classification.query_type == "mixed":
        focused.append(f"{entity} best practices {year}")
        focused.append(f"{entity} benchmark official")
    elif classification.is_ambiguous:
        focused.append(f"{entity} what is")
        focused.append(f"{entity} official docs")

    if decision.mode in {WebSearchMode.TARGETED_SEARCH, WebSearchMode.DEEP_SEARCH} and domains:
        for domain in domains[:2]:
            focused.append(f"{text} site:{domain}")

    if decision.mode in {WebSearchMode.VERIFY_ONLY, WebSearchMode.SOFT_SEARCH}:
        fallback.append(f"{entity} official")
    elif decision.mode in {WebSearchMode.TARGETED_SEARCH, WebSearchMode.DEEP_SEARCH}:
        fallback.append(f"{entity} documentation")
        fallback.append(f"{entity} github release")

    budget_queries = max(0, int(decision.budget.max_queries))
    scout = _dedupe_keep_order(scout)
    focused = _dedupe_keep_order(focused)
    fallback = _dedupe_keep_order(fallback)

    if budget_queries <= 0:
        scout = []
        focused = []
        fallback = []
    else:
        all_rows = scout + focused + fallback
        all_rows = _dedupe_keep_order(all_rows)[:budget_queries]
        scout, focused, fallback = _split_for_budget(all_rows, original_scout=scout, original_focused=focused)

    return WebQueryPlan(
        mode=decision.mode,
        scout_queries=scout,
        focused_queries=focused,
        fallback_queries=fallback,
        preferred_domains=domains,
    )


def _split_for_budget(
    all_rows: list[str],
    *,
    original_scout: list[str],
    original_focused: list[str],
) -> tuple[list[str], list[str], list[str]]:
    scout: list[str] = []
    focused: list[str] = []
    fallback: list[str] = []
    scout_set = set(original_scout)
    focused_set = set(original_focused)
    for row in list(all_rows):
        if row in scout_set:
            scout.append(row)
        elif row in focused_set:
            focused.append(row)
        else:
            fallback.append(row)
    return scout, focused, fallback


def _strip_prefix(text: str) -> str:
    src = str(text or "").strip()
    low = src.lower()
    for token in ("/web", "/no-web"):
        if low == token:
            return ""
        if low.startswith(token + " "):
            return src[len(token) :].strip()
    return src


def _extract_entity(text: str) -> str:
    src = str(text or "").strip()
    if not src:
        return "query"
    tokens = re.findall(r"[A-Za-zА-Яа-яЁёЇїІіЄєҐґ0-9\-\._]+", src)
    if not tokens:
        return src[:64]
    blacklist = {
        "какая",
        "какие",
        "какой",
        "что",
        "сейчас",
        "лучшие",
        "best",
        "latest",
        "version",
        "price",
        "news",
        "актуальные",
        "практики",
    }
    filtered = [t for t in tokens if t.lower() not in blacklist]
    if not filtered:
        filtered = tokens
    return " ".join(filtered[:5]).strip()


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

