from __future__ import annotations

import re
from typing import Any


_VALID_ENGINE_STATES = {"healthy", "degraded", "blocked", "suspended"}
_VALID_LOCALE_RE = re.compile(r"^[A-Za-z]{2,3}(?:-[A-Za-z]{2,4})?$")
_UA_FINANCE_MARKERS = ("uah", "грн", "грив", "usd/uah", "eur/uah", "gbp/uah")
_UA_GEO_MARKERS = ("ukraine", "kyiv", "kiev", "ua", "украин", "україн", "киев", "київ", "шостк", "shostk")
_RU_MARKERS = ("russia", "moscow", "rub", "рубл", "росси", "москв", "cbr", "цб рф")
_BLOCKED_REASON_MARKERS = ("access denied", "captcha", "forbidden", "403", "blocked")
_DEGRADED_REASON_MARKERS = ("dns", "resolve", "timeout", "tempor", "unavailable", "bad request", "429", "502", "503")
_UA_DOMAIN_HINTS = (
    "bank.gov.ua",
    "minfin.com.ua",
    "finance.ua",
    "kurs.com.ua",
    "sinoptik.ua",
    "meteo.ua",
    "ukrinform.ua",
)


def normalize_search_locale(value: str, *, default: str = "ru-RU") -> str:
    src = str(value or "").strip().replace("_", "-")
    fallback = str(default or "ru-RU").strip().replace("_", "-") or "ru-RU"
    if not src or src.lower() == "all":
        return fallback
    if not _VALID_LOCALE_RE.match(src):
        return fallback
    head, sep, tail = src.partition("-")
    if not sep:
        return head.lower()
    return f"{head.lower()}-{tail.upper()}"


def build_search_context(
    *,
    query_text: str,
    query_category: str = "",
    query_intent: str = "",
    geo_hint: str = "",
    policy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cfg = dict(policy or {})
    category = str(query_category or "").strip().lower()
    intent = str(query_intent or "").strip().lower()
    low = str(query_text or "").strip().lower()
    geo = str(geo_hint or "").strip().lower()
    reasons: list[str] = []

    explicit_ru = any(token in low for token in _RU_MARKERS) or any(token in geo for token in _RU_MARKERS)
    explicit_ua = any(token in low for token in _UA_GEO_MARKERS) or any(token in geo for token in _UA_GEO_MARKERS)
    has_uah_context = any(token in low for token in _UA_FINANCE_MARKERS)

    region_bias = ""
    if category == "finance" and has_uah_context and not explicit_ru:
        region_bias = "ua"
        reasons.append("uah_market_context")
    elif category in {"weather", "news"} and explicit_ua and not explicit_ru:
        region_bias = "ua"
        reasons.append("ua_geo_context")
    elif explicit_ru and not explicit_ua:
        region_bias = "ru"
        reasons.append("explicit_ru_market")

    default_locale = normalize_search_locale(str(cfg.get("default_locale") or "ru-RU"), default="ru-RU")
    locale_by_category = {
        str(k or "").strip().lower(): normalize_search_locale(str(v or ""), default=default_locale)
        for k, v in dict(cfg.get("locale_by_category") or {}).items()
        if str(k or "").strip()
    }
    region_locale_overrides = {
        str(k or "").strip().lower(): normalize_search_locale(str(v or ""), default=default_locale)
        for k, v in dict(cfg.get("region_locale_overrides") or {}).items()
        if str(k or "").strip()
    }

    locale = locale_by_category.get(category) or default_locale
    if region_bias:
        locale = region_locale_overrides.get(region_bias, locale)
    if category in {"docs", "version"} or intent == "news_release":
        locale = locale_by_category.get(category) or locale_by_category.get("docs") or locale

    return {
        "region_bias": str(region_bias or ""),
        "query_locale": str(locale or default_locale),
        "reasons": list(reasons),
        "explicit_region": bool(explicit_ua or explicit_ru),
        "engine_manual_states": _normalize_engine_states(cfg.get("engine_manual_states")),
        "suspended_engines": sorted(
            {
                str(x or "").strip().lower()
                for x in list(cfg.get("suspended_engines") or [])
                if str(x or "").strip()
            }
        ),
    }


def region_domain_adjustment(
    *,
    domain: str,
    region_bias: str,
    query_intent: str = "",
    query_category: str = "",
    query_text: str = "",
) -> float:
    host = str(domain or "").strip().lower()
    bias = str(region_bias or "").strip().lower()
    if not host or not bias:
        return 0.0
    if host.startswith("www."):
        host = host[4:]
    category = str(query_category or "").strip().lower()
    low = str(query_text or "").strip().lower()
    if bias == "ua":
        if any(token in low for token in _RU_MARKERS):
            return 0.0
        bonus = 0.0
        if host.endswith(".ua") or host.endswith(".com.ua"):
            bonus += 0.12 if category == "finance" else 0.08
        if any(hint in host for hint in _UA_DOMAIN_HINTS):
            bonus += 0.06 if category == "finance" else 0.03
        if host.endswith(".ru") or host.endswith(".su"):
            return bonus - (0.18 if category == "finance" else 0.10)
        return bonus
    if bias == "ru":
        if host.endswith(".ru") or host.endswith(".su"):
            return 0.10 if category in {"finance", "news"} else 0.06
        if host.endswith(".ua") or host.endswith(".com.ua"):
            return -0.08
    return 0.0


def classify_engine_state(*, engine: str, reason: str = "", manual_state: str = "") -> str:
    manual = str(manual_state or "").strip().lower()
    if manual in _VALID_ENGINE_STATES:
        return manual
    low = str(reason or "").strip().lower()
    if any(token in low for token in _BLOCKED_REASON_MARKERS):
        return "blocked"
    if any(token in low for token in _DEGRADED_REASON_MARKERS):
        return "degraded"
    if low:
        return "degraded"
    if str(engine or "").strip():
        return "healthy"
    return "degraded"


def compute_search_confidence(
    *,
    raw_results_count: int,
    usable_results_count: int,
    top_scores: list[float] | None = None,
    engine_health: dict[str, str] | None = None,
) -> float:
    raw_count = max(0, int(raw_results_count or 0))
    usable_count = max(0, int(usable_results_count or 0))
    scores = [max(0.0, min(1.0, float(x))) for x in list(top_scores or []) if float(x) >= 0.0]
    avg_top_score = (sum(scores[:3]) / max(1, len(scores[:3]))) if scores else 0.0
    states = {str(k or "").strip().lower(): str(v or "").strip().lower() for k, v in dict(engine_health or {}).items() if str(k or "").strip()}

    result_component = min(1.0, usable_count / 3.0) * 0.42
    raw_component = min(1.0, raw_count / 6.0) * 0.18
    quality_component = avg_top_score * 0.24

    if states:
        healthy = sum(1 for state in states.values() if state == "healthy")
        degraded = sum(1 for state in states.values() if state == "degraded")
        blocked = sum(1 for state in states.values() if state in {"blocked", "suspended"})
        total = max(1, len(states))
        engine_component = ((healthy + (0.45 * degraded)) / total) * 0.16
        engine_penalty = min(0.12, ((0.06 * degraded) + (0.10 * blocked)) / total)
    else:
        engine_component = 0.08 if usable_count > 0 else 0.0
        engine_penalty = 0.0

    confidence = result_component + raw_component + quality_component + engine_component - engine_penalty
    return max(0.0, min(1.0, confidence))


def _normalize_engine_states(value: Any) -> dict[str, str]:
    out: dict[str, str] = {}
    if not isinstance(value, dict):
        return out
    for key, state in value.items():
        engine = str(key or "").strip().lower()
        token = str(state or "").strip().lower()
        if engine and token in _VALID_ENGINE_STATES:
            out[engine] = token
    return out
