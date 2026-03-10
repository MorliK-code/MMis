from __future__ import annotations

import datetime as dt
import re
from typing import Any

from modules.internet.web.web_models import FreshnessAssessment, QueryClassification


_PRICE_MARKERS = ("price", "cost", "цена", "курс", "exchange", "usd", "eur", "uah")
_VERSION_MARKERS = ("version", "release", "changelog", "версия", "релиз", "обновлен")
_NEWS_MARKERS = ("news", "breaking", "анонс", "новости", "новин", "latest")
_MARKET_MARKERS = ("best", "market", "benchmark", "compare", "рынок", "бенчмарк", "сравни")


def assess_freshness(
    *,
    query: str,
    classification: QueryClassification,
    web_items: list[Any] | None = None,
    ttl_days: dict[str, int] | None = None,
    now_utc: dt.datetime | None = None,
) -> FreshnessAssessment:
    src = str(query or "").strip().lower()
    now = now_utc or dt.datetime.now(dt.timezone.utc)

    category = _detect_freshness_category(src)
    ttl_cfg = _normalized_ttl_days(ttl_days)
    ttl_for_category = ttl_cfg.get(category, ttl_cfg["default"])

    breakdown = {
        "base": 0.14,
        "temporal_marker_boost": 0.35 if classification.is_temporal else 0.0,
        "freshness_required_boost": 0.28 if classification.requires_freshness else 0.0,
        "external_fact_boost": 0.14 if classification.is_external_fact_question else 0.0,
        "local_scope_penalty": -0.20 if classification.is_local_project_question and not classification.requires_freshness else 0.0,
        "category_boost": _category_boost(category),
    }
    temporal_risk = 0.0
    for value in breakdown.values():
        temporal_risk += float(value)
    temporal_risk = max(0.0, min(1.0, temporal_risk))

    latest_age_days = _latest_age_days(web_items=web_items, now=now)
    stale_detected = False
    if latest_age_days is not None:
        stale_detected = bool(latest_age_days > float(ttl_for_category))
    elif classification.requires_freshness:
        stale_detected = True

    needs_refresh = bool(classification.requires_freshness or temporal_risk >= 0.55 or stale_detected)

    if temporal_risk >= 0.75:
        risk_level = "high"
    elif temporal_risk >= 0.45:
        risk_level = "medium"
    else:
        risk_level = "low"

    reasons: list[str] = []
    if classification.is_temporal:
        reasons.append("temporal_markers")
    if classification.requires_freshness:
        reasons.append("requires_freshness")
    if classification.is_external_fact_question:
        reasons.append("external_fact")
    if classification.is_local_project_question and not classification.requires_freshness:
        reasons.append("local_project_penalty")
    reasons.append(f"category:{category}")
    if latest_age_days is not None:
        reasons.append(f"latest_web_age_days:{latest_age_days:.2f}")
    if stale_detected:
        reasons.append("stale_web_fact_detected")

    return FreshnessAssessment(
        temporal_risk=temporal_risk,
        risk_level=risk_level,
        needs_refresh=needs_refresh,
        stale_web_fact_detected=stale_detected,
        category=category,
        reasons=reasons,
        breakdown=breakdown,
    )


def _detect_freshness_category(text: str) -> str:
    src = str(text or "").lower()
    if any(token in src for token in _PRICE_MARKERS):
        return "prices"
    if any(token in src for token in _VERSION_MARKERS):
        return "versions"
    if any(token in src for token in _NEWS_MARKERS):
        return "news"
    if any(token in src for token in _MARKET_MARKERS):
        return "market_compare"
    return "docs_summary"


def _category_boost(category: str) -> float:
    table = {
        "prices": 0.22,
        "versions": 0.20,
        "news": 0.22,
        "market_compare": 0.18,
        "docs_summary": 0.08,
    }
    return float(table.get(str(category or ""), 0.08))


def _normalized_ttl_days(ttl_days: dict[str, int] | None) -> dict[str, int]:
    src = dict(ttl_days or {})
    out = {
        "default": 7,
        "prices": 1,
        "versions": 14,
        "news": 2,
        "docs_summary": 30,
        "market_compare": 7,
    }
    for key in list(out.keys()):
        raw = src.get(key)
        try:
            if raw is None:
                continue
            out[key] = max(1, int(raw))
        except Exception:
            continue
    return out


def _latest_age_days(*, web_items: list[Any] | None, now: dt.datetime) -> float | None:
    if not web_items:
        return None
    ages: list[float] = []
    for row in list(web_items or []):
        item = dict(row) if isinstance(row, dict) else {}
        candidates = [
            str(item.get("fetched_at") or "").strip(),
            str(item.get("published_date") or "").strip(),
        ]
        dt_value = None
        for value in candidates:
            if not value:
                continue
            dt_value = _parse_date(value)
            if dt_value is not None:
                break
        if dt_value is None:
            continue
        delta = now - dt_value
        ages.append(max(0.0, float(delta.total_seconds()) / 86400.0))
    if not ages:
        return None
    return min(ages)


def _parse_date(value: str) -> dt.datetime | None:
    src = str(value or "").strip()
    if not src:
        return None
    try:
        # ISO timestamps from fetched_at.
        parsed = dt.datetime.fromisoformat(src.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=dt.timezone.utc)
        return parsed.astimezone(dt.timezone.utc)
    except Exception:
        pass
    if re.match(r"^\d{4}-\d{2}-\d{2}$", src):
        try:
            day = dt.datetime.strptime(src, "%Y-%m-%d")
            return day.replace(tzinfo=dt.timezone.utc)
        except Exception:
            return None
    return None

