from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Any

from modules.internet.search import SearchResult
from modules.internet.web.domain_reputation import DomainTrustPolicy
from modules.internet.web.numeric_facts import (
    currency_page_type_priority,
    detect_currency_page_type,
    detect_factual_page_type,
    detect_numeric_time_scope,
    detect_requested_currency_rate_type,
    infer_numeric_profile,
)
from modules.internet.web.topical_relevance import assess_source_topical_relevance


@dataclass(frozen=True)
class RankedSource:
    item: SearchResult
    trust_score: float
    trust_tier: str
    bonus: float
    quality_score: float = 0.0
    freshness_score: float = 0.0
    audit: dict[str, Any] = field(default_factory=dict)


_OFFICIAL_DOCS_DOMAINS = (
    "docs.python.org",
    "developer.mozilla.org",
    "docs.pytorch.org",
    "docs.docker.com",
    "kubernetes.io",
)
_OFFICIAL_REPO_DOMAINS = (
    "github.com",
    "gitlab.com",
    "bitbucket.org",
)
_VENDOR_DOMAINS = (
    "openai.com",
    "google.com",
    "microsoft.com",
    "aws.amazon.com",
    "cloudflare.com",
    "huggingface.co",
    "chroma-core.github.io",
)
_REPUTABLE_TECH = (
    "arxiv.org",
    "stackoverflow.com",
    "techcrunch.com",
    "reuters.com",
    "apnews.com",
    "bbc.com",
    "nature.com",
)
_FORUM_DOMAINS = (
    "reddit.com",
    "news.ycombinator.com",
    "community.",
    "forum.",
    "discuss.",
)
_LOW_TRUST_PATTERNS = (
    "blogspot.",
    "medium.com",
    "substack.com",
)
_DOCS_DOMAIN_HINTS = ("docs.", "developer.", "readthedocs", "github.com", "gitlab.com", "pypi.org", "npmjs.com")
_FINANCE_DOMAIN_HINTS = ("bank.", "bank.gov", "minfin", "finance.", "forex", "fx", "kurs", "invest", "marketwatch")
_WEATHER_DOMAIN_HINTS = ("weather", "meteo", "forecast", "sinoptik", "accuweather", "gismeteo")
_NEWS_DOMAIN_HINTS = ("news", "reuters", "apnews", "bbc", "ukrinform", "cnn", "nytimes", "wsj")
_HISTORICAL_DOMAIN_HINTS = ("wikipedia", "britannica", "history", "archive", "museum", ".gov", ".edu", "reuters", "apnews")

_MID_TRUST_TIERS = {"community_verified", "learned_mid", "policy_preferred"}
_OPEN_WEB_TIERS = {"general_web", "forum_discussion"}
_WEAK_TIERS = {"blog_random", "unknown", "degraded_source", "policy_risky", "policy_degraded"}


def rank_sources(
    *,
    items: list[SearchResult],
    preferred_domains: list[str] | None = None,
    blocked_domains: list[str] | None = None,
    trust_policy: DomainTrustPolicy | dict[str, Any] | None = None,
    reputation_scores: dict[str, float] | None = None,
    reputation_stats: dict[str, dict[str, Any]] | None = None,
    query_intent: str = "generic",
    query_category: str = "",
    query_text: str = "",
    geo_hint: str = "",
    limit_hint: int = 0,
) -> list[RankedSource]:
    preferred = [str(x or "").strip().lower() for x in list(preferred_domains or []) if str(x or "").strip()]
    blocked = [str(x or "").strip().lower() for x in list(blocked_domains or []) if str(x or "").strip()]
    reputation = {
        str(k or "").strip().lower(): _clamp(float(v), -1.0, 1.0)
        for k, v in dict(reputation_scores or {}).items()
        if str(k or "").strip()
    }
    policy = _resolve_trust_policy(trust_policy).with_runtime(
        preferred_domains=preferred,
        blocked_domains=blocked,
    )
    out: list[RankedSource] = []

    for item in list(items or []):
        domain = str(item.source or "").strip().lower()
        if not domain:
            continue

        audit = build_source_audit_entry(
            item=item,
            preferred_domains=preferred,
            blocked_domains=blocked,
            trust_policy=policy,
            reputation_scores=reputation,
            reputation_stats=reputation_stats,
            query_category=query_category,
            query_text=query_text,
            geo_hint=geo_hint,
        )
        if bool(audit.get("blocked_hit")):
            continue

        base_trust_score = float(audit.get("base_trust_score") or 0.0)
        base_trust_tier = str(audit.get("base_trust_tier") or "unknown")
        reputation_score = float(audit.get("reputation_score") or 0.0)
        reputation_bonus = float(audit.get("reputation_bonus") or 0.0)
        policy_bonus = float(audit.get("policy_bonus") or 0.0)
        trust_tier = str(audit.get("trust_tier") or "unknown")
        effective_trust_score = float(audit.get("trust_score") or 0.0)
        freshness_score = float(audit.get("freshness_score") or 0.0)
        quality_score = float(audit.get("quality_score") or 0.0)
        geo_score = float(audit.get("geo_bonus") or 0.0)
        category_bonus = float(audit.get("category_bonus") or 0.0)
        topical_bonus = float(audit.get("topical_bonus") or 0.0)
        currency_page_bonus = float(audit.get("currency_page_bonus") or 0.0)
        factual_page_bonus = float(audit.get("factual_page_bonus") or 0.0)
        direct_answer_bonus = float(audit.get("direct_answer_bonus") or 0.0)

        base_score = max(0.0, float(item.score or 0.0))
        total_score = (
            base_score
            + base_trust_score
            + policy_bonus
            + freshness_score
            + quality_score
            + reputation_bonus
            + geo_score
            + category_bonus
            + topical_bonus
            + currency_page_bonus
            + factual_page_bonus
            + direct_answer_bonus
        )
        preferred_bonus = float(audit.get("preferred_bonus") or 0.0)
        audit["ranking_score_total"] = round(float(total_score), 6)
        audit["filtered_out_reason"] = ""
        audit["selected_for_evidence"] = False
        score_breakdown = {
            **dict(item.score_breakdown or {}),
            "v2_source_trust_base": round(float(base_trust_score), 6),
            "v2_source_trust": round(float(effective_trust_score), 6),
            "v2_reputation_score": round(float(reputation_score), 6),
            "v2_reputation_bonus": round(float(reputation_bonus), 6),
            "v2_policy_bonus": round(float(policy_bonus), 6),
            "v2_preferred_bonus": round(float(preferred_bonus), 6),
            "v2_freshness_bonus": round(float(freshness_score), 6),
            "v2_quality_bonus": round(float(quality_score), 6),
            "v2_geo_bonus": round(float(geo_score), 6),
            "v2_category_bonus": round(float(category_bonus), 6),
            "v2_topical_bonus": round(float(topical_bonus), 6),
            "v2_currency_page_bonus": round(float(currency_page_bonus), 6),
            "v2_factual_page_bonus": round(float(factual_page_bonus), 6),
            "v2_direct_answer_bonus": round(float(direct_answer_bonus), 6),
        }
        row = SearchResult(
            title=item.title,
            snippet=item.snippet,
            url=item.url,
            source=item.source,
            published_date=item.published_date,
            score=total_score,
            score_breakdown=score_breakdown,
            raw={
                **dict(item.raw or {}),
                "v2_base_trust_tier": base_trust_tier,
                "v2_trust_tier": trust_tier,
                "v2_base_trust_score": float(base_trust_score),
                "v2_trust_score": float(effective_trust_score),
                "v2_policy_state": str(audit.get("policy_state") or "neutral"),
                "v2_policy_bonus": float(policy_bonus),
                "v2_policy_reasons": list(audit.get("policy_reasons") or []),
                "v2_manual_override": str(audit.get("manual_override") or ""),
                "v2_reputation_state": str(audit.get("reputation_state") or "neutral"),
                "v2_reputation_score": float(reputation_score),
                "v2_reputation_bonus": float(reputation_bonus),
                "v2_source_audit": dict(audit),
            },
        )
        out.append(
            RankedSource(
                item=row,
                trust_score=effective_trust_score,
                trust_tier=trust_tier,
                bonus=(policy_bonus + reputation_bonus),
                quality_score=quality_score,
                freshness_score=freshness_score,
                audit=dict(audit),
            )
        )

    out.sort(key=lambda x: float(x.item.score or 0.0), reverse=True)
    out = _rebalance_trust_mix(out, query_intent=query_intent, limit_hint=limit_hint)
    out = _rebalance_domain_streak(out, max_streak=2)
    return out


def build_source_audit_entry(
    *,
    item: SearchResult,
    preferred_domains: list[str] | None = None,
    blocked_domains: list[str] | None = None,
    trust_policy: DomainTrustPolicy | dict[str, Any] | None = None,
    reputation_scores: dict[str, float] | None = None,
    reputation_stats: dict[str, dict[str, Any]] | None = None,
    query_category: str = "",
    query_text: str = "",
    geo_hint: str = "",
) -> dict[str, Any]:
    preferred = [str(x or "").strip().lower() for x in list(preferred_domains or []) if str(x or "").strip()]
    blocked = [str(x or "").strip().lower() for x in list(blocked_domains or []) if str(x or "").strip()]
    reputation = {
        str(k or "").strip().lower(): _clamp(float(v), -1.0, 1.0)
        for k, v in dict(reputation_scores or {}).items()
        if str(k or "").strip()
    }
    policy = _resolve_trust_policy(trust_policy).with_runtime(
        preferred_domains=preferred,
        blocked_domains=blocked,
    )
    domain = str(item.source or "").strip().lower()
    url = str(item.url or "").strip()
    title = str(item.title or "").strip()
    base_trust_score, base_trust_tier = _trust_for(item=item)
    reputation_score = _reputation_score(domain=domain, reputation=reputation)
    trust_assessment = policy.evaluate(
        domain,
        reputation_score=reputation_score,
        query_category=query_category,
    )
    policy_bonus = _policy_bonus(trust_assessment=trust_assessment, base_tier=base_trust_tier)
    reputation_bonus = _reputation_bonus(
        reputation_score,
        base_tier=base_trust_tier,
        policy_state=str(trust_assessment.policy_state or ""),
    )
    trust_tier = (
        "policy_blocked"
        if bool(trust_assessment.hard_blocked)
        else _apply_effective_tier(
            base_tier=base_trust_tier,
            trust_assessment=trust_assessment,
            reputation_score=reputation_score,
        )
    )
    effective_trust_score = _clamp(base_trust_score + policy_bonus + reputation_bonus, 0.0, 1.0)
    freshness_score = _freshness_score(str(item.published_date or "").strip())
    source_quality_score = _quality_score(item=item)
    geo_bonus = _geo_bonus(domain=domain, geo_hint=geo_hint)
    category_bonus = _category_bonus(domain=domain, query_category=query_category)
    topical = assess_source_topical_relevance(
        domain=domain,
        url=url,
        title=title,
        snippet=str(item.snippet or "").strip(),
        query_category=query_category,
        query_text=query_text,
    )
    currency_page_type, requested_rate_type, currency_page_bonus = _currency_page_adjustment(
        url=url,
        title=title,
        snippet=str(item.snippet or "").strip(),
        query_category=query_category,
        query_text=query_text,
    )
    factual_page_type, factual_page_bonus, direct_answer_score, direct_answer_bonus = _factual_page_adjustment(
        url=url,
        title=title,
        snippet=str(item.snippet or "").strip(),
        query_category=query_category,
        query_text=query_text,
    )
    topical_bonus = float(topical.bonus) - min(0.30, 0.34 * float(topical.penalty))
    policy_state = str(trust_assessment.policy_state or "neutral")
    preferred_bonus = policy_bonus if policy_state == "preferred" else 0.0
    rep_stats = _reputation_stats_for(domain=domain, reputation_stats=reputation_stats)

    return {
        "domain": domain,
        "url": url,
        "title": title,
        "base_trust_tier": str(base_trust_tier),
        "trust_tier": str(trust_tier),
        "base_trust_score": float(base_trust_score),
        "trust_score": float(effective_trust_score),
        "quality_score": float(source_quality_score),
        "ranking_quality_score": float(source_quality_score),
        "freshness_score": float(freshness_score),
        "freshness_tag": _freshness_tag_from_score(freshness_score=freshness_score),
        "preferred_hit": bool(policy_state == "preferred"),
        "trusted_hit": bool(policy_state == "trusted"),
        "blocked_hit": bool(trust_assessment.hard_blocked),
        "risky_hit": bool(policy_state in {"risky", "degraded"}),
        "selected_for_evidence": False,
        "filtered_out_reason": "blocked_by_policy" if bool(trust_assessment.hard_blocked) else "",
        "selection_reason": "",
        "policy_state": policy_state,
        "policy_category": str(query_category or "").strip().lower(),
        "policy_bonus": float(policy_bonus),
        "preferred_bonus": float(preferred_bonus),
        "policy_reasons": [str(x or "").strip() for x in list(getattr(trust_assessment, "reasons", ()) or ()) if str(x or "").strip()],
        "manual_override": str(getattr(trust_assessment, "manual_override", "") or ""),
        "reputation_score": float(reputation_score),
        "reputation_bonus": float(reputation_bonus),
        "reputation_state": str(getattr(trust_assessment, "reputation_state", "neutral") or "neutral"),
        "reputation_stats": dict(rep_stats),
        "geo_bonus": float(geo_bonus),
        "category_bonus": float(category_bonus),
        "topical_profile": str(topical.profile),
        "page_type": str(topical.page_type),
        "topical_relevance_score": float(topical.score),
        "topical_bonus": float(topical_bonus),
        "topical_allow_selection": bool(topical.allow_selection),
        "topical_reasons": [str(x or "").strip() for x in list(topical.reasons or []) if str(x or "").strip()],
        "topical_flags": [str(x or "").strip() for x in list(topical.flags or []) if str(x or "").strip()],
        "currency_page_type": str(currency_page_type or ""),
        "currency_requested_rate_type": str(requested_rate_type or ""),
        "currency_page_bonus": float(currency_page_bonus),
        "factual_page_type": str(factual_page_type or ""),
        "factual_page_bonus": float(factual_page_bonus),
        "direct_answer_score": float(direct_answer_score),
        "direct_answer_bonus": float(direct_answer_bonus),
    }


def _trust_for(*, item: SearchResult) -> tuple[float, str]:
    domain = str(item.source or "").strip().lower()
    url = str(item.url or "").strip().lower()
    if not domain:
        return (0.0, "unknown")

    if _matches_any_domain(domain, _OFFICIAL_DOCS_DOMAINS) or domain.startswith("docs."):
        return (0.34, "official_docs")
    if _matches_any_domain(domain, _OFFICIAL_REPO_DOMAINS):
        if "/releases" in url or "/tags" in url:
            return (0.33, "official_repo_release")
        return (0.30, "official_repo")
    if _matches_any_domain(domain, _VENDOR_DOMAINS):
        return (0.28, "vendor_docs")
    if domain.endswith(".gov") or domain.endswith(".edu"):
        return (0.24, "institutional")
    if _matches_any_domain(domain, _REPUTABLE_TECH):
        return (0.18, "reputable_tech")
    if _matches_any_domain(domain, _FORUM_DOMAINS):
        return (0.07, "forum_discussion")
    if _matches_any_domain(domain, _LOW_TRUST_PATTERNS):
        return (0.01, "blog_random")
    return (0.03, "general_web")


def _reputation_score(*, domain: str, reputation: dict[str, float]) -> float:
    src = str(domain or "").strip().lower()
    if not src:
        return 0.0
    if src in reputation:
        return _clamp(float(reputation.get(src) or 0.0), -1.0, 1.0)
    for key, value in reputation.items():
        if src == key or src.endswith("." + key):
            return _clamp(float(value), -1.0, 1.0)
    return 0.0


def _resolve_trust_policy(value: DomainTrustPolicy | dict[str, Any] | None) -> DomainTrustPolicy:
    if isinstance(value, DomainTrustPolicy):
        return value
    if isinstance(value, dict):
        return DomainTrustPolicy.from_config(value)
    return DomainTrustPolicy()


def _policy_bonus(*, trust_assessment, base_tier: str) -> float:
    state = str(getattr(trust_assessment, "policy_state", "") or "").strip().lower()
    manual = bool(str(getattr(trust_assessment, "manual_override", "") or "").strip())
    tier = str(base_tier or "").strip().lower()
    if tier in _WEAK_TIERS or tier in _OPEN_WEB_TIERS:
        positive_scale = 1.0
        negative_scale = 1.0
    elif tier in _MID_TRUST_TIERS:
        positive_scale = 0.7
        negative_scale = 0.8
    else:
        positive_scale = 0.5
        negative_scale = 0.6

    manual_boost = 0.04 if manual else 0.0
    if state == "trusted":
        return (0.16 * positive_scale) + manual_boost
    if state == "preferred":
        return (0.10 * positive_scale) + (0.02 if manual else 0.0)
    if state == "risky":
        return -((0.14 * negative_scale) + manual_boost)
    if state == "degraded":
        return -((0.22 * negative_scale) + manual_boost)
    return 0.0


def _reputation_bonus(reputation_score: float, *, base_tier: str, policy_state: str = "") -> float:
    score = _clamp(float(reputation_score), -1.0, 1.0)
    tier = str(base_tier or "").strip().lower()
    policy = str(policy_state or "").strip().lower()
    if tier in {"official_docs", "official_repo_release", "official_repo", "vendor_docs", "institutional"}:
        scale = 0.04
    elif tier in {"reputable_tech"}:
        scale = 0.06
    else:
        scale = 0.12
    if policy in {"trusted", "preferred"} and score > 0.0:
        scale *= 0.85
    if policy in {"risky", "degraded"} and score < 0.0:
        scale *= 1.15
    return score * scale


def _apply_reputation_tier(*, base_tier: str, reputation_score: float) -> str:
    tier = str(base_tier or "").strip().lower()
    score = _clamp(float(reputation_score), -1.0, 1.0)
    if score >= 0.40 and tier in {"general_web", "forum_discussion", "blog_random", "unknown"}:
        return "learned_mid"
    if score <= -0.55 and tier in {"general_web", "forum_discussion", "blog_random", "unknown"}:
        return "degraded_source"
    return tier


def _apply_effective_tier(*, base_tier: str, trust_assessment, reputation_score: float) -> str:
    state = str(getattr(trust_assessment, "policy_state", "") or "").strip().lower()
    tier = str(base_tier or "").strip().lower()
    if state == "trusted":
        if tier in _OPEN_WEB_TIERS or tier in _WEAK_TIERS or tier in {"unknown"}:
            return "policy_trusted"
        return tier
    if state == "preferred":
        if tier in _OPEN_WEB_TIERS or tier in _WEAK_TIERS or tier in {"unknown"}:
            return "policy_preferred"
        return tier
    if state == "risky":
        return "policy_risky"
    if state == "degraded":
        return "policy_degraded"
    return _apply_reputation_tier(base_tier=tier, reputation_score=reputation_score)


def _quality_score(*, item: SearchResult) -> float:
    title = str(item.title or "").strip()
    snippet = str(item.snippet or "").strip()
    title_bonus = 0.04 if len(title) >= 12 else 0.0
    snippet_bonus = min(0.05, float(len(snippet)) / 700.0)
    has_digits_bonus = 0.02 if any(ch.isdigit() for ch in (title + " " + snippet)) else 0.0
    return max(0.0, min(0.12, title_bonus + snippet_bonus + has_digits_bonus))


def _reputation_stats_for(*, domain: str, reputation_stats: dict[str, dict[str, Any]] | None) -> dict[str, Any]:
    src = str(domain or "").strip().lower()
    if not src:
        return {}
    stats = dict(reputation_stats or {})
    if src in stats and isinstance(stats.get(src), dict):
        row = dict(stats.get(src) or {})
    else:
        row = {}
        for key, value in stats.items():
            token = str(key or "").strip().lower()
            if src == token or src.endswith("." + token):
                row = dict(value or {}) if isinstance(value, dict) else {}
                break
    if not row:
        return {}
    return {
        "score": float(row.get("score") or 0.0),
        "evidence_count": int(row.get("evidence_count") or 0),
        "positive_count": int(row.get("positive_count") or 0),
        "negative_count": int(row.get("negative_count") or 0),
        "last_updated": str(row.get("last_updated") or ""),
    }


def _freshness_score(published_date: str) -> float:
    parsed = _parse_day(published_date)
    if parsed is None:
        return 0.0
    age_days = max(0, (dt.datetime.now(dt.timezone.utc).date() - parsed).days)
    if age_days <= 2:
        return 0.05
    if age_days <= 7:
        return 0.035
    if age_days <= 30:
        return 0.02
    return 0.0


def _freshness_tag_from_score(*, freshness_score: float) -> str:
    value = max(0.0, float(freshness_score))
    if value >= 0.05:
        return "fresh"
    if value >= 0.02:
        return "recent"
    if value > 0.0:
        return "recent"
    return "unknown"


def _parse_day(value: str) -> dt.date | None:
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


def _is_blocked(*, domain: str, blocked_domains: list[str]) -> bool:
    src = str(domain or "").strip().lower()
    for row in list(blocked_domains or []):
        token = str(row or "").strip().lower()
        if not token:
            continue
        if src == token or src.endswith("." + token):
            return True
    return False


def _matches_any_domain(domain: str, candidates: tuple[str, ...] | list[str]) -> bool:
    src = str(domain or "").strip().lower()
    if not src:
        return False
    for candidate in list(candidates or []):
        cur = str(candidate or "").strip().lower()
        if not cur:
            continue
        if "." in cur:
            if src == cur or src.endswith("." + cur):
                return True
            continue
        if cur in src:
            return True
    return False


def _rebalance_trust_mix(rows: list[RankedSource], *, query_intent: str, limit_hint: int) -> list[RankedSource]:
    ranked = list(rows or [])
    if len(ranked) <= 2:
        return ranked

    trusted: list[RankedSource] = []
    mid: list[RankedSource] = []
    open_web: list[RankedSource] = []
    weak: list[RankedSource] = []

    for row in ranked:
        tier = str(row.trust_tier or "").strip().lower()
        if tier in _WEAK_TIERS:
            weak.append(row)
        elif tier in _MID_TRUST_TIERS:
            mid.append(row)
        elif tier in _OPEN_WEB_TIERS:
            open_web.append(row)
        else:
            trusted.append(row)

    if not trusted and not mid:
        return ranked

    high_stakes = _is_high_stakes_intent(query_intent)
    trusted_ratio = 0.60 if high_stakes else 0.40
    cap = max(3, int(limit_hint) if int(limit_hint or 0) > 0 else min(8, len(ranked)))
    desired_trusted = int(round(float(cap) * trusted_ratio))
    desired_open = max(0, cap - desired_trusted)

    trusted_pool: list[RankedSource] = []
    trusted_pool.extend(trusted)
    trusted_pool.extend(mid)
    open_pool = list(open_web)
    weak_pool = list(weak)

    selected: list[RankedSource] = []
    trusted_used = 0
    open_used = 0
    while len(selected) < cap and (trusted_pool or open_pool):
        candidate: RankedSource | None = None
        if trusted_used < desired_trusted and trusted_pool:
            candidate = trusted_pool.pop(0)
            trusted_used += 1
        elif open_used < desired_open and open_pool:
            candidate = open_pool.pop(0)
            open_used += 1
        elif trusted_pool:
            candidate = trusted_pool.pop(0)
            trusted_used += 1
        elif open_pool:
            candidate = open_pool.pop(0)
            open_used += 1
        if candidate is None:
            break
        selected.append(candidate)

    tail = list(trusted_pool) + list(open_pool) + list(weak_pool)
    return selected + tail


def _is_trusted_tier(value: str) -> bool:
    tier = str(value or "").strip().lower()
    return tier not in _WEAK_TIERS and tier not in _OPEN_WEB_TIERS


def _is_high_stakes_intent(query_intent: str) -> bool:
    token = str(query_intent or "").strip().lower()
    return token in {"weather", "fx_rate", "currency_rate", "news", "news_release"}


def _geo_bonus(*, domain: str, geo_hint: str) -> float:
    host = str(domain or "").strip().lower()
    geo = str(geo_hint or "").strip().lower()
    if not host or not geo:
        return 0.0
    if host.startswith("www."):
        host = host[4:]
    if any(token in geo for token in ("ukraine", "kyiv", "kiev", "украин", "україн", "киев", "київ", "ua")):
        if host.endswith(".ua") or host.endswith(".com.ua"):
            return 0.12
        if host.endswith(".pl") or host.endswith(".de"):
            return 0.02
    return 0.0


def _category_bonus(*, domain: str, query_category: str) -> float:
    category = str(query_category or "").strip().lower()
    host = str(domain or "").strip().lower()
    if not category or not host:
        return 0.0
    if _matches_any_domain(host, _LOW_TRUST_PATTERNS):
        return 0.0
    if category in {"version", "docs"}:
        if _matches_any_hint(host, _DOCS_DOMAIN_HINTS):
            return 0.06
        if _matches_any_hint(host, _FINANCE_DOMAIN_HINTS + _WEATHER_DOMAIN_HINTS):
            return -0.08
        return 0.0
    if category == "finance":
        if _matches_any_hint(host, _FINANCE_DOMAIN_HINTS):
            return 0.08
        if _matches_any_hint(host, _DOCS_DOMAIN_HINTS + _WEATHER_DOMAIN_HINTS):
            return -0.12
        return 0.0
    if category == "price":
        if _matches_any_hint(host, _FINANCE_DOMAIN_HINTS + _NEWS_DOMAIN_HINTS):
            return 0.06
        if _matches_any_hint(host, _DOCS_DOMAIN_HINTS + _WEATHER_DOMAIN_HINTS):
            return -0.12
        return 0.0
    if category == "weather":
        if _matches_any_hint(host, _WEATHER_DOMAIN_HINTS):
            return 0.08
        if _matches_any_hint(host, _DOCS_DOMAIN_HINTS + _FINANCE_DOMAIN_HINTS):
            return -0.12
        return 0.0
    if category == "news":
        if _matches_any_hint(host, _NEWS_DOMAIN_HINTS):
            return 0.06
        if _matches_any_hint(host, _DOCS_DOMAIN_HINTS):
            return -0.08
        return 0.0
    if category == "external":
        if _matches_any_hint(host, _HISTORICAL_DOMAIN_HINTS + _NEWS_DOMAIN_HINTS):
            return 0.05
        if _matches_any_hint(host, _DOCS_DOMAIN_HINTS):
            return -0.08
        return 0.0
    return 0.0


def _currency_page_adjustment(*, url: str, title: str, snippet: str, query_category: str, query_text: str) -> tuple[str, str, float]:
    if str(query_category or "").strip().lower() != "finance":
        return ("", "", 0.0)
    page_type = detect_currency_page_type(url=url, title=title, snippet=snippet)
    requested = detect_requested_currency_rate_type(query_text)
    time_scope = detect_numeric_time_scope(query_text, profile="fx_rate")
    priority = currency_page_type_priority(
        page_type=page_type,
        requested_rate_type=requested,
        query_time_scope=time_scope,
    )
    if requested and page_type == requested:
        return (page_type, requested, 0.08)
    if requested and page_type and page_type != requested:
        return (page_type, requested, -0.10)
    if time_scope == "current" and page_type == "historical_rate":
        return (page_type, requested, -0.12)
    if page_type == "currency_overview":
        return (page_type, requested, 0.06 * priority)
    if page_type in {"cash_rate", "bank_rate", "nbu_rate"}:
        return (page_type, requested, 0.05 * priority)
    if page_type == "currency_index":
        return (page_type, requested, -0.02)
    if page_type == "currency_page":
        return (page_type, requested, 0.01)
    return (page_type, requested, 0.0)


def _factual_page_adjustment(*, url: str, title: str, snippet: str, query_category: str, query_text: str) -> tuple[str, float, float, float]:
    profile = infer_numeric_profile(query_category=query_category, query_text=query_text)
    if profile == "generic":
        return ("", 0.0, 0.0, 0.0)
    page_type = detect_factual_page_type(
        url=url,
        title=title,
        snippet=snippet,
        query_category=query_category,
        query_text=query_text,
    )
    page_bonus = 0.0
    direct_answer_score = 0.0
    low = " ".join(part for part in (str(title or "").strip(), str(snippet or "").strip()) if part).lower()
    if profile == "fx_rate":
        if page_type in {"cash_rate", "official_rate", "nbu_rate", "bank_rate", "overview_page"}:
            page_bonus += 0.05
            direct_answer_score += 0.58
        elif page_type == "index_or_aggregate":
            page_bonus -= 0.03
            direct_answer_score += 0.22
        elif page_type == "historical_rate":
            page_bonus -= 0.08
        elif page_type == "news_or_analysis":
            page_bonus -= 0.06
            direct_answer_score += 0.12
        if any(token in low for token in ("usd/uah", "eur/uah", "exchange rate", "official rate", "cash rate")):
            direct_answer_score += 0.24
    elif profile == "historical":
        if page_type in {"historical_rate", "generic_reference"}:
            page_bonus += 0.05
            direct_answer_score += 0.52
        elif page_type == "news_or_analysis":
            page_bonus -= 0.02
            direct_answer_score += 0.18
        if any(token in low for token in ("timeline", "history", "historical", "date", "founded", "treaty")):
            direct_answer_score += 0.20
    elif profile == "weather":
        if page_type == "overview_page":
            page_bonus += 0.05
            direct_answer_score += 0.54
        elif page_type == "news_or_analysis":
            page_bonus -= 0.04
            direct_answer_score += 0.14
        elif page_type == "historical_rate":
            page_bonus -= 0.03
        if any(token in low for token in ("temperature", "forecast", "degrees", "weather")):
            direct_answer_score += 0.22
    elif profile == "price":
        if page_type in {"overview_page", "official_rate", "generic_reference"}:
            page_bonus += 0.04
            direct_answer_score += 0.44
        elif page_type == "news_or_analysis":
            page_bonus -= 0.03
            direct_answer_score += 0.16
        if any(token in low for token in ("price", "cost", "pricing")):
            direct_answer_score += 0.22
    direct_answer_score = _clamp(direct_answer_score, 0.0, 1.0)
    direct_answer_bonus = direct_answer_score * 0.04
    return (page_type, page_bonus, direct_answer_score, direct_answer_bonus)


def _matches_any_hint(domain: str, hints: tuple[str, ...]) -> bool:
    host = str(domain or "").strip().lower()
    if not host:
        return False
    return any(str(hint or "").strip().lower() in host for hint in hints)


def _rebalance_domain_streak(rows: list[RankedSource], *, max_streak: int) -> list[RankedSource]:
    if max_streak <= 0:
        return list(rows or [])
    pool = list(rows or [])
    if len(pool) <= max_streak:
        return pool
    out: list[RankedSource] = []
    while pool:
        pick_idx = 0
        for idx, row in enumerate(pool):
            domain = str(row.item.source or "").strip().lower()
            if not domain:
                pick_idx = idx
                break
            if len(out) < max_streak:
                pick_idx = idx
                break
            recent_domains = [str(x.item.source or "").strip().lower() for x in out[-max_streak:]]
            if all(domain == recent for recent in recent_domains):
                continue
            pick_idx = idx
            break
        out.append(pool.pop(pick_idx))
    return out


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))
