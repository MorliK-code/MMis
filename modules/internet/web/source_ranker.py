from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

from modules.internet.search import SearchResult


@dataclass(frozen=True)
class RankedSource:
    item: SearchResult
    trust_score: float
    trust_tier: str
    bonus: float
    quality_score: float = 0.0
    freshness_score: float = 0.0


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


def rank_sources(
    *,
    items: list[SearchResult],
    preferred_domains: list[str] | None = None,
    blocked_domains: list[str] | None = None,
) -> list[RankedSource]:
    preferred = [str(x or "").strip().lower() for x in list(preferred_domains or []) if str(x or "").strip()]
    blocked = [str(x or "").strip().lower() for x in list(blocked_domains or []) if str(x or "").strip()]
    out: list[RankedSource] = []

    for item in list(items or []):
        domain = str(item.source or "").strip().lower()
        if not domain:
            continue
        if _is_blocked(domain=domain, blocked_domains=blocked):
            continue

        trust_score, trust_tier = _trust_for(item=item)
        preferred_bonus = 0.08 if _matches_any_domain(domain, preferred) else 0.0
        freshness_score = _freshness_score(str(item.published_date or "").strip())
        quality_score = _quality_score(item=item)

        base_score = max(0.0, float(item.score or 0.0))
        total_score = base_score + trust_score + preferred_bonus + freshness_score + quality_score
        score_breakdown = {
            **dict(item.score_breakdown or {}),
            "v2_source_trust": round(float(trust_score), 6),
            "v2_preferred_bonus": round(float(preferred_bonus), 6),
            "v2_freshness_bonus": round(float(freshness_score), 6),
            "v2_quality_bonus": round(float(quality_score), 6),
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
                "v2_trust_tier": trust_tier,
                "v2_trust_score": float(trust_score),
            },
        )
        out.append(
            RankedSource(
                item=row,
                trust_score=trust_score,
                trust_tier=trust_tier,
                bonus=preferred_bonus,
                quality_score=quality_score,
                freshness_score=freshness_score,
            )
        )

    out.sort(key=lambda x: float(x.item.score or 0.0), reverse=True)
    return out


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


def _quality_score(*, item: SearchResult) -> float:
    title = str(item.title or "").strip()
    snippet = str(item.snippet or "").strip()
    title_bonus = 0.04 if len(title) >= 12 else 0.0
    snippet_bonus = min(0.05, float(len(snippet)) / 700.0)
    has_digits_bonus = 0.02 if any(ch.isdigit() for ch in (title + " " + snippet)) else 0.0
    return max(0.0, min(0.12, title_bonus + snippet_bonus + has_digits_bonus))


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
