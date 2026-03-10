from __future__ import annotations

from dataclasses import dataclass

from modules.internet.search import SearchResult


@dataclass(frozen=True)
class RankedSource:
    item: SearchResult
    trust_score: float
    trust_tier: str
    bonus: float


_OFFICIAL_DOMAINS = (
    "python.org",
    "docs.python.org",
    "openai.com",
    "github.com",
    "pytorch.org",
    "tensorflow.org",
    "huggingface.co",
)
_REPUTABLE_TECH = (
    "developer.mozilla.org",
    "stackoverflow.com",
    "arxiv.org",
    "towardsdatascience.com",
    "kdnuggets.com",
)
_FORUM_DOMAINS = ("reddit.com", "news.ycombinator.com", "community", "forum")


def rank_sources(
    *,
    items: list[SearchResult],
    preferred_domains: list[str] | None = None,
    blocked_domains: list[str] | None = None,
) -> list[RankedSource]:
    preferred = [str(x or "").strip().lower() for x in list(preferred_domains or []) if str(x or "").strip()]
    blocked = {str(x or "").strip().lower() for x in list(blocked_domains or []) if str(x or "").strip()}
    out: list[RankedSource] = []

    for item in list(items or []):
        domain = str(item.source or "").strip().lower()
        if not domain:
            continue
        if domain in blocked:
            continue
        if any(domain.endswith("." + b) for b in blocked):
            continue

        trust_score, trust_tier = _domain_trust(domain)
        preferred_bonus = 0.06 if _matches_any_domain(domain, preferred) else 0.0
        base_score = float(item.score or 0.0) + trust_score + preferred_bonus

        row = SearchResult(
            title=item.title,
            snippet=item.snippet,
            url=item.url,
            source=item.source,
            published_date=item.published_date,
            score=base_score,
            score_breakdown={
                **dict(item.score_breakdown or {}),
                "v2_source_trust": round(float(trust_score), 6),
                "v2_preferred_bonus": round(float(preferred_bonus), 6),
            },
            raw={**dict(item.raw or {}), "v2_trust_tier": trust_tier},
        )
        out.append(
            RankedSource(
                item=row,
                trust_score=trust_score,
                trust_tier=trust_tier,
                bonus=preferred_bonus,
            )
        )

    out.sort(key=lambda x: float(x.item.score or 0.0), reverse=True)
    return out


def _domain_trust(domain: str) -> tuple[float, str]:
    src = str(domain or "").strip().lower()
    if not src:
        return 0.0, "unknown"
    if _matches_any_domain(src, _OFFICIAL_DOMAINS):
        return 0.28, "official"
    if _matches_any_domain(src, _REPUTABLE_TECH):
        return 0.18, "reputable_tech"
    if _matches_any_domain(src, _FORUM_DOMAINS):
        return 0.06, "forum"
    if src.endswith(".gov") or src.endswith(".edu"):
        return 0.22, "institutional"
    return 0.02, "general_web"


def _matches_any_domain(domain: str, candidates: tuple[str, ...] | list[str]) -> bool:
    for candidate in list(candidates or []):
        cur = str(candidate or "").strip().lower()
        if not cur:
            continue
        if domain == cur or domain.endswith("." + cur):
            return True
    return False

