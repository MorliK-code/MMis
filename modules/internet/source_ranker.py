from __future__ import annotations

"""Legacy source ranker for the pre-web-v2 internet stack.

This file remains for compatibility with older adapters such as
``semantic_search_bridge``. The active ranking path lives in
``modules.internet.web.source_ranker``.
"""

import datetime as dt
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse


@dataclass(slots=True)
class _ScoredResult:
    result: Any
    domain: str
    source_class: str
    score: float


class SourceRanker:
    """
    Legacy ranker for already-fetched search results.

    Intent policies:
    - weather
    - currency_rate
    - news
    - generic
    """

    _INTENT_ALIASES: dict[str, str] = {
        "weather": "weather",
        "currency_rate": "currency_rate",
        "fx_rate": "currency_rate",
        "currency": "currency_rate",
        "news": "news",
        "news_release": "news",
        "generic": "generic",
    }

    _TRUSTED_GENERIC = {
        "docs.python.org",
        "developer.mozilla.org",
        "github.com",
        "openai.com",
        "wikipedia.org",
        "cloud.google.com",
        "learn.microsoft.com",
        "aws.amazon.com",
    }
    _TRUSTED_WEATHER = {
        "weather.com",
        "accuweather.com",
        "open-meteo.com",
        "meteo.ua",
        "sinoptik.ua",
        "gismeteo.ua",
        "weather.gov",
        "metoffice.gov.uk",
    }
    _TRUSTED_CURRENCY = {
        "bank.gov.ua",
        "minfin.com.ua",
        "finance.ua",
        "xe.com",
        "ecb.europa.eu",
        "federalreserve.gov",
        "investing.com",
        "bloomberg.com",
    }
    _TRUSTED_NEWS = {
        "reuters.com",
        "apnews.com",
        "bbc.com",
        "ft.com",
        "wsj.com",
        "nytimes.com",
        "forbes.com",
        "theverge.com",
    }

    _WEAK_PATTERNS = (
        "forum.",
        "community.",
        "discuss.",
        "reddit.com",
        "quora.com",
        "blogspot.",
        "pinterest.com",
        "xn--",
    )

    _DOMAIN_CLASS_BONUS = {
        "trusted": 0.24,
        "open_web": 0.06,
        "weak": -0.20,
    }

    _FRESHNESS_WEIGHT_BY_INTENT = {
        "weather": 0.22,
        "currency_rate": 0.22,
        "news": 0.18,
        "generic": 0.08,
    }

    def classify_domain(self, domain: str, query_intent: str) -> str:
        host = self._normalize_domain(domain)
        if not host:
            return "weak"

        if self._is_weak_domain(host):
            return "weak"

        trusted = self._trusted_domains_for_intent(query_intent)
        if self._matches_any_domain(host, trusted):
            return "trusted"

        if host.endswith(".gov") or host.endswith(".edu"):
            return "trusted"

        return "open_web"

    def score_result(self, result: Any, query_intent: str) -> float:
        intent = self._normalize_intent(query_intent)
        domain = self._extract_domain(result)
        source_class = self.classify_domain(domain, intent)

        base_score = self._extract_base_relevance(result)
        quality_score = self._quality_signal(result)
        freshness_score = self._freshness_signal(result, intent)

        domain_bonus = self._DOMAIN_CLASS_BONUS[source_class]
        total = base_score + quality_score + freshness_score + domain_bonus
        return round(total, 6)

    def select_diverse_results(
        self,
        results: list[Any],
        query_intent: str,
        limit: int = 5,
    ) -> list[Any]:
        max_items = max(1, int(limit))
        if not results:
            return []

        scored = self._score_all(results=results, query_intent=query_intent)
        if not scored:
            return []

        buckets = self._build_domain_buckets(scored)
        domain_order = sorted(
            buckets.keys(),
            key=lambda domain: buckets[domain][0].score if buckets[domain] else -1.0,
            reverse=True,
        )

        selected: list[_ScoredResult] = []
        while len(selected) < max_items:
            progressed = False
            for domain in domain_order:
                bucket = buckets.get(domain) or []
                if not bucket:
                    continue
                candidate = bucket[0]
                if self._would_make_three_in_row(selected, domain):
                    continue
                selected.append(candidate)
                del bucket[0]
                progressed = True
                if len(selected) >= max_items:
                    break

            if progressed:
                continue

            fallback = self._pick_best_non_triplet_candidate(buckets=buckets, selected=selected)
            if fallback is None:
                break
            selected.append(fallback)

        return [item.result for item in selected[:max_items]]

    def _score_all(self, results: list[Any], query_intent: str) -> list[_ScoredResult]:
        intent = self._normalize_intent(query_intent)
        scored: list[_ScoredResult] = []
        for result in results:
            domain = self._extract_domain(result)
            source_class = self.classify_domain(domain, intent)
            score = self.score_result(result, intent)
            scored.append(
                _ScoredResult(
                    result=result,
                    domain=domain,
                    source_class=source_class,
                    score=score,
                )
            )
        scored.sort(key=lambda item: item.score, reverse=True)
        return scored

    @staticmethod
    def _build_domain_buckets(scored: list[_ScoredResult]) -> dict[str, list[_ScoredResult]]:
        buckets: dict[str, list[_ScoredResult]] = {}
        for item in scored:
            key = item.domain or "unknown"
            buckets.setdefault(key, []).append(item)
        return buckets

    def _pick_best_non_triplet_candidate(
        self,
        *,
        buckets: dict[str, list[_ScoredResult]],
        selected: list[_ScoredResult],
    ) -> _ScoredResult | None:
        candidates: list[tuple[str, _ScoredResult]] = []
        for domain, rows in buckets.items():
            if not rows:
                continue
            candidates.append((domain, rows[0]))
        candidates.sort(key=lambda item: item[1].score, reverse=True)

        for domain, candidate in candidates:
            if self._would_make_three_in_row(selected, domain):
                continue
            bucket = buckets.get(domain)
            if bucket:
                del bucket[0]
            return candidate
        return None

    @staticmethod
    def _would_make_three_in_row(selected: list[_ScoredResult], next_domain: str) -> bool:
        if len(selected) < 2:
            return False
        return selected[-1].domain == next_domain and selected[-2].domain == next_domain

    def _trusted_domains_for_intent(self, query_intent: str) -> set[str]:
        intent = self._normalize_intent(query_intent)
        if intent == "weather":
            return self._TRUSTED_WEATHER
        if intent == "currency_rate":
            return self._TRUSTED_CURRENCY
        if intent == "news":
            return self._TRUSTED_NEWS
        return self._TRUSTED_GENERIC

    def _quality_signal(self, result: Any) -> float:
        title = self._safe_text(self._get_field(result, "title"))
        snippet = self._safe_text(self._get_field(result, "snippet"))
        url = self._safe_text(self._get_field(result, "url"))

        quality = 0.0
        if len(title) >= 16:
            quality += 0.08
        elif len(title) >= 8:
            quality += 0.04

        if len(snippet) >= 40:
            quality += 0.08
        elif len(snippet) >= 20:
            quality += 0.04

        if url.startswith("https://"):
            quality += 0.03

        # Weak sources can still participate, but need stronger content signals.
        domain = self._extract_domain(result)
        if self._is_weak_domain(domain):
            quality -= 0.03

        return max(-0.10, min(0.25, quality))

    def _freshness_signal(self, result: Any, query_intent: str) -> float:
        weight = self._FRESHNESS_WEIGHT_BY_INTENT[self._normalize_intent(query_intent)]
        raw_date = self._safe_text(
            self._get_field(result, "published_date")
            or self._get_field(result, "published")
            or self._get_field(result, "date")
        )
        if not raw_date:
            return 0.0

        parsed = self._parse_date(raw_date)
        if parsed is None:
            return 0.0

        age_days = max(0, (dt.datetime.now(dt.timezone.utc).date() - parsed).days)
        if age_days <= 1:
            freshness = 1.0
        elif age_days <= 3:
            freshness = 0.85
        elif age_days <= 7:
            freshness = 0.65
        elif age_days <= 30:
            freshness = 0.35
        else:
            freshness = 0.12

        return round(weight * freshness, 6)

    def _extract_base_relevance(self, result: Any) -> float:
        raw_score = self._get_field(result, "score")
        try:
            score = float(raw_score)
        except Exception:
            score = 0.0

        if score <= 0.0:
            title = self._safe_text(self._get_field(result, "title"))
            snippet = self._safe_text(self._get_field(result, "snippet"))
            # Small fallback relevance signal when upstream score is absent.
            content_len = len(title) + len(snippet)
            return min(0.35, content_len / 400.0)

        return score

    def _extract_domain(self, result: Any) -> str:
        source = self._safe_text(self._get_field(result, "source") or self._get_field(result, "domain"))
        source_host = self._normalize_domain(source)
        if source_host:
            return source_host

        url = self._safe_text(self._get_field(result, "url"))
        if not url:
            return ""
        return self._normalize_domain(urlparse(url).netloc)

    @staticmethod
    def _get_field(result: Any, key: str) -> Any:
        if isinstance(result, dict):
            return result.get(key)
        return getattr(result, key, None)

    def _normalize_intent(self, query_intent: str) -> str:
        token = self._safe_text(query_intent).lower()
        return self._INTENT_ALIASES.get(token, "generic")

    @staticmethod
    def _normalize_domain(domain: str) -> str:
        host = str(domain or "").strip().lower()
        if host.startswith("www."):
            host = host[4:]
        return host

    def _is_weak_domain(self, domain: str) -> bool:
        host = self._normalize_domain(domain)
        if not host:
            return True
        return any(token in host for token in self._WEAK_PATTERNS)

    @staticmethod
    def _matches_any_domain(host: str, trusted_domains: set[str]) -> bool:
        for trusted in trusted_domains:
            if host == trusted or host.endswith(f".{trusted}"):
                return True
        return False

    @staticmethod
    def _safe_text(value: Any) -> str:
        return str(value or "").strip()

    @staticmethod
    def _parse_date(value: str) -> dt.date | None:
        src = str(value or "").strip()
        if not src:
            return None
        try:
            return dt.datetime.fromisoformat(src.replace("Z", "+00:00")).date()
        except Exception:
            pass
        for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%d.%m.%Y"):
            try:
                return dt.datetime.strptime(src, fmt).date()
            except Exception:
                continue
        return None
