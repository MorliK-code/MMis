from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Any

from modules.internet.search import SearchClient, SearchResult
from modules.internet.scraper import WebScraper
from modules.internet.web.web_models import FreshnessAssessment, QueryClassification, WebPolicyDecision, WebQueryPlan, WebSearchMode


_FX_DOMAIN_FILTER = [
    "minfin.com.ua",
    "finance.ua",
    "kurs.com.ua",
    "obmenka.ua",
    "privatbank.ua",
    "monobank.ua",
    "bank.gov.ua",
]
_WEATHER_DOMAIN_FILTER = [
    "open-meteo.com",
    "sinoptik.ua",
    "meteo.ua",
    "weather.com",
    "accuweather.com",
    "gismeteo.ua",
]


@dataclass(frozen=True)
class SearchExecutionResult:
    results: list[SearchResult]
    fetched_pages: dict[str, dict[str, Any]]
    queries_used: list[str]
    domain_filter: list[str]
    recency_days: int | None


class SearchManager:
    def __init__(
        self,
        *,
        search_client: SearchClient,
        scraper: WebScraper,
        cooldown_seconds: int = 45,
    ):
        self._search = search_client
        self._scraper = scraper
        self._cooldown_seconds = max(0, int(cooldown_seconds))
        self._topic_last_ts: dict[str, float] = {}

    def execute(
        self,
        *,
        plan: WebQueryPlan,
        decision: WebPolicyDecision,
        classification: QueryClassification,
        freshness: FreshnessAssessment,
    ) -> SearchExecutionResult:
        if decision.mode == WebSearchMode.NO_SEARCH:
            return SearchExecutionResult(results=[], fetched_pages={}, queries_used=[], domain_filter=[], recency_days=None)

        budget_queries = max(0, int(decision.budget.max_queries))
        budget_sources = max(0, int(decision.budget.max_sources))
        budget_pages = max(0, int(decision.budget.max_pages))
        intent = _compat_query_intent(query=plan.all_queries()[0] if plan.all_queries() else "", classification=classification)
        recency_days = _recency_days(intent=intent, freshness=freshness)
        domain_filter = _domain_filter_for_intent(intent)

        queries = plan.all_queries()[:budget_queries]
        queries_used: list[str] = []
        collected: list[SearchResult] = []
        seen_urls: set[str] = set()

        for query in queries:
            if len(queries_used) >= budget_queries:
                break
            query_text = str(query or "").strip()
            if not query_text:
                continue
            queries_used.append(query_text)
            rows = self._search.search(
                query_text,
                recency_days=recency_days,
                domain_filter=domain_filter or None,
                k=max(1, budget_sources),
                volatile=bool(freshness.needs_refresh),
                query_intent=intent,
            )
            for item in list(rows or []):
                url = str(item.url or "").strip().lower()
                if not url or url in seen_urls:
                    continue
                seen_urls.add(url)
                collected.append(item)
                if len(collected) >= max(1, budget_sources):
                    break
            if len(collected) >= max(1, budget_sources):
                break

        if not collected and plan.fallback_queries:
            for query in list(plan.fallback_queries):
                if len(queries_used) >= budget_queries:
                    break
                query_text = str(query or "").strip()
                if not query_text:
                    continue
                queries_used.append(query_text)
                rows = self._search.search(
                    query_text,
                    recency_days=recency_days,
                    domain_filter=domain_filter or None,
                    k=max(1, budget_sources),
                    volatile=True,
                    query_intent=intent,
                )
                for item in list(rows or []):
                    url = str(item.url or "").strip().lower()
                    if not url or url in seen_urls:
                        continue
                    seen_urls.add(url)
                    collected.append(item)
                    if len(collected) >= max(1, budget_sources):
                        break
                if len(collected) >= max(1, budget_sources):
                    break

        fetched_pages: dict[str, dict[str, Any]] = {}
        fetched_at = dt.datetime.now(dt.timezone.utc).isoformat()
        for item in list(collected)[:budget_pages]:
            url = str(item.url or "").strip()
            if not url:
                continue
            domain = str(item.source or "").strip().lower()
            try:
                page = self._scraper.scrape(url)
            except Exception:
                fetched_pages[url] = {
                    "url": url,
                    "domain": domain,
                    "status": 0,
                    "final_url": url,
                    "title": str(item.title or "").strip(),
                    "text": "",
                    "snippet": str(item.snippet or "").strip(),
                    "published_date": str(item.published_date or "").strip(),
                    "fetched_at": fetched_at,
                    "clean_method": "search_snippet_fallback",
                    "removed_blocks": 0,
                    "raw_len": 0,
                    "clean_len": 0,
                }
                continue
            meta = dict(getattr(page, "metadata", {}) or {})
            fetched_pages[url] = {
                "url": url,
                "domain": domain,
                "status": int(getattr(page, "status_code", 200) or 200),
                "final_url": str(getattr(page, "final_url", "") or url),
                "title": str(getattr(page, "title", "") or item.title or "").strip(),
                "text": str(getattr(page, "text", "") or "").strip(),
                "snippet": str(item.snippet or "").strip(),
                "published_date": str(item.published_date or "").strip(),
                "fetched_at": fetched_at,
                "clean_method": str(meta.get("clean_method") or "").strip(),
                "removed_blocks": int(meta.get("removed_blocks") or 0),
                "raw_len": int(meta.get("raw_len") or 0),
                "clean_len": int(meta.get("clean_len") or 0),
            }

        return SearchExecutionResult(
            results=collected,
            fetched_pages=fetched_pages,
            queries_used=queries_used,
            domain_filter=list(domain_filter),
            recency_days=recency_days,
        )


def _recency_days(*, intent: str, freshness: FreshnessAssessment) -> int | None:
    if intent in {"fx_rate", "weather"}:
        return 1
    if intent == "news_release":
        return 3
    if freshness.needs_refresh:
        return 7
    return None


def _compat_query_intent(*, query: str, classification: QueryClassification) -> str:
    text = str(query or "").strip().lower()
    if any(token in text for token in ("курс", "usd", "eur", "uah", "exchange rate", "forex")):
        return "fx_rate"
    if any(token in text for token in ("weather", "forecast", "погод", "температур")):
        return "weather"
    if any(token in text for token in ("news", "release", "changelog", "новост", "релиз", "версия")):
        return "news_release"
    if classification.query_type == "external_factual" and classification.primary_category in {"version", "news"}:
        return "news_release"
    return "generic"


def _domain_filter_for_intent(intent: str) -> list[str]:
    if intent == "fx_rate":
        return list(_FX_DOMAIN_FILTER)
    if intent == "weather":
        return list(_WEATHER_DOMAIN_FILTER)
    return []

