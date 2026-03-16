from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from modules.internet.search import SearchClient, SearchResult, _build_search_url, _rank_results
from modules.internet.scraper import WebScraper
from modules.internet.web.query_classifier import classify_query
from modules.internet.web.query_planner import build_query_plan
from modules.internet.web.search_manager import SearchManager
from modules.internet.web.web_models import FreshnessAssessment, SearchBudget, WebPolicyDecision, WebQueryPlan, WebSearchMode


class _DummyPage:
    def __init__(self, url: str):
        self.final_url = url
        self.title = "stub"
        self.text = "stub text"
        self.status_code = 200
        self.metadata = {"clean_method": "stub", "removed_blocks": 0, "raw_len": 10, "clean_len": 9}


class _DummyScraper(WebScraper):
    def __init__(self):
        pass

    def scrape(self, url: str):  # noqa: ANN001
        return _DummyPage(url)


class _StubMetaSearchClient:
    def search_with_meta(self, query: str, **kwargs):  # noqa: ANN001
        return (
            [
                SearchResult(
                    title="USD/UAH current rate",
                    snippet="USD/UAH current exchange rate today.",
                    url="https://minfin.com.ua/currency/usd/",
                    source="minfin.com.ua",
                    score=0.71,
                )
            ],
            {
                "provider": "searxng",
                "raw_result_count": 3,
                "reported_result_count": 0,
                "usable_results_count": 1,
                "effective_success": True,
                "effective_success_reason": "results_present_number_of_results_zero",
                "effective_search_confidence": 0.66,
                "engine_success_count": 1,
                "engine_failure_count": 2,
                "engine_health": {"brave": "healthy", "qwant": "blocked", "bing": "degraded"},
                "engines_succeeded": ["brave"],
                "engines_failed": [
                    {"engine": "qwant", "reason": "Access denied", "state": "blocked"},
                    {"engine": "bing", "reason": "DNS resolve error", "state": "degraded"},
                ],
                "query_locale": str(kwargs.get("query_locale") or ""),
                "region_bias": str(kwargs.get("region_bias") or ""),
            },
        )


class SearchLayerPolicyTests(unittest.TestCase):
    @staticmethod
    def _decision() -> WebPolicyDecision:
        return WebPolicyDecision(
            mode=WebSearchMode.TARGETED_SEARCH,
            should_search=True,
            web_need_score=0.91,
            reason="test",
            budget=SearchBudget(max_queries=4, max_sources=4, max_pages=3, max_fetches=3),
        )

    @staticmethod
    def _freshness() -> FreshnessAssessment:
        return FreshnessAssessment(
            temporal_risk=0.42,
            risk_level="medium",
            needs_refresh=True,
            stale_web_fact_detected=False,
            category="finance",
        )

    def test_build_search_url_uses_normalized_language_param(self) -> None:
        url = _build_search_url(
            "http://127.0.0.1:8080/search?format=json",
            query="usd uah exchange rate",
            query_locale="uk_UA",
        )
        self.assertIn("language=uk-UA", url)
        self.assertNotIn("ZZ-ZZ", url)

    def test_engine_health_is_extracted_from_searxng_payload(self) -> None:
        payload = json.dumps(
            {
                "number_of_results": 0,
                "results": [
                    {
                        "url": "https://minfin.com.ua/currency/usd/",
                        "title": "USD cash rate",
                        "content": "USD/UAH current exchange rate.",
                        "engines": ["brave"],
                    }
                ],
                "unresponsive_engines": [
                    {"engine": "qwant", "reason": "Access denied"},
                    {"engine": "bing", "reason": "DNS resolve error"},
                ],
            }
        )
        client = SearchClient(
            endpoint="https://searx.example/search?format=json",
            strict_endpoint=True,
            use_disk_cache=False,
        )
        with patch("modules.internet.search._http_get", return_value=payload):
            rows, debug = client._search_endpoint_with_meta(
                "usd uah exchange rate",
                query_locale="uk-UA",
            )

        self.assertEqual(len(rows), 1)
        self.assertEqual(debug["engine_health"]["brave"], "healthy")
        self.assertEqual(debug["engine_health"]["qwant"], "blocked")
        self.assertEqual(debug["engine_health"]["bing"], "degraded")
        self.assertEqual(debug["engine_success_count"], 1)
        self.assertEqual(debug["engine_failure_count"], 2)

    def test_uah_region_bias_prefers_ua_domain_over_ru_domain(self) -> None:
        ranked = _rank_results(
            items=[
                SearchResult(
                    title="USD/UAH current exchange rate",
                    snippet="Current USD/UAH market rate.",
                    url="https://finance.ua/currency/usd",
                    source="finance.ua",
                ),
                SearchResult(
                    title="USD/UAH current exchange rate",
                    snippet="Current USD/UAH market rate.",
                    url="https://forex.ru/usd-uah",
                    source="forex.ru",
                ),
            ],
            query="курс доллара в грн",
            recency_days=1,
            query_intent="fx_rate",
            query_category="finance",
            region_bias="ua",
        )

        self.assertEqual(ranked[0].source, "finance.ua")
        self.assertGreater(ranked[0].score_breakdown.get("region_bias", 0.0), 0.0)
        self.assertLess(ranked[1].score_breakdown.get("region_bias", 0.0), 0.0)

    def test_search_manager_reports_region_bias_engine_health_and_confidence(self) -> None:
        manager = SearchManager(
            search_client=_StubMetaSearchClient(),
            scraper=_DummyScraper(),
            cooldown_seconds=0,
            search_policy={
                "default_locale": "ru-RU",
                "locale_by_category": {"finance": "uk-UA"},
                "region_locale_overrides": {"ua": "uk-UA"},
            },
        )
        classification = classify_query("курс доллара в грн")
        result = manager.execute(
            plan=WebQueryPlan(mode=WebSearchMode.TARGETED_SEARCH, scout_queries=["usd uah exchange rate"]),
            decision=self._decision(),
            classification=classification,
            freshness=self._freshness(),
            preferred_domains=[],
            geo_hint="",
        )

        self.assertTrue(result.search_debug["effective_success"])
        self.assertEqual(result.search_debug["region_bias"], "ua")
        self.assertEqual(result.search_debug["query_locale"], "uk-UA")
        self.assertEqual(result.search_debug["engine_failure_count"], 2)
        self.assertGreater(float(result.search_debug["effective_search_confidence"]), 0.0)
        self.assertEqual(result.query_runs[0]["query_locale"], "uk-UA")
        self.assertEqual(result.query_runs[0]["region_bias"], "ua")

    def test_query_planner_marks_uah_finance_region_bias(self) -> None:
        query = "курс доллара в грн"
        classification = classify_query(query)
        plan = build_query_plan(
            query=query,
            classification=classification,
            decision=self._decision(),
            preferred_domains=["forex.ru", "bank.gov.ua", "minfin.com.ua"],
        )

        self.assertEqual(plan.debug.get("region_bias"), "ua")
        domain_queries = list(plan.query_roles.get("domain_constrained") or [])
        self.assertTrue(domain_queries)
        self.assertIn("site:bank.gov.ua", domain_queries[0])


if __name__ == "__main__":
    unittest.main()
