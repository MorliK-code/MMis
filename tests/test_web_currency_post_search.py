from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from modules.internet.search import SearchClient, SearchResult
from modules.internet.web.query_classifier import classify_query
from modules.internet.web.result_processor import build_evidence_pack
from modules.internet.web.search_manager import SearchManager
from modules.internet.web.source_ranker import rank_sources
from modules.internet.web.stage import _assess_evidence_quality
from modules.internet.web.web_models import FreshnessAssessment, QueryClassification, SearchBudget, WebPolicyDecision, WebQueryPlan, WebSearchMode


class _DummyPage:
    def __init__(self, url: str):
        self.final_url = url
        self.title = "stub"
        self.text = "stub text"
        self.status_code = 200
        self.metadata = {"clean_method": "stub", "removed_blocks": 0, "raw_len": 10, "clean_len": 9}


class _DummyScraper:
    def scrape(self, url: str):  # noqa: ANN001
        return _DummyPage(url)


class _StubSearchClient:
    def search(self, query: str, **kwargs):  # noqa: ANN001
        _ = kwargs
        return [
            SearchResult(
                title="USD/UAH result",
                snippet="USD/UAH current exchange rate",
                url="https://minfin.com.ua/currency/usd/",
                source="minfin.com.ua",
                raw={
                    "provider": "searxng",
                    "reported_result_count": 0,
                    "raw_result_count": 3,
                    "effective_success_reason": "results_present_number_of_results_zero",
                },
            )
        ]


class WebCurrencyPostSearchTests(unittest.TestCase):
    @staticmethod
    def _decision(mode: WebSearchMode = WebSearchMode.TARGETED_SEARCH) -> WebPolicyDecision:
        return WebPolicyDecision(
            mode=mode,
            should_search=True,
            web_need_score=0.91,
            reason="test",
            budget=SearchBudget(max_queries=5, max_sources=3, max_pages=3, max_fetches=3),
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

    def test_searx_results_are_used_even_when_number_of_results_is_zero(self) -> None:
        payload = json.dumps(
            {
                "number_of_results": 0,
                "results": [
                    {
                        "url": "https://minfin.com.ua/currency/usd/",
                        "title": "USD cash rate today",
                        "content": "Cash USD/UAH rate today.",
                    }
                ],
            }
        )
        client = SearchClient(
            endpoint="https://searx.example/search?format=json",
            strict_endpoint=True,
            use_disk_cache=False,
        )
        with patch("modules.internet.search._http_get", return_value=payload):
            rows = client._search_endpoint("usd uah")

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].source, "minfin.com.ua")
        self.assertEqual(rows[0].raw.get("reported_result_count"), 0)
        self.assertEqual(rows[0].raw.get("raw_result_count"), 1)
        self.assertEqual(rows[0].raw.get("effective_success_reason"), "results_present_number_of_results_zero")

    def test_search_manager_keeps_success_when_rows_exist_but_reported_count_is_zero(self) -> None:
        manager = SearchManager(
            search_client=_StubSearchClient(),
            scraper=_DummyScraper(),
            cooldown_seconds=0,
        )
        classification = QueryClassification(
            query_type="external_factual",
            primary_category="finance",
            is_temporal=True,
            is_local_project_question=False,
            is_external_fact_question=True,
            is_ambiguous=False,
            is_correction_challenge=False,
            requires_freshness=True,
            stakes_level="high",
            expected_search_need="high",
            explicit_search_intent=True,
        )
        result = manager.execute(
            plan=WebQueryPlan(mode=WebSearchMode.TARGETED_SEARCH, scout_queries=["usd uah exchange rate"]),
            decision=self._decision(),
            classification=classification,
            freshness=self._freshness(),
            preferred_domains=[],
            geo_hint="",
        )

        self.assertEqual(len(result.results), 1)
        self.assertEqual(result.search_debug.get("reported_result_count"), 0)
        self.assertEqual(result.search_debug.get("raw_result_count"), 3)
        self.assertTrue(result.search_debug.get("effective_success"))
        self.assertEqual(result.search_debug.get("effective_success_reason"), "results_present_number_of_results_zero")
        self.assertEqual(result.query_runs[0].get("reported_result_count"), 0)
        self.assertEqual(result.query_runs[0].get("raw_result_count"), 3)
        self.assertTrue(result.query_runs[0].get("effective_success"))

    def test_explicit_cash_query_selects_cash_rate_page(self) -> None:
        query = "\u043d\u0430\u043b\u0438\u0447\u043d\u044b\u0439 \u043a\u0443\u0440\u0441 \u0434\u043e\u043b\u043b\u0430\u0440\u0430 \u0441\u0435\u0433\u043e\u0434\u043d\u044f"
        classification = classify_query(query)
        ranked = rank_sources(
            items=[
                SearchResult(
                    title="USD cash rate today",
                    snippet="Cash USD/UAH rate today.",
                    url="https://cash.example.com/usd-cash-rate-today",
                    source="cash.example.com",
                    score=0.64,
                ),
                SearchResult(
                    title="Official NBU USD/UAH rate today",
                    snippet="Official NBU exchange rate for USD/UAH.",
                    url="https://bank.example.com/nbu/usd-uah",
                    source="bank.example.com",
                    score=0.66,
                ),
            ],
            query_intent="fx_rate",
            query_category="finance",
            query_text=query,
            limit_hint=3,
        )
        evidence = build_evidence_pack(
            ranked_results=ranked,
            fetched_pages={
                "https://cash.example.com/usd-cash-rate-today": {
                    "text": "Cash USD/UAH rate today: 41.25 UAH per USD.",
                    "fetched_at": "2026-03-12T10:00:00+00:00",
                },
                "https://bank.example.com/nbu/usd-uah": {
                    "text": "Official NBU USD/UAH rate today: 39.80 UAH per USD.",
                    "fetched_at": "2026-03-12T10:00:00+00:00",
                },
            },
            decision=self._decision(),
            classification=classification,
            query_text=query,
        )

        audit_by_domain = {str(row.get("domain") or ""): row for row in evidence.source_audit}

        self.assertEqual([item.domain for item in evidence.items], ["cash.example.com"])
        self.assertEqual(evidence.selection_summary.get("currency_selected_rate_type"), "cash_rate")
        self.assertEqual(audit_by_domain["bank.example.com"].get("currency_page_type"), "nbu_rate")
        self.assertIn(
            audit_by_domain["bank.example.com"]["filtered_out_reason"],
            {"rate_type_mismatch", "missing_relevant_numeric"},
        )
        self.assertFalse(evidence.conflicting_sources)

    def test_current_rate_query_filters_historical_page(self) -> None:
        query = "\u043a\u0443\u0440\u0441 \u0434\u043e\u043b\u043b\u0430\u0440\u0430 \u0441\u0435\u0433\u043e\u0434\u043d\u044f"
        classification = classify_query(query)
        ranked = rank_sources(
            items=[
                SearchResult(
                    title="USD/UAH today overview",
                    snippet="Current USD/UAH exchange rate today.",
                    url="https://rates.example.com/currency/usd-uah",
                    source="rates.example.com",
                    score=0.68,
                ),
                SearchResult(
                    title="USD/UAH historical rates archive",
                    snippet="Historical USD/UAH rates by date.",
                    url="https://history.example.com/history/usd-uah-2024",
                    source="history.example.com",
                    score=0.70,
                ),
            ],
            query_intent="fx_rate",
            query_category="finance",
            query_text=query,
            limit_hint=3,
        )
        evidence = build_evidence_pack(
            ranked_results=ranked,
            fetched_pages={
                "https://rates.example.com/currency/usd-uah": {
                    "text": "USD/UAH exchange rate today: 41.20 UAH per USD.",
                    "fetched_at": "2026-03-12T10:00:00+00:00",
                },
                "https://history.example.com/history/usd-uah-2024": {
                    "text": "Historical USD/UAH rate in 2024 was 37.50 UAH per USD.",
                    "fetched_at": "2026-03-12T10:00:00+00:00",
                },
            },
            decision=self._decision(),
            classification=classification,
            query_text=query,
        )

        audit_by_domain = {str(row.get("domain") or ""): row for row in evidence.source_audit}

        self.assertEqual([item.domain for item in evidence.items], ["rates.example.com"])
        self.assertEqual(audit_by_domain["history.example.com"]["filtered_out_reason"], "historical_rate_mismatch")

    def test_same_rate_type_conflict_remains_true_numeric_conflict(self) -> None:
        query = "\u043d\u0430\u043b\u0438\u0447\u043d\u044b\u0439 \u043a\u0443\u0440\u0441 \u0434\u043e\u043b\u043b\u0430\u0440\u0430 \u0441\u0435\u0433\u043e\u0434\u043d\u044f"
        classification = classify_query(query)
        ranked = rank_sources(
            items=[
                SearchResult(
                    title="USD cash rate today",
                    snippet="Cash USD/UAH rate today.",
                    url="https://cash-a.example.com/usd-cash-rate-today",
                    source="cash-a.example.com",
                    score=0.65,
                ),
                SearchResult(
                    title="USD cash rate today - second source",
                    snippet="Cash USD/UAH rate today from another source.",
                    url="https://cash-b.example.com/usd-cash-rate-today",
                    source="cash-b.example.com",
                    score=0.63,
                ),
            ],
            query_intent="fx_rate",
            query_category="finance",
            query_text=query,
            limit_hint=3,
        )
        evidence = build_evidence_pack(
            ranked_results=ranked,
            fetched_pages={
                "https://cash-a.example.com/usd-cash-rate-today": {
                    "text": "Cash USD/UAH rate today: 41.25 UAH per USD.",
                    "fetched_at": "2026-03-12T10:00:00+00:00",
                },
                "https://cash-b.example.com/usd-cash-rate-today": {
                    "text": "Cash USD/UAH rate today: 58.56 UAH per USD.",
                    "fetched_at": "2026-03-12T10:00:00+00:00",
                },
            },
            decision=self._decision(),
            classification=classification,
            query_text=query,
        )
        quality = _assess_evidence_quality(
            ranked=ranked,
            evidence=evidence,
            freshness=self._freshness(),
            classification=classification,
            source_audit=evidence.source_audit,
            query_text=query,
        )

        self.assertEqual(evidence.selection_summary.get("currency_selected_rate_type"), "cash_rate")
        self.assertTrue(evidence.conflicting_sources)
        self.assertTrue(any(str(note).startswith("numeric_conflict_") for note in evidence.conflict_notes))
        self.assertEqual(quality["conflict_reason"], "true_numeric_conflict")


if __name__ == "__main__":
    unittest.main()
