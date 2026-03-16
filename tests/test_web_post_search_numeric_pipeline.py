from __future__ import annotations

import unittest

from modules.internet.search import SearchResult
from modules.internet.web.query_classifier import classify_query
from modules.internet.web.result_processor import build_evidence_pack
from modules.internet.web.source_ranker import rank_sources
from modules.internet.web.stage import _assess_evidence_quality
from modules.internet.web.web_models import FreshnessAssessment, SearchBudget, WebPolicyDecision, WebSearchMode


class WebPostSearchNumericPipelineTests(unittest.TestCase):
    @staticmethod
    def _decision(mode: WebSearchMode = WebSearchMode.TARGETED_SEARCH) -> WebPolicyDecision:
        return WebPolicyDecision(
            mode=mode,
            should_search=True,
            web_need_score=0.9,
            reason="test",
            budget=SearchBudget(max_queries=5, max_sources=3, max_pages=3, max_fetches=3),
        )

    @staticmethod
    def _freshness(category: str) -> FreshnessAssessment:
        return FreshnessAssessment(
            temporal_risk=0.4,
            risk_level="medium",
            needs_refresh=True,
            stale_web_fact_detected=False,
            category=category,
        )

    def test_currency_pipeline_distinguishes_type_mismatch_from_true_conflict(self) -> None:
        query = "\u043d\u0430\u043b\u0438\u0447\u043d\u044b\u0439 \u043a\u0443\u0440\u0441 \u0434\u043e\u043b\u043b\u0430\u0440\u0430 \u0441\u0435\u0433\u043e\u0434\u043d\u044f"
        classification = classify_query(query)
        ranked = rank_sources(
            items=[
                SearchResult(
                    title="USD cash rate today",
                    snippet="Cash USD/UAH rate today at exchange offices.",
                    url="https://cash.example.com/usd-cash-rate-today",
                    source="cash.example.com",
                    score=0.67,
                ),
                SearchResult(
                    title="Official NBU USD/UAH rate today",
                    snippet="Official NBU exchange rate for USD/UAH.",
                    url="https://bank.example.com/nbu/usd-uah",
                    source="bank.example.com",
                    score=0.69,
                ),
                SearchResult(
                    title="USD/UAH history archive",
                    snippet="Historical archive by date.",
                    url="https://history.example.com/history/usd-uah",
                    source="history.example.com",
                    score=0.72,
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
                    "fetched_at": "2026-03-13T10:00:00+00:00",
                },
                "https://bank.example.com/nbu/usd-uah": {
                    "text": "Official NBU USD/UAH rate today: 39.80 UAH per USD.",
                    "fetched_at": "2026-03-13T10:00:00+00:00",
                },
                "https://history.example.com/history/usd-uah": {
                    "text": "Historical USD/UAH rate in 2024 was 37.50 UAH per USD.",
                    "fetched_at": "2026-03-13T10:00:00+00:00",
                },
            },
            decision=self._decision(),
            classification=classification,
            query_text=query,
        )
        quality = _assess_evidence_quality(
            ranked=ranked,
            evidence=evidence,
            freshness=self._freshness("finance"),
            classification=classification,
            source_audit=evidence.source_audit,
            query_text=query,
        )

        self.assertEqual([item.domain for item in evidence.items], ["cash.example.com"])
        self.assertFalse(evidence.conflicting_sources)
        self.assertEqual(quality["conflict_reason"], "rate_type_mismatch")
        self.assertEqual(quality["selected_result_factual_page_type"], "cash_rate")
        self.assertTrue(quality["factual_basis"]["accepted_numeric_candidates"])
        self.assertTrue(quality["factual_basis"]["conflict_summary"]["type_mismatches"])

    def test_historical_query_prefers_reference_page_over_analysis(self) -> None:
        query = "\u043d\u0430\u0437\u043e\u0432\u0438 \u0442\u043e\u0447\u043d\u0443\u044e \u0434\u0430\u0442\u0443 \u043e\u0441\u043d\u043e\u0432\u0430\u043d\u0438\u044f \u0435\u0432\u0440\u043e\u0441\u043e\u044e\u0437\u0430"
        classification = classify_query(query)
        ranked = rank_sources(
            items=[
                SearchResult(
                    title="European Union - Encyclopedia",
                    snippet="Historical timeline and treaty dates.",
                    url="https://www.britannica.com/topic/European-Union",
                    source="britannica.com",
                    score=0.55,
                ),
                SearchResult(
                    title="Analysis: what the EU means today",
                    snippet="Opinionated market analysis with overlapping words.",
                    url="https://example.com/news/eu-analysis",
                    source="example.com",
                    score=0.78,
                ),
            ],
            query_intent="news_release",
            query_category="external",
            query_text=query,
            limit_hint=2,
        )
        evidence = build_evidence_pack(
            ranked_results=ranked,
            fetched_pages={
                "https://www.britannica.com/topic/European-Union": {
                    "text": "The European Union traces its origins to the Maastricht Treaty signed in 1992.",
                    "fetched_at": "2026-03-13T10:00:00+00:00",
                },
                "https://example.com/news/eu-analysis": {
                    "text": "This analysis discusses Europe today and mentions many numbers like 300 and 120 but no exact founding date.",
                    "fetched_at": "2026-03-13T10:00:00+00:00",
                },
            },
            decision=self._decision(),
            classification=classification,
            query_text=query,
        )

        self.assertEqual([item.domain for item in evidence.items], ["britannica.com"])
        self.assertEqual(evidence.selection_summary["primary_selected_factual_page_type"], "generic_reference")
        self.assertTrue(evidence.selection_summary["factual_basis"]["accepted_numeric_candidates"])
        rejected = evidence.selection_summary["factual_basis"]["rejected_results"]
        self.assertTrue(any(row["domain"] == "example.com" for row in rejected))

    def test_weather_pipeline_keeps_temperature_and_rejects_noise(self) -> None:
        query = "\u043a\u0430\u043a\u0430\u044f \u0442\u0435\u043c\u043f\u0435\u0440\u0430\u0442\u0443\u0440\u0430 \u0432 \u041a\u0438\u0435\u0432\u0435 \u0441\u0435\u0433\u043e\u0434\u043d\u044f"
        classification = classify_query(query)
        ranked = rank_sources(
            items=[
                SearchResult(
                    title="Kyiv weather today",
                    snippet="Temperature today in Kyiv, forecast and feels like.",
                    url="https://weather.example.com/kyiv/today",
                    source="weather.example.com",
                    score=0.63,
                ),
                SearchResult(
                    title="Cold wave analysis in Kyiv",
                    snippet="Weather analysis article with many unrelated statistics.",
                    url="https://news.example.com/kyiv-cold-wave",
                    source="news.example.com",
                    score=0.82,
                ),
            ],
            query_intent="weather",
            query_category="weather",
            query_text=query,
            limit_hint=2,
        )
        evidence = build_evidence_pack(
            ranked_results=ranked,
            fetched_pages={
                "https://weather.example.com/kyiv/today": {
                    "text": "Kyiv weather today: temperature -2\u00b0C, feels like -5\u00b0C. Wind 6 m/s.",
                    "fetched_at": "2026-03-13T10:00:00+00:00",
                },
                "https://news.example.com/kyiv-cold-wave": {
                    "text": "This analysis says 120 schools were affected and 45 crews worked overnight.",
                    "fetched_at": "2026-03-13T10:00:00+00:00",
                },
            },
            decision=self._decision(),
            classification=classification,
            query_text=query,
        )

        self.assertEqual([item.domain for item in evidence.items], ["weather.example.com"])
        self.assertEqual(evidence.selection_summary["primary_selected_factual_page_type"], "overview_page")
        accepted = evidence.selection_summary["factual_basis"]["accepted_numeric_candidates"][0]["candidates"]
        self.assertTrue(any(float(row["value"]) < 0 for row in accepted))
        rejected = evidence.selection_summary["factual_basis"]["rejected_results"]
        self.assertTrue(any(row["domain"] == "news.example.com" for row in rejected))

    def test_price_query_prefers_direct_price_page(self) -> None:
        query = "\u0441\u043a\u043e\u043b\u044c\u043a\u043e \u0441\u0442\u043e\u0438\u0442 ps5 \u0441\u0435\u0433\u043e\u0434\u043d\u044f"
        classification = classify_query(query)
        ranked = rank_sources(
            items=[
                SearchResult(
                    title="PS5 price today",
                    snippet="Current PS5 price in UAH.",
                    url="https://store.example.com/ps5-price",
                    source="store.example.com",
                    score=0.62,
                ),
                SearchResult(
                    title="PS5 market analysis",
                    snippet="Analysis with overlapping keywords and many unrelated figures.",
                    url="https://blog.example.com/ps5-market-analysis",
                    source="blog.example.com",
                    score=0.80,
                ),
            ],
            query_intent="price",
            query_category="price",
            query_text=query,
            limit_hint=2,
        )
        evidence = build_evidence_pack(
            ranked_results=ranked,
            fetched_pages={
                "https://store.example.com/ps5-price": {
                    "text": "PS5 price today is 19999 UAH. Delivery tomorrow.",
                    "fetched_at": "2026-03-13T10:00:00+00:00",
                },
                "https://blog.example.com/ps5-market-analysis": {
                    "text": "The console market reached 300 million users in 2024 and 150 analysts discussed the trend.",
                    "fetched_at": "2026-03-13T10:00:00+00:00",
                },
            },
            decision=self._decision(),
            classification=classification,
            query_text=query,
        )

        self.assertEqual([item.domain for item in evidence.items], ["store.example.com"])
        self.assertIn(
            evidence.selection_summary["primary_selected_factual_page_type"],
            {"overview_page", "official_rate", "generic_reference"},
        )
        self.assertTrue(evidence.selection_summary["factual_basis"]["accepted_numeric_candidates"])
        self.assertTrue(any(row["domain"] == "blog.example.com" for row in evidence.selection_summary["factual_basis"]["rejected_results"]))


if __name__ == "__main__":
    unittest.main()
