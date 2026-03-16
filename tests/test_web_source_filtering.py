from __future__ import annotations

import unittest

from modules.internet.search import SearchResult
from modules.internet.web.query_classifier import classify_query
from modules.internet.web.result_processor import build_evidence_pack
from modules.internet.web.source_ranker import rank_sources
from modules.internet.web.stage import _assess_evidence_quality, _build_web_evidence_context
from modules.internet.web.web_models import FreshnessAssessment, SearchBudget, WebPolicyDecision, WebSearchMode


class WebSourceFilteringTests(unittest.TestCase):
    def _decision(self, *, mode: WebSearchMode = WebSearchMode.TARGETED_SEARCH) -> WebPolicyDecision:
        return WebPolicyDecision(
            mode=mode,
            should_search=True,
            web_need_score=0.82,
            reason="test",
            budget=SearchBudget(max_queries=4, max_sources=3, max_pages=3, max_fetches=3),
        )

    def _freshness(self) -> FreshnessAssessment:
        return FreshnessAssessment(
            temporal_risk=0.25,
            risk_level="medium",
            needs_refresh=True,
            stale_web_fact_detected=False,
            category="finance",
        )

    def test_finance_evidence_filters_support_and_video_sources(self) -> None:
        query = "курс доллара сегодня"
        classification = classify_query(query)
        ranked = rank_sources(
            items=[
                SearchResult(
                    title="Google Help: курс доллара сегодня",
                    snippet="Support article with matched words but not an exchange-rate source.",
                    url="https://support.google.com/websearch/answer/123",
                    source="support.google.com",
                    score=0.93,
                ),
                SearchResult(
                    title="USD/UAH today - official exchange rate",
                    snippet="Official rate for USD to UAH today from the National Bank.",
                    url="https://bank.gov.ua/markets/exchangerates?date=2026-03-12",
                    source="bank.gov.ua",
                    published_date="2026-03-12",
                    score=0.61,
                ),
                SearchResult(
                    title="USD/UAH rate today video",
                    snippet="Video discussion about the dollar exchange rate today.",
                    url="https://www.youtube.com/watch?v=abcd",
                    source="youtube.com",
                    score=0.88,
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
                "https://support.google.com/websearch/answer/123": {
                    "text": "How to change Google settings and interface language.",
                    "fetched_at": "2026-03-12T10:00:00+00:00",
                },
                "https://bank.gov.ua/markets/exchangerates?date=2026-03-12": {
                    "text": "Official USD/UAH exchange rate on 2026-03-12 was 41.25 UAH per USD.",
                    "fetched_at": "2026-03-12T10:00:00+00:00",
                },
                "https://www.youtube.com/watch?v=abcd": {
                    "text": "",
                    "fetched_at": "2026-03-12T10:00:00+00:00",
                },
            },
            decision=self._decision(),
            classification=classification,
            query_text=query,
        )

        selected_domains = [item.domain for item in evidence.items]
        audit_by_domain = {str(row.get("domain") or ""): row for row in evidence.source_audit}

        self.assertIn("bank.gov.ua", selected_domains)
        self.assertNotIn("support.google.com", selected_domains)
        self.assertNotIn("youtube.com", selected_domains)
        self.assertEqual(audit_by_domain["support.google.com"]["filtered_out_reason"], "unsupported_source_type")
        self.assertEqual(audit_by_domain["youtube.com"]["filtered_out_reason"], "unsupported_source_type")

    def test_historical_query_filters_random_lexical_page(self) -> None:
        query = "назови точную дату основания евросоюза"
        classification = classify_query(query)
        ranked = rank_sources(
            items=[
                SearchResult(
                    title="European Union - history and timeline",
                    snippet="Historical timeline with founding dates and treaty milestones.",
                    url="https://www.britannica.com/topic/European-Union",
                    source="britannica.com",
                    score=0.54,
                ),
                SearchResult(
                    title="Random article mentioning exact date and euro",
                    snippet="A generic page with overlapping words but no real historical reference.",
                    url="https://example.com/blog/euro-date-random",
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
                    "fetched_at": "2026-03-12T10:00:00+00:00",
                },
                "https://example.com/blog/euro-date-random": {
                    "text": "This blog post talks loosely about Europe, money and random dates.",
                    "fetched_at": "2026-03-12T10:00:00+00:00",
                },
            },
            decision=self._decision(),
            classification=classification,
            query_text=query,
        )

        selected_domains = [item.domain for item in evidence.items]
        audit_by_domain = {str(row.get("domain") or ""): row for row in evidence.source_audit}

        self.assertIn("britannica.com", selected_domains)
        self.assertIn(
            audit_by_domain["example.com"]["filtered_out_reason"],
            {"topical_mismatch", "missing_relevant_numeric"},
        )

    def test_weak_strict_evidence_marks_cautious_synthesis(self) -> None:
        query = "курс доллара сегодня"
        classification = classify_query(query)
        ranked = rank_sources(
            items=[
                SearchResult(
                    title="Google Help: курс доллара сегодня",
                    snippet="Support article with matched words but not an exchange-rate source.",
                    url="https://support.google.com/websearch/answer/123",
                    source="support.google.com",
                    score=0.93,
                ),
                SearchResult(
                    title="USD/UAH rate today video",
                    snippet="Video discussion about the dollar exchange rate today.",
                    url="https://www.youtube.com/watch?v=abcd",
                    source="youtube.com",
                    score=0.88,
                ),
            ],
            query_intent="fx_rate",
            query_category="finance",
            query_text=query,
            limit_hint=2,
        )
        evidence = build_evidence_pack(
            ranked_results=ranked,
            fetched_pages={},
            decision=self._decision(mode=WebSearchMode.VERIFY_ONLY),
            classification=classification,
            query_text=query,
        )
        quality = _assess_evidence_quality(
            ranked=ranked,
            evidence=evidence,
            freshness=self._freshness(),
            classification=classification,
            source_audit=evidence.source_audit,
        )
        context = _build_web_evidence_context(
            evidence=evidence,
            prompt_items=[],
            mode=WebSearchMode.VERIFY_ONLY.value,
            query=query,
            source_audit=evidence.source_audit,
            quality=quality,
        )

        self.assertEqual(len(evidence.items), 0)
        self.assertTrue(quality["cautious_synthesis"])
        self.assertGreaterEqual(int(quality["topical_filtered_sources"]), 1)
        self.assertTrue(context["cautious_synthesis"])
        self.assertIn("synthesis_guidance: cautious", context["prompt_block"])


if __name__ == "__main__":
    unittest.main()
