from __future__ import annotations

import unittest

from core.response_pipeline import _apply_web_factual_caution_guard
from modules.internet.search import SearchResult
from modules.internet.web.query_classifier import classify_query
from modules.internet.web.query_planner import build_query_plan
from modules.internet.web.query_text import analyze_search_text
from modules.internet.web.result_processor import build_evidence_pack
from modules.internet.web.source_ranker import rank_sources
from modules.internet.web.stage import _assess_evidence_quality
from modules.internet.web.web_models import FreshnessAssessment, SearchBudget, WebPolicyDecision, WebSearchMode


class WebFactualNumericSafetyTests(unittest.TestCase):
    @staticmethod
    def _decision(mode: WebSearchMode = WebSearchMode.TARGETED_SEARCH) -> WebPolicyDecision:
        return WebPolicyDecision(
            mode=mode,
            should_search=True,
            web_need_score=0.92,
            reason="test",
            budget=SearchBudget(max_queries=6, max_sources=4, max_pages=4, max_fetches=4),
        )

    @staticmethod
    def _freshness(category: str = "finance") -> FreshnessAssessment:
        return FreshnessAssessment(
            temporal_risk=0.45,
            risk_level="medium",
            needs_refresh=True,
            stale_web_fact_detected=False,
            category=category,
        )

    def test_finance_query_core_removes_commands_and_wrapper_tail(self) -> None:
        query = "/mode_lock on найди точный курс доллара и евро в грн на сегодня. желательно до копеек"
        debug = analyze_search_text(query)
        classification = classify_query(query)
        plan = build_query_plan(
            query=query,
            classification=classification,
            decision=self._decision(),
        )

        self.assertEqual(debug["original_query"], query)
        self.assertNotIn("/mode_lock", str(debug["normalized_query"]))
        self.assertNotIn("желательно", str(debug["normalized_query"]).lower())
        self.assertIn("/mode_lock on", list(debug.get("removed_wrapper_fragments") or []))
        self.assertTrue(any("USD UAH" in row for row in plan.query_roles["validation"]))
        self.assertTrue(any("EUR UAH" in row for row in plan.query_roles["validation"]))
        self.assertFalse(any("official source" in row.lower() for row in plan.all_queries()))

    def test_unrelated_numbers_do_not_become_finance_evidence(self) -> None:
        query = "курс доллара сегодня"
        classification = classify_query(query)
        ranked = rank_sources(
            items=[
                SearchResult(
                    title="Official USD/UAH exchange rate",
                    snippet="Official rate for USD to UAH from the National Bank.",
                    url="https://bank.gov.ua/markets/exchangerates?date=2026-03-12",
                    source="bank.gov.ua",
                    published_date="2026-03-12",
                    score=0.62,
                ),
                SearchResult(
                    title="Forum thread about dollar",
                    snippet="Matched words but mostly random counters and thread metadata.",
                    url="https://example.com/thread/2040",
                    source="example.com",
                    score=0.90,
                ),
            ],
            query_intent="fx_rate",
            query_category="finance",
            query_text=query,
            limit_hint=4,
        )
        evidence = build_evidence_pack(
            ranked_results=ranked,
            fetched_pages={
                "https://bank.gov.ua/markets/exchangerates?date=2026-03-12": {
                    "text": "Official USD/UAH exchange rate on 2026-03-12 was 41.25 UAH per USD.",
                    "fetched_at": "2026-03-12T10:00:00+00:00",
                },
                "https://example.com/thread/2040": {
                    "text": "Thread id 2040. Views 12345. Comments 88. Support code 991.",
                    "fetched_at": "2026-03-12T10:00:00+00:00",
                },
            },
            decision=self._decision(),
            classification=classification,
            query_text=query,
        )

        audit_by_domain = {str(row.get("domain") or ""): row for row in evidence.source_audit}

        self.assertEqual([item.domain for item in evidence.items], ["bank.gov.ua"])
        self.assertIn(
            audit_by_domain["example.com"]["filtered_out_reason"],
            {"missing_relevant_numeric", "unsupported_source_type", "topical_mismatch"},
        )
        self.assertEqual(len(list(audit_by_domain["example.com"].get("numeric_candidates_selected") or [])), 0)
        self.assertGreaterEqual(len(list(audit_by_domain["example.com"].get("numeric_candidates_rejected") or [])), 1)
        self.assertFalse(evidence.conflicting_sources)

    def test_high_finance_numeric_conflict_forces_cautious_quality(self) -> None:
        query = "курс доллара в грн сегодня"
        classification = classify_query(query)
        ranked = rank_sources(
            items=[
                SearchResult(
                    title="USD/UAH today - Minfin",
                    snippet="Market USD/UAH rate today.",
                    url="https://minfin.com.ua/currency/usd/",
                    source="minfin.com.ua",
                    score=0.66,
                ),
                SearchResult(
                    title="USD/UAH today - Finance.ua",
                    snippet="Current USD/UAH exchange rate today.",
                    url="https://finance.ua/ru/currency/usd",
                    source="finance.ua",
                    score=0.64,
                ),
            ],
            query_intent="fx_rate",
            query_category="finance",
            query_text=query,
            limit_hint=4,
        )
        evidence = build_evidence_pack(
            ranked_results=ranked,
            fetched_pages={
                "https://minfin.com.ua/currency/usd/": {
                    "text": "USD/UAH exchange rate today: 41.25 UAH per USD.",
                    "fetched_at": "2026-03-12T10:00:00+00:00",
                },
                "https://finance.ua/ru/currency/usd": {
                    "text": "USD/UAH exchange rate today: 58.56 UAH per USD.",
                    "fetched_at": "2026-03-12T10:01:00+00:00",
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

        self.assertTrue(evidence.conflicting_sources)
        self.assertTrue(any("numeric_conflict" in note for note in evidence.conflict_notes))
        self.assertGreaterEqual(float(quality["conflict_severity"]), 0.55)
        self.assertTrue(bool(quality["cautious_synthesis"]))
        self.assertLess(float(quality["final_factual_confidence"]), 0.64)

    def test_postprocess_replaces_overconfident_numeric_claim_on_weak_web_evidence(self) -> None:
        fixed, changed = _apply_web_factual_caution_guard(
            "Точный курс доллара сегодня 58.56 грн.",
            meta={
                "web_used": True,
                "web_primary_category": "finance",
                "web_evidence_quality": {
                    "cautious_synthesis": True,
                    "final_factual_confidence": 0.41,
                    "conflict_severity": 0.92,
                    "numeric_profile": "fx_rate",
                },
            },
            web_evidence_context={
                "cautious_synthesis": True,
                "numeric_profile": "fx_rate",
                "conflict_notes": ["numeric_conflict_high:USD/UAH:minfin.com.ua=41.250 vs finance.ua=58.560"],
                "compact_citations": ["minfin.com.ua (2026-03-12)"],
                "final_factual_confidence": 0.41,
                "conflict_severity": 0.92,
            },
            web_intent="fx_rate",
        )

        self.assertTrue(changed)
        self.assertIn("Не удалось надежно подтвердить точное значение", fixed)
        self.assertIn("minfin.com.ua", fixed)


if __name__ == "__main__":
    unittest.main()
