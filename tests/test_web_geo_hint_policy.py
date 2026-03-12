from __future__ import annotations

import unittest

from modules.internet.web.query_classifier import classify_query
from modules.internet.web.query_planner import build_query_plan
from modules.internet.web.stage import _resolve_geo_hint
from modules.internet.web.web_models import SearchBudget, WebPolicyDecision, WebSearchMode


class WebGeoHintPolicyTests(unittest.TestCase):
    def _decision(self) -> WebPolicyDecision:
        return WebPolicyDecision(
            mode=WebSearchMode.TARGETED_SEARCH,
            should_search=True,
            web_need_score=0.8,
            reason="test",
            budget=SearchBudget(
                max_queries=6,
                max_sources=6,
                max_pages=3,
                max_fetches=3,
            ),
        )

    def test_general_fx_query_does_not_apply_stale_profile_geo(self) -> None:
        classification = classify_query("курс доллара")
        decision = _resolve_geo_hint(
            original_query="курс доллара",
            effective_query="курс доллара",
            classification=classification,
            state={
                "profile_summary": {"user": {"country": "Finland"}},
                "history": [{"role": "user", "text": "какая погода в Финляндии"}],
                "location": "Finland",
            },
            memory_context={},
            retrieved_memories=[],
        )

        self.assertEqual(decision["value"], "")
        self.assertEqual(decision["candidate"], "Finland")
        self.assertEqual(decision["reason"], "implicit_geo_disabled_for_finance")
        self.assertFalse(decision["applied"])

    def test_fx_query_with_explicit_location_keeps_query_geo(self) -> None:
        classification = classify_query("курс доллара в Киеве")
        decision = _resolve_geo_hint(
            original_query="курс доллара в Киеве",
            effective_query="курс доллара в Киеве",
            classification=classification,
            state={
                "profile_summary": {"user": {"country": "Finland"}},
                "location": "Finland",
            },
            memory_context={},
            retrieved_memories=[],
        )

        self.assertEqual(decision["value"], "Kyiv")
        self.assertEqual(decision["reason"], "explicit_query_location")
        self.assertTrue(decision["applied"])

    def test_finance_planner_does_not_append_geo_without_local_scope(self) -> None:
        query = "курс доллара сегодня"
        classification = classify_query(query)
        plan = build_query_plan(
            query=query,
            classification=classification,
            decision=self._decision(),
            geo_hint="Finland",
        )

        self.assertEqual(plan.query_roles["primary"], ["курс доллара сегодня"])
        self.assertFalse(any("finland" in q.lower() for q in plan.all_queries()))


if __name__ == "__main__":
    unittest.main()
