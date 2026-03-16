from __future__ import annotations

import unittest

from modules.internet.web.query_classifier import classify_query
from modules.internet.web.stage import _compat_query_intent, _resolve_intent_alignment
from modules.internet.web.web_models import ConfidenceAssessment, FreshnessAssessment, SearchBudget, WebPolicyDecision, WebSearchMode
from modules.internet.web.web_policy import WebPolicyEngine
from modules.nlu.intent_resolver import IntentResolver
from modules.nlu.types import Segment


class WebFxRoutingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = WebPolicyEngine()
        self.intent_resolver = IntentResolver()

    @staticmethod
    def _decision(mode: WebSearchMode = WebSearchMode.TARGETED_SEARCH) -> WebPolicyDecision:
        return WebPolicyDecision(
            mode=mode,
            should_search=mode != WebSearchMode.NO_SEARCH,
            web_need_score=0.72,
            reason="test",
            budget=SearchBudget(max_queries=3, max_sources=4, max_pages=2, max_fetches=2),
        )

    @staticmethod
    def _confidence() -> ConfidenceAssessment:
        return ConfidenceAssessment(score=0.88, level="high", reasons=[], breakdown={})

    @staticmethod
    def _freshness() -> FreshnessAssessment:
        return FreshnessAssessment(
            temporal_risk=0.65,
            risk_level="medium",
            needs_refresh=True,
            stale_web_fact_detected=False,
            category="finance",
            reasons=[],
            breakdown={},
        )

    def test_colloquial_dollar_today_is_classified_as_finance(self) -> None:
        text = "\u0447\u0442\u043e \u043f\u043e \u0434\u043e\u043b\u043b\u0430\u0440\u0443 \u0441\u0435\u0433\u043e\u0434\u043d\u044f?"
        classification = classify_query(text)

        self.assertEqual(classification.primary_category, "finance")
        self.assertEqual(classification.query_type, "external_factual")
        self.assertTrue(classification.requires_freshness)
        self.assertTrue(classification.explicit_search_intent)
        self.assertIn("colloquial_fx", classification.category_hits)
        self.assertEqual(_compat_query_intent(text, classification=classification), "fx_rate")

    def test_short_currency_pair_query_forces_web_policy_out_of_chat(self) -> None:
        text = "\u0434\u043e\u043b\u043b\u0430\u0440 \u0432 \u0433\u0440\u043d"
        classification = classify_query(text)
        decision = self.engine.decide(
            query=text,
            classification=classification,
            confidence=self._confidence(),
            freshness=self._freshness(),
            web_mode="auto",
            internet_enabled=True,
            user_override=None,
            policy_context={},
        )

        self.assertEqual(classification.primary_category, "finance")
        self.assertTrue(classification.is_external_fact_question)
        self.assertTrue(decision.should_search)
        self.assertNotEqual(decision.mode, WebSearchMode.NO_SEARCH)
        self.assertIn(decision.mode, {WebSearchMode.VERIFY_ONLY, WebSearchMode.TARGETED_SEARCH, WebSearchMode.DEEP_SEARCH})

    def test_currency_queries_align_chat_to_fx_rate(self) -> None:
        text = "\u0447\u0442\u043e \u043f\u043e \u0434\u043e\u043b\u043b\u0430\u0440\u0443 \u0441\u0435\u0433\u043e\u0434\u043d\u044f?"
        classification = classify_query(text)
        alignment = _resolve_intent_alignment(
            original_intent="chat",
            web_intent=_compat_query_intent(text, classification=classification),
            classification=classification,
            decision=self._decision(),
        )

        self.assertTrue(alignment["applied"])
        self.assertEqual(alignment["corrected_intent"], "fx_rate")
        self.assertIn("category:finance", alignment["signals"])

    def test_intent_resolver_promotes_currency_phrases_to_finance_query(self) -> None:
        segments = [
            Segment(
                text="\u0434\u043e\u043b\u043b\u0430\u0440 \u0432 \u0433\u0440\u043d",
                kind="request",
                confidence=0.95,
            )
        ]
        intents, concepts = self.intent_resolver.resolve(segments)

        self.assertTrue(any(item.name == "finance_query" for item in intents))
        self.assertTrue(any(item.name == "currency" for item in concepts))


if __name__ == "__main__":
    unittest.main()
