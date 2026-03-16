from __future__ import annotations

import unittest

from modules.internet.web.query_classifier import classify_query
from modules.internet.web.web_models import ConfidenceAssessment, FreshnessAssessment, WebSearchMode
from modules.internet.web.web_policy import WebPolicyEngine


class WebPolicyChallengeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = WebPolicyEngine()

    @staticmethod
    def _confidence(score: float = 0.9) -> ConfidenceAssessment:
        return ConfidenceAssessment(score=score, level="high", reasons=[], breakdown={})

    @staticmethod
    def _freshness(*, temporal_risk: float = 0.2, needs_refresh: bool = False, category: str = "external") -> FreshnessAssessment:
        return FreshnessAssessment(
            temporal_risk=temporal_risk,
            risk_level="medium" if temporal_risk >= 0.4 else "low",
            needs_refresh=needs_refresh,
            stale_web_fact_detected=False,
            category=category,
            reasons=[],
            breakdown={},
        )

    def test_short_correction_for_fx_rate_is_routed_to_recheck(self) -> None:
        classification = classify_query(
            "курс доллара Киев неправильно",
            original_text="неправильно",
        )

        decision = self.engine.decide(
            query="курс доллара Киев неправильно",
            classification=classification,
            confidence=self._confidence(0.95),
            freshness=self._freshness(temporal_risk=0.7, needs_refresh=True, category="finance"),
            web_mode="auto",
            internet_enabled=True,
            user_override=None,
            policy_context={},
        )

        self.assertTrue(classification.is_correction_challenge)
        self.assertEqual(classification.primary_category, "finance")
        self.assertEqual(classification.expected_search_need, "TARGETED_SEARCH")
        self.assertIn(decision.mode, {WebSearchMode.TARGETED_SEARCH, WebSearchMode.DEEP_SEARCH})
        self.assertTrue(decision.should_search)
        self.assertEqual(decision.reason, "factual_challenge_recheck")

    def test_exact_date_challenge_no_longer_stays_in_no_search(self) -> None:
        classification = classify_query(
            "\u043d\u0430\u0437\u043e\u0432\u0438 \u0442\u043e\u0447\u043d\u0443\u044e \u0434\u0430\u0442\u0443",
            original_text="\u043d\u0430\u0437\u043e\u0432\u0438 \u0442\u043e\u0447\u043d\u0443\u044e \u0434\u0430\u0442\u0443",
        )

        decision = self.engine.decide(
            query="\u043d\u0430\u0437\u043e\u0432\u0438 \u0442\u043e\u0447\u043d\u0443\u044e \u0434\u0430\u0442\u0443",
            classification=classification,
            confidence=self._confidence(0.88),
            freshness=self._freshness(temporal_risk=0.35, needs_refresh=False, category="external"),
            web_mode="auto",
            internet_enabled=True,
            user_override=None,
            policy_context={},
        )

        self.assertTrue(classification.is_correction_challenge)
        self.assertTrue(classification.is_external_fact_question)
        self.assertIn(classification.expected_search_need, {"VERIFY_ONLY", "TARGETED_SEARCH"})
        self.assertNotEqual(decision.mode, WebSearchMode.NO_SEARCH)
        self.assertTrue(decision.should_search)

    def test_you_are_wrong_weather_challenge_avoids_no_search(self) -> None:
        classification = classify_query(
            "weather kyiv you're wrong",
            original_text="you're wrong",
        )

        decision = self.engine.decide(
            query="weather kyiv you're wrong",
            classification=classification,
            confidence=self._confidence(0.91),
            freshness=self._freshness(temporal_risk=0.8, needs_refresh=True, category="weather"),
            web_mode="auto",
            internet_enabled=True,
            user_override=None,
            policy_context={},
        )

        self.assertTrue(classification.is_correction_challenge)
        self.assertEqual(classification.primary_category, "weather")
        self.assertIn(decision.mode, {WebSearchMode.TARGETED_SEARCH, WebSearchMode.DEEP_SEARCH})


if __name__ == "__main__":
    unittest.main()
