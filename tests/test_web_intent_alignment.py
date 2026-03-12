from __future__ import annotations

import unittest
from types import SimpleNamespace

from modules.internet.web.query_classifier import classify_query
from modules.internet.web.stage import _apply_intent_alignment, _resolve_intent_alignment
from modules.internet.web.web_models import SearchBudget, WebPolicyDecision, WebSearchMode


class WebIntentAlignmentTests(unittest.TestCase):
    def _decision(self, *, should_search: bool = True, score: float = 0.72) -> WebPolicyDecision:
        return WebPolicyDecision(
            mode=WebSearchMode.TARGETED_SEARCH,
            should_search=should_search,
            web_need_score=score,
            reason="test",
            budget=SearchBudget(max_queries=4, max_sources=4, max_pages=2, max_fetches=2),
        )

    def test_fx_rate_query_aligns_generic_chat_intent(self) -> None:
        classification = classify_query("курс доллара Киев")
        alignment = _resolve_intent_alignment(
            original_intent="chat",
            web_intent="fx_rate",
            classification=classification,
            decision=self._decision(),
        )

        self.assertTrue(alignment["applied"])
        self.assertEqual(alignment["corrected_intent"], "fx_rate")
        self.assertEqual(alignment["reason"], "web_external_factual_alignment")
        self.assertIn("category:finance", alignment["signals"])

    def test_apply_intent_alignment_updates_turn_summary_and_plan(self) -> None:
        ctx = SimpleNamespace(
            tags={"intent": "chat", "intent_conf": 0.41},
            meta={"turn_log_summaries": {"intent_summary": {"intent": "chat", "intent_confidence": 0.41}}},
            logs=[],
            plan="intent=chat; maintain conversational flow",
            state={},
        )
        classification = classify_query("курс доллара сегодня")
        alignment = _resolve_intent_alignment(
            original_intent="chat",
            web_intent="fx_rate",
            classification=classification,
            decision=self._decision(),
        )

        _apply_intent_alignment(ctx, alignment)

        self.assertEqual(ctx.tags["intent"], "fx_rate")
        self.assertEqual(ctx.tags["resolved_intent"], "fx_rate")
        self.assertEqual(ctx.meta["resolved_intent"], "fx_rate")
        self.assertEqual(ctx.plan, "intent=fx_rate; provide concise factual answer grounded in current evidence")
        intent_summary = ctx.meta["turn_log_summaries"]["intent_summary"]
        self.assertEqual(intent_summary["intent"], "fx_rate")
        self.assertEqual(intent_summary["original_intent"], "chat")
        self.assertEqual(intent_summary["corrected_intent"], "fx_rate")
        self.assertTrue(intent_summary["intent_alignment_applied"])

    def test_regular_chat_intent_is_not_forced_into_web_category(self) -> None:
        classification = classify_query("привет, как дела")
        alignment = _resolve_intent_alignment(
            original_intent="chat",
            web_intent="generic",
            classification=classification,
            decision=self._decision(should_search=False, score=0.08),
        )

        self.assertFalse(alignment["applied"])
        self.assertEqual(alignment["corrected_intent"], "chat")
        self.assertEqual(alignment["reason"], "web_intent_not_actionable")


if __name__ == "__main__":
    unittest.main()
