from __future__ import annotations

import unittest

from core.response_pipeline import (
    _apply_web_factual_caution_guard,
    _isolate_factual_prompt_context,
    _resolve_web_factual_response_mode,
)


class ResponsePipelineFactualIsolationTests(unittest.TestCase):
    def test_resolves_price_and_historical_factual_modes(self) -> None:
        self.assertEqual(
            _resolve_web_factual_response_mode(
                web_category="price",
                web_evidence_context={"numeric_profile": "price"},
                quality={},
            ),
            "price",
        )
        self.assertEqual(
            _resolve_web_factual_response_mode(
                web_category="external",
                web_evidence_context={"numeric_profile": "historical"},
                quality={},
            ),
            "historical_factual",
        )

    def test_fx_rate_context_isolation_drops_weather_blocks_and_tool_state(self) -> None:
        memory_context = {
            "blocks": {
                "session_summary": "Погода на неделю: от +2 до +15 градусов, дождь и ветер.",
                "working_memory": "Пользователь спрашивает курс доллара в грн на сегодня.",
                "retrieved_semantic": "USD/UAH currency rate and exchange context.",
                "active_tool_state": "weather_tool_result: tomorrow +2 to +15 degrees",
            },
            "selected": [
                {"text": "Weather tomorrow: +2 to +15 degrees.", "source": "message", "topic": "weather"},
                {"text": "USD/UAH exchange rate today is around 41.25.", "source": "web", "topic": "web:finance"},
            ],
        }

        filtered_context, filtered_memories, debug = _isolate_factual_prompt_context(
            memory_context=memory_context,
            retrieved_memories=list(memory_context["selected"]),
            web_intent="fx_rate",
        )

        blocks = filtered_context["blocks"]
        self.assertIn("working_memory", blocks)
        self.assertIn("retrieved_semantic", blocks)
        self.assertNotIn("session_summary", blocks)
        self.assertNotIn("active_tool_state", blocks)
        self.assertEqual(len(filtered_memories), 1)
        self.assertIn("usd/uah", str(filtered_memories[0]["text"]).lower())
        self.assertEqual(debug["target_category"], "fx_rate")
        self.assertTrue(any(row["block"] == "session_summary" for row in debug["dropped_memory_blocks"]))
        self.assertTrue(any(row["block"] == "active_tool_state" for row in debug["dropped_memory_blocks"]))
        self.assertTrue(any("cross_category" in row["reason"] for row in debug["dropped_retrieved_memories"]))

    def test_price_context_isolation_drops_weather_noise(self) -> None:
        memory_context = {
            "blocks": {
                "session_summary": "Weather tomorrow: +2 to +15 degrees with rain.",
                "working_memory": "User is asking how much the laptop costs today.",
                "retrieved_semantic": "Product price today for the laptop model is around 39999 UAH.",
                "active_tool_state": "weather_tool_result: +2 to +15 degrees",
            },
            "selected": [
                {"text": "Weather tomorrow: +2 to +15 degrees.", "source": "message", "topic": "weather"},
                {"text": "Current product price is 39999 UAH.", "source": "web", "topic": "web:price"},
            ],
        }

        filtered_context, filtered_memories, debug = _isolate_factual_prompt_context(
            memory_context=memory_context,
            retrieved_memories=list(memory_context["selected"]),
            web_intent="price",
        )

        blocks = filtered_context["blocks"]
        self.assertIn("working_memory", blocks)
        self.assertIn("retrieved_semantic", blocks)
        self.assertNotIn("session_summary", blocks)
        self.assertNotIn("active_tool_state", blocks)
        self.assertEqual(len(filtered_memories), 1)
        self.assertIn("39999", str(filtered_memories[0]["text"]))
        self.assertEqual(debug["target_category"], "price")

    def test_non_factual_intent_keeps_context_unchanged(self) -> None:
        memory_context = {
            "blocks": {
                "session_summary": "Погода на неделю: от +2 до +15 градусов.",
            },
            "selected": [{"text": "Weather tomorrow +2 to +15.", "source": "message", "topic": "weather"}],
        }

        filtered_context, filtered_memories, debug = _isolate_factual_prompt_context(
            memory_context=memory_context,
            retrieved_memories=list(memory_context["selected"]),
            web_intent="generic",
        )

        self.assertEqual(filtered_context["blocks"], memory_context["blocks"])
        self.assertEqual(filtered_memories, memory_context["selected"])
        self.assertEqual(debug, {})

    def test_price_caution_guard_replaces_overconfident_exact_claim(self) -> None:
        fixed, changed = _apply_web_factual_caution_guard(
            "The exact price is 39999 UAH.",
            meta={
                "web_used": True,
                "web_primary_category": "price",
                "factual_response_mode": "price",
                "web_evidence_quality": {
                    "score": 0.58,
                    "conflict_severity": 0.18,
                    "final_factual_confidence": 0.41,
                    "cautious_synthesis": True,
                },
            },
            web_evidence_context={
                "compact_citations": ["shop.example (2026-03-13)"],
                "summary": "Current listed price varies across sources.",
                "cautious_synthesis": True,
                "final_factual_confidence": 0.41,
            },
            web_intent="generic",
        )

        self.assertTrue(changed)
        self.assertNotIn("39999", fixed)
        self.assertTrue(str(fixed).strip())


if __name__ == "__main__":
    unittest.main()
