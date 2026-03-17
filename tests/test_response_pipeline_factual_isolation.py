from __future__ import annotations

import unittest

from core.character_runtime import CharacterRuntime, PromptPack
from prompt_engine.prompt_engine import PromptEngine
from core.response_pipeline import (
    PROFILE_BALANCED,
    PipelineContext,
    PromptBuildStage,
    PromptEngineStage,
    _apply_self_fact_factual_mode_guard,
    _isolate_self_memory_exact_context,
    _apply_web_evidence_to_prompt_pack,
    _apply_web_factual_caution_guard,
    _has_exact_self_facts,
    _isolate_factual_prompt_context,
    _resolve_web_factual_response_mode,
)


class ResponsePipelineFactualIsolationTests(unittest.TestCase):
    def test_prompt_engine_memory_block_puts_self_facts_before_general_memory(self) -> None:
        block = PromptEngine._build_memory_retrieval_block(
            blocks={},
            memory_blocks={
                "memory_recall_mode": "- mode: exact_fact_recall",
                "self_facts": "- environment_gpu_model: RTX 3050 Ti",
                "fact_expectation_check": "- expected_predicates: environment_gpu_model",
                "working_memory": "old working note",
                "retrieved_semantic": "old semantic noise",
            },
        )

        self.assertIn("[MEMORY_RECALL_MODE]\n- mode: exact_fact_recall", block)
        self.assertIn("[SELF_FACTS]\n- environment_gpu_model: RTX 3050 Ti", block)
        self.assertIn(
            "[FACT_EXPECTATION_CHECK]\n- expected_predicates: environment_gpu_model",
            block,
        )
        self.assertIn("[WORKING_MEMORY]\nold working note", block)
        self.assertIn("[SEMANTIC_FACTS]\nold semantic noise", block)
        self.assertLess(block.index("[SELF_FACTS]"), block.index("[WORKING_MEMORY]"))
        self.assertLess(block.index("[FACT_EXPECTATION_CHECK]"), block.index("[SEMANTIC_FACTS]"))

    def test_prompt_build_and_prompt_engine_keep_self_facts_in_final_memory_section(self) -> None:
        ctx = PipelineContext(
            route="chat",
            user_msg="какая у меня видеокарта?",
            clean_user_msg="какая у меня видеокарта?",
            state={},
            meta={},
            tags={},
            retrieved_memories=[
                {
                    "id": "fact:gpu",
                    "text": "user.environment_gpu_model=RTX 3050 Ti",
                    "memory_type": "fact",
                    "level": "l3_semantic",
                    "scope": "conversation",
                    "status": "active",
                    "score": 0.97,
                    "metadata": {
                        "fact": {
                            "predicate": "environment_gpu_model",
                            "subject": "user",
                            "value": "RTX 3050 Ti",
                        }
                    },
                },
                {
                    "id": "msg:assistant",
                    "text": "Чтобы узнать модель видеокарты, открой диспетчер устройств.",
                    "memory_type": "message",
                    "level": "l2_episodic",
                    "scope": "conversation",
                    "status": "active",
                    "metadata": {"source_kind": "assistant_reply"},
                },
            ],
            traits={},
            policies={},
            profile=PROFILE_BALANCED,
            memory_context={
                "blocks": {
                    "memory_recall_mode": "- mode: exact_fact_recall",
                    "self_facts": "- environment_gpu_model: RTX 3050 Ti",
                    "fact_expectation_check": "- expected_predicates: environment_gpu_model",
                    "working_memory": "old working note",
                    "retrieved_semantic": "old semantic noise",
                },
                "selected": [
                    {
                        "id": "fact:gpu",
                        "text": "user.environment_gpu_model=RTX 3050 Ti",
                        "memory_type": "fact",
                        "level": "l3_semantic",
                        "scope": "conversation",
                        "status": "active",
                        "score": 0.97,
                        "metadata": {
                            "fact": {
                                "predicate": "environment_gpu_model",
                                "subject": "user",
                                "value": "RTX 3050 Ti",
                            }
                        },
                    },
                    {
                        "id": "msg:assistant",
                        "text": "Чтобы узнать модель видеокарты, открой диспетчер устройств.",
                        "memory_type": "message",
                        "level": "l2_episodic",
                        "scope": "conversation",
                        "status": "active",
                        "metadata": {"source_kind": "assistant_reply"},
                    },
                ],
            },
        )

        build_stage = PromptBuildStage(character_runtime=CharacterRuntime())
        engine_stage = PromptEngineStage(prompt_engine=PromptEngine())

        ctx = build_stage.run(ctx)
        ctx = engine_stage.run(ctx)

        system_prompt = str(ctx.prompt_sections.get("system") or "")
        self.assertIn("<<<MEMORY>>>", system_prompt)
        self.assertIn("[SELF_FACTS]\n- environment_gpu_model: RTX 3050 Ti", system_prompt)
        self.assertIn(
            "[FACT_EXPECTATION_CHECK]\n- expected_predicates: environment_gpu_model",
            system_prompt,
        )
        self.assertNotIn("old working note", system_prompt)
        self.assertNotIn("<<<RECENT_CHAT>>>", system_prompt)
        self.assertNotIn("<<<LONG_SUMMARY>>>", system_prompt)

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
                "session_summary": "РџРѕРіРѕРґР° РЅР° РЅРµРґРµР»СЋ: РѕС‚ +2 РґРѕ +15 РіСЂР°РґСѓСЃРѕРІ, РґРѕР¶РґСЊ Рё РІРµС‚РµСЂ.",
                "working_memory": "РџРѕР»СЊР·РѕРІР°С‚РµР»СЊ СЃРїСЂР°С€РёРІР°РµС‚ РєСѓСЂСЃ РґРѕР»Р»Р°СЂР° РІ РіСЂРЅ РЅР° СЃРµРіРѕРґРЅСЏ.",
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
        self.assertIn("retrieved_semantic", blocks)
        self.assertNotIn("working_memory", blocks)
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
                "session_summary": "РџРѕРіРѕРґР° РЅР° РЅРµРґРµР»СЋ: РѕС‚ +2 РґРѕ +15 РіСЂР°РґСѓСЃРѕРІ.",
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

    def test_self_facts_block_marks_exact_self_fact_presence(self) -> None:
        memory_context = {
            "blocks": {
                "self_facts": "- environment_gpu_model: RTX 3050 Ti",
            },
            "fact_expectation": {
                "found_predicates": ["environment_gpu_model"],
            },
        }

        self.assertTrue(_has_exact_self_facts(memory_context))

    def test_self_facts_switch_to_self_memory_exact_mode_for_prompt(self) -> None:
        memory_context = {
            "blocks": {
                "self_facts": "- environment_gpu_model: RTX 3050 Ti",
            },
            "fact_expectation": {
                "found_predicates": ["environment_gpu_model"],
            },
        }

        mode, used, debug = _apply_self_fact_factual_mode_guard(
            factual_response_mode="price",
            web_used=True,
            memory_context=memory_context,
        )

        self.assertEqual(mode, "self_memory_exact")
        self.assertFalse(used)
        self.assertEqual(debug.get("reason"), "exact_self_fact_present")
        self.assertTrue(debug.get("skip_factual_context_isolation"))
        self.assertEqual(debug.get("factual_response_mode"), "self_memory_exact")

    def test_exact_self_facts_enable_self_memory_exact_even_without_web_mode(self) -> None:
        memory_context = {
            "blocks": {
                "self_facts": "- identity_name: РџР°С€Р°",
            },
            "fact_expectation": {
                "found_predicates": ["identity_name"],
            },
        }

        mode, used, debug = _apply_self_fact_factual_mode_guard(
            factual_response_mode="",
            web_used=False,
            memory_context=memory_context,
        )

        self.assertEqual(mode, "self_memory_exact")
        self.assertFalse(used)
        self.assertEqual(debug.get("reason"), "exact_self_fact_present")

    def test_web_evidence_is_not_injected_when_self_facts_are_present(self) -> None:
        pack = PromptPack(
            system_prompt="system",
            user_message="user",
            full_prompt="",
            messages=[],
            blocks={"self_facts": "- environment_gpu_model: RTX 3050 Ti"},
            token_usage={},
            budgets={},
        )

        updated = _apply_web_evidence_to_prompt_pack(
            pack,
            web_evidence_context={
                "summary": "Current web result says something else.",
                "compact_citations": ["example.com (2026-03-17)"],
            },
        )

        self.assertNotIn("web_evidence", updated.blocks)
        self.assertTrue(updated.cut_info.get("web_evidence_skipped_for_self_facts"))

    def test_prompt_engine_ignores_stale_web_context_when_web_not_used(self) -> None:
        block = PromptEngine._build_web_evidence_block(
            blocks={},
            memory_blocks={},
            state_map={
                "context_tags": {
                    "web_used": "false",
                    "web_query_intent": "generic",
                },
                "web_evidence_context": {
                    "prompt_block": "[WEB_EVIDENCE]\n- summary: old price context",
                    "numeric_profile": "price",
                },
            },
        )

        self.assertEqual(block, "")

    def test_prompt_engine_skips_web_evidence_for_self_memory_exact_even_if_web_used_tag_is_true(self) -> None:
        block = PromptEngine._build_web_evidence_block(
            blocks={},
            memory_blocks={
                "self_facts": "- environment_gpu_model: RTX 3050 Ti",
            },
            state_map={
                "context_tags": {
                    "web_used": "true",
                    "self_memory_exact": "true",
                    "factual_response_mode": "self_memory_exact",
                    "web_query_intent": "generic",
                },
                "web_evidence_context": {
                    "prompt_block": "[WEB_EVIDENCE]\n- summary: stale web context",
                    "numeric_profile": "price",
                },
            },
        )

        self.assertEqual(block, "")

    def test_missing_expected_self_fact_does_not_enable_self_memory_exact_mode(self) -> None:
        memory_context = {
            "blocks": {},
            "fact_expectation": {
                "expected_predicates": ["environment_ram_gb", "environment_memory_gb"],
                "found_predicates": [],
                "missing_predicates": ["environment_ram_gb", "environment_memory_gb"],
            },
            "self_facts_context": {},
        }

        mode, used, debug = _apply_self_fact_factual_mode_guard(
            factual_response_mode="price",
            web_used=True,
            memory_context=memory_context,
        )

        self.assertEqual(mode, "price")
        self.assertTrue(used)
        self.assertEqual(debug, {})

    def test_self_fact_presence_comes_from_relevant_self_facts_context(self) -> None:
        memory_context = {
            "blocks": {},
            "fact_expectation": {
                "expected_predicates": ["environment_ram_gb"],
                "found_predicates": [],
            },
            "self_facts_context": {
                "found_predicates": ["environment_ram_gb"],
                "found_facts": {
                    "environment_ram_gb": [
                        {"value": 32},
                    ],
                },
            },
        }

        self.assertTrue(_has_exact_self_facts(memory_context))

    def test_self_memory_exact_isolation_keeps_only_exact_fact_and_one_supporting_user_message(self) -> None:
        memory_context = {
            "blocks": {
                "memory_recall_mode": "- mode: exact_fact_recall",
                "self_facts": "- environment_gpu_model: RTX 3050 Ti",
                "fact_expectation_check": "- expected_predicates: environment_gpu_model",
                "working_memory": "old working note",
                "session_summary": "old summary",
                "retrieved_semantic": "old semantic noise",
                "retrieved_episodic": "old episodic noise",
                "active_tool_state": "WEB_TEMP: old web temp summary",
            },
            "selected": [],
        }
        retrieved = [
            {
                "id": "fact:gpu",
                "text": "user.environment_gpu_model=RTX 3050 Ti",
                "memory_type": "fact",
                "level": "l3_semantic",
                "scope": "conversation",
                "status": "active",
                "score": 0.97,
                "metadata": {
                    "fact": {
                        "predicate": "environment_gpu_model",
                        "subject": "user",
                        "value": "RTX 3050 Ti",
                    }
                },
            },
            {
                "id": "msg:user",
                "text": "у меня rtx 3050 ti",
                "memory_type": "message",
                "level": "l2_episodic",
                "scope": "conversation",
                "status": "active",
                "metadata": {"source_kind": "user"},
            },
            {
                "id": "msg:assistant",
                "text": "Чтобы узнать модель видеокарты, открой диспетчер устройств.",
                "memory_type": "message",
                "level": "l2_episodic",
                "scope": "conversation",
                "status": "active",
                "metadata": {"source_kind": "assistant_reply"},
            },
        ]

        filtered_context, filtered_memories, debug = _isolate_self_memory_exact_context(
            memory_context=memory_context,
            retrieved_memories=retrieved,
        )

        blocks = filtered_context["blocks"]
        self.assertEqual(
            set(blocks.keys()),
            {"memory_recall_mode", "self_facts", "fact_expectation_check", "exact_fact_evidence", "supporting_message"},
        )
        self.assertNotIn("working_memory", blocks)
        self.assertNotIn("session_summary", blocks)
        self.assertNotIn("retrieved_episodic", blocks)
        self.assertNotIn("active_tool_state", blocks)
        self.assertNotIn("retrieved_semantic", blocks)
        self.assertEqual([str(row.get("id") or "") for row in filtered_memories], ["fact:gpu", "msg:user"])
        assistant_drop = next(
            (row for row in debug["dropped_retrieved_memories"] if str(row.get("id") or "") == "msg:assistant"),
            {},
        )
        self.assertEqual(
            str(assistant_drop.get("reason") or ""),
            "assistant_reply_not_allowed_in_self_memory_exact",
        )

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
