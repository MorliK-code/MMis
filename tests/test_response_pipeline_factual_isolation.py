from __future__ import annotations

import unittest

from core import DebugTrace
from core.character_runtime import CharacterRuntime, PromptPack
from prompt_engine.prompt_engine import PromptEngine
from core.response_pipeline import (
    EpisodeContinuityStage,
    PROFILE_BALANCED,
    MemoryRetrieveStage,
    PipelineContext,
    PipelineStage,
    PromptBuildStage,
    PromptEngineStage,
    ResponsePipeline,
    _apply_self_fact_factual_mode_guard,
    _isolate_self_memory_exact_context,
    _apply_web_evidence_to_prompt_pack,
    _apply_web_factual_caution_guard,
    _has_exact_self_facts,
    _isolate_factual_prompt_context,
    _resolve_web_factual_response_mode,
)
from memory.governor import GovernorProfileSnapshot
from memory.identity_core import IdentityCoreSnapshot


class ResponsePipelineFactualIsolationTests(unittest.TestCase):
    def test_response_pipeline_initializes_debug_trace_in_state(self) -> None:
        class _ProbeStage(PipelineStage):
            name = "probe"

            def __init__(self) -> None:
                self.seen_trace = None

            def run(self, ctx: PipelineContext) -> PipelineContext:
                self.seen_trace = ctx.state.get("debug_trace")
                ctx.text = "ok"
                ctx.stop = True
                return ctx

        probe = _ProbeStage()
        pipeline = ResponsePipeline(provider=object())
        pipeline._stages = {"probe": probe}
        pipeline._profiles[PROFILE_BALANCED] = ("probe",)
        meta_map = {"conversation_id": "conv-debug-trace"}

        pipeline.run(
            route="chat",
            user_msg="hello memory",
            state={"conversation_id": "conv-debug-trace"},
            meta=meta_map,
            retrieved_memories=[],
            traits={},
            policies={},
        )

        self.assertIsInstance(probe.seen_trace, DebugTrace)
        self.assertEqual(str(probe.seen_trace.user_text), "hello memory")
        self.assertTrue(str(probe.seen_trace.request_id))
        self.assertEqual(str(dict(meta_map.get("debug_trace") or {}).get("user_text") or ""), "hello memory")
        self.assertEqual(str(dict(meta_map.get("memory_debug_snapshot") or {}).get("user_text") or ""), "hello memory")

    def test_memory_retrieve_stores_active_profile_snapshot_from_governor(self) -> None:
        class _Result:
            def to_dict(self):
                return {
                    "blocks": {"self_facts": "- identity_name: Паша"},
                    "selected": [
                        {
                            "id": "fact:name",
                            "memory_type": "fact",
                            "score": 0.97,
                            "level": "l3_semantic",
                            "scope": "conversation",
                            "status": "active",
                            "metadata": {
                                "fact": {
                                    "subject": "user",
                                    "predicate": "identity_name",
                                    "value": "Паша",
                                }
                            },
                        }
                    ],
                    "dropped": [{"id": "msg:old", "reason": "budget"}],
                    "truncation_log": [],
                    "fact_expectation": {"expected_predicates": ["identity_name"]},
                    "self_facts_context": {"found_predicates": ["identity_name"]},
                    "dialog_episode_hits": [],
                    "recall_mode": "exact_fact_recall",
                }

        class _Manager:
            def build_context(self, _request):
                return _Result()

            def get_governor_profile_snapshot(self, namespace: str):
                return GovernorProfileSnapshot(
                    namespace=namespace,
                    active_facts={
                        "identity.name": {
                            "predicate": "identity_name",
                            "value": "Паша",
                            "record_id": "fact:name",
                        }
                    },
                    conflicts=[],
                    updated_at=123.0,
                )

        ctx = PipelineContext(
            route="chat",
            user_msg="как меня зовут?",
            clean_user_msg="как меня зовут?",
            state={"conversation_id": "conv-profile"},
            meta={"conversation_id": "conv-profile"},
            tags={},
            retrieved_memories=[],
            traits={},
            policies={},
            profile=PROFILE_BALANCED,
        )

        ctx = MemoryRetrieveStage(memory_manager=_Manager()).run(ctx)

        snapshot = dict(ctx.state.get("active_profile_snapshot") or {})
        self.assertEqual(snapshot.get("namespace"), "conv-profile")
        self.assertEqual(
            dict(snapshot.get("active_facts") or {}).get("identity.name", {}).get("value"),
            "Паша",
        )
        trace = ctx.state.get("debug_trace")
        self.assertIsInstance(trace, DebugTrace)
        self.assertEqual(str(trace.memory_retrieval.get("query") or ""), "как меня зовут?")
        self.assertEqual(
            list(trace.memory_retrieval.get("selected_facts") or [{}])[0].get("predicate"),
            "identity_name",
        )
        self.assertEqual(
            list(trace.memory_retrieval.get("exact_self_fact_hits", {}).get("self_facts_context", {}).get("found_predicates") or []),
            ["identity_name"],
        )
        self.assertEqual(dict(trace.active_profile or {}).get("namespace"), "conv-profile")

    def _legacy_test_memory_retrieve_resolves_active_task_into_state(self) -> None:
        class _Result:
            def to_dict(self):
                return {
                    "blocks": {
                        "recalled_dialog": (
                            "Topic: memory design\n"
                            "Summary: Discussed staged memory architecture.\n"
                            "Decisions:\n"
                            "- сначала доделываем память, потом веб\n"
                            "Open questions:\n"
                            "- how to build fusion retrieval"
                        ),
                    },
                        "selected": [
                            {
                                "id": "episode:memory-plan",
                                "memory_type": "episode",
                                "score": 0.78,
                                "metadata": {
                                    "dialog_episode": {
                                        "id": "episode:memory-plan",
                                        "topic": "memory design",
                                    "summary_short": "Discussed staged memory architecture.",
                                    "decisions": ["сначала доделываем память, потом веб"],
                                    "open_questions": ["how to build fusion retrieval"],
                                }
                            },
                        }
                    ],
                    "dropped": [],
                    "truncation_log": [],
                }

        class _Manager:
            def build_context(self, _request):
                return _Result()

        ctx = PipelineContext(
            route="chat",
            user_msg="вернись к тому плану по памяти",
            clean_user_msg="вернись к тому плану по памяти",
            state={"conversation_id": "conv-active-task"},
            meta={"conversation_id": "conv-active-task", "now_ts": 100.0},
            tags={},
            retrieved_memories=[],
            traits={},
            policies={},
            profile=PROFILE_BALANCED,
        )

        ctx = MemoryRetrieveStage(memory_manager=_Manager()).run(ctx)

        active_task = dict(ctx.state.get("active_task") or {})
        self.assertEqual(active_task.get("task_id"), "episode:memory-plan")
        self.assertEqual(active_task.get("status"), "active")
        self.assertEqual(ctx.state.get("active_goal"), "сначала доделываем память, потом веб")
        self.assertEqual(list(dict(ctx.state.get("active_tasks", [{}])[0]).get("decisions") or []), ["сначала доделываем память, потом веб"])

    def _legacy_test_memory_retrieve_keeps_active_task_for_short_follow_up(self) -> None:
        class _Result:
            def to_dict(self):
                return {"blocks": {}, "selected": [], "dropped": [], "truncation_log": []}

        class _Manager:
            def build_context(self, _request):
                return _Result()

        ctx = PipelineContext(
            route="chat",
            user_msg="ну давай, дальше",
            clean_user_msg="ну давай, дальше",
            state={
                "conversation_id": "conv-active-task-followup",
                "active_task": {
                    "task_id": "episode:memory-plan",
                    "topic": "memory design",
                    "status": "active",
                    "summary_short": "Discussed staged memory architecture.",
                    "current_goal": "сначала доделываем память, потом веб",
                    "decisions": ["сначала доделываем память, потом веб"],
                },
                "_active_task_source": "episode_planner",
            },
            meta={"conversation_id": "conv-active-task-followup", "now_ts": 150.0},
            tags={},
            retrieved_memories=[],
            traits={},
            policies={},
            profile=PROFILE_BALANCED,
        )

        ctx = MemoryRetrieveStage(memory_manager=_Manager()).run(ctx)

        active_task = dict(ctx.state.get("active_task") or {})
        self.assertEqual(active_task.get("task_id"), "episode:memory-plan")
        self.assertEqual(active_task.get("status"), "active")
        self.assertEqual(active_task.get("updated_at"), 150.0)

    def test_episode_continuity_resolves_active_task_into_state_from_dialog_episode_hit(self) -> None:
        class _Result:
            def to_dict(self):
                return {
                    "dialog_episode_hits": [
                        {
                            "record_id": "episode:memory-plan",
                            "score": 0.78,
                            "summary_short": "Discussed staged memory architecture.",
                            "summary_reasoning": "Finish memory before web and keep retrieval clean.",
                            "decisions": ["finish memory before web"],
                            "episode": {
                                "id": "episode:memory-plan",
                                "topic": "memory design",
                                "summary_short": "Discussed staged memory architecture.",
                                "summary_reasoning": "Finish memory before web and keep retrieval clean.",
                                "decisions": ["finish memory before web"],
                                "open_questions": ["how to build fusion retrieval"],
                                "updated_at": 88.0,
                            },
                        }
                    ],
                    "blocks": {
                        "recalled_dialog": (
                            "Topic: memory design\n"
                            "Summary: Discussed staged memory architecture.\n"
                            "Decisions:\n"
                            "- finish memory before web\n"
                            "Open questions:\n"
                            "- how to build fusion retrieval"
                        ),
                    },
                    "selected": [],
                    "dropped": [],
                    "truncation_log": [],
                }

        class _Manager:
            def build_context(self, _request):
                return _Result()

        ctx = PipelineContext(
            route="chat",
            user_msg="return to that memory plan",
            clean_user_msg="return to that memory plan",
            state={"conversation_id": "conv-active-task"},
            meta={"conversation_id": "conv-active-task", "now_ts": 100.0},
            tags={},
            retrieved_memories=[],
            traits={},
            policies={},
            profile=PROFILE_BALANCED,
        )

        ctx = MemoryRetrieveStage(memory_manager=_Manager()).run(ctx)
        self.assertEqual(dict(ctx.state.get("active_task") or {}), {})
        ctx = EpisodeContinuityStage().run(ctx)

        active_task = dict(ctx.state.get("active_task") or {})
        self.assertEqual(active_task.get("task_id"), "task:episode:memory-plan")
        self.assertEqual(active_task.get("status"), "waiting_user")
        self.assertEqual(ctx.state.get("active_goal"), "Finish memory before web and keep retrieval clean.")
        self.assertEqual(
            list(dict(ctx.state.get("active_tasks", [{}])[0]).get("decisions") or []),
            ["finish memory before web"],
        )
        trace = ctx.state.get("debug_trace")
        self.assertIsInstance(trace, DebugTrace)
        self.assertEqual(str(dict(trace.active_task or {}).get("source") or ""), "episode_hit")
        self.assertEqual(str(dict(trace.active_task or {}).get("reason") or ""), "episode_open_questions")
        self.assertEqual(str(dict(trace.active_task or {}).get("event") or ""), "resolved")

    def test_episode_continuity_keeps_active_task_for_short_follow_up(self) -> None:
        class _Result:
            def to_dict(self):
                return {"blocks": {}, "selected": [], "dropped": [], "truncation_log": []}

        class _Manager:
            def build_context(self, _request):
                return _Result()

        ctx = PipelineContext(
            route="chat",
            user_msg="continue",
            clean_user_msg="continue",
            state={
                "conversation_id": "conv-active-task-followup",
                "active_task": {
                    "task_id": "task:episode:memory-plan",
                    "topic": "memory design",
                    "status": "active",
                    "summary_short": "Discussed staged memory architecture.",
                    "current_goal": "finish memory before web",
                    "decisions": ["finish memory before web"],
                },
                "_active_task_source": "episode_planner",
            },
            meta={"conversation_id": "conv-active-task-followup", "now_ts": 150.0},
            tags={},
            retrieved_memories=[],
            traits={},
            policies={},
            profile=PROFILE_BALANCED,
        )

        ctx = MemoryRetrieveStage(memory_manager=_Manager()).run(ctx)
        ctx = EpisodeContinuityStage().run(ctx)

        active_task = dict(ctx.state.get("active_task") or {})
        self.assertEqual(active_task.get("task_id"), "task:episode:memory-plan")
        self.assertEqual(active_task.get("status"), "active")
        self.assertEqual(active_task.get("updated_at"), 150.0)
        trace = ctx.state.get("debug_trace")
        self.assertIsInstance(trace, DebugTrace)
        self.assertEqual(str(dict(trace.active_task or {}).get("source") or ""), "continuation")
        self.assertEqual(str(dict(trace.active_task or {}).get("reason") or ""), "short_followup")

    def test_episode_continuity_clears_active_task_when_user_closes_it(self) -> None:
        class _Result:
            def to_dict(self):
                return {"blocks": {}, "selected": [], "dropped": [], "truncation_log": []}

        class _Manager:
            def build_context(self, _request):
                return _Result()

        ctx = PipelineContext(
            route="chat",
            user_msg="done",
            clean_user_msg="done",
            state={
                "conversation_id": "conv-active-task-close",
                "active_task": {
                    "task_id": "task:episode:memory-plan",
                    "topic": "memory design",
                    "status": "active",
                    "summary_short": "Discussed staged memory architecture.",
                    "current_goal": "finish memory before web",
                    "source_episode_id": "episode:memory-plan",
                },
                "_active_task_source": "episode_planner",
            },
            meta={"conversation_id": "conv-active-task-close", "now_ts": 160.0},
            tags={},
            retrieved_memories=[],
            traits={},
            policies={},
            profile=PROFILE_BALANCED,
        )

        ctx = MemoryRetrieveStage(memory_manager=_Manager()).run(ctx)
        ctx = EpisodeContinuityStage().run(ctx)

        self.assertEqual(dict(ctx.state.get("active_task") or {}), {})
        self.assertNotIn("active_goal", ctx.state)
        self.assertNotIn("active_tasks", ctx.state)
        self.assertNotIn("_active_task_source", ctx.state)
        trace = ctx.state.get("debug_trace")
        self.assertIsInstance(trace, DebugTrace)
        self.assertEqual(str(dict(trace.active_task or {}).get("event") or ""), "close")
        self.assertEqual(str(dict(trace.active_task or {}).get("reason") or ""), "explicit_close_phrase")
        self.assertTrue(any("active_task_closed" in line for line in list(ctx.logs or [])))

    def test_episode_continuity_clears_active_task_on_explicit_topic_switch(self) -> None:
        class _Result:
            def to_dict(self):
                return {"blocks": {}, "selected": [], "dropped": [], "truncation_log": []}

        class _Manager:
            def build_context(self, _request):
                return _Result()

        ctx = PipelineContext(
            route="chat",
            user_msg="by the way, another topic",
            clean_user_msg="by the way, another topic",
            state={
                "conversation_id": "conv-active-task-switch",
                "active_task": {
                    "task_id": "task:episode:memory-plan",
                    "topic": "memory design",
                    "status": "active",
                    "summary_short": "Discussed staged memory architecture.",
                    "current_goal": "finish memory before web",
                    "source_episode_id": "episode:memory-plan",
                },
                "_active_task_source": "episode_planner",
            },
            meta={"conversation_id": "conv-active-task-switch", "now_ts": 170.0},
            tags={},
            retrieved_memories=[],
            traits={},
            policies={},
            profile=PROFILE_BALANCED,
        )

        ctx = MemoryRetrieveStage(memory_manager=_Manager()).run(ctx)
        ctx = EpisodeContinuityStage().run(ctx)

        self.assertEqual(dict(ctx.state.get("active_task") or {}), {})
        self.assertNotIn("active_goal", ctx.state)
        self.assertNotIn("active_tasks", ctx.state)
        self.assertNotIn("_active_task_source", ctx.state)
        self.assertTrue(any("active_task_cleared" in line for line in list(ctx.logs or [])))
        trace = ctx.state.get("debug_trace")
        self.assertIsInstance(trace, DebugTrace)
        self.assertEqual(str(dict(trace.active_task or {}).get("event") or ""), "clear")
        self.assertEqual(str(dict(trace.active_task or {}).get("reason") or ""), "switch_topic_phrase")

    def test_episode_continuity_builds_active_task_from_runtime_memory_hints(self) -> None:
        class _Result:
            def to_dict(self):
                return {
                    "blocks": {},
                    "selected": [],
                    "dropped": [],
                    "truncation_log": [],
                    "open_questions": ["how to build fusion retrieval"],
                    "current_decisions": ["finish memory before web"],
                }

        class _Manager:
            def build_context(self, _request):
                return _Result()

        ctx = PipelineContext(
            route="chat",
            user_msg="return to the plan",
            clean_user_msg="return to the plan",
            state={"conversation_id": "conv-runtime-task-hints"},
            meta={"conversation_id": "conv-runtime-task-hints", "now_ts": 180.0},
            tags={},
            retrieved_memories=[],
            traits={},
            policies={},
            profile=PROFILE_BALANCED,
        )

        ctx = MemoryRetrieveStage(memory_manager=_Manager()).run(ctx)
        ctx = EpisodeContinuityStage().run(ctx)

        active_task = dict(ctx.state.get("active_task") or {})
        self.assertTrue(str(active_task.get("task_id") or "").startswith("task:runtime:"))
        self.assertEqual(active_task.get("status"), "waiting_user")
        self.assertEqual(active_task.get("current_goal"), "finish memory before web")
        self.assertEqual(list(active_task.get("decisions") or []), ["finish memory before web"])
        self.assertEqual(list(active_task.get("open_questions") or []), ["how to build fusion retrieval"])
        trace = ctx.state.get("debug_trace")
        self.assertIsInstance(trace, DebugTrace)
        self.assertEqual(str(dict(trace.active_task or {}).get("source") or ""), "runtime_hints")
        self.assertEqual(str(dict(trace.active_task or {}).get("reason") or ""), "runtime_open_questions_fallback")

    def test_prompt_engine_memory_block_puts_self_facts_before_general_memory(self) -> None:
        block = PromptEngine._build_memory_retrieval_block(
            blocks={},
            memory_blocks={
                "memory_recall_mode": "- mode: exact_fact_recall",
                "self_facts": "- environment_gpu_model: RTX 3050 Ti",
                "fact_expectation_check": "- expected_predicates: environment_gpu_model",
                "relevant_claims": "- user uses VS Code",
                "recalled_dialog": "Topic: memory design\nSummary: You discussed memory layers.",
                "document_evidence": "Chunk 12: where ollama is called",
                "supporting_messages": "- user: у меня rtx 3050 ti",
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
        self.assertIn("[RELEVANT_CLAIMS]\n- user uses VS Code", block)
        self.assertIn("[RECALLED_DIALOG]\nTopic: memory design\nSummary: You discussed memory layers.", block)
        self.assertIn("[DOCUMENT_EVIDENCE]\nChunk 12: where ollama is called", block)
        self.assertIn("[SUPPORTING_MESSAGES]\n- user: у меня rtx 3050 ti", block)
        self.assertIn("[WORKING_MEMORY]\nold working note", block)
        self.assertIn("[SEMANTIC_FACTS]\nold semantic noise", block)
        self.assertLess(block.index("[SELF_FACTS]"), block.index("[WORKING_MEMORY]"))
        self.assertLess(block.index("[FACT_EXPECTATION_CHECK]"), block.index("[SEMANTIC_FACTS]"))
        self.assertLess(block.index("[RELEVANT_CLAIMS]"), block.index("[WORKING_MEMORY]"))
        self.assertLess(block.index("[RECALLED_DIALOG]"), block.index("[SEMANTIC_FACTS]"))

    def test_prompt_engine_prefers_recalled_dialog_over_raw_episodic_snippets(self) -> None:
        block = PromptEngine._build_memory_retrieval_block(
            blocks={},
            memory_blocks={
                "recalled_dialog": (
                    "Topic: memory design\n"
                    "Summary: Discussed staged memory architecture.\n"
                    "Reasoning: Discussed memory layers. Decided to separate facts and claims.\n"
                    "Decisions:\n"
                    "- finish memory before web\n"
                    "Open questions:\n"
                    "- how to build fusion retrieval"
                ),
                "supporting_messages": (
                    "- user: сначала доделываем память\n"
                    "- assistant: потом отдельно вернемся к вебу"
                ),
                "retrieved_episodic": "raw episodic snippet\nanother raw snippet",
            },
        )

        self.assertIn("[RECALLED_DIALOG]", block)
        self.assertIn("Summary: Discussed staged memory architecture.", block)
        self.assertIn("Reasoning: Discussed memory layers.", block)
        self.assertIn("Decisions:", block)
        self.assertIn("Open questions:", block)
        self.assertIn("[SUPPORTING_MESSAGES]", block)
        self.assertNotIn("[EPISODIC_MEMORIES]", block)
        self.assertNotIn("raw episodic snippet", block)

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

    def test_prompt_build_creates_persona_snapshot_and_runtime_uses_overlay(self) -> None:
        runtime = CharacterRuntime()
        runtime.storage.save_identity_core(
            "asya",
            {
                "addressing": {
                    "canonical_name": "Паш",
                    "allowed_forms": ["Паш"],
                    "forbidden_forms": ["Пашка"],
                    "use_name_by_default": False,
                }
                ,
                "interaction_style": {
                    "prefers_directness": 0.82,
                    "prefers_short_answers": 0.55,
                    "allows_light_teasing": True,
                    "technical_collaboration_style": "high",
                },
                "boundaries": {
                    "avoid_overloaded_intros": True,
                    "avoid_baby_talk": True,
                    "do_not_invent_user_facts": True,
                },
                "emotional_handling": {
                    "deescalate_on_irritation": True,
                    "treat_short_replies_as_low_bandwidth": True,
                    "warmth_upshift_on_user_distress": 0.24,
                    "playfulness_downshift_on_user_distress": 0.33,
                },
            },
        )
        ctx = PipelineContext(
            route="chat",
            user_msg="как меня зовут?",
            clean_user_msg="как меня зовут?",
            state={
                "active_character_id": "asya",
                "active_profile_snapshot": {
                    "identity_name": "Паша",
                    "assistant_warmth": 0.81,
                    "assistant_empathy": 0.77,
                },
            },
            tags={},
            retrieved_memories=[],
            traits={},
            policies={},
            profile=PROFILE_BALANCED,
            memory_context={"blocks": {}, "selected": []},
            meta={"emotion": "frustrated"},
        )

        ctx = PromptBuildStage(character_runtime=runtime).run(ctx)

        persona_snapshot = dict(ctx.state.get("persona_snapshot") or {})
        self.assertEqual(
            dict(persona_snapshot.get("user_addressing") or {}).get("canonical_name"),
            "Паш",
        )
        self.assertEqual(
            dict(persona_snapshot.get("user_profile_hints") or {}).get("user_name"),
            "Паш",
        )
        self.assertEqual(
            dict(dict(ctx.state.get("identity_core") or {}).get("addressing") or {}).get("canonical_name"),
            "Паш",
        )
        self.assertEqual(
            dict(dict(ctx.state.get("identity_core") or {}).get("interaction_style") or {}).get("prefers_directness"),
            0.82,
        )
        self.assertTrue(
            bool(dict(dict(ctx.state.get("identity_core") or {}).get("boundaries") or {}).get("avoid_baby_talk"))
        )
        self.assertTrue(bool(dict(persona_snapshot.get("boundaries") or {}).get("avoid_baby_talk")))
        self.assertTrue(
            bool(dict(dict(ctx.state.get("identity_core") or {}).get("emotional_handling") or {}).get("deescalate_on_irritation"))
        )
        self.assertEqual(
            dict(persona_snapshot.get("emotional_handling") or {}).get("warmth_upshift_on_user_distress"),
            0.24,
        )
        self.assertEqual(
            float(dict(persona_snapshot.get("response_bias") or {}).get("warmth_upshift") or 0.0),
            0.24,
        )
        self.assertIn("persona_snapshot_sources", "\n".join(ctx.logs))
        self.assertIn("persona_snapshot_built", "\n".join(ctx.logs))
        self.assertIn("persona_snapshot_applied", "\n".join(ctx.logs))
        self.assertEqual(
            dict(ctx.meta.get("persona_snapshot_sources") or {}).get("mood_source"),
            "meta.emotion",
        )
        trace = ctx.state.get("debug_trace")
        self.assertIsInstance(trace, DebugTrace)
        self.assertEqual(
            dict(dict(trace.persona_snapshot or {}).get("addressing_source") or {}).get("canonical_name"),
            "identity_core",
        )
        self.assertEqual(
            dict(dict(trace.persona_snapshot or {}).get("trait_overrides") or {}).get("warmth", {}).get("source"),
            "identity_core_baseline",
        )
        self.assertEqual(
            dict(dict(trace.persona_snapshot or {}).get("trait_overrides") or {}).get("directness", {}).get("source"),
            "identity_core",
        )
        self.assertEqual(
            dict(dict(trace.persona_snapshot or {}).get("sources") or {}).get("interaction_style_source", {}).get("prefers_directness"),
            "identity_core",
        )
        self.assertEqual(
            dict(dict(trace.persona_snapshot or {}).get("boundary_source") or {}).get("avoid_baby_talk"),
            "identity_core",
        )
        self.assertEqual(
            dict(dict(trace.persona_snapshot or {}).get("emotional_handling_source") or {}).get("deescalate_on_irritation"),
            "identity_core",
        )
        self.assertEqual(
            dict(dict(trace.identity_core or {}).get("snapshot") or {}).get("character_id"),
            "asya",
        )
        self.assertIn(
            "canonical_name",
            list(dict(dict(trace.identity_core or {}).get("active_identity_keys") or {}).get("addressing") or []),
        )
        self.assertEqual(
            dict(dict(trace.identity_core or {}).get("sources") or {}).get("addressing", {}).get("canonical_name"),
            "identity_core",
        )
        self.assertEqual(
            float(dict(trace.identity_core or {}).get("trait_baselines", {}).get("warmth_baseline") or 0.0),
            0.81,
        )
        self.assertIn(
            "user_name",
            list(
                dict(ctx.meta.get("persona_snapshot_applied") or {}).get("persistent_fields")
                or []
            ),
        )
        self.assertIn(
            "technical_collaboration",
            list(
                dict(ctx.meta.get("persona_snapshot_applied") or {}).get("persistent_fields")
                or []
            ),
        )
        self.assertIn(
            "avoid_baby_talk",
            list(
                dict(ctx.meta.get("persona_snapshot_applied") or {}).get("persistent_fields")
                or []
            ),
        )

    def test_prompt_build_prefers_memory_identity_core_snapshot_over_runtime_storage_file(self) -> None:
        class _Manager:
            @staticmethod
            def get_identity_core_snapshot(namespace: str):
                if str(namespace or "") != "conv-memory-identity":
                    return None
                return IdentityCoreSnapshot(
                    addressing={
                        "canonical_name": "Паша",
                        "allowed_forms": ["Паша", "Паш"],
                    },
                    boundaries={
                        "avoid_baby_tone": True,
                        "avoid_inventing_user_facts": True,
                    },
                    emotional_rules={
                        "frustration_softening": 0.24,
                    },
                    assistant_trait_baseline={
                        "warmth_baseline": 0.64,
                    },
                )

        runtime = CharacterRuntime()
        runtime.storage.save_identity_core(
            "asya",
            {
                "addressing": {
                    "canonical_name": "Павел",
                    "allowed_forms": ["Павел"],
                },
                "boundaries": {
                    "avoid_baby_talk": False,
                },
            },
        )
        ctx = PipelineContext(
            route="chat",
            user_msg="как меня звать?",
            clean_user_msg="как меня звать?",
            state={
                "active_character_id": "asya",
                "conversation_id": "conv-memory-identity",
                "active_profile_snapshot": {
                    "identity_name": "Павел",
                },
            },
            tags={},
            retrieved_memories=[],
            traits={},
            policies={},
            profile=PROFILE_BALANCED,
            memory_context={"blocks": {}, "selected": []},
            meta={
                "conversation_id": "conv-memory-identity",
                "memory_manager": _Manager(),
            },
        )

        ctx = PromptBuildStage(character_runtime=runtime).run(ctx)

        identity_core = dict(ctx.state.get("identity_core") or {})
        self.assertEqual(
            dict(identity_core.get("addressing") or {}).get("canonical_name"),
            "Паша",
        )
        self.assertTrue(
            bool(dict(identity_core.get("boundaries") or {}).get("avoid_baby_talk"))
        )
        self.assertTrue(
            bool(dict(identity_core.get("boundaries") or {}).get("do_not_invent_user_facts"))
        )
        self.assertTrue(
            bool(dict(identity_core.get("emotional_handling") or {}).get("deescalate_on_irritation"))
        )
        self.assertEqual(
            float(dict(identity_core.get("assistant_trait_baseline") or {}).get("warmth_baseline") or 0.0),
            0.64,
        )
        self.assertEqual(
            dict(dict(ctx.state.get("identity_core_memory_snapshot") or {}).get("addressing") or {}).get("canonical_name"),
            "Паша",
        )
        trace = ctx.state.get("debug_trace")
        self.assertIsInstance(trace, DebugTrace)
        self.assertEqual(
            dict(dict(trace.identity_core or {}).get("memory_snapshot_input") or {}).get("addressing", {}).get("canonical_name"),
            "Паша",
        )
        self.assertEqual(
            dict(dict(trace.identity_core or {}).get("runtime_fallback_input") or {}).get("addressing", {}).get("canonical_name"),
            "Павел",
        )
        self.assertIn(
            "deescalate_on_irritation",
            list(
                dict(ctx.meta.get("persona_snapshot_applied") or {}).get("persistent_fields")
                or []
            ),
        )
        persona_snapshot = dict(ctx.state.get("persona_snapshot") or {})
        persona_block = CharacterRuntime(autosave=False)._build_persona_block(
            {
                "active_character_id": "asya",
                "persona_snapshot": dict(persona_snapshot),
            },
            {},
            {},
        )
        self.assertIn("canonical_name: Паш", persona_block)
        self.assertIn("[PERSONA_BOUNDARIES]", persona_block)
        self.assertIn("[PERSONA_EMOTIONAL_HANDLING]", persona_block)
        self.assertIn("Do not invent facts about the user", persona_block)
        self.assertIn("de-escalate first and then move to solving", persona_block)
        self.assertGreaterEqual(float(dict(persona_snapshot.get("stable_traits") or {}).get("directness") or 0.0), 0.82)

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
                "self_facts": "- identity_name: Паша",
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


class ActiveTaskPromptTests(unittest.TestCase):
    def test_prompt_engine_renders_active_task_even_without_memory_blocks(self) -> None:
        block = PromptEngine._build_memory_retrieval_block(
            blocks={},
            memory_blocks={},
            state_map={
                "active_task": {
                    "topic": "memory design",
                    "status": "active",
                    "current_goal": "Finish memory before web.",
                }
            },
        )

        self.assertIn("[ACTIVE_TASK]", block)
        self.assertIn("- topic: memory design", block)
        self.assertIn("- current_goal: Finish memory before web.", block)

    def test_prompt_engine_renders_active_task_block(self) -> None:
        block = PromptEngine._build_memory_retrieval_block(
            blocks={},
            memory_blocks={
                "memory_recall_mode": "- mode: contextual_recall",
                "recalled_dialog": "Topic: memory design\nSummary: You discussed memory layers.",
                "working_memory": "old working note",
            },
            state_map={
                "active_task": {
                    "topic": "memory design",
                    "status": "waiting_user",
                    "summary_short": "Discussed staged memory architecture.",
                    "current_goal": "Finish memory before web and keep retrieval clean.",
                    "next_steps": ["finish memory before web"],
                    "open_questions": ["how to build fusion retrieval"],
                }
            },
        )

        self.assertIn("[ACTIVE_TASK]", block)
        self.assertIn("- topic: memory design", block)
        self.assertIn("- status: waiting_user", block)
        self.assertIn("- current_goal: Finish memory before web and keep retrieval clean.", block)
        self.assertLess(block.index("[ACTIVE_TASK]"), block.index("[WORKING_MEMORY]"))

    def test_prompt_build_and_engine_include_active_task_block(self) -> None:
        ctx = PipelineContext(
            route="chat",
            user_msg="continue",
            clean_user_msg="continue",
            state={
                "active_task": {
                    "task_id": "task:episode:memory-plan",
                    "topic": "memory design",
                    "status": "waiting_user",
                    "summary_short": "Discussed staged memory architecture.",
                    "current_goal": "Finish memory before web and keep retrieval clean.",
                    "next_steps": ["finish memory before web"],
                    "open_questions": ["how to build fusion retrieval"],
                }
            },
            meta={},
            tags={},
            retrieved_memories=[],
            traits={},
            policies={},
            profile=PROFILE_BALANCED,
            memory_context={
                "blocks": {
                    "memory_recall_mode": "- mode: contextual_recall",
                    "recalled_dialog": "Topic: memory design\nSummary: You discussed memory layers.",
                },
                "selected": [],
            },
        )

        ctx = PromptBuildStage(character_runtime=CharacterRuntime()).run(ctx)
        ctx = PromptEngineStage(prompt_engine=PromptEngine()).run(ctx)

        system_prompt = str(ctx.prompt_messages[0].content if ctx.prompt_messages else "")
        self.assertIn("[ACTIVE_TASK]", system_prompt)
        self.assertIn("- topic: memory design", system_prompt)
        self.assertIn("- status: waiting_user", system_prompt)
        self.assertIn("- current_goal: Finish memory before web and keep retrieval clean.", system_prompt)
        trace = ctx.state.get("debug_trace")
        self.assertIsInstance(trace, DebugTrace)
        self.assertEqual(
            str(dict(trace.persona_snapshot or {}).get("active_mode") or ""),
            "chatting",
        )
        self.assertIn("[ACTIVE_TASK]", str(dict(trace.prompt_pack or {}).get("memory_block", {}).get("text") or ""))
        self.assertGreaterEqual(int(dict(trace.prompt_pack or {}).get("token_estimate") or 0), 0)
        self.assertIn("ACTIVE_TASK", list(dict(trace.prompt_pack or {}).get("included_memory_blocks") or []))
        self.assertIn("RECALLED_DIALOG", list(dict(trace.prompt_pack or {}).get("included_memory_blocks") or []))

    def test_prompt_trace_includes_omitted_memory_blocks_from_context_isolation(self) -> None:
        ctx = PipelineContext(
            route="chat",
            user_msg="which gpu do i have?",
            clean_user_msg="which gpu do i have?",
            state={},
            meta={
                "prompt_context_isolation": {
                    "dropped_memory_blocks": [
                        {
                            "block": "working_memory",
                            "reason": "self_memory_exact_noise",
                            "preview": "old working note",
                        }
                    ]
                }
            },
            tags={},
            retrieved_memories=[],
            traits={},
            policies={},
            profile=PROFILE_BALANCED,
            memory_context={
                "blocks": {
                    "memory_recall_mode": "- mode: exact_fact_recall",
                    "self_facts": "- environment_gpu_model: RTX 3050 Ti",
                    "working_memory": "old working note",
                },
                "selected": [],
            },
        )

        ctx = PromptBuildStage(character_runtime=CharacterRuntime()).run(ctx)
        ctx = PromptEngineStage(prompt_engine=PromptEngine()).run(ctx)

        trace = ctx.state.get("debug_trace")
        self.assertIsInstance(trace, DebugTrace)
        omitted = list(dict(trace.prompt_pack or {}).get("omitted_memory_blocks") or [])
        self.assertTrue(any(dict(row).get("block") == "WORKING_MEMORY" for row in omitted))
        self.assertTrue(
            any(
                dict(row).get("block") == "WORKING_MEMORY"
                and dict(row).get("reason") == "self_memory_exact_noise"
                for row in omitted
            )
        )


if __name__ == "__main__":
    unittest.main()
