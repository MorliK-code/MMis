import unittest
import importlib
import sys
import types

from core.response_pipeline import PROFILE_AUTONOMOUS, ResponsePipeline
from llm.provider_base import LLMChunk, LLMResponse, Message, Timings, ToolCall, Usage

if "ollama" not in sys.modules:
    stub = types.ModuleType("ollama")
    stub.Client = object
    sys.modules["ollama"] = stub

ollama_message_to_dict = importlib.import_module("llm.ollama_provider")._message_to_dict


class _MemoryContextResult:
    def to_dict(self):
        return {
            "blocks": {
                "self_facts": "- identity_name: Pasha",
                "relevant_claims": "- user_name -> Pasha",
                "working_memory": "full memory block that should stay out of compact reasoning snapshot",
            },
            "selected": [
                {
                    "id": "fact:name",
                    "memory_type": "fact",
                    "score": 0.98,
                    "level": "l3_semantic",
                    "scope": "conversation",
                    "status": "active",
                    "metadata": {
                        "fact": {
                            "subject": "user",
                            "predicate": "identity_name",
                            "value": "Pasha",
                        }
                    },
                }
            ],
            "dropped": [],
            "truncation_log": [],
            "fact_expectation": {"expected_predicates": ["identity_name"]},
            "self_facts_context": {"found_predicates": ["identity_name"]},
            "dialog_episode_hits": [],
            "recall_mode": "exact_fact_recall",
        }


class _MemoryManager:
    def __init__(self) -> None:
        self.calls = []

    def build_context(self, request):
        self.calls.append(request)
        return _MemoryContextResult()

    def get_identity_core_snapshot(self, namespace):
        if str(namespace or "") != "conv-agent-loop":
            return None
        return {
            "addressing": {
                "canonical_name": "Pasha",
            },
            "interaction_style": {
                "prefers_short_answers": 0.78,
                "prefers_examples_on_user_code": True,
            },
            "boundaries": {
                "do_not_invent_user_facts": True,
            },
        }


class _LoopProvider:
    def __init__(self) -> None:
        self.requests = []

    def generate(self, req):
        self.requests.append(req)
        if len(self.requests) == 1:
            return LLMResponse(
                text="",
                tool_calls=[
                    ToolCall(
                        id="call_memory_1",
                        name="memory_retrieve",
                        arguments={
                            "mode": "profile",
                            "topic_hints": ["name", "identity"],
                            "time_hint": "persistent",
                            "sources": ["profile", "facts"],
                            "top_k": 3,
                        },
                        raw_arguments='{"mode":"profile","topic_hints":["name","identity"],"time_hint":"persistent","sources":["profile","facts"],"top_k":3}',
                    )
                ],
                model="fake-loop",
                usage=Usage(prompt_tokens=10, completion_tokens=2, total_tokens=12),
                timings=Timings(latency_ms=5.0),
            )

        tool_message = req.messages[-1]
        assistant_message = req.messages[-2]
        assert assistant_message.role == "assistant"
        assert tool_message.role == "tool"
        assert tool_message.tool_call_id == "call_memory_1"
        assert assistant_message.tool_calls[0].name == "memory_retrieve"
        assert "Pasha" in tool_message.content
        assert '"mode":"profile"' in tool_message.content
        reasoning_messages = [
            message
            for message in req.messages
            if message.role == "system" and "[MEMORY_REASONING_CHECK]" in message.content
        ]
        assert len(reasoning_messages) == 1
        reasoning = reasoning_messages[0].content
        assert '"profile_facts"' in reasoning
        assert '"identity_core"' in reasoning
        assert '"active_task"' in reasoning
        assert '"prefers_short_answers"' in reasoning
        assert '"prefers_examples_on_user_code":true' in reasoning
        assert '"do_not_invent_user_facts":true' in reasoning
        assert '"status":"active"' in reasoning
        assert '"blocks"' not in reasoning
        assert '"selected"' not in reasoning
        assert "working_memory" not in reasoning

        return LLMResponse(
            text="I remember your name is Pasha.",
            model="fake-loop",
            usage=Usage(prompt_tokens=18, completion_tokens=6, total_tokens=24),
            timings=Timings(latency_ms=7.0),
        )


class _StreamingLoopProvider:
    def __init__(self) -> None:
        self.stream_requests = []
        self.generate_requests = []

    def generate(self, req):
        self.generate_requests.append(req)
        raise AssertionError("generate() should not be used when agent-loop streaming succeeds")

    def stream(self, req):
        self.stream_requests.append(req)
        if len(self.stream_requests) == 1:
            yield LLMChunk(
                tool_calls_delta=[
                    ToolCall(
                        id="call_memory_stream_1",
                        name="memory_retrieve",
                        arguments={
                            "mode": "profile",
                            "topic_hints": ["name"],
                            "time_hint": "persistent",
                            "sources": ["profile", "facts"],
                            "top_k": 3,
                        },
                    )
                ],
                model="fake-stream-loop",
                done=False,
            )
            yield LLMChunk(
                model="fake-stream-loop",
                usage=Usage(prompt_tokens=9, completion_tokens=1, total_tokens=10),
                timings=Timings(latency_ms=4.0),
                done=True,
            )
            return

        yield LLMChunk(text_delta="I ", model="fake-stream-loop", done=False)
        yield LLMChunk(text_delta="remember ", model="fake-stream-loop", done=False)
        yield LLMChunk(text_delta="your ", model="fake-stream-loop", done=False)
        yield LLMChunk(
            text_delta="name is Pasha.",
            model="fake-stream-loop",
            usage=Usage(prompt_tokens=18, completion_tokens=6, total_tokens=24),
            timings=Timings(latency_ms=7.0),
            done=True,
        )


class _StreamingFallbackProvider:
    def __init__(self) -> None:
        self.stream_requests = []
        self.generate_requests = []

    def stream(self, req):
        self.stream_requests.append(req)
        raise RuntimeError("broken stream")

    def generate(self, req):
        self.generate_requests.append(req)
        return LLMResponse(
            text="Fallback final answer.",
            model="fake-fallback-loop",
            usage=Usage(prompt_tokens=12, completion_tokens=4, total_tokens=16),
            timings=Timings(latency_ms=6.0),
        )


class ResponsePipelineAgentLoopTests(unittest.TestCase):
    def test_autonomous_profile_uses_full_pipeline_without_stage_memory_retrieve(self) -> None:
        pipeline = ResponsePipeline(provider=object())

        stages = pipeline._resolve_stage_names(PROFILE_AUTONOMOUS, {}, {})

        self.assertIn("personality", stages)
        self.assertIn("prompt_build", stages)
        self.assertIn("prompt_engine", stages)
        self.assertIn("memory_write", stages)
        self.assertNotIn("memory_retrieve", stages)

    def test_autonomous_agent_loop_executes_memory_recall_and_hides_tool_step(self) -> None:
        provider = _LoopProvider()
        manager = _MemoryManager()
        pipeline = ResponsePipeline(provider=provider, memory_manager=manager)
        meta = {
            "conversation_id": "conv-agent-loop",
            "turn_id": 1,
            "profile": PROFILE_AUTONOMOUS,
            "memory_manager": manager,
        }

        result = pipeline.run(
            route="chat",
            user_msg="What is my name?",
            state={
                "conversation_id": "conv-agent-loop",
                "turn_id": 1,
                "active_character_id": "asya",
                "active_profile_snapshot": {
                    "prefers_short_answers": True,
                    "prefers_examples_on_user_code": True,
                    "assistant_directness": 0.82,
                    "assistant_warmth": 0.64,
                },
                "active_task": {
                    "task_id": "task:memory-loop",
                    "topic": "memory loop",
                    "status": "active",
                    "current_goal": "Confirm the remembered user identity before answering.",
                    "next_steps": ["Check recalled self-facts against identity core"],
                },
                "identity_core": {
                    "addressing": {
                        "canonical_name": "Pasha",
                    },
                    "boundaries": {
                        "do_not_invent_user_facts": True,
                    },
                },
            },
            meta=meta,
            retrieved_memories=[],
            traits={},
            policies={},
        )

        self.assertEqual(result.text, "I remember your name is Pasha.")
        self.assertEqual(result.tool_calls, [])
        self.assertEqual(len(provider.requests), 2)
        self.assertEqual(len(manager.calls), 1)
        self.assertTrue(any(tool.name == "memory_retrieve" for tool in provider.requests[0].tools))
        self.assertEqual(provider.requests[1].messages[-2].role, "assistant")
        self.assertEqual(provider.requests[1].messages[-1].role, "tool")
        self.assertTrue(bool(result.stats.get("agent_loop")))
        self.assertEqual(int(result.stats.get("agent_tool_calls") or 0), 1)
        self.assertEqual(
            str(dict(meta.get("debug_trace") or {}).get("memory_retrieval", {}).get("stage") or ""),
            "agent_memory_retrieve",
        )
        self.assertTrue(
            bool(dict(dict(meta.get("debug_trace") or {}).get("final_answer_meta") or {}).get("memory_reasoning_used"))
        )
        self.assertEqual(
            int(dict(dict(meta.get("debug_trace") or {}).get("final_answer_meta") or {}).get("agent_tool_calls") or 0),
            1,
        )
        self.assertEqual(
            int(dict(dict(meta.get("memory_debug_snapshot") or {}).get("tool_loop") or {}).get("agent_passes") or 0),
            2,
        )

    def test_ollama_message_to_dict_preserves_assistant_tool_calls(self) -> None:
        message = Message(
            role="assistant",
            content="",
            tool_calls=[
                ToolCall(
                    id="call_memory_1",
                    name="memory_retrieve",
                    arguments={"query": "What is my name?"},
                )
            ],
        )

        payload = ollama_message_to_dict(message)

        self.assertEqual(payload["role"], "assistant")
        self.assertEqual(payload["tool_calls"][0]["function"]["name"], "memory_retrieve")
        self.assertEqual(payload["tool_calls"][0]["function"]["arguments"]["query"], "What is my name?")

    def test_agent_loop_streaming_keeps_tool_step_hidden_and_streams_final_pass(self) -> None:
        provider = _StreamingLoopProvider()
        manager = _MemoryManager()
        pipeline = ResponsePipeline(provider=provider, memory_manager=manager)
        streamed: list[str] = []
        meta = {
            "conversation_id": "conv-agent-loop-stream",
            "turn_id": 2,
            "profile": PROFILE_AUTONOMOUS,
            "memory_manager": manager,
            "stream_on_answer_chunk": lambda piece: streamed.append(piece),
        }

        result = pipeline.run(
            route="chat",
            user_msg="What is my name?",
            state={
                "conversation_id": "conv-agent-loop-stream",
                "turn_id": 2,
                "active_character_id": "asya",
            },
            meta=meta,
            retrieved_memories=[],
            traits={},
            policies={},
        )

        self.assertEqual(result.text, "I remember your name is Pasha.")
        self.assertEqual("".join(streamed), "I remember your name is Pasha.")
        self.assertEqual(len(provider.stream_requests), 2)
        self.assertEqual(len(provider.generate_requests), 0)
        self.assertEqual(len(manager.calls), 1)
        self.assertEqual(result.tool_calls, [])
        self.assertEqual(int(result.stats.get("agent_tool_calls") or 0), 1)

    def test_agent_loop_streaming_error_logged_not_fallback(self) -> None:
        """
        Проверить, что streaming ошибка логируется с деталями, а не маскируется fallback.
        """
        provider = _StreamingFallbackProvider()
        pipeline = ResponsePipeline(provider=provider)

        result = pipeline.run(
            route="chat",
            user_msg="Say hello",
            state={
                "conversation_id": "conv-agent-loop-stream-error",
                "turn_id": 3,
                "active_character_id": "asya",
            },
            meta={
                "conversation_id": "conv-agent-loop-stream-error",
                "turn_id": 3,
                "profile": PROFILE_AUTONOMOUS,
                "stream_on_answer_chunk": lambda _piece: None,
            },
            retrieved_memories=[],
            traits={},
            policies={},
        )

        # Ошибка должна быть залогирована с полным текстом
        self.assertTrue(any("agent_loop_stream_error=RuntimeError:" in row for row in list(result.logs or [])))
        # Не должно быть старого fallback лога
        self.assertFalse(any("agent_loop_stream_fallback=generate" in row for row in list(result.logs or [])))


if __name__ == "__main__":
    unittest.main()
