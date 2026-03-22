from __future__ import annotations

import unittest

from core.brain import Brain
from core.character_runtime import CharacterRuntime
from core.response_pipeline import GenerateStage, PipelineContext, PipelineResult
from llm.provider_base import LLMChunk, LLMResponse, Message, Timings, Usage


class _VerboseGenerateProvider:
    def __init__(self) -> None:
        self.requests = []

    def generate(self, req):
        self.requests.append(req)
        return LLMResponse(
            text="done",
            model="verbose-model",
            usage=Usage(prompt_tokens=11, completion_tokens=7, total_tokens=18),
            timings=Timings(
                latency_ms=123.0,
                total_duration_ms=123.0,
                load_duration_ms=5.0,
                prompt_eval_duration_ms=40.0,
                eval_duration_ms=70.0,
            ),
        )


class _VerboseStreamProvider:
    def __init__(self) -> None:
        self.requests = []

    def stream(self, req):
        self.requests.append(req)
        yield LLMChunk(text_delta="Hello ", model="stream-model")
        yield LLMChunk(
            text_delta="world",
            done=True,
            model="stream-model",
            usage=Usage(prompt_tokens=9, completion_tokens=5, total_tokens=14),
            timings=Timings(
                latency_ms=88.0,
                total_duration_ms=88.0,
                load_duration_ms=6.0,
                prompt_eval_duration_ms=20.0,
                eval_duration_ms=40.0,
            ),
        )


class _BrainPipeline:
    def __init__(self) -> None:
        self.calls = []

    def run(self, **kwargs):
        self.calls.append(kwargs)
        return PipelineResult(
            text="ok",
            stats={
                "served_model": "brain-model",
                "answer_ms": 55.0,
                "prompt_eval_count": 10,
                "eval_count": 4,
                "total_tokens": 14,
                "verbose_enabled": True,
                "eval_duration_ms": 20.0,
            },
        )


class _StatsRecorder:
    def __init__(self) -> None:
        self.calls = []

    def record_response_stats(self, **kwargs) -> None:
        self.calls.append(dict(kwargs))


class VerboseStatsPipelineTests(unittest.TestCase):
    def test_generate_stage_passes_verbose_and_collects_extended_timings(self) -> None:
        provider = _VerboseGenerateProvider()
        stage = GenerateStage(provider=provider, character_runtime=CharacterRuntime())
        ctx = PipelineContext(
            route="chat",
            user_msg="hi",
            clean_user_msg="hi",
            state={"conversation_id": "conv-verbose"},
            meta={"conversation_id": "conv-verbose", "verbose": True},
            retrieved_memories=[],
            traits={},
            policies={},
            profile="BALANCED",
            prompt_messages=[Message(role="user", content="hi")],
        )

        out = stage.run(ctx)

        self.assertEqual(len(provider.requests), 1)
        self.assertTrue(bool(provider.requests[0].metadata.get("verbose")))
        self.assertEqual(out.text, "done")
        self.assertEqual(float(out.stats.get("answer_ms") or 0.0), 123.0)
        self.assertEqual(float(out.stats.get("total_duration_ms") or 0.0), 123.0)
        self.assertEqual(float(out.stats.get("load_duration_ms") or 0.0), 5.0)
        self.assertEqual(float(out.stats.get("prompt_eval_duration_ms") or 0.0), 40.0)
        self.assertEqual(float(out.stats.get("eval_duration_ms") or 0.0), 70.0)
        self.assertEqual(float(out.stats.get("eval_tokens_per_sec") or 0.0), 100.0)
        self.assertTrue(bool(out.stats.get("verbose_enabled")))

    def test_generate_stage_stream_collects_final_verbose_stats(self) -> None:
        provider = _VerboseStreamProvider()
        stage = GenerateStage(provider=provider, character_runtime=CharacterRuntime())
        seen_chunks: list[str] = []
        ctx = PipelineContext(
            route="chat",
            user_msg="stream me",
            clean_user_msg="stream me",
            state={"conversation_id": "conv-stream-verbose"},
            meta={
                "conversation_id": "conv-stream-verbose",
                "verbose": True,
                "agent_loop": False,
                "stream_on_answer_chunk": seen_chunks.append,
            },
            retrieved_memories=[],
            traits={},
            policies={},
            profile="BALANCED",
            prompt_messages=[Message(role="user", content="stream me")],
        )

        out = stage.run(ctx)

        self.assertEqual(len(provider.requests), 1)
        self.assertTrue(bool(provider.requests[0].metadata.get("verbose")))
        self.assertEqual(out.text, "Hello world")
        self.assertEqual("".join(seen_chunks), "Hello world")
        self.assertTrue(bool(out.stats.get("streaming")))
        self.assertEqual(int(out.stats.get("prompt_eval_count") or 0), 9)
        self.assertEqual(int(out.stats.get("eval_count") or 0), 5)
        self.assertEqual(int(out.stats.get("total_tokens") or 0), 14)
        self.assertEqual(float(out.stats.get("answer_ms") or 0.0), 88.0)
        self.assertEqual(float(out.stats.get("eval_duration_ms") or 0.0), 40.0)
        self.assertEqual(float(out.stats.get("eval_tokens_per_sec") or 0.0), 125.0)
        self.assertTrue(bool(out.stats.get("verbose_enabled")))

    def test_brain_harvests_response_stats_to_event_store(self) -> None:
        pipeline = _BrainPipeline()
        manager = _StatsRecorder()
        brain = Brain(provider=object(), memory_manager=manager, response_pipeline=pipeline)
        result = brain.handle_message(
            "collect stats",
            meta={
                "conversation_id": "conv-brain-stats",
                "verbose": True,
                "store_turn": False,
            },
        )

        self.assertEqual(result.text, "ok")
        self.assertEqual(len(manager.calls), 1)
        payload = dict(manager.calls[0] or {})
        self.assertEqual(str(payload.get("namespace") or ""), "conv-brain-stats")
        self.assertEqual(str(payload.get("route") or ""), "chat")
        self.assertTrue(bool(dict(payload.get("meta") or {}).get("verbose")))
        self.assertEqual(float(dict(payload.get("stats") or {}).get("answer_ms") or 0.0), 55.0)
        self.assertEqual(str(payload.get("model") or ""), "brain-model")


if __name__ == "__main__":
    unittest.main()
