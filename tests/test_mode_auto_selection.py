from __future__ import annotations

from _output_utils import enable_unittest_json_output

enable_unittest_json_output()

import unittest

from core.response_pipeline import ResponsePipeline
from llm.provider_base import (
    LLMProviderBase,
    LLMRequest,
    LLMResponse,
    ModelInfo,
    ProviderHealth,
    Timings,
    Usage,
)


class _StubProvider(LLMProviderBase):
    def generate(self, req: LLMRequest) -> LLMResponse:
        _ = req
        return LLMResponse(
            text="ok",
            usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            timings=Timings(latency_ms=1.0),
            model="stub",
        )

    def healthcheck(self) -> ProviderHealth:
        return ProviderHealth(ok=True, provider="stub", detail="ok", model="stub")

    def model_info(self, model: str = "") -> ModelInfo:
        return ModelInfo(provider="stub", model=model or "stub")

    def list_models(self) -> list[str]:
        return ["stub"]


class ModeAutoSelectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.pipeline = ResponsePipeline(provider=_StubProvider())

    def _run_chat(self, user_msg: str, *, mode_lock: bool = False):
        return self.pipeline.run(
            route="chat",
            user_msg=user_msg,
            state={
                "history": [],
                "quality_profile": "BALANCED",
                "mode": "chat",
                "active_mode": "friend_chat",
                "mode_lock": mode_lock,
            },
            meta={"source": "test", "store_turn": False},
            retrieved_memories=[],
            traits={},
            policies={},
        )

    def test_traceback_switches_to_debugger(self) -> None:
        result = self._run_chat("Traceback (most recent call last): ValueError")
        ops = [x for x in result.memory_ops if str(x.get("op")) == "state_mode"]
        self.assertTrue(ops)
        self.assertEqual(str(ops[-1].get("value")), "debugger")

    def test_mode_lock_blocks_auto_switch(self) -> None:
        result = self._run_chat("Traceback (most recent call last): RuntimeError", mode_lock=True)
        ops = [x for x in result.memory_ops if str(x.get("op")) == "state_mode"]
        self.assertFalse(ops)


if __name__ == "__main__":
    unittest.main()
