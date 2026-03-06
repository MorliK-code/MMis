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
    def __init__(self) -> None:
        self.calls = 0

    def generate(self, req: LLMRequest) -> LLMResponse:
        _ = req
        self.calls += 1
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


class ModeCommandsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.provider = _StubProvider()
        self.pipeline = ResponsePipeline(provider=self.provider)

    def _run_command(self, cmd: str):
        return self.pipeline.run(
            route="command",
            user_msg=cmd,
            state={"history": [], "quality_profile": "BALANCED", "active_mode": "chatting"},
            meta={"source": "test", "store_turn": False},
            retrieved_memories=[],
            traits={},
            policies={},
        )

    def test_mode_command_sets_state_mode(self) -> None:
        result = self._run_command("/mode debugger")
        self.assertIn("Mode switched to: debugger", result.text)
        self.assertNotIn("[PARAMETERS]", result.text)
        ops = [x for x in result.memory_ops if str(x.get("op")) == "state_mode"]
        self.assertTrue(ops)
        self.assertEqual(str(ops[-1].get("value")), "debugger")
        lock_ops = [x for x in result.memory_ops if str(x.get("op")) == "state_mode_lock"]
        self.assertTrue(lock_ops)
        self.assertTrue(bool(lock_ops[-1].get("value")))

    def test_mode_auto_command_disables_lock(self) -> None:
        result = self._run_command("/mode auto")
        self.assertIn("Mode auto enabled", result.text)
        lock_ops = [x for x in result.memory_ops if str(x.get("op")) == "state_mode_lock"]
        self.assertTrue(lock_ops)
        self.assertFalse(bool(lock_ops[-1].get("value")))

    def test_custom_mode_is_accepted_and_locked(self) -> None:
        result = self._run_command("/mode my_custom_mode")
        self.assertIn("Mode switched to: my_custom_mode", result.text)
        mode_ops = [x for x in result.memory_ops if str(x.get("op")) == "state_mode"]
        self.assertTrue(mode_ops)
        self.assertEqual(str(mode_ops[-1].get("value")), "my_custom_mode")
        lock_ops = [x for x in result.memory_ops if str(x.get("op")) == "state_mode_lock"]
        self.assertTrue(lock_ops)
        self.assertTrue(bool(lock_ops[-1].get("value")))

    def test_mode_lock_command_sets_flag(self) -> None:
        result = self._run_command("/mode_lock on")
        self.assertIn("Mode lock enabled", result.text)
        ops = [x for x in result.memory_ops if str(x.get("op")) == "state_mode_lock"]
        self.assertTrue(ops)
        self.assertTrue(bool(ops[-1].get("value")))

    def test_unknown_command_does_not_call_llm(self) -> None:
        result = self._run_command("/not_a_command")
        self.assertIn("Unknown command. Use /help", result.text)
        self.assertEqual(int(self.provider.calls), 0)


if __name__ == "__main__":
    unittest.main()
