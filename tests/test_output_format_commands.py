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


class OutputFormatCommandsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.pipeline = ResponsePipeline(provider=_StubProvider())
        self.state = {
            "history": [],
            "quality_profile": "BALANCED",
            "active_mode": "debugger",
            "mode_lock": True,
            "output_format": {"show_parameters": None, "show_summary": None},
        }

    def _run_command(self, cmd: str):
        return self.pipeline.run(
            route="command",
            user_msg=cmd,
            state=dict(self.state),
            meta={"source": "test", "store_turn": False},
            retrieved_memories=[],
            traits={},
            policies={},
        )

    def test_output_parameters_command_emits_memory_op(self) -> None:
        result = self._run_command("/output parameters off")
        self.assertIn("Output parameters disabled", result.text)
        ops = [x for x in result.memory_ops if str(x.get("op")) == "state_output_format"]
        self.assertTrue(ops)
        self.assertFalse(bool(dict(ops[-1].get("value") or {}).get("show_parameters")))

    def test_output_summary_command_emits_memory_op(self) -> None:
        result = self._run_command("/output summary on")
        self.assertIn("Output summary enabled", result.text)
        ops = [x for x in result.memory_ops if str(x.get("op")) == "state_output_format"]
        self.assertTrue(ops)
        self.assertTrue(bool(dict(ops[-1].get("value") or {}).get("show_summary")))

    def test_output_status_returns_effective_values(self) -> None:
        result = self._run_command("/output status")
        self.assertIn("Output format:", result.text)
        self.assertIn("mode: debugger", result.text)
        self.assertIn("parameters: on", result.text)
        self.assertIn("summary: on", result.text)

    def test_output_usage_on_invalid_command(self) -> None:
        result = self._run_command("/output something maybe")
        self.assertIn("Usage: /output parameters on|off", result.text)


if __name__ == "__main__":
    unittest.main()
