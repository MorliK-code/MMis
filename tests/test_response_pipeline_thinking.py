from __future__ import annotations

try:
    from _output_utils import enable_unittest_json_output
except ModuleNotFoundError:
    from tests._output_utils import enable_unittest_json_output
enable_unittest_json_output()

import sys
import unittest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

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


class _ThinkingProvider(LLMProviderBase):
    def generate(self, req: LLMRequest) -> LLMResponse:
        _ = req
        return LLMResponse(
            text="visible answer",
            thinking="internal reasoning",
            usage=Usage(prompt_tokens=8, completion_tokens=4, total_tokens=12),
            timings=Timings(latency_ms=5.0),
            model="stub-thinking",
        )

    def healthcheck(self) -> ProviderHealth:
        return ProviderHealth(ok=True, provider="stub", detail="ok", model="stub-thinking")

    def model_info(self, model: str = "") -> ModelInfo:
        return ModelInfo(provider="stub", model=model or "stub-thinking")

    def list_models(self) -> list[str]:
        return ["stub-thinking"]


class ResponsePipelineThinkingTests(unittest.TestCase):
    def test_pipeline_result_contains_thinking_and_memory_op(self) -> None:
        pipeline = ResponsePipeline(provider=_ThinkingProvider())
        result = pipeline.run(
            route="chat",
            user_msg="hello",
            state={"mode": "chat", "history": [], "quality_profile": "BALANCED"},
            meta={"source": "test", "store_turn": True},
            retrieved_memories=[],
            traits={},
            policies={},
        )

        self.assertEqual(result.thinking, "internal reasoning")
        assistant_ops = [x for x in result.memory_ops if str(x.get("op")) == "turn_assistant"]
        self.assertTrue(assistant_ops)
        self.assertEqual(str(assistant_ops[-1].get("thinking") or ""), "internal reasoning")


if __name__ == "__main__":
    unittest.main()
