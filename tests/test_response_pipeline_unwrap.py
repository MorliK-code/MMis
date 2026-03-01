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


class _StubProvider(LLMProviderBase):
    def __init__(self, text: str):
        self._text = str(text)

    def generate(self, req: LLMRequest) -> LLMResponse:
        _ = req
        return LLMResponse(
            text=self._text,
            usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            timings=Timings(latency_ms=10.0),
            model="stub",
        )

    def healthcheck(self) -> ProviderHealth:
        return ProviderHealth(ok=True, provider="stub", detail="ok", model="stub")

    def model_info(self, model: str = "") -> ModelInfo:
        return ModelInfo(provider="stub", model=model or "stub")

    def list_models(self) -> list[str]:
        return ["stub"]


class ResponsePipelineUnwrapTests(unittest.TestCase):
    def test_unwraps_safety_output_json(self) -> None:
        provider = _StubProvider('{"safe":true,"reason":"Friendly and safe response","output":"Здравствуйте! Как дела? 😊"}')
        pipeline = ResponsePipeline(provider=provider)

        result = pipeline.run(
            route="chat",
            user_msg="привет",
            state={"mode": "chat", "history": [], "quality_profile": "BALANCED"},
            meta={"source": "test", "store_turn": False},
            retrieved_memories=[],
            traits={},
            policies={},
        )

        self.assertEqual(result.text, "Здравствуйте! Как дела? 😊")

    def test_keeps_json_when_json_mode_enabled(self) -> None:
        src = '{"safe":true,"reason":"ok","output":"Привет"}'
        provider = _StubProvider(src)
        pipeline = ResponsePipeline(provider=provider)

        result = pipeline.run(
            route="chat",
            user_msg="привет",
            state={"mode": "chat", "history": [], "quality_profile": "BALANCED"},
            meta={"source": "test", "json_mode": True, "store_turn": False},
            retrieved_memories=[],
            traits={},
            policies={},
        )

        self.assertEqual(result.text, src)

    def test_unwraps_json_even_with_thinking_block(self) -> None:
        src = '{"safe":true,"reason":"","output":"exit"}\n<think>internal note</think>'
        provider = _StubProvider(src)
        pipeline = ResponsePipeline(provider=provider)

        result = pipeline.run(
            route="chat",
            user_msg="exit",
            state={"mode": "chat", "history": [], "quality_profile": "BALANCED"},
            meta={"source": "test", "json_mode": False, "store_turn": False},
            retrieved_memories=[],
            traits={},
            policies={},
        )

        self.assertEqual(result.text, "exit\n<think>internal note</think>")

    def test_unwraps_in_fast_profile_too(self) -> None:
        provider = _StubProvider('{"safe":true,"reason":"","output":"Привет!"}')
        pipeline = ResponsePipeline(provider=provider)

        result = pipeline.run(
            route="chat",
            user_msg="привет",
            state={"mode": "chat", "history": [], "quality_profile": "FAST"},
            meta={"source": "test", "json_mode": False, "store_turn": False},
            retrieved_memories=[],
            traits={},
            policies={},
        )

        self.assertTrue(result.text.startswith("Привет"))

    def test_unwrap_and_terms_policy_are_compatible(self) -> None:
        provider = _StubProvider('{"safe":true,"reason":"","output":"Милашка, вот ответ."}')
        pipeline = ResponsePipeline(provider=provider)

        result = pipeline.run(
            route="chat",
            user_msg="разбор",
            state={"mode": "chat", "history": [], "quality_profile": "BALANCED"},
            meta={
                "source": "test",
                "json_mode": False,
                "store_turn": False,
                "address_terms_policy": {
                    "terms_list": ["милашка"],
                    "banned_terms_effective": ["милашка"],
                    "banned_terms_active": True,
                    "use_term_now": False,
                },
            },
            retrieved_memories=[],
            traits={},
            policies={},
        )

        self.assertNotIn("милашка", result.text.lower())


if __name__ == "__main__":
    unittest.main()


