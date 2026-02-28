from __future__ import annotations

import unittest

from core.response_pipeline import ResponsePipeline, _enforce_response_hygiene
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
            usage=Usage(prompt_tokens=8, completion_tokens=8, total_tokens=16),
            timings=Timings(latency_ms=5.0),
            model="stub",
        )

    def healthcheck(self) -> ProviderHealth:
        return ProviderHealth(ok=True, provider="stub", detail="ok", model="stub")

    def model_info(self, model: str = "") -> ModelInfo:
        return ModelInfo(provider="stub", model=model or "stub")

    def list_models(self) -> list[str]:
        return ["stub"]


class ResponseHygieneTests(unittest.TestCase):
    def test_hygiene_removes_service_lines_and_prefixes(self) -> None:
        src = (
            "thinking> internal note\n"
            "assistant> Привет\n"
            "[model: qwen3:8b]\n"
            "user> ignored\n"
            "[thinking]\n"
            "assistant> Как могу помочь?"
        )
        self.assertEqual(_enforce_response_hygiene(src), "Привет\nКак могу помочь?")

    def test_hygiene_dedupes_adjacent_repeats(self) -> None:
        src = "assistant> Простите, если что-то обидело.\nassistant> Простите, если что-то обидело."
        self.assertEqual(_enforce_response_hygiene(src), "Простите, если что-то обидело.")

    def test_pipeline_applies_hygiene_in_postprocess(self) -> None:
        provider = _StubProvider(
            "thinking> plan\nassistant> Простите, если что-то обидело.\nassistant> Простите, если что-то обидело."
        )
        pipeline = ResponsePipeline(provider=provider)

        result = pipeline.run(
            route="chat",
            user_msg="ok",
            state={"mode": "chat", "history": [], "quality_profile": "BALANCED"},
            meta={"source": "test", "store_turn": False},
            retrieved_memories=[],
            traits={},
            policies={},
        )

        self.assertEqual(result.text, "Простите, если что-то обидело.")


if __name__ == "__main__":
    unittest.main()
