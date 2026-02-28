from __future__ import annotations

import unittest

from llm.provider_base import (
    LLMProviderBase,
    LLMRequest,
    LLMResponse,
    Message,
    ModelInfo,
    ProviderHealth,
    Timings,
    Usage,
)


class FakeProvider(LLMProviderBase):
    def generate(self, req: LLMRequest) -> LLMResponse:
        last = req.messages[-1].content if req.messages else ""
        return LLMResponse(
            text=f"echo:{last}",
            usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            timings=Timings(latency_ms=12.5),
            model=req.model or "fake",
            raw={"ok": True},
        )

    def healthcheck(self) -> ProviderHealth:
        return ProviderHealth(ok=True, provider="fake", detail="ok", model="fake")

    def model_info(self, model: str = "") -> ModelInfo:
        return ModelInfo(provider="fake", model=model or "fake", capabilities={"tools": True})

    def list_models(self) -> list[str]:
        return ["fake"]


class ProviderContractTests(unittest.TestCase):
    def test_generate_returns_llmresponse(self) -> None:
        provider = FakeProvider()
        req = LLMRequest(messages=[Message(role="user", content="hello")], model="fake")
        res = provider.generate(req)

        self.assertIsInstance(res, LLMResponse)
        self.assertTrue(res.text.startswith("echo:"))
        self.assertIsInstance(res.usage.total_tokens, int)
        self.assertGreaterEqual(res.timings.latency_ms, 0.0)


if __name__ == "__main__":
    unittest.main()
