from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from core.brain import Brain
from core.state_manager import StateManager
from llm.provider_base import (
    LLMProviderBase,
    LLMRequest,
    LLMResponse,
    ModelInfo,
    ProviderHealth,
    Timings,
    Usage,
)


class FakeProvider(LLMProviderBase):
    def generate(self, req: LLMRequest) -> LLMResponse:
        text = ""
        if req.messages:
            text = req.messages[-1].content
        return LLMResponse(
            text=f"ok: {text[:40]}",
            usage=Usage(prompt_tokens=7, completion_tokens=6, total_tokens=13),
            timings=Timings(latency_ms=5.0),
            model=req.model or "fake",
            raw={"from": "fake"},
        )

    def healthcheck(self) -> ProviderHealth:
        return ProviderHealth(ok=True, provider="fake", detail="ok", model="fake")

    def model_info(self, model: str = "") -> ModelInfo:
        return ModelInfo(provider="fake", model=model or "fake")

    def list_models(self) -> list[str]:
        return ["fake"]


class BrainSmokeTests(unittest.TestCase):
    def test_brain_handles_chat_and_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            state = StateManager(state_path=Path(tmpdir) / "state.json", autosave=False)
            brain = Brain(provider=FakeProvider(), state_manager=state, throttle_sec=0.0)

            r1 = brain.handle_message("Hello there", meta={"source": "ui"})
            r2 = brain.handle_message("/nothink", meta={"source": "ui"})
            r3 = brain.handle_message("How are you?", meta={"source": "ui"})

            self.assertTrue(r1.text)
            self.assertTrue(r2.text)
            self.assertTrue(r3.text)
            self.assertIn(r2.route, {"command", "chat"})


if __name__ == "__main__":
    unittest.main()
