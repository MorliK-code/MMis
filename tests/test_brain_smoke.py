from __future__ import annotations

from _output_utils import enable_unittest_json_output
enable_unittest_json_output()

import tempfile
import unittest
from pathlib import Path

from core.brain import Brain
from core.character_runtime import CharacterRuntime
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
            state = CharacterRuntime(state_path=Path(tmpdir) / "state.json", autosave=False)
            brain = Brain(provider=FakeProvider(), state_manager=state, throttle_sec=0.0)

            r1 = brain.handle_message("Hello there", meta={"source": "ui"})
            r2 = brain.handle_message("/nothink", meta={"source": "ui"})
            r3 = brain.handle_message("How are you?", meta={"source": "ui"})

            self.assertTrue(r1.text)
            self.assertTrue(r2.text)
            self.assertTrue(r3.text)
            self.assertIn(r2.route, {"command", "chat"})

    def test_commands_are_not_throttled(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            state = CharacterRuntime(state_path=Path(tmpdir) / "state.json", autosave=False)
            brain = Brain(provider=FakeProvider(), state_manager=state, throttle_sec=60.0)

            r1 = brain.handle_message("/mode", meta={"source": "ui"})
            r2 = brain.handle_message("/mode_lock on", meta={"source": "ui"})

            self.assertEqual(r1.route, "command")
            self.assertEqual(r2.route, "command")
            self.assertFalse(bool(r1.throttled))
            self.assertFalse(bool(r2.throttled))
            self.assertTrue(str(r1.text or "").strip())
            self.assertTrue(str(r2.text or "").strip())

    def test_feedback_action_is_recorded_in_shared_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            state = CharacterRuntime(state_path=Path(tmpdir) / "state.json", autosave=False)
            brain = Brain(provider=FakeProvider(), state_manager=state, throttle_sec=0.0)

            brain.handle_message('Do not say phrase "could you clarify what you mean"', meta={"source": "ui"})
            actions = [dict(x) for x in list(state.snapshot().raw.get("last_actions") or []) if isinstance(x, dict)]

            self.assertTrue(any(str(x.get("type") or "").upper() == "FEEDBACK_RECEIVED" for x in actions))

    def test_command_with_bom_prefix_is_routed_as_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            state = CharacterRuntime(state_path=Path(tmpdir) / "state.json", autosave=False)
            brain = Brain(provider=FakeProvider(), state_manager=state, throttle_sec=0.0)

            result = brain.handle_message("\ufeff/brain_debug", meta={"source": "ui"})

            self.assertEqual(result.route, "command")
            self.assertIn("brain_debug:", str(result.text or ""))


if __name__ == "__main__":
    unittest.main()

