from __future__ import annotations

from _output_utils import enable_unittest_json_output
enable_unittest_json_output()

import tempfile
import unittest
from pathlib import Path

from core.character_runtime import CharacterRuntime
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


class DebugCommandsTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="mmis_debug_cmds_")
        self.runtime = CharacterRuntime(
            character_path=Path(self._tmp.name) / "characters",
            state_path=Path(self._tmp.name) / "brain_state.json",
            state_store_dir=Path(self._tmp.name) / "brain_state_store",
            autosave=False,
        )
        self.pipeline = ResponsePipeline(provider=_StubProvider(), character_runtime=self.runtime)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _run_command(self, cmd: str, state: dict):
        return self.pipeline.run(
            route="command",
            user_msg=cmd,
            state=state,
            meta={"source": "test", "store_turn": False},
            retrieved_memories=[],
            traits={},
            policies={},
        )

    def test_persona_debug_command_returns_expected_fields(self) -> None:
        state = {
            "history": [],
            "quality_profile": "BALANCED",
            "active_character_id": "asya",
            "active_mode": "debugger",
            "mode_lock": False,
            "characters": {
                "asya": {
                    "persona": {
                        "mood": "serious",
                        "traits": {
                            "warmth": 0.58,
                            "sarcasm": 0.18,
                            "teasing": 0.12,
                            "strictness": 0.62,
                            "verbosity": 0.55,
                        },
                        "locks": {"feminine": True, "informal_you": True},
                        "bans": ["artifact"],
                    },
                    "local_context": {},
                }
            },
            "last_actions": [
                {
                    "type": "MODE_CHANGED",
                    "state_diff": {"active_mode": {"old": "chatting", "new": "debugger"}},
                }
            ],
        }
        result = self._run_command("/persona_debug", state=state)
        self.assertIn("persona_debug:", result.text)
        self.assertIn("character=asya", result.text)
        self.assertIn("mode=debugger (auto)", result.text)
        self.assertIn("mood=serious", result.text)
        self.assertIn("traits:", result.text)
        self.assertIn("bans: artifact", result.text)

    def test_brain_debug_command_returns_signals_and_actions(self) -> None:
        state = {
            "history": [],
            "quality_profile": "QUALITY",
            "active_mode": "engineer",
            "mode_lock": True,
            "web_mode": "on",
            "thinking_enabled": True,
            "active_goal": "Fix traceback",
            "active_tasks": [{"id": "t1", "title": "Reproduce bug"}],
            "last_signals": {"intent": "bug_report", "emotion": "frustrated", "topic": "python"},
            "last_actions": [
                {
                    "type": "MODE_CHANGED",
                    "ts": "2026-03-03T10:00:00+02:00",
                    "state_diff": {"active_mode": {"old": "chatting", "new": "engineer"}},
                },
                {
                    "type": "FEEDBACK_RECEIVED",
                    "ts": "2026-03-03T10:00:01+02:00",
                    "feedback": ["no_teasing"],
                },
            ],
        }
        result = self._run_command("/brain_debug", state=state)
        self.assertIn("brain_debug:", result.text)
        self.assertNotIn("[PARAMETERS]", result.text)
        self.assertIn("active_mode=engineer (locked)", result.text)
        self.assertIn("web_mode=on", result.text)
        self.assertIn("thinking_enabled=true", result.text)
        self.assertIn("active_goal=Fix traceback", result.text)
        self.assertIn("last_signals:", result.text)
        self.assertIn("feedback: no_teasing", result.text)
        self.assertIn("FEEDBACK_RECEIVED@", result.text)
        self.assertIn("(no_teasing)", result.text)
        self.assertIn("last_actions:", result.text)

    def test_brain_debug_with_extra_tail_still_routes_to_debug_command(self) -> None:
        state = {
            "history": [],
            "quality_profile": "BALANCED",
            "active_mode": "chatting",
            "mode_lock": False,
        }
        result = self._run_command("/brain_debug /think", state=state)
        self.assertIn("brain_debug:", result.text)
        self.assertNotIn("Unknown command", result.text)


if __name__ == "__main__":
    unittest.main()
