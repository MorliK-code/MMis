from __future__ import annotations

from _output_utils import enable_unittest_json_output
enable_unittest_json_output()

import tempfile
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
from modules.character import CharacterEngine, CharacterStorage


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


class PersonalitySwitchingTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="mmis_style_single_source_")
        storage = CharacterStorage(root=self._tmp.name)
        self.characters = CharacterEngine(storage=storage)
        self.pipeline = ResponsePipeline(provider=_StubProvider(), character_engine=self.characters)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _run_command(self, cmd: str, state: dict | None = None):
        return self.pipeline.run(
            route="command",
            user_msg=cmd,
            state=state or {"mode": "chat", "history": [], "quality_profile": "BALANCED"},
            meta={"source": "test", "store_turn": False},
            retrieved_memories=[],
            traits={},
            policies={},
        )

    def test_persona_show_is_character_alias(self) -> None:
        result = self._run_command("/persona", state={"active_character_id": "default", "history": [], "mode": "chat"})
        self.assertIn("Character style source: default", result.text)

    def test_persona_switch_targets_character(self) -> None:
        result = self._run_command("/persona asya", state={"active_character_id": "default", "history": [], "mode": "chat"})
        self.assertIn("Character switched to: asya", result.text)
        char_ops = [x for x in result.memory_ops if str(x.get("op")) == "state_character"]
        self.assertTrue(char_ops)
        self.assertEqual(str(char_ops[-1].get("value")), "asya")

    def test_characters_list_contains_all_character_ids(self) -> None:
        result = self._run_command("/characters", state={"active_character_id": "default", "history": [], "mode": "chat"})
        self.assertIn("default", result.text)
        self.assertIn("asya", result.text)


if __name__ == "__main__":
    unittest.main()

