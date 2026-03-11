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
from modules.character.storage import CharacterStorage


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


class CharacterCommandsTests(unittest.TestCase):
    def _new_pipeline(self) -> tuple[ResponsePipeline, CharacterRuntime]:
        tmp = tempfile.TemporaryDirectory(prefix="mmis_char_cmd_")
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)

        storage = CharacterStorage(root=root / "characters_runtime", logs_root=root / "logs")
        storage.spec_root = (root / "specs" / "characters").resolve()
        storage.spec_root.mkdir(parents=True, exist_ok=True)

        runtime = CharacterRuntime(
            storage=storage,
            state_path=root / "brain_state.json",
            state_store_dir=root / "brain_state_store",
            autosave=False,
        )
        pipeline = ResponsePipeline(provider=_StubProvider(), character_runtime=runtime)
        return pipeline, runtime

    @staticmethod
    def _run_command(pipeline: ResponsePipeline, command: str, *, state: dict | None = None):
        row = {"history": [], "quality_profile": "BALANCED", "active_mode": "chatting"}
        if isinstance(state, dict):
            row.update(state)
        return pipeline.run(
            route="command",
            user_msg=command,
            state=row,
            meta={"source": "test", "store_turn": False},
            retrieved_memories=[],
            traits={},
            policies={},
        )

    def test_character_delete_removes_character(self) -> None:
        pipeline, runtime = self._new_pipeline()
        runtime.storage.ensure_character_structure("tempdel")
        runtime.storage.sync_manifest()
        self.assertIn("tempdel", runtime.list_ids())

        result = self._run_command(pipeline, "/character delete tempdel")
        self.assertIn("Character deleted: tempdel", str(result.text or ""))
        self.assertNotIn("tempdel", runtime.list_ids())

    def test_character_delete_requires_target(self) -> None:
        pipeline, _runtime = self._new_pipeline()
        result = self._run_command(pipeline, "/character delete")
        self.assertIn("Usage: /character delete <character_id>", str(result.text or ""))


if __name__ == "__main__":
    unittest.main()
