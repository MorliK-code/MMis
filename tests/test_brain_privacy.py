from __future__ import annotations

try:
    from _output_utils import enable_unittest_json_output
except ModuleNotFoundError:
    from tests._output_utils import enable_unittest_json_output

enable_unittest_json_output()

import json
import tempfile
import unittest
from pathlib import Path

from core.brain import Brain
from core.character_runtime import CharacterRuntime
from core.response_pipeline import ResponsePipeline
from llm.provider_base import LLMProviderBase, LLMRequest, LLMResponse, ModelInfo, ProviderHealth, Timings, Usage
from memory.memory_manager import MemoryManager
from memory.memory_models import DebugRequest
from modules.studio.studio_generator import StudioGenerator


class _StudioAwareProvider(LLMProviderBase):
    def __init__(self) -> None:
        self.main_calls = 0
        self.studio_calls = 0

    def generate(self, req: LLMRequest) -> LLMResponse:
        task = str((req.metadata or {}).get("studio_specs_task") or "").strip()
        if task:
            self.studio_calls += 1
            if task == "seed_extract":
                payload = {
                    "operation_type": "create_character",
                    "targets": {"character_ids": ["demo_spec"], "mode_ids": ["helper"], "scope": "character"},
                    "character": {
                        "character_id": "demo_spec",
                        "display_name": "Demo",
                        "default_mode": "helper",
                        "llm_profile": "BALANCED",
                        "set_active": False,
                    },
                    "mode_changes": [],
                    "confidence": {
                        "operation_type": 0.95,
                        "target_character_id": 0.95,
                        "display_name": 0.95,
                        "default_mode": 0.95,
                        "llm_profile": 0.95,
                    },
                }
            else:
                payload = {"options": ["вариант 1", "вариант 2", "вариант 3"]}
            text = json.dumps(payload, ensure_ascii=False)
        else:
            self.main_calls += 1
            text = "main-model-answer"
        return LLMResponse(
            text=text,
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


class BrainPrivacyTests(unittest.TestCase):
    def _new_brain(self) -> tuple[Brain, CharacterRuntime, MemoryManager]:
        tmp = tempfile.TemporaryDirectory(prefix="mmis_brain_privacy_")
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)

        memory_manager = MemoryManager(root_dir=root)
        self.addCleanup(memory_manager.close)
        state_manager = CharacterRuntime(
            state_path=root / "brain_state.json",
            state_store_dir=root / "brain_state_store",
            autosave=False,
        )
        provider = _StudioAwareProvider()
        studio = StudioGenerator(specs_root=root / "specs", character_specs_root=root / "specs" / "characters")
        pipeline = ResponsePipeline(
            provider=provider,
            character_runtime=state_manager,
            memory_manager=memory_manager,
            studio_generator=studio,
        )
        brain = Brain(
            provider=provider,
            state_manager=state_manager,
            memory_manager=memory_manager,
            response_pipeline=pipeline,
            throttle_sec=0.0,
        )
        return brain, state_manager, memory_manager

    def _assert_memory_stores_empty(self, manager: MemoryManager, *, namespace: str) -> None:
        snap = manager.debug_snapshot(DebugRequest(namespace=namespace, limit=50))
        self.assertEqual(int(snap.get("count") or 0), 0)

    def test_slash_commands_are_not_persisted_anywhere(self) -> None:
        brain, state_manager, manager = self._new_brain()
        result = brain.handle_message(
            "/mode debugger",
            meta={"source": "test", "store_turn": True, "conversation_id": "privacy"},
        )

        self.assertEqual(str(result.route or ""), "command")
        self.assertFalse(list(state_manager.snapshot().history or []))
        self._assert_memory_stores_empty(manager, namespace="privacy")

    def test_studio_dialog_is_not_persisted_anywhere(self) -> None:
        brain, state_manager, manager = self._new_brain()
        started = brain.handle_message(
            "/studio start demo_spec",
            meta={"source": "test", "store_turn": True, "conversation_id": "privacy"},
        )
        step = brain.handle_message(
            "обнови персонажа гарри",
            meta={"source": "test", "store_turn": True, "conversation_id": "privacy"},
        )

        self.assertEqual(str(started.route or ""), "command")
        self.assertEqual(str(step.route or ""), "chat")
        self.assertFalse(list(state_manager.snapshot().history or []))
        self._assert_memory_stores_empty(manager, namespace="privacy")

    def test_regular_chat_still_persists(self) -> None:
        brain, state_manager, manager = self._new_brain()
        result = brain.handle_message(
            "hello there",
            meta={"source": "test", "store_turn": True, "conversation_id": "privacy"},
        )

        self.assertEqual(str(result.route or ""), "chat")
        self.assertGreaterEqual(len(list(state_manager.snapshot().history or [])), 2)
        snap = manager.debug_snapshot(DebugRequest(namespace="privacy", limit=50))
        self.assertGreaterEqual(int(snap.get("count") or 0), 2)


if __name__ == "__main__":
    unittest.main()
