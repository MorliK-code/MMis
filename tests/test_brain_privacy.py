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
from memory.event_store import EventStore
from memory.long_memory import LongMemory
from memory.memory_manager import MemoryManager
from memory.short_memory import ShortMemory
from memory.vector_store import VectorStore
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


def _vector_record_count(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return 0
    rows = payload.get("records") if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        return 0
    return len(rows)


class BrainPrivacyTests(unittest.TestCase):
    def _new_brain(self) -> tuple[Brain, CharacterRuntime, MemoryManager]:
        tmp = tempfile.TemporaryDirectory(prefix="mmis_brain_privacy_")
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)

        short_memory = ShortMemory(path=root / "short_memory.json", autosave=True)
        long_memory = LongMemory(path=root / "long_memory_docs.json")
        vector_store = VectorStore(path=root / "vector_store.json", dim=64)
        event_store = EventStore(path=root / "events.jsonl")
        memory_manager = MemoryManager(
            short_memory=short_memory,
            long_memory=long_memory,
            vector_store=vector_store,
            event_store=event_store,
        )

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

    def _assert_memory_stores_empty(self, manager: MemoryManager) -> None:
        self.assertEqual(len(manager.event_store.list(50)), 0)
        self.assertEqual(manager.short_memory.size(), 0)
        self.assertEqual(len(manager.long_memory.list_docs(limit=20)), 0)
        self.assertEqual(_vector_record_count(manager.vector_store.path), 0)

    def test_slash_commands_are_not_persisted_anywhere(self) -> None:
        brain, state_manager, manager = self._new_brain()
        result = brain.handle_message("/mode debugger", meta={"source": "test", "store_turn": True})

        self.assertEqual(str(result.route or ""), "command")
        self.assertFalse(list(state_manager.snapshot().history or []))
        self._assert_memory_stores_empty(manager)

    def test_studio_dialog_is_not_persisted_anywhere(self) -> None:
        brain, state_manager, manager = self._new_brain()
        started = brain.handle_message("/studio start demo_spec", meta={"source": "test", "store_turn": True})
        step = brain.handle_message("обнови персонажа гарри", meta={"source": "test", "store_turn": True})

        self.assertEqual(str(started.route or ""), "command")
        self.assertEqual(str(step.route or ""), "chat")
        self.assertFalse(list(state_manager.snapshot().history or []))
        self._assert_memory_stores_empty(manager)

    def test_regular_chat_still_persists(self) -> None:
        brain, state_manager, manager = self._new_brain()
        result = brain.handle_message("hello there", meta={"source": "test", "store_turn": True})

        self.assertEqual(str(result.route or ""), "chat")
        self.assertGreaterEqual(len(list(state_manager.snapshot().history or [])), 2)
        self.assertGreaterEqual(len(manager.event_store.list(50)), 2)
        self.assertGreaterEqual(manager.short_memory.size(), 2)
        self.assertGreaterEqual(len(manager.long_memory.list_docs(limit=20)), 1)
        self.assertGreaterEqual(_vector_record_count(manager.vector_store.path), 2)


if __name__ == "__main__":
    unittest.main()

