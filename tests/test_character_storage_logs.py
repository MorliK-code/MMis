from __future__ import annotations

from _output_utils import enable_unittest_json_output

enable_unittest_json_output()

import json
import tempfile
import unittest
from pathlib import Path

from core.character_runtime import CharacterRuntime
from config.paths import DATA_DIR
from config.settings import load_config
from modules.character.storage import CharacterStorage


class CharacterStorageLogsTests(unittest.TestCase):
    def test_default_root_is_not_data_characters(self) -> None:
        storage = CharacterStorage()
        self.assertNotEqual(storage.root.resolve(), (DATA_DIR / "characters").resolve())

    def test_explicit_legacy_root_is_redirected_to_characters_runtime(self) -> None:
        storage = CharacterStorage(root=DATA_DIR / "characters")
        self.assertNotEqual(storage.root.resolve(), (DATA_DIR / "characters").resolve())
        self.assertEqual(storage.root.resolve(), (load_config().memory_dir / "characters_runtime").resolve())

    def test_append_event_writes_to_logs_folder(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mmis_char_logs_") as tmp:
            root = Path(tmp) / "characters"
            logs = Path(tmp) / "logs"
            storage = CharacterStorage(root=root, logs_root=logs)
            storage.ensure_defaults()
            storage.append_event("asya", {"type": "character_update", "changes": [{"kind": "test"}]})

            path = logs / "characters" / "asya" / "events.jsonl"
            self.assertTrue(path.exists())
            rows = [x for x in path.read_text(encoding="utf-8-sig").splitlines() if x.strip()]
            self.assertTrue(rows)
            last = json.loads(rows[-1])
            self.assertEqual(str(last.get("type")), "character_update")

    def test_runtime_reads_character_events_from_logs_folder(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mmis_char_logs_runtime_") as tmp:
            root = Path(tmp) / "characters"
            logs = Path(tmp) / "logs"
            storage = CharacterStorage(root=root, logs_root=logs)
            storage.ensure_defaults()
            storage.append_event(
                "asya",
                {
                    "type": "character_update",
                    "changes": [{"kind": "user_feedback", "feedback": ["no_teasing"]}],
                },
            )
            legacy = storage.character_dir("asya") / "events.jsonl"
            if legacy.exists():
                legacy.unlink()

            runtime = CharacterRuntime(
                storage=storage,
                state_path=Path(tmp) / "brain_state.json",
                state_store_dir=Path(tmp) / "brain_state_store",
                autosave=False,
            )
            payload = runtime.get_persona_debug(state={"active_character_id": "asya"}, max_deltas=1)
            feedback = [str(x).strip() for x in list(payload.get("feedback") or []) if str(x).strip()]
            self.assertIn("no_teasing", feedback)


if __name__ == "__main__":
    unittest.main()
