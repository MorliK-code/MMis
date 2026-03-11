from __future__ import annotations

from _output_utils import enable_unittest_json_output

enable_unittest_json_output()

import json
import tempfile
import unittest
from pathlib import Path

from core.character_runtime import CharacterRuntime
from modules.character.storage import CharacterStorage


class CharacterProfileSyncTests(unittest.TestCase):
    def test_storage_syncs_runtime_character_from_specs(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mmis_char_sync_") as tmp:
            root = Path(tmp)
            storage = CharacterStorage(root=root / "runtime", logs_root=root / "logs")
            storage.ensure_defaults()

            spec_path = storage.character_spec_dir("asya") / "character.json"
            runtime_path = storage.character_dir("asya") / "character.json"

            spec_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "character_id": "asya",
                        "name": "Asya",
                        "default_mood": "neutral",
                        "default_mode": "engineer",
                        "model_profile": "ASYA",
                        "custom_from_spec": "enabled",
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            runtime_path.write_text(
                json.dumps(
                    {
                        "id": "asya",
                        "name": "Asya stale",
                        "default_mood": "thoughtful",
                        "default_mode": "chatting",
                        "llm_profile": "QUALITY",
                        "runtime_only": "keep",
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

            loaded = storage.load_character("asya")
            self.assertEqual(str(loaded.get("llm_profile") or ""), "ASYA")
            self.assertEqual(str(loaded.get("model_profile") or ""), "ASYA")
            self.assertEqual(str(loaded.get("default_mode") or ""), "engineer")
            self.assertEqual(str(loaded.get("custom_from_spec") or ""), "enabled")
            self.assertEqual(str(loaded.get("runtime_only") or ""), "keep")

            runtime_synced = json.loads(runtime_path.read_text(encoding="utf-8"))
            self.assertEqual(str(runtime_synced.get("llm_profile") or ""), "ASYA")
            self.assertEqual(str(runtime_synced.get("default_mode") or ""), "engineer")

    def test_runtime_meta_refreshes_after_specs_change(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mmis_char_meta_refresh_") as tmp:
            root = Path(tmp)
            storage = CharacterStorage(root=root / "runtime", logs_root=root / "logs")
            storage.ensure_defaults()

            spec_path = storage.character_spec_dir("asya") / "character.json"
            spec_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "character_id": "asya",
                        "name": "Asya",
                        "default_mood": "neutral",
                        "default_mode": "chatting",
                        "llm_profile": "QUALITY",
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

            runtime = CharacterRuntime(
                storage=storage,
                state_path=root / "brain_state.json",
                state_store_dir=root / "brain_state_store",
                autosave=False,
            )

            first = runtime.get_meta("asya")
            self.assertEqual(str(first.llm_profile or ""), "QUALITY")

            spec_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "character_id": "asya",
                        "name": "Asya",
                        "default_mood": "neutral",
                        "default_mode": "chatting",
                        "model_profile": "ASYA",
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

            refreshed = runtime.get_meta("asya")
            self.assertEqual(str(refreshed.llm_profile or ""), "ASYA")

            debug = runtime.get_brain_debug(state={"active_character_id": "asya", "quality_profile": "QUALITY"})
            self.assertEqual(str(debug.get("quality_profile") or ""), "ASYA")


if __name__ == "__main__":
    unittest.main()
