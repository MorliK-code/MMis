from __future__ import annotations

from _output_utils import enable_unittest_json_output

enable_unittest_json_output()

import json
import tempfile
import unittest
from pathlib import Path

from core.character_runtime import CharacterRuntime


class BrainStateV2Tests(unittest.TestCase):
    def test_state_persists_v2_sections(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mmis_brain_state_v2_") as tmp:
            state_path = Path(tmp) / "brain_state.json"
            runtime = CharacterRuntime(state_path=state_path, autosave=False)
            runtime.set_mode("engineer")
            runtime.set_mode_lock(True)
            runtime.set_output_format(show_parameters=False, show_summary=True)
            runtime.update_on_user_message("check traceback", meta={"intent": "bug_report", "emotion": "frustrated"})
            runtime.save()

            payload = json.loads(state_path.read_text(encoding="utf-8"))
            self.assertEqual(int(payload.get("schema_version") or 0), 2)
            global_state = dict(payload.get("global") or {})
            self.assertEqual(str(global_state.get("active_mode")), "engineer")
            self.assertTrue(bool(global_state.get("mode_lock")))
            output_format = dict(global_state.get("output_format") or {})
            self.assertIn("show_parameters", output_format)
            self.assertIn("show_summary", output_format)
            self.assertFalse(bool(output_format.get("show_parameters")))
            self.assertTrue(bool(output_format.get("show_summary")))
            self.assertIn("last_actions", global_state)
            self.assertIsInstance(global_state.get("last_actions"), list)
            self.assertIn("characters", payload)
            self.assertIsInstance(payload.get("characters"), dict)

    def test_state_keeps_short_actions_and_places_context_block_higher(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mmis_brain_state_order_") as tmp:
            state_path = Path(tmp) / "brain_state.json"
            runtime = CharacterRuntime(state_path=state_path, autosave=False)
            runtime.set_mode("engineer")
            runtime.set_mode_lock(True)
            runtime.set_mode_lock(False)
            runtime.set_output_format(show_parameters=True, show_summary=False)
            runtime.update_on_user_message("hello", meta={"intent": "chat", "emotion": "neutral"})
            runtime.save()

            payload = json.loads(state_path.read_text(encoding="utf-8"))
            global_state = dict(payload.get("global") or {})
            self.assertLessEqual(len(list(global_state.get("last_actions") or [])), 3)
            self.assertLessEqual(len(list(payload.get("last_actions") or [])), 3)

            keys = list(payload.keys())
            self.assertIn("context_tags", keys)
            self.assertIn("cooldowns", keys)
            self.assertIn("address_terms", keys)
            self.assertLess(keys.index("context_tags"), keys.index("cooldowns"))
            self.assertLess(keys.index("context_tags"), keys.index("address_terms"))


if __name__ == "__main__":
    unittest.main()
