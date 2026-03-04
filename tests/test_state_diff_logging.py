from __future__ import annotations

from _output_utils import enable_unittest_json_output
enable_unittest_json_output()

import tempfile
import unittest
from pathlib import Path

from core.character_runtime import CharacterRuntime


class StateDiffLoggingTests(unittest.TestCase):
    def test_mode_change_records_state_diff(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mmis_state_diff_") as tmp:
            runtime = CharacterRuntime(
                character_path=Path(tmp) / "characters",
                state_path=Path(tmp) / "brain_state.json",
                state_store_dir=Path(tmp) / "brain_state_store",
                autosave=False,
            )
            runtime.set_mode("debugger")
            actions = list(runtime.snapshot().last_actions or [])
            self.assertTrue(actions)
            last = dict(actions[-1] or {})
            self.assertEqual(str(last.get("type")), "MODE_CHANGED")
            diff = dict(last.get("state_diff") or {})
            self.assertIn("active_mode", diff)
            self.assertEqual(str(dict(diff.get("active_mode") or {}).get("new")), "debugger")

    def test_turn_started_records_state_diff(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mmis_state_diff_turn_") as tmp:
            runtime = CharacterRuntime(
                character_path=Path(tmp) / "characters",
                state_path=Path(tmp) / "brain_state.json",
                state_store_dir=Path(tmp) / "brain_state_store",
                autosave=False,
            )
            runtime.update_on_user_message("check traceback", meta={"intent": "bug_report", "emotion": "frustrated"})
            actions = list(runtime.snapshot().last_actions or [])
            self.assertTrue(actions)
            last = dict(actions[-1] or {})
            self.assertEqual(str(last.get("type")), "TURN_STARTED")
            diff = dict(last.get("state_diff") or {})
            self.assertIn("turn_id", diff)
            self.assertEqual(int(dict(diff.get("turn_id") or {}).get("new") or 0), 1)

    def test_persona_update_records_state_diff(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mmis_state_diff_persona_") as tmp:
            runtime = CharacterRuntime(
                character_path=Path(tmp) / "characters",
                state_path=Path(tmp) / "brain_state.json",
                state_store_dir=Path(tmp) / "brain_state_store",
                autosave=False,
            )
            runtime._set_character_persona_entry(
                "asya",
                {
                    "persona": {
                        "traits": {
                            "warmth": 0.91,
                            "sarcasm": 0.05,
                            "strictness": 0.61,
                            "verbosity": 0.44,
                            "teasing": 0.08,
                            "empathy": 0.72,
                        },
                        "mood": "focused",
                        "locks": {"feminine": True, "informal_you": True},
                        "bans": ["artifact"],
                        "learned": {
                            "preferences_confirmed": [],
                            "preferences_pending": [],
                            "style_bias": {},
                        },
                    },
                    "local_context": {"last_topic_weights": {}, "last_seen_ts": ""},
                },
            )
            actions = list(runtime.snapshot().last_actions or [])
            self.assertTrue(actions)
            last = dict(actions[-1] or {})
            self.assertEqual(str(last.get("type")), "CHARACTER_PERSONA_UPDATED")
            diff = dict(last.get("state_diff") or {})
            self.assertTrue(diff)
            self.assertIn("traits", diff)


if __name__ == "__main__":
    unittest.main()
