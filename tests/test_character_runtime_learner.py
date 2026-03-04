from __future__ import annotations

from _output_utils import enable_unittest_json_output

enable_unittest_json_output()

import tempfile
import unittest
import json
from pathlib import Path

from core.character_runtime import CharacterRuntime


class CharacterRuntimeLearnerTests(unittest.TestCase):
    def test_persona_traits_drift_over_turns(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mmis_learner_") as tmp:
            runtime = CharacterRuntime(
                character_path=Path(tmp) / "characters",
                state_path=Path(tmp) / "brain_state.json",
                autosave=False,
            )
            cid = runtime.get_active_character_id({})
            runtime.build_personality_block(cid)
            before_traits = (
                dict(runtime.snapshot().raw.get("characters", {}).get(cid, {}).get("persona", {}).get("traits", {}))
            )
            before_strict = float(before_traits.get("strictness", 0.5))
            before_sarcasm = float(before_traits.get("sarcasm", 0.5))

            for _ in range(30):
                runtime.evolve(
                    text="Traceback (most recent call last): RuntimeError",
                    meta={
                        "intent": "bug_report",
                        "mood": "frustrated",
                        "metadata_tags": ["has_traceback", "has_logs", "topic_python"],
                    },
                    active_character_id=cid,
                )

            after_traits = (
                dict(runtime.snapshot().raw.get("characters", {}).get(cid, {}).get("persona", {}).get("traits", {}))
            )
            self.assertGreaterEqual(float(after_traits.get("strictness", 0.0)), before_strict)
            self.assertLessEqual(float(after_traits.get("sarcasm", 1.0)), before_sarcasm)

    def test_feedback_no_teasing_changes_persona(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mmis_learner_feedback_") as tmp:
            runtime = CharacterRuntime(
                character_path=Path(tmp) / "characters",
                state_path=Path(tmp) / "brain_state.json",
                autosave=False,
            )
            cid = runtime.get_active_character_id({})
            runtime.build_personality_block(cid)
            before = (
                dict(runtime.snapshot().raw.get("characters", {}).get(cid, {}).get("persona", {}).get("traits", {}))
            )
            runtime.evolve(
                text="Не подкалывай меня",
                meta={"intent": "chat", "mood": "neutral", "metadata_tags": []},
                active_character_id=cid,
            )
            after = (
                dict(runtime.snapshot().raw.get("characters", {}).get(cid, {}).get("persona", {}).get("traits", {}))
            )
            self.assertNotEqual(float(after.get("teasing", 1.0)), float(before.get("teasing", 1.0)))

            actions = [dict(x) for x in list(runtime.snapshot().raw.get("last_actions") or []) if isinstance(x, dict)]
            self.assertTrue(any(str(x.get("type") or "").upper() == "FEEDBACK_RECEIVED" for x in actions))

            events_path = runtime.storage.character_events_path(cid)
            self.assertTrue(events_path.exists())
            payload = events_path.read_text(encoding="utf-8")
            self.assertIn("user_feedback", payload)

    def test_evolve_dedupes_same_turn_id(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mmis_learner_dedupe_") as tmp:
            runtime = CharacterRuntime(
                character_path=Path(tmp) / "characters",
                state_path=Path(tmp) / "brain_state.json",
                autosave=False,
            )
            cid = runtime.get_active_character_id({})
            meta = {
                "intent": "bug_report",
                "mood": "frustrated",
                "turn_id": 12,
                "conversation_id": "conv-test-1",
                "metadata_tags": ["has_traceback", "has_logs", "topic_python"],
            }
            first = runtime.evolve(
                text="Traceback (most recent call last): RuntimeError",
                meta=dict(meta),
                active_character_id=cid,
            )
            second = runtime.evolve(
                text="Traceback (most recent call last): RuntimeError",
                meta=dict(meta),
                active_character_id=cid,
            )
            self.assertTrue(first.prompt_block)
            self.assertTrue(second.prompt_block)
            self.assertEqual(list(second.changes or []), [])

            events_path = runtime.storage.character_events_path(cid)
            self.assertTrue(events_path.exists())
            updates = 0
            for raw in events_path.read_text(encoding="utf-8-sig").splitlines():
                line = str(raw or "").strip()
                if not line:
                    continue
                row = json.loads(line)
                if str(row.get("type") or "").strip().lower() == "character_update":
                    updates += 1
            self.assertEqual(updates, 1)


if __name__ == "__main__":
    unittest.main()
