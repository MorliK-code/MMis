from __future__ import annotations

from _output_utils import enable_unittest_json_output
enable_unittest_json_output()

import tempfile
import unittest
from pathlib import Path

from memory.short_memory import ShortMemory


class ShortMemoryV2Tests(unittest.TestCase):
    def test_summary_is_meta_aware_for_technical_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            sm = ShortMemory(
                limit=10,
                summary_trigger=10,
                persona_snapshot_every=2,
                path=root / "short.json",
                autosave=False,
            )
            sm.append(
                {
                    "id": "1",
                    "role": "user",
                    "text": "Traceback in python script when loading config",
                    "intent": "bug_report",
                    "tags": ["has_traceback", "topic_python", "topic_windows"],
                    "meta": {"active_mode": "engineer", "turn_id": 1},
                }
            )
            sm.append(
                {
                    "id": "2",
                    "role": "assistant",
                    "text": "Use docker compose and check VS Code settings.json",
                    "intent": "task",
                    "tags": ["topic_docker", "topic_vscode", "has_command", "has_config"],
                    "meta": {"active_mode": "engineer", "turn_id": 1},
                }
            )
            sm.append(
                {
                    "id": "3",
                    "role": "user",
                    "text": "please give exact steps",
                    "intent": "task",
                    "tags": ["is_task", "needs_steps", "topic_docker"],
                    "meta": {"active_mode": "engineer", "turn_id": 2},
                }
            )
            sm.append(
                {
                    "id": "4",
                    "role": "assistant",
                    "text": "Done, step list provided",
                    "intent": "task",
                    "tags": ["topic_docker"],
                    "meta": {"active_mode": "engineer", "turn_id": 2},
                }
            )
            sm.append(
                {
                    "id": "5",
                    "role": "user",
                    "text": "thanks",
                    "intent": "chat",
                    "tags": [],
                    "meta": {"active_mode": "engineer", "turn_id": 3},
                }
            )
            for idx in range(6, 13):
                sm.append(
                    {
                        "id": str(idx),
                        "role": "assistant" if (idx % 2 == 0) else "user",
                        "text": f"filler {idx}",
                        "intent": "chat",
                        "tags": [],
                        "meta": {"active_mode": "engineer", "turn_id": idx},
                    }
                )

            summary = sm.rolling_summary()
            summary_meta = sm.rolling_summary_meta()

            self.assertIn("tasks:", summary)
            self.assertIn("errors:", summary)
            self.assertIn("focus: technical", summary)
            self.assertIn("topics:", summary)
            self.assertTrue(list(summary_meta.get("topics") or []))

    def test_persona_snapshot_event_is_not_saved(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            sm = ShortMemory(
                limit=10,
                summary_trigger=8,
                persona_snapshot_every=2,
                path=root / "short.json",
                autosave=False,
            )
            sm.append(
                {
                    "id": "a1",
                    "role": "assistant",
                    "text": "turn one",
                    "intent": "chat",
                    "meta": {
                        "turn_id": 1,
                        "active_mode": "friend_chat",
                        "persona_snapshot": {
                            "character_id": "asya",
                            "mood": "neutral",
                            "traits": {"warmth": 0.6, "sarcasm": 0.2},
                        },
                    },
                    "persona_snapshot": {
                        "character_id": "asya",
                        "mood": "neutral",
                        "traits": {"warmth": 0.6, "sarcasm": 0.2},
                    },
                }
            )
            sm.append(
                {
                    "id": "a2",
                    "role": "assistant",
                    "text": "turn two",
                    "intent": "task",
                    "meta": {
                        "turn_id": 2,
                        "active_mode": "engineer",
                        "persona_snapshot": {
                            "character_id": "asya",
                            "mood": "serious",
                            "traits": {"warmth": 0.52, "strictness": 0.7},
                        },
                    },
                    "persona_snapshot": {
                        "character_id": "asya",
                        "mood": "serious",
                        "traits": {"warmth": 0.52, "strictness": 0.7},
                    },
                }
            )

            snapshots = [x for x in sm.tail(20) if str(x.get("type") or "").strip().lower() == "persona_snapshot"]
            self.assertEqual(len(snapshots), 0)


if __name__ == "__main__":
    unittest.main()
