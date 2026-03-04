from __future__ import annotations

from _output_utils import enable_unittest_json_output

enable_unittest_json_output()

import json
import tempfile
import unittest
from pathlib import Path

from core.character_runtime import CharacterRuntime


class CharacterChangeKindsTests(unittest.TestCase):
    def test_playfulness_rule_change_logged_as_derived_axis(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mmis_change_kinds_") as tmp:
            runtime = CharacterRuntime(
                character_path=Path(tmp) / "characters",
                state_path=Path(tmp) / "brain_state.json",
                autosave=False,
            )
            cid = runtime.get_active_character_id({})
            runtime.evolve(
                text="hello there",
                meta={"intent": "chat", "mood": "neutral", "metadata_tags": []},
                active_character_id=cid,
            )

            events_path = runtime.storage.character_events_path(cid)
            self.assertTrue(events_path.exists())
            lines = [str(x).strip() for x in events_path.read_text(encoding="utf-8").splitlines() if str(x).strip()]
            self.assertTrue(lines)
            last = json.loads(lines[-1])
            changes = [dict(x) for x in list(last.get("changes") or []) if isinstance(x, dict)]

            rows = [x for x in changes if str(x.get("trait") or "").strip().lower() == "playfulness"]
            self.assertTrue(rows, msg=f"playfulness change not found in changes={changes!r}")
            self.assertTrue(all(str(x.get("kind") or "") == "derived_axis" for x in rows))


if __name__ == "__main__":
    unittest.main()

