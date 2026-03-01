<<<<<<< HEAD
﻿from __future__ import annotations

from _output_utils import enable_unittest_json_output
enable_unittest_json_output()
=======
from __future__ import annotations
>>>>>>> 51b8456b510b4061cb471a6e7b7574d205e99e04

import tempfile
import unittest

from modules.character import CharacterEngine, CharacterStorage


class CharacterEngineTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="mmis_char_")
        storage = CharacterStorage(root=self._tmp.name)
        self.engine = CharacterEngine(storage=storage)
<<<<<<< HEAD
        self.character_id = self.engine.get_active_character_id({})
=======
>>>>>>> 51b8456b510b4061cb471a6e7b7574d205e99e04

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_defaults_created_and_listed(self) -> None:
        ids = self.engine.list_ids()
<<<<<<< HEAD
        self.assertIn("default", ids)
        self.assertIn(self.character_id, ids)
        self.assertEqual(self.engine.get_active_character_id({}), self.character_id)

    def test_update_applies_chat_rule_and_builds_prompt(self) -> None:
        before = self.engine.list_traits(self.character_id)
=======
        self.assertIn("asya", ids)
        self.assertEqual(self.engine.get_active_character_id({}), "asya")

    def test_update_applies_chat_rule_and_builds_prompt(self) -> None:
        before = self.engine.list_traits("asya")
>>>>>>> 51b8456b510b4061cb471a6e7b7574d205e99e04
        before_play = float(before.get("playfulness", {}).get("value", 0.0))

        result = self.engine.update(
            text="hello there",
            meta={"intent": "chat", "mood": "positive_excited", "mode": "chat"},
<<<<<<< HEAD
            active_character_id=self.character_id,
=======
            active_character_id="asya",
>>>>>>> 51b8456b510b4061cb471a6e7b7574d205e99e04
        )

        after_play = float(result.traits.get("playfulness", {}).get("value", 0.0))
        self.assertGreaterEqual(after_play, before_play)
        self.assertTrue(result.prompt_block.strip())
        self.assertIn("[CHAR_BASE]", result.prompt_block)

    def test_set_and_remove_trait(self) -> None:
<<<<<<< HEAD
        updated = self.engine.set_trait(self.character_id, "debug_humor", value=0.8, confidence=0.9)
        self.assertAlmostEqual(float(updated.get("value", 0.0)), 0.8, places=3)

        rows = self.engine.list_traits(self.character_id)
        self.assertIn("debug_humor", rows)
        self.assertTrue(self.engine.remove_trait(self.character_id, "debug_humor"))
        rows2 = self.engine.list_traits(self.character_id)
=======
        updated = self.engine.set_trait("asya", "debug_humor", value=0.8, confidence=0.9)
        self.assertAlmostEqual(float(updated.get("value", 0.0)), 0.8, places=3)

        rows = self.engine.list_traits("asya")
        self.assertIn("debug_humor", rows)
        self.assertTrue(self.engine.remove_trait("asya", "debug_humor"))
        rows2 = self.engine.list_traits("asya")
>>>>>>> 51b8456b510b4061cb471a6e7b7574d205e99e04
        self.assertNotIn("debug_humor", rows2)


if __name__ == "__main__":
    unittest.main()

<<<<<<< HEAD

=======
>>>>>>> 51b8456b510b4061cb471a6e7b7574d205e99e04
