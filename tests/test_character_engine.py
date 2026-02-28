from __future__ import annotations

import tempfile
import unittest

from modules.character import CharacterEngine, CharacterStorage


class CharacterEngineTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="mmis_char_")
        storage = CharacterStorage(root=self._tmp.name)
        self.engine = CharacterEngine(storage=storage)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_defaults_created_and_listed(self) -> None:
        ids = self.engine.list_ids()
        self.assertIn("asya", ids)
        self.assertEqual(self.engine.get_active_character_id({}), "asya")

    def test_update_applies_chat_rule_and_builds_prompt(self) -> None:
        before = self.engine.list_traits("asya")
        before_play = float(before.get("playfulness", {}).get("value", 0.0))

        result = self.engine.update(
            text="hello there",
            meta={"intent": "chat", "mood": "positive_excited", "mode": "chat"},
            active_character_id="asya",
        )

        after_play = float(result.traits.get("playfulness", {}).get("value", 0.0))
        self.assertGreaterEqual(after_play, before_play)
        self.assertTrue(result.prompt_block.strip())
        self.assertIn("[CHAR_BASE]", result.prompt_block)

    def test_set_and_remove_trait(self) -> None:
        updated = self.engine.set_trait("asya", "debug_humor", value=0.8, confidence=0.9)
        self.assertAlmostEqual(float(updated.get("value", 0.0)), 0.8, places=3)

        rows = self.engine.list_traits("asya")
        self.assertIn("debug_humor", rows)
        self.assertTrue(self.engine.remove_trait("asya", "debug_humor"))
        rows2 = self.engine.list_traits("asya")
        self.assertNotIn("debug_humor", rows2)


if __name__ == "__main__":
    unittest.main()

