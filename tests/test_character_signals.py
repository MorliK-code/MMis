from __future__ import annotations

from _output_utils import enable_unittest_json_output

enable_unittest_json_output()

import unittest

from modules.character.signals import build_character_signals


class CharacterSignalsTests(unittest.TestCase):
    def test_lang_falls_back_to_lang_tag(self) -> None:
        sig = build_character_signals(
            text="",
            metadata={
                "lang": "",
                "metadata_tags": ["intent_chat", "lang_ru", "tone_neutral"],
            },
        )
        self.assertEqual(sig.lang, "ru")
        self.assertIn("lang_ru", sig.tags)

    def test_lang_tag_canonicalized_from_explicit_lang(self) -> None:
        sig = build_character_signals(
            text="",
            metadata={
                "lang": "en",
                "metadata_tags": ["lang_ru", "intent_question", "tone_neutral"],
            },
        )
        self.assertEqual(sig.lang, "en")
        self.assertIn("lang_en", sig.tags)
        self.assertNotIn("lang_ru", sig.tags)

    def test_mode_is_propagated_and_normalized(self) -> None:
        sig = build_character_signals(
            text="",
            metadata={
                "mode": "debug",
                "metadata_tags": [],
            },
        )
        self.assertEqual(sig.mode, "debugger")


if __name__ == "__main__":
    unittest.main()
