from __future__ import annotations

from _output_utils import enable_unittest_json_output

enable_unittest_json_output()

import unittest

from modules.character.feedback_detector import detect_feedback


class FeedbackDetectorTests(unittest.TestCase):
    def test_detect_no_teasing_and_ban_word(self) -> None:
        items = detect_feedback("Не подкалывай и не говори артефакт")
        self.assertIn("no_teasing", items)
        self.assertIn("ban_word:артефакт", items)

    def test_detect_lock_and_verbosity(self) -> None:
        items = detect_feedback("Пиши в женском роде и давай подробнее")
        self.assertIn("lock_feminine", items)
        self.assertIn("longer_answers", items)

    def test_detect_ban_phrase_from_quotes(self) -> None:
        items = detect_feedback('Так не говори фразу "уточни, что ты имеешь в виду"')
        self.assertIn("avoid_wording", items)
        self.assertIn("ban_word:уточни, что ты имеешь в виду", items)


if __name__ == "__main__":
    unittest.main()

