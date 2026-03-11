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


    def test_detect_more_feedback_patterns_phrase(self) -> None:
        ru = "\u043d\u0430\u0434\u043e \u0431\u043e\u043b\u044c\u0448\u0435 \u0444\u0438\u0434\u0431\u0435\u043a \u043f\u0430\u0442\u0442\u0435\u0440\u043d\u043e\u0432"
        items = detect_feedback(ru)
        self.assertIn("longer_answers", items)
        items_en = detect_feedback("add more feedback patterns")
        self.assertIn("longer_answers", items_en)

    def test_detect_compliments_feedback(self) -> None:
        less_ru = "\u043c\u0435\u043d\u044c\u0448\u0435 \u043a\u043e\u043c\u043f\u043b\u0438\u043c\u0435\u043d\u0442\u043e\u0432"
        more_ru = "\u0431\u043e\u043b\u044c\u0448\u0435 \u043a\u043e\u043c\u043f\u043b\u0438\u043c\u0435\u043d\u0442\u043e\u0432"
        none_ru = "\u0431\u0435\u0437 \u043a\u043e\u043c\u043f\u043b\u0438\u043c\u0435\u043d\u0442\u043e\u0432"

        self.assertIn("less_compliments", detect_feedback(less_ru))
        self.assertIn("more_compliments", detect_feedback(more_ru))
        self.assertIn("no_compliments", detect_feedback(none_ru))
        self.assertIn("no_compliments", detect_feedback("no compliments"))


if __name__ == "__main__":
    unittest.main()
