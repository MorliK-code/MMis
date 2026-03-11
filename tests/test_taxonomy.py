from __future__ import annotations

from _output_utils import enable_unittest_json_output
enable_unittest_json_output()

import unittest

from metadata.taxonomy import (
    normalize_emotion,
    normalize_intent,
    normalize_lang,
    normalize_topic_list,
)


class TaxonomyTests(unittest.TestCase):
    def test_normalize_lang(self) -> None:
        self.assertEqual(normalize_lang("ru-RU"), "ru")
        self.assertEqual(normalize_lang("english"), "en")
        self.assertEqual(normalize_lang(""), "unknown")

    def test_normalize_intent(self) -> None:
        self.assertEqual(normalize_intent("coding_help"), "task")
        self.assertEqual(normalize_intent("task_request"), "task")
        self.assertEqual(normalize_intent("review"), "code_review")
        self.assertEqual(normalize_intent(""), "chat")

    def test_normalize_emotion(self) -> None:
        self.assertEqual(normalize_emotion("frustrated_angry"), "frustrated")
        self.assertEqual(normalize_emotion("positive_excited"), "excited")
        self.assertEqual(normalize_emotion("confused"), "anxious")
        self.assertEqual(normalize_emotion(""), "neutral")

    def test_normalize_topics(self) -> None:
        topics = normalize_topic_list(["topic_python", "vs_code", "docker", "unknown_topic", "docker"])
        self.assertEqual(topics, ["python", "vscode", "docker"])


if __name__ == "__main__":
    unittest.main()
