from __future__ import annotations

from _output_utils import enable_unittest_json_output
enable_unittest_json_output()

import unittest

from metadata.metadata_extractor import MetadataExtractor
from metadata.taxonomy import EMOTIONS, INTENTS, LANGS


class MetadataTests(unittest.TestCase):
    def setUp(self) -> None:
        self.extractor = MetadataExtractor(cache_size=16, use_disk_cache=False)

    def test_extract_balanced_basic_tags(self) -> None:
        result = self.extractor.extract(
            text="How to fix python traceback error in project?",
            state={"mode": "coding", "quality_profile": "BALANCED"},
            last_messages=[{"role": "user", "content": "previous"}],
        )
        self.assertIn(result.lang, LANGS)
        self.assertIn(result.intent.label, INTENTS)
        self.assertIn(result.emotion.label, EMOTIONS)
        self.assertIn("intent_" + result.intent.label.lower(), [x.lower() for x in result.tags])
        self.assertIn("topic_python", [x.lower() for x in result.tags])
        self.assertIsInstance(result.entities, dict)

    def test_extract_fast_mode_is_lightweight(self) -> None:
        result = self.extractor.extract(
            text="ok",
            state={"mode": "chat", "quality_profile": "FAST"},
            last_messages=None,
        )
        self.assertEqual(result.entities, {})
        self.assertIn("lang_", " ".join(result.tags) if result.tags else "lang_")
        self.assertIn(result.intent.label, INTENTS)
        self.assertIn(result.emotion.label, EMOTIONS)

    def test_extract_entities_adds_entity_and_topic_tags(self) -> None:
        result = self.extractor.extract(
            text="Docker compose fails in VS Code on Windows with RTX 3060",
            state={"mode": "coding", "quality_profile": "BALANCED"},
            last_messages=None,
        )
        tags = {str(x).lower() for x in list(result.tags or [])}
        self.assertIn("topic_docker", tags)
        self.assertIn("topic_vscode", tags)
        self.assertIn("topic_windows", tags)
        self.assertIn("entity_software_docker", tags)
        self.assertIn("software", result.entities)

    def test_people_noise_does_not_leak_into_entity_tags(self) -> None:
        result = self.extractor.extract(
            text="Hello traceback how to fix docker on windows?",
            state={"mode": "coding", "quality_profile": "BALANCED"},
            last_messages=None,
        )
        tags = {str(x).lower() for x in list(result.tags or [])}
        self.assertFalse(any(tag.startswith("entity_people_") for tag in tags))
        people = [str(x).strip().lower() for x in list(dict(result.entities or {}).get("people") or []) if str(x).strip()]
        self.assertNotIn("hello", people)
        self.assertNotIn("traceback", people)
        self.assertNotIn("how", people)

    def test_lang_tag_is_canonical_and_matches_lang_field(self) -> None:
        result = self.extractor.extract(
            text="How to fix traceback in Python?",
            state={"mode": "coding", "quality_profile": "BALANCED"},
            last_messages=None,
        )
        lang_tags = [str(x).strip().lower() for x in list(result.tags or []) if str(x).strip().lower().startswith("lang_")]
        self.assertLessEqual(len(lang_tags), 1)
        if lang_tags:
            self.assertEqual(lang_tags[0], f"lang_{result.lang}")

    def test_bracket_smile_maps_to_positive_emotion(self) -> None:
        result = self.extractor.extract(
            text="Спасибо))",
            state={"mode": "chat", "quality_profile": "BALANCED"},
            last_messages=None,
        )
        self.assertIn(result.emotion.label, {"happy", "excited"})

    def test_bracket_sad_maps_to_negative_emotion(self) -> None:
        result = self.extractor.extract(
            text="Очень грустно(((",
            state={"mode": "chat", "quality_profile": "BALANCED"},
            last_messages=None,
        )
        self.assertIn(result.emotion.label, {"sad", "frustrated", "anxious"})

    def test_code_brackets_do_not_force_emotion_shift(self) -> None:
        result = self.extractor.extract(
            text="def f(x):\n    return (x + 1)\nprint(f(2))",
            state={"mode": "coding", "quality_profile": "BALANCED"},
            last_messages=None,
        )
        self.assertNotEqual(result.emotion.label, "happy")

    def test_question_smile_maps_to_positive_emotion(self) -> None:
        result = self.extractor.extract(
            text="Ну что, стартуем?)",
            state={"mode": "chat", "quality_profile": "BALANCED"},
            last_messages=None,
        )
        self.assertIn(result.emotion.label, {"happy", "excited"})

    def test_question_sad_maps_to_negative_emotion(self) -> None:
        result = self.extractor.extract(
            text="Почему так?(",
            state={"mode": "chat", "quality_profile": "BALANCED"},
            last_messages=None,
        )
        self.assertIn(result.emotion.label, {"sad", "frustrated", "anxious"})


if __name__ == "__main__":
    unittest.main()

