from __future__ import annotations

from _output_utils import enable_unittest_json_output
enable_unittest_json_output()

import unittest

from metadata.metadata_extractor import MetadataExtractor


class MetadataTests(unittest.TestCase):
    def setUp(self) -> None:
        self.extractor = MetadataExtractor(cache_size=16)

    def test_extract_balanced_basic_tags(self) -> None:
        result = self.extractor.extract(
            text="How to fix python traceback error in project?",
            state={"mode": "coding", "quality_profile": "BALANCED"},
            last_messages=[{"role": "user", "content": "previous"}],
        )
        self.assertTrue(result.lang)
        self.assertTrue(result.intent.label)
        self.assertIn("intent_" + result.intent.label.lower(), [x.lower() for x in result.tags])
        self.assertIsInstance(result.entities, dict)

    def test_extract_fast_mode_is_lightweight(self) -> None:
        result = self.extractor.extract(
            text="ok",
            state={"mode": "chat", "quality_profile": "FAST"},
            last_messages=None,
        )
        self.assertEqual(result.entities, {})
        self.assertIn("lang_", " ".join(result.tags) if result.tags else "lang_")


if __name__ == "__main__":
    unittest.main()

