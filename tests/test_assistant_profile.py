import json
import tempfile
import unittest
from pathlib import Path

from memory.assistant_profile import AssistantProfile


class AssistantProfileLoadTests(unittest.TestCase):
    def test_load_applies_base_defaults_without_wiping_learned_lists(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            profile_path = Path(tmpdir) / "assistant_profile.json"
            profile_path.write_text(
                json.dumps(
                    {
                        "name": "Старое имя",
                        "style": "сухо",
                        "tone": "формально",
                        "likes": ["кофе"],
                        "do_not_say": ["старая фраза"],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            profile = AssistantProfile(profile_path)

            self.assertEqual(profile.data["name"], "Вероника")
            self.assertEqual(profile.data["short_name"], "Ника")
            self.assertEqual(profile.data["style"], "лёгкая ирония")
            self.assertIn("кофе", profile.data["likes"])
            self.assertIn("старая фраза", profile.data["do_not_say"])
            self.assertIn("я ИИ", profile.data["do_not_say"])


if __name__ == "__main__":
    unittest.main()