from __future__ import annotations

from _output_utils import enable_unittest_json_output

enable_unittest_json_output()

import unittest

from modules.character.persona_compiler import compile_system_persona


class PersonaCompilerTests(unittest.TestCase):
    def test_compiler_includes_mode_and_locks(self) -> None:
        text, debug = compile_system_persona(
            character_id="asya",
            persona_state={
                "traits": {
                    "warmth": 0.8,
                    "sarcasm": 0.15,
                    "verbosity": 0.75,
                    "strictness": 0.55,
                    "teasing": 0.1,
                    "empathy": 0.85,
                    "professionalism": 0.8,
                },
                "mood": "neutral",
                "locks": {"feminine": True, "informal_you": True},
                "bans": ["артефакт"],
            },
            active_mode="engineer",
        )
        self.assertIn("Active mode: engineer.", text)
        self.assertIn("женский род", text.lower())
        self.assertIn("артефакт", text)
        self.assertNotIn("{name}", text)
        self.assertNotIn("{word}", text)
        self.assertNotIn("{ban-word}", text)
        # persona_spec for asya uses "lines": [...], not only "line".
        self.assertIn("Тон заметно тёплый и поддерживающий.", text)
        # Dynamic trait selection should include non-legacy traits too.
        self.assertIn("В задачах держи деловой стиль и структуру.", text)
        self.assertEqual(str(debug.get("active_mode")), "engineer")


if __name__ == "__main__":
    unittest.main()
