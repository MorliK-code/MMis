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
                "bans": ["artifact"],
            },
            active_mode="engineer",
        )
        self.assertIn("Active mode: engineer.", text)
        self.assertNotIn("{name}", text)
        self.assertNotIn("{word}", text)
        self.assertNotIn("{ban-word}", text)
        self.assertEqual(str(debug.get("active_mode")), "engineer")
        self.assertTrue(str(debug.get("mode_profile_source") or "").strip())
        self.assertTrue(str(debug.get("mode_profile_id") or "").strip())
        self.assertGreater(float(debug.get("mode_blend_alpha") or 0.0), 0.0)

    def test_missing_persona_mode_uses_runtime_profile_fallback(self) -> None:
        text, debug = compile_system_persona(
            character_id="assistant",
            persona_state={"traits": {"warmth": 0.55}, "mood": "neutral"},
            active_mode="spicy_chat",
        )
        self.assertIn("Active mode: spicy_chat.", text)
        self.assertNotEqual(str(debug.get("mode_profile_id") or ""), "friend_chat")
        self.assertIn(
            str(debug.get("mode_profile_source") or ""),
            {"builtin_preset", "semantic_preset", "modes_spec.persona_effects"},
        )

    def test_helper_and_spicy_mode_profiles_are_distinct(self) -> None:
        helper_text, helper_debug = compile_system_persona(
            character_id="assistant",
            persona_state={"traits": {"warmth": 0.55}, "mood": "neutral"},
            active_mode="helper",
        )
        spicy_text, spicy_debug = compile_system_persona(
            character_id="assistant",
            persona_state={"traits": {"warmth": 0.55}, "mood": "neutral"},
            active_mode="spicy_chat",
        )
        self.assertNotEqual(helper_text, spicy_text)
        self.assertEqual(str(helper_debug.get("mode_profile_id") or ""), "helper")
        self.assertEqual(str(spicy_debug.get("mode_profile_id") or ""), "spicy_chat")


if __name__ == "__main__":
    unittest.main()
