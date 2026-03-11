from __future__ import annotations

from _output_utils import enable_unittest_json_output

enable_unittest_json_output()

import unittest

from scripts.character_profile_builder import build_spec_payloads, _normalize_blueprint


class CharacterProfileBuilderTests(unittest.TestCase):
    def test_normalize_blueprint_coerces_core_fields(self) -> None:
        raw = {
            "character": {
                "name": "Кира",
                "default_mode": "engineer",
            },
            "persona_state": {
                "mood": "focused",
                "traits": {"warmth": 2, "sarcasm": -1, "humor": 0.55},
                "locks": {"feminine": True},
                "bans": ["artifact", "", "artifact"],
            },
            "persona_spec": {"identity": ["Ты — {name}."]},
            "evolution_spec": {"rules": []},
        }
        normalized = _normalize_blueprint(raw, character_id="kira", name_override="")
        character = dict(normalized.get("character") or {})
        state = dict(normalized.get("persona_state") or {})

        self.assertEqual(str(character.get("character_id")), "kira")
        self.assertEqual(str(character.get("id")), "kira")
        self.assertEqual(str(character.get("name")), "Кира")
        self.assertEqual(str(character.get("default_mode")), "engineer")
        self.assertEqual(str(state.get("mood")), "focused")

        traits = dict(state.get("traits") or {})
        self.assertAlmostEqual(float(traits.get("warmth", 0.0)), 1.0, places=6)
        self.assertAlmostEqual(float(traits.get("sarcasm", 1.0)), 0.0, places=6)
        self.assertAlmostEqual(float(traits.get("humor", 0.0)), 0.55, places=6)

        locks = dict(state.get("locks") or {})
        self.assertTrue(bool(locks.get("feminine")))
        self.assertTrue(bool(locks.get("informal_you")))
        self.assertEqual(list(state.get("bans") or []), ["artifact"])

    def test_build_spec_payloads_returns_required_files(self) -> None:
        normalized = _normalize_blueprint({}, character_id="demo", name_override="Demo")
        payloads = build_spec_payloads(normalized)
        self.assertEqual(
            sorted(payloads.keys()),
            sorted(["character.json", "persona_state.json", "persona_spec.json", "evolution_spec.json"]),
        )
        self.assertEqual(str((payloads.get("character.json") or {}).get("character_id")), "demo")

    def test_empty_blueprint_is_not_dry(self) -> None:
        normalized = _normalize_blueprint({}, character_id="neo", name_override="Нео")
        state = dict(normalized.get("persona_state") or {})
        spec = dict(normalized.get("persona_spec") or {})
        evo = dict(normalized.get("evolution_spec") or {})
        self.assertGreater(len(dict(state.get("traits") or {})), 8)
        self.assertGreater(len(list(spec.get("identity") or [])), 2)
        self.assertGreater(len(dict(spec.get("modes") or {})), 2)
        self.assertGreater(len(list(evo.get("rules") or [])), 0)


if __name__ == "__main__":
    unittest.main()
