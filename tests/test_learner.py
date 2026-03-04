from __future__ import annotations

from _output_utils import enable_unittest_json_output

enable_unittest_json_output()

import unittest

from modules.character.learner import update_persona


class LearnerTests(unittest.TestCase):
    def test_updates_traits_from_signals_and_feedback(self) -> None:
        persona = {
            "traits": {
                "warmth": 0.55,
                "sarcasm": 0.35,
                "teasing": 0.45,
                "verbosity": 0.45,
                "strictness": 0.45,
                "empathy": 0.5,
            },
            "locks": {"feminine": True, "informal_you": True},
            "bans": [],
            "learned": {},
        }
        signals = {
            "intent": "bug_report",
            "emotion": "frustrated",
            "mode": "debugger",
            "tags": ["has_traceback", "has_logs"],
            "user_feedback": ["no_teasing", "ban_word:\u0430\u0440\u0442\u0435\u0444\u0430\u043a\u0442"],
        }
        updated, debug = update_persona(persona, signals)
        traits = dict(updated.get("traits") or {})
        self.assertGreater(float(traits.get("strictness", 0.0)), 0.45)
        self.assertGreater(float(traits.get("warmth", 0.0)), 0.55)
        self.assertLess(float(traits.get("sarcasm", 1.0)), 0.35)
        self.assertLess(float(traits.get("teasing", 1.0)), 0.45)
        self.assertIn("\u0430\u0440\u0442\u0435\u0444\u0430\u043a\u0442", list(updated.get("bans") or []))
        self.assertIn("feedback_deltas", debug)

    def test_initializes_baseline_from_current_traits_when_missing(self) -> None:
        persona = {
            "traits": {
                "warmth": 0.31,
                "humor": 0.62,
            },
            "learned": {},
        }
        updated, _debug = update_persona(persona, {"intent": "chat", "emotion": "neutral", "tags": []})
        baselines = dict(updated.get("baseline_traits") or {})
        self.assertAlmostEqual(float(baselines.get("warmth", 0.0)), 0.31, places=3)
        self.assertAlmostEqual(float(baselines.get("humor", 0.0)), 0.62, places=3)

    def test_auto_adds_new_traits_to_baseline(self) -> None:
        persona = {
            "traits": {
                "warmth": 0.5,
                "professionalism": 0.73,
            },
            "baseline_traits": {"warmth": 0.5},
            "learned": {},
        }
        updated, _debug = update_persona(persona, {"intent": "chat", "emotion": "neutral", "tags": []})
        baselines = dict(updated.get("baseline_traits") or {})
        self.assertIn("professionalism", baselines)
        self.assertAlmostEqual(float(baselines.get("professionalism", 0.0)), 0.73, places=3)

    def test_mode_affects_implicit_dynamic_traits(self) -> None:
        persona = {
            "traits": {
                "professionalism": 0.50,
                "emoji_rate": 0.40,
                "teasing": 0.40,
            },
            "learned": {},
        }
        updated, debug = update_persona(
            persona,
            {
                "intent": "bug_report",
                "emotion": "frustrated",
                "mode": "debugger",
                "tags": ["has_traceback"],
            },
        )
        traits = dict(updated.get("traits") or {})
        self.assertGreater(float(traits.get("professionalism", 0.0)), 0.50)
        self.assertLess(float(traits.get("emoji_rate", 1.0)), 0.40)
        self.assertLess(float(traits.get("teasing", 1.0)), 0.40)
        self.assertIn("implicit_deltas", debug)


if __name__ == "__main__":
    unittest.main()
