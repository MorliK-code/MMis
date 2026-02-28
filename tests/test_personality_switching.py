from __future__ import annotations

import time
import unittest

from core.personality_engine import PersonalityEngine


class PersonalitySwitchingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = PersonalityEngine()

    def test_auto_switch_to_strict_for_coding_intent(self) -> None:
        now = time.time()
        decision = self.engine.decide(
            intent="coding_help",
            emotion="neutral",
            recent_context={"mode": "coding", "topic": "python"},
            active_personality_id="default",
            personality_locked=False,
            last_switch_ts=0.0,
            now_ts=now,
        )
        self.assertEqual(decision.target_personality_id, "strict")
        self.assertTrue(decision.switched)
        self.assertGreaterEqual(float(decision.confidence), 0.65)

    def test_cooldown_prevents_switch(self) -> None:
        now = time.time()
        decision = self.engine.decide(
            intent="chat",
            emotion="positive",
            recent_context={"mode": "chat", "topic": "smalltalk"},
            active_personality_id="strict",
            personality_locked=False,
            last_switch_ts=now - 5.0,
            now_ts=now,
        )
        self.assertFalse(decision.switched)
        self.assertEqual(decision.reason, "cooldown")
        self.assertEqual(decision.target_personality_id, "strict")

    def test_manual_override_locks_personality(self) -> None:
        now = time.time()
        decision = self.engine.decide(
            intent="chat",
            emotion="positive",
            recent_context={"mode": "chat"},
            active_personality_id="default",
            personality_locked=False,
            last_switch_ts=0.0,
            manual_personality="flirty",
            now_ts=now,
        )
        self.assertTrue(decision.switched)
        self.assertTrue(decision.lock_after_switch)
        self.assertEqual(decision.target_personality_id, "flirty")
        self.assertEqual(decision.reason, "manual")


if __name__ == "__main__":
    unittest.main()
