from __future__ import annotations

from _output_utils import enable_unittest_json_output

enable_unittest_json_output()

import unittest

from core.mode_selector import ModeSelector


class ModeSelectorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.selector = ModeSelector(confidence_threshold=0.8)

    def test_traceback_prefers_debugger(self) -> None:
        decision = self.selector.decide(
            active_mode="friend_chat",
            mode_lock=False,
            intent="bug_report",
            emotion="frustrated",
            tags=["has_traceback", "topic_python"],
        )
        self.assertEqual(decision.mode, "debugger")
        self.assertTrue(self.selector.should_switch(current_mode="friend_chat", decision=decision))

    def test_code_task_prefers_engineer(self) -> None:
        decision = self.selector.decide(
            active_mode="friend_chat",
            mode_lock=False,
            intent="task",
            emotion="neutral",
            tags=["has_code", "topic_python"],
        )
        self.assertEqual(decision.mode, "engineer")

    def test_planning_prefers_planner(self) -> None:
        decision = self.selector.decide(
            active_mode="friend_chat",
            mode_lock=False,
            intent="planning",
            emotion="neutral",
            tags=[],
        )
        self.assertEqual(decision.mode, "planner")

    def test_lock_keeps_current_mode(self) -> None:
        decision = self.selector.decide(
            active_mode="engineer",
            mode_lock=True,
            intent="chat",
            emotion="neutral",
            tags=[],
        )
        self.assertEqual(decision.mode, "engineer")
        self.assertFalse(self.selector.should_switch(current_mode="engineer", decision=decision))

    def test_casual_chat_returns_friend_mode(self) -> None:
        decision = self.selector.decide(
            active_mode="debugger",
            mode_lock=False,
            intent="chat",
            emotion="neutral",
            tags=["intent_chat", "emotion_neutral"],
        )
        self.assertEqual(decision.mode, "friend_chat")
        self.assertTrue(self.selector.should_switch(current_mode="debugger", decision=decision))


if __name__ == "__main__":
    unittest.main()
