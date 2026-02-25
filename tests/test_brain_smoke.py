import sys
import types
import unittest
from unittest.mock import patch

# `brain.py` imports ollama at module import time; provide a stub for isolated tests.
sys.modules.setdefault("ollama", types.SimpleNamespace(chat=lambda **_: {}))

from brain import Brain


class _DummyProfile:
    def summary(self):
        return "assistant summary"

    def get_summary(self):
        return "user summary"


class _DummyShortMemory:
    def get(self):
        return [{"role": "assistant", "content": "ok"}]


class _DummyMemoryManager:
    def __init__(self):
        self.assistant_profile = _DummyProfile()
        self.user_profile = _DummyProfile()
        self.short = _DummyShortMemory()
        self.stored = []

    def recall(self, _query, n_results=5):
        return []

    def last_events(self, _limit):
        return []

    def store_turn(self, user_input, reply):
        self.stored.append((user_input, reply))


class BrainSmokeTests(unittest.TestCase):
    def setUp(self):
        self.mm = _DummyMemoryManager()
        self.brain = Brain(self.mm)

    def test_typical_inputs_stay_short(self):
        mocked_response = {
            "message": {
                "content": (
                    "Конечно, я могу ответить очень подробно и с множеством уточнений, "
                    "но лучше сразу к сути. Давай разберёмся без лишней воды. "
                    "Если нужно, добавлю детали."
                )
            }
        }

        with patch("brain.ollama.chat", return_value=mocked_response):
            replies = [
                self.brain.think("Какой сегодня план?"),
                self.brain.think("Расскажи шутку"),
                self.brain.think("Уточни про дедлайн"),
            ]

        for reply in replies:
            self.assertTrue(reply)
            self.assertLessEqual(len(reply), 160)
            sentence_count = len(self.brain._segment_sentences(reply))
            self.assertLessEqual(sentence_count, 2)

    def test_ollama_options_include_brevity_controls(self):
        with patch("brain.ollama.chat", return_value={"message": {"content": "Коротко и ясно."}}) as chat_mock:
            self.brain.think("Что нового?")

        options = chat_mock.call_args.kwargs["options"]
        self.assertIn("num_predict", options)
        self.assertIn("stop", options)
        self.assertIn("mirostat", options)
        self.assertIn("presence_penalty", options)
        self.assertIn("frequency_penalty", options)


if __name__ == "__main__":
    unittest.main()
