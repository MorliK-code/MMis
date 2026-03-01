from __future__ import annotations

from _output_utils import enable_unittest_json_output
enable_unittest_json_output()

import unittest

from core.prompt_builder import PromptBuilder


class PromptBudgetTests(unittest.TestCase):
    def test_prompt_builder_respects_total_budget(self) -> None:
        state = {
            "mode": "chat",
            "history": [
                {"role": "user", "content": "hello " * 50},
                {"role": "assistant", "content": "response " * 50},
            ]
            * 8,
            "dialog_summary": "long summary " * 80,
            "context_tags": {"lang": "en", "topic": "python", "intent": "question", "mood": "neutral"},
        }
        retrieved = [
            {"text": "memory " * 50, "confidence": 0.9, "priority": 0.9, "source": "fact", "topic": "profile"}
            for _ in range(20)
        ]
        traits = {"tone": "friendly", "style": "short"}
        policies = {
            "budgets": {
                "total_tokens": 220,
                "system_tokens": 40,
                "persona_tokens": 30,
                "state_tokens": 30,
                "tags_tokens": 20,
                "memory_tokens": 35,
                "tail_tokens": 35,
                "user_tokens": 20,
                "output_schema_tokens": 20,
                "tail_turns": 4,
                "tail_turn_tokens": 16,
                "memory_item_tokens": 16,
            }
        }

        pack = PromptBuilder.build(
            state=state,
            user_msg="Please explain this error and give steps " * 10,
            retrieved_memories=retrieved,
            traits=traits,
            policies=policies,
        )

        self.assertLessEqual(pack.token_usage.get("full_prompt", 0), pack.budgets.get("total_tokens", 0))


if __name__ == "__main__":
    unittest.main()

