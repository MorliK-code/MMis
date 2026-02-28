from __future__ import annotations

import unittest

from prompt_engine.token_budget_manager import ContextBlock, TokenBudget, TokenBudgetManager


class TokenEconomyTests(unittest.TestCase):
    def test_fit_context_blocks_respects_total_minus_reserve(self) -> None:
        budget = TokenBudget(total=320, reserve=80, system_core=70, personality=40, rules=40, recent_chat=80, memory_retrieval=60)
        manager = TokenBudgetManager(budget=budget, chars_per_token=4.0)

        blocks = [
            ContextBlock(id="system_core", content="sys " * 200, bucket="system", required=True, priority=100),
            ContextBlock(id="personality_core", content="persona " * 120, bucket="personality", priority=90),
            ContextBlock(id="rules", content="rules " * 120, bucket="rules", required=True, priority=95),
            ContextBlock(id="recent_chat", content="chat " * 300, bucket="history", priority=60, shrink_strategy="summarize"),
            ContextBlock(id="memory", content="memory " * 300, bucket="memory", priority=50, shrink_strategy="drop"),
            ContextBlock(id="user", content="what should I do? " * 30, bucket="user", required=True, priority=100, min_tokens=12),
        ]

        fitted, stats = manager.fit_context_blocks(blocks)
        self.assertLessEqual(int(stats.get("total_tokens") or 0), int(stats.get("usable_total") or 0))
        self.assertTrue(str(fitted.get("user") or "").strip())


if __name__ == "__main__":
    unittest.main()
