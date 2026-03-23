"""Test thinking delta computation in Ollama provider."""

import unittest

from llm.ollama_provider import _stitch_thinking_delta


class TestThinkingDeltaComputation(unittest.TestCase):
    """Tests for _stitch_thinking_delta function."""

    def test_empty_current_returns_empty_delta(self):
        """When current is empty, should return empty delta (no new thinking)."""
        prev = "previous thinking"
        delta, new_full = _stitch_thinking_delta(prev, "")
        self.assertEqual(delta, "")
        self.assertEqual(new_full, prev)  # Keep prev when current is empty

    def test_empty_prev_returns_full_as_delta(self):
        """When prev is empty, should return full current as delta."""
        current = "This is thinking text"
        delta, new_full = _stitch_thinking_delta("", current)
        self.assertEqual(delta, current)
        self.assertEqual(new_full, current)

    def test_normal_prefix_case(self):
        """Normal case: current starts with prev."""
        prev = "This is "
        current = "This is thinking"
        delta, new_full = _stitch_thinking_delta(prev, current)
        self.assertEqual(delta, "thinking")
        self.assertEqual(new_full, current)

    def test_multiline_accumulation(self):
        """Multiline thinking accumulation."""
        prev = "Let me think\n"
        current = "Let me think\nStep 1: "
        delta, new_full = _stitch_thinking_delta(prev, current)
        self.assertEqual(delta, "Step 1: ")
        self.assertEqual(new_full, current)

    def test_no_duplicate_when_same(self):
        """When prev == current, delta should be empty."""
        text = "Same thinking"
        delta, new_full = _stitch_thinking_delta(text, text)
        self.assertEqual(delta, "")
        self.assertEqual(new_full, text)

    def test_edge_case_no_prefix(self):
        """Edge case: current doesn't start with prev."""
        prev = "Old thinking"
        current = "Completely different"
        delta, new_full = _stitch_thinking_delta(prev, current)
        # In this edge case, we return full current as delta
        self.assertEqual(delta, current)
        self.assertEqual(new_full, current)

    def test_whitespace_handling(self):
        """Whitespace should be preserved."""
        prev = "Think "
        current = "Think  about  it"  # Double spaces
        delta, new_full = _stitch_thinking_delta(prev, current)
        self.assertEqual(delta, " about  it")
        self.assertEqual(new_full, current)


if __name__ == "__main__":
    unittest.main()
