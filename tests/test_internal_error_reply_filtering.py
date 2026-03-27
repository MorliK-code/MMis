from __future__ import annotations

import sys
import tempfile
import unittest
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from core.character_runtime import CharacterRuntime, PromptBudgets
from memory_core.utils.text_sanitizer import is_internal_error_reply, sanitize_assistant_memory_text


FALLBACK_REPLY = "Я затупила. Повтори, пожалуйста, еще раз."


class InternalErrorReplyFilteringTests(unittest.TestCase):
    @staticmethod
    def _api_app():
        cached = sys.modules.get("api.app")
        if cached is not None:
            return cached
        with patch("llm.build_provider", return_value=SimpleNamespace()):
            return import_module("api.app")

    @staticmethod
    def _runtime() -> CharacterRuntime:
        temp_dir = tempfile.TemporaryDirectory()
        runtime = CharacterRuntime(
            state_path=Path(temp_dir.name) / "brain_state.json",
            autosave=False,
        )
        runtime._test_temp_dir = temp_dir  # keep alive for the test lifetime
        return runtime

    def test_sanitizer_drops_internal_error_reply(self) -> None:
        result = sanitize_assistant_memory_text(text=f"  {FALLBACK_REPLY}\n")

        self.assertTrue(is_internal_error_reply(FALLBACK_REPLY))
        self.assertEqual(result.text, "")
        self.assertEqual(result.reason, "internal_error_reply")

    def test_character_runtime_skips_internal_error_reply_in_history(self) -> None:
        runtime = self._runtime()

        runtime.update_on_assistant_message(FALLBACK_REPLY)

        self.assertEqual(runtime.snapshot().history, [])

    def test_prompt_tail_filters_legacy_internal_error_reply(self) -> None:
        runtime = self._runtime()

        block, selected, dropped = runtime._build_conversation_tail_block(
            {
                "history": [
                    {"role": "assistant", "content": FALLBACK_REPLY},
                    {"role": "user", "content": "привет"},
                    {"role": "assistant", "content": "И тебе привет."},
                ]
            },
            PromptBudgets(tail_turns=4, tail_tokens=256, tail_turn_tokens=64),
        )

        self.assertNotIn(FALLBACK_REPLY, block)
        self.assertEqual(
            selected,
            [
                {"role": "user", "content": "привет"},
                {"role": "assistant", "content": "И тебе привет."},
            ],
        )
        self.assertEqual(dropped, 0)

    def test_api_metadata_store_skips_internal_error_reply(self) -> None:
        api_app = self._api_app()

        self.assertFalse(api_app._should_store_metadata(role="assistant", text=FALLBACK_REPLY))
        self.assertTrue(api_app._should_store_metadata(role="user", text=FALLBACK_REPLY))


if __name__ == "__main__":
    unittest.main()
