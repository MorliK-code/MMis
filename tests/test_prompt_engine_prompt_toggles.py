from __future__ import annotations

try:
    from _output_utils import enable_unittest_json_output
except ModuleNotFoundError:
    from tests._output_utils import enable_unittest_json_output

enable_unittest_json_output()

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from prompt_engine.prompt_engine import PromptEngine
from prompt_engine.prompt_loader import PromptDocument


class _RegistryStub:
    def get_prompt(self, key: str, *, use_cache: bool = True, hot_reload: bool = True) -> PromptDocument:
        _ = (use_cache, hot_reload)
        mapping = {
            "system.base": "BASE_SYSTEM_TEXT",
            "response.safety_filter": "SAFETY_FILTER_TEXT",
            "response.formatting": "FORMAT_RULES_TEXT",
        }
        return PromptDocument(
            rel_path=f"{key}.txt",
            source_path=f"/virtual/{key}.txt",
            text=mapping.get(str(key), ""),
            id=str(key),
            version="1.0.0",
            role="system",
            tags=[],
            min_ctx=0,
            meta={},
        )

    def get_by_path(self, rel_path: str, *, use_cache: bool = True, hot_reload: bool = True) -> PromptDocument:
        _ = (rel_path, use_cache, hot_reload)
        return self.get_prompt("system.base")


class _PackStub:
    def __init__(self) -> None:
        self.blocks = {
            "user_message": "hello",
            "output_schema": "",
            "context_tags": "",
            "conversation_tail": "",
            "retrieved_memories": "",
            "long_summary": "",
        }


class PromptEnginePromptTogglesTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = PromptEngine(registry=_RegistryStub())
        self.pack = _PackStub()
        self.base_state = {
            "active_character_id": "asya",
            "active_personality_id": "asya",
            "character_prompt_block": "CHARACTER_CORE_TEXT",
            "dialog_summary": "",
            "context_tags": {},
        }

    def _compose_system_text(self, state: dict) -> tuple[str, dict]:
        result = self.engine.compose(
            prompt_pack=self.pack,
            state=state,
            traits={},
            policies={},
        )
        self.assertTrue(result.messages)
        return str(result.messages[0].content or ""), dict(result.sections or {})

    def test_disables_formatting_prompt_via_state_toggle(self) -> None:
        state = dict(self.base_state)
        state["prompt_response_formatting_enabled"] = False
        state["safety_mode"] = "read_only_tools"
        system_text, sections = self._compose_system_text(state)
        self.assertNotIn("FORMAT_RULES_TEXT", system_text)
        self.assertEqual(str(sections.get("rules_prompt_id") or ""), "none")

    def test_disables_safety_prompt_even_when_safety_mode_is_strict(self) -> None:
        state = dict(self.base_state)
        state["prompt_response_safety_filter_enabled"] = False
        state["prompt_response_formatting_enabled"] = True
        state["safety_mode"] = "strict"
        system_text, sections = self._compose_system_text(state)
        self.assertNotIn("SAFETY_FILTER_TEXT", system_text)
        self.assertIn("FORMAT_RULES_TEXT", system_text)
        self.assertEqual(str(sections.get("rules_prompt_id") or ""), "response.formatting")

    def test_enables_both_rule_prompts_when_toggles_are_on(self) -> None:
        state = dict(self.base_state)
        state["prompt_response_safety_filter_enabled"] = True
        state["prompt_response_formatting_enabled"] = True
        state["safety_mode"] = "strict"
        system_text, sections = self._compose_system_text(state)
        self.assertIn("SAFETY_FILTER_TEXT", system_text)
        self.assertIn("FORMAT_RULES_TEXT", system_text)
        self.assertEqual(
            str(sections.get("rules_prompt_id") or ""),
            "response.safety_filter+response.formatting",
        )

    def test_reads_prompt_toggles_from_system_spec(self) -> None:
        state = dict(self.base_state)
        state["safety_mode"] = "strict"
        with patch(
            "prompt_engine.prompt_engine.load_spec",
            return_value={
                "prompt_toggles": {
                    "response_safety_filter": True,
                    "response_formatting": False,
                }
            },
        ):
            system_text, sections = self._compose_system_text(state)
        self.assertIn("SAFETY_FILTER_TEXT", system_text)
        self.assertNotIn("FORMAT_RULES_TEXT", system_text)
        self.assertEqual(str(sections.get("rules_prompt_id") or ""), "response.safety_filter")

    def test_null_safety_toggle_disables_filter(self) -> None:
        state = dict(self.base_state)
        state["safety_mode"] = "strict"
        state["prompt_response_safety_filter_enabled"] = None
        with patch(
            "prompt_engine.prompt_engine.load_spec",
            return_value={
                "prompt_toggles": {
                    "response_safety_filter": True,
                    "response_formatting": True,
                }
            },
        ):
            system_text, sections = self._compose_system_text(state)
        self.assertNotIn("SAFETY_FILTER_TEXT", system_text)
        self.assertIn("FORMAT_RULES_TEXT", system_text)
        self.assertEqual(str(sections.get("rules_prompt_id") or ""), "response.formatting")


if __name__ == "__main__":
    unittest.main()
