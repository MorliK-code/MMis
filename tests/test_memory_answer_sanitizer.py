from __future__ import annotations

from _output_utils import enable_unittest_json_output

enable_unittest_json_output()

import unittest
from dataclasses import dataclass, field
from typing import Any

from memory.text_sanitizer import clean_assistant_text_for_memory


@dataclass
class _ResultStub:
    text: str
    structured_output: dict[str, Any] = field(default_factory=dict)


class MemoryAnswerSanitizerTests(unittest.TestCase):
    def test_prefers_structured_output_text(self) -> None:
        result = _ResultStub(
            text="[PARAMETERS]\nmode=debugger\n\n[RESPONSE]\nfrom-formatted",
            structured_output={"text": "from-structured"},
        )
        cleaned = clean_assistant_text_for_memory(result)
        self.assertEqual(cleaned.text, "from-structured")
        self.assertEqual(cleaned.reason, "structured_output_text")

    def test_extracts_response_block(self) -> None:
        result = _ResultStub(
            text="[PARAMETERS]\nmode=debugger\n\n[SUMMARY]\nshort\n\n[RESPONSE]\nfinal answer",
            structured_output={},
        )
        cleaned = clean_assistant_text_for_memory(result)
        self.assertEqual(cleaned.text, "final answer")
        self.assertEqual(cleaned.reason, "response_block_extract")

    def test_fallback_strip_removes_parameters_and_summary(self) -> None:
        result = _ResultStub(
            text="[PARAMETERS]\nmode=debugger\n\n[SUMMARY]\nshort summary\n\njust answer",
            structured_output={},
        )
        cleaned = clean_assistant_text_for_memory(result)
        self.assertEqual(cleaned.text, "just answer")
        self.assertEqual(cleaned.reason, "fallback_strip")

    def test_keeps_plain_text_unchanged(self) -> None:
        result = _ResultStub(text="plain answer only", structured_output={})
        cleaned = clean_assistant_text_for_memory(result)
        self.assertEqual(cleaned.text, "plain answer only")
        self.assertEqual(cleaned.reason, "plain_text")


if __name__ == "__main__":
    unittest.main()
