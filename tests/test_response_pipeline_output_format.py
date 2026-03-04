from __future__ import annotations

from _output_utils import enable_unittest_json_output

enable_unittest_json_output()

import unittest

from core.response_pipeline import ResponsePipeline
from llm.provider_base import (
    LLMProviderBase,
    LLMRequest,
    LLMResponse,
    ModelInfo,
    ProviderHealth,
    Timings,
    Usage,
)


class _SummaryProvider(LLMProviderBase):
    def __init__(self, *, main_text: str, summary_text: str = "short summary", fail_summary: bool = False):
        self.main_text = str(main_text or "")
        self.summary_text = str(summary_text or "")
        self.fail_summary = bool(fail_summary)

    def generate(self, req: LLMRequest) -> LLMResponse:
        is_summary = bool(dict(req.metadata or {}).get("summary_mini_pass", False))
        if is_summary:
            if self.fail_summary:
                raise RuntimeError("summary failed")
            text = self.summary_text
        else:
            text = self.main_text
        return LLMResponse(
            text=text,
            usage=Usage(prompt_tokens=8, completion_tokens=4, total_tokens=12),
            timings=Timings(latency_ms=5.0),
            model=req.model or "stub",
        )

    def healthcheck(self) -> ProviderHealth:
        return ProviderHealth(ok=True, provider="stub", detail="ok", model="stub")

    def model_info(self, model: str = "") -> ModelInfo:
        return ModelInfo(provider="stub", model=model or "stub")

    def list_models(self) -> list[str]:
        return ["stub"]


class ResponsePipelineOutputFormatTests(unittest.TestCase):
    def _run_chat(self, pipeline: ResponsePipeline, *, state: dict, meta: dict | None = None, text: str = "hello"):
        return pipeline.run(
            route="chat",
            user_msg=text,
            state=dict(state),
            meta={"source": "test", "store_turn": False, **dict(meta or {})},
            retrieved_memories=[],
            traits={},
            policies={},
        )

    def test_debugger_mode_uses_three_blocks_by_default(self) -> None:
        pipeline = ResponsePipeline(provider=_SummaryProvider(main_text="Main answer body.", summary_text="Debug summary."))
        result = self._run_chat(
            pipeline,
            state={
                "history": [],
                "quality_profile": "BALANCED",
                "active_mode": "debugger",
                "mode_lock": True,
            },
            text="traceback here",
        )
        self.assertIn("[PARAMETERS]", result.text)
        self.assertIn("[SUMMARY]", result.text)
        self.assertIn("[RESPONSE]", result.text)
        self.assertTrue(bool(dict(result.structured_output or {}).get("formatted")))

    def test_friend_chat_mode_keeps_plain_text_by_default(self) -> None:
        pipeline = ResponsePipeline(provider=_SummaryProvider(main_text="Plain friendly answer."))
        result = self._run_chat(
            pipeline,
            state={
                "history": [],
                "quality_profile": "BALANCED",
                "active_mode": "friend_chat",
                "mode_lock": True,
            },
            text="just chat",
        )
        self.assertEqual(result.text, "Plain friendly answer")
        self.assertFalse(bool(dict(result.structured_output or {}).get("formatted")))

    def test_manual_off_overrides_debugger_defaults(self) -> None:
        pipeline = ResponsePipeline(provider=_SummaryProvider(main_text="Main answer body."))
        result = self._run_chat(
            pipeline,
            state={
                "history": [],
                "quality_profile": "BALANCED",
                "active_mode": "debugger",
                "mode_lock": True,
                "output_format": {"show_parameters": False, "show_summary": False},
            },
            text="traceback here",
        )
        self.assertEqual(result.text, "Main answer body")
        self.assertFalse(bool(dict(result.structured_output or {}).get("formatted")))

    def test_summary_uses_mini_pass_when_available(self) -> None:
        pipeline = ResponsePipeline(
            provider=_SummaryProvider(main_text="Main answer body.", summary_text="Mini-pass summary text.")
        )
        result = self._run_chat(
            pipeline,
            state={
                "history": [],
                "quality_profile": "BALANCED",
                "active_mode": "debugger",
                "mode_lock": True,
            },
            text="traceback here",
        )
        self.assertIn("Mini-pass summary text.", result.text)
        self.assertEqual(str(dict(result.structured_output or {}).get("summary") or ""), "Mini-pass summary text.")

    def test_summary_falls_back_when_mini_pass_fails(self) -> None:
        pipeline = ResponsePipeline(
            provider=_SummaryProvider(
                main_text="First sentence. Second sentence. Third sentence.",
                fail_summary=True,
            )
        )
        result = self._run_chat(
            pipeline,
            state={
                "history": [],
                "quality_profile": "BALANCED",
                "active_mode": "debugger",
                "mode_lock": True,
            },
            text="traceback here",
        )
        self.assertIn("[SUMMARY]", result.text)
        self.assertIn("First sentence. Second sentence.", result.text)

    def test_json_mode_skips_output_formatter(self) -> None:
        pipeline = ResponsePipeline(provider=_SummaryProvider(main_text='{"ok":true}'))
        result = self._run_chat(
            pipeline,
            state={
                "history": [],
                "quality_profile": "BALANCED",
                "active_mode": "debugger",
                "mode_lock": True,
            },
            meta={"json_mode": True},
            text="traceback here",
        )
        self.assertEqual(result.text, '{"ok":true}')
        self.assertFalse(bool(dict(result.structured_output or {}).get("formatted")))


if __name__ == "__main__":
    unittest.main()
