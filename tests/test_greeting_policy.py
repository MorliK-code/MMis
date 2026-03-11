from __future__ import annotations

try:
    from _output_utils import enable_unittest_json_output
except ModuleNotFoundError:
    from tests._output_utils import enable_unittest_json_output

enable_unittest_json_output()

import sys
import time
import unittest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from core.response_pipeline import ResponsePipeline, _compute_greeting_flags, _local_day_kyiv
from llm.provider_base import (
    LLMProviderBase,
    LLMRequest,
    LLMResponse,
    ModelInfo,
    ProviderHealth,
    Timings,
    Usage,
)

RU_HELLO = "\u043f\u0440\u0438\u0432\u0435\u0442"
RU_ANALYSIS = "\u0432\u043e\u0442 \u043e\u0442\u0432\u0435\u0442 \u043f\u043e \u0434\u0435\u043b\u0443"
RU_TERM = "\u043c\u0438\u043b\u0430\u0448\u043a\u0430"


class _StubProvider(LLMProviderBase):
    def __init__(self, text: str):
        self._text = str(text)

    def generate(self, req: LLMRequest) -> LLMResponse:
        _ = req
        return LLMResponse(
            text=self._text,
            usage=Usage(prompt_tokens=8, completion_tokens=8, total_tokens=16),
            timings=Timings(latency_ms=5.0),
            model="stub",
        )

    def healthcheck(self) -> ProviderHealth:
        return ProviderHealth(ok=True, provider="stub", detail="ok", model="stub")

    def model_info(self, model: str = "") -> ModelInfo:
        return ModelInfo(provider="stub", model=model or "stub")

    def list_models(self) -> list[str]:
        return ["stub"]


class GreetingPolicyTests(unittest.TestCase):
    def test_user_greeting_has_priority_even_in_continuing_smalltalk(self) -> None:
        flags = _compute_greeting_flags(
            text=RU_HELLO,
            state={
                "conversation_id": "c1",
                "context_tags": {"conversation_state": "continuing_smalltalk"},
                "cooldowns": {
                    "prev_user_ts": time.time(),
                    "prev_session_id": "c1",
                    "greeting_date_local": _local_day_kyiv(),
                },
            },
            meta={},
            policies={},
        )
        self.assertTrue(flags["user_greeting"])
        self.assertTrue(flags["allow_greeting"])

    def test_new_session_allows_greeting_when_not_greeted_today(self) -> None:
        flags = _compute_greeting_flags(
            text="\u043f\u043e\u043a\u0430\u0436\u0438 \u043b\u043e\u0433",
            state={
                "conversation_id": "c1",
                "context_tags": {},
                "cooldowns": {
                    "prev_user_ts": time.time() - (7 * 3600),
                    "prev_session_id": "c1",
                    "greeting_date_local": "",
                },
            },
            meta={},
            policies={"greeting_session_timeout_sec": 6 * 3600},
        )
        self.assertTrue(flags["new_session"])
        self.assertFalse(flags["user_greeting"])
        self.assertTrue(flags["allow_greeting"])

    def test_continuing_smalltalk_blocks_auto_greeting(self) -> None:
        flags = _compute_greeting_flags(
            text="\u043d\u0430\u043f\u043e\u043c\u043d\u0438 \u0447\u0442\u043e \u043c\u044b \u0440\u0435\u0448\u0438\u043b\u0438",
            state={
                "conversation_id": "c1",
                "context_tags": {"conversation_state": "continuing_smalltalk"},
                "cooldowns": {
                    "prev_user_ts": time.time() - (7 * 3600),
                    "prev_session_id": "c1",
                    "greeting_date_local": "",
                },
            },
            meta={},
            policies={"greeting_session_timeout_sec": 6 * 3600},
        )
        self.assertTrue(flags["new_session"])
        self.assertFalse(flags["user_greeting"])
        self.assertFalse(flags["allow_greeting"])

    def test_postprocess_strips_leading_greeting_when_disallowed(self) -> None:
        provider = _StubProvider("\u041f\u0440\u0438\u0432\u0435\u0442! \u0427\u0435\u043c \u043f\u043e\u043c\u043e\u0447\u044c?\n\u0412\u043e\u0442 \u043e\u0442\u0432\u0435\u0442 \u043f\u043e \u0434\u0435\u043b\u0443.")
        pipeline = ResponsePipeline(provider=provider)

        result = pipeline.run(
            route="chat",
            user_msg="\u0441\u0434\u0435\u043b\u0430\u0439 \u0440\u0430\u0437\u0431\u043e\u0440",
            state={"mode": "chat", "history": [], "quality_profile": "BALANCED"},
            meta={"source": "test", "store_turn": False, "allow_greeting": False},
            retrieved_memories=[],
            traits={},
            policies={},
        )

        low = result.text.lower()
        self.assertNotIn(RU_HELLO, low)
        self.assertNotIn("\u0447\u0435\u043c \u043f\u043e\u043c\u043e\u0447\u044c", low)
        self.assertIn(RU_ANALYSIS, low)

    def test_pipeline_blocks_term_when_use_term_now_false(self) -> None:
        provider = _StubProvider("\u041c\u0438\u043b\u0430\u0448\u043a\u0430, \u0434\u0430\u0432\u0430\u0439 \u0441\u0440\u0430\u0437\u0443 \u043a \u0434\u0435\u043b\u0443.")
        pipeline = ResponsePipeline(provider=provider)

        result = pipeline.run(
            route="chat",
            user_msg="\u043f\u043e\u043a\u0430\u0436\u0438 \u043a\u043e\u043c\u0430\u043d\u0434\u044b",
            state={"mode": "chat", "history": [], "quality_profile": "BALANCED"},
            meta={
                "source": "test",
                "store_turn": False,
                "allow_greeting": False,
                "use_term_now": False,
                "allowed_term": RU_TERM,
                "address_terms_policy": {
                    "terms_list": [RU_TERM],
                    "use_term_now": False,
                    "allowed_term": RU_TERM,
                },
            },
            retrieved_memories=[],
            traits={},
            policies={},
        )
        self.assertNotIn(RU_TERM, result.text.lower())

    def test_pipeline_removes_term_when_session_ban_active(self) -> None:
        provider = _StubProvider("\u041c\u0438\u043b\u0430\u0448\u043a\u0430, \u0432\u043e\u0442 \u0440\u0430\u0437\u0431\u043e\u0440. \u0418 \u0434\u0430, \u043c\u0438\u043b\u0430\u0448\u043a\u0430, \u0432\u0442\u043e\u0440\u043e\u0439 \u0440\u0430\u0437.")
        pipeline = ResponsePipeline(provider=provider)

        result = pipeline.run(
            route="chat",
            user_msg="\u0434\u0430\u0439 \u0440\u0430\u0437\u0431\u043e\u0440",
            state={"mode": "chat", "history": [], "quality_profile": "BALANCED"},
            meta={
                "source": "test",
                "store_turn": False,
                "address_terms_policy": {
                    "terms_list": [RU_TERM],
                    "banned_terms_effective": [RU_TERM],
                    "banned_terms_active": True,
                    "use_term_now": False,
                    "allowed_term": RU_TERM,
                },
            },
            retrieved_memories=[],
            traits={},
            policies={},
        )
        self.assertNotIn(RU_TERM, result.text.lower())


if __name__ == "__main__":
    unittest.main()

