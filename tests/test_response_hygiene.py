from __future__ import annotations

try:
    from _output_utils import enable_unittest_json_output
except ModuleNotFoundError:
    from tests._output_utils import enable_unittest_json_output
enable_unittest_json_output()

import sys
import unittest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from core.response_pipeline import (
    ResponsePipeline,
    _apply_time_sensitive_web_failsafe,
    _enforce_response_hygiene,
    _filter_retrieved_memories_for_time_sensitive_web,
)
from llm.provider_base import (
    LLMProviderBase,
    LLMRequest,
    LLMResponse,
    ModelInfo,
    ProviderHealth,
    Timings,
    Usage,
)
from modules.character.evaluator import ResponseConstraintEvaluator


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


class ResponseHygieneTests(unittest.TestCase):
    def test_hygiene_removes_service_lines_and_prefixes(self) -> None:
        src = (
            "thinking> internal note\n"
            "assistant> Привет\n"
            "[model: qwen3:8b]\n"
            "user> ignored\n"
            "[thinking]\n"
            "assistant> Как могу помочь?"
        )
        self.assertEqual(_enforce_response_hygiene(src), "Привет\nКак могу помочь?")

    def test_hygiene_dedupes_adjacent_repeats(self) -> None:
        src = "assistant> Простите, если что-то обидело.\nassistant> Простите, если что-то обидело."
        self.assertEqual(_enforce_response_hygiene(src), "Простите, если что-то обидело.")

    def test_pipeline_applies_hygiene_in_postprocess(self) -> None:
        provider = _StubProvider(
            "thinking> plan\nassistant> Простите, если что-то обидело.\nassistant> Простите, если что-то обидело."
        )
        pipeline = ResponsePipeline(provider=provider)

        result = pipeline.run(
            route="chat",
            user_msg="ok",
            state={"mode": "chat", "history": [], "quality_profile": "BALANCED"},
            meta={"source": "test", "store_turn": False},
            retrieved_memories=[],
            traits={},
            policies={},
        )

        self.assertEqual(result.text, "Поняла. Перейду сразу к сути.")

    def test_pipeline_replaces_smalltalk_echo(self) -> None:
        provider = _StubProvider("Как у тебя дела?")
        pipeline = ResponsePipeline(provider=provider)

        result = pipeline.run(
            route="chat",
            user_msg="как у тебя дела?",
            state={"mode": "chat", "history": [], "quality_profile": "BALANCED"},
            meta={"source": "test", "store_turn": False},
            retrieved_memories=[],
            traits={},
            policies={},
        )

        self.assertEqual(result.text, "У меня все нормально, спасибо. Как ты?")

    def test_pipeline_replaces_generic_short_echo(self) -> None:
        provider = _StubProvider("что дальше")
        pipeline = ResponsePipeline(provider=provider)

        result = pipeline.run(
            route="chat",
            user_msg="что дальше",
            state={"mode": "chat", "history": [], "quality_profile": "BALANCED"},
            meta={"source": "test", "store_turn": False},
            retrieved_memories=[],
            traits={},
            policies={},
        )

        self.assertEqual(result.text, "Поняла. Я на связи и готова помочь. Уточни, что именно нужно.")

    def test_post_filter_removes_banned_term_in_prefix_and_body(self) -> None:
        evaluator = ResponseConstraintEvaluator()
        out, applied = evaluator.enforce(
            "Милашка, вот план. В конце снова милашка.",
            address_terms_policy={
                "terms_list": ["милашка"],
                "banned_terms_effective": ["милашка"],
                "banned_terms_active": True,
                "use_term_now": False,
            },
        )
        self.assertNotIn("милашка", out.lower())
        self.assertIn("terms_removed_banned", applied)

    def test_post_filter_leaves_max_one_term_when_use_term_now_true(self) -> None:
        evaluator = ResponseConstraintEvaluator()
        out, applied = evaluator.enforce(
            "Милашка, старт. Потом милашка снова.",
            address_terms_policy={
                "terms_list": ["милашка"],
                "allowed_term": "милашка",
                "use_term_now": True,
            },
        )
        self.assertEqual(out.lower().count("милашка"), 1)
        self.assertIn("terms_limited_to_one", applied)

    def test_time_sensitive_web_failsafe_removes_flirty_opener(self) -> None:
        src = "Ах, ты опять спрашиваешь о курсе? Курс USD/UAH: 43.10 (источник: minfin.com.ua, 2026-03-05)."
        out = _apply_time_sensitive_web_failsafe(
            src,
            web_intent="fx_rate",
            web_response_style="factual_direct",
        )
        self.assertFalse(out.lower().startswith("ах, ты опять"))
        self.assertIn("Курс USD/UAH", out)

    def test_time_sensitive_memory_filter_drops_noisy_message_blocks(self) -> None:
        items = [
            {"text": "[SUMMARY]\nАх, ты опять спрашиваешь о новостях?", "source": "message", "topic": "chat"},
            {
                "text": "[WEB] Rate\nsource_url: https://minfin.com.ua\nsource_domain: minfin.com.ua\nfetched_at: 2026-03-05T00:00:00Z\ntext: usd/uah 43.1",
                "source": "web",
                "topic": "web:fx_rate",
            },
        ]
        filtered, dropped = _filter_retrieved_memories_for_time_sensitive_web(items)
        self.assertEqual(dropped, 1)
        self.assertEqual(len(filtered), 1)
        self.assertEqual(str(filtered[0].get("topic") or ""), "web:fx_rate")


if __name__ == "__main__":
    unittest.main()
