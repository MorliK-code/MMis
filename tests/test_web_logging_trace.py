from __future__ import annotations

from _output_utils import enable_unittest_json_output

enable_unittest_json_output()

import re
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from core.character_runtime import CharacterRuntime
from core.response_pipeline import ResponsePipeline
from core.web_rag_stage import WebRagConfig, WebRetrieveStage
from llm.provider_base import (
    LLMProviderBase,
    LLMRequest,
    LLMResponse,
    ModelInfo,
    ProviderHealth,
    Timings,
    Usage,
)
from modules.internet.search import SearchResult


class _StubProvider(LLMProviderBase):
    def generate(self, req: LLMRequest) -> LLMResponse:
        _ = req
        return LLMResponse(
            text="Final answer from model.",
            usage=Usage(prompt_tokens=4, completion_tokens=6, total_tokens=10),
            timings=Timings(latency_ms=3.0),
            model="stub",
        )

    def healthcheck(self) -> ProviderHealth:
        return ProviderHealth(ok=True, provider="stub", detail="ok", model="stub")

    def model_info(self, model: str = "") -> ModelInfo:
        return ModelInfo(provider="stub", model=model or "stub")

    def list_models(self) -> list[str]:
        return ["stub"]


class _FakeSearch:
    def search(self, query: str, recency_days=None, domain_filter=None, k: int = 5, volatile: bool = False, query_intent: str = ""):
        _ = (recency_days, domain_filter, k, volatile, query_intent)
        return [
            SearchResult(
                title="Exchange Rate",
                snippet="USD to UAH market update.",
                url="https://example.com/rates",
                source="example.com",
            )
        ] if query else []


class _FakeScraper:
    def scrape(self, url: str):
        _ = url
        return SimpleNamespace(title="Rates", text="USD is 42.10 at sample source.")


class WebTraceLoggingTests(unittest.TestCase):
    def test_web_logs_cover_start_to_final_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            runtime = CharacterRuntime(
                state_path=root / "brain_state.json",
                state_store_dir=root / "brain_state_store",
                autosave=False,
            )
            pipeline = ResponsePipeline(provider=_StubProvider(), character_runtime=runtime)
            pipeline._stages["web_retrieve"] = WebRetrieveStage(
                search_client=_FakeSearch(),
                scraper=_FakeScraper(),
                cfg=WebRagConfig(k_search=3, k_fetch=1, max_text_chars=240),
            )

            result = pipeline.run(
                route="chat",
                user_msg="/web курс доллара сегодня",
                state={
                    "history": [],
                    "quality_profile": "BALANCED",
                    "active_mode": "friend_chat",
                    "web_mode": "on",
                },
                meta={"source": "test", "web_mode": "on", "store_turn": False},
                retrieved_memories=[],
                traits={},
                policies={},
            )

            self.assertTrue(result.text)
            logs = list(result.logs or [])
            joined = "\n".join(logs)
            self.assertIn("stage=web_trace start", joined)
            self.assertIn("stage=web_retrieve start", joined)
            self.assertIn("stage=web_retrieve search_start", joined)
            self.assertIn("stage=web_retrieve search_done", joined)
            self.assertIn("stage=web_retrieve fetch_ok", joined)
            self.assertIn("stage=web_trace end", joined)
            self.assertIn("web_used=true", joined)

            start_row = next((x for x in logs if "stage=web_trace start" in x), "")
            end_row = next((x for x in logs if "stage=web_trace end" in x), "")
            m1 = re.search(r"\btrace=([^\s]+)", start_row)
            m2 = re.search(r"\btrace=([^\s]+)", end_row)
            self.assertIsNotNone(m1)
            self.assertIsNotNone(m2)
            self.assertEqual(m1.group(1), m2.group(1))


if __name__ == "__main__":
    unittest.main()
