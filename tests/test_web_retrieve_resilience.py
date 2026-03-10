from __future__ import annotations

from _output_utils import enable_unittest_json_output

enable_unittest_json_output()

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from core.character_runtime import CharacterRuntime
from core.response_pipeline import ResponsePipeline
from modules.internet.web.stage import WebStageConfig, WebStageV2
from llm.provider_base import LLMProviderBase, LLMRequest, LLMResponse, ModelInfo, ProviderHealth, Timings, Usage


class _StubProvider(LLMProviderBase):
    def __init__(self):
        self.last_request: LLMRequest | None = None

    def generate(self, req: LLMRequest) -> LLMResponse:
        self.last_request = req
        return LLMResponse(
            text="Model answer without web data.",
            usage=Usage(prompt_tokens=4, completion_tokens=5, total_tokens=9),
            timings=Timings(latency_ms=3.0),
            model="stub",
        )

    def healthcheck(self) -> ProviderHealth:
        return ProviderHealth(ok=True, provider="stub", detail="ok", model="stub")

    def model_info(self, model: str = "") -> ModelInfo:
        return ModelInfo(provider="stub", model=model or "stub")

    def list_models(self) -> list[str]:
        return ["stub"]


class _FailSearch:
    def search(self, query: str, recency_days=None, domain_filter=None, k: int = 5, volatile: bool = False, query_intent: str = ""):
        _ = (query, recency_days, domain_filter, k, volatile, query_intent)
        raise RuntimeError("searx unavailable")


class _DummyScraper:
    def scrape(self, url: str):
        _ = url
        raise RuntimeError("blocked")


class WebRetrieveResilienceTests(unittest.TestCase):
    def test_search_failure_returns_local_guardrail_reply_without_model_dispatch(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mmis_web_resilience_") as tmpdir:
            root = Path(tmpdir)
            runtime = CharacterRuntime(
                state_path=root / "brain_state.json",
                state_store_dir=root / "brain_state_store",
                autosave=False,
            )
            provider = _StubProvider()
            pipeline = ResponsePipeline(provider=provider, character_runtime=runtime)
            pipeline._stages["web_retrieve"] = WebStageV2(
                search_client=_FailSearch(),
                scraper=_DummyScraper(),
                cfg=WebStageConfig(k_search=3, k_fetch=1, max_text_chars=240),
            )

            result = pipeline.run(
                route="chat",
                user_msg="/web курс доллара",
                state={
                    "history": [],
                    "quality_profile": "BALANCED",
                    "active_mode": "chatting",
                    "web_mode": "on",
                },
                meta={"source": "test", "web_mode": "on", "store_turn": False},
                retrieved_memories=[],
                traits={},
                policies={},
            )

            self.assertIn("Не смогла подтвердить актуальный курс USD/UAH", result.text)
            joined = "\n".join(list(result.logs or []))
            self.assertIn("stage=web_retrieve search_fail", joined)
            self.assertNotIn("stage=web_retrieve error=", joined)
            self.assertIn("stage=generate route=chat web_guardrail=local_reply", joined)

            req = provider.last_request
            self.assertIsNone(req)

    def test_scrape_failure_falls_back_to_search_snippet(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mmis_web_resilience_") as tmpdir:
            root = Path(tmpdir)
            runtime = CharacterRuntime(
                state_path=root / "brain_state.json",
                state_store_dir=root / "brain_state_store",
                autosave=False,
            )
            provider = _StubProvider()
            pipeline = ResponsePipeline(provider=provider, character_runtime=runtime)

            class _OkSearch:
                def search(self, query: str, recency_days=None, domain_filter=None, k: int = 5, volatile: bool = False, query_intent: str = ""):
                    _ = (query, recency_days, domain_filter, k, volatile, query_intent)
                    from modules.internet.search import SearchResult

                    return [
                        SearchResult(
                            title="USD/UAH market",
                            snippet="USD/UAH around 42.9",
                            url="https://minfin.com.ua/currency/usd/",
                            source="minfin.com.ua",
                            published_date="2026-03-04",
                        )
                    ]

            pipeline._stages["web_retrieve"] = WebStageV2(
                search_client=_OkSearch(),
                scraper=_DummyScraper(),
                cfg=WebStageConfig(k_search=3, k_fetch=1, max_text_chars=240),
            )

            result = pipeline.run(
                route="chat",
                user_msg="/web курс доллара",
                state={
                    "history": [],
                    "quality_profile": "BALANCED",
                    "active_mode": "chatting",
                    "web_mode": "on",
                },
                meta={"source": "test", "web_mode": "on", "store_turn": False},
                retrieved_memories=[],
                traits={},
                policies={},
            )

            joined = "\n".join(list(result.logs or []))
            self.assertIn("stage=web_retrieve fetch_snippet_fallback", joined)
            self.assertIn("web_used=true", joined)
            self.assertIsNotNone(provider.last_request)

    def test_time_sensitive_web_adds_factual_policy_rule(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mmis_web_resilience_") as tmpdir:
            root = Path(tmpdir)
            runtime = CharacterRuntime(
                state_path=root / "brain_state.json",
                state_store_dir=root / "brain_state_store",
                autosave=False,
            )
            provider = _StubProvider()
            pipeline = ResponsePipeline(provider=provider, character_runtime=runtime)

            class _OkSearch:
                def search(self, query: str, recency_days=None, domain_filter=None, k: int = 5, volatile: bool = False, query_intent: str = ""):
                    _ = (query, recency_days, domain_filter, k, volatile, query_intent)
                    from modules.internet.search import SearchResult

                    return [
                        SearchResult(
                            title="USD/UAH market",
                            snippet="USD/UAH around 43.2",
                            url="https://minfin.com.ua/currency/usd/",
                            source="minfin.com.ua",
                            published_date="2026-03-05",
                        )
                    ]

            class _OkScraper:
                def scrape(self, url: str):
                    _ = url
                    return SimpleNamespace(
                        title="USD/UAH",
                        text="Market quote page",
                        metadata={"clean_method": "bs4", "removed_blocks": 1},
                    )

            pipeline._stages["web_retrieve"] = WebStageV2(
                search_client=_OkSearch(),
                scraper=_OkScraper(),
                cfg=WebStageConfig(k_search=3, k_fetch=1, max_text_chars=240),
            )

            _ = pipeline.run(
                route="chat",
                user_msg="/web курс доллара",
                state={
                    "history": [],
                    "quality_profile": "BALANCED",
                    "active_mode": "chatting",
                    "web_mode": "on",
                },
                meta={"source": "test", "web_mode": "on", "store_turn": False},
                retrieved_memories=[],
                traits={},
                policies={},
            )

            req = provider.last_request
            self.assertIsNotNone(req)
            system_text = str((req.messages[0].content if req and req.messages else "") or "")
            self.assertTrue(
                ("avoid rhetorical openers" in system_text)
                or ("direct and factual without rhetorical/flirty openers" in system_text)
            )
            self.assertTrue(
                ("source domain and fetch/publish time" in system_text)
                or ("source domain and timestamp" in system_text)
            )
            self.assertIn("<<<WEB_EVIDENCE>>>", system_text)
            self.assertIn("[WEB_EVIDENCE]", system_text)


if __name__ == "__main__":
    unittest.main()
