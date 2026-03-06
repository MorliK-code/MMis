from __future__ import annotations

from _output_utils import enable_unittest_json_output

enable_unittest_json_output()

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from core.character_runtime import CharacterRuntime
from core.response_pipeline import ResponsePipeline
from core.web_rag_stage import WebRagConfig, WebRetrieveStage
from llm.provider_base import LLMProviderBase, LLMRequest, LLMResponse, ModelInfo, ProviderHealth, Timings, Usage
from modules.internet.search import SearchResult


class _ToggleProvider(LLMProviderBase):
    def __init__(self, *, first_text: str):
        self.first_text = str(first_text or "")
        self.calls = 0

    def generate(self, req: LLMRequest) -> LLMResponse:
        _ = req
        self.calls += 1
        text = self.first_text if self.calls == 1 else "Checked web sources and refined the answer."
        return LLMResponse(
            text=text,
            usage=Usage(prompt_tokens=8, completion_tokens=8, total_tokens=16),
            timings=Timings(latency_ms=2.0),
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
        _ = (query, recency_days, domain_filter, k, volatile, query_intent)
        return [
            SearchResult(
                title="Reference",
                snippet="web reference",
                url="https://example.com/reference",
                source="example.com",
                published_date="2026-03-05",
            )
        ]


class _FakeScraper:
    def scrape(self, url: str):
        _ = url
        return SimpleNamespace(
            title="Reference page",
            text="Clean reference text.",
            metadata={"clean_method": "bs4", "removed_blocks": 1},
        )


class WebSecondPassTests(unittest.TestCase):
    def _build_pipeline(self, provider: _ToggleProvider) -> ResponsePipeline:
        tmp = tempfile.TemporaryDirectory(prefix="mmis_web_second_pass_")
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        runtime = CharacterRuntime(
            state_path=root / "brain_state.json",
            state_store_dir=root / "brain_state_store",
            autosave=False,
        )
        pipeline = ResponsePipeline(provider=provider, character_runtime=runtime)
        pipeline._stages["web_retrieve"] = WebRetrieveStage(
            search_client=_FakeSearch(),
            scraper=_FakeScraper(),
            cfg=WebRagConfig(k_search=3, k_fetch=1, max_text_chars=220),
        )
        return pipeline

    def _run(self, pipeline: ResponsePipeline, *, profile: str, user_msg: str = "tell me about this topic"):
        return pipeline.run(
            route="chat",
            user_msg=user_msg,
            state={
                "history": [],
                "quality_profile": "BALANCED",
                "active_mode": "chatting",
                "web_mode": "auto",
                "web_auto_profile": profile,
            },
            meta={
                "source": "test",
                "store_turn": False,
                "web_mode": "auto",
                "web_auto_profile": profile,
            },
            retrieved_memories=[],
            traits={},
            policies={},
        )

    def test_second_pass_runs_only_for_aggressive_profile(self) -> None:
        provider = _ToggleProvider(first_text="Not sure, maybe details need to be verified.")
        pipeline = self._build_pipeline(provider)
        result = self._run(pipeline, profile="aggressive")
        self.assertGreaterEqual(provider.calls, 2)
        self.assertIn("stage=web_second_pass applied", list(result.logs or []))

    def test_second_pass_is_skipped_for_balanced_profile(self) -> None:
        provider = _ToggleProvider(first_text="Not sure, maybe details need to be verified.")
        pipeline = self._build_pipeline(provider)
        result = self._run(pipeline, profile="balanced")
        self.assertEqual(provider.calls, 1)
        self.assertTrue(any("stage=web_second_pass skipped(profile)" in str(x) for x in list(result.logs or [])))

    def test_second_pass_not_triggered_when_answer_is_confident(self) -> None:
        provider = _ToggleProvider(first_text="This is a stable answer without uncertainty markers.")
        pipeline = self._build_pipeline(provider)
        result = self._run(pipeline, profile="aggressive")
        self.assertEqual(provider.calls, 1)
        self.assertTrue(any("stage=web_second_pass skipped(confident)" in str(x) for x in list(result.logs or [])))

    def test_second_pass_not_triggered_for_smalltalk_in_aggressive(self) -> None:
        provider = _ToggleProvider(first_text="Not sure, maybe need to verify this.")
        pipeline = self._build_pipeline(provider)
        result = self._run(pipeline, profile="aggressive", user_msg="how are you?")
        self.assertEqual(provider.calls, 1)
        self.assertTrue(any("stage=web_second_pass skipped(confident)" in str(x) for x in list(result.logs or [])))


if __name__ == "__main__":
    unittest.main()
