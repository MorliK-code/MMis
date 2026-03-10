from __future__ import annotations

from _output_utils import enable_unittest_json_output

enable_unittest_json_output()

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from core.character_runtime import CharacterRuntime
from core.response_pipeline import ResponsePipeline
from modules.internet.web.stage import WebStageConfig, WebStageV2
from llm.provider_base import LLMProviderBase, LLMRequest, LLMResponse, ModelInfo, ProviderHealth, Timings, Usage
from modules.internet.search import SearchResult


class _StubProvider(LLMProviderBase):
    def generate(self, req: LLMRequest) -> LLMResponse:
        _ = req
        return LLMResponse(
            text="ok",
            usage=Usage(prompt_tokens=5, completion_tokens=5, total_tokens=10),
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
                title="USD/UAH",
                snippet="usd uah market snapshot",
                url="https://minfin.com.ua/currency/usd/",
                source="minfin.com.ua",
                published_date="2026-03-05",
                score=0.91,
                score_breakdown={"freshness": 0.3, "trust": 0.34, "lexical": 0.27},
            )
        ]


class _FakeScraper:
    def scrape(self, url: str):
        _ = url
        return SimpleNamespace(
            title="USD UAH page",
            text="USD/UAH 43.1",
            status_code=200,
            final_url="https://minfin.com.ua/currency/usd/",
            metadata={"clean_method": "bs4", "removed_blocks": 2, "raw_len": 1200, "clean_len": 210},
        )


class WebTraceEnvelopeTests(unittest.TestCase):
    def test_web_trace_envelope_and_correlation(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mmis_web_trace_") as tmpdir:
            root = Path(tmpdir)
            runtime = CharacterRuntime(
                state_path=root / "brain_state.json",
                state_store_dir=root / "brain_state_store",
                autosave=False,
            )
            pipeline = ResponsePipeline(provider=_StubProvider(), character_runtime=runtime)
            pipeline._stages["web_retrieve"] = WebStageV2(
                search_client=_FakeSearch(),
                scraper=_FakeScraper(),
                cfg=WebStageConfig(k_search=3, k_fetch=1, max_text_chars=240),
            )

            captured: list[dict] = []

            def _capture(_logger, event: str, **payload):
                captured.append({"event_name": str(event), **dict(payload or {})})

            with patch("core.response_pipeline.log_json", side_effect=_capture):
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

        self.assertTrue(captured)
        trace_rows = [x for x in captured if str(x.get("trace_id") or "").strip()]
        self.assertTrue(trace_rows)
        trace_id = str(trace_rows[0].get("trace_id") or "")
        self.assertTrue(trace_id)
        self.assertTrue(all(str(x.get("trace_id") or "") == trace_id for x in trace_rows))

        events = {str(x.get("event_name") or "") for x in trace_rows}
        self.assertIn("web_trace_start", events)
        self.assertIn("web_retrieve_decision", events)
        self.assertIn("web_search_request", events)
        self.assertIn("web_search_results", events)
        self.assertIn("web_budget_update", events)
        self.assertIn("web_evidence_pack", events)
        self.assertIn("web_memory_write", events)
        self.assertIn("web_fetch_item", events)
        self.assertIn("web_context_injected", events)
        self.assertIn("web_trace_end", events)

        search_row = next((x for x in trace_rows if str(x.get("event_name") or "") == "web_search_results"), {})
        search_payload = dict(search_row.get("payload") or {})
        top_results = list(search_payload.get("top_results") or [])
        self.assertTrue(top_results)
        first = dict(top_results[0] or {})
        self.assertIn("score_breakdown", first)
        self.assertIn("snippet_500", first)
        self.assertIn("snippet_sha1", first)
        self.assertIn("snippet_len", first)

        fetch_row = next((x for x in trace_rows if str(x.get("event_name") or "") == "web_fetch_item"), {})
        fetch_payload = dict(fetch_row.get("payload") or {})
        self.assertIn("text_500", fetch_payload)
        self.assertIn("text_sha1", fetch_payload)
        self.assertIn("text_len", fetch_payload)
        self.assertLessEqual(len(str(fetch_payload.get("text_500") or "")), 500)

        sample = trace_rows[0]
        self.assertIn("ts", sample)
        self.assertIn("route", sample)
        self.assertIn("profile", sample)
        self.assertIn("stage_profile", sample)
        self.assertIn("web_mode", sample)
        self.assertIn("web_query_intent", sample)
        self.assertIn("web_fresh_required", sample)
        self.assertIn("web_fresh_missing", sample)
        self.assertIn("payload", sample)
        self.assertIsInstance(sample.get("payload"), dict)

        self.assertEqual(str(result.stats.get("web_trace_id") or ""), trace_id)
        meta = dict(result.structured_output.get("meta") or {})
        self.assertEqual(str(meta.get("web_trace_id") or ""), trace_id)


if __name__ == "__main__":
    unittest.main()
