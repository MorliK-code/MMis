from __future__ import annotations

from _output_utils import enable_unittest_json_output

enable_unittest_json_output()

import unittest
from types import SimpleNamespace

from modules.internet.web.stage import WebStageConfig, WebStageV2
from modules.internet.search import SearchResult


class _FakeSearch:
    def __init__(self):
        self.last_call = {}

    def search(self, query: str, recency_days=None, domain_filter=None, k: int = 5, volatile: bool = False, query_intent: str = ""):
        self.last_call = {
            "query": query,
            "recency_days": recency_days,
            "domain_filter": list(domain_filter or []),
            "k": k,
            "volatile": volatile,
            "query_intent": query_intent,
        }
        return [
            SearchResult(
                title="Rate update",
                snippet="Snippet text",
                url="https://example.com/rates",
                source="example.com",
                published_date="2026-03-04",
            )
        ]


class _FakeScraper:
    def scrape(self, url: str):
        _ = url
        return SimpleNamespace(
            title="Rate page",
            text="Clean article body text.",
            metadata={"clean_method": "bs4", "removed_blocks": 4},
        )


class WebRetrieveMemoryPayloadTests(unittest.TestCase):
    def test_web_retrieve_writes_source_markers_to_memory(self) -> None:
        fake_search = _FakeSearch()
        stage = WebStageV2(
            search_client=fake_search,
            scraper=_FakeScraper(),
            cfg=WebStageConfig(k_search=3, k_fetch=1, max_text_chars=240),
        )
        ctx = SimpleNamespace(
            clean_user_msg="/web курс доллара",
            user_msg="/web курс доллара",
            state={"web_mode": "on"},
            meta={"web_mode": "on"},
            tags={},
            logs=[],
            retrieved_memories=[],
        )
        out = stage.run(ctx)
        self.assertEqual(list(out.retrieved_memories or []), [])
        evidence_ctx = dict(out.meta.get("web_evidence_context") or {})
        self.assertTrue(evidence_ctx)
        sources = list(evidence_ctx.get("sources") or [])
        self.assertTrue(sources)
        row = dict(sources[0] or {})
        self.assertEqual(str(row.get("url") or ""), "https://example.com/rates")
        self.assertEqual(str(row.get("domain") or ""), "example.com")
        self.assertEqual(str(row.get("clean_method") or ""), "bs4")
        self.assertEqual(str(row.get("published_at") or ""), "2026-03-04")
        self.assertTrue(str(row.get("fetched_at") or ""))
        prompt_block = str(evidence_ctx.get("prompt_block") or "")
        self.assertIn("[WEB_EVIDENCE]", prompt_block)
        self.assertIn("example.com", prompt_block)
        self.assertEqual(str(out.tags.get("web_query_intent") or ""), "fx_rate")
        self.assertEqual(str(out.tags.get("web_fresh_required") or ""), "true")
        self.assertEqual(str(out.tags.get("web_fresh_missing") or ""), "false")
        self.assertEqual(str(out.tags.get("web_response_style") or ""), "factual_direct")
        self.assertEqual(str(fake_search.last_call.get("query_intent") or ""), "fx_rate")
        self.assertTrue(bool(fake_search.last_call.get("volatile")))

    def test_weather_query_in_auto_mode_uses_web_even_with_chat_intent(self) -> None:
        fake_search = _FakeSearch()
        stage = WebStageV2(
            search_client=fake_search,
            scraper=_FakeScraper(),
            cfg=WebStageConfig(k_search=3, k_fetch=1, max_text_chars=240),
        )
        ctx = SimpleNamespace(
            clean_user_msg="какая погода в Киеве?",
            user_msg="какая погода в Киеве?",
            state={"web_mode": "auto"},
            meta={"web_mode": "auto"},
            tags={"intent": "chat"},
            logs=[],
            retrieved_memories=[],
        )
        out = stage.run(ctx)
        self.assertEqual(str(out.tags.get("web_used") or ""), "true")
        self.assertEqual(str(out.tags.get("web_query_intent") or ""), "weather")
        self.assertEqual(str(out.tags.get("web_response_style") or ""), "factual_direct")
        self.assertEqual(str(fake_search.last_call.get("query_intent") or ""), "weather")
        self.assertEqual(int(fake_search.last_call.get("recency_days") or 0), 1)

    def test_recipe_lookup_in_auto_mode_uses_web_for_generic_query(self) -> None:
        fake_search = _FakeSearch()
        stage = WebStageV2(
            search_client=fake_search,
            scraper=_FakeScraper(),
            cfg=WebStageConfig(k_search=3, k_fetch=1, max_text_chars=240),
        )
        ctx = SimpleNamespace(
            clean_user_msg="найди мне точный рецепт безе",
            user_msg="найди мне точный рецепт безе",
            state={"web_mode": "auto", "web_auto_profile": "balanced"},
            meta={"web_mode": "auto", "web_auto_profile": "balanced"},
            tags={"intent": "question", "intent_conf": 0.93},
            logs=[],
            retrieved_memories=[],
        )
        out = stage.run(ctx)
        self.assertEqual(str(out.tags.get("web_used") or ""), "true")
        self.assertEqual(str(out.tags.get("web_query_intent") or ""), "generic")
        self.assertEqual(str(fake_search.last_call.get("query_intent") or ""), "generic")

    def test_smalltalk_in_auto_aggressive_does_not_use_web(self) -> None:
        fake_search = _FakeSearch()
        stage = WebStageV2(
            search_client=fake_search,
            scraper=_FakeScraper(),
            cfg=WebStageConfig(k_search=3, k_fetch=1, max_text_chars=240),
        )
        ctx = SimpleNamespace(
            clean_user_msg="how are you?",
            user_msg="how are you?",
            state={"web_mode": "auto", "web_auto_profile": "aggressive"},
            meta={"web_mode": "auto", "web_auto_profile": "aggressive"},
            tags={"intent": "chat", "intent_conf": 0.95},
            logs=[],
            retrieved_memories=[],
        )
        out = stage.run(ctx)
        self.assertEqual(str(out.tags.get("web_used") or ""), "false")
        self.assertEqual(fake_search.last_call, {})

    def test_fx_query_retries_with_alternative_when_first_search_empty(self) -> None:
        class _RetrySearch:
            def __init__(self):
                self.calls = []

            def search(self, query: str, recency_days=None, domain_filter=None, k: int = 5, volatile: bool = False, query_intent: str = ""):
                self.calls.append(str(query))
                if len(self.calls) == 1:
                    return []
                return [
                    SearchResult(
                        title="USD/UAH",
                        snippet="42.8",
                        url="https://minfin.com.ua/currency/usd/",
                        source="minfin.com.ua",
                        published_date="2026-03-04",
                    )
                ]

        retry_search = _RetrySearch()
        stage = WebStageV2(
            search_client=retry_search,
            scraper=_FakeScraper(),
            cfg=WebStageConfig(k_search=3, k_fetch=1, max_text_chars=240),
        )
        ctx = SimpleNamespace(
            clean_user_msg="посмотри актуальный курс доллара в Украине",
            user_msg="посмотри актуальный курс доллара в Украине",
            state={"web_mode": "on"},
            meta={"web_mode": "on"},
            tags={"intent": "chat"},
            logs=[],
            retrieved_memories=[],
        )
        out = stage.run(ctx)
        self.assertGreaterEqual(len(retry_search.calls), 2)
        self.assertEqual(str(out.tags.get("web_used") or ""), "true")


if __name__ == "__main__":
    unittest.main()
