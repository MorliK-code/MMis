from __future__ import annotations

from _output_utils import enable_unittest_json_output

enable_unittest_json_output()

import unittest
from types import SimpleNamespace

from modules.internet.search import SearchResult
from modules.internet.web.freshness import assess_freshness
from modules.internet.web.query_classifier import classify_query
from modules.internet.web.stage import WebRagConfig, WebRetrieveStage
from modules.internet.web.web_models import ConfidenceAssessment, WebSearchMode
from modules.internet.web.web_policy import WebPolicyEngine, config_from_dict


class WebPolicyV2Tests(unittest.TestCase):
    def _engine(self) -> WebPolicyEngine:
        cfg = config_from_dict(
            {
                "never_search_categories": ["architecture", "reasoning"],
                "category_penalties": {"architecture": 0.42},
            }
        )
        return WebPolicyEngine(cfg)

    def test_category_penalty_is_soft_and_can_be_overridden_by_temporal_signals(self) -> None:
        engine = self._engine()
        query = "какие сейчас лучшие практики архитектуры RAG в 2026"
        classification = classify_query(query)
        freshness = assess_freshness(query=query, classification=classification)
        confidence = ConfidenceAssessment(score=0.32, level="low", reasons=["test"], breakdown={})
        decision = engine.decide(
            query=query,
            classification=classification,
            confidence=confidence,
            freshness=freshness,
            web_mode="auto",
            web_auto_profile="balanced",
            internet_enabled=True,
        )
        self.assertGreater(float(decision.category_penalty_applied), 0.0)
        self.assertTrue(decision.should_search)
        self.assertIn(decision.mode, {WebSearchMode.TARGETED_SEARCH, WebSearchMode.DEEP_SEARCH})

    def test_local_architecture_without_freshness_stays_shallow(self) -> None:
        engine = self._engine()
        query = "объясни архитектуру нашего локального пайплайна памяти"
        classification = classify_query(query)
        freshness = assess_freshness(query=query, classification=classification)
        confidence = ConfidenceAssessment(score=0.22, level="low", reasons=["test"], breakdown={})
        decision = engine.decide(
            query=query,
            classification=classification,
            confidence=confidence,
            freshness=freshness,
            web_mode="auto",
            web_auto_profile="balanced",
            internet_enabled=True,
        )
        self.assertIn(decision.mode, {WebSearchMode.NO_SEARCH, WebSearchMode.SOFT_SEARCH})

    def test_no_web_mode_is_hard_gate(self) -> None:
        engine = self._engine()
        query = "latest chromadb version"
        classification = classify_query(query)
        freshness = assess_freshness(query=query, classification=classification)
        confidence = ConfidenceAssessment(score=0.95, level="high", reasons=["test"], breakdown={})
        decision = engine.decide(
            query=query,
            classification=classification,
            confidence=confidence,
            freshness=freshness,
            web_mode="off",
            web_auto_profile="balanced",
            internet_enabled=True,
        )
        self.assertEqual(decision.mode, WebSearchMode.NO_SEARCH)

    def test_web_mode_on_forces_minimum_verify(self) -> None:
        engine = self._engine()
        query = "объясни архитектуру локального модуля"
        classification = classify_query(query)
        freshness = assess_freshness(query=query, classification=classification)
        confidence = ConfidenceAssessment(score=0.91, level="high", reasons=["test"], breakdown={})
        decision = engine.decide(
            query=query,
            classification=classification,
            confidence=confidence,
            freshness=freshness,
            web_mode="on",
            web_auto_profile="balanced",
            internet_enabled=True,
        )
        self.assertNotEqual(decision.mode, WebSearchMode.NO_SEARCH)
        self.assertIn(
            decision.mode,
            {
                WebSearchMode.VERIFY_ONLY,
                WebSearchMode.SOFT_SEARCH,
                WebSearchMode.TARGETED_SEARCH,
                WebSearchMode.DEEP_SEARCH,
            },
        )


class WebStageV2IntegrationTests(unittest.TestCase):
    def test_local_architecture_query_avoids_web_in_auto_mode(self) -> None:
        class _SpySearch:
            def __init__(self):
                self.calls = 0

            def search(self, query: str, recency_days=None, domain_filter=None, k: int = 5, volatile: bool = False, query_intent: str = ""):
                _ = (query, recency_days, domain_filter, k, volatile, query_intent)
                self.calls += 1
                return []

        search = _SpySearch()
        stage = WebRetrieveStage(search_client=search, scraper=SimpleNamespace(scrape=lambda _url: None), cfg=WebRagConfig())
        ctx = SimpleNamespace(
            clean_user_msg="объясни архитектуру нашего пайплайна памяти",
            user_msg="объясни архитектуру нашего пайплайна памяти",
            state={"web_mode": "auto", "web_auto_profile": "balanced"},
            meta={"web_mode": "auto", "web_auto_profile": "balanced"},
            tags={"intent": "question"},
            logs=[],
            retrieved_memories=[],
            memory_context={},
        )
        out = stage.run(ctx)
        self.assertEqual(search.calls, 0)
        self.assertEqual(str(out.tags.get("web_used") or ""), "false")

    def test_temporal_architecture_query_uses_web_despite_category_penalty(self) -> None:
        class _SpySearch:
            def __init__(self):
                self.calls = 0

            def search(self, query: str, recency_days=None, domain_filter=None, k: int = 5, volatile: bool = False, query_intent: str = ""):
                _ = (query, recency_days, domain_filter, k, volatile, query_intent)
                self.calls += 1
                return [
                    SearchResult(
                        title="RAG architecture best practices",
                        snippet="A short summary.",
                        url="https://example.com/rag-architecture",
                        source="example.com",
                        published_date="2026-03-08",
                    )
                ]

        class _StubScraper:
            def scrape(self, url: str):
                return SimpleNamespace(
                    title="RAG architecture best practices",
                    text="Detailed page body.",
                    status_code=200,
                    final_url=url,
                    metadata={"clean_method": "bs4", "removed_blocks": 1, "raw_len": 1000, "clean_len": 320},
                )

        search = _SpySearch()
        stage = WebRetrieveStage(search_client=search, scraper=_StubScraper(), cfg=WebRagConfig())
        ctx = SimpleNamespace(
            clean_user_msg="какие сейчас лучшие практики архитектуры RAG в 2026",
            user_msg="какие сейчас лучшие практики архитектуры RAG в 2026",
            state={"web_mode": "auto", "web_auto_profile": "balanced"},
            meta={"web_mode": "auto", "web_auto_profile": "balanced"},
            tags={"intent": "question"},
            logs=[],
            retrieved_memories=[],
            memory_context={},
        )
        out = stage.run(ctx)
        self.assertGreaterEqual(search.calls, 1)
        self.assertEqual(str(out.tags.get("web_used") or ""), "true")
        self.assertIn(str(out.tags.get("web_search_mode") or ""), {"TARGETED_SEARCH", "DEEP_SEARCH", "SOFT_SEARCH"})


if __name__ == "__main__":
    unittest.main()

