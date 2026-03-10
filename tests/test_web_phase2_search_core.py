from __future__ import annotations

from _output_utils import enable_unittest_json_output

enable_unittest_json_output()

import unittest
from types import SimpleNamespace

from modules.internet.search import SearchResult
from modules.internet.web.freshness import assess_freshness
from modules.internet.web.query_classifier import classify_query
from modules.internet.web.query_planner import build_query_plan
from modules.internet.web.result_processor import build_evidence_pack
from modules.internet.web.search_manager import SearchManager
from modules.internet.web.source_ranker import RankedSource, rank_sources
from modules.internet.web.web_models import SearchBudget, WebPolicyDecision, WebSearchMode


class WebPhase2SearchCoreTests(unittest.TestCase):
    def _decision(
        self,
        *,
        mode: WebSearchMode,
        max_queries: int,
        max_sources: int,
        max_fetches: int,
    ) -> WebPolicyDecision:
        return WebPolicyDecision(
            mode=mode,
            should_search=True,
            web_need_score=0.8,
            reason="test",
            budget=SearchBudget(
                max_queries=max_queries,
                max_sources=max_sources,
                max_pages=max_fetches,
                max_fetches=max_fetches,
            ),
        )

    def test_query_planner_generates_role_queries(self) -> None:
        query = "latest chromadb version release notes"
        classification = classify_query(query)
        decision = self._decision(
            mode=WebSearchMode.TARGETED_SEARCH,
            max_queries=6,
            max_sources=5,
            max_fetches=3,
        )
        plan = build_query_plan(
            query=query,
            classification=classification,
            decision=decision,
            preferred_domains=["github.com", "docs.trychroma.com"],
        )
        roles = set(plan.query_roles.keys())
        self.assertIn("primary", roles)
        self.assertIn("validation", roles)
        self.assertIn("release_notes", roles)
        self.assertIn("exact_entity", roles)
        self.assertIn("domain_constrained", roles)
        self.assertTrue(any("site:github.com" in q for q in plan.all_queries()))
        self.assertTrue(plan.scout_queries)

    def test_source_ranker_trust_ordering(self) -> None:
        ranked = rank_sources(
            items=[
                SearchResult(
                    title="Forum opinion",
                    snippet="I think this might work",
                    url="https://reddit.com/r/python/comments/123",
                    source="reddit.com",
                ),
                SearchResult(
                    title="Official docs page",
                    snippet="API reference",
                    url="https://docs.python.org/3/library/pathlib.html",
                    source="docs.python.org",
                ),
                SearchResult(
                    title="Random blog post",
                    snippet="My personal notes",
                    url="https://example-blog.com/post",
                    source="example-blog.com",
                ),
            ]
        )
        self.assertGreaterEqual(len(ranked), 3)
        self.assertEqual(str(ranked[0].trust_tier), "official_docs")
        tiers = [str(x.trust_tier) for x in ranked]
        self.assertIn("forum_discussion", tiers)

    def test_bounded_search_manager_uses_two_stage_and_limits(self) -> None:
        class _FakeSearch:
            def __init__(self):
                self.calls: list[str] = []
                self.fail_once: dict[str, bool] = {"validation": True}

            def search(self, query: str, recency_days=None, domain_filter=None, k: int = 5, volatile: bool = False, query_intent: str = ""):
                _ = (recency_days, domain_filter, k, volatile, query_intent)
                self.calls.append(str(query))
                if "validation" in str(query).lower() and self.fail_once["validation"]:
                    self.fail_once["validation"] = False
                    raise RuntimeError("transient")
                if "primary" in str(query).lower():
                    return [
                        SearchResult(
                            title="Primary result",
                            snippet="value 42.8",
                            url="https://source-a.example/1",
                            source="source-a.example",
                            published_date="2026-03-10",
                        )
                    ]
                if "validation" in str(query).lower():
                    return [
                        SearchResult(
                            title="Validation result",
                            snippet="value 43.1",
                            url="https://source-b.example/2",
                            source="source-b.example",
                            published_date="2026-03-10",
                        )
                    ]
                return []

        class _FakeScraper:
            def scrape(self, url: str):
                return SimpleNamespace(
                    title="Scraped page",
                    text=f"Body for {url}",
                    status_code=200,
                    final_url=url,
                    metadata={"clean_method": "bs4", "removed_blocks": 1, "raw_len": 500, "clean_len": 120},
                )

        manager = SearchManager(
            search_client=_FakeSearch(),
            scraper=_FakeScraper(),
            cooldown_seconds=45,
            retry_policy={"default_attempts": 1, "deep_search": 2, "backoff_ms": 10},
        )
        decision = self._decision(
            mode=WebSearchMode.DEEP_SEARCH,
            max_queries=3,
            max_sources=2,
            max_fetches=1,
        )
        classification = classify_query("latest release notes for product")
        freshness = assess_freshness(query="latest release notes for product", classification=classification)

        result = manager.execute(
            plan=SimpleNamespace(
                mode=WebSearchMode.DEEP_SEARCH,
                scout_queries=["primary query"],
                focused_queries=["validation query"],
                fallback_queries=["fallback query"],
                all_queries=lambda: ["primary query", "validation query", "fallback query"],
            ),
            decision=decision,
            classification=classification,
            freshness=freshness,
        )

        self.assertLessEqual(len(result.queries_used), 3)
        self.assertLessEqual(len(result.results), 2)
        self.assertLessEqual(len(result.fetched_pages), 1)
        self.assertIn("primary query", result.scout_queries_used)
        self.assertTrue(result.focused_queries_used)
        self.assertGreaterEqual(int(result.retries_used), 1)

    def test_bounded_search_manager_cooldown_signal(self) -> None:
        class _Search:
            def search(self, query: str, recency_days=None, domain_filter=None, k: int = 5, volatile: bool = False, query_intent: str = ""):
                _ = (query, recency_days, domain_filter, k, volatile, query_intent)
                return [
                    SearchResult(
                        title="Result",
                        snippet="value 10",
                        url="https://a.example/page",
                        source="a.example",
                        published_date="2026-03-10",
                    )
                ]

        class _Scraper:
            def scrape(self, url: str):
                return SimpleNamespace(
                    title="Page",
                    text="Body text.",
                    status_code=200,
                    final_url=url,
                    metadata={"clean_method": "bs4"},
                )

        manager = SearchManager(search_client=_Search(), scraper=_Scraper(), cooldown_seconds=3600)
        decision = self._decision(mode=WebSearchMode.SOFT_SEARCH, max_queries=3, max_sources=3, max_fetches=1)
        classification = classify_query("latest sdk version")
        freshness = assess_freshness(query="latest sdk version", classification=classification)
        plan = SimpleNamespace(
            mode=WebSearchMode.SOFT_SEARCH,
            scout_queries=["sdk primary"],
            focused_queries=["sdk focused"],
            fallback_queries=[],
            all_queries=lambda: ["sdk primary", "sdk focused"],
        )
        first = manager.execute(plan=plan, decision=decision, classification=classification, freshness=freshness)
        second = manager.execute(plan=plan, decision=decision, classification=classification, freshness=freshness)
        self.assertFalse(first.cooldown_applied)
        self.assertTrue(second.cooldown_applied)

    def test_result_processor_builds_compact_evidence_pack(self) -> None:
        decision = self._decision(mode=WebSearchMode.VERIFY_ONLY, max_queries=1, max_sources=3, max_fetches=2)
        classification = classify_query("latest usd uah exchange rate")
        ranked = [
            RankedSource(
                item=SearchResult(
                    title="Rate update A",
                    snippet="USD/UAH is 42.8 today",
                    url="https://source-a.example/rate",
                    source="source-a.example",
                    published_date="2026-03-10",
                    score=0.8,
                ),
                trust_score=0.24,
                trust_tier="vendor_docs",
                bonus=0.0,
                quality_score=0.4,
                freshness_score=0.05,
            ),
            RankedSource(
                item=SearchResult(
                    title="Rate update B",
                    snippet="USD/UAH is 55.1 today",
                    url="https://source-b.example/rate",
                    source="source-b.example",
                    published_date="2026-03-10",
                    score=0.7,
                ),
                trust_score=0.18,
                trust_tier="reputable_tech",
                bonus=0.0,
                quality_score=0.35,
                freshness_score=0.05,
            ),
            RankedSource(
                item=SearchResult(
                    title="Rate update A duplicate",
                    snippet="USD/UAH is 42.8 today",
                    url="https://source-a.example/rate?utm_source=test",
                    source="source-a.example",
                    published_date="2026-03-10",
                    score=0.6,
                ),
                trust_score=0.24,
                trust_tier="vendor_docs",
                bonus=0.0,
                quality_score=0.3,
                freshness_score=0.05,
            ),
        ]
        fetched_pages = {
            "https://source-a.example/rate": {
                "text": "Official statement: USD/UAH 42.8.",
                "fetched_at": "2026-03-10T08:00:00+00:00",
                "clean_method": "bs4",
            },
            "https://source-b.example/rate": {
                "text": "Market note: USD/UAH 55.1.",
                "fetched_at": "2026-03-10T08:10:00+00:00",
                "clean_method": "bs4",
            },
        }

        pack = build_evidence_pack(
            ranked_results=ranked,
            fetched_pages=fetched_pages,
            decision=decision,
            classification=classification,
        )
        self.assertTrue(pack.items)
        self.assertLessEqual(len(pack.items), 3)
        self.assertTrue(pack.key_facts)
        self.assertTrue(pack.trust_hints)
        self.assertTrue(pack.freshness_summary)
        self.assertTrue(pack.conflicting_sources)


if __name__ == "__main__":
    unittest.main()
