from __future__ import annotations

import unittest

from modules.internet.web.query_classifier import classify_query
from modules.internet.web.query_planner import build_query_plan
from modules.internet.web.query_text import analyze_search_text, extract_search_core, normalize_search_text
from modules.internet.web.web_models import SearchBudget, WebPolicyDecision, WebSearchMode


class WebQueryCoreExtractionTests(unittest.TestCase):
    def _decision(self) -> WebPolicyDecision:
        return WebPolicyDecision(
            mode=WebSearchMode.TARGETED_SEARCH,
            should_search=True,
            web_need_score=0.85,
            reason="test",
            budget=SearchBudget(
                max_queries=6,
                max_sources=6,
                max_pages=3,
                max_fetches=3,
            ),
        )

    def test_prefers_final_question_clause_over_long_intro(self) -> None:
        text = (
            "ты же видишь интернет, значит уже можно наконец проверить "
            "какой курс доллара сегодня?"
        )

        self.assertEqual(normalize_search_text(text), "какой курс доллара сегодня")
        self.assertEqual(extract_search_core(text), "какой курс доллара сегодня")

    def test_splits_sentence_and_takes_last_question(self) -> None:
        text = "теперь ты точно можешь увидеть, что интернет есть. какой курс доллара сегодня?"

        self.assertEqual(normalize_search_text(text), "какой курс доллара сегодня")
        self.assertEqual(extract_search_core(text), "какой курс доллара сегодня")

    def test_short_query_stays_unchanged(self) -> None:
        text = "курс доллара сегодня"

        self.assertEqual(normalize_search_text(text), text)
        self.assertEqual(extract_search_core(text), text)

    def test_planner_exposes_query_text_debug(self) -> None:
        query = "теперь ты точно можешь увидеть, что интернет есть. какой курс доллара сегодня?"
        classification = classify_query(query)
        plan = build_query_plan(
            query=query,
            classification=classification,
            decision=self._decision(),
        )

        debug = dict(plan.debug or {}).get("query_text") or {}
        self.assertEqual(debug.get("original_query"), query)
        self.assertIn("какой курс доллара сегодня", str(debug.get("normalized_query") or ""))
        self.assertEqual(debug.get("extracted_search_core"), "какой курс доллара сегодня")
        self.assertEqual(plan.query_roles["primary"], ["какой курс доллара сегодня"])

    def test_analyze_search_text_strips_web_command_but_keeps_original(self) -> None:
        debug = analyze_search_text("/web а найди мне какой курс доллара сегодня?")

        self.assertEqual(debug["original_query"], "/web а найди мне какой курс доллара сегодня?")
        self.assertEqual(debug["normalized_query"], "какой курс доллара сегодня")
        self.assertEqual(debug["extracted_search_core"], "какой курс доллара сегодня")


if __name__ == "__main__":
    unittest.main()
