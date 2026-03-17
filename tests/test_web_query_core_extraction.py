from __future__ import annotations

import unittest

from modules.internet.web.continuity import ContinuityConfig, build_continuity_patch, resolve_continuation
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
        query = "/web а найди мне какой курс доллара сегодня?"
        debug = analyze_search_text(query)

        self.assertEqual(debug["original_query"], query)
        self.assertEqual(debug["normalized_query"], "какой курс доллара сегодня")
        self.assertEqual(debug["extracted_search_core"], "какой курс доллара сегодня")

    def test_strips_service_prefixes_and_reports_removed_wrapper_text(self) -> None:
        text = "/mode_lock on /web какой курс доллара сегодня?"
        debug = analyze_search_text(text)

        self.assertEqual(debug["original_query"], text)
        self.assertEqual(debug["normalized_query"], "какой курс доллара сегодня")
        self.assertEqual(debug["extracted_search_core"], "какой курс доллара сегодня")
        self.assertIn("/mode_lock on /web", str(debug.get("removed_wrapper_text") or ""))

    def test_strips_mode_argument_prefix_from_search_core(self) -> None:
        text = "/mode engineer РЅР°Р№РґРё РІРµСЂСЃРёСЋ python 3.11"
        debug = analyze_search_text(text)

        self.assertNotIn("/mode", str(debug["normalized_query"]))
        self.assertNotIn("/mode", str(debug["extracted_search_core"]))
        self.assertIn("python 3.11", str(debug["normalized_query"]).lower())
        self.assertIn("python 3.11", str(debug["extracted_search_core"]).lower())
        self.assertIn("/mode engineer", str(debug.get("removed_wrapper_text") or ""))

    def test_continuity_patch_stores_sanitized_task_queries(self) -> None:
        patch = build_continuity_patch(
            state={},
            query="/mode_lock on /web РєСѓСЂСЃ РґРѕР»Р»Р°СЂР° СЃРµРіРѕРґРЅСЏ",
            resolved_intent="fx_rate",
            active_task={
                "task_id": "task_demo",
                "query": "/mode_lock on /web РєСѓСЂСЃ РґРѕР»Р»Р°СЂР° СЃРµРіРѕРґРЅСЏ",
                "base_query": "/mode_lock on /web РєСѓСЂСЃ РґРѕР»Р»Р°СЂР° СЃРµРіРѕРґРЅСЏ",
                "latest_query": "/mode_lock on /web РєСѓСЂСЃ РґРѕР»Р»Р°СЂР° СЃРµРіРѕРґРЅСЏ",
            },
        )

        active = dict(patch.get("web_active_task") or {})
        self.assertEqual(active.get("query"), "РєСѓСЂСЃ РґРѕР»Р»Р°СЂР° СЃРµРіРѕРґРЅСЏ")
        self.assertEqual(active.get("base_query"), "РєСѓСЂСЃ РґРѕР»Р»Р°СЂР° СЃРµРіРѕРґРЅСЏ")
        self.assertEqual(active.get("latest_query"), "РєСѓСЂСЃ РґРѕР»Р»Р°СЂР° СЃРµРіРѕРґРЅСЏ")

    def test_removes_model_argument_wrapper_from_search_core(self) -> None:
        text = "ты точно смотришь евро-гривны?"
        debug = analyze_search_text(text)

        self.assertEqual(debug["normalized_query"], "ты точно смотришь евро-гривны")
        self.assertEqual(debug["extracted_search_core"], "евро-гривны")
        self.assertIn("ты точно смотришь", str(debug.get("removed_wrapper_text") or ""))

    def test_planner_debug_includes_removed_wrapper_text(self) -> None:
        query = (
            "/mode_lock on /web "
            "ты опять врёшь, теперь ты точно можешь увидеть что интернет есть, "
            "так вот скажи наконец какой курс доллара сегодня?"
        )
        classification = classify_query(query)
        plan = build_query_plan(
            query=query,
            classification=classification,
            decision=self._decision(),
        )

        debug = dict(plan.debug or {}).get("query_text") or {}
        self.assertEqual(debug.get("extracted_search_core"), "какой курс доллара сегодня")
        self.assertIn("/mode_lock on /web", str(debug.get("removed_wrapper_text") or ""))
        self.assertEqual(plan.query_roles["primary"], ["какой курс доллара сегодня"])


    def test_self_memory_query_does_not_continue_old_price_task(self) -> None:
        state = {
            "web_active_task": {
                "task_id": "task_old_price",
                "query": "цена rtx 3050 сегодня",
                "base_query": "цена rtx 3050 сегодня",
                "latest_query": "цена rtx 3050 сегодня",
                "resolved_intent": "generic",
                "turn_timestamp": "2026-03-17T10:00:00+00:00",
            }
        }

        resolution = resolve_continuation(
            query="подскажи мою видеокарту",
            state=state,
            config=ContinuityConfig(ttl_minutes=999, max_user_turns=99),
        )

        self.assertFalse(resolution.used)
        self.assertEqual(resolution.resolved_query, "мою видеокарту")
        self.assertEqual(resolution.base_query, "мою видеокарту")
        self.assertEqual(resolution.reason, "self_memory_query")

    def test_followup_shaped_self_memory_query_still_breaks_price_continuation(self) -> None:
        state = {
            "web_active_task": {
                "task_id": "task_old_price",
                "query": "цена rtx 3050 сегодня",
                "base_query": "цена rtx 3050 сегодня",
                "latest_query": "цена rtx 3050 сегодня",
                "resolved_intent": "generic",
                "turn_timestamp": "2026-03-17T10:00:00+00:00",
            }
        }

        resolution = resolve_continuation(
            query="а моя видеокарта?",
            state=state,
            config=ContinuityConfig(ttl_minutes=999, max_user_turns=99),
        )

        self.assertFalse(resolution.used)
        self.assertEqual(resolution.resolved_query, "моя видеокарта")
        self.assertEqual(resolution.base_query, "моя видеокарта")
        self.assertEqual(resolution.reason, "self_memory_query")


if __name__ == "__main__":
    unittest.main()
