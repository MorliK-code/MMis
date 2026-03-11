from __future__ import annotations

import datetime as dt
import unittest

from _output_utils import enable_unittest_json_output

enable_unittest_json_output()

from config.settings import AppSettings, _validate_settings
from modules.internet.web.confidence import assess_confidence
from modules.internet.web.freshness import assess_freshness
from modules.internet.web.query_classifier import classify_query
from modules.internet.web.web_models import SearchBudget, WebEvidence, WebSearchMode, WebSearchRequest, WebSearchResult
from modules.internet.web.web_policy import WebPolicyEngine, config_from_dict


class WebPhase1FoundationTests(unittest.TestCase):
    def test_web_models_contract_has_request_result_and_evidence(self) -> None:
        request = WebSearchRequest(
            query="latest chromadb release",
            mode=WebSearchMode.TARGETED_SEARCH,
            budget=SearchBudget(max_queries=3, max_sources=5, max_pages=2),
            preferred_domains=["github.com"],
        )
        result = WebSearchResult(
            request=request,
            evidence=[
                WebEvidence(
                    title="Release notes",
                    url="https://github.com/chroma-core/chroma/releases",
                    domain="github.com",
                    snippet="Latest stable release information.",
                )
            ],
            queries_used=["chromadb release notes"],
            success=True,
        )

        payload = result.to_dict()
        self.assertEqual(payload["request"]["mode"], WebSearchMode.TARGETED_SEARCH.value)
        self.assertEqual(payload["queries_used"], ["chromadb release notes"])
        self.assertEqual(payload["evidence"][0]["domain"], "github.com")

    def test_query_classifier_contract_fields_present(self) -> None:
        classification = classify_query("latest OpenAI API pricing in 2026")
        expected_fields = (
            "query_type",
            "is_temporal",
            "is_local_project_question",
            "is_external_fact_question",
            "is_ambiguous",
            "requires_freshness",
            "stakes_level",
            "expected_search_need",
        )
        for field in expected_fields:
            self.assertTrue(hasattr(classification, field), field)
        self.assertTrue(classification.requires_freshness)

    def test_confidence_assessment_uses_phase1_factors(self) -> None:
        now = dt.datetime.now(dt.timezone.utc)
        classification = classify_query("latest chromadb version for our project")
        assessment = assess_confidence(
            query="latest chromadb version for our project",
            classification=classification,
            retrieved_memories=[
                {
                    "text": "Our project uses chromadb in memory backend.",
                    "source": "document",
                    "memory_type": "document_chunk",
                    "metadata": {"entities": {"product": ["chromadb", "backend"]}},
                },
                {
                    "text": "Older note about vector backend.",
                    "source": "memory",
                    "status": "superseded",
                    "metadata": {"fact": {"subject": "project", "predicate": "backend", "value": "json"}},
                },
                {
                    "text": "[web] latest release note",
                    "source": "web",
                    "fetched_at": (now - dt.timedelta(hours=2)).isoformat(),
                },
            ],
        )

        keys = set(assessment.breakdown.keys())
        self.assertIn("memory_hits_bonus", keys)
        self.assertIn("local_docs_bonus", keys)
        self.assertIn("entity_overlap_bonus", keys)
        self.assertIn("conflicts_penalty", keys)
        self.assertIn("fresh_web_cache_bonus", keys)
        self.assertIn("query_specificity_bonus", keys)
        self.assertTrue(0.0 <= float(assessment.score) <= 1.0)

    def test_freshness_assessment_detects_stale_web_fact(self) -> None:
        classification = classify_query("latest chromadb version")
        assessed = assess_freshness(
            query="latest chromadb version",
            classification=classification,
            web_items=[{"published_date": "2025-01-01", "status": "ok"}],
            ttl_days={"default": 30, "versions": 14},
            now_utc=dt.datetime(2026, 3, 10, tzinfo=dt.timezone.utc),
        )
        self.assertTrue(assessed.stale_web_fact_detected)
        self.assertTrue(assessed.needs_refresh)
        self.assertTrue(0.0 <= float(assessed.temporal_risk) <= 1.0)

    def test_policy_engine_supports_user_overrides(self) -> None:
        engine = WebPolicyEngine(config_from_dict({}))
        query = "explain local architecture for memory subsystem"
        classification = classify_query(query)
        confidence = assess_confidence(query=query, classification=classification, retrieved_memories=[])
        freshness = assess_freshness(query=query, classification=classification)

        no_web = engine.decide(
            query=query,
            classification=classification,
            confidence=confidence,
            freshness=freshness,
            web_mode="auto",
            web_auto_profile="balanced",
            internet_enabled=True,
            user_override="no-web",
        )
        self.assertEqual(no_web.mode, WebSearchMode.NO_SEARCH)
        self.assertEqual(no_web.reason, "user_override_no_web")

        force_web = engine.decide(
            query=query,
            classification=classification,
            confidence=confidence,
            freshness=freshness,
            web_mode="auto",
            web_auto_profile="balanced",
            internet_enabled=True,
            user_override="/web",
        )
        self.assertNotEqual(force_web.mode, WebSearchMode.NO_SEARCH)
        self.assertEqual(force_web.reason, "user_override_web")

    def test_web_v2_settings_validation(self) -> None:
        valid = AppSettings(
            web_v2={
                "enabled": True,
                "thresholds": {
                    "no_search_max": 0.25,
                    "verify_max": 0.45,
                    "soft_max": 0.65,
                    "targeted_max": 0.85,
                },
                "budgets": {
                    "verify_only": {"max_queries": 1, "max_sources": 2, "max_pages": 1},
                },
                "cooldown_seconds": 30,
                "force_search_keywords": ["latest", "today"],
                "preferred_domains": ["openai.com"],
                "blocked_domains": ["example.invalid"],
                "ttl_days": {"default": 7, "news": 2},
            }
        )
        _validate_settings(valid)

        invalid = AppSettings(
            web_v2={
                "thresholds": {
                    "no_search_max": 0.9,
                    "verify_max": 0.4,
                    "soft_max": 0.65,
                    "targeted_max": 0.8,
                },
                "budgets": {"soft_search": {"max_queries": -1}},
                "cooldown_seconds": -5,
                "ttl_days": {"default": 0},
            }
        )
        with self.assertRaises(ValueError):
            _validate_settings(invalid)


if __name__ == "__main__":
    unittest.main()
