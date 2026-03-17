from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from threading import RLock
from types import SimpleNamespace

from core.brain import Brain
from metadata.metadata_extractor import extract_message_metadata
from memory.ingest_analyzer import analyze_message_for_memory
from memory.memory_manager import MemoryManager
from memory.memory_models import (
    FactRecordV2,
    LifecycleDecision,
    MemoryEvent,
    MemoryLevel,
    MemoryScope,
    MemorySourceKind,
    MemoryStatus,
    MemoryType,
)
from memory.memory_policy import MemoryPolicy
from memory.memory_scoring import SalienceWeights


class _FakeStore:
    def __init__(self) -> None:
        self.records = []

    def upsert(self, record) -> None:
        self.records.append(record)

    def iter_records(self, namespace: str | None = None):
        if namespace is None:
            return list(self.records)
        return [row for row in list(self.records) if str(getattr(row, "namespace", "")) == str(namespace)]


class _FakeEventStore:
    def __init__(self) -> None:
        self.entries = []

    def append(self, row) -> None:
        self.entries.append(dict(row or {}))


class _FakeFactExtractor:
    def extract_v2(self, *, text, metadata, speaker, scope, mode, analysis=None):
        if not str(text or "").strip():
            return []
        speaker_norm = str(speaker or "").strip().lower()
        if speaker_norm == "assistant":
            return [
                FactRecordV2(
                    subject="assistant",
                    predicate="environment_os",
                    value="windows",
                    scope=scope,
                    confidence=0.88,
                    importance=0.62,
                    evidence=str(text or ""),
                    source_event_id=str(metadata.get("event_id") or ""),
                    canonical_key="assistant.environment_os",
                    relation="environment",
                )
            ]
        return [
            FactRecordV2(
                subject=str(speaker or "assistant"),
                predicate="task",
                value="dummy_fact",
                scope=scope,
                confidence=0.88,
                importance=0.62,
                evidence=str(text or ""),
                source_event_id=str(metadata.get("event_id") or ""),
                canonical_key=f"{speaker}.task",
                relation="task",
            )
        ]


class _FakeLifecycle:
    def decide(self, record, *, now_ts):
        return LifecycleDecision(
            promote_to=MemoryLevel.L2_EPISODIC,
            reason="message_keep_working",
            route="default",
            decision_debug={"composite_score": 0.52, "composite_threshold": 0.47},
        )

    def resolve_conflict(self, old, new):
        raise AssertionError("Fact conflict resolution should not be called in these tests")


class MemoryWritePolicyTests(unittest.TestCase):
    def _manager(self) -> MemoryManager:
        manager = MemoryManager.__new__(MemoryManager)
        manager._cfg = SimpleNamespace()
        temp_dir = Path(tempfile.mkdtemp(prefix="mmis-memory-policy-"))
        manager._root = temp_dir
        manager._state_path = temp_dir / "manager_state.json"
        manager._lock = RLock()
        manager._store = _FakeStore()
        manager._event_store = _FakeEventStore()
        manager._fact_extractor = _FakeFactExtractor()
        manager._policy = MemoryPolicy()
        manager._lifecycle = _FakeLifecycle()
        manager._working_records = []
        manager._session_summary = ""
        manager._open_questions = []
        manager._current_decisions = []
        manager._active_preferences = []
        manager._private_runtime = {}
        manager._temporary_ttl_sec = 3600
        manager._private_runtime_ttl_sec = 900
        manager._working_limit = 120
        manager._importance_weights = {
            "base": 0.42,
            "decision": 0.24,
            "remember": 0.18,
            "project": 0.10,
        }
        manager._salience_weights = SalienceWeights(
            novelty=0.22,
            permanence=0.20,
            repetition=0.14,
            project_relevance=0.16,
            task_relevance=0.16,
            explicit_save_signal=0.12,
        )
        manager._save_state = lambda: None
        manager._cleanup_expired = lambda: None
        return manager

    def test_unverified_assistant_fx_claim_is_skipped(self) -> None:
        manager = self._manager()

        result = manager.ingest_event(
            MemoryEvent(
                role="assistant",
                text="Курс евро сегодня составляет около 46.20 UAH.",
                namespace="conv-1",
                scope=MemoryScope.CONVERSATION,
                memory_type=MemoryType.MESSAGE,
                metadata={},
            )
        )

        self.assertEqual(result.stored_ids, [])
        self.assertEqual(len(result.dropped_ids), 1)
        self.assertEqual(manager._store.records, [])
        self.assertEqual(manager._event_store.entries[-1]["type"], "memory_ingest_skipped_v2")
        self.assertEqual(
            manager._event_store.entries[-1]["payload"]["reason"],
            "assistant_factual_unverified_no_web",
        )

    def test_verified_assistant_fx_claim_is_temporary_only(self) -> None:
        manager = self._manager()

        result = manager.ingest_event(
            MemoryEvent(
                role="assistant",
                text="Курс доллара сегодня в Киеве составляет около 35.70 UAH по данным minfin.com.ua.",
                namespace="conv-2",
                scope=MemoryScope.CONVERSATION,
                memory_type=MemoryType.MESSAGE,
                metadata={
                    "web_used": True,
                    "web_query_intent": "fx_rate",
                    "web_primary_category": "finance",
                    "web_evidence_quality_score": 0.81,
                    "web_conflict_flags": [],
                },
            )
        )

        self.assertEqual(len(result.stored_ids), 1)
        self.assertEqual(result.promoted_ids, [])
        self.assertEqual(result.extracted_facts, [])
        stored = manager._store.records[0]
        self.assertEqual(stored.scope, MemoryScope.TEMPORARY)
        self.assertEqual(stored.metadata["assistant_write_policy"]["action"], "temporary_only")
        self.assertEqual(stored.metadata["assistant_write_policy"]["reason"], "assistant_factual_temporary_only")
        self.assertTrue(bool(stored.metadata.get("assistant_fact_records_blocked")))

    def test_conflicted_assistant_historical_claim_is_skipped(self) -> None:
        manager = self._manager()

        result = manager.ingest_event(
            MemoryEvent(
                role="assistant",
                text="В 2023 году EUR/UAH был около 41.20, по данным X.",
                namespace="conv-3",
                scope=MemoryScope.CONVERSATION,
                memory_type=MemoryType.MESSAGE,
                metadata={
                    "web_used": True,
                    "web_primary_category": "external",
                    "web_evidence_quality_score": 0.92,
                    "web_conflict_flags": ["numeric_conflict"],
                },
            )
        )

        self.assertEqual(result.stored_ids, [])
        self.assertEqual(len(result.dropped_ids), 1)
        self.assertEqual(manager._store.records, [])
        self.assertEqual(
            manager._event_store.entries[-1]["payload"]["reason"],
            "assistant_factual_conflicted_evidence",
        )

    def test_verified_attributed_historical_claim_can_be_saved_without_fact_records(self) -> None:
        manager = self._manager()

        result = manager.ingest_event(
            MemoryEvent(
                role="assistant",
                text="По данным NASA, Apollo 11 landed in 1969.",
                namespace="conv-4",
                scope=MemoryScope.CONVERSATION,
                memory_type=MemoryType.MESSAGE,
                metadata={
                    "web_used": True,
                    "web_primary_category": "external",
                    "web_evidence_quality_score": 0.87,
                    "web_conflict_flags": [],
                },
            )
        )

        self.assertEqual(len(result.stored_ids), 1)
        self.assertEqual(result.extracted_facts, [])
        stored = manager._store.records[0]
        self.assertEqual(stored.scope, MemoryScope.CONVERSATION)
        self.assertEqual(stored.metadata["assistant_write_policy"]["action"], "allow")
        self.assertFalse(stored.metadata["assistant_write_policy"]["allow_fact_records"])
        self.assertEqual(stored.metadata["source_kind"], "assistant_reply")

    def test_cautious_numeric_web_answer_is_skipped_from_long_term_memory(self) -> None:
        manager = self._manager()

        result = manager.ingest_event(
            MemoryEvent(
                role="assistant",
                text="Точный курс доллара сегодня 58.56 грн.",
                namespace="conv-5",
                scope=MemoryScope.CONVERSATION,
                memory_type=MemoryType.MESSAGE,
                metadata={
                    "web_used": True,
                    "web_query_intent": "fx_rate",
                    "web_primary_category": "finance",
                    "web_evidence_quality_score": 0.58,
                    "web_final_factual_confidence": 0.41,
                    "web_cautious_synthesis": True,
                    "web_conflict_severity": 0.92,
                    "web_conflict_flags": ["numeric_conflict"],
                },
            )
        )

        self.assertEqual(result.stored_ids, [])
        self.assertEqual(len(result.dropped_ids), 1)
        self.assertEqual(
            manager._event_store.entries[-1]["payload"]["reason"],
            "assistant_factual_conflicted_evidence",
        )

    def test_cautious_weather_factual_answer_is_skipped_from_long_term_memory(self) -> None:
        manager = self._manager()

        result = manager.ingest_event(
            MemoryEvent(
                role="assistant",
                text="Tomorrow will be exactly +15 degrees and dry.",
                namespace="conv-5b",
                scope=MemoryScope.CONVERSATION,
                memory_type=MemoryType.MESSAGE,
                metadata={
                    "web_used": True,
                    "web_query_intent": "weather",
                    "web_primary_category": "weather",
                    "web_factual_mode": "weather",
                    "web_evidence_quality_score": 0.57,
                    "web_final_factual_confidence": 0.39,
                    "web_cautious_synthesis": True,
                    "web_conflict_severity": 0.31,
                    "web_conflict_flags": [],
                },
            )
        )

        self.assertEqual(result.stored_ids, [])
        self.assertEqual(len(result.dropped_ids), 1)
        self.assertEqual(
            manager._event_store.entries[-1]["payload"]["reason"],
            "assistant_factual_cautious_synthesis",
        )

    def test_generic_assistant_reply_strips_thinking_and_blocks_fact_records(self) -> None:
        manager = self._manager()

        result = manager.ingest_event(
            MemoryEvent(
                role="assistant",
                text="Ок, давай сделаем это по шагам.",
                namespace="conv-generic",
                scope=MemoryScope.CONVERSATION,
                memory_type=MemoryType.MESSAGE,
                metadata={},
                thinking="сначала прикину план и проверю логику",
            )
        )

        self.assertEqual(len(result.stored_ids), 1)
        self.assertEqual(result.extracted_facts, [])
        stored = manager._store.records[0]
        self.assertEqual(stored.metadata["source_kind"], "assistant_reply")
        self.assertTrue(bool(stored.metadata.get("assistant_thinking_stripped")))
        self.assertTrue(bool(stored.metadata.get("assistant_fact_records_blocked")))
        self.assertNotIn("thinking", stored.metadata)

    def test_assistant_memory_miss_help_reply_is_temporary_only(self) -> None:
        manager = self._manager()

        result = manager.ingest_event(
            MemoryEvent(
                role="assistant",
                text="Не помню точную модель. Проверь сам через winver или python --version.",
                namespace="conv-memory-help",
                scope=MemoryScope.CONVERSATION,
                memory_type=MemoryType.MESSAGE,
                metadata={"memory_recall_mode": "exact_fact_recall"},
            )
        )

        self.assertEqual(len(result.stored_ids), 1)
        self.assertEqual(result.promoted_ids, [])
        stored = manager._store.records[0]
        self.assertEqual(stored.scope, MemoryScope.TEMPORARY)
        self.assertEqual(stored.metadata["assistant_write_policy"]["reason"], "assistant_memory_miss_help_temporary_only")
        self.assertEqual(stored.metadata["assistant_reply_kind"], "memory_miss_help")
        self.assertTrue(bool(stored.metadata.get("assistant_memory_help_noise")))

    def test_explicit_assistant_thought_event_is_skipped(self) -> None:
        manager = self._manager()

        result = manager.ingest_event(
            MemoryEvent(
                role="assistant",
                text="похоже пользователь на windows и надо проверить web",
                namespace="conv-thought",
                scope=MemoryScope.CONVERSATION,
                memory_type=MemoryType.MESSAGE,
                metadata={"source_kind": "assistant_thought"},
            )
        )

        self.assertEqual(result.stored_ids, [])
        self.assertEqual(len(result.dropped_ids), 1)
        self.assertEqual(manager._store.records, [])
        self.assertEqual(
            manager._event_store.entries[-1]["payload"]["reason"],
            "assistant_thought_not_persisted",
        )

    def test_direct_fact_write_policy_denies_assistant_environment_fact(self) -> None:
        manager = self._manager()

        written = manager._write_fact_records(
            facts=[
                FactRecordV2(
                    subject="assistant",
                    predicate="environment_os",
                    value="windows",
                    scope=MemoryScope.CONVERSATION,
                    confidence=0.9,
                    importance=0.7,
                    evidence="you are on windows",
                    source_event_id="evt:assistant-direct",
                    canonical_key="assistant.environment_os",
                    relation="environment",
                    metadata={"source_role": "assistant", "source_kind": "structured_fact"},
                )
            ],
            namespace="conv-direct-fact",
            now_ts=1.0,
            event_id="evt:assistant-direct",
            source_role="assistant",
            source_kind="assistant_reply",
        )

        self.assertEqual(written, [])
        self.assertEqual(
            [row for row in manager._store.records if getattr(row, "memory_type", None) == MemoryType.FACT],
            [],
        )

    def test_direct_fact_write_policy_allows_assistant_decision_fact(self) -> None:
        manager = self._manager()

        written = manager._write_fact_records(
            facts=[
                FactRecordV2(
                    subject="assistant",
                    predicate="decision",
                    value="store_search_text_and_canonical_text",
                    scope=MemoryScope.CONVERSATION,
                    confidence=0.9,
                    importance=0.82,
                    evidence="we will store search_text and canonical_text",
                    source_event_id="evt:assistant-decision",
                    canonical_key="assistant.decision",
                    relation="decision",
                    metadata={"source_role": "assistant", "source_kind": "structured_fact"},
                )
            ],
            namespace="conv-direct-decision",
            now_ts=1.0,
            event_id="evt:assistant-decision",
            source_role="assistant",
            source_kind="assistant_reply",
        )

        self.assertEqual(len(written), 1)
        fact_rows = [row for row in manager._store.records if getattr(row, "memory_type", None) == MemoryType.FACT]
        self.assertEqual(len(fact_rows), 1)
        stored_fact = dict(dict(fact_rows[0].metadata or {}).get("fact") or {})
        self.assertEqual(str(stored_fact.get("predicate") or ""), "decision")
        self.assertEqual(str(stored_fact.get("value") or ""), "store_search_text_and_canonical_text")

    def test_direct_fact_write_policy_supersedes_same_group_ram_fact(self) -> None:
        manager = self._manager()

        first = manager._write_fact_records(
            facts=[
                FactRecordV2(
                    subject="user",
                    predicate="environment_memory_gb",
                    value="16",
                    scope=MemoryScope.CONVERSATION,
                    confidence=0.78,
                    importance=0.7,
                    evidence="i have 16 gb memory",
                    source_event_id="evt:ram-1",
                    canonical_key="user.environment_memory_gb",
                    relation="environment",
                    metadata={"source_role": "user", "source_kind": "structured_fact"},
                )
            ],
            namespace="conv-direct-ram",
            now_ts=1.0,
            event_id="evt:ram-1",
            source_role="user",
            source_kind="structured_fact",
        )
        second = manager._write_fact_records(
            facts=[
                FactRecordV2(
                    subject="user",
                    predicate="environment_ram_gb",
                    value="32",
                    scope=MemoryScope.CONVERSATION,
                    confidence=0.89,
                    importance=0.78,
                    evidence="i have 32 gb ram",
                    source_event_id="evt:ram-2",
                    canonical_key="user.environment_ram_gb",
                    relation="environment",
                    metadata={"source_role": "user", "source_kind": "structured_fact"},
                )
            ],
            namespace="conv-direct-ram",
            now_ts=2.0,
            event_id="evt:ram-2",
            source_role="user",
            source_kind="structured_fact",
        )

        self.assertEqual(len(first), 1)
        self.assertEqual(len(second), 1)
        latest_by_id = {}
        for row in manager._store.records:
            if getattr(row, "memory_type", None) == MemoryType.FACT:
                latest_by_id[str(getattr(row, "id", ""))] = row
        final_rows = list(latest_by_id.values())
        active_facts = [row for row in final_rows if getattr(row, "status", None) == MemoryStatus.ACTIVE]
        superseded_facts = [row for row in final_rows if getattr(row, "status", None) == MemoryStatus.SUPERSEDED]
        self.assertEqual(len(active_facts), 1)
        self.assertEqual(len(superseded_facts), 1)
        self.assertEqual(
            str(dict(dict(active_facts[0].metadata or {}).get("fact") or {}).get("predicate") or ""),
            "environment_ram_gb",
        )
        self.assertEqual(
            str(dict(dict(superseded_facts[0].metadata or {}).get("fact") or {}).get("predicate") or ""),
            "environment_memory_gb",
        )

    def test_ingest_analysis_tags_are_stored_separately_from_runtime_tags(self) -> None:
        manager = self._manager()
        analysis = analyze_message_for_memory("I use RTX 3050 Ti with 4 GB VRAM.")

        merged = manager._merge_ingest_analysis_into_metadata(
            metadata={"tags": ["intent_chat", "lang_en"], "memory_tags": ["old_memory_tag"]},
            analysis=analysis,
        )

        self.assertEqual(merged["tags"], ["intent_chat", "lang_en"])
        self.assertIn("old_memory_tag", list(merged.get("memory_tags") or []))
        self.assertIn("topic_hardware", list(merged.get("memory_tags") or []))
        self.assertIn("entity_gpu_model_rtx_3050_ti", list(merged.get("memory_tags") or []))

    def test_sanitize_metadata_moves_legacy_entities_to_runtime_entities(self) -> None:
        policy = MemoryPolicy()

        sanitized = policy.sanitize_metadata_for_storage(
            metadata={
                "entities": {"software": ["Python"], "os": ["Windows"]},
                "tags": ["intent_chat"],
            },
            source_kind=MemorySourceKind.USER,
        )

        self.assertNotIn("entities", sanitized)
        self.assertEqual(
            sanitized.get("runtime_entities"),
            {"software": ["Python"], "os": ["Windows"]},
        )
        self.assertTrue(bool(sanitized.get("legacy_runtime_entities_stripped")))

    def test_brain_flatten_turn_metadata_uses_runtime_entities_not_entities(self) -> None:
        flattened = Brain._flatten_turn_metadata(
            {
                "lang": "en",
                "intent": {"label": "chat", "conf": 0.5},
                "emotion": {"label": "neutral"},
                "tags": ["topic_python"],
                "entities": {"software": ["Python"]},
                "meta": {},
            }
        )

        self.assertNotIn("entities", flattened)
        self.assertEqual(flattened.get("runtime_entities"), {"software": ["Python"]})

    def test_metadata_extractor_uses_runtime_entities_and_no_entity_tags(self) -> None:
        payload = extract_message_metadata("I use Python on Windows with Docker.", state={})

        self.assertIn("runtime_entities", payload)
        self.assertNotIn("entities", payload)
        tags = [str(x).strip().lower() for x in list(payload.get("tags") or []) if str(x).strip()]
        self.assertFalse(any(tag.startswith("entity_") for tag in tags))

    def test_brain_extracts_compact_web_verification_meta_for_assistant_write_policy(self) -> None:
        payload = Brain._assistant_memory_web_meta(
            {
                "web_used": True,
                "web_query_intent": "fx_rate",
                "web_evidence_quality": {
                    "score": 0.81,
                    "usable_results": 2,
                    "unique_domains": 2,
                    "trusted_count": 1,
                    "has_conflict": False,
                },
                "web_evidence_context": {
                    "conflicting_sources": True,
                    "conflict_notes": ["numeric_conflict:minfin=35.7 vs finance=36.5"],
                },
                "turn_log_summaries": {
                    "web_summary": {
                        "web_used": True,
                        "mode": "DEEP_SEARCH",
                        "issues": ["source_conflict"],
                        "policy": {"primary_category": "finance"},
                        "sources": {"scanned": 12, "selected": 8},
                        "evidence": {"count": 8},
                    }
                },
            }
        )

        self.assertTrue(payload["web_used"])
        self.assertEqual(payload["web_query_intent"], "fx_rate")
        self.assertEqual(payload["web_primary_category"], "finance")
        self.assertEqual(payload["web_sources_scanned"], 12)
        self.assertEqual(payload["web_evidence_count"], 8)
        self.assertIn("source_conflict", payload["web_conflict_flags"])
        self.assertIn("numeric_conflict", payload["web_conflict_flags"])
        self.assertEqual(payload["web_conflict_severity"], 0.0)

    def test_brain_extracts_extended_factual_safety_meta(self) -> None:
        payload = Brain._assistant_memory_web_meta(
            {
                "web_used": True,
                "web_query_intent": "fx_rate",
                "web_evidence_quality": {
                    "score": 0.54,
                    "usable_results": 2,
                    "unique_domains": 2,
                    "trusted_count": 1,
                    "has_conflict": True,
                    "selected_avg_quality": 0.49,
                    "topical_filtered_sources": 3,
                    "conflict_severity": 0.88,
                    "evidence_strength": 0.43,
                    "final_factual_confidence": 0.31,
                    "cautious_synthesis": True,
                    "numeric_candidates_selected": 2,
                    "numeric_candidates_rejected": 5,
                },
                "web_evidence_context": {
                    "conflicting_sources": True,
                    "conflict_notes": ["numeric_conflict_high:USD/UAH:minfin=41.2 vs finance=58.5"],
                },
                "turn_log_summaries": {
                    "web_summary": {
                        "web_used": True,
                        "mode": "DEEP_SEARCH",
                        "issues": ["source_conflict"],
                        "policy": {"primary_category": "finance"},
                        "sources": {"scanned": 12, "selected": 2},
                        "evidence": {"count": 2},
                    }
                },
            }
        )

        self.assertTrue(payload["web_cautious_synthesis"])
        self.assertEqual(payload["web_conflict_severity"], 0.88)
        self.assertEqual(payload["web_final_factual_confidence"], 0.31)
        self.assertEqual(payload["web_numeric_candidates_selected"], 2)
        self.assertEqual(payload["web_numeric_candidates_rejected"], 5)

    def test_brain_extracts_factual_mode_and_basis_meta(self) -> None:
        payload = Brain._assistant_memory_web_meta(
            {
                "factual_response_mode": "price",
                "web_used": True,
                "web_evidence_quality": {
                    "score": 0.66,
                    "selected_result_factual_page_type": "overview_page",
                    "type_mismatch_notes": ["type_mismatch:shop:overview_page"],
                    "true_conflict_notes": [],
                },
                "web_evidence_context": {
                    "factual_basis": {
                        "selected_result": {"domain": "shop.example", "page_type": "overview_page"},
                    },
                    "selected_result_factual_page_type": "overview_page",
                },
                "turn_log_summaries": {
                    "web_summary": {
                        "web_used": True,
                        "policy": {"primary_category": "price"},
                        "sources": {"scanned": 4, "selected": 1},
                        "evidence": {"count": 1},
                    }
                },
            }
        )

        self.assertEqual(payload["web_factual_mode"], "price")
        self.assertEqual(payload["web_selected_result_factual_page_type"], "overview_page")
        self.assertEqual(payload["web_type_mismatch_notes"], ["type_mismatch:shop:overview_page"])
        self.assertEqual(
            payload["web_factual_basis"]["selected_result"]["domain"],
            "shop.example",
        )


if __name__ == "__main__":
    unittest.main()
