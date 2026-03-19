from __future__ import annotations

import tempfile
from pathlib import Path
from threading import RLock
from types import SimpleNamespace

from memory.anchor_extractor import extract_anchors
from memory.claim_candidate_extractor import extract_claim_candidates
from memory.claim_promoter import promote_claim_candidates
from memory.claim_models import ClaimCandidate
from memory.claim_normalizer import normalize_claim_object
from memory.claim_promotion_policy import decide_claim_promotion_level
from memory.claim_validator import validate_claim_candidate
from memory.fact_extractor import FactExtractor
from memory.governor import MemoryGovernor
from memory.ingest_analyzer import analyze_message_for_memory
from memory.memory_lifecycle import MemoryLifecycleManager
from memory.memory_manager import MemoryManager
from memory.memory_models import (
    ConflictDecision,
    LifecycleDecision,
    MemoryEvent,
    MemoryRecord,
    MemoryScope,
    MemoryType,
)
from memory.memory_policy import MemoryPolicy
from memory.memory_scoring import SalienceWeights
from memory.numeric_extractor import extract_numeric_facts
from memory.entity_resolver import resolve_entities
from memory.contextual_resolver import resolve_anchor_context
from memory.retrieval_projection import build_memory_views


class _ClaimStore:
    def __init__(self) -> None:
        self._rows: dict[str, MemoryRecord] = {}
        self._order: list[str] = []

    def upsert(self, record: MemoryRecord) -> None:
        if str(record.id) not in self._rows:
            self._order.append(str(record.id))
        self._rows[str(record.id)] = record

    def iter_records(self, namespace: str | None = None) -> list[MemoryRecord]:
        rows = [self._rows[row_id] for row_id in list(self._order)]
        if namespace is None:
            return rows
        return [row for row in rows if str(row.namespace or "") == str(namespace or "")]

    def close(self) -> None:
        return None


class _ClaimEventStore:
    def __init__(self) -> None:
        self.entries: list[dict] = []

    def append(self, row) -> None:  # noqa: ANN001
        self.entries.append(dict(row or {}))


class _ClaimLifecycle:
    def __init__(self) -> None:
        self._delegate = MemoryLifecycleManager()

    def decide(self, record: MemoryRecord, *, now_ts: float) -> LifecycleDecision:
        _ = (record, now_ts)
        return LifecycleDecision(reason="claim_test_no_promotion", route="claim_test")

    def resolve_conflict(self, *, old: MemoryRecord, new: MemoryRecord) -> ConflictDecision:
        return self._delegate.resolve_conflict(old=old, new=new)


def _manager() -> MemoryManager:
    manager = MemoryManager.__new__(MemoryManager)
    manager._cfg = SimpleNamespace()
    temp_dir = Path(tempfile.mkdtemp(prefix="mmis-memory-claims-"))
    manager._root = temp_dir
    manager._state_path = temp_dir / "manager_state.json"
    manager._lock = RLock()
    manager._storage_profile = "compact"
    manager._DEBUG_METADATA_FIELDS = {
        "web_used", "web_factual_mode", "web_query_intent", "web_search_mode",
        "web_primary_category", "web_evidence_quality_score", "web_evidence_quality",
        "web_sources_scanned", "web_sources_selected", "web_evidence_count",
        "web_conflicting_sources", "web_low_evidence_quality", "web_selected_avg_quality",
        "web_topical_filtered_sources", "web_conflict_severity", "web_evidence_strength",
        "web_final_factual_confidence", "web_cautious_synthesis",
        "web_numeric_candidates_selected", "web_numeric_candidates_rejected",
        "_persona_snapshot_debug", "persona_snapshot",
        "promotion_project_signal", "promotion_task_signal", "promotion_decision_signal",
        "promotion_preference_signal", "promotion_issue_signal", "promotion_technical_signal",
        "promotion_repeated_topic_signal", "promotion_smalltalk_signal",
        "promotion_signal_score", "promotion_stable_fact_signal", "promotion_fact_signal",
        "promotion_fact_relation_diversity",
        "extracted_facts_count", "extracted_fact_relations",
        "memory_analysis", "memory_views_debug",
        "lifecycle_decision", "decision_debug",
        "assistant_write_policy", "assistant_write_blocked", "assistant_write_reason",
    }
    manager._store = _ClaimStore()
    manager._event_store = _ClaimEventStore()
    manager._fact_extractor = FactExtractor()
    manager._policy = MemoryPolicy()
    manager._lifecycle = _ClaimLifecycle()
    manager._governor = MemoryGovernor(lifecycle=manager._lifecycle)
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


def _extract_candidates(text: str):
    anchors = extract_anchors(text)
    context = resolve_anchor_context(text, anchors=anchors).to_dict()
    entities = resolve_entities(text, anchors=anchors, context=context)
    numeric_facts = extract_numeric_facts(
        text,
        entities=entities,
        anchors=anchors,
        context=context,
    )
    candidates = extract_claim_candidates(
        text,
        entities=entities,
        numeric_facts=numeric_facts,
        anchors=anchors,
        context=context,
    )
    return candidates


def _first_candidate(candidates, predicate: str):
    for item in list(candidates or []):
        if str(item.predicate or "").strip().lower() == str(predicate or "").strip().lower():
            return item
    raise AssertionError(f"claim candidate {predicate!r} not found")


def test_ingest_analysis_extracts_four_core_claim_groups() -> None:
    analysis = analyze_message_for_memory(
        "Не люблю шум. Люблю смотреть в глаза Розе. У меня холодильник Samsung. Я использую VS Code."
    )

    predicates = {str(item.predicate or "") for item in list(analysis.claim_candidates or [])}
    assert predicates == {"likes", "dislikes", "owns", "uses"}


def test_claim_candidate_extractor_bounds_like_phrase_and_builds_alternatives() -> None:
    candidates = _extract_candidates("Я люблю смотреть в глаза Розе, а потом молчать.")

    item = _first_candidate(candidates, "likes")
    assert item.object_surface == "смотреть в глаза Розе"
    assert "потом" not in item.object_surface.lower()
    assert "Розе" in list(item.alternatives or [])


def test_claim_candidate_extractor_keeps_cleaned_sensory_like_phrase() -> None:
    candidates = _extract_candidates("Мне нравится запах цветка лотоса.")

    predicates = [str(item.predicate or "") for item in list(candidates or [])]
    assert predicates == ["likes"]

    item = candidates[0]
    assert item.object_surface == "запах цветка лотоса"
    assert item.object_type == "sensory_object"
    assert "лотоса" in list(item.alternatives or [])


def test_claim_normalizer_trims_discourse_tail_and_canonicalizes_windows() -> None:
    item = normalize_claim_object("windows 11. кстати")

    assert item is not None
    assert item.object_surface == "Windows 11"
    assert item.normalized_object == "windows 11"


def test_claim_normalizer_rejects_stopword_only_objects() -> None:
    assert normalize_claim_object("что") is None
    assert normalize_claim_object("ну это") is None


def test_claim_validator_rejects_empty_or_stopword_claim_object() -> None:
    empty = ClaimCandidate(
        subject="user",
        predicate="likes",
        object_surface="",
        normalized_object="",
        confidence=0.84,
        evidence_text="like",
        topic_keys=["likes", "preference"],
    )
    stopword = ClaimCandidate(
        subject="user",
        predicate="likes",
        object_surface="that",
        normalized_object="that",
        confidence=0.84,
        evidence_text="I like that",
        topic_keys=["likes", "preference"],
    )

    assert validate_claim_candidate(empty).reason == "empty_object"
    assert validate_claim_candidate(stopword).reason == "stopword_only_object"


def test_claim_validator_rejects_bad_predicate_low_confidence_and_missing_evidence() -> None:
    bad_predicate = ClaimCandidate(
        subject="user",
        predicate="prefers",
        object_surface="dark theme",
        normalized_object="dark theme",
        confidence=0.90,
        evidence_text="prefer dark theme",
        topic_keys=["preference"],
    )
    low_conf = ClaimCandidate(
        subject="user",
        predicate="uses",
        object_surface="VS Code",
        normalized_object="vs code",
        confidence=0.30,
        evidence_text="I use VS Code",
        topic_keys=["uses", "tool", "usage"],
    )
    missing_evidence = ClaimCandidate(
        subject="user",
        predicate="owns",
        object_surface="fridge samsung",
        normalized_object="fridge samsung",
        confidence=0.80,
        evidence_text="",
        topic_keys=["owns", "appliance", "ownership"],
    )

    assert validate_claim_candidate(bad_predicate).reason == "invalid_predicate"
    assert validate_claim_candidate(low_conf).reason == "low_confidence"
    assert validate_claim_candidate(missing_evidence).reason == "missing_evidence_text"


def test_claim_validator_rejects_clause_explosion_and_missing_topics() -> None:
    clause = ClaimCandidate(
        subject="user",
        predicate="likes",
        object_surface="long thing because it is complicated and when I am tired if needed I keep thinking about it constantly",
        normalized_object="long thing because it is complicated and when i am tired if needed i keep thinking about it constantly",
        confidence=0.84,
        evidence_text="I like long thing because it is complicated and when I am tired if needed I keep thinking about it constantly",
        topic_keys=["likes", "preference"],
    )
    no_topics = ClaimCandidate(
        subject="user",
        predicate="uses",
        object_surface="VS Code",
        normalized_object="vs code",
        confidence=0.80,
        evidence_text="I use VS Code",
        topic_keys=[],
    )

    assert validate_claim_candidate(clause).reason == "clause_explosion"
    assert validate_claim_candidate(no_topics).reason == "missing_topic_keys"


def test_claim_promotion_policy_returns_weak_claim_for_soft_preference() -> None:
    candidate = ClaimCandidate(
        subject="user",
        predicate="likes",
        object_surface="запах цветка лотоса",
        object_type="sensory_object",
        head="запах",
        normalized_object="запах цветка лотоса",
        alternatives=["запах цветка лотоса", "лотоса"],
        confidence=0.84,
        specificity=0.85,
        evidence_text="мне нравится запах цветка лотоса",
        topic_keys=["likes", "sensory_object", "preference"],
        trigger_keys=["likes", "запах", "лотоса"],
    )

    decision = decide_claim_promotion_level(candidate)
    assert decision.action == "weak_claim"
    assert decision.allow_write is True


def test_claim_promotion_policy_returns_strong_claim_for_explicit_person_like() -> None:
    candidate = ClaimCandidate(
        subject="user",
        predicate="likes",
        object_surface="смотреть в глаза Розе",
        object_type="activity",
        head="смотреть",
        normalized_object="смотреть в глаза розе",
        alternatives=["смотреть в глаза Розе", "Розе", "глаза розе"],
        confidence=0.84,
        specificity=1.0,
        evidence_text="я люблю смотреть в глаза Розе",
        topic_keys=["likes", "activity", "preference"],
        trigger_keys=["likes", "смотреть", "розе"],
    )

    decision = decide_claim_promotion_level(candidate)
    assert decision.action == "strong_claim"
    assert decision.allow_write is True


def test_claim_promotion_policy_returns_episodic_only_for_weak_contextual_preference() -> None:
    candidate = ClaimCandidate(
        subject="user",
        predicate="likes",
        object_surface="мягкий свет вечером",
        object_type="thing",
        head="мягкий",
        normalized_object="мягкий свет вечером",
        alternatives=["мягкий свет вечером"],
        confidence=0.68,
        specificity=0.34,
        evidence_text="мне нравится мягкий свет вечером",
        topic_keys=["likes", "preference"],
        trigger_keys=["likes", "свет"],
    )

    decision = decide_claim_promotion_level(candidate)
    assert decision.action == "episodic_only"
    assert decision.allow_write is False


def test_promote_claim_candidates_skips_episodic_only_and_marks_levels() -> None:
    weak_candidate = ClaimCandidate(
        subject="user",
        predicate="likes",
        object_surface="запах цветка лотоса",
        object_type="sensory_object",
        head="запах",
        normalized_object="запах цветка лотоса",
        alternatives=["запах цветка лотоса", "лотоса"],
        confidence=0.84,
        specificity=0.85,
        evidence_text="мне нравится запах цветка лотоса",
        topic_keys=["likes", "sensory_object", "preference"],
        trigger_keys=["likes", "запах", "лотоса"],
    )
    episodic_candidate = ClaimCandidate(
        subject="user",
        predicate="likes",
        object_surface="мягкий свет вечером",
        object_type="thing",
        head="мягкий",
        normalized_object="мягкий свет вечером",
        alternatives=["мягкий свет вечером"],
        confidence=0.68,
        specificity=0.34,
        evidence_text="мне нравится мягкий свет вечером",
        topic_keys=["likes", "preference"],
        trigger_keys=["likes", "свет"],
    )

    records = promote_claim_candidates(
        [weak_candidate, episodic_candidate],
        event_id="evt:test",
        namespace="claims",
    )

    assert len(records) == 1
    assert records[0].promotion_level == "weak_claim"
    assert records[0].recall_mode == "contextual"


def test_promote_claim_candidates_assigns_exact_recall_to_owns_and_uses() -> None:
    uses_candidate = ClaimCandidate(
        subject="user",
        predicate="uses",
        object_surface="VS Code",
        object_type="tool",
        head="VS Code",
        normalized_object="vs code",
        alternatives=["VS Code", "Code"],
        confidence=0.86,
        specificity=0.82,
        evidence_text="I use VS Code.",
        topic_keys=["uses", "tool", "usage"],
        trigger_keys=["uses", "vs", "code"],
    )
    owns_candidate = ClaimCandidate(
        subject="user",
        predicate="owns",
        object_surface="fridge Samsung",
        object_type="appliance",
        head="fridge",
        normalized_object="fridge samsung",
        alternatives=["fridge Samsung", "Samsung"],
        confidence=0.88,
        specificity=0.80,
        evidence_text="I have a Samsung fridge.",
        topic_keys=["owns", "appliance", "ownership"],
        trigger_keys=["owns", "fridge", "samsung"],
    )

    records = promote_claim_candidates(
        [uses_candidate, owns_candidate],
        event_id="evt:claim-recall",
        namespace="claims",
    )

    recall_modes = {
        str(item.predicate or ""): str(item.recall_mode or "")
        for item in list(records or [])
    }
    assert recall_modes["uses"] == "exact"
    assert recall_modes["owns"] == "exact"


def test_promote_claim_candidates_assigns_ambient_recall_to_strong_like_claims() -> None:
    strong_like = ClaimCandidate(
        subject="user",
        predicate="likes",
        object_surface="смотреть в глаза Розе",
        object_type="activity",
        head="смотреть",
        normalized_object="смотреть в глаза розе",
        alternatives=["смотреть в глаза Розе", "Розе"],
        confidence=0.84,
        specificity=1.0,
        evidence_text="Я люблю смотреть в глаза Розе.",
        topic_keys=["likes", "activity", "preference"],
        trigger_keys=["likes", "смотреть", "розе"],
    )

    records = promote_claim_candidates(
        [strong_like],
        event_id="evt:strong-like",
        namespace="claims",
    )

    assert len(records) == 1
    assert records[0].recall_mode == "ambient"
    assert records[0].spontaneous_recall is True


def test_claim_candidate_extractor_extracts_entity_backed_tool_usage() -> None:
    candidates = _extract_candidates("Я использую VS Code.")

    item = _first_candidate(candidates, "uses")
    assert item.object_surface == "VS Code"
    assert item.object_type == "tool"
    matched_entities = list(dict(item.qualifiers or {}).get("matched_entities") or [])
    assert matched_entities
    assert str(matched_entities[0].get("type") or "") == "tool_name"
    assert "VS Code" in list(item.alternatives or [])


def test_claim_candidate_extractor_extracts_owned_object_with_bounded_span() -> None:
    candidates = _extract_candidates("У меня холодильник Samsung, а не старый LG.")

    item = _first_candidate(candidates, "owns")
    assert item.object_surface == "холодильник Samsung"
    assert item.object_type == "appliance"
    assert "Samsung" in list(item.alternatives or [])


def test_claim_candidates_contribute_projection_keys() -> None:
    analysis = analyze_message_for_memory("Я использую VS Code и не люблю шум.")
    views = build_memory_views(
        analysis.raw_text,
        metadata={
            "claim_candidates": [item.to_dict() for item in list(analysis.claim_candidates or [])],
        },
    )

    entity_keys = set(str(x or "") for x in list(views.get("entity_keys") or []))
    assert "claim" in entity_keys
    assert "uses" in entity_keys
    assert "tool" in entity_keys
    assert "dislikes" in entity_keys


def test_compact_claims_contribute_projection_keys() -> None:
    views = build_memory_views(
        "I use VS Code and dislike noise.",
        metadata={
            "claims": [
                {
                    "subject": "user",
                    "predicate": "uses",
                    "obj": "vs code",
                    "object_surface": "VS Code",
                    "object_type": "tool",
                    "topic_keys": ["uses", "tool", "usage"],
                    "trigger_keys": ["uses", "vs", "code"],
                    "confidence": 0.84,
                },
                {
                    "subject": "user",
                    "predicate": "dislikes",
                    "obj": "noise",
                    "object_surface": "noise",
                    "object_type": "thing",
                    "topic_keys": ["dislikes", "preference"],
                    "trigger_keys": ["dislikes", "noise"],
                    "confidence": 0.76,
                },
            ],
        },
    )

    entity_keys = set(str(x or "") for x in list(views.get("entity_keys") or []))
    assert "claim" in entity_keys
    assert "uses" in entity_keys
    assert "tool" in entity_keys
    assert "dislikes" in entity_keys


def test_memory_manager_persists_claim_records_for_user_messages() -> None:
    manager = _manager()

    manager.ingest_event(
        MemoryEvent(
            role="user",
            text="Не люблю шум. У меня холодильник Samsung. Я использую VS Code.",
            namespace="claims-demo",
            scope=MemoryScope.CONVERSATION,
            memory_type=MemoryType.MESSAGE,
            metadata={},
        )
    )

    claim_rows = [
        row
        for row in manager._store.iter_records(namespace="claims-demo")
        if row.memory_type == MemoryType.CLAIM
    ]
    predicates = {str(dict(row.metadata or {}).get("claim", {}).get("predicate") or "") for row in claim_rows}
    assert predicates == {"dislikes", "owns", "uses"}
    for row in list(claim_rows or []):
        metadata = dict(row.metadata or {})
        claim = dict(metadata.get("claim") or {})
        write_policy = dict(metadata.get("write_policy") or {})
        assert "promotion_signals" not in claim
        assert "evidence_text" not in claim
        assert "evidence_span" not in claim
        assert "signals" not in write_policy
