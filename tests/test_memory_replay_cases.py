from __future__ import annotations

import tempfile
from pathlib import Path
from threading import RLock
from types import SimpleNamespace

from memory.ingest_analyzer import IngestAnalysis, analyze_message_for_memory
from memory.fact_extractor import FactExtractor
from memory.governor import MemoryGovernor
from memory.memory_lifecycle import MemoryLifecycleManager
from memory.memory_manager import MemoryManager
from memory.memory_models import (
    ConflictDecision,
    IngestResult,
    LifecycleDecision,
    MemoryEvent,
    MemoryLevel,
    MemoryRecord,
    MemoryScope,
    MemoryStatus,
    MemoryType,
    RetrievalCandidate,
    RetrievalQuery,
    ScoreBreakdown,
)
from memory.memory_policy import MemoryPolicy
from memory.memory_scoring import SalienceWeights


class _ReplayStore:
    def __init__(self) -> None:
        self._rows: dict[str, MemoryRecord] = {}
        self._order: list[str] = []

    def upsert(self, record: MemoryRecord) -> None:
        if str(record.id) not in self._rows:
            self._order.append(str(record.id))
        self._rows[str(record.id)] = record

    def batch_upsert(self, records: list[MemoryRecord]) -> None:
        for record in list(records or []):
            self.upsert(record)

    def iter_records(self, namespace: str | None = None) -> list[MemoryRecord]:
        rows = [self._rows[row_id] for row_id in list(self._order)]
        if namespace is None:
            return rows
        return [row for row in rows if str(row.namespace or "") == str(namespace or "")]

    def close(self) -> None:
        return None


class _ReplayEventStore:
    def __init__(self) -> None:
        self.entries: list[dict] = []

    def append(self, row) -> None:  # noqa: ANN001
        self.entries.append(dict(row or {}))


class _ReplayLifecycle:
    def __init__(self) -> None:
        self._delegate = MemoryLifecycleManager()

    def decide(self, record: MemoryRecord, *, now_ts: float) -> LifecycleDecision:
        _ = (record, now_ts)
        return LifecycleDecision(reason="replay_test_no_promotion", route="replay")

    def resolve_conflict(self, *, old: MemoryRecord, new: MemoryRecord) -> ConflictDecision:
        return self._delegate.resolve_conflict(old=old, new=new)


def _manager() -> MemoryManager:
    manager = MemoryManager.__new__(MemoryManager)
    manager._cfg = SimpleNamespace()
    temp_dir = Path(tempfile.mkdtemp(prefix="mmis-memory-replay-"))
    manager._root = temp_dir
    manager._state_path = temp_dir / "manager_state.json"
    manager._lock = RLock()
    # Storage profile (must match MemoryManager)
    manager._storage_profile = "compact"
    # Debug metadata fields to strip before storage (must match MemoryManager)
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
    manager._store = _ReplayStore()
    manager._event_store = _ReplayEventStore()
    manager._fact_extractor = FactExtractor()
    manager._policy = MemoryPolicy()
    manager._lifecycle = _ReplayLifecycle()
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
    # Debug metadata fields to strip before storage
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
    }
    return manager


def _ingest_turn(
    manager: MemoryManager,
    *,
    text: str,
    role: str = "user",
    namespace: str = "replay",
    metadata: dict | None = None,
) -> IngestResult:
    payload = {
        "conversation_id": namespace,
        **dict(metadata or {}),
    }
    return manager.ingest_event(
        MemoryEvent(
            role=role,
            text=text,
            namespace=namespace,
            scope=MemoryScope.CONVERSATION,
            memory_type=MemoryType.MESSAGE,
            metadata=payload,
        )
    )


def _records(
    manager: MemoryManager,
    *,
    namespace: str = "replay",
    memory_type: MemoryType | None = None,
    status: MemoryStatus | None = None,
) -> list[MemoryRecord]:
    rows = list(manager._store.iter_records(namespace=namespace))
    if memory_type is not None:
        rows = [row for row in rows if row.memory_type == memory_type]
    if status is not None:
        rows = [row for row in rows if row.status == status]
    return rows


def _fact_rows(
    manager: MemoryManager,
    *,
    namespace: str = "replay",
    predicate: str | None = None,
    status: MemoryStatus | None = None,
) -> list[MemoryRecord]:
    rows = _records(manager, namespace=namespace, memory_type=MemoryType.FACT, status=status)
    if not predicate:
        return rows
    expected = str(predicate or "").strip().lower()
    out: list[MemoryRecord] = []
    for row in rows:
        fact = dict(row.metadata or {}).get("fact") or {}
        if str(dict(fact).get("predicate") or "").strip().lower() == expected:
            out.append(row)
    return out


def _fact_values(
    manager: MemoryManager,
    *,
    predicate: str,
    namespace: str = "replay",
    status: MemoryStatus | None = MemoryStatus.ACTIVE,
) -> list[str]:
    out: list[str] = []
    for row in _fact_rows(manager, namespace=namespace, predicate=predicate, status=status):
        fact = dict(row.metadata or {}).get("fact") or {}
        value = dict(fact).get("value")
        out.append(str(value))
    return out


def _root_message_rows(manager: MemoryManager, *, namespace: str = "replay") -> list[MemoryRecord]:
    rows = _records(manager, namespace=namespace, memory_type=MemoryType.MESSAGE)
    return [row for row in rows if ":p:" not in str(row.id)]


def _extract(text: str) -> tuple[IngestAnalysis, list]:
    analysis = analyze_message_for_memory(text)
    facts = FactExtractor().extract_v2(
        text=text,
        metadata={"event_id": "evt:test", "namespace": "default"},
        speaker="user",
        scope=MemoryScope.CONVERSATION,
        mode="BALANCED",
        analysis=analysis,
    )
    return analysis, facts


def _has_entity(analysis: IngestAnalysis, entity_type: str, canonical_contains: str) -> bool:
    target = str(canonical_contains or "").lower()
    for item in list(analysis.entities or []):
        if str(item.type or "").lower() != str(entity_type or "").lower():
            continue
        if target in str(item.canonical or "").lower():
            return True
    return False


def _has_numeric(analysis: IngestAnalysis, kind: str, value) -> bool:  # noqa: ANN001
    for item in list(analysis.numeric_facts or []):
        if str(item.kind or "").lower() == str(kind or "").lower() and item.value == value:
            return True
    return False


def _has_fact(facts: list, predicate: str, value=None) -> bool:  # noqa: ANN001
    for item in list(facts or []):
        if str(item.predicate or "").lower() != str(predicate or "").lower():
            continue
        if value is None:
            return True
        if item.value == value or str(item.value) == str(value):
            return True
    return False


def _has_tag(analysis: IngestAnalysis, tag_prefix: str) -> bool:
    prefix = str(tag_prefix or "").lower()
    return any(str(tag or "").lower().startswith(prefix) for tag in list(analysis.tags or []))


def test_replay_builds_structured_environment_profile_from_short_turns() -> None:
    manager = _manager()

    first = _ingest_turn(manager, text="I use Windows.", namespace="profile")
    second = _ingest_turn(manager, text="Python 3.11", namespace="profile")
    third = _ingest_turn(manager, text="RTX 3050 Ti with 4 GB VRAM", namespace="profile")
    fourth = _ingest_turn(manager, text="32 GB RAM", namespace="profile")

    assert any(row.predicate == "environment_os" for row in first.extracted_facts)
    assert any(row.predicate == "environment_runtime_python" for row in second.extracted_facts)
    assert any(row.predicate == "environment_gpu_model" for row in third.extracted_facts)
    assert any(row.predicate == "environment_ram_gb" for row in fourth.extracted_facts)

    assert _fact_values(manager, namespace="profile", predicate="environment_os") == ["windows"]
    assert _fact_values(manager, namespace="profile", predicate="environment_runtime_python") == ["python 3.11"]
    assert _fact_values(manager, namespace="profile", predicate="environment_gpu_model") == ["RTX 3050 Ti"]
    assert _fact_values(manager, namespace="profile", predicate="environment_gpu_vram_gb") == ["4"]
    assert _fact_values(manager, namespace="profile", predicate="environment_ram_gb") == ["32"]

    root_messages = _root_message_rows(manager, namespace="profile")
    assert len(root_messages) == 4
    # Message metadata stays compact: no duplicated raw projections.
    for row in root_messages:
        meta = dict(row.metadata or {})
        assert "search_text" not in meta
        assert "normalized_text" not in meta
        assert "canonical_text" not in meta
        views = dict(meta.get("memory_views") or {})
        has_keys = bool(views.get("entity_keys") or views.get("numeric_keys"))
        assert has_keys, f"Missing compact memory_views keys for record {row.id}"


def test_replay_persists_compact_fact_metadata() -> None:
    manager = _manager()

    _ingest_turn(manager, text="я сижу на python 3.11", namespace="compact-facts")

    fact_rows = _fact_rows(
        manager,
        namespace="compact-facts",
        predicate="environment_runtime_python",
        status=MemoryStatus.ACTIVE,
    )
    assert fact_rows

    metadata = dict(fact_rows[0].metadata or {})
    fact = dict(metadata.get("fact") or {})
    write_policy = dict(metadata.get("write_policy") or {})

    assert "evidence" not in fact
    assert "metadata" not in fact
    assert "text" not in fact
    assert "created_at" not in fact
    assert "updated_at" not in fact
    assert "signals" not in write_policy


def test_replay_false_positive_guards_block_identity_and_age_noise() -> None:
    manager = _manager()

    _ingest_turn(manager, text="I'm tired and frustrated.", namespace="noise")
    _ingest_turn(manager, text="I am on Windows.", namespace="noise")
    _ingest_turn(manager, text="I have 32 GB RAM and i am 3 commits behind", namespace="noise")

    assert _fact_values(manager, namespace="noise", predicate="identity_name") == []
    assert _fact_values(manager, namespace="noise", predicate="identity_age_years") == []
    assert _fact_values(manager, namespace="noise", predicate="environment_os") == ["windows"]
    assert _fact_values(manager, namespace="noise", predicate="environment_ram_gb") == ["32"]


def test_replay_singleton_python_fact_supersedes_previous_version() -> None:
    manager = _manager()

    _ingest_turn(manager, text="Python 3.10", namespace="python")
    _ingest_turn(manager, text="Python 3.11", namespace="python")

    active = _fact_rows(
        manager,
        namespace="python",
        predicate="environment_runtime_python",
        status=MemoryStatus.ACTIVE,
    )
    superseded = _fact_rows(
        manager,
        namespace="python",
        predicate="environment_runtime_python",
        status=MemoryStatus.SUPERSEDED,
    )

    assert [str(dict(dict(row.metadata or {}).get("fact") or {}).get("value")) for row in active] == ["python 3.11"]
    assert [str(dict(dict(row.metadata or {}).get("fact") or {}).get("value")) for row in superseded] == ["python 3.10"]
    assert len(active) == 1
    assert len(superseded) == 1


def test_replay_supersedes_ram_group_across_memory_and_ram_predicates() -> None:
    manager = _manager()

    _ingest_turn(manager, text="I have 16 GB memory.", namespace="ram-gap")
    _ingest_turn(manager, text="I have 32 GB RAM.", namespace="ram-gap")

    assert _fact_values(manager, namespace="ram-gap", predicate="environment_memory_gb") == []
    assert _fact_values(manager, namespace="ram-gap", predicate="environment_ram_gb") == ["32"]
    superseded = _fact_rows(manager, namespace="ram-gap", predicate="environment_memory_gb", status=MemoryStatus.SUPERSEDED)
    assert [str(dict(dict(row.metadata or {}).get("fact") or {}).get("value")) for row in superseded] == ["16"]
    assert len(_fact_rows(manager, namespace="ram-gap", status=MemoryStatus.ACTIVE)) >= 1


def test_replay_does_not_store_assistant_environment_restatement() -> None:
    manager = _manager()

    result = _ingest_turn(
        manager,
        text="You use Windows, Python 3.11, and RTX 3050 Ti with 4 GB VRAM.",
        role="assistant",
        namespace="assistant-noise",
    )

    assert all(not str(row.predicate or "").startswith("environment_") for row in list(result.extracted_facts or []))
    assert _fact_values(manager, namespace="assistant-noise", predicate="environment_os") == []
    assert _fact_values(manager, namespace="assistant-noise", predicate="environment_runtime_python") == []
    assert _fact_values(manager, namespace="assistant-noise", predicate="environment_gpu_model") == []


def test_fact_expectation_check_prefers_exact_gpu_fact_over_message_similarity() -> None:
    manager = _manager()
    manager._store.upsert(
        MemoryRecord(
            id="msg:gpu-only",
            text="у меня видяха rtx 3050 ti",
            memory_type=MemoryType.MESSAGE,
            level=MemoryLevel.L0_WORKING,
            scope=MemoryScope.CONVERSATION,
            namespace="fact-check",
            metadata={"memory_views": {"entity_keys": ["gpu", "rtx_3050_ti"], "numeric_keys": []}},
        )
    )

    row = manager._build_fact_expectation_check(
        query_text="какая у меня видюха?",
        namespace="fact-check",
    )

    assert row["exact_fact_required"] is True
    assert row["expected_predicates"] == ["environment_gpu_model"]
    assert row["found_predicates"] == []
    assert row["missing_predicates"] == ["environment_gpu_model"]
    assert row["should_answer_cautiously"] is True


def test_fact_expectation_check_finds_exact_python_fact() -> None:
    manager = _manager()
    _ingest_turn(manager, text="я сижу на python 3.11", namespace="fact-check-python")

    row = manager._build_fact_expectation_check(
        query_text="напомни, какой у меня python",
        namespace="fact-check-python",
    )

    assert row["expected_predicates"] == ["environment_runtime_python"]
    assert row["missing_predicates"] == []
    assert row["found_predicates"] == ["environment_runtime_python"]
    values = [str(item.get("value") or "") for item in list(row["found_facts"]["environment_runtime_python"] or [])]
    assert "python 3.11" in values


def test_fact_expectation_check_matches_live_gpu_wording() -> None:
    manager = _manager()

    row = manager._build_fact_expectation_check(
        query_text="помнишь, что у меня за карточка?",
        namespace="fact-check-gpu-live",
    )

    assert row["expected_predicates"] == ["environment_gpu_model"]
    assert row["missing_predicates"] == ["environment_gpu_model"]


def test_fact_expectation_check_matches_live_python_wording() -> None:
    manager = _manager()

    row = manager._build_fact_expectation_check(
        query_text="помнишь мой питон?",
        namespace="fact-check-python-live",
    )

    assert row["expected_predicates"] == ["environment_runtime_python"]
    assert row["missing_predicates"] == ["environment_runtime_python"]


def test_fact_expectation_check_matches_colloquial_python_wording_with_paiton() -> None:
    manager = _manager()

    row = manager._build_fact_expectation_check(
        query_text="напомни мой пайтон",
        namespace="fact-check-python-paiton",
    )

    assert row["expected_predicates"] == ["environment_runtime_python"]
    assert row["missing_predicates"] == ["environment_runtime_python"]


def test_fact_expectation_check_matches_live_os_wording() -> None:
    manager = _manager()

    row = manager._build_fact_expectation_check(
        query_text="что у меня за винда?",
        namespace="fact-check-os-live",
    )

    assert row["expected_predicates"] == ["environment_os"]
    assert row["missing_predicates"] == ["environment_os"]


def test_fact_expectation_check_matches_colloquial_os_wording_with_systema() -> None:
    manager = _manager()

    row = manager._build_fact_expectation_check(
        query_text="что у меня за система, версия винды?",
        namespace="fact-check-os-systema",
    )

    assert row["expected_predicates"] == ["environment_os"]
    assert row["missing_predicates"] == ["environment_os"]


def test_fact_expectation_check_matches_live_ram_wording() -> None:
    manager = _manager()

    row = manager._build_fact_expectation_check(
        query_text="помнишь мою оперативку?",
        namespace="fact-check-ram-live",
    )

    assert row["expected_predicates"] == ["environment_ram_gb", "environment_memory_gb"]
    assert row["missing_predicates"] == ["environment_ram_gb", "environment_memory_gb"]


def test_fact_expectation_check_falls_back_to_relevant_predicates_for_exact_recall() -> None:
    manager = _manager()

    row = manager._build_fact_expectation_check(
        query_text="что там у меня по питону?",
        namespace="fact-check-python-fallback",
    )

    assert row["expected_predicates"] == ["environment_runtime_python"]
    assert row["missing_predicates"] == ["environment_runtime_python"]
    assert "self_fact_fallback" in list(row["intents"] or [])


def test_fact_expectation_prioritizes_exact_fact_before_message_hits() -> None:
    manager = _manager()
    _ingest_turn(manager, text="у меня rtx 3050 ti", namespace="fact-order")
    message_record = MemoryRecord(
        id="msg:gpu-chat",
        text="кажется, мы обсуждали мою видеокарту раньше",
        memory_type=MemoryType.MESSAGE,
        level=MemoryLevel.L2_EPISODIC,
        scope=MemoryScope.CONVERSATION,
        namespace="fact-order",
    )
    candidate = RetrievalCandidate(
        record=message_record,
        score_breakdown=ScoreBreakdown(final_score=0.95),
        source="hybrid",
    )

    fact_expectation = manager._build_fact_expectation_check(
        query_text="какая у меня видюха?",
        namespace="fact-order",
    )
    ordered = manager._prioritize_retrieval_candidates_for_fact_expectation(
        query=RetrievalQuery(query_text="какая у меня видюха?", namespace="fact-order", top_k=8),
        candidates=[candidate],
        exact_fact_candidates=manager._exact_fact_candidates_for_expectation(
            query=RetrievalQuery(query_text="какая у меня видюха?", namespace="fact-order", top_k=8),
            namespace="fact-order",
            fact_expectation=fact_expectation,
        ),
        fact_expectation=fact_expectation,
    )

    assert ordered
    assert ordered[0].record.memory_type == MemoryType.FACT
    first_fact = dict(dict(ordered[0].record.metadata or {}).get("fact") or {})
    assert str(first_fact.get("predicate") or "") == "environment_gpu_model"


def test_self_fact_recall_uses_exact_facts_without_message_fallback_when_found() -> None:
    manager = _manager()
    _ingest_turn(manager, text="у меня rtx 3050 ti", namespace="self-recall")
    message_candidate = RetrievalCandidate(
        record=MemoryRecord(
            id="msg:self-recall",
            text="мы вроде обсуждали мою видеокарту раньше",
            memory_type=MemoryType.MESSAGE,
            level=MemoryLevel.L2_EPISODIC,
            scope=MemoryScope.CONVERSATION,
            namespace="self-recall",
        ),
        score_breakdown=ScoreBreakdown(final_score=0.99),
        source="hybrid",
    )
    fact_expectation = manager._build_fact_expectation_check(
        query_text="какая у меня видюха?",
        namespace="self-recall",
    )
    exact_fact_candidates = manager._exact_fact_candidates_for_expectation(
        query=RetrievalQuery(query_text="какая у меня видюха?", namespace="self-recall", top_k=8),
        namespace="self-recall",
        fact_expectation=fact_expectation,
    )
    prioritized = manager._prioritize_retrieval_candidates_for_fact_expectation(
        query=RetrievalQuery(query_text="какая у меня видюха?", namespace="self-recall", top_k=8),
        candidates=[message_candidate],
        exact_fact_candidates=exact_fact_candidates,
        fact_expectation=fact_expectation,
    )
    selected = manager._select_candidates_for_self_fact_recall(
        fallback_candidates=prioritized,
        exact_fact_candidates=exact_fact_candidates,
        fact_expectation=fact_expectation,
    )

    assert selected
    assert all(item.record.memory_type == MemoryType.FACT for item in selected)
    assert not any(str(item.record.id or "") == "msg:self-recall" for item in selected)


def test_self_fact_recall_falls_back_to_message_retrieval_when_exact_fact_missing() -> None:
    manager = _manager()
    message_candidate = RetrievalCandidate(
        record=MemoryRecord(
            id="msg:self-recall-missing",
            text="я вроде упоминал свою видеокарту",
            memory_type=MemoryType.MESSAGE,
            level=MemoryLevel.L2_EPISODIC,
            scope=MemoryScope.CONVERSATION,
            namespace="self-recall-missing",
        ),
        score_breakdown=ScoreBreakdown(final_score=0.81),
        source="hybrid",
    )
    fact_expectation = manager._build_fact_expectation_check(
        query_text="какая у меня видюха?",
        namespace="self-recall-missing",
    )
    exact_fact_candidates = manager._exact_fact_candidates_for_expectation(
        query=RetrievalQuery(query_text="какая у меня видюха?", namespace="self-recall-missing", top_k=8),
        namespace="self-recall-missing",
        fact_expectation=fact_expectation,
    )
    selected = manager._select_candidates_for_self_fact_recall(
        fallback_candidates=[message_candidate],
        exact_fact_candidates=exact_fact_candidates,
        fact_expectation=fact_expectation,
    )

    assert exact_fact_candidates == []
    assert len(selected) == 1
    assert str(selected[0].record.id or "") == "msg:self-recall-missing"


def test_self_fact_recall_missing_exact_fact_filters_unrelated_fact_candidates() -> None:
    manager = _manager()
    message_candidate = RetrievalCandidate(
        record=MemoryRecord(
            id="msg:ram-fallback",
            text="you mentioned 32 gb ram before",
            memory_type=MemoryType.MESSAGE,
            level=MemoryLevel.L2_EPISODIC,
            scope=MemoryScope.CONVERSATION,
            namespace="self-recall-ram-filter",
        ),
        score_breakdown=ScoreBreakdown(final_score=0.71),
        source="hybrid",
    )
    unrelated_fact = RetrievalCandidate(
        record=MemoryRecord(
            id="fact:name",
            text="user.identity_name=Pasha",
            memory_type=MemoryType.FACT,
            level=MemoryLevel.L3_SEMANTIC,
            scope=MemoryScope.CONVERSATION,
            namespace="self-recall-ram-filter",
            metadata={"fact": {"predicate": "identity_name", "subject": "user", "value": "Pasha"}},
            status=MemoryStatus.ACTIVE,
        ),
        score_breakdown=ScoreBreakdown(final_score=0.95),
        source="fact_channel",
    )
    working_fact = RetrievalCandidate(
        record=MemoryRecord(
            id="fact:ram-working",
            text="user.environment_ram_gb=4",
            memory_type=MemoryType.FACT,
            level=MemoryLevel.L0_WORKING,
            scope=MemoryScope.CONVERSATION,
            namespace="self-recall-ram-filter",
            metadata={"fact": {"predicate": "environment_ram_gb", "subject": "user", "value": "4"}},
            status=MemoryStatus.ACTIVE,
        ),
        score_breakdown=ScoreBreakdown(final_score=0.96),
        source="fact_channel",
    )

    selected = manager._select_candidates_for_self_fact_recall(
        fallback_candidates=[unrelated_fact, working_fact, message_candidate],
        exact_fact_candidates=[],
        fact_expectation={
            "expected_predicates": ["environment_ram_gb", "environment_memory_gb"],
            "found_predicates": [],
        },
    )

    assert len(selected) == 1
    assert str(selected[0].record.id or "") == "msg:ram-fallback"


def test_self_facts_context_can_be_built_from_live_gpu_rule_match() -> None:
    manager = _manager()
    _ingest_turn(manager, text="у меня rtx 3050 ti", namespace="self-facts-soft")
    fact_rows = _fact_rows(manager, namespace="self-facts-soft", predicate="environment_gpu_model", status=MemoryStatus.ACTIVE)
    candidates = [
        RetrievalCandidate(
            record=row,
            score_breakdown=ScoreBreakdown(final_score=0.92),
            source="fact_channel",
        )
        for row in list(fact_rows or [])
    ]

    fact_expectation = manager._build_fact_expectation_check(
        query_text="подскажи мою карточку",
        namespace="self-facts-soft",
    )
    self_facts = manager._build_self_facts_context(
        query_text="подскажи мою карточку",
        selected_candidates=list(candidates or []),
        fact_expectation=fact_expectation,
    )

    assert fact_expectation["expected_predicates"] == ["environment_gpu_model"]
    assert "environment_gpu_model" in list(self_facts.get("found_predicates") or [])
    values = [str(item.get("value") or "") for item in list(dict(self_facts.get("found_facts") or {}).get("environment_gpu_model") or [])]
    assert "RTX 3050 Ti" in values


def test_self_facts_context_does_not_attach_unrelated_user_facts_for_generic_query() -> None:
    manager = _manager()
    _ingest_turn(manager, text="у меня rtx 3050 ti", namespace="self-facts-always")
    fact_rows = _fact_rows(manager, namespace="self-facts-always", predicate="environment_gpu_model", status=MemoryStatus.ACTIVE)
    candidates = [
        RetrievalCandidate(
            record=row,
            score_breakdown=ScoreBreakdown(final_score=0.88),
            source="fact_channel",
        )
        for row in list(fact_rows or [])
    ]

    self_facts = manager._build_self_facts_context(
        query_text="what should we fix next",
        selected_candidates=list(candidates or []),
        fact_expectation={},
    )

    assert self_facts == {}


def test_self_facts_context_filters_to_ram_predicates_for_ozu_query() -> None:
    manager = _manager()
    _ingest_turn(manager, text="my name is Pasha", namespace="self-facts-ram-filter")
    _ingest_turn(manager, text="i have rtx 3050 ti", namespace="self-facts-ram-filter")
    fact_rows = list(_fact_rows(manager, namespace="self-facts-ram-filter", predicate="identity_name", status=MemoryStatus.ACTIVE))
    fact_rows.extend(_fact_rows(manager, namespace="self-facts-ram-filter", predicate="environment_gpu_model", status=MemoryStatus.ACTIVE))
    candidates = [
        RetrievalCandidate(
            record=row,
            score_breakdown=ScoreBreakdown(final_score=0.91),
            source="fact_channel",
        )
        for row in list(fact_rows or [])
    ]

    fact_expectation = manager._build_fact_expectation_check(
        query_text="what is my ram",
        namespace="self-facts-ram-filter",
    )
    self_facts = manager._build_self_facts_context(
        query_text="what is my ram",
        selected_candidates=list(candidates or []),
        fact_expectation=fact_expectation,
    )

    assert fact_expectation["expected_predicates"] == ["environment_ram_gb", "environment_memory_gb"]
    assert fact_expectation["found_predicates"] == []
    assert self_facts == {}


def test_memory_recall_mode_detects_exact_fact_recall() -> None:
    manager = _manager()

    mode = manager._classify_memory_recall_mode(query_text="какая у меня видеокарта?")

    assert mode == "exact_fact_recall"


def test_memory_recall_mode_detects_contextual_recall() -> None:
    manager = _manager()

    mode = manager._classify_memory_recall_mode(query_text="что мы обсуждали про память?")

    assert mode == "contextual_recall"


def test_memory_recall_mode_detects_contextual_recall_for_conclusion_query() -> None:
    manager = _manager()

    mode = manager._classify_memory_recall_mode(query_text="к чему пришли по памяти?")

    assert mode == "contextual_recall"


def test_memory_recall_mode_detects_contextual_recall_for_plan_query_variant() -> None:
    manager = _manager()

    mode = manager._classify_memory_recall_mode(query_text="какой у нас был план по памяти?")

    assert mode == "contextual_recall"


def test_memory_recall_mode_detects_document_recall() -> None:
    manager = _manager()

    mode = manager._classify_memory_recall_mode(query_text="where in code is ollama called?")

    assert mode == "document_recall"


def test_contextual_recall_prioritizes_episodes_then_messages_then_decision_facts() -> None:
    manager = _manager()
    candidates = [
        RetrievalCandidate(
            record=MemoryRecord(
                id="episode:memory",
                text="memory design episode",
                memory_type=MemoryType.EPISODE,
                level=MemoryLevel.L2_EPISODIC,
                scope=MemoryScope.CONVERSATION,
                namespace="contextual-recall",
                metadata={"episode": {"topic": "memory design"}},
                status=MemoryStatus.ACTIVE,
            ),
            score_breakdown=ScoreBreakdown(final_score=0.64),
            source="message_channel",
        ),
        RetrievalCandidate(
            record=MemoryRecord(
                id="fact:gpu",
                text="user.environment_gpu_model=RTX 3050 Ti",
                memory_type=MemoryType.FACT,
                level=MemoryLevel.L3_SEMANTIC,
                scope=MemoryScope.CONVERSATION,
                namespace="contextual-recall",
                metadata={"fact": {"predicate": "environment_gpu_model", "subject": "user", "value": "RTX 3050 Ti"}},
                status=MemoryStatus.ACTIVE,
            ),
            score_breakdown=ScoreBreakdown(final_score=0.98),
            source="fact_channel",
        ),
        RetrievalCandidate(
            record=MemoryRecord(
                id="fact:decision",
                text="assistant.decision=store_search_text_and_canonical_text",
                memory_type=MemoryType.FACT,
                level=MemoryLevel.L3_SEMANTIC,
                scope=MemoryScope.CONVERSATION,
                namespace="contextual-recall",
                metadata={"fact": {"predicate": "decision", "subject": "assistant", "value": "store_search_text_and_canonical_text"}},
                status=MemoryStatus.ACTIVE,
            ),
            score_breakdown=ScoreBreakdown(final_score=0.72),
            source="fact_channel",
        ),
        RetrievalCandidate(
            record=MemoryRecord(
                id="msg:discussion",
                text="мы обсуждали, что сначала доделываем память, потом веб",
                memory_type=MemoryType.MESSAGE,
                level=MemoryLevel.L2_EPISODIC,
                scope=MemoryScope.CONVERSATION,
                namespace="contextual-recall",
            ),
            score_breakdown=ScoreBreakdown(final_score=0.70),
            source="message_channel",
        ),
    ]

    ordered = manager._prioritize_contextual_recall_candidates(
        query_text="что мы обсуждали про память?",
        candidates=candidates,
    )

    assert [str(item.record.id or "") for item in ordered[:3]] == [
        "episode:memory",
        "msg:discussion",
        "fact:decision",
    ]


def test_render_memory_recall_mode_contextual_includes_context_rule() -> None:
    manager = _manager()

    block = manager._render_memory_recall_mode("contextual_recall")

    assert "mode: contextual_recall" in block
    assert "dialog episodes" in block


def test_render_memory_recall_mode_document_includes_document_rule() -> None:
    manager = _manager()

    block = manager._render_memory_recall_mode("document_recall")

    assert "mode: document_recall" in block
    assert "document chunks" in block


def test_self_facts_context_uses_exact_ram_fact_for_ozu_query() -> None:
    manager = _manager()
    _ingest_turn(manager, text="i have 32 gb ram", namespace="self-facts-ram-hit")
    fact_rows = _fact_rows(manager, namespace="self-facts-ram-hit", predicate="environment_ram_gb", status=MemoryStatus.ACTIVE)
    candidates = [
        RetrievalCandidate(
            record=row,
            score_breakdown=ScoreBreakdown(final_score=0.89),
            source="fact_channel",
        )
        for row in list(fact_rows or [])
    ]

    fact_expectation = manager._build_fact_expectation_check(
        query_text="what is my ram",
        namespace="self-facts-ram-hit",
    )
    self_facts = manager._build_self_facts_context(
        query_text="what is my ram",
        selected_candidates=list(candidates or []),
        fact_expectation=fact_expectation,
    )

    assert "environment_ram_gb" in list(self_facts.get("found_predicates") or [])
    values = [str(item.get("value") or "") for item in list(dict(self_facts.get("found_facts") or {}).get("environment_ram_gb") or [])]
    assert "32" in values


def test_render_self_facts_includes_strict_response_rules() -> None:
    manager = _manager()

    block = manager._render_self_facts(
        {
            "found_predicates": ["environment_gpu_model"],
            "found_facts": {
                "environment_gpu_model": [
                    {"value": "RTX 3050 Ti"},
                ]
            },
        }
    )

    assert "- trust_level: exact_active_self_facts" in block
    assert "- response_rule: answer directly from these facts." in block
    assert "- response_rule: do not suggest ways to check manually." in block
    assert "- response_rule: do not say you cannot see it in memory." in block
    assert "- response_rule: ignore weaker ordinary memory snippets if they conflict." in block
    assert "- environment_gpu_model: RTX 3050 Ti" in block


def test_identity_name_and_age_ru() -> None:
    analysis, facts = _extract("меня зовут Паша, мне 21 год")

    assert _has_entity(analysis, "person_name", "паша")
    assert _has_numeric(analysis, "age_years", 21)
    assert _has_fact(facts, "identity_name", "Паша")
    assert _has_fact(facts, "identity_age_years", 21)


def test_identity_name_en() -> None:
    analysis, facts = _extract("my name is Pasha")

    assert _has_entity(analysis, "person_name", "pasha")
    assert _has_fact(facts, "identity_name", "Pasha")


def test_identity_age_en() -> None:
    analysis, facts = _extract("i am 21 years old")

    assert _has_numeric(analysis, "age_years", 21)
    assert _has_fact(facts, "identity_age_years", 21)


def test_not_capture_name_from_im_tired() -> None:
    analysis, facts = _extract("I'm tired today")

    assert not any(str(item.type or "").lower() == "person_name" for item in list(analysis.entities or []))
    assert not _has_fact(facts, "identity_name")


def test_not_capture_age_from_ram_phrase() -> None:
    analysis, facts = _extract("I have 32 GB RAM and i am 3 commits behind")

    assert not _has_numeric(analysis, "age_years", 32)
    assert not _has_fact(facts, "identity_age_years", 32)


def test_gpu_and_vram_full_phrase() -> None:
    analysis, facts = _extract("у меня видяха rtx 3050 ti на 4 gb vram")

    assert _has_entity(analysis, "gpu_model", "rtx 3050 ti")
    assert _has_numeric(analysis, "vram_gb", 4)
    assert _has_fact(facts, "environment_gpu_model", "RTX 3050 Ti")
    assert _has_fact(facts, "environment_gpu_vram_gb", 4)
    assert _has_tag(analysis, "topic_hardware")


def test_gpu_short_phrase_with_nvidia_context() -> None:
    analysis, facts = _extract("у меня nvidia 3050 ti")

    assert _has_entity(analysis, "gpu_model", "3050 ti")
    assert _has_fact(facts, "environment_gpu_model")


def test_cpu_model() -> None:
    analysis, facts = _extract("у меня intel i5-11400h")

    assert _has_entity(analysis, "cpu_model", "i5-11400h")
    assert _has_fact(facts, "environment_cpu_model", "I5-11400H")


def test_ram_amount() -> None:
    analysis, facts = _extract("I have 32 GB RAM")

    assert _has_numeric(analysis, "ram_gb", 32) or _has_numeric(analysis, "memory_gb", 32)
    assert _has_fact(facts, "environment_ram_gb", 32) or _has_fact(facts, "environment_memory_gb", 32)


def test_os_detection() -> None:
    analysis, facts = _extract("я сижу на windows 11")

    assert _has_entity(analysis, "os_name", "windows")
    assert _has_fact(facts, "environment_os", "windows")


def test_tool_detection() -> None:
    analysis, facts = _extract("I use ollama and docker")

    assert _has_entity(analysis, "tool_name", "ollama")
    assert _has_entity(analysis, "tool_name", "docker")
    assert _has_fact(facts, "environment_tool", "ollama") or _has_fact(facts, "environment_tool", "Ollama")


def test_python_version() -> None:
    analysis, facts = _extract("я работаю на python 3.11")

    assert _has_entity(analysis, "python_version", "3.11")
    assert _has_numeric(analysis, "python_version", "3.11")
    assert _has_fact(facts, "environment_runtime_python", "python 3.11")


def test_project_name_mmis() -> None:
    analysis, facts = _extract("я развиваю проект MMis")

    assert _has_entity(analysis, "project_name", "mmis")
    assert _has_fact(facts, "project_name", "MMis")


def test_multiple_environment_items_in_one_phrase() -> None:
    analysis, facts = _extract("I use windows, python 3.11, ollama and chromadb")

    assert _has_entity(analysis, "os_name", "windows")
    assert _has_entity(analysis, "python_version", "3.11")
    assert _has_entity(analysis, "tool_name", "ollama")
    assert _has_entity(analysis, "tool_name", "chromadb")

    assert _has_fact(facts, "environment_os", "windows")
    assert _has_fact(facts, "environment_runtime_python", "python 3.11")


def test_not_capture_gpu_from_plain_number() -> None:
    analysis, facts = _extract("у меня 3050 сообщений в логе")

    assert not any(str(item.type or "").lower() == "gpu_model" for item in list(analysis.entities or []))
    assert not _has_fact(facts, "environment_gpu_model")


def test_not_capture_python_from_unrelated_number() -> None:
    analysis, facts = _extract("сегодня 3.11 часа ждал конвертацию")

    assert not any(str(item.type or "").lower() == "python_version" for item in list(analysis.entities or []))
    assert not _has_fact(facts, "environment_runtime_python")


def test_emotion_frustrated() -> None:
    analysis, facts = _extract("это уже бесит, всё криво работает")

    assert analysis.emotion is not None
    assert str(analysis.emotion.primary or "").strip() != ""
    assert len(list(analysis.tags or [])) >= 1
    assert isinstance(facts, list)


def test_hardware_tags_present() -> None:
    analysis, facts = _extract("у меня rtx 3050 ti и 4 gb vram")

    assert _has_tag(analysis, "topic_hardware")
    assert isinstance(facts, list)


def test_single_message_no_duplicate_python_fact() -> None:
    _analysis, facts = _extract("я сижу на python 3.11")

    python_facts = [item for item in list(facts or []) if str(item.predicate or "") == "environment_runtime_python"]
    assert len(python_facts) == 1


def test_single_message_no_duplicate_gpu_fact() -> None:
    _analysis, facts = _extract("у меня rtx 3050 ti")

    gpu_facts = [item for item in list(facts or []) if str(item.predicate or "") == "environment_gpu_model"]
    assert len(gpu_facts) == 1


def test_single_message_no_duplicate_age_fact() -> None:
    _analysis, facts = _extract("меня зовут Паша, мне 21 год")

    age_facts = [item for item in list(facts or []) if str(item.predicate or "") == "identity_age_years"]
    assert len(age_facts) == 1
def test_decision_fact_canonicalizes_store_search_and_canonical_text() -> None:
    _analysis, facts = _extract("Ok, we will store search_text and canonical_text")

    assert _has_fact(facts, "decision", "store_search_text_and_canonical_text")


def test_decision_fact_canonicalizes_do_not_store_assistant_thoughts() -> None:
    _analysis, facts = _extract("Договорились не сохранять мысли ассистента")

    assert _has_fact(facts, "decision", "do_not_store_assistant_thoughts")


def test_agreed_plan_fact_canonicalizes_memory_before_web() -> None:
    _analysis, facts = _extract("Сначала доделываем память, потом веб")

    assert _has_fact(facts, "agreed_plan", "finish_memory_before_web")


def test_task_goal_fact_canonicalizes_next_write_policy() -> None:
    _analysis, facts = _extract("Следующим делом делаем write policy")

    assert _has_fact(facts, "task_goal", "next_write_policy")
