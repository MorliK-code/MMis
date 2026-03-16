from __future__ import annotations

import tempfile
from pathlib import Path
from threading import RLock
from types import SimpleNamespace

from memory.ingest_analyzer import IngestAnalysis, analyze_message_for_memory
from memory.fact_extractor import FactExtractor
from memory.memory_lifecycle import MemoryLifecycleManager
from memory.memory_manager import MemoryManager
from memory.memory_models import (
    ConflictDecision,
    IngestResult,
    LifecycleDecision,
    MemoryEvent,
    MemoryRecord,
    MemoryScope,
    MemoryStatus,
    MemoryType,
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
    manager._store = _ReplayStore()
    manager._event_store = _ReplayEventStore()
    manager._fact_extractor = FactExtractor()
    manager._policy = MemoryPolicy()
    manager._lifecycle = _ReplayLifecycle()
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
    assert all(str(dict(row.metadata or {}).get("analysis_version") or "") == "memory_ingest_v3" for row in root_messages)
    assert all(bool(dict(dict(row.metadata or {}).get("memory_views") or {}).get("search_text")) for row in root_messages)


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


def test_replay_documents_ram_group_supersede_gap_between_memory_and_ram_predicates() -> None:
    manager = _manager()

    _ingest_turn(manager, text="I have 16 GB memory.", namespace="ram-gap")
    _ingest_turn(manager, text="I have 32 GB RAM.", namespace="ram-gap")

    # Current behavior keeps both because singleton handling is predicate-level, not group-level.
    assert _fact_values(manager, namespace="ram-gap", predicate="environment_memory_gb") == ["16"]
    assert _fact_values(manager, namespace="ram-gap", predicate="environment_ram_gb") == ["32"]
    assert len(_fact_rows(manager, namespace="ram-gap", status=MemoryStatus.ACTIVE)) >= 2


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
