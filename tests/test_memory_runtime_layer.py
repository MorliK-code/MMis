from __future__ import annotations

import time
from types import SimpleNamespace

from core.brain import Brain
from memory_core.adapter import MemoryCoreAdapter
from memory_core.episode_manager import EpisodeManager
from memory_core.facade import MemoryService
from memory_core.retrieval.query_models import ContextPack
from memory_core.runtime_session_store import RuntimeSessionStore
from memory_core.schemas import MemoryEnvelope, MemoryQuery


class _FakeEventStore:
    def __init__(self) -> None:
        self.events = []

    def append(self, envelope: MemoryEnvelope) -> None:
        self.events.append(envelope)


class _FakeJobQueue:
    TYPE_MEMORY_LLM_PROCESS = "memory_llm_process"

    def __init__(self) -> None:
        self.jobs = []

    def enqueue(self, **kwargs):
        self.jobs.append(dict(kwargs))
        return "job-runtime-1"


class _FakeRetrievalService:
    def query(self, query: MemoryQuery):
        return ContextPack(), []


def _service(*, should_process: bool = True) -> MemoryService:
    service = MemoryService.__new__(MemoryService)
    service.event_store = _FakeEventStore()
    service.analyzer = SimpleNamespace(analyze=lambda envelope: {"should_process": should_process})
    service.job_queue = _FakeJobQueue()
    service.worker = None
    service.retrieval_service = _FakeRetrievalService()
    service.runtime_session_store = RuntimeSessionStore()
    service.episode_manager = EpisodeManager()
    service._include_pending_facts_in_retrieval = lambda: True
    return service


def test_ingest_updates_runtime_before_worker_and_query_uses_it() -> None:
    service = _service()
    envelope = MemoryEnvelope(
        text="Please fix verbose so tok/sec comes from Ollama?",
        source_kind="user",
        payload_type="message",
        workspace_id="asya",
        session_id="chat-1",
    )

    ingest = service.ingest_event(envelope)
    result = service.query(MemoryQuery(text="what is active?", workspace_id="asya", session_id="chat-1"))

    assert ingest["queued"] is True
    assert ingest["runtime_updated"] is True
    assert ingest["current_episode_id"]
    assert "working_memory" in result.blocks
    assert "Last user turn" in result.blocks["working_memory"]
    assert any(dict(row.get("metadata") or {}).get("source") == "runtime" for row in result.selected)
    assert result.debug["selected_by_source"]["runtime"] >= 1


def test_skipped_ingest_still_updates_runtime_layer() -> None:
    service = _service(should_process=False)
    envelope = MemoryEnvelope(
        text="/status but keep current runtime visible",
        source_kind="user",
        payload_type="message",
        workspace_id="asya",
        session_id="chat-2",
    )

    ingest = service.ingest_event(envelope)
    state = service.runtime_session_store.get_state("asya", "chat-2")

    assert ingest["processed"] is False
    assert ingest["queued"] is False
    assert state["last_user_turn"]["text"] == envelope.text


def test_adapter_exposes_runtime_state_as_structured_query_fields() -> None:
    service = _service()
    service.ingest_event(
        MemoryEnvelope(
            text="Can you keep the verbose fix as the active task?",
            source_kind="user",
            payload_type="message",
            workspace_id="asya",
            session_id="chat-adapter",
        )
    )
    adapter = MemoryCoreAdapter.__new__(MemoryCoreAdapter)
    adapter.service = service
    adapter.config = SimpleNamespace(top_k=8)
    adapter._current_workspace = "asya"
    adapter._current_session = "chat-adapter"

    result = MemoryCoreAdapter.query(
        adapter,
        text="what is pending?",
        workspace_id="asya",
        session_id="chat-adapter",
    )

    assert result["open_questions"]
    assert result["task_continuity"]["active_task"]
    assert result["task_continuity"]["current_episode_id"]


def test_episode_manager_switches_hidden_episode_on_topic_thread_change() -> None:
    service = _service()
    first = MemoryEnvelope(
        text="Let's finish the streaming bug",
        workspace_id="asya",
        session_id="chat-3",
        metadata={"topic_thread_id": "streaming"},
    )
    second = MemoryEnvelope(
        text="Now a different topic: memory runtime layer",
        workspace_id="asya",
        session_id="chat-3",
        metadata={"topic_thread_id": "memory-runtime"},
    )

    first_result = service.ingest_event(first)
    second_result = service.ingest_event(second)

    assert first_result["current_episode_id"] != second_result["current_episode_id"]
    episodes = service.inspect(SimpleNamespace(kind="episodes", workspace_id="asya", limit=10)).get("items", [])
    assert len(episodes) == 2


def test_brain_ingest_metrics_accept_dict_and_object_results() -> None:
    brain = Brain.__new__(Brain)
    summary = Brain._empty_memory_write_summary()

    brain._capture_memory_ingest(summary, {"queued": True, "artifact_ids": []}, bucket="user_turn")
    brain._capture_memory_ingest(
        summary,
        SimpleNamespace(stored_ids=["stored-1"], promoted_ids=["promoted-1"], extracted_facts=["fact"]),
        bucket="assistant_turn",
    )

    assert summary["attempted_writes"] == 2
    assert summary["stored_records"] == 2
    assert summary["user_turns_written"] == 1
    assert summary["assistant_turns_written"] == 1
    assert summary["facts_extracted"] == 1
    assert summary["promotions"] == 1


def test_query_runtime_merge_can_be_disabled_by_flag() -> None:
    service = _service()
    service.ingest_event(
        MemoryEnvelope(
            text="Please keep this runtime note visible",
            source_kind="user",
            payload_type="message",
            workspace_id="asya",
            session_id="chat-flag-off",
        )
    )
    service._include_pending_facts_in_retrieval = lambda: False

    result = service.query(
        MemoryQuery(
            text="what was the latest note?",
            workspace_id="asya",
            session_id="chat-flag-off",
        )
    )

    assert result.debug["include_pending_facts_in_retrieval"] is False
    assert "working_memory" not in result.blocks
    assert not any(dict(row.get("metadata") or {}).get("source") == "runtime" for row in result.selected)


def test_query_runtime_merge_enabled_keeps_runtime_rows() -> None:
    service = _service()
    service.ingest_event(
        MemoryEnvelope(
            text="Can you keep the runtime continuity in mind?",
            source_kind="user",
            payload_type="message",
            workspace_id="asya",
            session_id="chat-flag-on",
        )
    )
    service._include_pending_facts_in_retrieval = lambda: True

    result = service.query(
        MemoryQuery(
            text="runtime continuity",
            workspace_id="asya",
            session_id="chat-flag-on",
        )
    )

    assert result.debug["include_pending_facts_in_retrieval"] is True
    assert "working_memory" in result.blocks
    assert any(dict(row.get("metadata") or {}).get("source") == "runtime" for row in result.selected)


def test_runtime_session_store_cleanup_expired_and_close_task_questions() -> None:
    store = RuntimeSessionStore(session_ttl_sec=1, recent_user_state_ttl_sec=1)
    now = time.time()
    store.update_from_event(
        MemoryEnvelope(
            text="Please fix this bug urgently",
            source_kind="user",
            payload_type="message",
            workspace_id="asya",
            session_id="chat-cleanup",
            ts=now,
        )
    )
    store.update_from_event(
        MemoryEnvelope(
            text="done, fixed, закрыто",
            source_kind="assistant",
            payload_type="message",
            workspace_id="asya",
            session_id="chat-cleanup",
            ts=now + 0.2,
        )
    )
    layer = store.build_query_layer("asya", "chat-cleanup")
    assert not any("active_task" in str(row.get("artifact_id")) for row in layer.get("selected", []))

    removed = store.cleanup_expired(now=now + 2.0)
    assert removed == 1
    assert store.get_state("asya", "chat-cleanup") == {}


def test_recent_user_state_decay_hides_stale_signal() -> None:
    store = RuntimeSessionStore(recent_user_state_ttl_sec=1)
    now = time.time()
    store.update_from_event(
        MemoryEnvelope(
            text="I am very tired today",
            source_kind="user",
            payload_type="message",
            workspace_id="asya",
            session_id="chat-decay",
            ts=now,
        )
    )
    state = store.get_state("asya", "chat-decay")
    state["recent_user_state"]["ts"] = now - 5.0
    store._states[("asya", "chat-decay")].recent_user_state = dict(state["recent_user_state"])

    layer = store.build_query_layer("asya", "chat-decay")
    assert layer["recent_user_state"] == {}
    assert not any("recent_user_state" in str(row.get("artifact_id")) for row in layer.get("selected", []))
