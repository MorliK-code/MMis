from __future__ import annotations

from memory_core.facade import MemoryService
from memory_core.inspect.memory_inspector import MemoryInspector
from memory_core.schemas import MemoryArtifact, MemoryEnvelope, MemoryInspectRequest, MemoryTrace


class _FakeEventStore:
    def list_events(
        self,
        workspace_id: str | None = None,
        session_id: str | None = None,
        source_kind: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ):
        event = MemoryEnvelope(
            event_id="evt-1",
            source_kind=source_kind or "user",
            payload_type="message",
            text=f"hello:{workspace_id or session_id or 'all'}",
            workspace_id=workspace_id or "global",
            session_id=session_id or "sess-1",
        )
        return [event][offset : offset + limit]

    def count(self):
        return 1

    def get_by_id(self, event_id: str):
        return MemoryEnvelope(
            event_id=event_id,
            source_kind="assistant",
            payload_type="message",
            text="trace-event",
            workspace_id="global",
            session_id="sess-1",
        )


class _FakeArtifactStore:
    def list_artifacts(
        self,
        artifact_type: str | None = None,
        workspace_id: str | None = None,
        status: str | None = None,
        namespace: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ):
        row = MemoryArtifact(
            artifact_id="art-1",
            artifact_type=artifact_type or "profile_fact",
            source_event_id="evt-1",
            text=f"{artifact_type or 'profile_fact'}:{workspace_id or 'global'}:{status or 'active'}",
            summary="artifact",
            workspace_id=workspace_id or "global",
            status=status or "active",
            namespace=namespace or "default",
        )
        return [row][offset : offset + limit]

    def get_by_source_event(self, event_id: str):
        return [
            MemoryArtifact(
                artifact_id="art-trace",
                artifact_type="profile_fact",
                source_event_id=event_id,
                text=f"from:{event_id}",
                summary="trace artifact",
                workspace_id="global",
                status="active",
            )
        ]

    def count(self):
        return 1


class _FakeWorkspaceStore:
    def list_workspaces(self):
        return [{"workspace_id": "global"}]


class _FakeJobQueue:
    def get_stats(self):
        return {"queued": 0, "processing": 0, "done": 0, "dead": 0}

    def list_jobs(self, status=None, limit: int = 50):
        return []


def test_memory_inspector_uses_current_store_api_for_all_public_views() -> None:
    inspector = MemoryInspector(
        event_store=_FakeEventStore(),
        artifact_store=_FakeArtifactStore(),
        workspace_store=_FakeWorkspaceStore(),
        job_queue=_FakeJobQueue(),
    )

    events = inspector.inspect(kind="events", workspace_id="asya", limit=5)
    artifacts = inspector.inspect(kind="artifacts", workspace_id="asya", limit=5)
    workspaces = inspector.inspect(kind="workspaces", limit=5)
    profile = inspector.inspect(kind="profile", workspace_id="asya", limit=5)
    trace = inspector.inspect(kind="trace", event_id="evt-1")

    assert events["items"][0]["workspace_id"] == "asya"
    assert artifacts["items"][0]["artifact_type"] == "profile_fact"
    assert workspaces["workspaces"][0]["workspace_id"] == "global"
    assert profile["profile_facts"][0]["artifact_type"] == "profile_fact"
    assert profile["preferences"][0]["artifact_type"] == "preference"
    assert trace["event"]["event_id"] == "evt-1"
    assert trace["artifacts"][0]["source_event_id"] == "evt-1"


def test_new_memory_inspector_exposes_legacy_methods_for_facade() -> None:
    inspector = MemoryInspector(
        event_store=_FakeEventStore(),
        artifact_store=_FakeArtifactStore(),
        workspace_store=_FakeWorkspaceStore(),
        job_queue=_FakeJobQueue(),
    )

    assert inspector.list_events(limit=5)[0]["event_id"] == "evt-1"
    assert inspector.list_artifacts(limit=5)[0]["artifact_type"] == "profile_fact"
    assert inspector.list_workspaces()[0]["workspace_id"] == "global"
    assert inspector.get_profile_facts(workspace_id="asya", limit=5)[0]["artifact_type"] == "profile_fact"

    trace = inspector.trace_event("evt-1")
    assert isinstance(trace, MemoryTrace)
    assert trace.to_dict()["event"]["event_id"] == "evt-1"


def test_memory_service_inspect_accepts_new_inspector_via_legacy_contract() -> None:
    inspector = MemoryInspector(
        event_store=_FakeEventStore(),
        artifact_store=_FakeArtifactStore(),
        workspace_store=_FakeWorkspaceStore(),
        job_queue=_FakeJobQueue(),
    )
    service = MemoryService.__new__(MemoryService)
    service.inspector = inspector

    result = MemoryService.inspect(service, MemoryInspectRequest(kind="events", limit=5))

    assert "items" in result
    assert result["items"][0]["event_id"] == "evt-1"
