from __future__ import annotations

import tempfile
from pathlib import Path
from threading import RLock

from memory.memory_manager import MemoryManager


class _FakeEventStore:
    def __init__(self) -> None:
        self.entries = []

    def append(self, row) -> None:
        self.entries.append(dict(row or {}))


def _manager(root: Path | None = None) -> MemoryManager:
    manager = MemoryManager.__new__(MemoryManager)
    temp_dir = Path(root or Path(tempfile.mkdtemp(prefix="mmis-task-continuity-")))
    manager._root = temp_dir
    manager._state_path = temp_dir / "manager_state.json"
    manager._lock = RLock()
    manager._event_store = _FakeEventStore()
    manager._session_summary = ""
    manager._open_questions = []
    manager._current_decisions = []
    manager._active_preferences = []
    manager._task_continuity_by_namespace = {}
    manager._private_runtime = {}
    manager._working_records = []
    return manager


def test_memory_manager_task_continuity_tracks_active_and_previous_tasks() -> None:
    manager = _manager()

    manager.update_task_continuity(
        namespace="conv-task-history",
        active_task={
            "task_id": "task:memory-loop",
            "topic": "memory loop",
            "status": "active",
            "current_goal": "Finish Memory -> Persona -> Prompt wiring.",
            "open_questions": ["how to persist task state"],
            "decisions": ["keep episode planner central"],
        },
        source="episode_continuity_resolved",
        now_ts=10.0,
    )
    manager.update_task_continuity(
        namespace="conv-task-history",
        active_task={
            "task_id": "task:web-isolation",
            "topic": "web isolation",
            "status": "blocked",
            "current_goal": "Finish web isolation cleanup.",
            "decisions": ["keep self facts stronger than web"],
        },
        previous_active_task={
            "task_id": "task:memory-loop",
            "topic": "memory loop",
            "status": "active",
            "current_goal": "Finish Memory -> Persona -> Prompt wiring.",
            "open_questions": ["how to persist task state"],
            "decisions": ["keep episode planner central"],
        },
        source="episode_continuity_resolved",
        now_ts=20.0,
    )

    snapshot = manager.get_task_continuity_snapshot("conv-task-history")

    assert dict(snapshot.get("active_task") or {}).get("task_id") == "task:web-isolation"
    assert dict(snapshot.get("active_task") or {}).get("status") == "blocked"
    previous_tasks = list(snapshot.get("previous_tasks") or [])
    assert list(dict(previous_tasks[0] if previous_tasks else {}).get("open_questions") or []) == [
        "how to persist task state"
    ]
    assert list(snapshot.get("current_decisions") or []) == ["keep self facts stronger than web"]


def test_memory_manager_task_continuity_persists_to_state_file() -> None:
    root = Path(tempfile.mkdtemp(prefix="mmis-task-continuity-persist-"))
    manager = _manager(root=root)

    manager.update_task_continuity(
        namespace="conv-task-persist",
        active_task={
            "task_id": "task:memory-loop",
            "topic": "memory loop",
            "status": "waiting_user",
            "current_goal": "Need a continuity plan.",
            "open_questions": ["how do we persist task state?"],
        },
        source="episode_continuity_resolved",
        now_ts=30.0,
    )

    restored = _manager(root=root)
    restored._load_state()
    snapshot = restored.get_task_continuity_snapshot("conv-task-persist")

    assert dict(snapshot.get("active_task") or {}).get("task_id") == "task:memory-loop"
    assert dict(snapshot.get("active_task") or {}).get("status") == "waiting_user"
    assert list(snapshot.get("open_questions") or []) == ["how do we persist task state?"]
