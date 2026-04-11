from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any

from memory_core.schemas import MemoryArtifact, MemoryEnvelope
from memory_core.storage.state_store import StateStore


def _clean_text(value: Any, *, limit: int = 400) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


def _clean_list(value: Any, *, limit: int = 10, item_limit: int = 240) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for item in list(value or []):
        text = _clean_text(item, limit=item_limit)
        if not text:
            continue
        key = text.casefold()
        if key in seen:
            continue
        seen.add(key)
        result.append(text)
        if len(result) >= max(1, int(limit or 10)):
            break
    return result


def _clean_mapping(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    return {
        str(key): item
        for key, item in dict(value).items()
        if item not in (None, "", [], {})
    }


def _normalize_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return None


def _extract_recent_user_state(metadata: dict[str, Any]) -> dict[str, Any]:
    explicit = _clean_mapping(metadata.get("recent_user_state"))
    if explicit:
        sources = _clean_list(list(explicit.get("sources") or []) + ["runtime_session_store"])
        if sources:
            explicit["sources"] = sources
        return explicit

    emotion = _clean_text(metadata.get("emotion"), limit=80).lower()
    low_bandwidth = _normalize_bool(metadata.get("low_bandwidth"))
    frustrated = _normalize_bool(metadata.get("frustrated"))
    if emotion in {"sad", "tired", "anxious"} and low_bandwidth is None:
        low_bandwidth = True
    if emotion in {"frustrated", "angry"} and frustrated is None:
        frustrated = True

    row = {
        "emotion": emotion,
        "low_bandwidth": low_bandwidth,
        "frustrated": frustrated,
    }
    out = {
        key: value
        for key, value in row.items()
        if value not in (None, "", [], {})
    }
    if out:
        out["sources"] = ["runtime_session_store"]
    return out


@dataclass(slots=True)
class RuntimeSessionSnapshot:
    namespace: str = "default"
    workspace_id: str = "global"
    session_id: str = "default"
    last_user_turn: dict[str, Any] = field(default_factory=dict)
    last_assistant_turn: dict[str, Any] = field(default_factory=dict)
    recent_turns: list[dict[str, Any]] = field(default_factory=list)
    active_topic: dict[str, Any] = field(default_factory=dict)
    active_task: dict[str, Any] = field(default_factory=dict)
    open_questions: list[str] = field(default_factory=list)
    recent_decisions: list[str] = field(default_factory=list)
    recent_user_state: dict[str, Any] = field(default_factory=dict)
    current_episode_id: str = ""
    task_history: list[dict[str, Any]] = field(default_factory=list)
    last_turn_ts: float = 0.0
    updated_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "RuntimeSessionSnapshot":
        row = dict(data or {})
        return cls(
            namespace=str(row.get("namespace") or "default").strip() or "default",
            workspace_id=str(row.get("workspace_id") or "global").strip() or "global",
            session_id=str(row.get("session_id") or "default").strip() or "default",
            last_user_turn=_clean_mapping(row.get("last_user_turn")),
            last_assistant_turn=_clean_mapping(row.get("last_assistant_turn")),
            recent_turns=[
                _clean_mapping(item)
                for item in list(row.get("recent_turns") or [])
                if isinstance(item, dict)
            ][:3],
            active_topic=_clean_mapping(row.get("active_topic")),
            active_task=_clean_mapping(row.get("active_task")),
            open_questions=_clean_list(row.get("open_questions"), limit=10),
            recent_decisions=_clean_list(row.get("recent_decisions"), limit=10),
            recent_user_state=_clean_mapping(row.get("recent_user_state")),
            current_episode_id=_clean_text(row.get("current_episode_id"), limit=120),
            task_history=[
                _clean_mapping(item)
                for item in list(row.get("task_history") or [])
                if isinstance(item, dict)
            ][:6],
            last_turn_ts=float(row.get("last_turn_ts") or 0.0),
            updated_at=float(row.get("updated_at") or time.time()),
        )


class RuntimeSessionStore:
    def __init__(self, state_store: StateStore):
        self.state_store = state_store

    @staticmethod
    def _make_key(*, namespace: str, workspace_id: str, session_id: str) -> str:
        clean_namespace = str(namespace or "default").strip() or "default"
        clean_workspace = str(workspace_id or "global").strip() or "global"
        clean_session = str(session_id or "default").strip() or "default"
        return f"runtime_session:{clean_namespace}:{clean_workspace}:{clean_session}"

    def get_session(
        self,
        *,
        namespace: str = "default",
        workspace_id: str = "global",
        session_id: str = "default",
    ) -> RuntimeSessionSnapshot:
        payload = self.state_store.get_state(
            self._make_key(namespace=namespace, workspace_id=workspace_id, session_id=session_id),
            None,
        )
        snapshot = RuntimeSessionSnapshot.from_dict(payload if isinstance(payload, dict) else {})
        snapshot.namespace = str(namespace or snapshot.namespace or "default").strip() or "default"
        snapshot.workspace_id = str(workspace_id or snapshot.workspace_id or "global").strip() or "global"
        snapshot.session_id = str(session_id or snapshot.session_id or "default").strip() or "default"
        return snapshot

    def save_session(self, snapshot: RuntimeSessionSnapshot) -> RuntimeSessionSnapshot:
        snapshot.updated_at = float(time.time())
        self.state_store.set_state(
            self._make_key(
                namespace=snapshot.namespace,
                workspace_id=snapshot.workspace_id,
                session_id=snapshot.session_id,
            ),
            snapshot.to_dict(),
        )
        return snapshot

    def update_from_event(
        self,
        envelope: MemoryEnvelope,
        *,
        current_episode_id: str = "",
    ) -> RuntimeSessionSnapshot:
        snapshot = self.get_session(
            namespace=envelope.namespace,
            workspace_id=envelope.workspace_id,
            session_id=envelope.session_id,
        )
        metadata = dict(envelope.metadata or {})
        turn = self._build_turn_snapshot(envelope)
        if envelope.source_kind == "assistant":
            snapshot.last_assistant_turn = turn
        else:
            snapshot.last_user_turn = turn
            extracted_state = _extract_recent_user_state(metadata)
            if extracted_state:
                snapshot.recent_user_state = extracted_state

        snapshot.recent_turns = (list(snapshot.recent_turns or []) + [turn])[-3:]
        active_topic = self._build_active_topic(metadata, current=snapshot.active_topic)
        if active_topic:
            snapshot.active_topic = active_topic
        if str(current_episode_id or "").strip():
            snapshot.current_episode_id = str(current_episode_id).strip()

        event_task = _clean_mapping(metadata.get("active_task"))
        if event_task:
            snapshot.active_task = event_task
            snapshot.open_questions = _clean_list(event_task.get("open_questions"), limit=10)
            snapshot.recent_decisions = _clean_list(event_task.get("decisions"), limit=10)
        else:
            event_questions = _clean_list(metadata.get("open_questions"), limit=10)
            event_decisions = _clean_list(
                metadata.get("recent_decisions") or metadata.get("current_decisions") or metadata.get("decisions"),
                limit=10,
            )
            if event_questions:
                snapshot.open_questions = event_questions
            if event_decisions:
                snapshot.recent_decisions = event_decisions

        snapshot.last_turn_ts = float(envelope.ts or time.time())
        return self.save_session(snapshot)

    def update_task_continuity(
        self,
        *,
        namespace: str = "default",
        workspace_id: str = "global",
        session_id: str = "default",
        active_task: dict[str, Any] | None,
        previous_active_task: dict[str, Any] | None = None,
        source: str = "",
        now_ts: float | None = None,
    ) -> dict[str, Any]:
        snapshot = self.get_session(
            namespace=namespace,
            workspace_id=workspace_id,
            session_id=session_id,
        )
        clean_previous = self._normalize_task(previous_active_task)
        clean_active = self._normalize_task(active_task)

        if clean_previous:
            previous_id = str(clean_previous.get("task_id") or "").strip()
            active_id = str(clean_active.get("task_id") or "").strip()
            if previous_id and previous_id != active_id:
                history = list(snapshot.task_history or [])
                history.insert(
                    0,
                    {
                        **clean_previous,
                        "updated_at": float(now_ts or time.time()),
                        "source": str(source or "").strip(),
                    },
                )
                snapshot.task_history = history[:6]

        snapshot.active_task = clean_active
        snapshot.open_questions = _clean_list(clean_active.get("open_questions"), limit=10)
        snapshot.recent_decisions = _clean_list(clean_active.get("decisions"), limit=10)
        snapshot.last_turn_ts = float(now_ts or snapshot.last_turn_ts or time.time())
        self.save_session(snapshot)
        return self.build_task_continuity(snapshot=snapshot, source=source, now_ts=now_ts)

    def build_task_continuity(
        self,
        *,
        snapshot: RuntimeSessionSnapshot,
        source: str = "",
        now_ts: float | None = None,
    ) -> dict[str, Any]:
        return {
            "active_task": dict(snapshot.active_task or {}),
            "task_history": [dict(item) for item in list(snapshot.task_history or []) if isinstance(item, dict)],
            "source": str(source or "").strip(),
            "updated_at": float(now_ts or snapshot.updated_at or time.time()),
        }

    def list_sessions(self, *, limit: int = 50) -> list[dict[str, Any]]:
        keys = self.state_store.list_keys(prefix="runtime_session:")
        rows: list[dict[str, Any]] = []
        for key in list(keys or []):
            payload = self.state_store.get_state(key, None)
            if not isinstance(payload, dict):
                continue
            snapshot = RuntimeSessionSnapshot.from_dict(payload)
            rows.append(snapshot.to_dict())
        rows.sort(key=lambda item: float(item.get("updated_at") or item.get("last_turn_ts") or 0.0), reverse=True)
        return rows[: max(1, int(limit or 50))]

    def to_retrieval_artifacts(
        self,
        snapshot: RuntimeSessionSnapshot,
        *,
        include_pending: bool = False,
        topic_details: dict[str, Any] | None = None,
    ) -> list[MemoryArtifact]:
        now = float(snapshot.last_turn_ts or snapshot.updated_at or time.time())
        metadata_base = {
            "session_id": snapshot.session_id,
            "runtime_session_id": snapshot.session_id,
            "retrieval_source": "runtime_session",
            "confidence": 0.96,
            "decay": "fast",
        }
        artifacts: list[MemoryArtifact] = []

        merged_open_questions = _clean_list(
            list(snapshot.open_questions or []) + list(dict(topic_details or {}).get("open_questions") or []),
            limit=10,
        )
        merged_decisions = _clean_list(
            list(snapshot.recent_decisions or []) + list(dict(topic_details or {}).get("current_decisions") or []),
            limit=10,
        )

        active_topic = dict(snapshot.active_topic or {})
        thread = dict(dict(topic_details or {}).get("thread") or {})
        topic_title = _clean_text(
            active_topic.get("title")
            or active_topic.get("topic_title")
            or thread.get("title")
            or active_topic.get("topic_key")
            or thread.get("topic_key")
            or "",
            limit=180,
        )
        topic_summary = _clean_text(
            dict(topic_details or {}).get("summary")
            or active_topic.get("summary")
            or thread.get("summary")
            or "",
            limit=240,
        )
        if topic_title or topic_summary or snapshot.current_episode_id:
            summary_parts: list[str] = []
            if topic_title:
                summary_parts.append(f"Active topic: {topic_title}")
            if topic_summary:
                summary_parts.append(topic_summary)
            if merged_decisions:
                summary_parts.append("Recent decisions: " + "; ".join(merged_decisions[:2]))
            if merged_open_questions:
                summary_parts.append("Open questions: " + "; ".join(merged_open_questions[:2]))
            artifacts.append(
                MemoryArtifact(
                    artifact_id=f"runtime:{snapshot.session_id}:active-episode",
                    artifact_type="episode_event",
                    source_event_id="runtime_session_store",
                    text=topic_title or snapshot.current_episode_id or "active episode",
                    summary=" | ".join(summary_parts),
                    metadata={
                        **metadata_base,
                        "retrieval_source": "active_episode",
                        "episode_id": str(snapshot.current_episode_id or thread.get("thread_id") or "").strip(),
                        "topic_thread_id": _clean_text(
                            active_topic.get("thread_id") or thread.get("thread_id") or "",
                            limit=120,
                        ),
                        "topic_key": _clean_text(active_topic.get("topic_key") or thread.get("topic_key") or "", limit=120),
                        "topic_title": topic_title,
                        "open_questions": merged_open_questions,
                        "current_decisions": merged_decisions,
                        "related_topic_ids": _clean_list(
                            active_topic.get("related_topic_ids") or thread.get("related_thread_ids") or [],
                            limit=8,
                            item_limit=120,
                        ),
                    },
                    namespace=snapshot.namespace,
                    workspace_id=snapshot.workspace_id,
                    status="active",
                    created_at=now,
                    updated_at=now,
                )
            )

        if snapshot.active_task:
            active_task = dict(snapshot.active_task or {})
            summary_parts = []
            if _clean_text(active_task.get("current_goal"), limit=220):
                summary_parts.append(f"Active task: {_clean_text(active_task.get('current_goal'), limit=220)}")
            if merged_decisions:
                summary_parts.append("Decisions: " + "; ".join(merged_decisions[:2]))
            if merged_open_questions:
                summary_parts.append("Open questions: " + "; ".join(merged_open_questions[:2]))
            artifacts.append(
                MemoryArtifact(
                    artifact_id=f"runtime:{snapshot.session_id}:active-task",
                    artifact_type="task_state",
                    source_event_id="runtime_session_store",
                    text=_clean_text(
                        active_task.get("current_goal")
                        or active_task.get("summary_short")
                        or active_task.get("topic")
                        or "active task",
                        limit=240,
                    ),
                    summary=" | ".join(summary_parts),
                    metadata={
                        **metadata_base,
                        "retrieval_source": "active_task",
                        "task_status": _clean_text(active_task.get("status") or "open", limit=40) or "open",
                        "task_id": _clean_text(active_task.get("task_id"), limit=120),
                        "topic": _clean_text(active_task.get("topic"), limit=160),
                        "decisions": merged_decisions,
                        "open_questions": merged_open_questions,
                    },
                    namespace=snapshot.namespace,
                    workspace_id=snapshot.workspace_id,
                    status="active",
                    created_at=now,
                    updated_at=now,
                )
            )

        if snapshot.recent_user_state:
            state = dict(snapshot.recent_user_state or {})
            emotion = _clean_text(state.get("emotion"), limit=80)
            state_summary = "Runtime user state hint"
            if emotion:
                state_summary = f"Runtime user state: {emotion}"
            artifacts.append(
                MemoryArtifact(
                    artifact_id=f"runtime:{snapshot.session_id}:recent-user-state",
                    artifact_type="emotional_state",
                    source_event_id="runtime_session_store",
                    text=state_summary,
                    summary=state_summary,
                    metadata={
                        **metadata_base,
                        "retrieval_source": "runtime_user_state",
                        **state,
                    },
                    namespace=snapshot.namespace,
                    workspace_id=snapshot.workspace_id,
                    status="active",
                    created_at=now,
                    updated_at=now,
                )
            )

        if include_pending:
            recent_turns = list(snapshot.recent_turns or [])[-3:]
            for index, row in enumerate(recent_turns, start=1):
                role = _clean_text(row.get("source_kind") or "turn", limit=40).lower() or "turn"
                text = _clean_text(row.get("text"), limit=220)
                if not text:
                    continue
                artifacts.append(
                    MemoryArtifact(
                        artifact_id=f"runtime:{snapshot.session_id}:pending:{index}",
                        artifact_type="episode_event",
                        source_event_id=str(row.get("event_id") or "runtime_session_store"),
                        text=text,
                        summary=f"Recent {role} turn: {text}",
                        metadata={
                            **metadata_base,
                            "retrieval_source": "pending_turn",
                            "source_kind": role,
                            "ts": float(row.get("ts") or now),
                        },
                        namespace=snapshot.namespace,
                        workspace_id=snapshot.workspace_id,
                        status="active",
                        created_at=float(row.get("ts") or now),
                        updated_at=float(row.get("ts") or now),
                    )
                )

        return artifacts

    @staticmethod
    def _build_turn_snapshot(envelope: MemoryEnvelope) -> dict[str, Any]:
        metadata = dict(envelope.metadata or {})
        return {
            "event_id": str(envelope.event_id or "").strip(),
            "source_kind": str(envelope.source_kind or "").strip(),
            "payload_type": str(envelope.payload_type or "").strip(),
            "text": _clean_text(envelope.text, limit=280),
            "ts": float(envelope.ts or time.time()),
            "topic_thread_id": _clean_text(metadata.get("topic_thread_id"), limit=120),
            "topic_title": _clean_text(metadata.get("topic_title") or metadata.get("topic"), limit=160),
        }

    @staticmethod
    def _build_active_topic(metadata: dict[str, Any], *, current: dict[str, Any] | None = None) -> dict[str, Any]:
        current_row = dict(current or {})
        candidate = {
            "thread_id": _clean_text(metadata.get("topic_thread_id"), limit=120),
            "topic_key": _clean_text(metadata.get("topic_key") or metadata.get("topic"), limit=120),
            "title": _clean_text(
                metadata.get("topic_title")
                or metadata.get("topic_thread_title")
                or metadata.get("topic")
                or "",
                limit=160,
            ),
            "visible_chat_id": _clean_text(
                metadata.get("visible_chat_id") or metadata.get("conversation_id") or "",
                limit=120,
            ),
            "related_topic_ids": _clean_list(metadata.get("related_topic_thread_ids"), limit=8, item_limit=120),
        }
        merged = dict(current_row)
        for key, value in candidate.items():
            if value not in ("", [], {}, None):
                merged[key] = value
        return _clean_mapping(merged)

    @staticmethod
    def _normalize_task(value: dict[str, Any] | None) -> dict[str, Any]:
        task = _clean_mapping(value)
        if not task:
            return {}
        normalized = {
            "task_id": _clean_text(task.get("task_id"), limit=120),
            "topic": _clean_text(task.get("topic"), limit=160),
            "status": _clean_text(task.get("status") or "active", limit=40) or "active",
            "summary_short": _clean_text(task.get("summary_short"), limit=220),
            "current_goal": _clean_text(task.get("current_goal"), limit=240),
            "source_episode_id": _clean_text(task.get("source_episode_id"), limit=120),
            "decisions": _clean_list(task.get("decisions"), limit=10, item_limit=220),
            "open_questions": _clean_list(task.get("open_questions"), limit=10, item_limit=220),
            "next_steps": _clean_list(task.get("next_steps"), limit=10, item_limit=220),
        }
        return _clean_mapping(normalized)
