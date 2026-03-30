from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from memory_core.schemas import MemoryArtifact


def _unique_strings(items: list[str] | None) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in list(items or []):
        value = str(item or "").strip()
        if not value or value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


@dataclass(slots=True)
class TopicThread:
    thread_id: str
    visible_chat_id: str
    workspace_id: str
    session_id: str
    topic_key: str = ""
    title: str = ""
    status: str = "active"
    summary: str = ""
    tags: list[str] = field(default_factory=list)
    related_thread_ids: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        *,
        visible_chat_id: str,
        workspace_id: str,
        session_id: str,
        topic_key: str,
        title: str,
        tags: list[str] | None = None,
        related_thread_ids: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "TopicThread":
        now = time.time()
        return cls(
            thread_id=f"thr_{uuid.uuid4().hex[:12]}",
            visible_chat_id=str(visible_chat_id or "").strip() or "default",
            workspace_id=str(workspace_id or "").strip() or "global",
            session_id=str(session_id or "").strip() or "default",
            topic_key=str(topic_key or "").strip(),
            title=str(title or "").strip(),
            tags=_unique_strings(tags),
            related_thread_ids=_unique_strings(related_thread_ids),
            created_at=now,
            updated_at=now,
            metadata=dict(metadata or {}),
        )

    def to_artifact(self) -> MemoryArtifact:
        return MemoryArtifact(
            artifact_id=self.thread_id,
            artifact_type="topic_thread",
            source_event_id="topic_router",
            text=self.title or self.topic_key or self.thread_id,
            summary=self.summary,
            metadata={
                "thread_id": self.thread_id,
                "visible_chat_id": self.visible_chat_id,
                "workspace_id": self.workspace_id,
                "session_id": self.session_id,
                "topic_key": self.topic_key,
                "title": self.title,
                "status": self.status,
                "tags": _unique_strings(self.tags),
                "related_thread_ids": _unique_strings(self.related_thread_ids),
                **dict(self.metadata or {}),
            },
            namespace="default",
            workspace_id=self.workspace_id,
            status="active" if self.status == "active" else "archived",
            created_at=self.created_at,
            updated_at=self.updated_at,
        )

    @classmethod
    def from_artifact(cls, artifact: MemoryArtifact) -> "TopicThread":
        meta = dict(artifact.metadata or {})
        return cls(
            thread_id=str(meta.get("thread_id") or artifact.artifact_id or "").strip(),
            visible_chat_id=str(meta.get("visible_chat_id") or meta.get("session_id") or "").strip() or "default",
            workspace_id=str(meta.get("workspace_id") or artifact.workspace_id or "").strip() or "global",
            session_id=str(meta.get("session_id") or "").strip() or "default",
            topic_key=str(meta.get("topic_key") or "").strip(),
            title=str(meta.get("title") or artifact.text or "").strip(),
            status=str(meta.get("status") or artifact.status or "active").strip() or "active",
            summary=str(artifact.summary or "").strip(),
            tags=_unique_strings(meta.get("tags")),
            related_thread_ids=_unique_strings(meta.get("related_thread_ids")),
            created_at=float(artifact.created_at or time.time()),
            updated_at=float(artifact.updated_at or artifact.created_at or time.time()),
            metadata={
                key: value
                for key, value in meta.items()
                if key
                not in {
                    "thread_id",
                    "visible_chat_id",
                    "workspace_id",
                    "session_id",
                    "topic_key",
                    "title",
                    "status",
                    "tags",
                    "related_thread_ids",
                }
            },
        )


@dataclass(slots=True)
class TopicRouteDecision:
    thread_id: str
    topic_key: str
    title: str
    reason: str
    score: float
    is_new_thread: bool = False
    related_thread_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
