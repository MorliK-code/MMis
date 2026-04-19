from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from threading import RLock
from typing import Any

from memory_core.memory_types import EPISODE_EVENT, enrich_metadata_with_memory_type
from memory_core.schemas import MemoryEnvelope


@dataclass(slots=True)
class RuntimeEpisode:
    episode_id: str
    workspace_id: str
    session_id: str
    title: str
    topic_key: str = ""
    summary: str = ""
    status: str = "active"
    turns: list[dict[str, Any]] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "workspace_id": self.workspace_id,
            "session_id": self.session_id,
            "title": self.title,
            "topic_key": self.topic_key,
            "summary": self.summary,
            "status": self.status,
            "turns": [dict(x) for x in self.turns],
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


class EpisodeManager:
    """
    Hidden episode layer inside one visible chat.

    It uses deterministic heuristics only. The heavy LLM worker can still
    create long-term episode artifacts later, but query() gets the active
    episode immediately.
    """

    def __init__(self, turn_limit: int = 8, idle_timeout_sec: float = 45 * 60):
        self._turn_limit = max(2, int(turn_limit or 8))
        self._idle_timeout_sec = max(60.0, float(idle_timeout_sec or 2700.0))
        self._episodes: dict[str, RuntimeEpisode] = {}
        self._active_by_session: dict[tuple[str, str], str] = {}
        self._lock = RLock()

    def update_from_event(self, envelope: MemoryEnvelope) -> dict[str, Any]:
        workspace = _clean_id(envelope.workspace_id, "global")
        session = _clean_id(envelope.session_id, "default")
        source = str(envelope.source_kind or "user").strip().lower() or "user"
        payload = str(envelope.payload_type or "message").strip().lower() or "message"
        text = str(envelope.text or "").strip()
        meta = dict(envelope.metadata or {})
        ts = float(envelope.ts or time.time())

        with self._lock:
            current = self.get_active_episode(workspace, session)
            topic_key = self._topic_key(meta)
            if current is None or self._should_start_new(current, text, meta, ts):
                current = self._create_episode_locked(workspace, session, text, topic_key)
            elif topic_key and not current.topic_key:
                current.topic_key = topic_key

            if text and payload in {"message", "tool_result", "state_update", "doc_text"}:
                current.turns.append(
                    {
                        "role": source,
                        "text": _truncate_text(text, 500),
                        "event_id": envelope.event_id,
                        "ts": ts,
                    }
                )
                current.turns = current.turns[-self._turn_limit :]
                current.summary = self._summarize_turns(current.turns)
                if not current.title:
                    current.title = self._title_from_text(text)

            current.updated_at = time.time()
            self._active_by_session[(workspace, session)] = current.episode_id
            return {
                "episode_updated": True,
                "episode_id": current.episode_id,
                "workspace_id": workspace,
                "session_id": session,
                "title": current.title,
                "topic_key": current.topic_key,
            }

    def get_active_episode(self, workspace_id: str | None, session_id: str | None) -> RuntimeEpisode | None:
        workspace = _clean_id(workspace_id, "global")
        session = _clean_id(session_id, "default")
        episode_id = self._active_by_session.get((workspace, session))
        if not episode_id:
            return None
        return self._episodes.get(episode_id)

    def get_episode(self, episode_id: str | None) -> dict[str, Any]:
        episode_key = str(episode_id or "").strip()
        with self._lock:
            episode = self._episodes.get(episode_key)
            return episode.to_dict() if episode is not None else {}

    def list_episodes(self, workspace_id: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        workspace = _clean_id(workspace_id, "") if workspace_id else ""
        cap = max(1, int(limit or 50))
        with self._lock:
            rows = []
            for episode in self._episodes.values():
                if workspace and episode.workspace_id != workspace:
                    continue
                rows.append(episode.to_dict())
            rows.sort(key=lambda x: float(x.get("updated_at") or 0.0), reverse=True)
            return rows[:cap]

    def build_query_layer(self, workspace_id: str | None, session_id: str | None) -> dict[str, Any]:
        with self._lock:
            episode = self.get_active_episode(workspace_id, session_id)
            if episode is None:
                return {"blocks": {}, "selected": [], "debug": {"source": "episode_manager", "selected": [], "dropped": []}}
            row = episode.to_dict()

        summary = str(row.get("summary") or "").strip()
        title = str(row.get("title") or "").strip()
        episode_id = str(row.get("episode_id") or "").strip()
        if not summary and not title:
            return {"blocks": {}, "selected": [], "debug": {"source": "episode_manager", "selected": [], "dropped": []}}

        hint = _join_non_empty(
            [
                f"Active hidden episode: {title}" if title else "",
                f"Episode summary: {summary}" if summary else "",
            ]
        )
        blocks = {"continuity_hints": f"- {hint}"} if hint else {}
        metadata = enrich_metadata_with_memory_type(
            {
                "source": "episode",
                "why_selected": "active hidden episode in current session",
                "retrieve_reason": "active hidden episode",
                "session_id": row.get("session_id") or "",
                "current_episode_id": episode_id,
                "topic_key": row.get("topic_key") or "",
            },
            artifact_type=EPISODE_EVENT,
        )
        selected = [
            {
                "artifact_id": f"runtime_episode:{episode_id}",
                "artifact_type": EPISODE_EVENT,
                "source_event_id": "",
                "text": summary or title,
                "summary": summary or title,
                "prompt_view": summary or title,
                "exposure_mode": "prompt_safe",
                "sensitivity": "low",
                "channel": "runtime",
                "score": 0.98,
                "confidence": 0.9,
                "metadata": metadata,
            }
        ]
        return {
            "blocks": blocks,
            "selected": selected,
            "debug": {
                "source": "episode_manager",
                "active_episode": row,
                "selected": [
                    {
                        "artifact_id": selected[0]["artifact_id"],
                        "type": EPISODE_EVENT,
                        "why_selected": "active hidden episode in current session",
                    }
                ],
                "dropped": [],
            },
        }

    def _create_episode_locked(
        self,
        workspace_id: str,
        session_id: str,
        text: str,
        topic_key: str,
    ) -> RuntimeEpisode:
        episode_id = f"episode_{uuid.uuid4().hex[:12]}"
        episode = RuntimeEpisode(
            episode_id=episode_id,
            workspace_id=workspace_id,
            session_id=session_id,
            title=self._title_from_text(text) or "Current episode",
            topic_key=topic_key,
        )
        self._episodes[episode_id] = episode
        self._active_by_session[(workspace_id, session_id)] = episode_id
        return episode

    def _should_start_new(
        self,
        current: RuntimeEpisode,
        text: str,
        metadata: dict[str, Any],
        ts: float,
    ) -> bool:
        topic_key = self._topic_key(metadata)
        if topic_key and current.topic_key and topic_key != current.topic_key:
            return True
        if current.updated_at and ts - float(current.updated_at or 0.0) > self._idle_timeout_sec:
            return True
        low = text.lower()
        explicit_markers = (
            "новая тема",
            "другая тема",
            "сменим тему",
            "вернемся к",
            "кстати",
            "new topic",
            "different topic",
            "by the way",
        )
        return any(marker in low for marker in explicit_markers)

    def _topic_key(self, metadata: dict[str, Any]) -> str:
        for key in ("topic_thread_id", "topic_id", "topic", "active_topic"):
            value = str(metadata.get(key) or "").strip()
            if value:
                return value
        return ""

    def _title_from_text(self, text: str) -> str:
        return _truncate_text(text, 90)

    def _summarize_turns(self, turns: list[dict[str, Any]]) -> str:
        rows = []
        for turn in list(turns or [])[-4:]:
            role = str(turn.get("role") or "").strip() or "turn"
            text = _truncate_text(turn.get("text"), 180)
            if text:
                rows.append(f"{role}: {text}")
        return " | ".join(rows)


def _clean_id(value: Any, default: str) -> str:
    text = str(value or "").strip()
    return text or default


def _truncate_text(value: Any, limit: int) -> str:
    text = " ".join(str(value or "").strip().split())
    cap = max(16, int(limit or 160))
    if len(text) <= cap:
        return text
    return text[: cap - 3].rstrip() + "..."


def _join_non_empty(items: list[str]) -> str:
    return "\n".join(str(item or "").strip() for item in items if str(item or "").strip())
