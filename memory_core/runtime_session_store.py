from __future__ import annotations

import time
from dataclasses import dataclass, field
from threading import RLock
from typing import Any

from memory_core.memory_types import (
    EMOTIONAL_STATE,
    EPISODE_EVENT,
    TASK_STATE,
    enrich_metadata_with_memory_type,
)
from memory_core.schemas import MemoryEnvelope


@dataclass(slots=True)
class RuntimeTurn:
    role: str
    text: str
    event_id: str
    ts: float
    memory_type: str = EPISODE_EVENT

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "text": self.text,
            "event_id": self.event_id,
            "ts": self.ts,
            "memory_type": self.memory_type,
        }


@dataclass(slots=True)
class RuntimeSessionState:
    workspace_id: str
    session_id: str
    last_user_turn: dict[str, Any] | None = None
    last_assistant_turn: dict[str, Any] | None = None
    active_topic: str = ""
    active_task: dict[str, Any] | None = None
    open_questions: list[dict[str, Any]] = field(default_factory=list)
    recent_decisions: list[dict[str, Any]] = field(default_factory=list)
    recent_user_state: dict[str, Any] = field(default_factory=dict)
    current_episode_id: str = ""
    last_turn_ts: float = 0.0
    recent_turns: list[dict[str, Any]] = field(default_factory=list)
    updated_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "workspace_id": self.workspace_id,
            "session_id": self.session_id,
            "last_user_turn": dict(self.last_user_turn or {}),
            "last_assistant_turn": dict(self.last_assistant_turn or {}),
            "active_topic": self.active_topic,
            "active_task": dict(self.active_task or {}),
            "open_questions": [dict(x) for x in self.open_questions],
            "recent_decisions": [dict(x) for x in self.recent_decisions],
            "recent_user_state": dict(self.recent_user_state or {}),
            "current_episode_id": self.current_episode_id,
            "last_turn_ts": self.last_turn_ts,
            "recent_turns": [dict(x) for x in self.recent_turns],
            "updated_at": self.updated_at,
        }


class RuntimeSessionStore:
    """
    Synchronous working memory for the current chat session.

    This store is intentionally cheap: no embeddings, no worker, no LLM.
    It exists so the next retrieval can see the live conversation state
    immediately after ingest_event().
    """

    def __init__(
        self,
        recent_turn_limit: int = 8,
        question_limit: int = 6,
        decision_limit: int = 6,
        session_ttl_sec: int = 4 * 3600,
        recent_user_state_ttl_sec: int = 20 * 60,
    ):
        self._recent_turn_limit = max(2, int(recent_turn_limit or 8))
        self._question_limit = max(1, int(question_limit or 6))
        self._decision_limit = max(1, int(decision_limit or 6))
        self._session_ttl_sec = max(1, int(session_ttl_sec or 4 * 3600))
        self._recent_user_state_ttl_sec = max(1, int(recent_user_state_ttl_sec or 20 * 60))
        self._states: dict[tuple[str, str], RuntimeSessionState] = {}
        self._lock = RLock()

    def update_from_event(
        self,
        envelope: MemoryEnvelope,
        *,
        current_episode_id: str | None = None,
    ) -> dict[str, Any]:
        workspace = _clean_id(envelope.workspace_id, "global")
        session = _clean_id(envelope.session_id, "default")
        source = str(envelope.source_kind or "user").strip().lower() or "user"
        payload = str(envelope.payload_type or "message").strip().lower() or "message"
        text = str(envelope.text or "").strip()
        meta = dict(envelope.metadata or {})
        ts = float(envelope.ts or time.time())

        with self._lock:
            self.cleanup_expired(now=ts)
            state = self._get_or_create_locked(workspace, session)
            if current_episode_id:
                state.current_episode_id = str(current_episode_id)

            if text and payload in {"message", "tool_result", "state_update", "doc_text"}:
                turn = RuntimeTurn(
                    role=source,
                    text=text,
                    event_id=envelope.event_id,
                    ts=ts,
                ).to_dict()
                state.recent_turns.append(turn)
                state.recent_turns = state.recent_turns[-self._recent_turn_limit :]
                state.last_turn_ts = ts

                if source == "user":
                    state.last_user_turn = dict(turn)
                    self._clear_irrelevant_questions_on_topic_switch(state, text)
                    self._capture_open_question(state, text, envelope.event_id, ts)
                    self._capture_recent_user_state(state, text, meta, ts)
                elif source == "assistant":
                    state.last_assistant_turn = dict(turn)
                    self._close_answered_question(state, text)

                self._capture_decision(state, text, envelope.event_id, ts)

            topic = self._extract_topic(meta, text)
            if topic:
                state.active_topic = topic

            task = self._extract_task(meta, text, source)
            if task == {}:
                state.active_task = None
            elif task:
                state.active_task = task

            state.updated_at = time.time()
            return {
                "runtime_updated": True,
                "workspace_id": workspace,
                "session_id": session,
                "current_episode_id": state.current_episode_id,
                "active_topic": state.active_topic,
                "open_questions": len(state.open_questions),
                "recent_decisions": len(state.recent_decisions),
            }

    def get_state(self, workspace_id: str | None, session_id: str | None) -> dict[str, Any]:
        workspace = _clean_id(workspace_id, "global")
        session = _clean_id(session_id, "default")
        with self._lock:
            self.cleanup_expired()
            state = self._states.get((workspace, session))
            return state.to_dict() if state is not None else {}

    def list_states(self, workspace_id: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        workspace = _clean_id(workspace_id, "") if workspace_id else ""
        cap = max(1, int(limit or 50))
        with self._lock:
            self.cleanup_expired()
            rows = []
            for state in self._states.values():
                if workspace and state.workspace_id != workspace:
                    continue
                rows.append(state.to_dict())
            rows.sort(key=lambda x: float(x.get("updated_at") or 0.0), reverse=True)
            return rows[:cap]

    def build_query_layer(self, workspace_id: str | None, session_id: str | None) -> dict[str, Any]:
        state = self.get_state(workspace_id, session_id)
        if not state:
            return {"blocks": {}, "selected": [], "recent_user_state": {}, "debug": {}}

        blocks: dict[str, str] = {}
        selected: list[dict[str, Any]] = []
        continuity_lines: list[str] = []
        working_lines: list[str] = []
        unresolved_lines: list[str] = []

        active_topic = str(state.get("active_topic") or "").strip()
        current_episode_id = str(state.get("current_episode_id") or "").strip()
        if active_topic:
            continuity_lines.append(f"Active topic: {active_topic}")
        if current_episode_id:
            continuity_lines.append(f"Active hidden episode: {current_episode_id}")

        last_user = dict(state.get("last_user_turn") or {})
        if last_user.get("text"):
            text = _truncate_text(last_user.get("text"), 320)
            working_lines.append(f"Last user turn: {text}")
            selected.append(
                self._selected_row(
                    state,
                    name="last_user_turn",
                    text=text,
                    source_event_id=str(last_user.get("event_id") or ""),
                    memory_type=EPISODE_EVENT,
                    reason="latest user turn in runtime session",
                    score=1.0,
                )
            )

        last_assistant = dict(state.get("last_assistant_turn") or {})
        if last_assistant.get("text"):
            text = _truncate_text(last_assistant.get("text"), 320)
            working_lines.append(f"Last assistant turn: {text}")
            selected.append(
                self._selected_row(
                    state,
                    name="last_assistant_turn",
                    text=text,
                    source_event_id=str(last_assistant.get("event_id") or ""),
                    memory_type=EPISODE_EVENT,
                    reason="latest assistant turn in runtime session",
                    score=0.92,
                )
            )

        active_task = dict(state.get("active_task") or {})
        if active_task:
            task_text = str(active_task.get("text") or active_task.get("title") or "").strip()
            if task_text:
                task_text = _truncate_text(task_text, 260)
                working_lines.append(f"Active task: {task_text}")
                selected.append(
                    self._selected_row(
                        state,
                        name="active_task",
                        text=task_text,
                        source_event_id=str(active_task.get("event_id") or ""),
                        memory_type=TASK_STATE,
                        reason="active task in runtime session",
                        score=0.95,
                    )
                )

        for item in list(state.get("open_questions") or [])[: self._question_limit]:
            question = _truncate_text(dict(item).get("text"), 220)
            if question:
                unresolved_lines.append(f"Open question: {question}")

        for item in list(state.get("recent_decisions") or [])[: self._decision_limit]:
            decision = _truncate_text(dict(item).get("text"), 220)
            if decision:
                working_lines.append(f"Recent decision: {decision}")

        if continuity_lines:
            blocks["continuity_hints"] = "\n".join(f"- {line}" for line in continuity_lines)
        if working_lines:
            blocks["working_memory"] = "\n".join(f"- {line}" for line in working_lines)
        if unresolved_lines:
            blocks["unresolved_items"] = "\n".join(f"- {line}" for line in unresolved_lines)

        recent_user_state = dict(state.get("recent_user_state") or {})
        recent_user_state = self._live_recent_user_state(state, recent_user_state)
        if recent_user_state:
            selected.append(
                self._selected_row(
                    state,
                    name="recent_user_state",
                    text=_truncate_text(recent_user_state.get("text") or recent_user_state.get("signal"), 220),
                    source_event_id=str(recent_user_state.get("event_id") or ""),
                    memory_type=EMOTIONAL_STATE,
                    reason="recent user state in runtime session",
                    score=0.75,
                )
            )

        return {
            "blocks": blocks,
            "selected": selected,
            "recent_user_state": recent_user_state,
            "debug": {
                "source": "runtime_session_store",
                "selected": [
                    {
                        "artifact_id": row.get("artifact_id"),
                        "type": dict(row.get("metadata") or {}).get("memory_type"),
                        "why_selected": dict(row.get("metadata") or {}).get("why_selected"),
                    }
                    for row in selected
                ],
                "dropped": [],
            },
        }

    def _get_or_create_locked(self, workspace_id: str, session_id: str) -> RuntimeSessionState:
        key = (workspace_id, session_id)
        state = self._states.get(key)
        if state is None:
            state = RuntimeSessionState(workspace_id=workspace_id, session_id=session_id)
            self._states[key] = state
        return state

    def _selected_row(
        self,
        state: dict[str, Any],
        *,
        name: str,
        text: str,
        source_event_id: str,
        memory_type: str,
        reason: str,
        score: float,
    ) -> dict[str, Any]:
        workspace = str(state.get("workspace_id") or "global")
        session = str(state.get("session_id") or "default")
        artifact_id = f"runtime:{workspace}:{session}:{name}"
        metadata = enrich_metadata_with_memory_type(
            {
                "source": "runtime",
                "why_selected": reason,
                "retrieve_reason": reason,
                "session_id": session,
                "current_episode_id": state.get("current_episode_id") or "",
            },
            artifact_type=memory_type,
        )
        return {
            "artifact_id": artifact_id,
            "artifact_type": memory_type,
            "source_event_id": source_event_id,
            "text": text,
            "summary": text,
            "prompt_view": text,
            "exposure_mode": "prompt_safe",
            "sensitivity": "low",
            "channel": "runtime",
            "score": score,
            "confidence": 0.9,
            "metadata": metadata,
        }

    def _extract_topic(self, metadata: dict[str, Any], text: str) -> str:
        for key in ("active_topic", "topic", "topic_title", "topic_thread_title", "topic_thread_id"):
            value = str(metadata.get(key) or "").strip()
            if value:
                return _truncate_text(value, 120)
        if len(text) >= 24:
            return _truncate_text(text, 80)
        return ""

    def _extract_task(self, metadata: dict[str, Any], text: str, source: str) -> dict[str, Any] | None:
        low = text.lower()
        close_markers = ("готово", "сделано", "закрыто", "не актуально", "done", "fixed")
        if source in {"user", "assistant"} and any(marker in low for marker in close_markers):
            return {}

        for key in ("active_task", "task", "task_state"):
            value = metadata.get(key)
            if isinstance(value, dict):
                return dict(value)
            if str(value or "").strip():
                return {"text": str(value).strip(), "source": "metadata"}

        task_markers = (
            "fix",
            "todo",
            "plan",
            "implement",
            "bug",
            "ошибка",
            "исправ",
            "реализ",
            "план",
            "задач",
        )
        if source == "user" and any(marker in low for marker in task_markers):
            return {"text": _truncate_text(text, 260), "source": "heuristic"}
        return None

    def _capture_open_question(self, state: RuntimeSessionState, text: str, event_id: str, ts: float) -> None:
        low = text.lower()
        close_markers = ("решено", "понятно", "ответили", "спасибо, ясно")
        if any(marker in low for marker in close_markers):
            state.open_questions = []
            return
        question_markers = ("?", "можешь", "как ", "почему", "что ", "why ", "how ", "can you")
        if not any(marker in low for marker in question_markers):
            return
        row = {"text": _truncate_text(text, 260), "event_id": event_id, "ts": ts}
        state.open_questions = _dedupe_by_text([row, *state.open_questions])[: self._question_limit]

    def _capture_decision(self, state: RuntimeSessionState, text: str, event_id: str, ts: float) -> None:
        low = text.lower()
        markers = ("решили", "сделал", "готово", "fixed", "done", "decision", "decided")
        if not any(marker in low for marker in markers):
            return
        row = {"text": _truncate_text(text, 260), "event_id": event_id, "ts": ts}
        state.recent_decisions = _dedupe_by_text([row, *state.recent_decisions])[: self._decision_limit]
        close_task_markers = ("готово", "сделано", "закрыто", "не актуально", "done", "fixed")
        if any(marker in low for marker in close_task_markers):
            state.active_task = None

    def _capture_recent_user_state(
        self,
        state: RuntimeSessionState,
        text: str,
        metadata: dict[str, Any],
        ts: float,
    ) -> None:
        user_state = metadata.get("user_state")
        if isinstance(user_state, dict):
            state.recent_user_state = {
                **dict(user_state),
                "ts": ts,
                "current_episode_id": state.current_episode_id,
            }
            return

        low = text.lower()
        signal = ""
        if any(token in low for token in ("устал", "выгор", "tired", "exhausted")):
            signal = "tired"
        elif any(token in low for token in ("бесит", "раздраж", "злю", "frustrated", "angry")):
            signal = "frustrated"
        elif any(token in low for token in ("рад", "ура", "класс", "great", "finally")):
            signal = "relieved"
        if signal:
            state.recent_user_state = {
                "signal": signal,
                "text": _truncate_text(text, 220),
                "ts": ts,
                "current_episode_id": state.current_episode_id,
            }

    def _clear_irrelevant_questions_on_topic_switch(self, state: RuntimeSessionState, text: str) -> None:
        if not state.open_questions:
            return
        low = str(text or "").lower()
        switch_markers = (
            "new topic",
            "другая тема",
            "теперь другой",
            "switch topic",
            "сменим тему",
        )
        if any(marker in low for marker in switch_markers):
            state.open_questions = []

    def _close_answered_question(self, state: RuntimeSessionState, assistant_text: str) -> None:
        if not state.open_questions:
            return
        answer = str(assistant_text or "").strip().lower()
        if not answer:
            return
        remained: list[dict[str, Any]] = []
        for item in state.open_questions:
            text = str(item.get("text") or "").strip()
            if not text:
                continue
            normalized = text.lower().rstrip("?")
            probe = " ".join(normalized.split()[:5])
            if probe and probe in answer:
                continue
            remained.append(dict(item))
        state.open_questions = remained[: self._question_limit]

    def cleanup_expired(self, *, now: float | None = None) -> int:
        now_ts = float(now or time.time())
        removed = 0
        for key, state in list(self._states.items()):
            if now_ts - float(state.updated_at or 0.0) > float(self._session_ttl_sec):
                self._states.pop(key, None)
                removed += 1
        return removed

    def _live_recent_user_state(self, state: dict[str, Any], recent_state: dict[str, Any]) -> dict[str, Any]:
        if not recent_state:
            return {}
        now_ts = time.time()
        state_ts = float(recent_state.get("ts") or 0.0)
        if not state_ts or now_ts - state_ts > float(self._recent_user_state_ttl_sec):
            return {}
        state_episode = str(state.get("current_episode_id") or "").strip()
        recent_episode = str(recent_state.get("current_episode_id") or "").strip()
        follow_up = bool(recent_state.get("follow_up"))
        if recent_episode and state_episode and recent_episode != state_episode and not follow_up:
            return {}
        return dict(recent_state)


def _clean_id(value: Any, default: str) -> str:
    text = str(value or "").strip()
    return text or default


def _truncate_text(value: Any, limit: int) -> str:
    text = " ".join(str(value or "").strip().split())
    cap = max(16, int(limit or 160))
    if len(text) <= cap:
        return text
    return text[: cap - 3].rstrip() + "..."


def _dedupe_by_text(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in items:
        text = str(item.get("text") or "").strip().lower()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(dict(item))
    return out
