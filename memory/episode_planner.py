from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any

from modules.nlu.normalizer import normalize_text


_CONTINUE_RE = re.compile(
    r"^(?:ну|ладно|давай|погнали|ок(?:ей)?|дальше|продолжай|go on|continue|next)\b",
    re.I,
)
_SWITCH_TOPIC_RE = re.compile(
    r"(?:кстати|другая тема|новая тема|теперь про|забей|неважно|сменим тему|by the way|another topic|new topic)",
    re.I,
)
_CLOSE_RE = re.compile(
    r"(?:\bdone\b|\bresolved\b|\bfinished\b|\bthat's it\b|\bwe are done\b|\bготово\b|\bсделано\b|\bразобрались\b|\bзакончили\b|\bзакрыли\b|\bвсё\b|\bвсе\b)",
    re.I,
)
_BLOCKED_RE = re.compile(
    r"(?:\bblocked\b|\bstuck\b|\bcan't continue\b|\bне получается\b|\bзастряли\b|\bуперлись\b|\bупёрлись\b|\bне можем дальше\b)",
    re.I,
)
_WAITING_USER_RE = re.compile(
    r"(?:\bwaiting\b|\bneed your input\b|\banswer first\b|\bжду\b|\bнужен твой ответ\b|\bответь сначала\b)",
    re.I,
)
_RETURN_RE = re.compile(
    r"(?:\breturn to\b|\bback to\b|\bgo back to\b|\bвернись\b|\bвернемся\b|\bвернёмся\b|\bтот план\b|\bтому плану\b)",
    re.I,
)
_RETURN_PREVIOUS_RE = re.compile(
    r"(?:\bprevious task\b|\bearlier task\b|\blast task\b|\bprevious plan\b|\bearlier plan\b|"
    r"\bРїСЂРѕС€Р»(?:РѕР№|Р°СЏ|СѓСЋ)\s+Р·Р°РґР°С‡\w*\b|\bРїСЂРµРґС‹РґСѓС‰\w*\s+Р·Р°РґР°С‡\w*\b|"
    r"\bРїСЂРѕС€Р»(?:РѕРјСѓ|С‹Р№)\s+РїР»Р°РЅ\w*\b|\bРїСЂРµРґС‹РґСѓС‰\w*\s+РїР»Р°РЅ\w*\b)",
    re.I,
)
_TASK_BULLET_RE = re.compile(r"^\s*(?:[-*]|\d+[.)])\s*(.+?)\s*$")
_TASK_DECISION_RE = re.compile(
    r"(?:\b(?:decided|agreed|we will|let'?s|next step|plan|будем|сделаем|решили|договорились|следующим шагом|сначала)\b)",
    re.I,
)


_ASSISTANT_CONTINUATION_PREFACE_RE = re.compile(
    r"^(?:let'?s\s+(?:continue|keep going|go back|return)\b|continue\b|we can continue\b|back to\b|return to\b)",
    re.I,
)


@dataclass(frozen=True)
class ActiveTaskState:
    task_id: str
    topic: str
    status: str  # active | blocked | waiting_user | done | abandoned
    source_episode_id: str = ""
    summary_short: str = ""
    current_goal: str = ""
    next_steps: list[str] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)
    decisions: list[str] = field(default_factory=list)
    confidence: float = 0.0
    updated_at: float = 0.0
    planner_source: str = ""
    planner_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "topic": self.topic,
            "status": self.status,
            "source_episode_id": self.source_episode_id,
            "summary_short": self.summary_short,
            "current_goal": self.current_goal,
            "next_steps": list(self.next_steps or []),
            "open_questions": list(self.open_questions or []),
            "decisions": list(self.decisions or []),
            "confidence": float(self.confidence or 0.0),
            "updated_at": float(self.updated_at or 0.0),
            "planner_source": str(self.planner_source or ""),
            "planner_reason": str(self.planner_reason or ""),
        }


class EpisodePlanner:
    episode_score_floor: float = 0.46

    def resolve_active_task(
        self,
        *,
        user_text: str,
        memory_context: dict[str, Any] | None,
        state: dict[str, Any] | None,
        meta: dict[str, Any] | None,
    ) -> ActiveTaskState | None:
        state_map = dict(state or {})
        memory_map = dict(memory_context or {})
        meta_map = dict(meta or {})
        active_task = self.resolve_context_active_task(
            state=state_map,
            memory_context=memory_map,
        )
        now_ts = self._resolve_now_ts(meta_map)
        close_reason = self.close_reason(user_text=user_text, active_task=active_task)
        continuation_reason = self.continuation_reason(
            user_text=user_text,
            active_task=active_task,
            meta=meta_map,
        )

        previous_task = self._previous_task_from_continuity(
            user_text=user_text,
            memory_context=memory_map,
            state=state_map,
            active_task=active_task,
            now_ts=now_ts,
        )
        if previous_task is not None:
            return self._coerce_active_task(previous_task)

        closed = self.maybe_close_active_task(user_text=user_text, active_task=active_task)
        if closed:
            closed["updated_at"] = now_ts
            closed["planner_source"] = "continuity"
            closed["planner_reason"] = str(close_reason or "explicit_close_signal")
            return self._coerce_active_task(closed)

        if continuation_reason in {"short_followup", "followup_like_meta"}:
            enriched = self._merge_active_task_with_context(
                active_task=active_task,
                memory_context=memory_map,
                user_text=user_text,
                now_ts=now_ts,
            )
            enriched["planner_source"] = "continuation"
            enriched["planner_reason"] = continuation_reason
            return self._coerce_active_task(enriched)

        episode_task = self._active_task_from_memory_context(
            user_text=user_text,
            memory_context=memory_map,
            now_ts=now_ts,
        )
        if episode_task:
            episode_task["planner_source"] = "episode_hit"
            episode_task["planner_reason"] = (
                "episode_open_questions"
                if list(episode_task.get("open_questions") or [])
                else "episode_decisions"
            )
            return self._coerce_active_task(episode_task)
        runtime_hint_task = self._active_task_from_runtime_hints(
            user_text=user_text,
            memory_context=memory_map,
            state=state_map,
            now_ts=now_ts,
        )
        if runtime_hint_task:
            runtime_hint_task["planner_source"] = "runtime_hints"
            runtime_hint_task["planner_reason"] = (
                "runtime_open_questions_fallback"
                if list(runtime_hint_task.get("open_questions") or [])
                else "runtime_decisions_fallback"
            )
            return self._coerce_active_task(runtime_hint_task)
        return None

    def should_continue_active_task(
        self,
        *,
        user_text: str,
        active_task: dict[str, Any] | None,
        meta: dict[str, Any] | None,
    ) -> bool:
        return self.continuation_reason(
            user_text=user_text,
            active_task=active_task,
            meta=meta,
        ) in {"short_followup", "followup_like_meta"}

    def resolve_context_active_task(
        self,
        *,
        state: dict[str, Any] | None,
        memory_context: dict[str, Any] | None,
    ) -> dict[str, Any]:
        state_map = dict(state or {})
        explicit = self._normalize_task_row(state_map.get("active_task"))
        if explicit:
            return explicit
        continuity = self._task_continuity_snapshot(memory_context=memory_context, state=state_map)
        return self._normalize_task_row(dict(continuity.get("active_task") or {}))

    def update_active_task_after_assistant_reply(
        self,
        *,
        assistant_text: str,
        active_task: dict[str, Any] | None,
        memory_context: dict[str, Any] | None = None,
        state: dict[str, Any] | None = None,
        meta: dict[str, Any] | None = None,
    ) -> ActiveTaskState | None:
        text = str(assistant_text or "").strip()
        task = self.resolve_context_active_task(state=state or {"active_task": dict(active_task or {})}, memory_context=memory_context)
        if not task or not text:
            return self._coerce_active_task(task) if task else None

        now_ts = self._resolve_now_ts(dict(meta or {}))
        updated = dict(task)
        action_items = self._assistant_action_items(text)
        question_items = self._assistant_question_items(text)
        progress_items = self._assistant_progress_items(text, action_items=action_items)
        existing_decisions = [str(x).strip() for x in list(updated.get("decisions") or []) if str(x).strip()]
        existing_open_questions = [str(x).strip() for x in list(updated.get("open_questions") or []) if str(x).strip()]
        existing_next_steps = [str(x).strip() for x in list(updated.get("next_steps") or []) if str(x).strip()]

        decisions = self._merge_unique_items(
            existing_decisions,
            progress_items or action_items,
            limit=8,
        )
        next_steps = self._merge_unique_items(
            action_items,
            [],
            limit=4,
        ) or existing_next_steps
        if question_items:
            open_questions = self._merge_unique_items(existing_open_questions, question_items, limit=6)
        elif action_items or progress_items:
            open_questions = []
        else:
            open_questions = existing_open_questions

        previous_status = str(updated.get("status") or "active").strip().lower() or "active"
        status = previous_status
        if open_questions:
            status = "waiting_user"
        elif action_items or progress_items:
            status = "active"
        elif previous_status in {"active", "blocked", "waiting_user"}:
            status = previous_status

        updated["decisions"] = decisions
        updated["next_steps"] = next_steps
        updated["open_questions"] = open_questions
        updated["status"] = status
        updated["current_goal"] = str(
            self._pick_first_non_empty(
                next_steps[0] if next_steps else "",
                decisions[-1] if decisions else "",
                updated.get("current_goal"),
                updated.get("summary_short"),
                updated.get("topic"),
            )
            or ""
        ).strip()
        if not str(updated.get("summary_short") or "").strip():
            updated["summary_short"] = str(
                self._pick_first_non_empty(
                    next_steps[0] if next_steps else "",
                    decisions[-1] if decisions else "",
                    updated.get("current_goal"),
                    updated.get("topic"),
                )
                or ""
            ).strip()
        updated["updated_at"] = now_ts
        updated["planner_source"] = "assistant_reply"
        updated["planner_reason"] = (
            "assistant_waiting_user_update"
            if open_questions
            else ("assistant_progress_update" if (action_items or progress_items) else "assistant_keep_state")
        )
        return self._coerce_active_task(updated)

    def continuation_reason(
        self,
        *,
        user_text: str,
        active_task: dict[str, Any] | None,
        meta: dict[str, Any] | None,
    ) -> str:
        text = normalize_text(str(user_text or "")).strip().lower()
        task = dict(active_task or {})
        if not text or not task:
            return "no_active_task"
        status = str(task.get("status") or "active").strip().lower()
        if status in {"done", "abandoned"}:
            return "inactive_task"
        meta_map = dict(meta or {})
        if bool(meta_map.get("new_topic")) or bool(meta_map.get("hard_topic_switch")):
            return "meta_topic_switch"
        if _SWITCH_TOPIC_RE.search(text):
            return "switch_topic_phrase"
        if len(text) <= 32 and _CONTINUE_RE.search(text):
            return "short_followup"
        if bool(meta_map.get("followup_like")):
            return "followup_like_meta"
        return "no_match"

    def maybe_close_active_task(
        self,
        *,
        user_text: str,
        active_task: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        task = dict(active_task or {})
        if not task:
            return None
        status = str(task.get("status") or "active").strip().lower()
        if status in {"done", "abandoned"}:
            return None
        close_reason = self.close_reason(user_text=user_text, active_task=active_task)
        if close_reason == "explicit_close_phrase":
            updated = dict(task)
            updated["status"] = "done"
            return updated
        if close_reason == "explicit_abandon_phrase":
            updated = dict(task)
            updated["status"] = "abandoned"
            return updated
        if close_reason == "blocked_phrase":
            updated = dict(task)
            updated["status"] = "blocked"
            return updated
        if close_reason == "waiting_user_phrase":
            updated = dict(task)
            updated["status"] = "waiting_user"
            return updated
        return None

    def close_reason(
        self,
        *,
        user_text: str,
        active_task: dict[str, Any] | None,
    ) -> str:
        task = dict(active_task or {})
        if not task:
            return "no_active_task"
        status = str(task.get("status") or "active").strip().lower()
        if status in {"done", "abandoned"}:
            return "inactive_task"
        text = normalize_text(str(user_text or "")).lower()
        if not text:
            return "empty_text"
        if _CLOSE_RE.search(text):
            return "explicit_close_phrase"
        if "abandon" in text or "забей" in text or "неважно" in text:
            return "explicit_abandon_phrase"
        if _BLOCKED_RE.search(text):
            return "blocked_phrase"
        if _WAITING_USER_RE.search(text):
            return "waiting_user_phrase"
        return "no_close_signal"

    def _previous_task_from_continuity(
        self,
        *,
        user_text: str,
        memory_context: dict[str, Any] | None,
        state: dict[str, Any] | None,
        active_task: dict[str, Any] | None,
        now_ts: float,
    ) -> dict[str, Any] | None:
        if not _RETURN_PREVIOUS_RE.search(normalize_text(str(user_text or "")).lower()):
            return None
        current_task = self._normalize_task_row(active_task)
        continuity = self._task_continuity_snapshot(memory_context=memory_context, state=state)
        history = [self._normalize_task_row(x) for x in list(continuity.get("task_history") or [])]
        history = [row for row in history if row]
        if not history:
            return None
        current_task_id = str(current_task.get("task_id") or "").strip()
        candidates = [
            row for row in history
            if str(row.get("task_id") or "").strip()
            and str(row.get("task_id") or "").strip() != current_task_id
            and str(row.get("status") or "").strip().lower() not in {"done", "abandoned"}
        ]
        if not candidates:
            return None
        selected = dict(candidates[0])
        selected["updated_at"] = now_ts
        selected["planner_source"] = "task_history"
        selected["planner_reason"] = "return_to_previous_task"
        return selected

    def _task_continuity_snapshot(
        self,
        *,
        memory_context: dict[str, Any] | None,
        state: dict[str, Any] | None,
    ) -> dict[str, Any]:
        memory = dict(memory_context or {})
        state_map = dict(state or {})
        continuity = dict(memory.get("task_continuity") or state_map.get("task_continuity") or {})
        if continuity:
            active_task = self._normalize_task_row(dict(continuity.get("active_task") or {}))
            history = [
                self._normalize_task_row(x)
                for x in list(continuity.get("task_history") or continuity.get("previous_tasks") or [])
            ]
            return {
                "active_task": active_task,
                "task_history": [row for row in history if row],
                "updated_at": self._to_float(continuity.get("updated_at"), 0.0),
                "source": str(continuity.get("source") or "").strip(),
            }
        return {}

    @staticmethod
    def _normalize_task_row(value: Any) -> dict[str, Any]:
        row = dict(value or {})
        if not row:
            return {}
        out = {
            "task_id": str(row.get("task_id") or "").strip(),
            "topic": str(row.get("topic") or "").strip(),
            "status": str(row.get("status") or "active").strip().lower() or "active",
            "source_episode_id": str(row.get("source_episode_id") or "").strip(),
            "summary_short": str(row.get("summary_short") or "").strip(),
            "current_goal": str(row.get("current_goal") or "").strip(),
            "next_steps": [str(x).strip() for x in list(row.get("next_steps") or []) if str(x).strip()],
            "open_questions": [str(x).strip() for x in list(row.get("open_questions") or []) if str(x).strip()],
            "decisions": [str(x).strip() for x in list(row.get("decisions") or []) if str(x).strip()],
            "confidence": float(EpisodePlanner._to_float(row.get("confidence"), 0.0)),
            "updated_at": float(EpisodePlanner._to_float(row.get("updated_at"), 0.0)),
            "planner_source": str(row.get("planner_source") or "").strip(),
            "planner_reason": str(row.get("planner_reason") or "").strip(),
        }
        return {
            key: value
            for key, value in out.items()
            if value not in ("", []) or key in {"status", "confidence", "updated_at"}
        }

    def _assistant_action_items(self, text: str) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()
        for raw_line in re.split(r"[\r\n]+", str(text or "")):
            match = _TASK_BULLET_RE.match(str(raw_line or "").strip())
            if not match:
                continue
            item = str(match.group(1) or "").strip(" .")
            if not item or item.endswith("?"):
                continue
            key = normalize_text(item).lower()
            if not key or key in seen:
                continue
            seen.add(key)
            out.append(item[:220])
        return out[:4]

    def _assistant_question_items(self, text: str) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()
        for chunk in re.split(r"[\r\n]+|(?<=[?!])\s+", str(text or "")):
            item = str(chunk or "").strip()
            if not item or "?" not in item:
                continue
            key = normalize_text(item).lower()
            if not key or key in seen:
                continue
            seen.add(key)
            out.append(item[:220])
        return out[:4]

    def _assistant_progress_items(self, text: str, *, action_items: list[str]) -> list[str]:
        out = list(action_items or [])
        seen = {normalize_text(str(item or "")).lower() for item in out if str(item or "").strip()}
        for chunk in re.split(r"[\r\n]+|(?<=[.!?])\s+", str(text or "")):
            item = str(chunk or "").strip()
            if (
                not item
                or item.endswith("?")
                or _ASSISTANT_CONTINUATION_PREFACE_RE.search(item)
                or not _TASK_DECISION_RE.search(item)
            ):
                continue
            key = normalize_text(item).lower()
            if not key or key in seen:
                continue
            seen.add(key)
            out.append(item[:220])
        return out[:4]

    @staticmethod
    def _merge_unique_items(primary: list[str], secondary: list[str], *, limit: int) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()
        for source in (primary, secondary):
            for item in list(source or []):
                text = str(item or "").strip()
                if not text:
                    continue
                key = normalize_text(text).lower()
                if not key or key in seen:
                    continue
                seen.add(key)
                out.append(text)
        return out[: max(1, int(limit))]

    @staticmethod
    def _pick_first_non_empty(*values: Any) -> str:
        for value in values:
            text = str(value or "").strip()
            if text:
                return text
        return ""

    def _active_task_from_memory_context(
        self,
        *,
        user_text: str,
        memory_context: dict[str, Any] | None,
        now_ts: float,
    ) -> dict[str, Any] | None:
        memory = dict(memory_context or {})
        normalized_text = normalize_text(str(user_text or "")).lower()
        if not (
            _RETURN_RE.search(normalized_text)
            or _CONTINUE_RE.search(normalized_text)
            or self._is_plan_like_query(user_text)
        ):
            return None
        top_episode_hit = self._top_episode_hit(memory_context=memory, user_text=user_text)
        if not top_episode_hit:
            return None
        task = self._from_episode_hit(top_episode_hit)
        if task is None:
            return None
        payload = task.to_dict()
        payload["updated_at"] = now_ts
        return payload

    def _active_task_from_runtime_hints(
        self,
        *,
        user_text: str,
        memory_context: dict[str, Any] | None,
        state: dict[str, Any] | None,
        now_ts: float,
    ) -> dict[str, Any] | None:
        normalized_text = normalize_text(str(user_text or "")).lower()
        if not (
            _RETURN_RE.search(normalized_text)
            or _CONTINUE_RE.search(normalized_text)
            or self._is_plan_like_query(user_text)
        ):
            return None
        open_questions, current_decisions = self._runtime_task_hints(
            memory_context=memory_context,
            state=state,
        )
        if not open_questions and not current_decisions:
            return None

        topic_source = (
            current_decisions[-1]
            if current_decisions
            else open_questions[0]
        )
        topic = self._runtime_task_topic(topic_source)
        summary_short = ""
        if current_decisions:
            summary_short = current_decisions[-1]
        elif open_questions:
            summary_short = open_questions[0]
        current_goal = current_decisions[-1] if current_decisions else summary_short
        next_steps = [str(x).strip() for x in list(current_decisions[-3:] or []) if str(x).strip()]
        status = "waiting_user" if open_questions else "active"
        task_id = f"task:runtime:{self._slug(topic or current_goal or summary_short or 'current-task')}"
        return {
            "task_id": task_id,
            "topic": topic,
            "status": status,
            "source_episode_id": "",
            "summary_short": summary_short,
            "current_goal": str(current_goal or summary_short or topic).strip(),
            "next_steps": next_steps,
            "open_questions": [str(x).strip() for x in list(open_questions or []) if str(x).strip()],
            "decisions": [str(x).strip() for x in list(current_decisions or []) if str(x).strip()],
            "confidence": 0.41,
            "updated_at": now_ts,
        }

    def _merge_active_task_with_context(
        self,
        *,
        active_task: dict[str, Any],
        memory_context: dict[str, Any] | None,
        user_text: str,
        now_ts: float,
    ) -> dict[str, Any]:
        task = dict(active_task or {})
        episode_hit = self._episode_hit_for_active_task(
            memory_context=memory_context,
            active_task=task,
            user_text=user_text,
        )
        episode_payload = self._episode_like_payload(episode_hit)
        if episode_payload and self._episode_looks_closed(episode_payload):
            closed = dict(task)
            closed["status"] = "done"
            closed["updated_at"] = now_ts
            if str(episode_payload.get("source_episode_id") or "").strip():
                closed["source_episode_id"] = str(episode_payload.get("source_episode_id") or "").strip()
            return closed

        hit_task = self._from_episode_hit(episode_hit)
        if hit_task is None:
            task["updated_at"] = now_ts
            return task

        merged = dict(task)
        hit_payload = hit_task.to_dict()
        for key in (
            "task_id",
            "topic",
            "status",
            "source_episode_id",
            "summary_short",
            "current_goal",
            "next_steps",
            "open_questions",
            "decisions",
        ):
            value = hit_payload.get(key)
            if isinstance(value, list):
                if value:
                    merged[key] = list(value)
            elif str(value or "").strip():
                merged[key] = value
        merged["confidence"] = max(
            self._to_float(merged.get("confidence"), 0.0),
            self._to_float(hit_payload.get("confidence"), 0.0),
        )
        merged["updated_at"] = now_ts
        return merged

    def _episode_hit_for_active_task(
        self,
        *,
        memory_context: dict[str, Any] | None,
        active_task: dict[str, Any] | None,
        user_text: str,
    ) -> dict[str, Any] | None:
        memory = dict(memory_context or {})
        task = dict(active_task or {})
        source_episode_id = str(task.get("source_episode_id") or "").strip()
        if source_episode_id:
            matched = self._episode_hit_by_id(memory_context=memory, episode_id=source_episode_id)
            if matched is not None:
                return matched
        top_hit = self._top_episode_hit(memory_context=memory, user_text=user_text)
        if top_hit is None:
            return None
        task_topic = str(task.get("topic") or "").strip()
        if not task_topic:
            return top_hit
        hit_payload = self._episode_like_payload(top_hit)
        hit_topic = str(hit_payload.get("topic") or "").strip()
        hit_summary = str(hit_payload.get("summary_short") or "").strip()
        if self._topic_overlap(task_topic, hit_topic) > 0:
            return top_hit
        if self._topic_overlap(task_topic, hit_summary) > 0:
            return top_hit
        return None

    def _top_episode_hit(
        self,
        *,
        memory_context: dict[str, Any] | None,
        user_text: str = "",
    ) -> dict[str, Any] | None:
        structured_hits = self._structured_episode_hits(memory_context)
        if structured_hits:
            structured_hits.sort(
                key=lambda row: float(self._to_float(dict(row or {}).get("score"), 0.0)),
                reverse=True,
            )
            top_hit = dict(structured_hits[0])
            if self._episode_is_relevant(row=top_hit, user_text=user_text):
                return top_hit

        episode_rows = self._episode_rows(memory_context)
        if not episode_rows:
            return None
        episode_rows.sort(key=lambda row: float(self._to_float(row.get("score"), 0.0)), reverse=True)
        top = dict(episode_rows[0])
        if not self._episode_is_relevant(row=top, user_text=user_text):
            return None
        return top

    def _episode_hit_by_id(
        self,
        *,
        memory_context: dict[str, Any] | None,
        episode_id: str,
    ) -> dict[str, Any] | None:
        target_id = str(episode_id or "").strip()
        if not target_id:
            return None
        for row in self._structured_episode_hits(memory_context):
            payload = self._episode_like_payload(row)
            row_id = str(payload.get("source_episode_id") or "").strip()
            if row_id == target_id:
                return row
        for row in self._episode_rows(memory_context):
            payload = self._episode_like_payload(row)
            row_id = str(payload.get("source_episode_id") or row.get("id") or "").strip()
            if row_id == target_id:
                return row
        return None

    def _structured_episode_hits(self, memory_context: dict[str, Any] | None) -> list[dict[str, Any]]:
        memory = dict(memory_context or {})
        hits = [dict(x) for x in list(memory.get("dialog_episode_hits") or []) if isinstance(x, dict)]
        return [row for row in hits if isinstance(dict(row.get("episode") or {}), dict) and dict(row.get("episode") or {})]

    def _episode_rows(self, memory_context: dict[str, Any] | None) -> list[dict[str, Any]]:
        memory = dict(memory_context or {})
        selected = [dict(x) for x in list(memory.get("selected") or []) if isinstance(x, dict)]
        return [
            row
            for row in selected
            if str(row.get("memory_type") or "").strip().lower() == "episode"
            and isinstance(
                dict(row.get("metadata") or {}).get("dialog_episode")
                or dict(row.get("metadata") or {}).get("episode")
                or None,
                dict,
            )
        ]

    def _from_episode_hit(self, hit: Any) -> ActiveTaskState | None:
        payload = self._episode_like_payload(hit)
        if not payload or not self._episode_has_active_thread_signals(payload):
            return None
        source_episode_id = str(payload.get("source_episode_id") or "").strip()
        topic = str(payload.get("topic") or "").strip() or "dialog_task"
        summary_short = str(payload.get("summary_short") or "").strip()
        summary_reasoning = str(payload.get("summary_reasoning") or "").strip()
        decisions = [str(x).strip() for x in list(payload.get("decisions") or []) if str(x).strip()]
        open_questions = [str(x).strip() for x in list(payload.get("open_questions") or []) if str(x).strip()]
        next_steps = [str(x).strip() for x in list(decisions[-3:] or []) if str(x).strip()]
        status = "waiting_user" if open_questions else "active"
        task_id = f"task:{source_episode_id}" if source_episode_id else f"task:{self._slug(topic or summary_short or 'dialog-task')}"
        return ActiveTaskState(
            task_id=str(task_id),
            topic=topic,
            status=status,
            source_episode_id=source_episode_id,
            summary_short=summary_short,
            current_goal=str(summary_reasoning or summary_short or topic).strip(),
            next_steps=next_steps,
            open_questions=open_questions,
            decisions=decisions,
            confidence=float(self._to_float(payload.get("score"), 0.0)),
            updated_at=float(self._to_float(payload.get("updated_at"), 0.0)),
        )

    def _episode_like_payload(self, value: Any) -> dict[str, Any]:
        if isinstance(value, dict) and isinstance(dict(value.get("episode") or {}), dict) and dict(value.get("episode") or {}):
            row = dict(value or {})
            episode = dict(row.get("episode") or {})
            return {
                "source_episode_id": str(episode.get("id") or row.get("record_id") or "").strip(),
                "topic": str(episode.get("topic") or "").strip(),
                "summary_short": str(row.get("summary_short") or episode.get("summary_short") or "").strip(),
                "summary_reasoning": str(row.get("summary_reasoning") or episode.get("summary_reasoning") or "").strip(),
                "decisions": [str(x).strip() for x in list(row.get("decisions") or episode.get("decisions") or []) if str(x).strip()],
                "open_questions": [str(x).strip() for x in list(episode.get("open_questions") or []) if str(x).strip()],
                "status": str(episode.get("status") or "").strip(),
                "score": self._to_float(row.get("score"), 0.0),
                "updated_at": self._to_float(episode.get("updated_at"), 0.0),
            }
        return self._episode_payload_from_row(value if isinstance(value, dict) else {})

    def _episode_payload_from_row(self, row: dict[str, Any] | None) -> dict[str, Any]:
        source = dict(row or {})
        meta = dict(source.get("metadata") or {})
        episode = dict(meta.get("dialog_episode") or meta.get("episode") or {})
        if not episode:
            return {}
        return {
            "source_episode_id": str(episode.get("id") or source.get("id") or "").strip(),
            "topic": str(episode.get("topic") or meta.get("topic") or "").strip(),
            "summary_short": str(episode.get("summary_short") or meta.get("summary_short") or source.get("text") or "").strip(),
            "summary_reasoning": str(episode.get("summary_reasoning") or meta.get("summary_reasoning") or "").strip(),
            "decisions": [str(x).strip() for x in list(episode.get("decisions") or meta.get("decisions") or []) if str(x).strip()],
            "open_questions": [str(x).strip() for x in list(episode.get("open_questions") or meta.get("open_questions") or []) if str(x).strip()],
            "status": str(episode.get("status") or meta.get("status") or "").strip(),
            "score": self._to_float(source.get("score"), 0.0),
            "updated_at": self._to_float(episode.get("updated_at") or source.get("updated_at"), 0.0),
        }

    def _episode_is_relevant(self, *, row: dict[str, Any], user_text: str) -> bool:
        score = self._to_float(row.get("score"), 0.0)
        if score < self.episode_score_floor:
            return False
        payload = self._episode_like_payload(row)
        summary_short = str(payload.get("summary_short") or "").strip()
        decisions = [str(x).strip() for x in list(payload.get("decisions") or []) if str(x).strip()]
        open_questions = [str(x).strip() for x in list(payload.get("open_questions") or []) if str(x).strip()]
        if not any((summary_short, decisions, open_questions)):
            return False
        if open_questions or decisions:
            return True
        return self._is_plan_like_query(user_text)

    def _episode_has_active_thread_signals(self, episode_payload: dict[str, Any] | None) -> bool:
        payload = dict(episode_payload or {})
        if self._episode_looks_closed(payload):
            return False
        open_questions = [str(x).strip() for x in list(payload.get("open_questions") or []) if str(x).strip()]
        decisions = [str(x).strip() for x in list(payload.get("decisions") or []) if str(x).strip()]
        if open_questions:
            return True
        if decisions:
            return True
        return False

    def _episode_looks_closed(self, episode_payload: dict[str, Any] | None) -> bool:
        payload = dict(episode_payload or {})
        status = str(payload.get("status") or "").strip().lower()
        if status in {"done", "closed", "completed", "finished", "resolved", "abandoned"}:
            return True
        summary = normalize_text(str(payload.get("summary_short") or "")).lower()
        if summary and _CLOSE_RE.search(summary):
            return True
        return False

    def _runtime_task_hints(
        self,
        *,
        memory_context: dict[str, Any] | None,
        state: dict[str, Any] | None,
    ) -> tuple[list[str], list[str]]:
        memory = dict(memory_context or {})
        state_map = dict(state or {})
        open_questions = [
            str(x).strip()
            for x in list(memory.get("open_questions") or state_map.get("open_questions") or [])
            if str(x).strip()
        ]
        current_decisions = [
            str(x).strip()
            for x in list(memory.get("current_decisions") or state_map.get("current_decisions") or [])
            if str(x).strip()
        ]
        return open_questions, current_decisions

    @staticmethod
    def _runtime_task_topic(value: str) -> str:
        text = normalize_text(str(value or "")).strip()
        if not text:
            return "current_task"
        tokens = [token for token in re.split(r"\s+", text) if token]
        return " ".join(tokens[:6]) or "current_task"

    @staticmethod
    def _topic_overlap(user_text: str, topic: str) -> int:
        user_tokens = {x for x in re.split(r"\s+", normalize_text(str(user_text or "")).lower()) if len(x) >= 4}
        topic_tokens = {x for x in re.split(r"\s+", normalize_text(str(topic or "")).lower()) if len(x) >= 4}
        return len(user_tokens.intersection(topic_tokens))

    @staticmethod
    def _is_plan_like_query(text: str) -> bool:
        low = normalize_text(str(text or "")).lower()
        return any(
            marker in low
            for marker in (
                "plan",
                "memory",
                "web",
                "task",
                "задач",
                "план",
                "памят",
                "веб",
                "шаг",
            )
        )

    @staticmethod
    def _resolve_now_ts(meta: dict[str, Any]) -> float:
        for key in ("now_ts", "ts", "updated_at"):
            try:
                value = float(meta.get(key))
            except Exception:
                value = 0.0
            if value > 0.0:
                return value
        return float(time.time())

    @staticmethod
    def _coerce_active_task(value: dict[str, Any]) -> ActiveTaskState:
        row = dict(value or {})
        return ActiveTaskState(
            task_id=str(row.get("task_id") or "").strip(),
            topic=str(row.get("topic") or "").strip(),
            status=str(row.get("status") or "active").strip().lower() or "active",
            source_episode_id=str(row.get("source_episode_id") or "").strip(),
            summary_short=str(row.get("summary_short") or "").strip(),
            current_goal=str(row.get("current_goal") or "").strip(),
            next_steps=[str(x).strip() for x in list(row.get("next_steps") or []) if str(x).strip()],
            open_questions=[str(x).strip() for x in list(row.get("open_questions") or []) if str(x).strip()],
            decisions=[str(x).strip() for x in list(row.get("decisions") or []) if str(x).strip()],
            confidence=float(EpisodePlanner._to_float(row.get("confidence"), 0.0)),
            updated_at=float(EpisodePlanner._to_float(row.get("updated_at"), 0.0)),
            planner_source=str(row.get("planner_source") or "").strip(),
            planner_reason=str(row.get("planner_reason") or "").strip(),
        )

    @staticmethod
    def _slug(value: str) -> str:
        text = normalize_text(str(value or "")).lower()
        tokens = [x for x in re.split(r"\s+", text) if x]
        return "-".join(tokens[:6]) or "active-thread"

    @staticmethod
    def _to_float(value: Any, default: float) -> float:
        try:
            return float(value)
        except Exception:
            return float(default)
