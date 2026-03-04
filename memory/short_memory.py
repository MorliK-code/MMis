from __future__ import annotations

import json
from pathlib import Path
from threading import RLock
from typing import Any

from config.settings import load_config
from utils.datetime_local import now_local_ts, to_local_iso


class ShortMemory:
    """Short-lived rolling window of recent messages/events."""

    def __init__(
        self,
        limit: int = 60,
        *,
        summary_trigger: int = 50,
        persona_snapshot_every: int = 6,
        path: str | Path | None = None,
        autosave: bool = True,
    ):
        cfg = load_config()
        default_path = cfg.memory_dir / "short_memory.json"
        self.path = Path(path).expanduser() if path is not None else default_path
        self.path.parent.mkdir(parents=True, exist_ok=True)

        self.limit = max(10, int(limit))
        self.summary_trigger = max(10, int(summary_trigger))
        self.persona_snapshot_every = max(1, int(persona_snapshot_every))
        self.autosave = bool(autosave)
        self._lock = RLock()

        self._items: list[dict[str, Any]] = []
        self._rolling_summary: str = ""
        self._rolling_summary_meta: dict[str, Any] = {}
        self._last_persona_snapshot_turn: int = 0
        self.load()

    def append(self, item: dict[str, Any]) -> None:
        row = self._normalize_item(item)
        if not row.get("text"):
            return
        with self._lock:
            self._items.append(row)
            snapshot_row = self._build_persona_snapshot_row(row)
            if snapshot_row is not None:
                self._items.append(snapshot_row)
            self._trim_and_update_summary()
        self._autosave()

    def tail(self, n: int = 12) -> list[dict[str, Any]]:
        count = max(1, int(n))
        with self._lock:
            return [dict(x) for x in self._items[-count:]]

    def clear(self) -> None:
        with self._lock:
            self._items = []
            self._rolling_summary = ""
            self._rolling_summary_meta = {}
            self._last_persona_snapshot_turn = 0
        self._autosave()

    def rolling_summary(self) -> str:
        with self._lock:
            return str(self._rolling_summary or "")

    def rolling_summary_meta(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._rolling_summary_meta or {})

    def size(self) -> int:
        with self._lock:
            return len(self._items)

    def load(self) -> None:
        if not self.path.exists():
            return
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8-sig"))
        except Exception:
            return
        if not isinstance(payload, dict):
            return
        items = payload.get("items")
        summary = payload.get("rolling_summary")
        summary_meta = payload.get("rolling_summary_meta")
        with self._lock:
            self._items = [self._normalize_item(x) for x in list(items or []) if isinstance(x, dict)]
            self._items = [x for x in self._items if x.get("text")]
            if len(self._items) > self.limit:
                self._items = self._items[-self.limit :]
            self._rolling_summary = str(summary or "")
            self._rolling_summary_meta = dict(summary_meta or {})
            self._restore_persona_snapshot_marker()

    def save(self) -> None:
        with self._lock:
            payload = {
                "items": [dict(x) for x in self._items],
                "rolling_summary": str(self._rolling_summary or ""),
                "rolling_summary_meta": dict(self._rolling_summary_meta or {}),
            }
        self.path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _autosave(self) -> None:
        if self.autosave:
            self.save()

    def _trim_and_update_summary(self) -> None:
        overflow: list[dict[str, Any]] = []
        if len(self._items) > self.limit:
            overflow = self._items[:-self.limit]
            self._items = self._items[-self.limit :]
        elif len(self._items) > self.summary_trigger:
            overflow = self._items[:-self.summary_trigger]
        if overflow:
            self._update_summary(overflow)

    def _update_summary(self, items: list[dict[str, Any]]) -> None:
        if not items:
            return
        payload = self._build_summary_payload(items)
        block = self._render_summary_payload(payload)
        if not block:
            return
        self._rolling_summary_meta = payload
        if self._rolling_summary:
            merged = f"{self._rolling_summary} || {block}"
        else:
            merged = block
        self._rolling_summary = merged[-1600:]

    def _build_persona_snapshot_row(self, row: dict[str, Any]) -> dict[str, Any] | None:
        if str(row.get("role") or "").strip().lower() != "assistant":
            return None
        turn_id = _to_int(row.get("meta", {}).get("turn_id"), 0)
        if turn_id <= 0:
            return None
        if (turn_id % self.persona_snapshot_every) != 0:
            return None
        if turn_id == self._last_persona_snapshot_turn:
            return None
        persona = _normalize_persona_snapshot(row.get("persona_snapshot"))
        if not persona:
            persona = _normalize_persona_snapshot(row.get("meta", {}).get("persona_snapshot"))
        if not persona:
            return None
        self._last_persona_snapshot_turn = turn_id
        mood = str(persona.get("mood") or "neutral").strip().lower() or "neutral"
        traits = dict(persona.get("traits") or {})
        trait_parts: list[str] = []
        for key in ("warmth", "sarcasm", "verbosity", "strictness", "teasing", "empathy"):
            if key not in traits:
                continue
            try:
                trait_parts.append(f"{key}={float(traits.get(key)):.2f}")
            except Exception:
                continue
        text = f"persona snapshot: mood={mood}"
        if trait_parts:
            text += ", " + ", ".join(trait_parts[:6])
        return self._normalize_item(
            {
                "id": f"{str(row.get('id') or '')}:persona_snapshot",
                "role": "system",
                "type": "persona_snapshot",
                "text": text,
                "ts": row.get("ts"),
                "tags": ["persona_snapshot", f"mood_{mood}"],
                "topic": "",
                "topics": [],
                "active_mode": row.get("active_mode") or row.get("meta", {}).get("active_mode"),
                "persona_snapshot": persona,
                "meta": {
                    "source": "persona_snapshot",
                    "turn_id": turn_id,
                    "active_mode": row.get("active_mode") or row.get("meta", {}).get("active_mode"),
                    "persona_snapshot": persona,
                    "personality_id": row.get("meta", {}).get("personality_id"),
                },
            }
        )

    def _restore_persona_snapshot_marker(self) -> None:
        last_turn = 0
        for row in reversed(self._items):
            if str(row.get("type") or "").strip().lower() != "persona_snapshot":
                continue
            turn_id = _to_int(row.get("meta", {}).get("turn_id"), 0)
            if turn_id > 0:
                last_turn = turn_id
                break
        self._last_persona_snapshot_turn = last_turn

    @staticmethod
    def _build_summary_payload(items: list[dict[str, Any]]) -> dict[str, Any]:
        recent = [dict(x) for x in list(items or []) if isinstance(x, dict)][-28:]
        tasks: list[str] = []
        errors: list[str] = []
        topic_counts: dict[str, int] = {}
        fallback_lines: list[str] = []
        last_mode = ""
        technical_focus = False
        persona_snapshot: dict[str, Any] = {}

        for row in recent:
            meta_map = dict(row.get("meta") or {}) if isinstance(row.get("meta"), dict) else {}
            intent = str(row.get("intent") or "").strip().lower()
            role = str(row.get("role") or "").strip().lower()
            text = _short_text(row.get("text"), 110)
            tags = {str(x).strip().lower() for x in list(row.get("tags") or []) if str(x).strip()}
            mode = str(row.get("active_mode") or meta_map.get("active_mode") or "").strip().lower()
            if mode:
                last_mode = mode
            if mode in {"engineer", "debugger", "planner"}:
                technical_focus = True
            for topic in list(row.get("topics") or []):
                key = str(topic or "").strip().lower()
                if not key:
                    continue
                topic_counts[key] = int(topic_counts.get(key, 0)) + 1
            if str(row.get("type") or "").strip().lower() == "persona_snapshot":
                persona_snapshot = _normalize_persona_snapshot(row.get("persona_snapshot") or meta_map.get("persona_snapshot"))
            if _is_error_row(intent=intent, tags=tags, text=text):
                if text:
                    errors.append(text)
            if role == "user" and _is_task_row(intent=intent, tags=tags):
                if text:
                    tasks.append(text)
            if text and str(row.get("type") or "").strip().lower() == "message":
                if technical_focus and intent == "chat" and not _is_error_row(intent=intent, tags=tags, text=text):
                    continue
                fallback_lines.append(f"{role}: {text}")

        unique_tasks = _dedupe_str_list(tasks)[-3:]
        unique_errors = _dedupe_str_list(errors)[-2:]
        ranked_topics = sorted(topic_counts.items(), key=lambda x: (int(x[1]), x[0]), reverse=True)
        topics = [str(key) for key, _ in ranked_topics[:4]]
        fallback = _dedupe_str_list(fallback_lines)[-3:]

        return {
            "tasks": unique_tasks,
            "errors": unique_errors,
            "topics": topics,
            "active_mode": last_mode,
            "technical_focus": bool(technical_focus),
            "persona_snapshot": persona_snapshot,
            "recent": fallback,
        }

    @staticmethod
    def _render_summary_payload(payload: dict[str, Any]) -> str:
        data = dict(payload or {})
        lines: list[str] = []
        tasks = [str(x) for x in list(data.get("tasks") or []) if str(x).strip()]
        errors = [str(x) for x in list(data.get("errors") or []) if str(x).strip()]
        topics = [str(x) for x in list(data.get("topics") or []) if str(x).strip()]
        recent = [str(x) for x in list(data.get("recent") or []) if str(x).strip()]
        mode = str(data.get("active_mode") or "").strip().lower()
        persona = _normalize_persona_snapshot(data.get("persona_snapshot"))

        if tasks:
            lines.append("tasks: " + " | ".join(tasks[:3]))
        if errors:
            lines.append("errors: " + " | ".join(errors[:2]))
        if topics:
            lines.append("topics: " + ", ".join(topics[:4]))
        if mode:
            lines.append(f"mode: {mode}")
        if bool(data.get("technical_focus")):
            lines.append("focus: technical")
        if persona:
            mood = str(persona.get("mood") or "neutral").strip().lower() or "neutral"
            traits = dict(persona.get("traits") or {})
            trait_bits: list[str] = []
            for key in ("warmth", "sarcasm", "verbosity", "strictness", "teasing", "empathy"):
                if key not in traits:
                    continue
                try:
                    trait_bits.append(f"{key}={float(traits.get(key)):.2f}")
                except Exception:
                    continue
            line = f"persona: mood={mood}"
            if trait_bits:
                line += ", " + ", ".join(trait_bits[:4])
            lines.append(line)
        if recent:
            lines.append("recent: " + " | ".join(recent[:3]))
        return " || ".join(lines).strip()

    @staticmethod
    def _normalize_item(item: dict[str, Any]) -> dict[str, Any]:
        row = dict(item or {})
        meta_map = dict(row.get("meta") or {}) if isinstance(row.get("meta"), dict) else {}
        text = str(row.get("text") or row.get("content") or "").strip()
        ts_value = to_local_iso(row.get("ts"), default="")
        if not ts_value:
            ts_value = now_local_ts()
        tags = _dedupe_str_list([str(x).strip().lower() for x in list(row.get("tags") or []) if str(x).strip()])
        topic = str(row.get("topic") or meta_map.get("topic") or "").strip().lower()
        if topic.startswith("topic_"):
            topic = topic.replace("topic_", "", 1)
        topics = [str(x).strip().lower().replace("topic_", "", 1) for x in list(row.get("topics") or []) if str(x).strip()]
        topics.extend(_topic_list_from_tags(tags))
        topics = _dedupe_str_list([x for x in topics if x])
        active_mode = str(row.get("active_mode") or meta_map.get("active_mode") or "").strip().lower()
        persona_snapshot = _normalize_persona_snapshot(row.get("persona_snapshot") or meta_map.get("persona_snapshot"))
        return {
            "id": str(row.get("id") or row.get("event_id") or ""),
            "role": str(row.get("role") or "user"),
            "type": str(row.get("type") or "message"),
            "text": text,
            "thinking": str(row.get("thinking") or ""),
            "ts": ts_value,
            "lang": str(row.get("lang") or ""),
            "intent": str(row.get("intent") or "").strip().lower(),
            "emotion": str(row.get("emotion") or "").strip().lower(),
            "topic": topic,
            "topics": topics,
            "active_mode": active_mode,
            "tags": tags,
            "has_code": bool(row.get("has_code", False)),
            "persona_snapshot": persona_snapshot,
            "meta": dict(meta_map),
        }


def _dedupe_str_list(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in list(values or []):
        item = str(value or "").strip()
        if not item:
            continue
        low = item.lower()
        if low in seen:
            continue
        seen.add(low)
        out.append(item)
    return out


def _topic_list_from_tags(tags: list[str]) -> list[str]:
    out: list[str] = []
    for tag in list(tags or []):
        item = str(tag or "").strip().lower()
        if not item.startswith("topic_"):
            continue
        out.append(item.replace("topic_", "", 1))
    return _dedupe_str_list(out)


def _short_text(value: Any, limit: int = 100) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if len(text) <= int(limit):
        return text
    return text[: max(8, int(limit) - 3)].rstrip() + "..."


def _to_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _is_task_row(*, intent: str, tags: set[str]) -> bool:
    if intent in {"task", "bug_report", "question", "planning", "code_review", "clarification"}:
        return True
    if tags.intersection({"is_task", "is_question", "needs_steps"}):
        return True
    return False


def _is_error_row(*, intent: str, tags: set[str], text: str) -> bool:
    if intent == "bug_report":
        return True
    if tags.intersection({"has_traceback", "has_stacktrace", "has_logs"}):
        return True
    lowered = str(text or "").strip().lower()
    return any(token in lowered for token in ("traceback", "exception", "error", "failed", "crash"))


def _normalize_persona_snapshot(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    mood = str(value.get("mood") or "").strip().lower()
    traits_raw = dict(value.get("traits") or {})
    traits: dict[str, float] = {}
    for key in ("warmth", "sarcasm", "verbosity", "strictness", "teasing", "empathy"):
        if key not in traits_raw:
            continue
        try:
            traits[key] = max(0.0, min(1.0, float(traits_raw.get(key))))
        except Exception:
            continue
    out = {}
    if mood:
        out["mood"] = mood
    if traits:
        out["traits"] = traits
    if "character_id" in value and str(value.get("character_id") or "").strip():
        out["character_id"] = str(value.get("character_id")).strip().lower()
    if "active_mode" in value and str(value.get("active_mode") or "").strip():
        out["active_mode"] = str(value.get("active_mode")).strip().lower()
    return out
