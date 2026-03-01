from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Any

from modules.character.dialog_policies import (
    local_date_kyiv as dialog_local_date,
    starts_with_greeting as dialog_starts_with_greeting,
)
from utils.datetime_local import now_local_iso, now_local_ts, parse_time_to_epoch, to_local_iso
from config.settings import load_config


VALID_MODES = {"chat", "task", "coding", "voice", "silent", "debug"}
VALID_PROFILES = {"FAST", "BALANCED", "QUALITY"}


@dataclass(frozen=True)
class StateSnapshot:
    conversation_id: str
    turn_id: int
    mode: str
    active_character_id: str
    character_locked: bool
    character_last_switch_ts: float
    active_personality_id: str
    personality_blend: dict[str, Any]
    personality_locked: bool
    personality_last_switch_ts: float
    active_goal: str
    quality_profile: str
    active_tasks: list[dict[str, Any]]
    dialog_summary: str
    history: list[dict[str, str]]
    context_tags: dict[str, str]
    traits: dict[str, Any]
    policies: dict[str, Any]
    retrieved_memories: list[dict[str, Any]]
    context_stack: list[dict[str, Any]]
    last_tool_result: Any
    cooldowns: dict[str, Any]
    address_terms: dict[str, Any]
    raw: dict[str, Any]


class StateManager:
    def __init__(
        self,
        *,
        state_path: str | Path | None = None,
        state_store_dir: str | Path | None = None,
        autosave: bool = True,
        history_limit: int = 120,
    ):
        cfg = load_config()
        default_path = cfg.memory_dir / "brain_state.json"
        self.state_path = Path(state_path).expanduser() if state_path is not None else default_path
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        default_store_dir = cfg.memory_dir / "brain_state_store"
        if state_store_dir is not None:
            self.state_store_dir = Path(state_store_dir).expanduser()
        else:
            self.state_store_dir = default_store_dir
        self.state_store_dir.mkdir(parents=True, exist_ok=True)

        self.autosave = bool(autosave)
        self.history_limit = max(20, int(history_limit))
        self._lock = RLock()
        self._state = self._default_state()
        self.load()

    def _default_state(self) -> dict[str, Any]:
        return {
            "conversation_id": self._new_conversation_id(),
            "turn_id": 0,
            "mode": "chat",
            "active_character_id": "asya",
            "character_locked": False,
            "character_last_switch_ts": "",
            "active_personality_id": "default",
            "personality_blend": {
                "active": False,
                "from": "default",
                "to": "default",
                "step": 0,
                "steps": 4,
                "old_weight": 0.0,
                "new_weight": 1.0,
            },
            "personality_locked": False,
            "personality_last_switch_ts": "",
            "personality_cooldown_sec": 90.0,
            "active_goal": "",
            "quality_profile": "BALANCED",
            "active_tasks": [],
            "dialog_summary": "",
            "history": [],
            "context_tags": {},
            "traits": {},
            "policies": {},
            "retrieved_memories": [],
            "context_stack": [],
            "last_tool_result": None,
            "cooldowns": {
                "last_user_message": "",
                "last_user_hash": "",
                "last_user_ts": "",
                "prev_user_ts": "",
                "last_assistant_message": "",
                "last_assistant_ts": "",
                "last_action": "",
                "last_action_ts": "",
                "repeat_count": 0,
                "session_id": "",
                "prev_session_id": "",
                "greeting_date_local": "",
                "greeting_ts": "",
            },
            "address_terms": {
                "last_term_used_at": "",
                "term_used_turn_index": 0,
                "terms_used_count_session": 0,
                "session_id_snapshot": "",
                "banned_terms": [],
                "banned_terms_until": {},
                "last_disable_directive_ts": "",
                "last_enable_directive_ts": "",
            },
        }

    def load(self) -> None:
        with self._lock:
            payload = self._load_payload()
            if not isinstance(payload, dict):
                return
            merged = self._default_state()
            merged.update({k: v for k, v in payload.items() if k in merged})
            merged["mode"] = self._normalize_mode(merged.get("mode"))
            merged["quality_profile"] = self._normalize_profile(merged.get("quality_profile"))
            merged["conversation_id"] = str(merged.get("conversation_id") or self._new_conversation_id())
            merged["turn_id"] = int(merged.get("turn_id") or 0)
            merged["history"] = self._coerce_history(merged.get("history"))
            merged["active_tasks"] = self._coerce_dict_list(merged.get("active_tasks"))
            merged["retrieved_memories"] = self._coerce_dict_list(merged.get("retrieved_memories"))
            merged["context_stack"] = self._coerce_dict_list(merged.get("context_stack"))
            merged["context_tags"] = self._coerce_string_dict(merged.get("context_tags"))
            merged["cooldowns"] = self._coerce_cooldowns(merged.get("cooldowns"))
            merged["address_terms"] = self._coerce_address_terms(
                merged.get("address_terms"),
                conversation_id=str(merged.get("conversation_id") or ""),
            )
            merged["active_personality_id"] = str(merged.get("active_personality_id") or "default").strip().lower() or "default"
            merged["active_character_id"] = str(merged.get("active_character_id") or "asya").strip().lower() or "asya"
            merged["character_locked"] = bool(merged.get("character_locked", False))
            merged["character_last_switch_ts"] = to_local_iso(merged.get("character_last_switch_ts"), default="")
            merged["personality_blend"] = self._coerce_personality_blend(merged.get("personality_blend"))
            merged["personality_locked"] = bool(merged.get("personality_locked", False))
            merged["personality_last_switch_ts"] = to_local_iso(merged.get("personality_last_switch_ts"), default="")
            try:
                merged["personality_cooldown_sec"] = max(0.0, float(merged.get("personality_cooldown_sec") or 90.0))
            except Exception:
                merged["personality_cooldown_sec"] = 90.0
            self._state = merged

    def save(self) -> None:
        with self._lock:
            payload = self._to_json_safe(self._state)
            self._save_split_state(payload)
            self.state_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    def _load_payload(self) -> dict[str, Any]:
        split_payload = self._load_split_state()
        if split_payload:
            return split_payload
        if not self.state_path.exists():
            return {}
        try:
            payload = json.loads(self.state_path.read_text(encoding="utf-8-sig"))
        except Exception:
            return {}
        return payload if isinstance(payload, dict) else {}

    def _load_split_state(self) -> dict[str, Any]:
        root = self.state_store_dir
        if not root.exists():
            return {}
        index_path = root / "_index.json"
        if index_path.exists():
            try:
                index = json.loads(index_path.read_text(encoding="utf-8-sig"))
                keys = [str(x) for x in list(index.get("keys") or []) if str(x).strip()]
            except Exception:
                keys = []
        else:
            keys = []

        if not keys:
            keys = [p.stem for p in sorted(root.glob("*.json")) if p.name != "_index.json"]
        if not keys:
            return {}

        payload: dict[str, Any] = {}
        for key in keys:
            if key.startswith("_"):
                continue
            part_path = root / f"{key}.json"
            if not part_path.exists():
                continue
            try:
                value = json.loads(part_path.read_text(encoding="utf-8-sig"))
            except Exception:
                continue
            payload[key] = value
        return payload

    def _save_split_state(self, payload: dict[str, Any]) -> None:
        root = self.state_store_dir
        root.mkdir(parents=True, exist_ok=True)
        keys = sorted(str(k) for k in payload.keys())
        for key in keys:
            part_path = root / f"{key}.json"
            part_path.write_text(json.dumps(payload.get(key), ensure_ascii=False, indent=2), encoding="utf-8")

        for old_path in root.glob("*.json"):
            if old_path.name == "_index.json":
                continue
            if old_path.stem not in payload:
                try:
                    old_path.unlink()
                except Exception:
                    pass

        index = {
            "version": 1,
            "updated_at": now_local_iso(),
            "keys": keys,
        }
        (root / "_index.json").write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")

    def get(self, key: str, default=None):
        with self._lock:
            return self._state.get(key, default)

    def set(self, key: str, value) -> None:
        with self._lock:
            self._state[key] = value
        self._autosave()

    def patch(self, updates: dict[str, Any]) -> None:
        with self._lock:
            for key, value in dict(updates or {}).items():
                self._state[key] = value
        self._autosave()

    def new_conversation(self, conversation_id: str | None = None) -> str:
        conv_id = str(conversation_id or self._new_conversation_id()).strip()
        with self._lock:
            self._state["conversation_id"] = conv_id
            self._state["turn_id"] = 0
            self._state["history"] = []
            self._state["dialog_summary"] = ""
            self._state["address_terms"] = self._coerce_address_terms({}, conversation_id=conv_id)
        self._autosave()
        return conv_id

    def set_mode(self, mode: str) -> None:
        with self._lock:
            self._state["mode"] = self._normalize_mode(mode)
            self._touch_action(f"mode:{self._state['mode']}")
        self._autosave()

    def get_mode(self) -> str:
        with self._lock:
            return str(self._state.get("mode") or "chat")

    def set_active_character(self, character_id: str, *, locked: bool | None = None, switch_ts: float | None = None) -> None:
        cid = str(character_id or "asya").strip().lower() or "asya"
        with self._lock:
            self._state["active_character_id"] = cid
            if locked is not None:
                self._state["character_locked"] = bool(locked)
            if switch_ts is not None:
                self._state["character_last_switch_ts"] = to_local_iso(switch_ts, default=now_local_ts())
            elif not str(self._state.get("character_last_switch_ts") or "").strip():
                self._state["character_last_switch_ts"] = now_local_ts()
            self._touch_action(f"character:{cid}")
        self._autosave()

    def get_active_character(self) -> str:
        with self._lock:
            return str(self._state.get("active_character_id") or "asya").strip().lower() or "asya"

    def set_character_lock(self, locked: bool) -> None:
        with self._lock:
            self._state["character_locked"] = bool(locked)
            self._touch_action("character_lock" if locked else "character_unlock")
        self._autosave()

    def get_character_lock(self) -> bool:
        with self._lock:
            return bool(self._state.get("character_locked", False))

    def set_active_personality(
        self,
        personality_id: str,
        *,
        locked: bool | None = None,
        blend: dict[str, Any] | None = None,
        switch_ts: float | None = None,
    ) -> None:
        pid = str(personality_id or "default").strip().lower() or "default"
        with self._lock:
            self._state["active_personality_id"] = pid
            if locked is not None:
                self._state["personality_locked"] = bool(locked)
            if blend is not None:
                self._state["personality_blend"] = self._coerce_personality_blend(blend)
            if switch_ts is not None:
                self._state["personality_last_switch_ts"] = to_local_iso(switch_ts, default=now_local_ts())
            elif not str(self._state.get("personality_last_switch_ts") or "").strip():
                self._state["personality_last_switch_ts"] = now_local_ts()
            self._touch_action(f"personality:{pid}")
        self._autosave()

    def get_active_personality(self) -> str:
        with self._lock:
            return str(self._state.get("active_personality_id") or "default").strip().lower() or "default"

    def set_personality_lock(self, locked: bool) -> None:
        with self._lock:
            self._state["personality_locked"] = bool(locked)
            self._touch_action("personality_lock" if locked else "personality_unlock")
        self._autosave()

    def get_personality_lock(self) -> bool:
        with self._lock:
            return bool(self._state.get("personality_locked", False))

    def set_personality_cooldown(self, cooldown_sec: float) -> None:
        value = max(0.0, float(cooldown_sec))
        with self._lock:
            self._state["personality_cooldown_sec"] = value
        self._autosave()

    def get_personality_cooldown(self) -> float:
        with self._lock:
            try:
                return max(0.0, float(self._state.get("personality_cooldown_sec") or 0.0))
            except Exception:
                return 0.0

    def update_personality_blend(self, blend: dict[str, Any]) -> None:
        with self._lock:
            self._state["personality_blend"] = self._coerce_personality_blend(blend)
        self._autosave()

    def get_personality_blend(self) -> dict[str, Any]:
        with self._lock:
            return self._coerce_personality_blend(self._state.get("personality_blend"))

    def set_quality_profile(self, profile: str) -> None:
        with self._lock:
            self._state["quality_profile"] = self._normalize_profile(profile)
        self._autosave()

    def get_quality_profile(self) -> str:
        with self._lock:
            return str(self._state.get("quality_profile") or "BALANCED")

    def set_active_goal(self, goal: str) -> None:
        with self._lock:
            self._state["active_goal"] = str(goal or "").strip()
        self._autosave()

    def get_active_goal(self) -> str:
        with self._lock:
            return str(self._state.get("active_goal") or "")

    def set_dialog_summary(self, value: str) -> None:
        with self._lock:
            self._state["dialog_summary"] = str(value or "").strip()
        self._autosave()

    def get_dialog_summary(self) -> str:
        with self._lock:
            return str(self._state.get("dialog_summary") or "")

    def set_active_tasks(self, tasks) -> None:
        with self._lock:
            self._state["active_tasks"] = self._coerce_dict_list(tasks)
        self._autosave()

    def get_active_tasks(self) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(x) for x in self._coerce_dict_list(self._state.get("active_tasks"))]

    def set_last_tool_result(self, value) -> None:
        with self._lock:
            self._state["last_tool_result"] = value
            self._touch_action("tool_result")
        self._autosave()

    def get_last_tool_result(self):
        with self._lock:
            return self._state.get("last_tool_result")

    def set_address_terms(self, value: dict[str, Any]) -> None:
        with self._lock:
            self._state["address_terms"] = self._coerce_address_terms(
                value,
                conversation_id=str(self._state.get("conversation_id") or ""),
            )
        self._autosave()

    def push_context(self, ctx: dict[str, Any]) -> int:
        item = dict(ctx or {})
        with self._lock:
            stack = self._coerce_dict_list(self._state.get("context_stack"))
            stack.append(item)
            self._state["context_stack"] = stack
            depth = len(stack)
            self._touch_action("push_context")
        self._autosave()
        return depth

    def pop_context(self) -> dict[str, Any] | None:
        with self._lock:
            stack = self._coerce_dict_list(self._state.get("context_stack"))
            if not stack:
                return None
            item = dict(stack.pop())
            self._state["context_stack"] = stack
            self._touch_action("pop_context")
        self._autosave()
        return item

    def update_on_event(self, event_name: str, meta: dict[str, Any] | None = None) -> None:
        event = str(event_name or "").strip().lower()
        meta_map = dict(meta or {})
        with self._lock:
            if meta_map.get("mode"):
                self._state["mode"] = self._normalize_mode(meta_map.get("mode"))
            if event in {"voice_start", "voice.start", "voice:started"}:
                self._state["mode"] = "voice"
            elif event in {"voice_stop", "voice.stop", "voice:stopped"}:
                self._state["mode"] = "chat"
            self._touch_action(event or "event")
        self._autosave()

    def update_on_user_message(self, message: str, meta: dict[str, Any] | None = None) -> int:
        text = str(message or "").strip()
        if not text:
            with self._lock:
                return int(self._state.get("turn_id") or 0)

        meta_map = dict(meta or {})
        with self._lock:
            if meta_map.get("conversation_id"):
                self._state["conversation_id"] = str(meta_map.get("conversation_id")).strip()
            if meta_map.get("mode"):
                self._state["mode"] = self._normalize_mode(meta_map.get("mode"))
            if meta_map.get("quality_profile"):
                self._state["quality_profile"] = self._normalize_profile(meta_map.get("quality_profile"))
            if meta_map.get("active_goal"):
                self._state["active_goal"] = str(meta_map.get("active_goal") or "").strip()

            self._state["turn_id"] = int(self._state.get("turn_id") or 0) + 1
            self._append_history_item(role="user", content=text)

            cooldowns = self._coerce_cooldowns(self._state.get("cooldowns"))
            digest = hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()
            now_epoch = time.time()
            now_iso = now_local_ts()
            prev_hash = str(cooldowns.get("last_user_hash") or "")
            prev_raw = cooldowns.get("last_user_ts")
            prev_ts = parse_time_to_epoch(prev_raw, 0.0)
            prev_session = str(cooldowns.get("session_id") or "")
            current_session = str(self._state.get("conversation_id") or "")
            repeated = digest == prev_hash and (now_epoch - prev_ts) < 3.0
            cooldowns["repeat_count"] = int(cooldowns.get("repeat_count") or 0) + 1 if repeated else 0
            cooldowns["prev_user_ts"] = to_local_iso(prev_raw, default="")
            cooldowns["prev_session_id"] = prev_session
            cooldowns["session_id"] = current_session
            cooldowns["last_user_message"] = text
            cooldowns["last_user_hash"] = digest
            cooldowns["last_user_ts"] = now_iso
            self._state["cooldowns"] = cooldowns
            address_terms = self._coerce_address_terms(
                self._state.get("address_terms"),
                conversation_id=current_session,
            )
            self._state["address_terms"] = address_terms
            self._touch_action("user_message")
            turn_id = int(self._state.get("turn_id") or 0)
        self._autosave()
        return turn_id

    def update_on_assistant_message(self, message: str, meta: dict[str, Any] | None = None) -> None:
        text = str(message or "").strip()
        if not text:
            return
        meta_map = dict(meta or {})
        now_iso = now_local_ts()
        with self._lock:
            self._append_history_item(role="assistant", content=text)
            if meta_map.get("summary"):
                self._state["dialog_summary"] = str(meta_map.get("summary") or "").strip()
            if "last_tool_result" in meta_map:
                self._state["last_tool_result"] = meta_map.get("last_tool_result")
            cooldowns = self._coerce_cooldowns(self._state.get("cooldowns"))
            cooldowns["last_assistant_message"] = text
            cooldowns["last_assistant_ts"] = now_iso
            if _starts_with_greeting(text):
                cooldowns["greeting_date_local"] = _local_today()
                cooldowns["greeting_ts"] = now_iso
            self._state["cooldowns"] = cooldowns
            address_terms = self._coerce_address_terms(
                self._state.get("address_terms"),
                conversation_id=str(self._state.get("conversation_id") or ""),
            )
            if isinstance(meta_map.get("address_terms"), dict):
                address_terms = self._coerce_address_terms(
                    meta_map.get("address_terms"),
                    conversation_id=str(self._state.get("conversation_id") or ""),
                )
            self._state["address_terms"] = address_terms
            self._touch_action("assistant_message")
        self._autosave()

    def append_turn(self, role: str, content: str, *, max_items: int | None = None) -> None:
        role_norm = str(role or "user").strip().lower()
        if role_norm == "assistant":
            self.update_on_assistant_message(content, meta={})
            return
        if role_norm == "user":
            self.update_on_user_message(content, meta={})
            return
        with self._lock:
            self._append_history_item(role=role_norm, content=str(content or "").strip(), max_items=max_items)
        self._autosave()

    def get_history_tail(self, n: int = 8) -> list[dict[str, str]]:
        n_safe = max(1, int(n))
        with self._lock:
            history = self._coerce_history(self._state.get("history"))
            return history[-n_safe:]

    def snapshot(self) -> StateSnapshot:
        with self._lock:
            raw = dict(self._state)
            return StateSnapshot(
                conversation_id=str(self._state.get("conversation_id") or self._new_conversation_id()),
                turn_id=int(self._state.get("turn_id") or 0),
                mode=self._normalize_mode(self._state.get("mode")),
                active_character_id=str(self._state.get("active_character_id") or "asya"),
                character_locked=bool(self._state.get("character_locked", False)),
                character_last_switch_ts=parse_time_to_epoch(self._state.get("character_last_switch_ts"), 0.0),
                active_personality_id=str(self._state.get("active_personality_id") or "default"),
                personality_blend=self._coerce_personality_blend(self._state.get("personality_blend")),
                personality_locked=bool(self._state.get("personality_locked", False)),
                personality_last_switch_ts=parse_time_to_epoch(self._state.get("personality_last_switch_ts"), 0.0),
                active_goal=str(self._state.get("active_goal") or ""),
                quality_profile=self._normalize_profile(self._state.get("quality_profile")),
                active_tasks=[dict(x) for x in self._coerce_dict_list(self._state.get("active_tasks"))],
                dialog_summary=str(self._state.get("dialog_summary") or ""),
                history=self._coerce_history(self._state.get("history")),
                context_tags=self._coerce_string_dict(self._state.get("context_tags")),
                traits=dict(self._state.get("traits") or {}),
                policies=dict(self._state.get("policies") or {}),
                retrieved_memories=[dict(x) for x in self._coerce_dict_list(self._state.get("retrieved_memories"))],
                context_stack=[dict(x) for x in self._coerce_dict_list(self._state.get("context_stack"))],
                last_tool_result=self._state.get("last_tool_result"),
                cooldowns=self._coerce_cooldowns(self._state.get("cooldowns")),
                address_terms=self._coerce_address_terms(
                    self._state.get("address_terms"),
                    conversation_id=str(self._state.get("conversation_id") or ""),
                ),
                raw=raw,
            )

    def _append_history_item(self, *, role: str, content: str, max_items: int | None = None) -> None:
        text = str(content or "").strip()
        if not text:
            return
        history = self._coerce_history(self._state.get("history"))
        history.append({"role": str(role or "user"), "content": text})
        limit = self.history_limit if max_items is None else max(10, int(max_items))
        if len(history) > limit:
            history = history[-limit:]
        self._state["history"] = history

    def _touch_action(self, action: str) -> None:
        cooldowns = self._coerce_cooldowns(self._state.get("cooldowns"))
        cooldowns["last_action"] = str(action or "")
        cooldowns["last_action_ts"] = now_local_ts()
        self._state["cooldowns"] = cooldowns

    def _autosave(self) -> None:
        if self.autosave:
            self.save()

    @staticmethod
    def _normalize_mode(value) -> str:
        mode = str(value or "chat").strip().lower()
        return mode if mode in VALID_MODES else "chat"

    @staticmethod
    def _normalize_profile(value) -> str:
        profile = str(value or "BALANCED").strip().upper()
        return profile if profile in VALID_PROFILES else "BALANCED"

    @staticmethod
    def _new_conversation_id() -> str:
        return f"conv-{uuid.uuid4().hex[:12]}"

    @staticmethod
    def _coerce_history(value) -> list[dict[str, str]]:
        out: list[dict[str, str]] = []
        for row in list(value or []):
            if isinstance(row, dict):
                role = str(row.get("role") or "user")
                content = str(row.get("content") or "")
            elif isinstance(row, (list, tuple)) and len(row) >= 2:
                role = str(row[0] or "user")
                content = str(row[1] or "")
            else:
                continue
            role = role.lower().strip()
            if role in {"ai", "bot"}:
                role = "assistant"
            if role not in {"user", "assistant", "system"}:
                role = "user"
            content = content.strip()
            if not content:
                continue
            out.append({"role": role, "content": content})
        return out

    @staticmethod
    def _coerce_dict_list(value) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for row in list(value or []):
            if isinstance(row, dict):
                out.append(dict(row))
            else:
                text = str(row or "").strip()
                if text:
                    out.append({"value": text})
        return out

    @staticmethod
    def _coerce_string_dict(value) -> dict[str, str]:
        out: dict[str, str] = {}
        for key, val in dict(value or {}).items():
            k = str(key or "").strip()
            if not k:
                continue
            out[k] = str(val or "").strip()
        return out

    @staticmethod
    def _coerce_cooldowns(value) -> dict[str, Any]:
        base = {
            "last_user_message": "",
            "last_user_hash": "",
            "last_user_ts": "",
            "prev_user_ts": "",
            "last_assistant_message": "",
            "last_assistant_ts": "",
            "last_action": "",
            "last_action_ts": "",
            "repeat_count": 0,
            "session_id": "",
            "prev_session_id": "",
            "greeting_date_local": "",
            "greeting_ts": "",
        }
        data = dict(value or {})
        for key in base:
            if key in data:
                base[key] = data.get(key)
        base["last_user_ts"] = to_local_iso(base.get("last_user_ts"), default="")
        base["last_assistant_ts"] = to_local_iso(base.get("last_assistant_ts"), default="")
        base["prev_user_ts"] = to_local_iso(base.get("prev_user_ts"), default="")
        base["last_action_ts"] = to_local_iso(base.get("last_action_ts"), default="")
        base["greeting_ts"] = to_local_iso(base.get("greeting_ts"), default="")
        try:
            base["repeat_count"] = int(base["repeat_count"] or 0)
        except Exception:
            base["repeat_count"] = 0
        base["session_id"] = str(base.get("session_id") or "").strip()
        base["prev_session_id"] = str(base.get("prev_session_id") or "").strip()
        base["greeting_date_local"] = str(base.get("greeting_date_local") or "").strip()
        return base

    @staticmethod
    def _coerce_address_terms(value, *, conversation_id: str) -> dict[str, Any]:
        row = dict(value or {})
        out = {
            "last_term_used_at": "",
            "term_used_turn_index": 0,
            "terms_used_count_session": 0,
            "session_id_snapshot": "",
            "banned_terms": [],
            "banned_terms_until": {},
            "last_disable_directive_ts": "",
            "last_enable_directive_ts": "",
        }
        for key in out:
            if key in row:
                out[key] = row.get(key)

        out["last_term_used_at"] = to_local_iso(out.get("last_term_used_at"), default="")
        try:
            out["term_used_turn_index"] = int(out.get("term_used_turn_index") or 0)
        except Exception:
            out["term_used_turn_index"] = 0
        try:
            out["terms_used_count_session"] = max(0, int(out.get("terms_used_count_session") or 0))
        except Exception:
            out["terms_used_count_session"] = 0
        out["last_disable_directive_ts"] = to_local_iso(out.get("last_disable_directive_ts"), default="")
        out["last_enable_directive_ts"] = to_local_iso(out.get("last_enable_directive_ts"), default="")

        banned_terms: list[str] = []
        seen_terms: set[str] = set()
        for term in list(out.get("banned_terms") or []):
            item = str(term or "").strip().lower()
            if not item or item in seen_terms:
                continue
            seen_terms.add(item)
            banned_terms.append(item)
        out["banned_terms"] = banned_terms

        banned_until: dict[str, Any] = {}
        for term, marker in dict(out.get("banned_terms_until") or {}).items():
            key = str(term or "").strip().lower()
            if not key:
                continue
            banned_until[key] = marker
        out["banned_terms_until"] = banned_until

        current_session = str(conversation_id or "").strip()
        previous_session = str(out.get("session_id_snapshot") or "").strip()
        if current_session and previous_session and current_session != previous_session:
            out["terms_used_count_session"] = 0
            keep_terms: list[str] = []
            keep_markers: dict[str, Any] = {}
            for term in list(out.get("banned_terms") or []):
                marker = str(out["banned_terms_until"].get(term) or "").strip()
                if marker.startswith("session:"):
                    continue
                keep_terms.append(term)
                if marker:
                    keep_markers[term] = marker
            out["banned_terms"] = keep_terms
            out["banned_terms_until"] = keep_markers

        if current_session:
            out["session_id_snapshot"] = current_session
        else:
            out["session_id_snapshot"] = previous_session
        return out

    @staticmethod
    def _coerce_personality_blend(value) -> dict[str, Any]:
        row = dict(value or {})
        out = {
            "active": bool(row.get("active", False)),
            "from": str(row.get("from") or "default").strip().lower() or "default",
            "to": str(row.get("to") or "default").strip().lower() or "default",
            "step": 0,
            "steps": 4,
            "old_weight": 0.0,
            "new_weight": 1.0,
        }
        try:
            out["step"] = max(0, int(row.get("step") or 0))
        except Exception:
            out["step"] = 0
        try:
            out["steps"] = max(1, int(row.get("steps") or 4))
        except Exception:
            out["steps"] = 4
        try:
            out["old_weight"] = max(0.0, min(1.0, float(row.get("old_weight") or 0.0)))
        except Exception:
            out["old_weight"] = 0.0
        try:
            out["new_weight"] = max(0.0, min(1.0, float(row.get("new_weight") or 1.0)))
        except Exception:
            out["new_weight"] = 1.0
        return out

    @classmethod
    def _to_json_safe(cls, value):
        if value is None:
            return None
        if isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, dict):
            out = {}
            for k, v in value.items():
                out[str(k)] = cls._to_json_safe(v)
            return out
        if isinstance(value, (list, tuple)):
            return [cls._to_json_safe(x) for x in value]
        return str(value)


def _local_today() -> str:
    return str(dialog_local_date(time.time()))


def _starts_with_greeting(text: str) -> bool:
    return bool(dialog_starts_with_greeting(str(text or "")))
