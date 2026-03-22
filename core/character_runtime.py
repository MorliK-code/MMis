"""
Единый runtime для персонажа.

Объединяет:
- PersonalityEngine (personality profiles, switching, blending)
- StateManager (character state, mood, traits, context)
- PromptBuilder (persona block construction)
- CharacterEngine (traits, mood, evolution rules)
- CharacterStorage (state, learned traits)
- CharacterComposer (prompt building)
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock
from typing import Any

from config.settings import load_config
from core.mode_selector import normalize_mode_name
from metadata.taxonomy import normalize_emotion
from memory_core.utils.summary_quality import is_meaningful_summary_turn, sanitize_session_summary_text
from modules.character.composer import CharacterComposeResult, CharacterComposer, compute_context_trait_modifiers
from modules.character.dialog_policies import (
    local_date_kyiv as dialog_local_date,
    starts_with_greeting as dialog_starts_with_greeting,
)
from modules.character.evaluator import RuleEvaluator
from modules.character.learner import update_persona
from modules.character.signals import build_character_signals
from modules.character.persona_compiler import compile_system_persona
from modules.character.storage import CharacterStorage
from modules.character.trait_policy import apply_trait_delta, clamp_trait_map, clamp_trait_scalar, normalize_trait_record
from utils.datetime_local import now_local_iso, now_local_ts, parse_time_to_epoch, to_local_iso


_EMOTION_TARGETS: dict[str, dict[str, float]] = {
    "neutral": {"valence": 0.0, "arousal": 0.08},
    "happy": {"valence": 0.56, "arousal": 0.36},
    "excited": {"valence": 0.74, "arousal": 0.82},
    "frustrated": {"valence": -0.46, "arousal": 0.62},
    "angry": {"valence": -0.82, "arousal": 0.90},
    "sad": {"valence": -0.64, "arousal": 0.22},
    "anxious": {"valence": -0.52, "arousal": 0.70},
    "tired": {"valence": -0.24, "arousal": 0.10},
}


# =============================================================================
# META
# =============================================================================

@dataclass(frozen=True)
class CharacterMeta:
    """
    Метаданные персонажа.
    """
    id: str
    name: str
    version: str
    default_mood: str
    llm_profile: str
    prompt_files: dict[str, Any] = field(default_factory=dict)
    system_prompt: str = ""
    style_prompt: str = ""
    rules_prompt: str = ""
    voice_style: str = "neutral"
    traits: dict[str, float] = field(default_factory=dict)
    triggers: dict[str, list[str]] = field(default_factory=dict)


@dataclass(frozen=True)
class PersonalityProfile:
    """
    Профиль личности (из personality_engine).
    """
    id: str
    name: str
    version: str
    system_prompt: str
    style_prompt: str
    rules_prompt: str
    voice_style: str
    llm_profile: str
    traits: dict[str, float] = field(default_factory=dict)
    triggers: dict[str, list[str]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "version": self.version,
            "system_prompt": self.system_prompt,
            "style_prompt": self.style_prompt,
            "rules_prompt": self.rules_prompt,
            "voice_style": self.voice_style,
            "llm_profile": self.llm_profile,
            "traits": dict(self.traits or {}),
            "triggers": {
                "auto_enable_intents": list((self.triggers or {}).get("auto_enable_intents") or []),
                "auto_disable_intents": list((self.triggers or {}).get("auto_disable_intents") or []),
            },
        }


@dataclass(frozen=True)
class PersonalityDecision:
    """
    Решение о переключении личности (из personality_engine).
    """
    target_personality_id: str
    confidence: float
    reason: str
    switched: bool
    lock_after_switch: bool = False
    blend: dict[str, Any] = field(default_factory=dict)
    ts: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_personality_id": self.target_personality_id,
            "confidence": float(self.confidence),
            "reason": self.reason,
            "switched": bool(self.switched),
            "lock_after_switch": bool(self.lock_after_switch),
            "blend": dict(self.blend or {}),
            "ts": float(self.ts or 0.0),
        }


@dataclass(frozen=True)
class StateSnapshot:
    """
    Снимок состояния (из state_manager).
    """
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
    raw: dict[str, Any] = field(default_factory=dict)
    active_mode: str = "chatting"
    mode_lock: bool = False
    mode_until: str = ""
    web_mode: str = "auto"
    thinking_enabled: bool = False
    output_format: dict[str, Any] = field(default_factory=dict)
    last_signals: dict[str, Any] = field(default_factory=dict)
    last_actions: list[dict[str, Any]] = field(default_factory=list)

    @property
    def web_auto_profile(self) -> str:
        """
        Backward-compat shim for legacy callers that still read web_auto_profile.

        Web auto profiles are removed; keep a harmless empty value instead of
        raising AttributeError in older UI/API paths.
        """
        return ""


@dataclass(frozen=True)
class CharacterRuntimeResult:
    """
    Результат обновления CharacterRuntime.
    """
    character_id: str
    mood: str
    llm_profile: str
    traits: dict[str, Any]
    trait_values: dict[str, Any]
    state: dict[str, Any]
    user_addressing: dict[str, Any]
    prompt_block: str
    used_prompt_files: list[str] = field(default_factory=list)
    changes: list[dict[str, Any]] = field(default_factory=list)
    effective_traits: dict[str, float] = field(default_factory=dict)
    style_coefficients: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class PromptBudgets:
    """
    Бюджеты токенов для prompt (из prompt_builder).
    """
    total_tokens: int = 2200
    system_tokens: int = 240
    persona_tokens: int = 200
    state_tokens: int = 180
    tags_tokens: int = 100
    memory_tokens: int = 600
    long_summary_tokens: int = 180
    tail_tokens: int = 500
    user_tokens: int = 240
    output_schema_tokens: int = 220
    tail_turns: int = 8
    tail_turn_tokens: int = 120
    memory_item_tokens: int = 110

    @classmethod
    def from_policies(cls, policies: dict[str, Any]) -> PromptBudgets:
        p = _as_dict(policies)
        b = _as_dict(p.get("budgets"))
        return cls(
            total_tokens=_to_int(b.get("total_tokens", p.get("total_tokens")), 2200, 256),
            system_tokens=_to_int(b.get("system_tokens", p.get("system_tokens")), 240, 32),
            persona_tokens=_to_int(b.get("persona_tokens", p.get("persona_tokens")), 200, 32),
            state_tokens=_to_int(b.get("state_tokens", p.get("state_tokens")), 180, 32),
            tags_tokens=_to_int(b.get("tags_tokens", p.get("tags_tokens")), 100, 16),
            memory_tokens=_to_int(b.get("memory_tokens", p.get("memory_tokens")), 600, 64),
            long_summary_tokens=_to_int(b.get("long_summary_tokens", p.get("long_summary_tokens")), 180, 32),
            tail_tokens=_to_int(b.get("tail_tokens", p.get("tail_tokens")), 500, 64),
            user_tokens=_to_int(b.get("user_tokens", p.get("user_tokens")), 240, 32),
            output_schema_tokens=_to_int(b.get("output_schema_tokens", p.get("output_schema_tokens")), 220, 32),
            tail_turns=_to_int(b.get("tail_turns", p.get("tail_turns", p.get("conversation_tail_n"))), 8, 1),
            tail_turn_tokens=_to_int(b.get("tail_turn_tokens", p.get("tail_turn_tokens")), 120, 24),
            memory_item_tokens=_to_int(b.get("memory_item_tokens", p.get("memory_item_tokens")), 110, 24),
        )

    def as_dict(self) -> dict[str, int]:
        return {
            "total_tokens": self.total_tokens,
            "system_tokens": self.system_tokens,
            "persona_tokens": self.persona_tokens,
            "state_tokens": self.state_tokens,
            "tags_tokens": self.tags_tokens,
            "memory_tokens": self.memory_tokens,
            "long_summary_tokens": self.long_summary_tokens,
            "tail_tokens": self.tail_tokens,
            "user_tokens": self.user_tokens,
            "output_schema_tokens": self.output_schema_tokens,
            "tail_turns": self.tail_turns,
            "tail_turn_tokens": self.tail_turn_tokens,
            "memory_item_tokens": self.memory_item_tokens,
        }


@dataclass(frozen=True)
class PromptPack:
    """
    Результат построения prompt (из prompt_builder).
    """
    system_prompt: str
    user_message: str
    full_prompt: str
    messages: list[dict[str, str]]
    blocks: dict[str, str]
    token_usage: dict[str, int]
    budgets: dict[str, int]
    cut_info: dict[str, Any] = field(default_factory=dict)
    context_tags: dict[str, str] = field(default_factory=dict)
    selected_memories: list[dict[str, Any]] = field(default_factory=list)
    conversation_tail: list[dict[str, str]] = field(default_factory=list)


# =============================================================================
# CHARACTER RUNTIME
# =============================================================================

class CharacterRuntime:
    """
    Единый runtime для управления персонажем.

    Объединяет:
    - PersonalityEngine (personality profiles, switching, blending)
    - StateManager (character state, mood, traits, context)
    - PromptBuilder (persona block construction)
    - CharacterEngine (traits, mood, evolution rules)
    - CharacterStorage (state, learned traits)
    - CharacterComposer (prompt building)
    """

    DEFAULT_COOLDOWN_SEC = 90.0
    DEFAULT_MIN_CONFIDENCE = 0.65
    DEFAULT_BLEND_STEPS = 4

    def __init__(
        self,
        character_path: str | Path | None = None,
        *,
        storage: CharacterStorage | None = None,
        evaluator: RuleEvaluator | None = None,
        composer: CharacterComposer | None = None,
        trait_threshold: float = 0.55,
        max_trait_overlays: int = 4,
        cooldown_sec: float = DEFAULT_COOLDOWN_SEC,
        min_confidence: float = DEFAULT_MIN_CONFIDENCE,
        blend_steps: int = DEFAULT_BLEND_STEPS,
        state_path: str | Path | None = None,
        state_store_dir: str | Path | None = None,
        autosave: bool = True,
        history_limit: int = 120,
        action_history_limit: int = 3,
    ):
        """
        ������������ CharacterRuntime.

        Args:
            character_path: Путь к директории персонажей.
            storage: Хранилище персонажей.
            evaluator: Оценщик правил эволюции.
            composer: Композитор prompt.
            trait_threshold: Порог включения traits.
            max_trait_overlays: Максимум trait overlays.
            cooldown_sec: Кулдаун между переключениями personality.
            min_confidence: Минимальная уверенность для переключения.
            blend_steps: Шагов для blending personality.
        """
        if storage is not None:
            self.storage = storage
        else:
            logs_root = None
            if character_path is not None:
                try:
                    logs_root = Path(character_path).expanduser().resolve().parent / "logs"
                except Exception:
                    logs_root = None
            self.storage = CharacterStorage(root=character_path, logs_root=logs_root)
        self.evaluator = evaluator or RuleEvaluator()
        self.composer = composer or CharacterComposer(
            trait_threshold=trait_threshold,
            max_trait_overlays=max_trait_overlays,
        )
        self.storage.ensure_defaults()

        # Personality engine settings
        self.cooldown_sec = max(0.0, float(cooldown_sec))
        self.min_confidence = max(0.0, min(1.0, float(min_confidence)))
        self.blend_steps = max(1, int(blend_steps))

        # Caches
        self._meta_cache: dict[str, CharacterMeta] = {}
        self._meta_cache_revision: dict[str, str] = {}
        self._profiles_cache: dict[str, PersonalityProfile] = {}
        self._profiles_cache_revision: dict[str, str] = {}
        cfg = load_config()
        default_path = cfg.memory_dir / "brain_state.json"
        self.state_path = Path(state_path).expanduser() if state_path is not None else default_path
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        default_store_dir = cfg.memory_dir / "brain_state_store"
        # Legacy split-store location used only for one-way migration to single-file state.
        self.state_store_dir = Path(state_store_dir).expanduser() if state_store_dir is not None else default_store_dir
        self.autosave = bool(autosave)
        self.history_limit = max(20, int(history_limit))
        self.action_history_limit = max(1, int(action_history_limit))
        self._lock = RLock()

        self._state = self._default_state()
        self.load()

    # -------------------------------------------------------------------------
    # STATE MANAGEMENT (РёР· state_manager)
    # -------------------------------------------------------------------------

    def _default_state(self) -> dict[str, Any]:
        """Создать состояние по умолчанию."""
        conversation_id = self._new_conversation_id()
        output_format = {
            "show_parameters": None,
            "show_summary": None,
        }
        global_state = {
            "conversation_id": conversation_id,
            "turn_id": 0,
            "active_character_id": "asya",
            "active_mode": "chatting",
            "mode_lock": False,
            "mode_until": "",
            "web_mode": "auto",
            "thinking_enabled": False,
            "output_format": dict(output_format),
            "quality_profile": "BALANCED",
            "active_goal": "",
            "active_tasks": [],
            "context_stack": [],
            "last_signals": {},
            "last_actions": [],
        }
        return {
            "schema_version": 2,
            "global": dict(global_state),
            "characters": {},
            "conversation_id": conversation_id,
            "turn_id": 0,
            "mode": "chat",
            "active_mode": str(global_state["active_mode"]),
            "mode_lock": bool(global_state["mode_lock"]),
            "mode_until": "",
            "web_mode": str(global_state["web_mode"]),
            "thinking_enabled": bool(global_state["thinking_enabled"]),
            "output_format": dict(output_format),
            "last_signals": {},
            "last_actions": [],
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
            "personality_cooldown_sec": self.cooldown_sec,
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

    def _coerce_web_mode(self, value: Any) -> str:
        text = str(value or "auto").strip().lower()
        return text if text in {"auto", "on", "off"} else "auto"

    @staticmethod
    def _coerce_nullable_bool(value) -> bool | None:
        if isinstance(value, bool):
            return value
        if value is None:
            return None
        text = str(value).strip().lower()
        if text in {"none", "null", ""}:
            return None
        if text in {"1", "true", "yes", "on"}:
            return True
        if text in {"0", "false", "no", "off"}:
            return False
        return None

    @classmethod
    def _coerce_bool(cls, value, *, default: bool = False) -> bool:
        parsed = cls._coerce_nullable_bool(value)
        if isinstance(parsed, bool):
            return parsed
        return bool(default)

    def _coerce_output_format(self, value: Any) -> dict[str, Any]:
        row = dict(value or {})
        out = {
            "show_parameters": self._coerce_nullable_bool(row.get("show_parameters")),
            "show_summary": self._coerce_nullable_bool(row.get("show_summary")),
        }
        return out

    def _coerce_last_actions(self, value: Any) -> list[dict[str, Any]]:
        if not isinstance(value, list):
            return []
        out: list[dict[str, Any]] = []
        for row in value:
            if not isinstance(row, dict):
                continue
            item = dict(row)
            atype = str(item.get("type") or "").strip().upper()
            if atype:
                item["type"] = atype
            item["ts"] = to_local_iso(item.get("ts"), default=now_local_ts())
            out.append(item)
        return self._trim_last_actions(out)

    @staticmethod
    def _is_feedback_action(row: dict[str, Any]) -> bool:
        return str(dict(row or {}).get("type") or "").strip().upper() == "FEEDBACK_RECEIVED"

    def _trim_last_actions(self, actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
        rows = [dict(x) for x in list(actions or []) if isinstance(x, dict)]
        limit = max(1, int(self.action_history_limit))
        if len(rows) <= limit:
            return rows
        tail = list(rows[-limit:])
        if any(self._is_feedback_action(x) for x in tail):
            return tail
        feedback_index = -1
        for idx in range(len(rows) - 1, -1, -1):
            if self._is_feedback_action(rows[idx]):
                feedback_index = idx
                break
        if feedback_index < 0:
            return tail
        feedback_row = dict(rows[feedback_index])
        replace_index = 0
        for idx, item in enumerate(tail):
            if not self._is_feedback_action(item):
                replace_index = idx
                break
        tail[replace_index] = feedback_row
        return tail

    def _debug_state_snapshot(self) -> dict[str, Any]:
        active_character = str(self._state.get("active_character_id") or "asya").strip().lower() or "asya"
        characters = dict(self._state.get("characters") or {})
        entry = dict(characters.get(active_character) or {})
        persona = dict(entry.get("persona") or {})
        locks = dict(persona.get("locks") or {})
        trait_src = dict(persona.get("traits") or {})
        traits: dict[str, float] = {}
        for key in ("warmth", "sarcasm", "verbosity", "strictness", "teasing", "empathy"):
            if key not in trait_src:
                continue
            try:
                traits[key] = round(max(0.0, min(1.0, float(trait_src.get(key)))), 3)
            except Exception:
                continue
        return {
            "active_mode": self._normalize_active_mode(self._state.get("active_mode")),
            "mode_lock": self._coerce_bool(self._state.get("mode_lock"), default=False),
            "output_format": self._coerce_output_format(self._state.get("output_format")),
            "active_character_id": active_character,
            "active_personality_id": str(self._state.get("active_personality_id") or "default").strip().lower() or "default",
            "mood": str(self._state.get("mood") or persona.get("mood") or "neutral").strip().lower() or "neutral",
            "active_goal": str(self._state.get("active_goal") or "").strip(),
            "thinking_enabled": self._coerce_bool(self._state.get("thinking_enabled"), default=False),
            "web_mode": self._coerce_web_mode(self._state.get("web_mode")),
            "turn_id": int(self._state.get("turn_id") or 0),
            "traits": traits,
            "locks": {str(k): bool(v) for k, v in locks.items() if str(k).strip()},
            "bans": [str(x).strip() for x in list(persona.get("bans") or []) if str(x).strip()],
        }

    @staticmethod
    def _build_state_diff(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
        prev = dict(before or {})
        curr = dict(after or {})
        keys = set(prev.keys()).union(curr.keys())
        out: dict[str, Any] = {}
        for key in sorted(keys):
            old = prev.get(key)
            new = curr.get(key)
            if old == new:
                continue
            out[str(key)] = {"old": old, "new": new}
        return out

    def _append_last_action(self, action: dict[str, Any]) -> None:
        row = dict(action or {})
        row["type"] = str(row.get("type") or "EVENT").strip().upper() or "EVENT"
        row["ts"] = to_local_iso(row.get("ts"), default=now_local_ts())
        before = row.pop("_state_before", None)
        if isinstance(before, dict):
            diff = self._build_state_diff(before, self._debug_state_snapshot())
            if diff:
                row["state_diff"] = diff
        actions = self._coerce_last_actions(self._state.get("last_actions"))
        actions.append(row)
        self._state["last_actions"] = self._trim_last_actions(actions)

    def _build_global_section(self, state_map: dict[str, Any]) -> dict[str, Any]:
        src = dict(state_map or {})
        active_mode = self._normalize_active_mode(src.get("active_mode") or src.get("mode"))
        try:
            turn_id = int(src.get("turn_id") or 0)
        except Exception:
            turn_id = 0
        out = {
            "conversation_id": str(src.get("conversation_id") or self._new_conversation_id()),
            "turn_id": turn_id,
            "active_character_id": str(src.get("active_character_id") or "asya").strip().lower() or "asya",
            "active_mode": active_mode,
            "mode_lock": self._coerce_bool(src.get("mode_lock"), default=False),
            "mode_until": to_local_iso(src.get("mode_until"), default=""),
            "web_mode": self._coerce_web_mode(src.get("web_mode")),
            "thinking_enabled": self._coerce_bool(src.get("thinking_enabled"), default=False),
            "output_format": self._coerce_output_format(src.get("output_format")),
            "quality_profile": self._normalize_profile(src.get("quality_profile")),
            "active_goal": str(src.get("active_goal") or "").strip(),
            "active_tasks": self._coerce_dict_list(src.get("active_tasks")),
            "context_stack": self._coerce_dict_list(src.get("context_stack")),
            "last_signals": dict(src.get("last_signals") or {}),
            "last_actions": self._coerce_last_actions(src.get("last_actions")),
        }
        return out

    def _sync_flat_with_global(self, *, prefer_global: bool) -> None:
        src_global = self._state.get("global")
        if not isinstance(src_global, dict):
            src_global = {}
        if prefer_global:
            flat = dict(self._state)
            flat.update({
                "conversation_id": src_global.get("conversation_id", flat.get("conversation_id")),
                "turn_id": src_global.get("turn_id", flat.get("turn_id")),
                "active_character_id": src_global.get("active_character_id", flat.get("active_character_id")),
                "active_mode": src_global.get("active_mode", flat.get("active_mode")),
                "mode_lock": src_global.get("mode_lock", flat.get("mode_lock")),
                "mode_until": src_global.get("mode_until", flat.get("mode_until")),
                "web_mode": src_global.get("web_mode", flat.get("web_mode")),
                "thinking_enabled": src_global.get("thinking_enabled", flat.get("thinking_enabled")),
                "output_format": src_global.get("output_format", flat.get("output_format")),
                "quality_profile": src_global.get("quality_profile", flat.get("quality_profile")),
                "active_goal": src_global.get("active_goal", flat.get("active_goal")),
                "active_tasks": src_global.get("active_tasks", flat.get("active_tasks")),
                "context_stack": src_global.get("context_stack", flat.get("context_stack")),
                "last_signals": src_global.get("last_signals", flat.get("last_signals")),
                "last_actions": src_global.get("last_actions", flat.get("last_actions")),
            })
            self._state.update(flat)
        global_payload = self._build_global_section(self._state)
        self._state["global"] = global_payload
        self._state["schema_version"] = 2
        self._state["conversation_id"] = str(global_payload.get("conversation_id") or "")
        self._state["turn_id"] = int(global_payload.get("turn_id") or 0)
        self._state["active_character_id"] = str(global_payload.get("active_character_id") or "asya")
        self._state["active_mode"] = str(global_payload.get("active_mode") or "chatting")
        self._state["mode_lock"] = self._coerce_bool(global_payload.get("mode_lock"), default=False)
        self._state["mode_until"] = str(global_payload.get("mode_until") or "")
        self._state["web_mode"] = str(global_payload.get("web_mode") or "auto")
        self._state["thinking_enabled"] = self._coerce_bool(global_payload.get("thinking_enabled"), default=False)
        self._state["output_format"] = self._coerce_output_format(global_payload.get("output_format"))
        self._state["quality_profile"] = str(global_payload.get("quality_profile") or "BALANCED")
        self._state["active_goal"] = str(global_payload.get("active_goal") or "")
        self._state["active_tasks"] = list(global_payload.get("active_tasks") or [])
        self._state["context_stack"] = list(global_payload.get("context_stack") or [])
        self._state["last_signals"] = dict(global_payload.get("last_signals") or {})
        self._state["last_actions"] = list(global_payload.get("last_actions") or [])
        legacy_mode = str(self._state.get("mode") or "").strip().lower()
        if legacy_mode not in {"voice", "silent"}:
            self._state["mode"] = self._active_to_legacy_mode(self._state.get("active_mode"))

    def _ensure_characters_state(self) -> None:
        payload = self._state.get("characters")
        out: dict[str, Any] = {}
        if isinstance(payload, dict):
            for cid_raw, row in payload.items():
                cid = str(cid_raw or "").strip().lower()
                if not cid:
                    continue
                item = dict(row or {})
                persona = dict(item.get("persona") or self.storage.load_persona_state(cid) or {})
                persona.setdefault("traits", {})
                persona.setdefault("mood", "neutral")
                persona.setdefault("locks", {"feminine": True, "informal_you": True})
                persona.setdefault("bans", [])
                persona.setdefault(
                    "learned",
                    {
                        "preferences_confirmed": [],
                        "preferences_pending": [],
                        "style_bias": {},
                    },
                )
                persona.setdefault("baseline_traits", {})
                local = dict(item.get("local_context") or {})
                local.setdefault("last_topic_weights", {})
                local.setdefault("last_seen_ts", "")
                out[cid] = {"persona": persona, "local_context": local}
        active = str(self._state.get("active_character_id") or "asya").strip().lower() or "asya"
        if active not in out:
            persona_seed = self.storage.load_persona_state(active)
            out[active] = {
                "persona": {
                    "traits": dict(persona_seed.get("traits") or {}),
                    "mood": str(persona_seed.get("mood") or self._state.get("mood") or "neutral"),
                    "locks": dict(persona_seed.get("locks") or {"feminine": True, "informal_you": True}),
                    "bans": [str(x).strip() for x in list(persona_seed.get("bans") or []) if str(x).strip()],
                    "learned": {
                        "preferences_confirmed": list((dict(persona_seed.get("learned") or {}).get("preferences_confirmed") or [])),
                        "preferences_pending": list((dict(persona_seed.get("learned") or {}).get("preferences_pending") or [])),
                        "style_bias": dict((dict(persona_seed.get("learned") or {}).get("style_bias") or {})),
                    },
                    "baseline_traits": dict(persona_seed.get("baseline_traits") or {}),
                },
                "local_context": {"last_topic_weights": {}, "last_seen_ts": now_local_ts()},
            }
        self._state["characters"] = out

    def dispatch_action(self, action: dict[str, Any] | None) -> None:
        row = dict(action or {})
        atype = str(row.get("type") or "").strip().upper()
        if not atype:
            return
        with self._lock:
            before_state = self._debug_state_snapshot()
            row["type"] = atype
            row["ts"] = to_local_iso(row.get("ts"), default=now_local_ts())
            if atype == "TURN_STARTED":
                signals = dict(row.get("signals") or {})
                if signals:
                    self._state["last_signals"] = signals
            elif atype == "MODE_CHANGED":
                if "active_mode" in row:
                    self._state["active_mode"] = self._normalize_active_mode(row.get("active_mode"))
                if "mode_lock" in row:
                    self._state["mode_lock"] = self._coerce_bool(row.get("mode_lock"), default=False)
            elif atype == "OUTPUT_FORMAT_CHANGED":
                value = row.get("output_format")
                if isinstance(value, dict):
                    current = self._coerce_output_format(self._state.get("output_format"))
                    merged = dict(current)
                    merged.update({k: v for k, v in value.items() if k in {"show_parameters", "show_summary"}})
                    self._state["output_format"] = self._coerce_output_format(merged)
            elif atype == "CHARACTER_SWITCH":
                target = str(row.get("character_id") or "").strip().lower()
                if target:
                    self._state["active_character_id"] = target
            row["_state_before"] = before_state
            self._append_last_action(row)
            self._sync_flat_with_global(prefer_global=False)
            self._ensure_characters_state()
        self._autosave()

    def load(self) -> None:
        with self._lock:
            payload = self._load_payload()
            if not isinstance(payload, dict):
                return
            merged = self._default_state()
            merged.update({k: v for k, v in payload.items() if k in merged})
            if isinstance(payload.get("global"), dict):
                merged["global"] = dict(payload.get("global") or {})
            if isinstance(payload.get("characters"), dict):
                merged["characters"] = dict(payload.get("characters") or {})
            if "schema_version" in payload:
                merged["schema_version"] = int(payload.get("schema_version") or 2)
            self._state = dict(merged)
            self._sync_flat_with_global(prefer_global=bool(payload.get("global")))
            self._ensure_characters_state()
            merged = dict(self._state)
            merged["mode"] = self._normalize_mode(merged.get("mode"))
            merged["active_mode"] = self._normalize_active_mode(merged.get("active_mode"))
            merged["mode_lock"] = self._coerce_bool(merged.get("mode_lock"), default=False)
            merged["mode_until"] = to_local_iso(merged.get("mode_until"), default="")
            merged["web_mode"] = self._coerce_web_mode(merged.get("web_mode"))
            merged["thinking_enabled"] = self._coerce_bool(merged.get("thinking_enabled"), default=False)
            merged["output_format"] = self._coerce_output_format(merged.get("output_format"))
            merged["last_signals"] = dict(merged.get("last_signals") or {})
            merged["last_actions"] = self._coerce_last_actions(merged.get("last_actions"))
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
            merged["quality_profile"] = self._resolve_quality_profile_from_character(
                character_id=merged.get("active_character_id"),
                fallback=merged.get("quality_profile"),
            )
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
            self._sync_flat_with_global(prefer_global=False)
            self._ensure_characters_state()

    def save(self) -> None:
        with self._lock:
            self._sync_flat_with_global(prefer_global=False)
            self._ensure_characters_state()
            payload = self._to_json_safe(self._ordered_state_map(self._state))
            self.state_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    def _ordered_state_map(self, state_map: dict[str, Any]) -> dict[str, Any]:
        src = dict(state_map or {})
        if isinstance(src.get("global"), dict):
            src["global"] = self._ordered_global_map(src.get("global") or {})
        ordered_keys = [
            "schema_version",
            "global",
            "characters",
            "conversation_id",
            "turn_id",
            "active_character_id",
            "mode",
            "active_mode",
            "mode_lock",
            "mode_until",
            "web_mode",
            "thinking_enabled",
            "output_format",
            "quality_profile",
            "active_goal",
            "active_tasks",
            "last_signals",
            "last_actions",
            "context_tags",
            "traits",
            "policies",
            "retrieved_memories",
            "context_stack",
            "last_tool_result",
            "cooldowns",
            "address_terms",
            "dialog_summary",
            "history",
            "character_locked",
            "character_last_switch_ts",
            "active_personality_id",
            "personality_blend",
            "personality_locked",
            "personality_last_switch_ts",
            "personality_cooldown_sec",
        ]
        out: dict[str, Any] = {}
        for key in ordered_keys:
            if key in src:
                out[key] = src.get(key)
        for key, value in src.items():
            if key in out:
                continue
            out[key] = value
        return out

    def _ordered_global_map(self, section: dict[str, Any]) -> dict[str, Any]:
        src = dict(section or {})
        ordered_keys = [
            "conversation_id",
            "turn_id",
            "active_character_id",
            "active_mode",
            "mode_lock",
            "mode_until",
            "web_mode",
            "thinking_enabled",
            "output_format",
            "quality_profile",
            "active_goal",
            "active_tasks",
            "last_signals",
            "last_actions",
            "context_stack",
        ]
        out: dict[str, Any] = {}
        for key in ordered_keys:
            if key in src:
                out[key] = src.get(key)
        for key, value in src.items():
            if key in out:
                continue
            out[key] = value
        return out

    def _load_payload(self) -> dict[str, Any]:
        if self.state_path.exists():
            try:
                payload = json.loads(self.state_path.read_text(encoding="utf-8-sig"))
            except Exception:
                payload = {}
            if isinstance(payload, dict):
                return payload

        # Legacy fallback: import split state once when single-file state is absent.
        split_payload = self._load_split_state()
        if split_payload:
            try:
                normalized = self._to_json_safe(self._ordered_state_map(split_payload))
                self.state_path.write_text(
                    json.dumps(normalized, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            except Exception:
                pass
            return split_payload
        return {}

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

    def _autosave(self) -> None:
        if self.autosave:
            self.save()

    def snapshot(self) -> StateSnapshot:
        """Получить снимок состояния."""
        with self._lock:
            self._sync_flat_with_global(prefer_global=False)
            self._ensure_characters_state()
            raw = dict(self._state)
        return StateSnapshot(
            conversation_id=str(raw.get("conversation_id") or self._new_conversation_id()),
            turn_id=int(raw.get("turn_id") or 0),
            mode=self._normalize_mode(raw.get("mode")),
            active_character_id=str(raw.get("active_character_id") or "asya"),
            character_locked=bool(raw.get("character_locked", False)),
            character_last_switch_ts=parse_time_to_epoch(raw.get("character_last_switch_ts"), 0.0),
            active_personality_id=str(raw.get("active_personality_id") or "default"),
            personality_blend=self._coerce_personality_blend(raw.get("personality_blend")),
            personality_locked=bool(raw.get("personality_locked", False)),
            personality_last_switch_ts=parse_time_to_epoch(raw.get("personality_last_switch_ts"), 0.0),
            active_goal=str(raw.get("active_goal") or ""),
            quality_profile=self._normalize_profile(raw.get("quality_profile")),
            active_tasks=[dict(x) for x in self._coerce_dict_list(raw.get("active_tasks"))],
            dialog_summary=str(raw.get("dialog_summary") or ""),
            history=self._coerce_history(raw.get("history")),
            context_tags=self._coerce_string_dict(raw.get("context_tags")),
            traits=dict(raw.get("traits") or {}),
            policies=dict(raw.get("policies") or {}),
            retrieved_memories=[dict(x) for x in self._coerce_dict_list(raw.get("retrieved_memories"))],
            context_stack=[dict(x) for x in self._coerce_dict_list(raw.get("context_stack"))],
            last_tool_result=raw.get("last_tool_result"),
            cooldowns=self._coerce_cooldowns(raw.get("cooldowns")),
            address_terms=self._coerce_address_terms(
                raw.get("address_terms"),
                conversation_id=str(raw.get("conversation_id") or ""),
            ),
            raw=raw,
            active_mode=self._normalize_active_mode(raw.get("active_mode")),
            mode_lock=self._coerce_bool(raw.get("mode_lock"), default=False),
            mode_until=to_local_iso(raw.get("mode_until"), default=""),
            web_mode=self._coerce_web_mode(raw.get("web_mode")),
            thinking_enabled=self._coerce_bool(raw.get("thinking_enabled"), default=False),
            output_format=self._coerce_output_format(raw.get("output_format")),
            last_signals=dict(raw.get("last_signals") or {}),
            last_actions=self._coerce_last_actions(raw.get("last_actions")),
        )

    def get(self, key: str, default=None):
        with self._lock:
            return self._state.get(key, default)

    def set(self, key: str, value) -> None:
        with self._lock:
            self._state[key] = value
            self._sync_flat_with_global(prefer_global=False)
            self._ensure_characters_state()
        self._autosave()

    def patch(self, updates: dict[str, Any]) -> None:
        with self._lock:
            for key, value in dict(updates or {}).items():
                self._state[key] = value
            self._sync_flat_with_global(prefer_global=False)
            self._ensure_characters_state()
        self._autosave()

    def new_conversation(self, conversation_id: str | None = None) -> str:
        conv_id = str(conversation_id or self._new_conversation_id()).strip()
        with self._lock:
            self._state["conversation_id"] = conv_id
            self._state["turn_id"] = 0
            self._state["history"] = []
            self._state["dialog_summary"] = ""
            self._state["last_signals"] = {}
            self._state["last_actions"] = []
            self._state["address_terms"] = self._coerce_address_terms({}, conversation_id=conv_id)
            self._sync_flat_with_global(prefer_global=False)
        self._autosave()
        return conv_id

    def set_mode(self, mode: str) -> None:
        raw = str(mode or "").strip().lower()
        with self._lock:
            before_state = self._debug_state_snapshot()
            if raw in {"voice", "silent"}:
                self._state["mode"] = raw
            else:
                active = self._normalize_active_mode(raw)
                self._state["active_mode"] = active
                self._state["mode"] = self._active_to_legacy_mode(active)
            self._sync_flat_with_global(prefer_global=False)
            self._touch_action(f"mode:{self._state.get('active_mode') or self._state.get('mode')}")
            self._append_last_action(
                {
                    "type": "MODE_CHANGED",
                    "active_mode": str(self._state.get("active_mode") or "chatting"),
                    "mode_lock": bool(self._state.get("mode_lock", False)),
                    "_state_before": before_state,
                }
            )
        self._autosave()

    def get_mode(self) -> str:
        with self._lock:
            return str(self._state.get("mode") or "chat")

    def get_active_mode(self) -> str:
        with self._lock:
            return self._normalize_active_mode(self._state.get("active_mode"))

    def set_mode_lock(self, enabled: bool) -> None:
        with self._lock:
            before_state = self._debug_state_snapshot()
            self._state["mode_lock"] = self._coerce_bool(enabled, default=False)
            self._sync_flat_with_global(prefer_global=False)
            self._touch_action(f"mode_lock:{str(bool(enabled)).lower()}")
            self._append_last_action(
                {
                    "type": "MODE_CHANGED",
                    "active_mode": str(self._state.get("active_mode") or "chatting"),
                    "mode_lock": bool(enabled),
                    "_state_before": before_state,
                }
            )
        self._autosave()

    def get_mode_lock(self) -> bool:
        with self._lock:
            return self._coerce_bool(self._state.get("mode_lock"), default=False)

    def set_output_format(
        self,
        *,
        show_parameters: bool | None = None,
        show_summary: bool | None = None,
    ) -> None:
        with self._lock:
            before_state = self._debug_state_snapshot()
            current = self._coerce_output_format(self._state.get("output_format"))
            updates: dict[str, bool | None] = {}
            if show_parameters is not None:
                updates["show_parameters"] = bool(show_parameters)
            if show_summary is not None:
                updates["show_summary"] = bool(show_summary)
            if not updates:
                return
            merged = dict(current)
            merged.update(updates)
            self._state["output_format"] = self._coerce_output_format(merged)
            self._sync_flat_with_global(prefer_global=False)
            self._touch_action("output_format")
            self._append_last_action(
                {
                    "type": "OUTPUT_FORMAT_CHANGED",
                    "output_format": dict(self._state.get("output_format") or {}),
                    "_state_before": before_state,
                }
            )
        self._autosave()

    def _set_active_character_state(self, character_id: str, *, locked: bool | None = None, switch_ts: float | None = None) -> None:
        before_state = self._debug_state_snapshot()
        cid = str(character_id or "asya").strip().lower() or "asya"
        self._state["active_character_id"] = cid
        self._state["quality_profile"] = self._resolve_quality_profile_from_character(
            character_id=cid,
            fallback=self._state.get("quality_profile"),
        )
        if locked is not None:
            self._state["character_locked"] = bool(locked)
        if switch_ts is not None:
            self._state["character_last_switch_ts"] = to_local_iso(switch_ts, default=now_local_ts())
        elif not str(self._state.get("character_last_switch_ts") or "").strip():
            self._state["character_last_switch_ts"] = now_local_ts()
        self._ensure_characters_state()
        characters = dict(self._state.get("characters") or {})
        entry = dict(characters.get(cid) or {})
        local = dict(entry.get("local_context") or {})
        local["last_seen_ts"] = now_local_ts()
        entry["local_context"] = local
        characters[cid] = entry
        self._state["characters"] = characters
        self._append_last_action({"type": "CHARACTER_SWITCH", "character_id": cid, "_state_before": before_state})
        self._sync_flat_with_global(prefer_global=False)
        self._touch_action(f"character:{cid}")

    def get_active_character(self) -> str:
        with self._lock:
            return str(self._state.get("active_character_id") or "asya").strip().lower() or "asya"

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
            before_state = self._debug_state_snapshot()
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
            self._append_last_action(
                {
                    "type": "PERSONALITY_SWITCH",
                    "personality_id": pid,
                    "locked": bool(self._state.get("personality_locked", False)),
                    "_state_before": before_state,
                }
            )
        self._autosave()

    def get_active_personality(self) -> str:
        with self._lock:
            return str(self._state.get("active_personality_id") or "default").strip().lower() or "default"

    def set_quality_profile(self, profile: str) -> None:
        with self._lock:
            self._state["quality_profile"] = self._normalize_profile(profile)
        self._autosave()

    def get_quality_profile(self) -> str:
        with self._lock:
            return str(self._state.get("quality_profile") or "BALANCED")

    def update_personality_blend(self, blend: dict[str, Any]) -> None:
        with self._lock:
            self._state["personality_blend"] = self._coerce_personality_blend(blend)
        self._autosave()

    def get_personality_blend(self) -> dict[str, Any]:
        with self._lock:
            return self._coerce_personality_blend(self._state.get("personality_blend"))

    def set_dialog_summary(self, value: str) -> None:
        with self._lock:
            before_state = self._debug_state_snapshot()
            self._state["dialog_summary"] = str(value or "").strip()
            self._append_last_action(
                {
                    "type": "SUMMARY_UPDATED",
                    "summary_chars": len(str(value or "").strip()),
                    "_state_before": before_state,
                }
            )
            self._sync_flat_with_global(prefer_global=False)
        self._autosave()

    def get_dialog_summary(self) -> str:
        with self._lock:
            return str(self._state.get("dialog_summary") or "")

    def set_last_tool_result(self, value) -> None:
        with self._lock:
            before_state = self._debug_state_snapshot()
            self._state["last_tool_result"] = value
            self._append_last_action({"type": "TOOL_RESULT", "_state_before": before_state})
            self._sync_flat_with_global(prefer_global=False)
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

    def get_user_addressing(self, character_id: str | None = None) -> dict[str, Any]:
        cid = self._validate_character(character_id)
        try:
            return dict(self.storage.load_user_addressing(cid) or {})
        except Exception:
            return {
                "canonical_name": "",
                "allowed_forms": [],
                "forbidden_forms": [],
                "allow_diminutives": False,
                "use_name_by_default": False,
                "updated_at": "",
            }

    def set_context_tags(self, tags: dict[str, str]) -> None:
        with self._lock:
            self._state["context_tags"] = self._coerce_string_dict(tags)
        self._autosave()

    def get_context_tags(self) -> dict[str, str]:
        with self._lock:
            return self._coerce_string_dict(self._state.get("context_tags"))

    def get_persona_debug(self, state: dict[str, Any] | None = None, *, max_deltas: int = 3) -> dict[str, Any]:
        with self._lock:
            merged = dict(self._state)
            if isinstance(state, dict):
                merged.update(dict(state))
            active_character = str(merged.get("active_character_id") or "asya").strip().lower() or "asya"
            active_mode = self._normalize_active_mode(merged.get("active_mode") or merged.get("mode") or "chatting")
            mode_lock = bool(merged.get("mode_lock", False))
            characters = dict(merged.get("characters") or {})
            entry = dict(characters.get(active_character) or {})
            persona = dict(entry.get("persona") or {})
            mood = str(merged.get("mood") or persona.get("mood") or "neutral").strip().lower() or "neutral"
            traits_raw = dict(persona.get("traits") or {})
            traits: dict[str, float] = {}
            for key in ("warmth", "sarcasm", "teasing", "strictness", "verbosity", "empathy"):
                if key not in traits_raw:
                    continue
                try:
                    traits[key] = round(max(0.0, min(1.0, float(traits_raw.get(key)))), 3)
                except Exception:
                    continue
            locks = dict(persona.get("locks") or {})
            bans = [str(x).strip() for x in list(persona.get("bans") or []) if str(x).strip()]
            actions = list(self._coerce_last_actions(merged.get("last_actions")))
            deltas = [dict(x.get("state_diff") or {}) for x in reversed(actions) if isinstance(x, dict) and isinstance(x.get("state_diff"), dict)]
            deltas = deltas[: max(1, int(max_deltas))]
            events = self._tail_character_events(active_character, limit=8)
            last_event = dict(events[-1] or {}) if events else {}
            event_changes = [dict(x) for x in list(last_event.get("changes") or []) if isinstance(x, dict)][:8]
            learner_delta = dict(last_event.get("learner_delta") or {})
            user_addressing = self.get_user_addressing(active_character)
            feedback = []
            for row in event_changes:
                if str(row.get("kind") or "").strip().lower() == "user_feedback":
                    feedback.extend([str(x).strip() for x in list(row.get("feedback") or []) if str(x).strip()])
            return {
                "active_character_id": active_character,
                "active_mode": active_mode,
                "mode_lock": self._coerce_bool(mode_lock, default=False),
                "mood": mood,
                "traits": traits,
                "locks": locks,
                "bans": bans,
                "user_addressing": user_addressing,
                "last_deltas": deltas,
                "last_event_changes": event_changes,
                "learner_delta": learner_delta,
                "feedback": feedback[:8],
            }

    def get_brain_debug(self, state: dict[str, Any] | None = None, *, max_actions: int = 3) -> dict[str, Any]:
        with self._lock:
            merged = dict(self._state)
            if isinstance(state, dict):
                merged.update(dict(state))
            active_character_id = str(merged.get("active_character_id") or "asya").strip().lower() or "asya"
            effective_quality_profile = self._resolve_quality_profile_from_character(
                character_id=active_character_id,
                fallback=merged.get("quality_profile"),
            )
            actions = self._coerce_last_actions(merged.get("last_actions"))
            tail_actions = actions[-max(1, int(max_actions)) :]
            feedback: list[str] = []
            for item in reversed(actions):
                atype = str(item.get("type") or "").strip().upper()
                if atype != "FEEDBACK_RECEIVED":
                    continue
                rows = [str(x).strip() for x in list(item.get("feedback") or []) if str(x).strip()]
                if rows:
                    feedback = rows[:8]
                    break
            active_tasks = [dict(x) for x in self._coerce_dict_list(merged.get("active_tasks"))][:8]
            return {
                "active_character_id": active_character_id,
                "active_mode": self._normalize_active_mode(merged.get("active_mode") or merged.get("mode") or "chatting"),
                "mode_lock": self._coerce_bool(merged.get("mode_lock"), default=False),
                "web_mode": self._coerce_web_mode(merged.get("web_mode")),
                "thinking_enabled": self._coerce_bool(merged.get("thinking_enabled"), default=False),
                "output_format": self._coerce_output_format(merged.get("output_format")),
                "quality_profile": effective_quality_profile,
                "active_goal": str(merged.get("active_goal") or "").strip(),
                "active_tasks": active_tasks,
                "last_signals": dict(merged.get("last_signals") or {}),
                "feedback": feedback,
                "last_actions": tail_actions,
            }

    @staticmethod
    def _character_cache_revision(payload: dict[str, Any] | None) -> str:
        row = dict(payload or {})
        marker = {
            "id": str(row.get("id") or row.get("character_id") or "").strip().lower(),
            "name": str(row.get("name") or "").strip(),
            "version": str(row.get("version") or "").strip(),
            "default_mood": str(row.get("default_mood") or "").strip(),
            "default_mode": str(row.get("default_mode") or "").strip(),
            "llm_profile": str(row.get("llm_profile") or row.get("performance") or "").strip().upper(),
            "model_profile": str(row.get("model_profile") or row.get("llm_profile") or row.get("performance") or "").strip().upper(),
            "system_prompt": str(row.get("system_prompt") or "").strip(),
            "style_prompt": str(row.get("style_prompt") or "").strip(),
            "rules_prompt": str(row.get("rules_prompt") or "").strip(),
            "voice_style": str(row.get("voice_style") or "").strip(),
            "locks": dict(row.get("locks") or {}),
        }
        try:
            raw = json.dumps(marker, ensure_ascii=False, sort_keys=True)
        except Exception:
            raw = str(marker)
        return hashlib.sha1(raw.encode("utf-8", errors="ignore")).hexdigest()

    def _resolve_quality_profile_from_character(self, *, character_id: Any, fallback: Any = "BALANCED") -> str:
        cid = str(character_id or "").strip().lower()
        if not cid:
            return self._normalize_profile(fallback)
        try:
            payload = self.storage.load_character(cid)
            value = payload.get("performance") or payload.get("llm_profile") or payload.get("model_profile") or fallback
        except Exception:
            value = fallback
        return self._normalize_profile(value)

    def _tail_character_events(self, character_id: str, *, limit: int = 8) -> list[dict[str, Any]]:
        cid = self._validate_character(character_id)
        path = self.storage.character_events_path(cid)
        if not path.exists():
            path = self.storage.character_dir(cid) / "events.jsonl"
        if not path.exists():
            return []
        try:
            lines = path.read_text(encoding="utf-8-sig").splitlines()
        except Exception:
            return []
        out: list[dict[str, Any]] = []
        for raw in lines[-max(1, int(limit)) :]:
            line = str(raw or "").strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            if isinstance(row, dict):
                out.append(dict(row))
        return out

    def append_history(self, role: str, content: str, *, max_items: int = 120) -> None:
        with self._lock:
            self._append_history_item(role=str(role or "user").strip().lower(), content=str(content or "").strip(), max_items=max_items)
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
            before_state = self._debug_state_snapshot()
            if meta_map.get("mode"):
                mode_raw = str(meta_map.get("mode") or "").strip().lower()
                if mode_raw in {"voice", "silent"}:
                    self._state["mode"] = mode_raw
                else:
                    active = self._normalize_active_mode(mode_raw)
                    self._state["active_mode"] = active
                    self._state["mode"] = self._active_to_legacy_mode(active)
            if event in {"voice_start", "voice.start", "voice:started"}:
                self._state["mode"] = "voice"
            elif event in {"voice_stop", "voice.stop", "voice:stopped"}:
                self._state["mode"] = "chat"
                self._state["active_mode"] = "chatting"
            self._append_last_action(
                {
                    "type": "EVENT",
                    "event": event or "event",
                    "_state_before": before_state,
                }
            )
            self._sync_flat_with_global(prefer_global=False)
            self._touch_action(event or "event")
        self._autosave()

    def update_on_user_message(self, message: str, meta: dict[str, Any] | None = None) -> int:
        text = str(message or "").strip()
        if not text:
            with self._lock:
                return int(self._state.get("turn_id") or 0)

        meta_map = dict(meta or {})
        with self._lock:
            before_state = self._debug_state_snapshot()
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
            self._state["address_terms"] = self._coerce_address_terms(
                self._state.get("address_terms"),
                conversation_id=current_session,
            )
            # Emotion: prefer emotion_profile (memory), fallback to emotion (NLU)
            emotion_profile = meta_map.get("emotion_profile") or {}
            signals = {
                "intent": str(meta_map.get("intent") or "").strip().lower(),
                "emotion": str(
                    emotion_profile.get("primary") or
                    emotion_profile.get("label") or
                    meta_map.get("emotion") or
                    meta_map.get("mood") or
                    ""
                ).strip().lower(),
                "emotion_intensity": self._clamp01(self._to_float(
                    emotion_profile.get("intensity") or meta_map.get("emotion_intensity"), 0.0
                )),
                "emotion_arousal": self._clamp01(self._to_float(
                    emotion_profile.get("arousal") or meta_map.get("emotion_arousal"), 0.0
                )),
                "topic": str(meta_map.get("topic") or "").strip().lower(),
                "active_mode": str(self._state.get("active_mode") or "chatting"),
                "has_code": bool(meta_map.get("has_code", False)),
                "has_traceback": bool(meta_map.get("has_traceback", False)),
            }
            self._state["last_signals"] = {k: v for k, v in signals.items() if v not in {"", None}}
            self._append_last_action(
                {
                    "type": "TURN_STARTED",
                    "turn_id": int(self._state.get("turn_id") or 0),
                    "signals": dict(self._state.get("last_signals") or {}),
                    "_state_before": before_state,
                }
            )
            self._sync_flat_with_global(prefer_global=False)
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
            before_state = self._debug_state_snapshot()
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
            self._append_last_action(
                {
                    "type": "TURN_FINISHED",
                    "turn_id": int(self._state.get("turn_id") or 0),
                    "assistant_chars": len(text),
                    "_state_before": before_state,
                }
            )
            self._sync_flat_with_global(prefer_global=False)
            self._touch_action("assistant_message")
        self._autosave()

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

    # -------------------------------------------------------------------------
    # CHARACTER META & PROFILES
    # -------------------------------------------------------------------------

    def list_ids(self) -> list[str]:
        """Список доступных персонажей."""
        return self.storage.list_character_ids(include_disabled=False)

    def get_manifest(self) -> dict[str, Any]:
        """Получить манифест всех персонажей."""
        return self.storage.sync_manifest()

    def get_active_character_id(self, state: dict[str, Any] | None = None) -> str:
        """Получить ID активного персонажа."""
        state_map = dict(state or {})
        from_state = str(state_map.get("active_character_id") or "").strip().lower()
        if from_state:
            return from_state
        from_internal = self.get_active_character()
        if from_internal:
            return from_internal
        manifest = self.storage.load_manifest()
        from_manifest = str(manifest.get("active_character_id") or "").strip().lower()
        if from_manifest:
            return from_manifest
        ids = self.list_ids()
        return ids[0] if ids else "asya"

    def set_active_character(
        self,
        character_id: str,
        *,
        locked: bool | None = None,
        switch_ts: float | None = None,
    ) -> str:
        """Установить активного персонажа."""
        target = str(character_id or "").strip().lower()
        known = set(self.list_ids())
        if target not in known:
            raise ValueError(f"unknown character: {target}")
        manifest = self.storage.load_manifest()
        previous = str(manifest.get("active_character_id") or "").strip().lower()
        manifest["active_character_id"] = target
        self.storage.save_manifest(manifest)
        self.storage.append_manifest_event({
            "type": "active_character_changed",
            "from": previous,
            "to": target,
            "source": "manual",
        })
        if previous and previous != target:
            self.storage.append_event(previous, {"type": "switch_out", "source": "manual", "to": target})
        self.storage.append_event(target, {"type": "switch", "source": "manual", "target": target})
        for cid in {previous, target}:
            if not cid:
                continue
            self._meta_cache.pop(cid, None)
            self._meta_cache_revision.pop(cid, None)
            self._profiles_cache.pop(cid, None)
            self._profiles_cache_revision.pop(cid, None)
        with self._lock:
            self._set_active_character_state(target, locked=locked, switch_ts=switch_ts)
            self._state["active_personality_id"] = target
        self._autosave()
        return target

    def get_meta(self, character_id: str | None = None) -> CharacterMeta:
        """Получить метаданные персонажа."""
        cid = self._validate_character(character_id)
        character = self.storage.load_character(cid)
        revision = self._character_cache_revision(character)
        cached_meta = self._meta_cache.get(cid)
        if cached_meta is not None and self._meta_cache_revision.get(cid) == revision:
            return cached_meta
        profile = self._load_profile(cid, character=character, revision=revision)

        meta = CharacterMeta(
            id=cid,
            name=str(character.get("name") or cid).strip() or cid,
            version=str(character.get("version") or "1.0.0").strip() or "1.0.0",
            default_mood=str(character.get("default_mood") or "thoughtful").strip() or "thoughtful",
            llm_profile=str(character.get("performance") or character.get("llm_profile") or "BALANCED").strip().upper() or "BALANCED",
            prompt_files=dict(character.get("prompt_files") or {}),
            system_prompt=profile.system_prompt if profile else "",
            style_prompt=profile.style_prompt if profile else "",
            rules_prompt=profile.rules_prompt if profile else "",
            voice_style=profile.voice_style if profile else "neutral",
            traits=dict(profile.traits) if profile else {},
            triggers=dict(profile.triggers) if profile else {},
        )
        self._meta_cache[cid] = meta
        self._meta_cache_revision[cid] = revision
        return meta

    def _load_profile(
        self,
        character_id: str,
        *,
        character: dict[str, Any] | None = None,
        revision: str | None = None,
    ) -> PersonalityProfile | None:
        """Загрузить профиль личности."""
        cid = self._validate_character(character_id)
        payload = dict(character or {}) if isinstance(character, dict) else self.storage.load_character(cid)
        cache_revision = str(revision or self._character_cache_revision(payload))
        cached_profile = self._profiles_cache.get(cid)
        if cached_profile is not None and self._profiles_cache_revision.get(cid) == cache_revision:
            return cached_profile
        profile = PersonalityProfile(
            id=cid,
            name=str(payload.get("name") or cid).strip() or cid,
            version=str(payload.get("version") or "1.0.0").strip() or "1.0.0",
            system_prompt=str(payload.get("system_prompt") or "").strip(),
            style_prompt=str(payload.get("style_prompt") or "").strip(),
            rules_prompt=str(payload.get("rules_prompt") or "").strip(),
            voice_style=str(payload.get("voice_style") or "neutral").strip(),
            llm_profile=str(payload.get("performance") or payload.get("llm_profile") or "BALANCED").strip().upper() or "BALANCED",
            traits={},
            triggers={},
        )
        self._profiles_cache[cid] = profile
        self._profiles_cache_revision[cid] = cache_revision
        return profile

    # -------------------------------------------------------------------------
    # PERSONALITY ENGINE (РёР· personality_engine)
    # -------------------------------------------------------------------------

    def get_profile(self, profile_id: str) -> PersonalityProfile:
        """Получить профиль личности."""
        key = str(profile_id or "").strip().lower()
        return self._load_profile(key) or self._load_profile("default")

    def decide_personality(
        self,
        *,
        intent: str,
        emotion: str,
        recent_context: dict[str, Any] | None,
        user_pref: str = "",
        active_personality_id: str = "default",
        personality_locked: bool = False,
        last_switch_ts: float = 0.0,
        manual_personality: str = "",
        now_ts: float | None = None,
    ) -> PersonalityDecision:
        """
        Принять решение о переключении личности.
        """
        now = float(now_ts or time.time())
        active_id = self._normalize_personality_id(active_personality_id)
        manual = self._normalize_personality_id(manual_personality)

        # Manual override
        if manual and manual != "auto":
            target = manual if manual in self._profiles_cache or self._validate_character(manual) else active_id
            switched = target != active_id
            return PersonalityDecision(
                target_personality_id=target,
                confidence=1.0,
                reason="manual",
                switched=switched,
                lock_after_switch=True,
                blend=self._build_blend(active_id, target, switched),
                ts=now,
            )

        # Locked
        if personality_locked and manual != "auto":
            return PersonalityDecision(
                target_personality_id=active_id,
                confidence=1.0,
                reason="locked",
                switched=False,
                lock_after_switch=True,
                blend={},
                ts=now,
            )

        # Score personalities
        scores = self._score_personalities(
            intent=str(intent or "").strip().lower(),
            emotion=str(emotion or "").strip().lower(),
            recent_context=dict(recent_context or {}),
            user_pref=self._normalize_personality_id(user_pref),
        )
        ordered = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        if not ordered:
            return PersonalityDecision(
                target_personality_id=active_id,
                confidence=0.0,
                reason="no_scores",
                switched=False,
                blend={},
                ts=now,
            )

        target, best = ordered[0]
        second = ordered[1][1] if len(ordered) > 1 else 0.0
        confidence = best / max(0.001, best + second)
        confidence = max(0.0, min(1.0, confidence))

        if confidence < self.min_confidence:
            return PersonalityDecision(
                target_personality_id=active_id,
                confidence=confidence,
                reason="low_confidence",
                switched=False,
                blend={},
                ts=now,
            )

        if target == active_id:
            return PersonalityDecision(
                target_personality_id=active_id,
                confidence=confidence,
                reason="already_active",
                switched=False,
                blend={},
                ts=now,
            )

        elapsed = now - float(last_switch_ts or 0.0)
        if elapsed < self.cooldown_sec:
            return PersonalityDecision(
                target_personality_id=active_id,
                confidence=confidence,
                reason="cooldown",
                switched=False,
                blend={},
                ts=now,
            )

        return PersonalityDecision(
            target_personality_id=target,
            confidence=confidence,
            reason="auto",
            switched=True,
            lock_after_switch=False,
            blend=self._build_blend(active_id, target, True),
            ts=now,
        )

    def advance_blend(self, blend: dict[str, Any] | None) -> dict[str, Any]:
        """Продвинуть blend personality."""
        row = dict(blend or {})
        if not bool(row.get("active")):
            return {
                "active": False,
                "from": str(row.get("from") or ""),
                "to": str(row.get("to") or ""),
                "step": int(row.get("step") or 0),
                "steps": int(row.get("steps") or 0),
                "old_weight": float(row.get("old_weight") or 0.0),
                "new_weight": float(row.get("new_weight") or 1.0),
            }

        step = int(row.get("step") or 0) + 1
        steps = max(1, int(row.get("steps") or self.blend_steps))
        frac = min(1.0, step / float(steps))
        active = step < steps
        return {
            "active": active,
            "from": str(row.get("from") or ""),
            "to": str(row.get("to") or ""),
            "step": step,
            "steps": steps,
            "old_weight": round(max(0.0, 1.0 - frac), 2),
            "new_weight": round(min(1.0, frac), 2),
        }

    def _score_personalities(
        self,
        *,
        intent: str,
        emotion: str,
        recent_context: dict[str, Any],
        user_pref: str,
    ) -> dict[str, float]:
        """Оценить личности."""
        scores = {pid: 0.2 for pid in self.list_ids()}
        scores.setdefault("default", 0.25)

        mode = str(recent_context.get("mode") or "").strip().lower()
        topic = str(recent_context.get("topic") or "").strip().lower()

        if intent in {"task", "bug_report", "code_review", "question", "planning"}:
            scores["strict"] = scores.get("strict", 0.2) + 0.45
        if intent in {"chat"}:
            scores["flirty"] = scores.get("flirty", 0.2) + 0.36
        if emotion in {"frustrated", "angry", "sad", "anxious", "tired"}:
            scores["supportive"] = scores.get("supportive", 0.2) + 0.5
            scores["flirty"] = scores.get("flirty", 0.2) - 0.2
        if emotion in {"happy", "excited"} and intent in {"chat"}:
            scores["flirty"] = scores.get("flirty", 0.2) + 0.28

        if mode in {"task"}:
            scores["strict"] = scores.get("strict", 0.2) + 0.22
            scores["flirty"] = scores.get("flirty", 0.2) - 0.15
        if topic in {"python", "git", "ui", "llm", "security"}:
            scores["strict"] = scores.get("strict", 0.2) + 0.1

        if user_pref and user_pref in scores:
            scores[user_pref] += 0.22

        for pid in self.list_ids():
            profile = self.get_profile(pid)
            enables = {str(x).strip().lower() for x in list((profile.triggers or {}).get("auto_enable_intents") or [])}
            disables = {str(x).strip().lower() for x in list((profile.triggers or {}).get("auto_disable_intents") or [])}
            if intent and intent in enables:
                scores[pid] = scores.get(pid, 0.2) + 0.18
            if intent and intent in disables:
                scores[pid] = scores.get(pid, 0.2) - 0.18

        out = {}
        for pid, value in scores.items():
            out[pid] = max(0.01, min(1.5, float(value)))
        return out

    def update_emotional_state(
        self,
        character_id: str,
        *,
        state: dict[str, Any],
        signals,
        traits: dict[str, Any],
        context: dict[str, Any] | None = None,
        mood_hints: list[str] | None = None,
        now_ts: float | None = None,
        current_state: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        cid = self._validate_character(character_id)
        now = float(now_ts or time.time())
        current = self._coerce_emotion_state(current_state or self.storage.load_emotion_state(cid) or {})

        label = normalize_emotion(getattr(signals, "emotion", "") or "neutral")
        signal_intensity = self._clamp01(self._to_float(getattr(signals, "emotion_intensity", 0.0), 0.0))
        signal_arousal = self._clamp01(self._to_float(getattr(signals, "emotion_arousal", 0.0), 0.0))
        if label != "neutral" and signal_intensity <= 0.0:
            signal_intensity = 0.35

        target_base = dict(_EMOTION_TARGETS.get(label) or _EMOTION_TARGETS["neutral"])
        amplitude = signal_intensity if label != "neutral" else 0.0
        target_valence = self._clamp(float(target_base.get("valence", 0.0)) * amplitude, -1.0, 1.0)
        base_arousal = float(target_base.get("arousal", 0.0)) * (0.25 + (0.75 * amplitude))
        target_arousal = self._clamp01((base_arousal * 0.45) + ((signal_arousal or base_arousal) * 0.55))
        target_intensity = amplitude

        prev_label = normalize_emotion(current.get("trigger"))
        prev_valence = self._clamp(self._to_float(current.get("valence"), 0.0), -1.0, 1.0)
        prev_arousal = self._clamp01(self._to_float(current.get("arousal"), 0.0))
        prev_intensity = self._clamp01(self._to_float(current.get("intensity"), 0.0))
        cooldown_until = parse_time_to_epoch(current.get("cooldown_until_ts"), 0.0)

        alpha = 0.14 + (target_intensity * 0.42)
        if label == "neutral":
            alpha = 0.10 + min(0.06, prev_intensity * 0.08)
        if prev_label and prev_label == label and label != "neutral":
            alpha += 0.08
        if cooldown_until > now and label not in {"", "neutral", prev_label}:
            alpha *= 0.6
        if label == "neutral" and prev_intensity >= 0.65:
            alpha = min(alpha, 0.12)
        alpha = self._clamp(alpha, 0.08, 0.72)

        next_valence = prev_valence + ((target_valence - prev_valence) * alpha)
        next_arousal = prev_arousal + ((target_arousal - prev_arousal) * alpha)
        next_intensity = prev_intensity + ((target_intensity - prev_intensity) * alpha)
        if label == "neutral" and target_intensity <= 0.0:
            next_valence *= 0.96
            next_arousal = max(0.0, (next_arousal * 0.96) - 0.01)
            next_intensity = max(0.0, (next_intensity * 0.95) - 0.01)

        if label != "neutral" and target_intensity >= 0.08:
            trigger = label
        elif next_intensity < 0.10:
            trigger = ""
        else:
            trigger = prev_label if prev_label != "neutral" else ""

        if label != "neutral" and target_intensity >= 0.65:
            cooldown_until_ts = to_local_iso(now + 180.0 + (target_intensity * 120.0), default="")
        elif next_intensity <= 0.15:
            cooldown_until_ts = ""
        else:
            cooldown_until_ts = to_local_iso(cooldown_until, default="") if cooldown_until > now else ""

        next_state = {
            "schema_version": 1,
            "mood": "neutral",
            "valence": float(self._clamp(next_valence, -1.0, 1.0)),
            "arousal": float(self._clamp01(next_arousal)),
            "intensity": float(self._clamp01(next_intensity)),
            "trigger": str(trigger or ""),
            "last_update_ts": now_local_ts(),
            "cooldown_until_ts": cooldown_until_ts,
        }
        next_state["mood"] = self._derive_mood_from_emotion_state(
            emotion_state=next_state,
            signals=signals,
            traits=traits,
            context=context,
            mood_hints=mood_hints,
        )
        state["mood"] = str(next_state.get("mood") or "neutral")
        state["emotion"] = str(next_state.get("trigger") or label or "neutral")
        return next_state, {
            "label": label,
            "alpha": float(alpha),
            "target": {
                "valence": float(target_valence),
                "arousal": float(target_arousal),
                "intensity": float(target_intensity),
            },
            "previous": {
                "mood": str(current.get("mood") or "neutral"),
                "valence": float(prev_valence),
                "arousal": float(prev_arousal),
                "intensity": float(prev_intensity),
                "trigger": str(prev_label or ""),
            },
            "next": dict(next_state),
            "mood_hints": [str(x).strip().lower() for x in list(mood_hints or []) if str(x).strip()],
        }

    def _derive_mood_from_emotion_state(
        self,
        *,
        emotion_state: dict[str, Any],
        signals,
        traits: dict[str, Any],
        context: dict[str, Any] | None = None,
        mood_hints: list[str] | None = None,
    ) -> str:
        row = self._coerce_emotion_state(emotion_state)
        ctx = dict(context or {})
        trigger = normalize_emotion(row.get("trigger"))
        valence = self._clamp(self._to_float(row.get("valence"), 0.0), -1.0, 1.0)
        arousal = self._clamp01(self._to_float(row.get("arousal"), 0.0))
        intensity = self._clamp01(self._to_float(row.get("intensity"), 0.0))
        intent = str(ctx.get("intent") or getattr(signals, "intent", "") or "").strip().lower()
        mode = self._normalize_active_mode(str(ctx.get("mode") or getattr(signals, "mode", "") or "chatting"))
        tags = {str(x).strip().lower() for x in list(ctx.get("tags") or getattr(signals, "tags", []) or []) if str(x).strip()}
        hints = {self._normalize_runtime_mood(x) for x in list(mood_hints or []) if str(x).strip()}
        hints.discard("")

        focused_bias = 0.0
        if intent in {"task", "bug_report", "code_review", "planning", "question"}:
            focused_bias += 0.18
        if mode in {"engineer", "debugger", "planner"}:
            focused_bias += 0.12
        if {"has_traceback", "has_stacktrace", "has_logs", "has_code"} & tags:
            focused_bias += 0.14
        if "focused" in hints:
            focused_bias += 0.18

        supportive_bias = 0.0
        if "soft_supportive" in hints:
            supportive_bias += 0.18
        if "thoughtful" in hints:
            supportive_bias += 0.08

        teasing_bias = 0.0
        if {"teasing", "playful"} & hints:
            teasing_bias += 0.18
        if self._trait_scalar(traits, "playfulness") >= 0.52:
            teasing_bias += 0.08

        ironic_bias = 0.0
        if "ironic" in hints:
            ironic_bias += 0.14
        if self._trait_scalar(traits, "sarcasm") >= 0.32:
            ironic_bias += 0.06

        if intensity < 0.16 and abs(valence) < 0.14 and arousal < 0.24:
            if focused_bias >= 0.28:
                return "focused"
            if teasing_bias >= 0.24 and intent == "chat":
                return "teasing"
            if supportive_bias >= 0.24:
                return "soft_supportive"
            return "neutral"
        if trigger in {"angry", "frustrated"}:
            if intensity >= 0.55 or arousal >= 0.65 or focused_bias >= 0.28:
                return "focused"
            if ironic_bias >= 0.16 and intensity <= 0.45:
                return "ironic"
            return "thoughtful"
        if trigger == "anxious":
            if arousal >= 0.56 and focused_bias >= 0.20:
                return "focused"
            return "thoughtful"
        if trigger in {"sad", "tired"}:
            if intensity >= 0.22 or valence <= -0.18:
                return "soft_supportive"
            return "thoughtful"
        if valence >= 0.28:
            if arousal >= 0.45 and (teasing_bias >= 0.18 or intent == "chat"):
                return "teasing"
            if intensity >= 0.24:
                return "thoughtful"
            return "neutral"
        if valence <= -0.30:
            if arousal >= 0.58 and focused_bias >= 0.12:
                return "focused"
            if arousal <= 0.34:
                return "soft_supportive"
            return "thoughtful"
        if focused_bias >= 0.28:
            return "focused"
        if supportive_bias >= 0.22 and valence < 0.0:
            return "soft_supportive"
        if teasing_bias >= 0.24 and intent == "chat":
            return "teasing"
        if intensity >= 0.20:
            return "thoughtful"
        return "neutral"

    def _coerce_emotion_state(self, value: dict[str, Any] | None) -> dict[str, Any]:
        row = dict(value or {})
        mood = self._normalize_runtime_mood(row.get("mood"))
        return {
            "schema_version": 1,
            "mood": mood or "neutral",
            "valence": float(self._clamp(self._to_float(row.get("valence"), 0.0), -1.0, 1.0)),
            "arousal": float(self._clamp01(self._to_float(row.get("arousal"), 0.0))),
            "intensity": float(self._clamp01(self._to_float(row.get("intensity"), 0.0))),
            "trigger": str(normalize_emotion(row.get("trigger")) if row.get("trigger") else ""),
            "last_update_ts": to_local_iso(row.get("last_update_ts"), default=""),
            "cooldown_until_ts": to_local_iso(row.get("cooldown_until_ts"), default=""),
        }

    @staticmethod
    def _normalize_runtime_mood(value: Any) -> str:
        mood = str(value or "").strip().lower()
        if not mood:
            return ""
        if mood == "romantic_soft":
            return "soft_supportive"
        return mood

    def _build_blend(self, old_id: str, new_id: str, active: bool) -> dict[str, Any]:
        """Построить blend personality."""
        if not active or old_id == new_id:
            return {
                "active": False,
                "from": old_id,
                "to": new_id,
                "step": 0,
                "steps": self.blend_steps,
                "old_weight": 0.0,
                "new_weight": 1.0,
            }
        return {
            "active": True,
            "from": old_id,
            "to": new_id,
            "step": 1,
            "steps": self.blend_steps,
            "old_weight": round(max(0.0, 1.0 - 1.0 / self.blend_steps), 2),
            "new_weight": round(min(1.0, 1.0 / self.blend_steps), 2),
        }

    # -------------------------------------------------------------------------
    # TRAITS & EVOLUTION (РёР· character_engine)
    # -------------------------------------------------------------------------

    def get_traits(self, character_id: str | None = None) -> dict[str, Any]:
        """Получить все traits персонажа (merged builtin + learned)."""
        cid = self._validate_character(character_id)
        builtin = dict(self.storage.load_builtin_traits(cid))
        learned = dict(self.storage.load_learned_traits(cid))
        merged: dict[str, Any] = {}

        for name, row in builtin.items():
            key = self._normalize_trait_name(name)
            if not key:
                continue
            payload = self._normalize_trait(row, trait_name=key)
            payload["_source"] = "builtin"
            merged[key] = payload

        for name, row in learned.items():
            key = self._normalize_trait_name(name)
            if not key:
                continue
            payload = self._normalize_trait(row, trait_name=key)
            existing = dict(merged.get(key) or {})
            existing.update(payload)
            existing["_source"] = "learned"
            merged[key] = existing

        return {k: self._clean_trait(v) for k, v in merged.items()}

    def list_traits(self, character_id: str | None = None) -> dict[str, Any]:
        """Compatibility alias for command handlers expecting list_traits()."""
        return self.get_traits(character_id)

    def set_trait(
        self,
        character_id: str,
        trait_name: str,
        *,
        value: Any,
        confidence: float = 0.8,
        trait_type: str | None = None,
    ) -> dict[str, Any]:
        """Установить значение trait."""
        cid = self._validate_character(character_id)
        builtin, _, merged = self._load_traits(cid)
        name = self._normalize_trait_name(trait_name)
        if not name:
            raise ValueError("trait name is empty")
        now = time.time()
        prev_row = dict(merged.get(name) or {})
        prev_value = prev_row.get("value")

        row = dict(merged.get(name) or {})
        ttype = str(trait_type or row.get("type") or "scalar").strip().lower()
        row["type"] = "flag" if ttype == "flag" else "scalar"
        row.setdefault("min", 0.0)
        row.setdefault("max", 1.0)
        row["confidence"] = self._clamp01(confidence)
        row["updated_at"] = now_local_iso()
        row["last_used_ts"] = now_local_ts()
        row["disabled"] = False

        if row["type"] == "flag":
            row["value"] = bool(value)
        else:
            row["value"] = clamp_trait_scalar(
                name,
                value,
                minimum=self._to_float(row.get("min"), 0.0),
                maximum=self._to_float(row.get("max"), 1.0),
            )
        merged[name] = row

        learned_delta = self._extract_learned_delta(builtin, merged)
        self.storage.save_learned_traits(cid, learned_delta)
        state = self.storage.load_state(cid)
        state["last_update_ts"] = now_local_ts()
        self._refresh_active_lists(state=state, traits=merged)
        self.storage.save_state(cid, state)
        self.storage.append_event(cid, {
            "type": "trait_set",
            "trait": name,
            "value": row.get("value"),
            "confidence": row.get("confidence"),
            "source": "manual",
        })
        with self._lock:
            before_state = self._debug_state_snapshot()
            self._append_last_action(
                {
                    "type": "CHARACTER_TRAIT_SET",
                    "character_id": cid,
                    "trait": name,
                    "old_value": prev_value,
                    "new_value": row.get("value"),
                    "_state_before": before_state,
                }
            )
            self._touch_action(f"trait_set:{name}")
            self._sync_flat_with_global(prefer_global=False)
        self._autosave()
        return self._clean_trait(row)

    def remove_trait(self, character_id: str, trait_name: str) -> bool:
        """Удалить trait."""
        cid = self._validate_character(character_id)
        builtin, _, merged = self._load_traits(cid)
        name = self._normalize_trait_name(trait_name)
        if not name or name not in merged:
            return False
        now = time.time()
        prev_value = dict(merged.get(name) or {}).get("value")

        if name in builtin:
            row = dict(merged.get(name) or {})
            row["disabled"] = True
            if str(row.get("type") or "scalar").lower() == "flag":
                row["value"] = False
            else:
                row["value"] = self._to_float(row.get("min"), 0.0)
            row["updated_at"] = now_local_iso()
            merged[name] = row
        else:
            merged.pop(name, None)

        learned_delta = self._extract_learned_delta(builtin, merged)
        self.storage.save_learned_traits(cid, learned_delta)
        state = self.storage.load_state(cid)
        state["last_update_ts"] = now_local_ts()
        self._refresh_active_lists(state=state, traits=merged)
        self.storage.save_state(cid, state)
        self.storage.append_event(cid, {"type": "trait_remove", "trait": name, "source": "manual"})
        with self._lock:
            before_state = self._debug_state_snapshot()
            self._append_last_action(
                {
                    "type": "CHARACTER_TRAIT_REMOVE",
                    "character_id": cid,
                    "trait": name,
                    "old_value": prev_value,
                    "new_value": None,
                    "_state_before": before_state,
                }
            )
            self._touch_action(f"trait_remove:{name}")
            self._sync_flat_with_global(prefer_global=False)
        self._autosave()
        return True

    def evolve(
        self,
        *,
        text: str,
        meta: dict[str, Any] | None = None,
        active_character_id: str | None = None,
    ) -> CharacterRuntimeResult:
        """
        Обновить состояние персонажа на основе взаимодействия.

        Применяет правила эволюции, decay, conflict resolution.
        """
        meta_map = dict(meta or {})
        cid = self._validate_character(active_character_id or self.get_active_character_id(meta_map))
        character = self.storage.load_character(cid)
        state = self.storage.load_state(cid)
        builtin, _, merged = self._load_traits(cid)
        rules_payload = self.storage.load_rules(cid)
        existing_entry = self._get_character_persona_entry(cid)
        turn_id = self._coerce_turn_id(meta_map.get("turn_id"))
        conversation_id = str(
            meta_map.get("conversation_id")
            or self._state.get("conversation_id")
            or ""
        ).strip()
        if self._is_duplicate_evolve_turn(
            existing_entry,
            turn_id=turn_id,
            conversation_id=conversation_id,
        ):
            return self._build_evolve_result_from_current(
                character_id=cid,
                character=character,
                state=state,
                traits=merged,
                meta=meta_map,
                existing_entry=existing_entry,
                changes=[],
            )

        now = time.time()
        changes: list[dict[str, Any]] = []
        emotion_state = self.storage.load_emotion_state(cid)
        emotional_state_before = dict(emotion_state or {})
        signals = build_character_signals(
            text=str(text or ""),
            metadata={
                "lang": meta_map.get("lang"),
                "intent": meta_map.get("intent"),
                "emotion": meta_map.get("emotion"),
                "mood": meta_map.get("mood"),
                "emotion_profile": meta_map.get("emotion_profile"),
                "emotion_intensity": meta_map.get("emotion_intensity"),
                "emotion_arousal": meta_map.get("emotion_arousal"),
                "mode": meta_map.get("active_mode") or meta_map.get("mode") or self._state.get("active_mode"),
                "topic": meta_map.get("topic"),
                "topics": meta_map.get("topics"),
                "tags": meta_map.get("tags"),
            },
        )
        last_signals = signals.to_dict()

        # Apply decay
        self._apply_decay(merged, now=now, changes=changes)

        # Build context and apply rules
        ctx = self._build_context(text=text, meta=meta_map, state=state)
        mood_hints: list[str] = []
        applied_rules: list[dict[str, Any]] = []
        for index, rule in enumerate(list(rules_payload.get("rules") or []), start=1):
            if not isinstance(rule, dict):
                continue
            when = dict(rule.get("when") or {})
            if not self.evaluator.matches(when, ctx=ctx, traits=merged):
                continue
            rule_name = str(
                rule.get("id")
                or rule.get("name")
                or rule.get("label")
                or f"rule_{index}"
            ).strip() or f"rule_{index}"
            applied_rules.append(
                {
                    "rule": rule_name,
                    "action_count": len([x for x in list(rule.get("apply") or []) if isinstance(x, dict)]),
                }
            )
            self._apply_actions(
                merged=merged,
                state=state,
                actions=list(rule.get("apply") or []),
                now=now,
                changes=changes,
                mood_hints=mood_hints,
            )

        prev_mood = str(state.get("mood") or "").strip().lower()
        emotion_state, emotion_debug = self.update_emotional_state(
            character_id=cid,
            state=state,
            signals=signals,
            traits=merged,
            context=ctx,
            mood_hints=mood_hints,
            now_ts=now,
            current_state=emotion_state,
        )
        next_mood = str(state.get("mood") or "").strip().lower()
        if prev_mood != next_mood:
            changes.append({"kind": "mood", "from": prev_mood, "to": next_mood, "source": "emotion_state"})
        self.storage.save_emotion_state(cid, emotion_state)
        self._resolve_conflicts(merged=merged, state=state, now=now, changes=changes)
        self._refresh_active_lists(state=state, traits=merged)
        self._apply_cleanup(
            builtin=builtin,
            merged=merged,
            state=state,
            rules=rules_payload,
            now=now,
            changes=changes,
        )
        self._refresh_active_lists(state=state, traits=merged)
        state["last_update_ts"] = now_local_ts()
        if not str(state.get("mood") or "").strip():
            state["mood"] = str(character.get("default_mood") or "thoughtful")

        learned_delta = self._extract_learned_delta(builtin, merged)
        self.storage.save_learned_traits(cid, learned_delta)
        self.storage.save_state(cid, state)

        composed = self.composer.compose(
            storage=self.storage,
            character_id=cid,
            character=character,
            state=state,
            traits=merged,
            dialog_mode=dict(meta_map.get("dialog_mode") or {}),
            context_meta={
                "intent": str(meta_map.get("intent") or ""),
                "is_technical": bool(meta_map.get("is_technical", False)),
                "active_mode": str(meta_map.get("active_mode") or self._state.get("active_mode") or "chatting"),
            },
            persona_snapshot=dict(
                meta_map.get("persona_snapshot")
                or state.get("persona_snapshot")
                or self._state.get("persona_snapshot")
                or {}
            ),
            identity_core_snapshot=dict(
                meta_map.get("identity_core_snapshot")
                or meta_map.get("identity_core")
                or state.get("identity_core")
                or self._state.get("identity_core")
                or {}
            ),
        )
        trait_values = self._flat_trait_values(merged)
        persona_entry = self._build_persona_entry(
            character_id=cid,
            mood=str(state.get("mood") or composed.mood or "neutral"),
            trait_values=trait_values,
            existing=existing_entry,
        )
        user_addressing = self.storage.load_user_addressing(cid)
        feedback_items = [str(x).strip() for x in list(signals.user_feedback or []) if str(x).strip()]
        persona_feedback_items = [self._normalize_feedback_token(x) for x in feedback_items if not self._is_user_addressing_feedback(x)]
        user_addressing, addressing_debug = self._apply_user_addressing_feedback(
            current=user_addressing,
            feedback_items=feedback_items,
        )
        if persona_feedback_items:
            learning_signals = {
                "user_feedback": list(persona_feedback_items),
            }
            learned_persona, learner_debug = update_persona(
                dict(persona_entry.get("persona") or {}),
                learning_signals,
                max_delta_per_turn=0.02,
                decay_to_baseline=0.0,
            )
        else:
            learned_persona = dict(persona_entry.get("persona") or {})
            learner_debug = {
                "decay_deltas": {},
                "implicit_deltas": {},
                "feedback_deltas": {},
                "feedback_applied": [],
                "lock_changes": [],
                "ban_changes": [],
                "transient_persona_learning_enabled": False,
                "transient_signals_ignored": {},
                "max_delta_per_turn": 0.02,
                "decay_to_baseline": 0.0,
                "baseline_size": len(
                    dict(
                        learned_persona.get("baseline_traits")
                        or (dict(learned_persona.get("learned") or {}).get("baseline_traits") or {})
                    )
                ),
                "skipped_reason": "no_explicit_feedback",
            }
        persona_feedback_applied = [str(x).strip() for x in list(persona_feedback_items) if str(x).strip()]
        persona_entry["persona"] = learned_persona
        local_ctx = dict(persona_entry.get("local_context") or {})
        topic_weights = dict(local_ctx.get("last_topic_weights") or {})
        next_topic_weights: dict[str, float] = {}
        for key, value in topic_weights.items():
            topic = str(key or "").strip().lower()
            if not topic:
                continue
            decayed = self._clamp01(self._to_float(value, 0.0) * 0.94)
            if decayed >= 0.01:
                next_topic_weights[topic] = decayed
        for topic in list(signals.topics or []):
            tkey = str(topic or "").strip().lower()
            if not tkey:
                continue
            next_topic_weights[tkey] = self._clamp01(self._to_float(next_topic_weights.get(tkey), 0.0) + 0.15)
        local_ctx["last_topic_weights"] = next_topic_weights
        local_ctx["last_seen_ts"] = now_local_ts()
        if turn_id > 0:
            local_ctx["last_update_turn_id"] = int(turn_id)
        if conversation_id:
            local_ctx["last_update_conversation_id"] = conversation_id
        persona_entry["local_context"] = local_ctx
        state["last_signals"] = dict(last_signals)
        state["applied_rules"] = list(applied_rules)
        state["emotional_state_before"] = dict(emotional_state_before)
        state["emotional_state_after"] = dict(emotion_state)
        state["persona_feedback_applied"] = list(persona_feedback_applied)
        self.storage.save_state(cid, state)
        
        # Emotion: prefer emotion_profile (memory), fallback to emotion (NLU)
        emotion_profile = meta_map.get("emotion_profile") or {}
        emotion_detector_payload = {
            "emotion": str(
                emotion_profile.get("primary") or
                emotion_profile.get("label") or
                meta_map.get("emotion") or
                meta_map.get("mood") or
                ""
            ).strip().lower(),
            "emotion_intensity": self._clamp01(self._to_float(
                emotion_profile.get("intensity") or meta_map.get("emotion_intensity"), 0.0
            )),
            "emotion_arousal": self._clamp01(self._to_float(
                emotion_profile.get("arousal") or meta_map.get("emotion_arousal"), 0.0
            )),
            "source": "metadata",
        }
        emotional_state_delta = self._build_state_delta(
            before=emotional_state_before,
            after=emotion_state,
            keys=("mood", "trigger", "valence", "arousal", "intensity"),
        )
        persona_feedback_delta = {
            "applied": list(persona_feedback_applied),
            "feedback_deltas": dict(learner_debug.get("feedback_deltas") or {}),
            "lock_changes": list(learner_debug.get("lock_changes") or []),
            "ban_changes": list(learner_debug.get("ban_changes") or []),
            "skipped_reason": str(learner_debug.get("skipped_reason") or ""),
        }

        if feedback_items:
            changes.append(
                {
                    "kind": "user_feedback",
                    "feedback": feedback_items[:8],
                    "trait_deltas": dict(learner_debug.get("feedback_deltas") or {}),
                }
            )
            if addressing_debug.get("applied"):
                changes.append(
                    {
                        "kind": "user_addressing",
                        "feedback": list(addressing_debug.get("applied") or [])[:8],
                        "state": dict(user_addressing),
                    }
                )
            self.dispatch_action(
                {
                    "type": "FEEDBACK_RECEIVED",
                    "character_id": cid,
                    "feedback": feedback_items[:12],
                }
            )
        active_mode = self._normalize_active_mode(meta_map.get("active_mode") or self._state.get("active_mode"))
        persona_prompt_payload = self._build_compiler_persona_payload(
            character_id=cid,
            persona_state=dict(persona_entry.get("persona") or {}),
            mood=str(state.get("mood") or "neutral"),
            emotional_state=emotion_state,
        )
        compiled_prompt, _ = compile_system_persona(
            character_id=cid,
            persona_state=persona_prompt_payload,
            active_mode=active_mode,
            user_addressing=user_addressing,
        )
        self._set_character_persona_entry(cid, persona_entry)
        self.storage.save_user_addressing(cid, user_addressing)
        self.storage.append_event(cid, {
            "type": "character_update",
            "intent": ctx.get("intent"),
            "emotion": ctx.get("emotion"),
            "mode": ctx.get("mode"),
            "mood": state.get("mood"),
            "emotion_detector": emotion_detector_payload,
            "signals": dict(last_signals),
            "character_signals": dict(last_signals),
            "last_signals": dict(last_signals),
            "applied_rules": list(applied_rules),
            "emotional_state_before": dict(emotional_state_before),
            "emotional_state_after": dict(emotion_state),
            "emotional_state_delta": emotional_state_delta,
            "persona_feedback_applied": list(persona_feedback_applied),
            "persona_feedback_delta": persona_feedback_delta,
            "emotion_debug": emotion_debug,
            "learner_delta": learner_debug,
            "user_addressing": user_addressing,
            "user_addressing_debug": addressing_debug,
            "changes": changes[:16],
        })

        llm_profile = str(character.get("performance") or character.get("llm_profile") or "BALANCED").strip().upper() or "BALANCED"

        return CharacterRuntimeResult(
            character_id=cid,
            mood=str(composed.mood or state.get("mood") or "thoughtful"),
            llm_profile=llm_profile,
            traits={k: self._clean_trait(v) for k, v in merged.items()},
            trait_values=trait_values,
            state=dict(state),
            user_addressing=dict(user_addressing),
            prompt_block=compiled_prompt or composed.prompt,
            used_prompt_files=list(composed.used_files),
            changes=changes,
            effective_traits=dict(composed.effective_traits or {}),
            style_coefficients=dict(composed.style_coefficients or {}),
        )

    def _get_character_persona_entry(self, character_id: str) -> dict[str, Any]:
        self._ensure_characters_state()
        characters = dict(self._state.get("characters") or {})
        cid = str(character_id).strip().lower() or "asya"
        hit = dict(characters.get(cid) or {})
        if hit:
            return hit
        persona_seed = self.storage.load_persona_state(cid)
        return {
            "persona": {
                "traits": dict(persona_seed.get("traits") or {}),
                "mood": str(persona_seed.get("mood") or "neutral").strip().lower() or "neutral",
                "locks": dict(persona_seed.get("locks") or {"feminine": True, "informal_you": True}),
                "bans": [str(x).strip() for x in list(persona_seed.get("bans") or []) if str(x).strip()],
                "learned": dict(persona_seed.get("learned") or {}),
            },
            "local_context": {"last_topic_weights": {}, "last_seen_ts": now_local_ts()},
            "character_id": cid,
        }

    def _coerce_turn_id(self, value: Any) -> int:
        try:
            out = int(value)
        except Exception:
            out = 0
        return max(0, out)

    def _is_duplicate_evolve_turn(
        self,
        existing_entry: dict[str, Any] | None,
        *,
        turn_id: int,
        conversation_id: str,
    ) -> bool:
        if int(turn_id or 0) <= 0:
            return False
        row = dict(existing_entry or {})
        local_ctx = dict(row.get("local_context") or {})
        last_turn = self._coerce_turn_id(local_ctx.get("last_update_turn_id"))
        if last_turn <= 0 or last_turn != int(turn_id):
            return False
        last_conversation = str(local_ctx.get("last_update_conversation_id") or "").strip()
        current_conversation = str(conversation_id or "").strip()
        if last_conversation and current_conversation and last_conversation != current_conversation:
            return False
        return True

    def _build_evolve_result_from_current(
        self,
        *,
        character_id: str,
        character: dict[str, Any],
        state: dict[str, Any],
        traits: dict[str, Any],
        meta: dict[str, Any],
        existing_entry: dict[str, Any] | None = None,
        changes: list[dict[str, Any]] | None = None,
    ) -> CharacterRuntimeResult:
        cid = self._validate_character(character_id)
        row = dict(existing_entry or self._get_character_persona_entry(cid))
        persona = dict(row.get("persona") or {})
        active_mode = self._normalize_active_mode(meta.get("active_mode") or self._state.get("active_mode"))
        composed = self.composer.compose(
            storage=self.storage,
            character_id=cid,
            character=character,
            state=state,
            traits=traits,
            dialog_mode=dict(meta.get("dialog_mode") or {}),
            context_meta={
                "intent": str(meta.get("intent") or meta.get("meta", {}).get("intent_label") or ""),
                "is_technical": self._coerce_bool(meta.get("is_technical"), default=False),
                "active_mode": active_mode,
            },
            persona_snapshot=dict(
                meta.get("persona_snapshot")
                or state.get("persona_snapshot")
                or self._state.get("persona_snapshot")
                or {}
            ),
            identity_core_snapshot=dict(
                meta.get("identity_core_snapshot")
                or meta.get("identity_core")
                or state.get("identity_core")
                or self._state.get("identity_core")
                or {}
            ),
        )
        trait_values = self._flat_trait_values(traits)
        mood = str(state.get("mood") or composed.mood or persona.get("mood") or character.get("default_mood") or "neutral").strip().lower() or "neutral"
        persona_prompt_payload = self._build_compiler_persona_payload(
            character_id=cid,
            persona_state=dict(persona or self.storage.load_persona_state(cid) or {}),
            mood=mood,
        )
        user_addressing = self.storage.load_user_addressing(cid)
        prompt_block, _ = compile_system_persona(
            character_id=cid,
            persona_state=persona_prompt_payload,
            active_mode=active_mode,
            user_addressing=user_addressing,
        )
        llm_profile = str(character.get("performance") or character.get("llm_profile") or "BALANCED").strip().upper() or "BALANCED"
        return CharacterRuntimeResult(
            character_id=cid,
            mood=mood,
            llm_profile=llm_profile,
            traits={k: self._clean_trait(v) for k, v in dict(traits or {}).items()},
            trait_values=trait_values,
            state=dict(state or {}),
            user_addressing=dict(user_addressing),
            prompt_block=prompt_block or composed.prompt,
            used_prompt_files=list(composed.used_files),
            changes=[dict(x) for x in list(changes or []) if isinstance(x, dict)],
            effective_traits=dict(composed.effective_traits or {}),
            style_coefficients=dict(composed.style_coefficients or {}),
        )

    def _set_character_persona_entry(self, character_id: str, entry: dict[str, Any]) -> None:
        cid = str(character_id or "").strip().lower() or "asya"
        with self._lock:
            before_state = self._debug_state_snapshot()
            self._ensure_characters_state()
            characters = dict(self._state.get("characters") or {})
            payload = dict(entry or {})
            persona = dict(payload.get("persona") or {})
            characters[cid] = payload
            self._state["characters"] = characters
            self._sync_flat_with_global(prefer_global=False)
            self._touch_action(f"persona_state:{cid}")
            self._append_last_action(
                {
                    "type": "CHARACTER_PERSONA_UPDATED",
                    "character_id": cid,
                    "_state_before": before_state,
                }
            )
        try:
            self.storage.save_persona_state(cid, persona)
        except Exception:
            pass

    def _build_persona_entry(
        self,
        *,
        character_id: str,
        mood: str,
        trait_values: dict[str, Any],
        existing: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        row = dict(existing or {})
        persona = dict(row.get("persona") or {})
        local = dict(row.get("local_context") or {})
        base_traits = dict(persona.get("traits") or {})
        next_traits = dict(base_traits)
        seed_runtime_traits = not bool(base_traits)
        for raw_key, raw_value in dict(trait_values or {}).items():
            key = self._normalize_trait_name(raw_key)
            if not key or isinstance(raw_value, bool):
                continue
            try:
                source = float(clamp_trait_scalar(key, raw_value, minimum=0.0, maximum=1.0))
            except Exception:
                continue
            if seed_runtime_traits and key not in next_traits:
                # Only bootstrap an empty persona once. Existing persona traits
                # must not absorb transient runtime reactions on each turn.
                next_traits[key] = source
        if seed_runtime_traits and "teasing" not in next_traits and "playfulness" in trait_values:
            next_traits["teasing"] = float(clamp_trait_scalar("teasing", trait_values.get("playfulness"), minimum=0.0, maximum=1.0))
        if seed_runtime_traits and "empathy" not in next_traits and "thoughtfulness" in trait_values:
            next_traits["empathy"] = float(clamp_trait_scalar("empathy", trait_values.get("thoughtfulness"), minimum=0.0, maximum=1.0))
        persona["traits"] = dict(clamp_trait_map(next_traits))
        persona["relation_state"] = self._coerce_relation_state_payload(
            persona.get("relation_state"),
            traits=persona["traits"],
        )
        if "mood" not in persona:
            persona["mood"] = str(mood or "neutral").strip().lower() or "neutral"
        locks = dict(persona.get("locks") or {})
        locks.setdefault("feminine", True)
        locks.setdefault("informal_you", True)
        persona["locks"] = locks
        persona["bans"] = [str(x).strip() for x in list(persona.get("bans") or []) if str(x).strip()]
        learned = dict(persona.get("learned") or {})
        learned.setdefault("preferences_confirmed", [])
        learned.setdefault("preferences_pending", [])
        learned.setdefault("style_bias", {})
        persona["learned"] = learned
        persona["baseline_traits"] = dict(clamp_trait_map(persona.get("baseline_traits") or {}))
        local.setdefault("last_topic_weights", {})
        local["last_seen_ts"] = now_local_ts()
        return {"persona": persona, "local_context": local, "character_id": str(character_id or "")}

    def _build_compiler_persona_payload(
        self,
        *,
        character_id: str,
        persona_state: dict[str, Any] | None,
        mood: str = "",
        emotional_state: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        cid = self._validate_character(character_id)
        payload = dict(self.storage.load_persona_state(cid) or {})
        payload.update(dict(persona_state or {}))
        traits = dict(clamp_trait_map(payload.get("traits")))
        payload["traits"] = traits
        payload["relation_state"] = self._coerce_relation_state_payload(
            payload.get("relation_state"),
            traits=traits,
        )
        next_mood = str(
            mood
            or payload.get("mood")
            or dict(payload.get("emotional_state") or {}).get("mood")
            or "neutral"
        ).strip().lower() or "neutral"
        if next_mood == "romantic_soft":
            next_mood = "soft_supportive"
        payload["mood"] = next_mood
        emotion_row = dict(emotional_state or payload.get("emotional_state") or self.storage.load_emotion_state(cid) or {})
        emotion_row.setdefault("mood", next_mood)
        if str(emotion_row.get("mood") or "").strip().lower() == "romantic_soft":
            emotion_row["mood"] = "soft_supportive"
        payload["emotional_state"] = emotion_row
        return payload

    def _coerce_relation_state_payload(
        self,
        value: dict[str, Any] | None,
        *,
        traits: dict[str, Any] | None = None,
    ) -> dict[str, float]:
        row = dict(traits or {})
        warmth = float(clamp_trait_scalar("warmth", row.get("warmth", 0.58), minimum=0.0, maximum=1.0))
        empathy = float(clamp_trait_scalar("empathy", row.get("empathy", 0.62), minimum=0.0, maximum=1.0))
        teasing_seed = row.get("teasing", row.get("playfulness", 0.35))
        teasing = float(clamp_trait_scalar("teasing", teasing_seed, minimum=0.0, maximum=1.0))
        defaults = {
            "familiarity": self._clamp01(0.28 + max(0.0, warmth - 0.5) * 0.12 + max(0.0, empathy - 0.5) * 0.08),
            "trust": self._clamp01(0.54 + max(0.0, empathy - 0.5) * 0.16),
            "teasing_permission": self._clamp01(0.08 + max(0.0, teasing - 0.3) * 0.55),
            "softness_bias": self._clamp01((warmth * 0.55) + (empathy * 0.45)),
        }
        current = dict(value or {})
        out = dict(defaults)
        for key in ("familiarity", "trust", "teasing_permission", "softness_bias"):
            if key not in current:
                continue
            out[key] = self._clamp01(self._to_float(current.get(key), out[key]))
        return {k: float(v) for k, v in out.items()}

    def _build_state_delta(
        self,
        *,
        before: dict[str, Any] | None,
        after: dict[str, Any] | None,
        keys: tuple[str, ...],
    ) -> dict[str, dict[str, Any]]:
        prev = dict(before or {})
        nxt = dict(after or {})
        out: dict[str, dict[str, Any]] = {}
        for key in keys:
            left = prev.get(key)
            right = nxt.get(key)
            if isinstance(left, (int, float)) or isinstance(right, (int, float)):
                lval = self._to_float(left, 0.0)
                rval = self._to_float(right, 0.0)
                if abs(rval - lval) <= 1e-9:
                    continue
                out[str(key)] = {"before": lval, "after": rval}
                continue
            if str(left or "") == str(right or ""):
                continue
            out[str(key)] = {"before": left, "after": right}
        return out

    def _apply_user_addressing_feedback(
        self,
        *,
        current: dict[str, Any] | None,
        feedback_items: list[str],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        before = self._coerce_user_addressing(current)
        updated = dict(before)
        applied: list[str] = []

        for raw in list(feedback_items or []):
            item = str(raw or "").strip()
            if not item:
                continue
            kind = self._feedback_kind(item)
            payload = self._feedback_payload(item)
            if kind == "set_canonical_name":
                name = self._normalize_user_name_form(payload)
                if not name:
                    continue
                updated["canonical_name"] = name
                updated["allowed_forms"] = self._merge_name_forms(updated.get("allowed_forms"), [name])
                updated["forbidden_forms"] = [
                    x for x in list(updated.get("forbidden_forms") or [])
                    if str(x).casefold() != name.casefold()
                ]
                applied.append(f"set_canonical_name:{name}")
                continue
            if kind == "allow_name_form":
                name = self._normalize_user_name_form(payload)
                if not name:
                    continue
                updated["allowed_forms"] = self._merge_name_forms(updated.get("allowed_forms"), [name])
                updated["forbidden_forms"] = [
                    x for x in list(updated.get("forbidden_forms") or [])
                    if str(x).casefold() != name.casefold()
                ]
                applied.append(f"allow_name_form:{name}")
                continue
            if kind == "forbid_name_form":
                name = self._normalize_user_name_form(payload)
                canonical = str(updated.get("canonical_name") or "").strip()
                if not name or (canonical and name.casefold() == canonical.casefold()):
                    continue
                updated["forbidden_forms"] = self._merge_name_forms(updated.get("forbidden_forms"), [name])
                updated["allowed_forms"] = [
                    x for x in list(updated.get("allowed_forms") or [])
                    if str(x).casefold() != name.casefold()
                ]
                applied.append(f"forbid_name_form:{name}")
                continue
            if kind == "disable_diminutives":
                updated["allow_diminutives"] = False
                applied.append("disable_diminutives")
                continue
            if kind == "enable_diminutives":
                updated["allow_diminutives"] = True
                applied.append("enable_diminutives")

        normalized = self._coerce_user_addressing(updated)
        if normalized != before:
            normalized["updated_at"] = now_local_iso()
        return normalized, {
            "applied": applied,
            "before": before,
            "after": normalized,
        }

    @staticmethod
    def _is_user_addressing_feedback(value: str) -> bool:
        return CharacterRuntime._feedback_kind(value) in {
            "set_canonical_name",
            "allow_name_form",
            "forbid_name_form",
            "disable_diminutives",
            "enable_diminutives",
        }

    @staticmethod
    def _feedback_kind(value: str) -> str:
        text = str(value or "").strip()
        if not text:
            return ""
        if ":" not in text:
            return text.lower()
        kind, _ = text.split(":", 1)
        return kind.strip().lower()

    @staticmethod
    def _feedback_payload(value: str) -> str:
        text = str(value or "").strip()
        if ":" not in text:
            return ""
        _, payload = text.split(":", 1)
        return payload.strip()

    @staticmethod
    def _normalize_feedback_token(value: str) -> str:
        text = str(value or "").strip()
        if not text:
            return ""
        if ":" not in text:
            return text.lower()
        kind, payload = text.split(":", 1)
        return f"{kind.strip().lower()}:{payload.strip().lower()}"

    @classmethod
    def _coerce_user_addressing(cls, value: dict[str, Any] | None) -> dict[str, Any]:
        row = dict(value or {})
        canonical = cls._normalize_user_name_form(row.get("canonical_name"))
        allowed = cls._merge_name_forms([], list(row.get("allowed_forms") or []))
        forbidden = cls._merge_name_forms([], list(row.get("forbidden_forms") or []))
        if canonical:
            allowed = cls._merge_name_forms(allowed, [canonical])
            forbidden = [x for x in forbidden if str(x).casefold() != canonical.casefold()]
        return {
            "canonical_name": canonical,
            "allowed_forms": allowed,
            "forbidden_forms": forbidden,
            "allow_diminutives": bool(row.get("allow_diminutives", False)),
            "use_name_by_default": bool(row.get("use_name_by_default", False)),
            "updated_at": str(row.get("updated_at") or "").strip(),
        }

    @staticmethod
    def _normalize_user_name_form(value: Any) -> str:
        text = str(value or "").strip()
        text = text.strip(" \t\r\n.,!?;:()[]{}\"'`«»")
        text = " ".join(text.split())
        if not text or " " in text:
            return ""
        if len(text) < 2 or len(text) > 40:
            return ""
        if any(ch.isdigit() for ch in text):
            return ""
        return text

    @classmethod
    def _merge_name_forms(cls, current: Any, extra: list[Any]) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()
        for row in [*list(current or []), *list(extra or [])]:
            item = cls._normalize_user_name_form(row)
            key = item.casefold()
            if not item or key in seen:
                continue
            seen.add(key)
            out.append(item)
        return out

    # -------------------------------------------------------------------------
    # PROMPT BUILDER (РёР· prompt_builder)
    # -------------------------------------------------------------------------

    def build_personality_block(self, character_id: str | None = None) -> str:
        """
        Построить prompt block для персонажа.

        Это основной метод для интеграции с Brain/PromptBuilder.
        """
        cid = self._validate_character(character_id)
        state = self.storage.load_state(cid)
        _, _, merged = self._load_traits(cid)
        trait_values = self._flat_trait_values(merged)
        persona_entry = self._build_persona_entry(
            character_id=cid,
            mood=str(state.get("mood") or "neutral"),
            trait_values=trait_values,
            existing=self._get_character_persona_entry(cid),
        )
        self._set_character_persona_entry(cid, persona_entry)
        text, _ = compile_system_persona(
            character_id=cid,
            persona_state=self._build_compiler_persona_payload(
                character_id=cid,
                persona_state=dict(persona_entry.get("persona") or {}),
                mood=str(state.get("mood") or "neutral"),
            ),
            active_mode=self.get_active_mode(),
            user_addressing=self.storage.load_user_addressing(cid),
        )
        return text

    @classmethod
    def build(
        cls,
        state,
        user_msg: str,
        retrieved_memories,
        traits,
        policies,
    ) -> PromptPack:
        runtime = cls(autosave=False)
        runtime.patch(_as_dict(state))
        return runtime.build_prompt(
            user_msg=user_msg,
            retrieved_memories=retrieved_memories,
            traits=traits,
            policies=policies,
        )

    def build_prompt(
        self,
        user_msg: str,
        *,
        retrieved_memories: list[dict[str, Any]] | None = None,
        traits: dict[str, Any] | None = None,
        policies: dict[str, Any] | None = None,
        budgets: PromptBudgets | None = None,
    ) -> PromptPack:
        """
        Построить полный prompt.
        """
        state_map = dict(self._state)
        traits_map = _as_dict(traits)
        policies_map = _coerce_policies(policies)
        budgets = budgets or PromptBudgets.from_policies(policies_map)

        context_tags = self._extract_context_tags(state_map, traits_map, policies_map)
        memories_block, selected_memories, dropped_memories = self._build_memories_block(
            retrieved_memories,
            budgets,
        )
        tail_block, conversation_tail, dropped_tail = self._build_conversation_tail_block(
            state_map,
            budgets,
        )
        long_summary_block = self._build_long_summary_block(state_map, budgets, dropped_tail=dropped_tail)

        blocks = {
            "system_role": self._build_system_role_block(policies_map),
            "persona": self._build_persona_block(state_map, traits_map, policies_map),
            "state_summary": self._build_state_summary_block(state_map),
            "context_tags": self._build_context_tags_block(context_tags),
            "retrieved_memories": memories_block,
            "long_summary": long_summary_block,
            "conversation_tail": tail_block,
            "user_message": _normalize_text(user_msg),
            "output_schema": self._build_output_schema_block(policies_map),
        }
        blocks, cut_info = self._apply_block_budgets(blocks, budgets)
        blocks = self._enforce_total_budget(blocks, budgets, cut_info)

        from llm.tokenizer import estimate_tokens

        full_prompt = self._render_blocks(blocks)
        if estimate_tokens(full_prompt) > budgets.total_tokens:
            full_prompt, _ = _clip_to_tokens(full_prompt, budgets.total_tokens)
            cut_info["full_prompt_hard_clip"] = True
        system_prompt = self._render_blocks(blocks, system=True)
        user_message = blocks.get("user_message", "")
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]

        token_usage = {name: estimate_tokens(text) for name, text in blocks.items()}
        token_usage["system_prompt"] = estimate_tokens(system_prompt)
        token_usage["full_prompt"] = estimate_tokens(full_prompt)

        cut_info["dropped_memories"] = dropped_memories
        cut_info["dropped_tail_messages"] = dropped_tail

        return PromptPack(
            system_prompt=system_prompt,
            user_message=user_message,
            full_prompt=full_prompt,
            messages=messages,
            blocks=blocks,
            token_usage=token_usage,
            budgets=budgets.as_dict(),
            cut_info=cut_info,
            context_tags=context_tags,
            selected_memories=selected_memories,
            conversation_tail=conversation_tail,
        )

    def _extract_context_tags(
        self,
        state: dict[str, Any],
        traits: dict[str, Any],
        policies: dict[str, Any],
    ) -> dict[str, str]:
        """������ ����������� ����."""
        state_tags = _as_dict(state.get("context_tags"))
        policy_tags = _as_dict(policies.get("context_tags"))
        trait_tags = _as_dict(traits.get("context_tags"))

        def _pick(*values) -> str:
            for value in values:
                text = _normalize_text(value)
                if text:
                    return text
            return ""

        tags = {
            "lang": _pick(
                state_tags.get("lang"),
                state.get("lang"),
                state.get("language"),
                policy_tags.get("lang"),
                trait_tags.get("lang"),
            ),
            "mood": _pick(
                state_tags.get("mood"),
                state.get("mood"),
                policy_tags.get("mood"),
                trait_tags.get("mood"),
            ),
            "intent": _pick(
                state_tags.get("intent"),
                state.get("intent"),
                policy_tags.get("intent"),
                trait_tags.get("intent"),
            ),
            "active_mode": _pick(
                state_tags.get("active_mode"),
                state.get("active_mode"),
                state.get("mode"),
                policy_tags.get("active_mode"),
                trait_tags.get("active_mode"),
            ),
            "topic": _pick(
                state_tags.get("topic"),
                state.get("topic"),
                state.get("current_topic"),
                policy_tags.get("topic"),
                trait_tags.get("topic"),
            ),
            "user_greeting": _pick(
                state_tags.get("user_greeting"),
                state.get("user_greeting"),
                policy_tags.get("user_greeting"),
                trait_tags.get("user_greeting"),
            ),
            "allow_greeting": _pick(
                state_tags.get("allow_greeting"),
                state.get("allow_greeting"),
                policy_tags.get("allow_greeting"),
                trait_tags.get("allow_greeting"),
            ),
            "new_session": _pick(
                state_tags.get("new_session"),
                state.get("new_session"),
                policy_tags.get("new_session"),
                trait_tags.get("new_session"),
            ),
            "greeted_today": _pick(
                state_tags.get("greeted_today"),
                state.get("greeted_today"),
                policy_tags.get("greeted_today"),
                trait_tags.get("greeted_today"),
            ),
            "local_date": _pick(
                state_tags.get("local_date"),
                state.get("local_date"),
                policy_tags.get("local_date"),
                trait_tags.get("local_date"),
            ),
            "is_technical": _pick(
                state_tags.get("is_technical"),
                state.get("is_technical"),
                policy_tags.get("is_technical"),
                trait_tags.get("is_technical"),
            ),
            "allowed_term": _pick(
                state_tags.get("allowed_term"),
                state.get("allowed_term"),
                policy_tags.get("allowed_term"),
                trait_tags.get("allowed_term"),
            ),
            "use_term_now": _pick(
                state_tags.get("use_term_now"),
                state.get("use_term_now"),
                policy_tags.get("use_term_now"),
                trait_tags.get("use_term_now"),
            ),
        }
        return {k: v for k, v in tags.items() if v}

    def _build_system_role_block(self, policies: dict[str, Any]) -> str:
        """Построить блок system role."""
        lines = [
            "You are MMis assistant.",
            "Follow system rules and active policies strict.",
            "Prefer concise, clear, and actionable replies.",
            "If context is insufficient, ask a clarifying question.",
        ]
        for key in ("system_role", "role", "global_rule"):
            value = _normalize_text(policies.get(key))
            if value:
                lines.append(value)

        for key in ("rules", "policy_rules", "constraints"):
            for item in _as_list(policies.get(key)):
                value = _normalize_text(item)
                if value:
                    lines.append(value)

        unique: list[str] = []
        seen: set[str] = set()
        for line in lines:
            low = line.lower()
            if low in seen:
                continue
            seen.add(low)
            unique.append(line)
        return "\n".join(f"- {line}" for line in unique)

    def _build_persona_block(
        self,
        state: dict[str, Any],
        traits: dict[str, Any],
        policies: dict[str, Any],
    ) -> str:
        """Построить блок persona."""
        character_prompt = _normalize_text(state.get("character_prompt_block"))
        if character_prompt:
            return character_prompt

        character = _normalize_text(
            state.get("active_character_id")
            or state.get("character")
            or traits.get("character")
            or policies.get("character")
            or "default"
        )
        characters = dict(state.get("characters") or {})
        entry = dict(characters.get(character) or {})
        persona_state = dict(entry.get("persona") or {})
        persona_snapshot = _as_dict(state.get("persona_snapshot"))

        trait_map = dict(persona_state.get("traits") or {})
        snapshot_traits = _as_dict(persona_snapshot.get("stable_traits"))
        for raw_key, raw_value in dict(snapshot_traits or {}).items():
            key = self._normalize_trait_name(raw_key)
            if not key or isinstance(raw_value, bool):
                continue
            try:
                value = self._clamp01(float(raw_value))
            except Exception:
                continue
            trait_map[key] = value
        for raw_key, raw_value in dict(traits or {}).items():
            key = self._normalize_trait_name(raw_key)
            if not key or isinstance(raw_value, bool):
                continue
            try:
                value = self._clamp01(float(raw_value))
            except Exception:
                continue
            trait_map[key] = value
        if "teasing" not in trait_map and "playfulness" in traits:
            trait_map["teasing"] = self._clamp01(self._to_float(traits.get("playfulness"), 0.35))
        if "empathy" not in trait_map and "thoughtfulness" in traits:
            trait_map["empathy"] = self._clamp01(self._to_float(traits.get("thoughtfulness"), 0.55))

        if not trait_map:
            local_cid = self._validate_character(character)
            _, _, merged_traits = self._load_traits(local_cid)
            flat_traits = self._flat_trait_values(merged_traits)
            for raw_key, raw_value in dict(flat_traits or {}).items():
                key = self._normalize_trait_name(raw_key)
                if not key or isinstance(raw_value, bool):
                    continue
                try:
                    value = self._clamp01(float(raw_value))
                except Exception:
                    continue
                trait_map[key] = value
            if "teasing" not in trait_map and "playfulness" in flat_traits:
                trait_map["teasing"] = self._clamp01(self._to_float(flat_traits.get("playfulness"), 0.35))
            if "empathy" not in trait_map and "thoughtfulness" in flat_traits:
                trait_map["empathy"] = self._clamp01(self._to_float(flat_traits.get("thoughtfulness"), 0.55))

        persona_payload = dict(persona_state)
        persona_payload["traits"] = trait_map
        snapshot_relation_state = _as_dict(persona_snapshot.get("relation_state"))
        if snapshot_relation_state:
            persona_payload["relation_state"] = dict(snapshot_relation_state)
        snapshot_boundaries = _as_dict(persona_snapshot.get("boundaries"))
        if snapshot_boundaries:
            merged_boundaries = dict(persona_payload.get("boundaries") or {})
            for raw_key, raw_value in dict(snapshot_boundaries or {}).items():
                key = str(raw_key or "").strip().lower()
                if not key:
                    continue
                merged_boundaries[key] = bool(raw_value)
            persona_payload["boundaries"] = merged_boundaries
        snapshot_emotional_handling = _as_dict(persona_snapshot.get("emotional_handling"))
        if snapshot_emotional_handling:
            merged_emotional_handling = dict(persona_payload.get("emotional_handling") or {})
            for raw_key, raw_value in dict(snapshot_emotional_handling or {}).items():
                key = str(raw_key or "").strip().lower()
                if not key:
                    continue
                if key in {"deescalate_on_irritation", "treat_short_replies_as_low_bandwidth"}:
                    merged_emotional_handling[key] = bool(raw_value)
                else:
                    try:
                        merged_emotional_handling[key] = self._clamp01(float(raw_value))
                    except Exception:
                        continue
            persona_payload["emotional_handling"] = merged_emotional_handling
        snapshot_mood = str(persona_snapshot.get("mood") or "").strip().lower()
        persona_payload = self._build_compiler_persona_payload(
            character_id=character,
            persona_state=persona_payload,
            mood=str(
                snapshot_mood
                or persona_payload.get("mood")
                or state.get("mood")
                or traits.get("mood")
                or "neutral"
            ).strip().lower(),
        )
        locks = dict(persona_payload.get("locks") or {})
        locks.setdefault("feminine", True)
        locks.setdefault("informal_you", True)
        persona_payload["locks"] = locks
        persona_payload["bans"] = [str(x).strip() for x in list(persona_payload.get("bans") or []) if str(x).strip()]
        mode = self._normalize_active_mode(
            persona_snapshot.get("active_mode")
            or state.get("active_mode")
            or state.get("mode")
            or "chatting"
        )
        user_addressing = dict(self.storage.load_user_addressing(character) or {})
        snapshot_user_addressing = _as_dict(persona_snapshot.get("user_addressing"))
        if snapshot_user_addressing:
            for key in ("canonical_name", "use_name_by_default", "allow_diminutives"):
                if key in snapshot_user_addressing:
                    user_addressing[key] = snapshot_user_addressing.get(key)
            for key in ("allowed_forms", "forbidden_forms"):
                values = [str(x).strip() for x in list(snapshot_user_addressing.get(key) or []) if str(x).strip()]
                if values:
                    user_addressing[key] = values
        text, _ = compile_system_persona(
            character_id=character,
            persona_state=persona_payload,
            active_mode=mode,
            user_addressing=user_addressing,
        )
        return _normalize_text(text)

    def _build_state_summary_block(self, state: dict[str, Any]) -> str:
        """
        Построить блок state summary.
        
        Теперь включает always-on memory state — рабочую память личности,
        которая всегда доступна модели (не через retrieval).
        """
        mode = _normalize_text(state.get("active_mode") or state.get("mode")) or "chatting"
        task = (
            _normalize_text(state.get("current_task"))
            or _normalize_text(state.get("task"))
            or "none"
        )
        summary = (
            _normalize_text(state.get("dialog_summary"))
            or _normalize_text(state.get("summary"))
            or "none"
        )
        focus = _normalize_text(state.get("focus") or state.get("active_goal") or "")
        
        # Always-on memory state: рабочая память личности
        # Это не retrieval, а внутреннее состояние, которое всегда с моделью
        identity_core = self._build_identity_core_block(state)
        open_questions = self._build_open_questions_block(state)
        current_decisions = self._build_current_decisions_block(state)
        recent_topics = self._build_recent_topics_block(state)
        relation_state = self._build_relation_state_block(state)
        
        lines = [
            f"- mode: {mode}",
            f"- mode_lock: {str(bool(state.get('mode_lock', False))).lower()}",
            f"- current_task: {task}",
        ]
        
        # Добавляем identity_core только если есть данные
        if identity_core and identity_core != "- none":
            lines.append(f"- identity_core: {identity_core}")
        
        # Добавляем open_questions только если есть активные вопросы
        if open_questions and open_questions != "- none":
            lines.append(f"- open_questions: {open_questions}")
        
        # Добавляем current_decisions только если есть активные решения
        if current_decisions and current_decisions != "- none":
            lines.append(f"- current_decisions: {current_decisions}")
        
        # Добавляем recent_topics только если есть история тем
        if recent_topics and recent_topics != "- none":
            lines.append(f"- recent_topics: {recent_topics}")
        
        # Добавляем relation_state только если есть данные
        if relation_state and relation_state != "- none":
            lines.append(f"- relation_state: {relation_state}")
        
        lines.append(f"- dialog_summary: {summary}")
        
        personality = _normalize_text(state.get("active_personality_id") or state.get("personality") or "default")
        lines.append(f"- active_personality: {personality}")
        character = _normalize_text(state.get("active_character_id") or state.get("character") or personality)
        lines.append(f"- active_character: {character}")
        blend = _as_dict(state.get("personality_blend"))
        if blend and bool(blend.get("active")):
            lines.append(
                f"- personality_blend: from={blend.get('from')} to={blend.get('to')} "
                f"step={blend.get('step')}/{blend.get('steps')}"
            )
        if focus:
            lines.append(f"- focus: {focus}")
        return "\n".join(lines)
    
    def _build_identity_core_block(self, state: dict[str, Any]) -> str:
        """
        Построить блок identity core — ключевые факты о пользователе и отношениях.
        
        Это всегда в prompt и является частью 'самосознания' модели:
        - canonical_name: как пользователь представился
        - preferred_addressing: предпочтительные формы обращения
        - key_boundaries: установленные границы (не сюсюкать, не повторять вопросы)
        - interaction_preferences: предпочтения в общении
        """
        addressing = _as_dict(state.get("addressing") or state.get("user_addressing") or {})
        if not addressing:
            return "- none"
        
        parts = []
        canonical_name = addressing.get("canonical_name")
        if canonical_name:
            parts.append(f"name={canonical_name}")
        
        allowed_forms = addressing.get("allowed_forms")
        if allowed_forms:
            if isinstance(allowed_forms, list):
                parts.append(f"address_as={','.join(allowed_forms[:3])}")
            else:
                parts.append(f"address_as={allowed_forms}")
        
        forbidden_forms = addressing.get("forbidden_forms")
        if forbidden_forms:
            if isinstance(forbidden_forms, list):
                parts.append(f"do_not_call={','.join(forbidden_forms[:3])}")
            else:
                parts.append(f"do_not_call={forbidden_forms}")
        
        boundaries = _as_dict(state.get("boundaries") or {})
        boundary_flags = []
        if boundaries.get("avoid_baby_tone"):
            boundary_flags.append("no_baby_talk")
        if boundaries.get("avoid_repeating_question"):
            boundary_flags.append("no_repeat_questions")
        if boundaries.get("avoid_inventing_user_facts"):
            boundary_flags.append("no_invent_facts")
        if boundary_flags:
            parts.append(f"boundaries={','.join(boundary_flags)}")
        
        interaction = _as_dict(state.get("interaction_style") or {})
        if interaction.get("prefers_directness"):
            parts.append("direct_style")
        if interaction.get("prefers_short_answers"):
            parts.append("short_answers")
        if interaction.get("allows_light_teasing"):
            parts.append("allows_teasing")
        
        if not parts:
            return "- none"
        
        return " ".join(parts)
    
    def _build_open_questions_block(self, state: dict[str, Any]) -> str:
        """
        Построить блок unresolved items — открытые вопросы и незавершённые задачи.
        
        Помогает модели помнить о том, что ещё требует ответа или завершения.
        """
        open_questions = state.get("open_questions")
        if not open_questions:
            # Пробуем альтернативные ключи
            open_questions = state.get("unresolved_questions")
        if not open_questions:
            return "- none"
        
        if isinstance(open_questions, list):
            # Фильтруем только недавние и релевантные (максимум 3)
            recent = [q for q in open_questions[:5] if q]
            if not recent:
                return "- none"
            return "; ".join(str(q) for q in recent[:3])
        elif isinstance(open_questions, str):
            return open_questions[:200]
        
        return "- none"
    
    def _build_current_decisions_block(self, state: dict[str, Any]) -> str:
        """
        Построить блок current_decisions — текущие решения и договорённости.
        
        Помогает модели помнить о принятых решениях и планах.
        """
        decisions = state.get("current_decisions")
        if not decisions:
            decisions = state.get("active_decisions")
        if not decisions:
            return "- none"
        
        if isinstance(decisions, list):
            recent = [d for d in decisions[:5] if d]
            if not recent:
                return "- none"
            return "; ".join(str(d) for d in recent[:3])
        elif isinstance(decisions, str):
            return decisions[:200]
        
        return "- none"
    
    def _build_recent_topics_block(self, state: dict[str, Any]) -> str:
        """
        Построить блок recent_topics — последние обсуждавшиеся темы.
        
        Помогает модели поддерживать контекст беседы.
        """
        topics = state.get("recent_topics")
        if not topics:
            # Пробуем извлечь из context_tags
            context_tags = _as_dict(state.get("context_tags") or {})
            topic = context_tags.get("topic")
            if topic:
                return topic
            return "- none"
        
        if isinstance(topics, list):
            recent = [t for t in topics[:5] if t]
            if not recent:
                return "- none"
            return "; ".join(str(t) for t in recent[:3])
        elif isinstance(topics, str):
            return topics[:200]
        
        return "- none"
    
    def _build_relation_state_block(self, state: dict[str, Any]) -> str:
        """
        Построить блок relation_state — состояние отношений с пользователем.
        
        Включает:
        - rapport_level: уровень доверия/близости
        - last_interaction_tone: тон последнего взаимодействия
        - user_mood_pattern: паттерн настроения пользователя
        """
        relation = _as_dict(state.get("relation_state") or {})
        if not relation:
            # Пробуем извлечь из user_addressing
            user_addressing = _as_dict(state.get("user_addressing") or {})
            if user_addressing.get("rapport_level"):
                relation["rapport_level"] = user_addressing["rapport_level"]
            if user_addressing.get("last_interaction_tone"):
                relation["last_interaction_tone"] = user_addressing["last_interaction_tone"]
        
        if not relation:
            return "- none"
        
        parts = []
        rapport = relation.get("rapport_level")
        if rapport:
            parts.append(f"rapport={rapport}")
        
        tone = relation.get("last_interaction_tone")
        if tone:
            parts.append(f"last_tone={tone}")
        
        mood_pattern = relation.get("user_mood_pattern")
        if mood_pattern:
            parts.append(f"mood_pattern={mood_pattern}")
        
        if not parts:
            return "- none"
        
        return " ".join(parts)

    def _build_context_tags_block(self, tags: dict[str, str]) -> str:
        """Построить блок context tags."""
        if not tags:
            return "- none"
        return "\n".join(f"- {k}: {v}" for k, v in tags.items())

    def _build_memories_block(
        self,
        retrieved_memories,
        budgets: PromptBudgets,
    ) -> tuple[str, list[dict[str, Any]], int]:
        """Построить блок memories."""
        from llm.tokenizer import estimate_tokens

        raw_items = []
        for item in _as_list(retrieved_memories):
            row = _coerce_memory(item)
            if not row["text"] or not row["relevant"]:
                continue
            raw_items.append(row)

        raw_items.sort(
            key=lambda x: (x["priority"], x["confidence"], x["score"], x["recency_ts"], len(x["text"])),
            reverse=True,
        )

        selected: list[dict[str, Any]] = []
        lines: list[str] = []
        used_tokens = 0
        for row in raw_items:
            memory_text, _ = _clip_to_tokens(row["text"], budgets.memory_item_tokens)
            parts = [f"conf={row['confidence']:.2f}"]
            if row["source"]:
                parts.append(f"src={row['source']}")
            if row["topic"]:
                parts.append(f"topic={row['topic']}")
            line = f"- ({', '.join(parts)}) {memory_text}"
            line_tokens = estimate_tokens(line)
            if used_tokens + line_tokens > budgets.memory_tokens:
                continue
            used_tokens += line_tokens
            lines.append(line)
            selected.append(row)

        if not lines:
            return "- none", [], len(raw_items)
        return "\n".join(lines), selected, max(0, len(raw_items) - len(selected))

    def _build_conversation_tail_block(
        self,
        state: dict[str, Any],
        budgets: PromptBudgets,
    ) -> tuple[str, list[dict[str, str]], int]:
        """Построить блок conversation tail."""
        from llm.tokenizer import estimate_tokens

        source = (
            state.get("conversation_tail")
            or state.get("history")
            or state.get("messages")
            or []
        )
        parsed = [_coerce_turn(item) for item in _as_list(source)]
        parsed = [x for x in parsed if x["content"]]
        if not parsed:
            return "- none", [], 0

        tail = parsed[-budgets.tail_turns :]
        selected_rev: list[dict[str, str]] = []
        used_tokens = 0
        for row in reversed(tail):
            clipped, _ = _clip_to_tokens(row["content"], budgets.tail_turn_tokens)
            line = f"- {row['role']}: {clipped}"
            line_tokens = estimate_tokens(line)
            if used_tokens + line_tokens > budgets.tail_tokens:
                continue
            used_tokens += line_tokens
            selected_rev.append({"role": row["role"], "content": clipped})

        selected = list(reversed(selected_rev))
        if not selected:
            return "- none", [], len(tail)
        block = "\n".join(f"- {x['role']}: {x['content']}" for x in selected)
        return block, selected, max(0, len(tail) - len(selected))

    def _build_long_summary_block(
        self,
        state: dict[str, Any],
        budgets: PromptBudgets,
        *,
        dropped_tail: int,
    ) -> str:
        """Построить блок long summary."""
        explicit = sanitize_session_summary_text(
            state.get("long_summary")
            or state.get("rolling_summary")
            or state.get("dialog_summary")
            or "",
            max_chars=max(64, int(budgets.long_summary_tokens * 6)),
        )
        if explicit:
            clipped, _ = _clip_to_tokens(explicit, budgets.long_summary_tokens)
            return clipped

        history = [_coerce_turn(item) for item in _as_list(state.get("history"))]
        history = [x for x in history if x["content"]]
        if dropped_tail <= 0 or len(history) <= budgets.tail_turns:
            return "- none"

        older = history[: max(0, len(history) - budgets.tail_turns)]
        if not older:
            return "- none"
        meaningful: list[dict[str, str]] = []
        for row in older:
            content = _normalize_text(row["content"])
            if not is_meaningful_summary_turn(content):
                continue
            meaningful.append({"role": row["role"], "content": content})
            if len(meaningful) >= 4:
                break
        if not meaningful:
            return "- none"

        parts: list[str] = []
        for row in meaningful:
            role = str(row.get("role") or "").strip().lower()
            content = str(row.get("content") or "").strip()
            if not content:
                continue
            if role == "assistant":
                parts.append(f"Assistant replied: {content}")
            elif role == "tool":
                parts.append(f"Tool result: {content}")
            else:
                parts.append(f"User said: {content}")
        if not parts:
            return "- none"

        text = "Earlier context: " + " ".join(parts)
        if len(older) > len(meaningful):
            text += f" ({len(older) - len(meaningful)} more earlier turns compressed.)"
        clipped, _ = _clip_to_tokens(text, budgets.long_summary_tokens)
        return clipped if clipped else "- none"

    def _build_output_schema_block(self, policies: dict[str, Any]) -> str:
        """Построить блок output schema."""
        for key in ("output_schema", "response_schema", "tool_schema"):
            value = policies.get(key)
            if isinstance(value, (dict, list)):
                return _normalize_text(json.dumps(value, ensure_ascii=False, indent=2))
            text = _normalize_text(value)
            if text:
                return text
        return "Return plain text by default. For tool invocation return strict JSON object."

    def _apply_block_budgets(
        self,
        blocks: dict[str, str],
        budgets: PromptBudgets,
    ) -> tuple[dict[str, str], dict[str, Any]]:
        """Применить бюджеты к блокам."""
        from llm.tokenizer import estimate_tokens

        limits = {
            "system_role": budgets.system_tokens,
            "persona": budgets.persona_tokens,
            "state_summary": budgets.state_tokens,
            "context_tags": budgets.tags_tokens,
            "retrieved_memories": budgets.memory_tokens,
            "long_summary": budgets.long_summary_tokens,
            "conversation_tail": budgets.tail_tokens,
            "user_message": budgets.user_tokens,
            "output_schema": budgets.output_schema_tokens,
        }
        out: dict[str, str] = {}
        cut_info: dict[str, Any] = {}
        for key, value in blocks.items():
            clipped, was_cut = _clip_to_tokens(_normalize_text(value), limits.get(key, 240))
            out[key] = clipped
            if was_cut:
                cut_info[f"{key}_trimmed"] = True
        return out, cut_info

    def _enforce_total_budget(
        self,
        blocks: dict[str, str],
        budgets: PromptBudgets,
        cut_info: dict[str, Any],
    ) -> dict[str, str]:
        """Обеспечить общий бюджет."""
        out = dict(blocks)
        while self._full_token_count(out) > budgets.total_tokens:
            changed = False
            for key in ("conversation_tail", "long_summary", "state_summary", "persona", "context_tags", "retrieved_memories"):
                current = out.get(key, "")
                if not current or current == "- none":
                    continue
                shrunk = _shrink_block(current, key)
                if shrunk == current:
                    continue
                out[key] = shrunk
                cut_info[f"{key}_trimmed_total"] = True
                changed = True
                break
            if not changed:
                break
        return out

    def _render_blocks(self, blocks: dict[str, str], system: bool = False) -> str:
        """Рендер блоков."""
        order = (
            "system_role",
            "persona",
            "state_summary",
            "context_tags",
            "retrieved_memories",
            "long_summary",
            "conversation_tail",
            "output_schema" if system else "user_message",
        )
        parts: list[str] = []
        for key in order:
            value = _normalize_text(blocks.get(key))
            if not value:
                continue
            parts.append(f"<<<{key.upper()}>>>\n{value}\n<<<END_{key.upper()}>>>")
        return "\n\n".join(parts).strip()

    def _full_token_count(self, blocks: dict[str, str]) -> int:
        """Получить общее количество токенов."""
        from llm.tokenizer import estimate_tokens
        return estimate_tokens(self._render_blocks(blocks))

    # -------------------------------------------------------------------------
    # INTERNAL HELPERS
    # -------------------------------------------------------------------------

    def _validate_character(self, character_id: str | None = None) -> str:
        """Валидировать и получить ID персонажа."""
        cid = str(character_id or "").strip().lower()
        ids = set(self.list_ids())
        if cid in ids:
            return cid
        if ids:
            return sorted(ids)[0]
        return "asya"

    def _load_traits(self, character_id: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        """Загрузить builtin, learned и merged traits."""
        builtin = dict(self.storage.load_builtin_traits(character_id))
        learned = dict(self.storage.load_learned_traits(character_id))
        merged: dict[str, Any] = {}

        for name, row in builtin.items():
            key = self._normalize_trait_name(name)
            if not key:
                continue
            payload = self._normalize_trait(row, trait_name=key)
            payload["_source"] = "builtin"
            merged[key] = payload

        for name, row in learned.items():
            key = self._normalize_trait_name(name)
            if not key:
                continue
            payload = self._normalize_trait(row, trait_name=key)
            existing = dict(merged.get(key) or {})
            existing.update(payload)
            existing["_source"] = "learned"
            merged[key] = existing

        return builtin, learned, merged

    def _build_context(self, *, text: str, meta: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
        """Построить контекст для rule evaluator."""
        # Tags: prefer top-level tags, fallback to metadata_tags (legacy)
        tags = list(meta.get("tags") or meta.get("metadata_tags") or [])
        
        # Emotion: prefer emotion_profile (memory), fallback to emotion (NLU)
        emotion_profile = meta.get("emotion_profile") or {}
        emotion_val = (
            emotion_profile.get("primary") or
            emotion_profile.get("label") or
            meta.get("emotion") or
            meta.get("mood") or
            ""
        )
        emotion_intensity = (
            emotion_profile.get("intensity") or
            meta.get("emotion_intensity") or
            0.0
        )
        emotion_arousal = (
            emotion_profile.get("arousal") or
            meta.get("emotion_arousal") or
            0.0
        )
        
        return {
            "text": str(text or ""),
            "intent": str(meta.get("intent") or meta.get("meta", {}).get("intent_label") or "").strip().lower(),
            "emotion": str(emotion_val).strip().lower(),
            "emotion_intensity": self._clamp01(self._to_float(emotion_intensity, 0.0)),
            "emotion_arousal": self._clamp01(self._to_float(emotion_arousal, 0.0)),
            "mode": str(
                meta.get("active_mode")
                or meta.get("mode")
                or state.get("active_mode")
                or state.get("mode")
                or "chatting"
            ).strip().lower(),
            "topic": str(meta.get("topic") or "").strip().lower(),
            "tags": [str(x).strip().lower() for x in tags if str(x).strip()],
        }

    def _apply_actions(
        self,
        *,
        merged: dict[str, Any],
        state: dict[str, Any],
        actions: list[Any],
        now: float,
        changes: list[dict[str, Any]],
        mood_hints: list[str] | None = None,
    ) -> None:
        """Применить actions из rules."""
        for row in list(actions or []):
            if not isinstance(row, dict):
                continue
            if "set_mood" in row:
                mood = str(row.get("set_mood") or "").strip().lower()
                if mood:
                    if mood == "romantic_soft":
                        mood = "soft_supportive"
                    if isinstance(mood_hints, list) and mood not in mood_hints:
                        mood_hints.append(mood)
                    changes.append({"kind": "mood_hint", "hint": mood, "source": "rule"})
                continue

            trait_name = self._normalize_trait_name(row.get("trait"))
            if not trait_name:
                continue
            op = str(row.get("op") or "add").strip().lower()
            value = row.get("value", 0.0)
            ttype = str(row.get("type") or "").strip().lower()

            trait = dict(merged.get(trait_name) or {})
            if not trait:
                trait = self._normalize_trait({
                    "type": ("flag" if ttype == "flag" else "scalar"),
                    "value": (False if ttype == "flag" else 0.0),
                    "min": 0.0,
                    "max": 1.0,
                    "confidence": 0.45,
                }, trait_name=trait_name)
                trait["_source"] = "learned"

            trait_type = str(trait.get("type") or "scalar").strip().lower()
            before = trait.get("value")

            if op == "remove":
                trait["disabled"] = True
                trait["value"] = False if trait_type == "flag" else self._to_float(trait.get("min"), 0.0)
            elif trait_type == "flag":
                if op == "toggle":
                    trait["value"] = not bool(trait.get("value"))
                elif op in {"set", "add", "update"}:
                    trait["value"] = bool(value)
                elif op == "mul":
                    trait["value"] = bool(trait.get("value")) and bool(value)
                trait["disabled"] = False
            else:
                cur = self._to_float(trait.get("value"), 0.0)
                min_v = self._to_float(trait.get("min"), 0.0)
                max_v = self._to_float(trait.get("max"), 1.0)
                val = self._to_float(value, 0.0)
                if op in {"set", "update"}:
                    nxt = clamp_trait_scalar(trait_name, val, minimum=min_v, maximum=max_v)
                elif op == "mul":
                    nxt = clamp_trait_scalar(trait_name, cur * val, minimum=min_v, maximum=max_v)
                else:
                    nxt = apply_trait_delta(
                        trait_name,
                        cur,
                        val,
                        minimum=min_v,
                        maximum=max_v,
                        soften=True,
                    )
                trait["value"] = nxt
                trait["disabled"] = False

            trait["confidence"] = self._clamp01(self._to_float(trait.get("confidence"), 0.5) + 0.03)
            trait["updated_at"] = now_local_iso()
            trait["last_used_ts"] = now_local_ts()
            merged[trait_name] = trait

            if before != trait.get("value"):
                change_kind = self._trait_change_kind(trait_name)
                changes.append({
                    "kind": change_kind,
                    "trait": trait_name,
                    "op": op,
                    "from": before,
                    "to": trait.get("value"),
                    "source": "rule",
                })

    def _apply_decay(self, merged: dict[str, Any], *, now: float, changes: list[dict[str, Any]]) -> None:
        """Применить decay к scalar traits."""
        for name, row in list(merged.items()):
            trait = dict(row or {})
            if str(trait.get("type") or "scalar").strip().lower() != "scalar":
                continue
            decay = self._to_float(trait.get("decay_per_day"), 0.0)
            if decay <= 0:
                continue
            last_ts = self._to_float(trait.get("decay_ts"), self._to_float(trait.get("updated_at"), now))
            if last_ts <= 0:
                last_ts = now
            days = max(0.0, (now - last_ts) / 86400.0)
            if days < 0.02:
                continue
            cur = self._to_float(trait.get("value"), 0.0)
            min_v = self._to_float(trait.get("min"), 0.0)
            max_v = self._to_float(trait.get("max"), 1.0)
            nxt = self._clamp(cur - (decay * days), min_v, max_v)
            if abs(nxt - cur) < 1e-6:
                trait["decay_ts"] = now_local_ts()
                merged[name] = trait
                continue
            trait["value"] = nxt
            trait["decay_ts"] = now_local_ts()
            merged[name] = trait
            changes.append({"kind": "decay", "trait": name, "from": cur, "to": nxt})

    def _resolve_conflicts(self, *, merged: dict[str, Any], state: dict[str, Any], now: float, changes: list[dict[str, Any]]) -> None:
        """Разрешить конфликты между traits."""
        sarcasm = self._trait_scalar(merged, "sarcasm")
        romance = self._trait_scalar(merged, "romance")

        if sarcasm > 0.72 and romance > 0.68:
            row = dict(merged.get("sarcasm") or {})
            before = self._to_float(row.get("value"), sarcasm)
            row["value"] = self._clamp(before - 0.12, self._to_float(row.get("min"), 0.0), self._to_float(row.get("max"), 1.0))
            row["updated_at"] = now_local_iso()
            row["last_used_ts"] = now_local_ts()
            row["_source"] = "learned"
            merged["sarcasm"] = row
            changes.append({
                "kind": "conflict",
                "trait": "sarcasm",
                "from": before,
                "to": row.get("value"),
                "reason": "romance_vs_sarcasm",
            })

        mood = str(state.get("mood") or "").strip().lower()
        if mood == "focused":
            play = self._trait_scalar(merged, "playfulness")
            if play > 0.55:
                row = dict(merged.get("playfulness") or {})
                before = self._to_float(row.get("value"), play)
                row["value"] = self._clamp(before - 0.1, self._to_float(row.get("min"), 0.0), self._to_float(row.get("max"), 1.0))
                row["updated_at"] = now_local_iso()
                row["last_used_ts"] = now_local_ts()
                row["_source"] = "learned"
                merged["playfulness"] = row
                changes.append({
                    "kind": "conflict",
                    "trait": "playfulness",
                    "from": before,
                    "to": row.get("value"),
                    "reason": "focused_mode",
                })

    def _apply_cleanup(
        self,
        *,
        builtin: dict[str, Any],
        merged: dict[str, Any],
        state: dict[str, Any],
        rules: dict[str, Any],
        now: float,
        changes: list[dict[str, Any]],
    ) -> None:
        """Очистка старых/неактивных traits."""
        cleanup = dict(rules.get("cleanup") or {})
        conf_threshold = self._to_float(cleanup.get("remove_if_confidence_below"), 0.25)
        unused_days = max(1.0, self._to_float(cleanup.get("remove_if_unused_days"), 45.0))
        remove_after_sec = unused_days * 86400.0

        for name in list(merged.keys()):
            row = dict(merged.get(name) or {})
            is_builtin = name in builtin
            conf = self._to_float(row.get("confidence"), 0.0)
            last_ts = max(self._to_float(row.get("last_used_ts"), 0.0), self._to_float(row.get("updated_at"), 0.0))
            stale = (now - last_ts) > remove_after_sec if last_ts > 0 else False
            ttl_days = self._to_float(row.get("ttl_days"), 0.0)
            ttl_expired = False
            if ttl_days > 0 and last_ts > 0:
                ttl_expired = (now - last_ts) > (ttl_days * 86400.0)

            remove = False
            if conf < conf_threshold and not is_builtin:
                remove = True
            if stale and not is_builtin:
                remove = True
            if ttl_expired:
                remove = True

            if remove:
                merged.pop(name, None)
                changes.append({"kind": "cleanup", "trait": name, "reason": "confidence_or_ttl"})

        counters = dict(state.get("counters") or {})
        counters.setdefault("banter_hits", 0)
        counters.setdefault("comfort_hits", 0)
        state["counters"] = counters

    def _refresh_active_lists(self, *, state: dict[str, Any], traits: dict[str, Any]) -> None:
        """Обновить списки active/disabled traits."""
        active: list[str] = []
        disabled: list[str] = []

        for name, row in list(traits.items()):
            trait = dict(row or {})
            if bool(trait.get("disabled", False)):
                disabled.append(name)
                continue
            ttype = str(trait.get("type") or "scalar").strip().lower()
            value = trait.get("value")
            if ttype == "flag":
                if bool(value):
                    active.append(name)
            else:
                if self._to_float(value, 0.0) > 0.05:
                    active.append(name)

        state["active_traits"] = sorted(set(active))
        state["disabled_traits"] = sorted(set(disabled))

    def _extract_learned_delta(self, builtin: dict[str, Any], merged: dict[str, Any]) -> dict[str, Any]:
        """������ delta learned traits."""
        out: dict[str, Any] = {}
        for name, row in list(merged.items()):
            clean = self._clean_trait(row)
            base = dict(builtin.get(name) or {})
            if name not in builtin:
                out[name] = clean
                continue
            if self._has_override(base, clean):
                out[name] = clean
        return out

    def _flat_trait_values(self, traits: dict[str, Any]) -> dict[str, Any]:
        """Плоские значения traits."""
        out: dict[str, Any] = {}
        for name, row in dict(traits or {}).items():
            if str(name).startswith("_"):
                continue
            payload = dict(row or {})
            value = payload.get("value")
            out[str(name)] = bool(value) if str(payload.get("type") or "").lower() == "flag" else self._to_float(value, 0.0)
        return out

    # -------------------------------------------------------------------------
    # STATIC HELPERS
    # -------------------------------------------------------------------------

    @staticmethod
    def _normalize_trait_name(value: str) -> str:
        """Нормализовать имя trait."""
        raw = str(value or "").strip().lower()
        if not raw:
            return ""
        return "".join(ch for ch in raw if ch.isalnum() or ch in {"_", "-"})

    @staticmethod
    def _normalize_trait(value: dict[str, Any] | None, *, trait_name: str = "") -> dict[str, Any]:
        """Нормализовать trait."""
        return normalize_trait_record(trait_name, value)

    @staticmethod
    def _clean_trait(value: dict[str, Any]) -> dict[str, Any]:
        """Очистить trait от служебных полей."""
        row = dict(value or {})
        row.pop("_source", None)
        return row

    @staticmethod
    def _has_override(base: dict[str, Any], current: dict[str, Any]) -> bool:
        """Проверить есть ли override над base."""
        base_n = CharacterRuntime._normalize_trait(base)
        cur_n = CharacterRuntime._normalize_trait(current)
        for key in ("type", "value", "disabled", "prompt_file"):
            if base_n.get(key) != cur_n.get(key):
                return True
        if abs(CharacterRuntime._to_float(base_n.get("confidence"), 0.0) - CharacterRuntime._to_float(cur_n.get("confidence"), 0.0)) >= 0.08:
            return True
        return False

    @staticmethod
    def _trait_scalar(traits: dict[str, Any], name: str) -> float:
        """Получить scalar значение trait."""
        row = dict(traits.get(str(name).strip().lower()) or {})
        value = row.get("value")
        if isinstance(value, bool):
            return 1.0 if value else 0.0
        return CharacterRuntime._to_float(value, 0.0)

    @staticmethod
    def _to_float(value, default: float) -> float:
        """Конвертировать в float."""
        try:
            return float(value)
        except Exception:
            return float(default)

    @staticmethod
    def _clamp(value: float, minimum: float, maximum: float) -> float:
        """Clamp значения."""
        lo = float(min(minimum, maximum))
        hi = float(max(minimum, maximum))
        return max(lo, min(hi, float(value)))

    @staticmethod
    def _clamp01(value: float) -> float:
        """Clamp Рє [0, 1]."""
        return CharacterRuntime._clamp(float(value), 0.0, 1.0)

    @staticmethod
    def _normalize_mode(value: Any) -> str:
        """Нормализовать режим."""
        text = str(value or "chat").strip().lower()
        valid = {"chat", "task", "coding", "voice", "silent", "debug"}
        return text if text in valid else "chat"

    @staticmethod
    def _normalize_active_mode(value: Any) -> str:
        text = str(value or "").strip().lower()
        if text in {"voice", "silent"}:
            return "chatting"
        return normalize_mode_name(text, allow_custom=True)

    @staticmethod
    def _active_to_legacy_mode(value: Any) -> str:
        mode = CharacterRuntime._normalize_active_mode(value)
        mapping = {
            "chatting": "chat",
            "helper": "task",
            "engineer": "coding",
            "debugger": "debug",
            "planner": "task",
        }
        return mapping.get(mode, "chat")

    @staticmethod
    def _normalize_profile(value: Any) -> str:
        """Нормализовать профиль."""
        text = str(value or "BALANCED").strip().upper()
        valid = {"FAST", "BALANCED", "QUALITY", "ECONOM", "AUTONOMOUS", "ASYA"}
        return text if text in valid else "BALANCED"

    @staticmethod
    def _normalize_personality_id(value: Any) -> str:
        """Нормализовать ID личности."""
        text = str(value or "").strip().lower()
        if not text:
            return ""
        return "".join(ch for ch in text if ch.isalnum() or ch in {"_", "-"})

    @staticmethod
    def _new_conversation_id() -> str:
        """Создать новый ID разговора."""
        import uuid
        return str(uuid.uuid4())

    @staticmethod
    def _coerce_history(value) -> list[dict[str, str]]:
        """Привести историю к правильному формату."""
        if not value:
            return []
        if isinstance(value, list):
            return [dict(x) for x in value if isinstance(x, dict)]
        return []

    @staticmethod
    def _coerce_dict_list(value) -> list[dict[str, Any]]:
        """Привести список dict."""
        if not value:
            return []
        if isinstance(value, list):
            return [dict(x) for x in value if isinstance(x, dict)]
        return []

    @staticmethod
    def _coerce_string_dict(value) -> dict[str, str]:
        """Привести dict[str, str]."""
        if not value:
            return {}
        if isinstance(value, dict):
            return {str(k): str(v) for k, v in value.items()}
        return {}

    @staticmethod
    def _coerce_personality_blend(value) -> dict[str, Any]:
        """Привести personality blend."""
        default = {
            "active": False,
            "from": "default",
            "to": "default",
            "step": 0,
            "steps": 4,
            "old_weight": 0.0,
            "new_weight": 1.0,
        }
        if not value:
            return default
        if not isinstance(value, dict):
            return default
        out = dict(default)
        out.update({k: v for k, v in value.items() if k in default})
        out["active"] = bool(out.get("active", False))
        out["step"] = int(out.get("step", 0))
        out["steps"] = int(out.get("steps", 4))
        try:
            out["old_weight"] = float(out.get("old_weight", 0.0))
            out["new_weight"] = float(out.get("new_weight", 1.0))
        except Exception:
            pass
        return out

    @staticmethod
    def _coerce_cooldowns(value) -> dict[str, Any]:
        """Привести cooldowns."""
        default = {
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
        if not value:
            return default
        if not isinstance(value, dict):
            return default
        out = dict(default)
        out.update({k: v for k, v in value.items() if k in default})
        try:
            out["repeat_count"] = int(out.get("repeat_count", 0))
        except Exception:
            pass
        return out

    @staticmethod
    def _coerce_address_terms(value, *, conversation_id: str) -> dict[str, Any]:
        """Привести address terms."""
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
    def _trait_change_kind(trait_name: str) -> str:
        key = str(trait_name or "").strip().lower()
        if key in {"playfulness", "thoughtfulness"}:
            return "derived_axis"
        return "trait"


# =============================================================================
# MODULE HELPERS
# =============================================================================

def _local_today() -> str:
    return str(dialog_local_date(time.time()))


def _starts_with_greeting(text: str) -> bool:
    return bool(dialog_starts_with_greeting(str(text or "")))


def _normalize_text(value) -> str:
    """Нормализовать текст."""
    if value is None:
        return ""
    text = str(value).strip()
    for bom in ("\ufeff", "\ufffe", "ГЇВ»Вї"):
        text = text.replace(bom, "")
    import re
    text = re.sub(r"[\u200B-\u200F\u2060\uFEFF]", "", text)
    text = re.sub(r"[ \t\f\v]+", " ", text)
    return text


def _as_dict(value) -> dict[str, Any]:
    """Привести к dict."""
    if isinstance(value, dict):
        return dict(value)
    if value is None:
        return {}
    if isinstance(value, (list, tuple, set)):
        return {"items": list(value)}
    try:
        return dict(vars(value))
    except Exception:
        return {}


def _as_list(value) -> list[Any]:
    """Привести к list."""
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _coerce_policies(policies) -> dict[str, Any]:
    """Привести policies."""
    if isinstance(policies, dict):
        return dict(policies)
    if isinstance(policies, (list, tuple)):
        return {"rules": list(policies)}
    text = _normalize_text(policies)
    return {"rules": [text]} if text else {}


def _coerce_memory(item) -> dict[str, Any]:
    """Привести memory item."""
    if isinstance(item, dict):
        metadata = _as_dict(item.get("metadata"))
        score_raw = item.get("score") if item.get("score") is not None else metadata.get("score")
        if score_raw is None:
            score_raw = 0.0
        score = CharacterRuntime._clamp01(CharacterRuntime._to_float(score_raw, 0.0))
        confidence_raw = item.get("confidence") if item.get("confidence") is not None else metadata.get("confidence")
        priority_raw = item.get("priority") if item.get("priority") is not None else metadata.get("priority")
        confidence = CharacterRuntime._to_float(confidence_raw, -1.0)
        priority = CharacterRuntime._to_float(priority_raw, -1.0)
        if confidence < 0.0:
            confidence = score
        if priority < 0.0:
            priority = score
        ts_value = (
            item.get("ts")
            if item.get("ts") is not None
            else item.get("updated_at")
            if item.get("updated_at") is not None
            else metadata.get("ts")
            if metadata.get("ts") is not None
            else metadata.get("updated_at")
        )
        recency_ts = parse_time_to_epoch(
            ts_value,
            0.0,
        )
        return {
            "text": _normalize_text(item.get("text") or item.get("content") or ""),
            "source": _normalize_text(item.get("source") or ""),
            "topic": _normalize_text(item.get("topic") or ""),
            "score": score,
            "confidence": CharacterRuntime._clamp01(confidence),
            "priority": CharacterRuntime._clamp01(priority),
            "recency_ts": float(recency_ts),
            "relevant": bool(item.get("relevant", True)),
        }
    return {
        "text": _normalize_text(item),
        "source": "",
        "topic": "",
        "score": 0.5,
        "confidence": 0.5,
        "priority": 0.5,
        "recency_ts": 0.0,
        "relevant": True,
    }


def _coerce_turn(item) -> dict[str, str]:
    """Привести turn."""
    if isinstance(item, dict):
        return {
            "role": str(item.get("role") or "user").strip().lower(),
            "content": _normalize_text(item.get("content") or item.get("text") or ""),
        }
    return {"role": "user", "content": _normalize_text(item)}


def _to_int(value, default: int, minimum: int = 0) -> int:
    """Конвертировать в int."""
    try:
        return max(minimum, int(value))
    except Exception:
        return max(minimum, int(default))


def _clip_to_tokens(text: str, max_tokens: int) -> tuple[str, bool]:
    """Обрезать текст до max_tokens."""
    from llm.tokenizer import estimate_tokens

    if not text:
        return "", True
    if max_tokens <= 0:
        return "", True

    current_tokens = estimate_tokens(text)
    if current_tokens <= max_tokens:
        return text, False

    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi) // 2
        clipped = text[:mid]
        if estimate_tokens(clipped) <= max_tokens:
            lo = mid + 1
        else:
            hi = mid

    result = text[: max(0, lo - 1)]
    return result, True


def _shrink_block(text: str, block_type: str) -> str:
    """Сжать блок."""
    if not text or text == "- none":
        return text

    lines = text.split("\n")
    if len(lines) <= 2:
        if block_type == "conversation_tail":
            return "- none"
        if block_type == "retrieved_memories":
            return str(lines[0]).strip() if lines else "- none"
        return text

    # Remove every other line
    keep = [lines[0]]
    for i, line in enumerate(lines[1:], 1):
        if i % 2 == 0:
            keep.append(line)

    if len(keep) <= 2:
        if block_type == "retrieved_memories":
            return str(keep[0]).strip() if keep else "- none"
        return "- none"
    return "\n".join(keep)


__all__ = [
    "CharacterMeta",
    "CharacterRuntimeResult",
    "CharacterRuntime",
    "PersonalityProfile",
    "PersonalityDecision",
    "StateSnapshot",
    "PromptBudgets",
    "PromptPack",
]
