from __future__ import annotations

import json
import os
import re
import time
import unicodedata
import datetime as dt
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from pathlib import Path
from threading import RLock
from types import SimpleNamespace
from typing import Any

from core.spec_registry import invalidate_spec_cache, load_character_spec, load_spec
from config.settings import load_config
from modules.character.dialog_policies import (
    compute_dialog_flags as dialog_compute_dialog_flags,
    extract_term_directive as dialog_extract_term_directive,
    local_date_kyiv as dialog_local_date_kyiv,
    local_region_name as dialog_local_region_name,
    trim_leading_greeting as dialog_trim_leading_greeting,
)
from modules.character import IdentityCoreBuilder, PersonaSnapshotBuilder
from core.debug_trace import DebugTrace
from core.character_runtime import CharacterRuntime
from core.character_runtime import PromptPack
from core.mode_selector import ModeSelector, list_runtime_modes, normalize_mode_name
from llm.provider_base import LLMProviderBase, LLMRequest, LLMResponse, Message, Timings, ToolCall, ToolSpec, Usage
from llm.task_router import run_task_model
from llm.tokenizer import estimate_tokens
from memory_core.adapter import MemoryCoreAdapter
from memory_core.processors.episode_processor import EpisodeProcessor
from memory_core.retrieval.history_tools import history_tools_list, HistoryReadResult
from memory_core.topic import TopicRouter, TopicStore, TopicToolService, topic_tools_list
from memory_core.utils.summary_quality import is_low_quality_session_summary, sanitize_session_summary_text
from metadata.metadata_extractor import MetadataExtractor
from modules.character.evaluator import ResponseConstraintEvaluator
from modules.studio.studio_generator import StudioGenerator
from modules.internet.web.query_text import strip_service_command_prefix
from modules.internet.web.result_processor import format_postprocess_citation_suffix
from prompt_engine import PromptEngine
from utils.datetime_local import now_local_ts, parse_time_to_epoch
from utils.logger import get_logger, log_json
from modules.internet.web.stage import WebStageV2


PROFILE_FAST = "FAST"
PROFILE_BALANCED = "BALANCED"
PROFILE_QUALITY = "QUALITY"
PROFILE_ECONOM = "ECONOM"
PROFILE_ASYA = "ASYA"
PROFILE_AUTONOMOUS = "AUTONOMOUS"
_SELF_MEMORY_EXACT_MODE = "self_memory_exact"
_MEMORY_TOOL_NAME = "memory_retrieve"
_TOPIC_TOOL_NAMES = {"topic_read", "topic_search", "topic_related"}
_MEMORY_REASONING_TAG = "[MEMORY_REASONING_CHECK]"
_AGENT_LOOP_MIN_TOOL_CALLS = 1
_AGENT_LOOP_MAX_TOOL_CALLS = 2
LOGGER = get_logger(__name__)
WEB_TRACE_LOGGER = get_logger("web.trace")
_WEB_TRACE_JSONL_LOCK = RLock()


@dataclass
class PipelineContext:
    route: str
    user_msg: str
    state: dict[str, Any]
    meta: dict[str, Any]
    retrieved_memories: list[Any]
    traits: dict[str, Any]
    policies: dict[str, Any]
    profile: str
    clean_user_msg: str = ""
    tags: dict[str, Any] = field(default_factory=dict)
    plan: str = ""
    personality: dict[str, Any] = field(default_factory=dict)
    memory_context: dict[str, Any] = field(default_factory=dict)
    prompt_pack: PromptPack | None = None
    prompt_messages: list[Message] = field(default_factory=list)
    prompt_sections: dict[str, str] = field(default_factory=dict)
    raw_output: str = ""
    text: str = ""
    thinking: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    tool_results: list[dict[str, Any]] = field(default_factory=list)
    structured_output: dict[str, Any] = field(default_factory=dict)
    memory_ops: list[dict[str, Any]] = field(default_factory=list)
    ui_actions: list[dict[str, Any]] = field(default_factory=list)
    logs: list[str] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    stop: bool = False


@dataclass(frozen=True)
class PipelineResult:
    text: str
    thinking: str = ""
    structured_output: dict[str, Any] = field(default_factory=dict)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    memory_ops: list[dict[str, Any]] = field(default_factory=list)
    ui_actions: list[dict[str, Any]] = field(default_factory=list)
    logs: list[str] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)
    debug_trace: dict[str, Any] = field(default_factory=dict)
    memory_debug_snapshot: dict[str, Any] = field(default_factory=dict)


class PipelineStage(ABC):
    name = "stage"

    @abstractmethod
    def run(self, ctx: PipelineContext) -> PipelineContext:
        raise NotImplementedError


class PreprocessStage(PipelineStage):
    name = "preprocess"

    def __init__(self, metadata_extractor: MetadataExtractor | None = None):
        self._metadata = metadata_extractor or MetadataExtractor(cache_size=240)

    def run(self, ctx: PipelineContext) -> PipelineContext:
        clean = _normalize_text(ctx.user_msg)
        ctx.clean_user_msg = clean
        meta_state = dict(ctx.state or {})
        meta_state.setdefault("source", _pick(ctx.meta.get("source"), ctx.meta.get("input_source"), "text"))
        metadata = self._metadata.extract(text=clean, state=meta_state, last_messages=ctx.state.get("history"))
        topic = _pick(
            _as_dict(ctx.meta).get("topic"),
            _as_dict(ctx.state).get("topic"),
            _as_dict(_as_dict(ctx.state).get("context_tags")).get("topic"),
        )
        ctx.tags = {
            "lang": metadata.lang,
            "mood": metadata.emotion.label,
            "emotion": metadata.emotion.label,
            "intent": metadata.intent.label,
            "topic": topic or "",
            "intent_conf": float(metadata.intent.conf),
            "emotion_intensity": float(metadata.emotion.intensity),
            "emotion_arousal": float(metadata.emotion.arousal),
            "metadata_tags": list(metadata.tags),
        }
        for key in ("now_iso", "timezone", "previous_user_at", "minutes_since_previous", "same_calendar_day"):
            value = ctx.meta.get(key)
            if value is None:
                continue
            text_value = str(value).strip()
            if not text_value:
                continue
            ctx.tags[key] = text_value
        greeting_flags = _compute_greeting_flags(
            text=clean,
            state=ctx.state,
            meta=ctx.meta,
            policies=ctx.policies,
        )
        dialog_mode = dict(greeting_flags.get("dialog_mode") or {})
        address_terms_policy = _as_dict(greeting_flags.get("address_terms_policy"))
        ctx.state["address_terms"] = _apply_address_term_updates(
            ctx.state.get("address_terms"),
            address_terms_policy=address_terms_policy,
            conversation_id=str(ctx.state.get("conversation_id") or ctx.meta.get("conversation_id") or ""),
        )
        allowed_term = str(greeting_flags.get("allowed_term") or address_terms_policy.get("allowed_term") or "")
        use_term_now = _to_bool(greeting_flags.get("use_term_now"), default=_to_bool(address_terms_policy.get("use_term_now"), default=False))
        address_terms_policy["allowed_term"] = allowed_term
        address_terms_policy["use_term_now"] = use_term_now
        ctx.tags["user_greeting"] = _bool_to_text(greeting_flags["user_greeting"])
        ctx.tags["new_session"] = _bool_to_text(greeting_flags["new_session"])
        ctx.tags["greeted_today"] = _bool_to_text(greeting_flags["greeted_today"])
        ctx.tags["allow_greeting"] = _bool_to_text(greeting_flags["allow_greeting"])
        ctx.tags["smalltalk_allowed"] = _bool_to_text(greeting_flags.get("smalltalk_allowed", True))
        ctx.tags["should_ask_back"] = _bool_to_text(greeting_flags.get("should_ask_back", False))
        ctx.tags["local_date"] = str(greeting_flags.get("local_date") or _local_day_kyiv())
        ctx.tags["local_region"] = str(greeting_flags.get("local_region") or _local_region_name())
        ctx.tags["greeting_allowed"] = _bool_to_text(bool(dialog_mode.get("greeting_allowed", greeting_flags["allow_greeting"])))
        ctx.tags["dialog_sarcasm_level"] = f"{_to_float(dialog_mode.get('sarcasm_level'), 0.0) or 0.0:.3f}"
        ctx.tags["dialog_warmth_level"] = f"{_to_float(dialog_mode.get('warmth_level'), 0.0) or 0.0:.3f}"
        ctx.tags["dialog_strictness_level"] = f"{_to_float(dialog_mode.get('strictness_level'), 0.0) or 0.0:.3f}"
        ctx.tags["dialog_verbosity_level"] = f"{_to_float(dialog_mode.get('verbosity_level'), 0.0) or 0.0:.3f}"
        ctx.tags["is_technical"] = _bool_to_text(bool(greeting_flags.get("is_technical", False)))
        ctx.tags["allowed_term"] = allowed_term
        ctx.tags["use_term_now"] = _bool_to_text(use_term_now)
        ctx.tags["address_terms_policy"] = _compact_json(address_terms_policy)
        if greeting_flags["conversation_state"]:
            ctx.tags["conversation_state"] = str(greeting_flags["conversation_state"])

        state_tags = _as_dict(ctx.state.get("context_tags"))
        state_tags["user_greeting"] = ctx.tags["user_greeting"]
        state_tags["new_session"] = ctx.tags["new_session"]
        state_tags["greeted_today"] = ctx.tags["greeted_today"]
        state_tags["allow_greeting"] = ctx.tags["allow_greeting"]
        state_tags["smalltalk_allowed"] = ctx.tags["smalltalk_allowed"]
        state_tags["should_ask_back"] = ctx.tags["should_ask_back"]
        state_tags["local_date"] = ctx.tags["local_date"]
        state_tags["local_region"] = ctx.tags["local_region"]
        state_tags["greeting_allowed"] = ctx.tags["greeting_allowed"]
        state_tags["dialog_sarcasm_level"] = ctx.tags["dialog_sarcasm_level"]
        state_tags["dialog_warmth_level"] = ctx.tags["dialog_warmth_level"]
        state_tags["dialog_strictness_level"] = ctx.tags["dialog_strictness_level"]
        state_tags["dialog_verbosity_level"] = ctx.tags["dialog_verbosity_level"]
        state_tags["is_technical"] = ctx.tags["is_technical"]
        state_tags["allowed_term"] = ctx.tags["allowed_term"]
        state_tags["use_term_now"] = ctx.tags["use_term_now"]
        for key in ("now_iso", "timezone", "previous_user_at", "minutes_since_previous", "same_calendar_day"):
            value = str(ctx.tags.get(key) or "").strip()
            if value:
                state_tags[key] = value
        if greeting_flags["conversation_state"]:
            state_tags["conversation_state"] = str(greeting_flags["conversation_state"])
        ctx.state["context_tags"] = state_tags
        ctx.state["dialog_mode"] = dict(dialog_mode)
        ctx.state["address_terms_policy"] = dict(address_terms_policy)
        ctx.state["allowed_term"] = allowed_term
        ctx.state["use_term_now"] = use_term_now

        ctx.meta["user_greeting"] = bool(greeting_flags["user_greeting"])
        ctx.meta["new_session"] = bool(greeting_flags["new_session"])
        ctx.meta["greeted_today"] = bool(greeting_flags["greeted_today"])
        ctx.meta["allow_greeting"] = bool(greeting_flags["allow_greeting"])
        ctx.meta["smalltalk_allowed"] = bool(greeting_flags.get("smalltalk_allowed", True))
        ctx.meta["should_ask_back"] = bool(greeting_flags.get("should_ask_back", False))
        ctx.meta["local_date"] = ctx.tags["local_date"]
        ctx.meta["local_region"] = ctx.tags["local_region"]
        ctx.meta["dialog_mode"] = dict(dialog_mode)
        ctx.meta["is_technical"] = bool(greeting_flags.get("is_technical", False))
        ctx.meta["verbosity_level"] = _to_float(dialog_mode.get("verbosity_level"), 0.0) or 0.0
        ctx.meta["strictness_level"] = _to_float(dialog_mode.get("strictness_level"), 0.0) or 0.0
        ctx.meta["warmth_level"] = _to_float(dialog_mode.get("warmth_level"), 0.0) or 0.0
        ctx.meta["sarcasm_level"] = _to_float(dialog_mode.get("sarcasm_level"), 0.0) or 0.0
        ctx.meta["address_terms_policy"] = dict(address_terms_policy)
        ctx.meta["allowed_term"] = allowed_term
        ctx.meta["use_term_now"] = use_term_now
        if greeting_flags["conversation_state"]:
            ctx.meta["conversation_state"] = str(greeting_flags["conversation_state"])
        ctx.meta["metadata"] = metadata.to_dict()
        if not clean and ctx.route in {"chat", "command"}:
            ctx.text = "Empty request. Please send text."
            ctx.stop = True
        ctx.logs.append(
            "stage=preprocess "
            f"lang={ctx.tags.get('lang')} intent={ctx.tags.get('intent')} "
            f"mood={ctx.tags.get('mood')} allow_greeting={ctx.tags.get('allow_greeting')} "
            f"use_term_now={ctx.tags.get('use_term_now')} verbosity={ctx.tags.get('dialog_verbosity_level')}"
        )
        _emit_turn_summary(
            ctx,
            "intent_summary",
            summary=(
                f"intent={ctx.tags.get('intent') or '-'} lang={ctx.tags.get('lang') or '-'} "
                f"mood={ctx.tags.get('mood') or '-'} topic={ctx.tags.get('topic') or '-'} "
                f"intent_conf={float(_to_float(ctx.tags.get('intent_conf'), 0.0) or 0.0):.2f}"
            ),
            route=str(ctx.route or ""),
            intent=str(ctx.tags.get("intent") or ""),
            intent_confidence=float(_to_float(ctx.tags.get("intent_conf"), 0.0) or 0.0),
            lang=str(ctx.tags.get("lang") or ""),
            mood=str(ctx.tags.get("mood") or ""),
            topic=str(ctx.tags.get("topic") or ""),
            metadata_tags=list(_as_list(ctx.tags.get("metadata_tags"))),
            is_technical=_to_bool(ctx.meta.get("is_technical"), default=False),
        )
        return ctx


class ModeSelectStage(PipelineStage):
    name = "mode_select"

    def __init__(self, selector: ModeSelector | None = None):
        self._selector = selector or ModeSelector()

    def run(self, ctx: PipelineContext) -> PipelineContext:
        if ctx.route != "chat":
            ctx.logs.append("stage=mode_select skipped(route)")
            return ctx

        active_mode = normalize_mode_name(
            _pick(
                ctx.state.get("active_mode"),
                ctx.meta.get("active_mode"),
                ctx.state.get("mode"),
                "chatting",
            ),
            allow_custom=True,
        )
        mode_lock = _to_bool(
            _pick_value(ctx.state.get("mode_lock"), ctx.meta.get("mode_lock"), False),
            default=False,
        )
        tags = [str(x).strip().lower() for x in _as_list(ctx.tags.get("metadata_tags")) if str(x).strip()]
        decision = self._selector.decide(
            active_mode=active_mode,
            mode_lock=mode_lock,
            intent=str(ctx.tags.get("intent") or ""),
            emotion=str(ctx.tags.get("mood") or ""),
            tags=tags,
        )

        target = normalize_mode_name(decision.mode, allow_custom=True)
        should_switch = self._selector.should_switch(current_mode=active_mode, decision=decision)
        if should_switch:
            ctx.memory_ops.append(
                {
                    "op": "state_mode",
                    "value": target,
                    "reason": f"auto_mode:{decision.reason}",
                    "confidence": float(decision.confidence),
                }
            )
            ctx.state["active_mode"] = target
            if target == "debugger":
                ctx.state["mode"] = "debug"
            elif target == "engineer":
                ctx.state["mode"] = "coding"
            elif target in {"helper", "planner"}:
                ctx.state["mode"] = "task"
            else:
                ctx.state["mode"] = "chat"
        else:
            ctx.state["active_mode"] = active_mode
            target = active_mode

        ctx.tags["active_mode"] = target
        ctx.meta["active_mode"] = target
        ctx.meta["mode_lock"] = bool(mode_lock)
        ctx.logs.append(
            "stage=mode_select "
            f"mode={target} lock={int(bool(mode_lock))} "
            f"decision={decision.reason} conf={decision.confidence:.2f}"
        )
        return ctx


class PlanStage(PipelineStage):
    name = "plan"

    def run(self, ctx: PipelineContext) -> PipelineContext:
        mode = normalize_mode_name(_pick(ctx.state.get("active_mode"), ctx.state.get("mode"), "chatting"), allow_custom=True)
        goal = _pick(
            ctx.state.get("active_goal"),
            ctx.state.get("current_task"),
            ctx.state.get("task"),
        )
        intent = str(ctx.tags.get("intent") or "chat")
        if mode in {"planner", "helper"} and goal:
            ctx.plan = f"goal={goal}; intent={intent}; style=step-by-step"
        elif intent in {"question", "task", "bug_report", "code_review", "planning"}:
            ctx.plan = f"intent={intent}; provide concise actionable answer"
        else:
            ctx.plan = f"intent={intent}; maintain conversational flow"
        ctx.logs.append("stage=plan")
        return ctx


class PersonalityStage(PipelineStage):
    name = "personality"

    def __init__(self, character_runtime: CharacterRuntime | None = None):
        self._characters = character_runtime or CharacterRuntime()

    def run(self, ctx: PipelineContext) -> PipelineContext:
        if ctx.route != "chat":
            ctx.logs.append("stage=personality skipped(route)")
            return ctx

        active_character = _pick(
            ctx.meta.get("character"),
            ctx.meta.get("character_id"),
            ctx.state.get("active_character_id"),
            ctx.state.get("active_personality_id"),
            "asya",
        ).lower()
        known = set(self._characters.list_ids())
        if known and active_character not in known:
            active_character = sorted(known)[0]

        update = self._characters.evolve(
            text=str(ctx.clean_user_msg or ctx.user_msg or ""),
            meta={
                "lang": str(ctx.tags.get("lang") or ""),
                "intent": str(ctx.tags.get("intent") or ""),
                "mood": str(ctx.tags.get("mood") or ""),
                "emotion": str(ctx.tags.get("emotion") or ctx.tags.get("mood") or ""),
                "emotion_intensity": float(_to_float(ctx.tags.get("emotion_intensity"), 0.0)),
                "emotion_arousal": float(_to_float(ctx.tags.get("emotion_arousal"), 0.0)),
                "mode": str(ctx.state.get("mode") or "chat"),
                "active_mode": str(ctx.state.get("active_mode") or "chatting"),
                "turn_id": ctx.meta.get("turn_id"),
                "conversation_id": ctx.meta.get("conversation_id") or ctx.state.get("conversation_id"),
                "topic": str(ctx.tags.get("topic") or ""),
                "metadata_tags": list(ctx.tags.get("metadata_tags") or []),
                "dialog_mode": dict(ctx.meta.get("dialog_mode") or ctx.state.get("dialog_mode") or {}),
                "is_technical": _to_bool(_pick_value(ctx.tags.get("is_technical"), ctx.meta.get("is_technical"), False), default=False),
            },
            active_character_id=active_character,
        )

        prev_character = str(ctx.state.get("active_character_id") or "").strip().lower()
        ctx.state["active_character_id"] = update.character_id
        ctx.state["active_personality_id"] = update.character_id
        ctx.state["character_state"] = dict(update.state)
        ctx.state["user_addressing"] = dict(update.user_addressing or {})
        ctx.meta["user_addressing"] = dict(update.user_addressing or {})
        try:
            compiled_persona = str(self._characters.build_personality_block(update.character_id) or "").strip()
        except Exception:
            compiled_persona = ""
        ctx.state["character_prompt_block"] = compiled_persona or str(update.prompt_block or "")
        ctx.state["mood"] = str(update.mood or "")
        if update.llm_profile:
            # Character preset is source-of-truth for LLM profile.
            ctx.state["quality_profile"] = str(update.llm_profile).strip().upper()
            ctx.meta.setdefault("personality_llm_profile", update.llm_profile)
            ctx.policies.setdefault("personality_llm_profile", update.llm_profile)

        merged_traits = _as_dict(ctx.traits)
        merged_traits.setdefault("character", update.character_id)
        merged_traits.setdefault("personality", update.character_id)
        merged_traits.setdefault("profile", update.character_id)
        merged_traits.setdefault("mood", update.mood)
        for key, value in dict(update.trait_values or {}).items():
            if key not in merged_traits:
                merged_traits[key] = value
        merged_traits["character_traits_meta"] = dict(update.traits or {})
        if update.style_coefficients:
            merged_traits["style_coefficients"] = dict(update.style_coefficients)
        if update.effective_traits:
            merged_traits["effective_traits"] = dict(update.effective_traits)
        ctx.traits = merged_traits
        ctx.personality = {
            "target_personality_id": update.character_id,
            "reason": "character_engine",
            "changes": list(update.changes or []),
            "mood": update.mood,
        }

        if update.character_id != prev_character:
            ctx.memory_ops.append(
                {
                    "op": "state_character",
                    "value": update.character_id,
                    "locked": bool(ctx.state.get("character_locked", False)),
                    "ts": now_local_ts(),
                    "reason": "character_stage",
                }
            )
            ctx.memory_ops.append(
                {
                    "op": "state_personality",
                    "value": update.character_id,
                    "locked": bool(ctx.state.get("personality_locked", False)),
                    "ts": now_local_ts(),
                    "reason": "character_stage_mirror",
                }
            )

        ctx.logs.append(
            "stage=personality "
            f"character={update.character_id} mood={update.mood} "
            f"changes={len(update.changes)}"
        )
        return ctx


class TopicRoutingStage(PipelineStage):
    name = "topic_routing"

    def __init__(self, topic_router: TopicRouter | None):
        self.topic_router = topic_router

    def run(self, ctx: PipelineContext) -> PipelineContext:
        if self.topic_router is None:
            ctx.logs.append("stage=topic_routing skipped(no_router)")
            return ctx
        if ctx.route != "chat":
            ctx.logs.append("stage=topic_routing skipped(route)")
            return ctx

        text = str(ctx.clean_user_msg or ctx.user_msg or "").strip()
        if not text:
            ctx.logs.append("stage=topic_routing skipped(empty)")
            return ctx

        conversation_id = str(
            _pick(ctx.meta.get("conversation_id"), ctx.state.get("conversation_id"), "default")
        ).strip() or "default"
        workspace_id = str(
            _pick(ctx.meta.get("workspace_id"), ctx.state.get("active_character_id"), "global")
        ).strip() or "global"
        previous_thread_id = str(
            _pick(ctx.state.get("active_topic_thread_id"), ctx.state.get("topic_thread_id"), "")
        ).strip()
        decision = self.topic_router.route_turn(
            text=text,
            visible_chat_id=conversation_id,
            workspace_id=workspace_id,
            session_id=conversation_id,
            current_state=ctx.state,
            meta=ctx.meta,
        )

        ctx.state["topic_thread_id"] = decision.thread_id
        ctx.state["topic_key"] = decision.topic_key
        ctx.state["topic_thread_title"] = decision.title
        ctx.state["active_topic_thread_id"] = decision.thread_id
        ctx.state["active_topic_key"] = decision.topic_key
        ctx.state["active_topic_title"] = decision.title
        ctx.state["topic_route_reason"] = decision.reason
        ctx.state["topic_route_score"] = float(decision.score)
        ctx.state["related_topic_thread_ids"] = list(decision.related_thread_ids or [])
        ctx.state["topic_candidates"] = list(decision.related_thread_ids or [])
        ctx.state["topic_last_route_reason"] = decision.reason
        ctx.state["topic_last_route_score"] = float(decision.score)
        if previous_thread_id and previous_thread_id != decision.thread_id:
            ctx.state["topic_last_switch_at"] = now_local_ts()
        elif not previous_thread_id:
            ctx.state["topic_last_switch_at"] = now_local_ts()
        ctx.state["topic_stack"] = _update_topic_stack(
            ctx.state.get("topic_stack"),
            thread_id=decision.thread_id,
            title=decision.title,
            topic_key=decision.topic_key,
        )

        ctx.meta["visible_chat_id"] = conversation_id
        ctx.meta["topic_thread_id"] = decision.thread_id
        ctx.meta["topic_key"] = decision.topic_key
        ctx.meta["topic_title"] = decision.title
        ctx.meta["topic_thread_title"] = decision.title
        ctx.meta["topic_route_reason"] = decision.reason
        ctx.meta["topic_route_score"] = float(decision.score)
        ctx.meta["related_topic_thread_ids"] = list(decision.related_thread_ids or [])

        ctx.tags["topic"] = decision.topic_key
        ctx.tags["topic_thread_id"] = decision.thread_id
        ctx.tags["topic_title"] = decision.title
        state_tags = _as_dict(ctx.state.get("context_tags"))
        state_tags["topic"] = decision.topic_key
        ctx.state["context_tags"] = state_tags

        ctx.logs.append(
            f"stage=topic_routing thread={decision.thread_id} "
            f"reason={decision.reason} score={float(decision.score):.2f}"
        )
        return ctx


class MemoryNativeStateStage(PipelineStage):
    """
    Построение memory_native_state — always-on memory layer.

    Этот этап ВСЕГДА строит компактный, структурированный слой памяти,
    который модель ощущает как своё внутреннее состояние.

    Важно: memory_native_state строится ДО любого retrieval и доступен
    даже если memory retrieval не сработал.
    """
    name = "memory_native_state"

    def __init__(self, memory_core: MemoryCoreAdapter | None = None):
        self.memory_core = memory_core or MemoryCoreAdapter()
        self._cfg = load_config()

    def run(self, ctx: PipelineContext) -> PipelineContext:
        """
        Построить memory_native_state из текущего state.

        Это always-on слой — строится всегда, даже для не-chat маршрутов.
        """
        _ensure_debug_trace(ctx)

        # Строим memory_native_state из текущего state и memory_context
        # Это always-on слой — доступен даже без retrieval и для не-chat маршрутов
        native_state = {
            "hits": ctx.memory_context.get("hits", []) if ctx.memory_context else [],
            "citations": ctx.memory_context.get("citations", []) if ctx.memory_context else [],
        }
        ctx.state["memory_native_state"] = native_state

        # Логируем, что было построено
        has_hits = bool(native_state.get("hits"))
        has_citations = bool(native_state.get("citations"))

        if ctx.route != "chat":
            ctx.logs.append(
                f"stage=memory_native_state built (non-chat route={ctx.route}) "
                f"hits={int(has_hits)} citations={int(has_citations)}"
            )
        else:
            ctx.logs.append(
                f"stage=memory_native_state built "
                f"hits={int(has_hits)} citations={int(has_citations)}"
            )
        return ctx


class MemoryRetrieveStage(PipelineStage):
    name = "memory_retrieve"

    def __init__(self, memory_core: MemoryCoreAdapter | None = None, memory_manager: Any | None = None):
        self.memory_core = memory_core or memory_manager
        self._cfg = load_config()

    def run(self, ctx: PipelineContext) -> PipelineContext:
        """
        Memory retrieve stage через memory_core.

        Выполняет запрос к memory_core и сохраняет результат в ctx.memory_context.
        """
        _ensure_debug_trace(ctx)
        if ctx.route != "chat":
            ctx.logs.append("stage=memory_retrieve skipped(route)")
            return ctx

        query = str(ctx.clean_user_msg or ctx.user_msg or "").strip()
        if not query:
            ctx.logs.append("stage=memory_retrieve skipped(empty)")
            return ctx

        memory_core = ctx.meta.get("memory_core") or ctx.meta.get("memory_manager") or self.memory_core
        if memory_core is None:
            ctx.logs.append("stage=memory_retrieve skipped(no_memory_core)")
            return ctx

        # Определяем workspace и session
        workspace_id = str(ctx.meta.get("workspace_id") or ctx.state.get("active_character_id") or "global")
        session_id = str(ctx.meta.get("conversation_id") or ctx.state.get("conversation_id") or "default")
        
        # Выполняем запрос через memory_core
        try:
            if hasattr(memory_core, "query"):
                result = memory_core.query(
                    text=query,
                    workspace_id=workspace_id,
                    session_id=session_id,
                    top_k=int(self._cfg.memory_core_top_k or 8),
                    include_citations=True,
                    topic_thread_id=str(ctx.state.get("topic_thread_id") or ctx.meta.get("topic_thread_id") or ""),
                    related_topic_ids=list(ctx.state.get("related_topic_thread_ids") or []),
                )
            else:
                request = {
                    "text": query,
                    "workspace_id": workspace_id,
                    "session_id": session_id,
                    "topic_thread_id": str(ctx.state.get("topic_thread_id") or ctx.meta.get("topic_thread_id") or ""),
                    "related_topic_ids": list(ctx.state.get("related_topic_thread_ids") or []),
                    "top_k": int(self._cfg.memory_core_top_k or 8),
                    "include_citations": True,
                }
                raw_result = memory_core.build_context(request)
                result = raw_result.to_dict() if hasattr(raw_result, "to_dict") else dict(raw_result or {})
            result = dict(result or {})
            result_blocks = dict(result.get("blocks", {}) or {})
            if is_low_quality_session_summary(str(result_blocks.get("session_summary") or "").strip()):
                result_blocks.pop("session_summary", None)

            ctx.memory_context = {
                "context_blocks": list(result.get("context_blocks", []) or []),
                "hits": list(result.get("hits", []) or []),
                "citations": list(result.get("citations", []) or []),
                "blocks": result_blocks,
                "selected": list(result.get("selected", []) or []),
                "dropped": list(result.get("dropped", []) or []),
                "recent_user_state": dict(result.get("recent_user_state", {}) or {}),
                "response_bias": dict(result.get("response_bias", {}) or {}),
                "dialog_episode_hits": list(result.get("dialog_episode_hits", []) or []),
                "task_continuity": dict(result.get("task_continuity", {}) or {}),
                "open_questions": list(result.get("open_questions", []) or []),
                "current_decisions": list(result.get("current_decisions", []) or []),
                "fact_expectation": dict(result.get("fact_expectation", {}) or {}),
                "self_facts_context": dict(result.get("self_facts_context", {}) or {}),
                "recall_mode": str(result.get("recall_mode") or ""),
                "debug": dict(result.get("debug", {}) or {}),
            }
            _augment_memory_context_with_topic_summary(
                ctx,
                workspace_id=workspace_id,
                memory_core=memory_core,
            )

            governor_snapshot: dict[str, Any] = {}
            if hasattr(memory_core, "get_governor_profile_snapshot"):
                snapshot = memory_core.get_governor_profile_snapshot(session_id)
                if snapshot is not None:
                    if hasattr(snapshot, "to_dict"):
                        governor_snapshot = dict(snapshot.to_dict() or {})
                    elif hasattr(snapshot, "__dict__"):
                        governor_snapshot = dict(vars(snapshot) or {})
                    else:
                        governor_snapshot = dict(snapshot or {})
            if governor_snapshot:
                ctx.state["active_profile_snapshot"] = dict(governor_snapshot)
                ctx.meta["active_profile_snapshot"] = dict(governor_snapshot)

            trace = _ensure_debug_trace(ctx)
            selected_facts = []
            for item in list(ctx.memory_context.get("selected") or []):
                metadata = _as_dict(_as_dict(item).get("metadata"))
                fact = _as_dict(metadata.get("fact"))
                if fact:
                    selected_facts.append(
                        {
                            "predicate": str(fact.get("predicate") or "").strip(),
                            "value": fact.get("value"),
                            "subject": str(fact.get("subject") or "").strip(),
                        }
                    )
            trace.memory_retrieval = {
                "query": query,
                "selected_facts": selected_facts,
                "exact_self_fact_hits": {
                    "self_facts_context": dict(ctx.memory_context.get("self_facts_context") or {}),
                    "fact_expectation": dict(ctx.memory_context.get("fact_expectation") or {}),
                },
            }
            if governor_snapshot:
                trace.active_profile = dict(governor_snapshot)
            
            ctx.logs.append(
                f"stage=memory_retrieve done hits={len(result.get('hits', []))} "
                f"blocks={len(result.get('context_blocks', []))}"
            )
        except Exception as exc:
            ctx.logs.append(f"stage=memory_retrieve error={type(exc).__name__}")
        
        return ctx


def _memory_cfg_int(cfg: Any, name: str, *, minimum: int) -> int:
    try:
        value = int(getattr(cfg, name))
    except Exception:
        value = minimum
    return max(minimum, value)


def _pick_memory_int(*values: Any, default: int, minimum: int) -> int:
    chosen = _pick_value(*values, default)
    try:
        value = int(chosen)
    except Exception:
        value = default
    return max(minimum, value)


class EpisodeContinuityStage(PipelineStage):
    """
    EpisodeContinuityStage — мост к memory_core для continuity.

    Берёт session_id из context и получает continuity pack из memory_core.
    """
    name = "episode_continuity"

    def __init__(self, memory_core: MemoryCoreAdapter | None = None, memory_manager: Any | None = None):
        self.memory_core = memory_core or memory_manager

    def run(self, ctx: PipelineContext) -> PipelineContext:
        trace = _ensure_debug_trace(ctx)
        memory_core = ctx.meta.get("memory_core") or ctx.meta.get("memory_manager") or self.memory_core
        session_id = str(ctx.meta.get("conversation_id") or ctx.state.get("conversation_id") or "").strip()
        workspace_id = str(ctx.meta.get("workspace_id") or ctx.state.get("active_character_id") or "global").strip()
        now_ts = _pick_value(ctx.meta.get("now_ts"), now_local_ts())
        user_text = str(ctx.clean_user_msg or ctx.user_msg or "").strip().lower()

        if not session_id:
            ctx.logs.append("stage=episode_continuity skipped(no_session_id)")
            return ctx

        continuity = {}
        if memory_core is not None and hasattr(memory_core, "get_episode_continuity"):
            try:
                continuity = memory_core.get_episode_continuity(
                    session_id=session_id,
                    workspace_id=workspace_id,
                    topic_thread_id=str(ctx.state.get("topic_thread_id") or ctx.meta.get("topic_thread_id") or ""),
                )
            except Exception:
                continuity = {}
        if not continuity:
            continuity = dict(_as_dict(ctx.memory_context).get("task_continuity") or {})

        ctx.state["episode_continuity"] = continuity
        existing_task = dict(ctx.state.get("active_task") or {})
        if existing_task:
            if user_text in {"done", "готово", "закрыли", "finished"}:
                ctx.state["active_task"] = {}
                ctx.state.pop("active_goal", None)
                ctx.state.pop("active_tasks", None)
                ctx.state.pop("current_decisions", None)
                ctx.state.pop("_active_task_source", None)
                trace.active_task = {"event": "close", "reason": "explicit_close_phrase", "source": "continuation"}
                ctx.logs.append("stage=episode_continuity active_task_closed(reason=explicit_close_phrase)")
                return ctx
            if "another topic" in user_text or "другая тема" in user_text:
                ctx.state["active_task"] = {}
                ctx.state.pop("active_goal", None)
                ctx.state.pop("active_tasks", None)
                ctx.state.pop("current_decisions", None)
                ctx.state.pop("_active_task_source", None)
                trace.active_task = {"event": "clear", "reason": "switch_topic_phrase", "source": "continuation"}
                ctx.logs.append("stage=episode_continuity active_task_cleared(reason=switch_topic_phrase)")
                return ctx
            if len(user_text.split()) <= 3:
                kept_task = dict(existing_task)
                kept_task["updated_at"] = now_ts
                ctx.state["active_task"] = kept_task
                ctx.state["active_goal"] = str(
                    kept_task.get("current_goal") or kept_task.get("summary_short") or ctx.state.get("active_goal") or ""
                ).strip()
                if list(kept_task.get("decisions") or []):
                    ctx.state["active_tasks"] = [dict(kept_task)]
                    ctx.state["current_decisions"] = list(kept_task.get("decisions") or [])
                trace.active_task = {"event": "keep", "reason": "short_followup", "source": "continuation"}
                ctx.logs.append("stage=episode_continuity kept_active_task(reason=short_followup)")
                return ctx

        persisted_task = dict(_as_dict(continuity).get("active_task") or {})
        if persisted_task:
            ctx.state["active_task"] = persisted_task
            ctx.state["active_goal"] = str(
                persisted_task.get("current_goal") or persisted_task.get("summary_short") or ""
            ).strip()
            if list(persisted_task.get("decisions") or []):
                ctx.state["current_decisions"] = list(persisted_task.get("decisions") or [])
                ctx.state["active_tasks"] = [dict(persisted_task)]
            trace.active_task = {"event": "restore", "reason": "persisted_task_continuity", "source": "continuation"}
            ctx.logs.append("stage=episode_continuity restored_active_task(source=continuation)")
            return ctx

        memory_context = _as_dict(ctx.memory_context)
        dialog_episode_hits = list(memory_context.get("dialog_episode_hits") or [])
        if dialog_episode_hits:
            top_hit = _as_dict(dialog_episode_hits[0])
            episode = _as_dict(top_hit.get("episode"))
            episode_id = str(episode.get("id") or top_hit.get("record_id") or "").strip()
            active_task = {
                "task_id": f"task:{episode_id}" if episode_id and not episode_id.startswith("task:") else episode_id,
                "topic": str(episode.get("topic") or "").strip(),
                "status": "waiting_user" if list(episode.get("open_questions") or []) else "active",
                "summary_short": str(episode.get("summary_short") or top_hit.get("summary_short") or "").strip(),
                "current_goal": str(top_hit.get("summary_reasoning") or "").strip(),
                "decisions": list(top_hit.get("decisions") or episode.get("decisions") or []),
                "open_questions": list(episode.get("open_questions") or []),
                "source_episode_id": episode_id,
                "updated_at": now_ts,
            }
            ctx.state["active_task"] = active_task
            ctx.state["active_goal"] = active_task["current_goal"]
            ctx.state["active_tasks"] = [dict(active_task)]
            ctx.state["current_decisions"] = list(active_task.get("decisions") or [])
            trace.active_task = {"event": "resolved", "reason": "episode_open_questions", "source": "episode_hit"}
            ctx.logs.append("stage=episode_continuity resolved_active_task(source=episode_hit)")
            return ctx

        open_questions = list(memory_context.get("open_questions") or [])
        current_decisions = list(memory_context.get("current_decisions") or [])
        if open_questions or current_decisions:
            active_task = {
                "task_id": f"task:runtime:{session_id}",
                "topic": "",
                "status": "waiting_user",
                "current_goal": str((current_decisions or [""])[0] or "").strip(),
                "decisions": current_decisions,
                "open_questions": open_questions,
                "updated_at": now_ts,
            }
            ctx.state["active_task"] = active_task
            ctx.state["active_goal"] = active_task["current_goal"]
            ctx.state["active_tasks"] = [dict(active_task)]
            ctx.state["current_decisions"] = list(current_decisions)
            trace.active_task = {"event": "resolved", "reason": "runtime_open_questions_fallback", "source": "runtime_hints"}
            ctx.logs.append("stage=episode_continuity resolved_active_task(source=runtime_hints)")
            return ctx

        ctx.logs.append(
            f"stage=episode_continuity active={bool(continuity.get('active_episode'))} "
            f"episode={continuity.get('active_episode', {}).get('episode_id', 'none')[:8]}"
        )
        return ctx


def _memory_tool_blocks(pack: dict[str, Any]) -> dict[str, str]:
    blocks = _as_dict(pack.get("blocks"))
    keys = (
        "memory_recall_mode",
        "self_facts",
        "fact_expectation_check",
        "relevant_claims",
        "exact_recall",
        "answer_support",
        "continuity_hints",
        "tone_hints",
        "recalled_dialog",
        "document_evidence",
        "exact_fact_evidence",
        "supporting_messages",
        "supporting_message",
        "working_memory",
        "session_summary",
        "retrieved_semantic",
        "retrieved_episodic",
        "retrieved_docs",
        "active_tool_state",
        "unresolved_items",
    )
    out: dict[str, str] = {}
    for key in keys:
        text = str(blocks.get(key) or "").strip()
        if text:
            out[key] = text
    return out


class PromptBuildStage(PipelineStage):
    name = "prompt_build"

    def __init__(
        self,
        character_runtime: CharacterRuntime,
        persona_snapshot_builder: PersonaSnapshotBuilder | None = None,
        identity_core_builder: IdentityCoreBuilder | None = None,
    ):
        self.character_runtime = character_runtime
        self.persona_snapshot_builder = persona_snapshot_builder or PersonaSnapshotBuilder()
        self.identity_core_builder = identity_core_builder or IdentityCoreBuilder()

    @staticmethod
    def _build_persona_memory_context(memory_context: dict[str, Any] | None) -> dict[str, Any]:
        """
        Строит memory context для persona builder.

        Передаёт не только block_keys, но и distilled memory signals.
        """
        row = _as_dict(memory_context)
        blocks = _as_dict(row.get("blocks"))
        block_keys = sorted(str(key) for key in list(blocks.keys()) if str(key).strip())
        recall_mode = str(
            _pick_value(
                row.get("recall_mode"),
                row.get("mode"),
                row.get("memory_recall_mode"),
                blocks.get("memory_recall_mode"),
                "",
            )
            or ""
        ).strip()
        
        # Извлекаем selected артефакты для signals
        selected = [dict(x) for x in _as_list(row.get("selected")) if isinstance(x, dict)]
        structured_recent_user_state = _as_dict(row.get("recent_user_state"))
        structured_response_bias = _as_dict(row.get("response_bias"))
        
        # Группируем по типам артефактов
        profile_signals: list[str] = []
        task_signals: list[str] = []
        episode_signals: list[str] = []
        emotion_signals: list[str] = []
        
        for item in selected:
            artifact_type = str(item.get("artifact_type") or item.get("memory_type") or "").strip().lower()
            exposure_mode = str(item.get("exposure_mode") or "prompt_safe").strip().lower()
            text = str(item.get("prompt_view") or item.get("summary") or "").strip()
            if not text:
                continue
            if exposure_mode == "latent" and artifact_type != "emotional_state":
                continue
            
            # Типизированные signals (усечённые)
            if artifact_type == "profile_fact":
                profile_signals.append(text)
            elif artifact_type in {"task_state", "task"}:
                task_signals.append(text)
            elif artifact_type in {"episode_event", "episode"}:
                episode_signals.append(text)
            elif artifact_type == "emotional_state":
                emotion_signals.append(text)
        
        return {
            "input_mode": "distilled",
            "block_keys": block_keys,
            "selected_count": int(len(selected)),
            "recall_mode": recall_mode,
            "profile_signals": profile_signals[:5],  # Усекаем
            "task_signals": task_signals[:2],
            "episode_signals": episode_signals[:3],
            "emotion_signals": emotion_signals[:2],
            "recent_user_state": dict(structured_recent_user_state),
            "response_bias": dict(structured_response_bias),
        }

    def _apply_persona_snapshot_prompt_policies(
        self,
        *,
        ctx: PipelineContext,
        payload: dict[str, Any],
    ) -> None:
        ctx.state["persona_snapshot_required"] = True
        ctx.meta["persona_snapshot_required"] = True
        ctx.tags["persona_snapshot_required"] = "true"
        active_task = _as_dict(payload.get("active_task"))
        relation_continuity = _as_dict(payload.get("relation_continuity"))
        recent_user_state = _as_dict(payload.get("recent_user_state"))
        identity_core_interaction = _as_dict(_as_dict(ctx.state.get("identity_core")).get("interaction_style"))

        if active_task:
            _append_policy_rule(
                ctx.policies,
                "Treat PERSONA_SNAPSHOT and ACTIVE_TASK as continuity anchors only when they remain relevant to the current user message; prefer them over loose recalled snippets, but do not force old context onto a clear topic shift.",
            )
            _append_policy_rule(
                ctx.policies,
                "Use PERSONA_SNAPSHOT and ACTIVE_TASK as the primary continuity anchors for the current code or task when they still match the user message.",
            )
        if bool(relation_continuity.get("is_followup")):
            _append_policy_rule(
                ctx.policies,
                "This turn is likely a follow-up. Preserve relation and task continuity only while the current user message stays on the same topic.",
            )
        if bool(recent_user_state.get("low_bandwidth")):
            _append_policy_rule(
                ctx.policies,
                "The user currently appears low-bandwidth. Keep the answer compact, avoid repeated clarification loops, and prefer direct next steps.",
            )
        if bool(recent_user_state.get("frustrated")):
            _append_policy_rule(
                ctx.policies,
                "The user seems frustrated or worn down. Keep the tone steady, reduce teasing, and prioritize concrete help over flourish.",
            )
        if bool(identity_core_interaction.get("prefers_examples_on_user_code")):
            _append_policy_rule(
                ctx.policies,
                "When examples would help, prefer examples grounded in the user's current code or task instead of generic toy snippets.",
            )

    def _build_persona_snapshot_payload(
        self,
        *,
        ctx: PipelineContext,
        prompt_state: dict[str, Any],
    ) -> dict[str, Any]:
        active_character_id = str(
            _pick_value(
                ctx.meta.get("character_id"),
                prompt_state.get("active_character_id"),
                prompt_state.get("active_personality_id"),
                ctx.state.get("active_character_id"),
                ctx.state.get("active_personality_id"),
                "asya",
            )
            or "asya"
        ).strip().lower() or "asya"
        namespace = str(
            _pick_value(
                ctx.meta.get("conversation_id"),
                ctx.state.get("conversation_id"),
                "default",
            )
            or "default"
        ).strip().lower() or "default"
        stored_identity_core: dict[str, Any] = {}
        memory_identity_core_snapshot: dict[str, Any] = {}
        
        # Используем memory_core вместо legacy memory_manager
        memory_core = ctx.meta.get("memory_core")
        if memory_core is not None and hasattr(memory_core, "get_identity_core_snapshot"):
            try:
                memory_identity_core = memory_core.get_identity_core_snapshot(namespace)
                if memory_identity_core is not None:
                    memory_identity_core_snapshot = (
                        dict(memory_identity_core or {})
                        if isinstance(memory_identity_core, dict)
                        else {}
                    )
            except Exception:
                memory_identity_core_snapshot = {}
        
        # Legacy fallback для обратной совместимости
        memory_manager = ctx.meta.get("memory_manager")
        if not memory_identity_core_snapshot and memory_manager is not None and hasattr(memory_manager, "get_identity_core_snapshot"):
            try:
                memory_identity_core = memory_manager.get_identity_core_snapshot(namespace)
                if memory_identity_core is not None:
                    memory_identity_core_snapshot = (
                        dict(memory_identity_core.to_dict() or {})
                        if hasattr(memory_identity_core, "to_dict")
                        else (
                            dict(vars(memory_identity_core) or {})
                            if hasattr(memory_identity_core, "__dict__")
                            else dict(memory_identity_core or {})
                        )
                    )
            except Exception:
                memory_identity_core_snapshot = {}
        runtime_storage = getattr(self.character_runtime, "storage", None)
        if runtime_storage is not None and hasattr(runtime_storage, "load_identity_core"):
            try:
                stored_identity_core = dict(runtime_storage.load_identity_core(active_character_id) or {})
            except Exception:
                stored_identity_core = {}
        identity_core_snapshot = self.identity_core_builder.build(
            character_id=active_character_id,
            active_profile_snapshot=dict(ctx.state.get("active_profile_snapshot") or {}),
            stored_identity_core=stored_identity_core,
            memory_identity_core_snapshot=memory_identity_core_snapshot,
        )
        character_trait_defaults: dict[str, Any] = {}
        try:
            persona_state_spec = dict(load_character_spec(active_character_id, "persona_state", required=False) or {})
        except Exception:
            persona_state_spec = {}
        character_trait_defaults = dict(
            _pick_value(
                _as_dict(_as_dict(persona_state_spec.get("learned")).get("baseline_traits")),
                _as_dict(persona_state_spec.get("traits")),
                {},
            )
            or {}
        )
        identity_core_payload = identity_core_snapshot.to_dict()
        ctx.state["identity_core"] = dict(identity_core_payload)
        ctx.state["identity_core_memory_snapshot"] = dict(memory_identity_core_snapshot)
        ctx.state["identity_core_runtime_fallback"] = dict(stored_identity_core)
        ctx.state["character_trait_defaults"] = dict(character_trait_defaults)
        prompt_state["identity_core"] = dict(identity_core_payload)
        prompt_state["character_trait_defaults"] = dict(character_trait_defaults)
        ctx.meta["identity_core_snapshot"] = dict(identity_core_payload)
        ctx.meta["identity_core_memory_snapshot"] = dict(memory_identity_core_snapshot)
        ctx.meta["identity_core_runtime_fallback"] = dict(stored_identity_core)
        ctx.meta["character_trait_defaults"] = dict(character_trait_defaults)
        persona_meta = dict(ctx.meta or {})
        persona_meta["current_user_message"] = str(_pick_value(ctx.clean_user_msg, ctx.user_msg, "") or "")
        persona_meta["route"] = str(ctx.route or "").strip().lower()
        persona_meta["context_tags"] = dict(ctx.tags or {})
        
        # Явно передаём текущие сигналы текущего turn-а
        persona_meta["emotion"] = str(
            _pick_value(ctx.tags.get("emotion"), ctx.tags.get("mood"), ctx.meta.get("emotion"), ctx.meta.get("mood"), "")
            or ""
        ).strip().lower()
        persona_meta["mood"] = str(
            _pick_value(ctx.tags.get("mood"), ctx.tags.get("emotion"), ctx.meta.get("mood"), ctx.meta.get("emotion"), "")
            or ""
        ).strip().lower()
        persona_meta["metadata_tags"] = list(
            _as_list(_pick_value(ctx.tags.get("metadata_tags"), ctx.meta.get("metadata_tags"), []))
        )
        persona_meta["emotion_intensity"] = float(_to_float(ctx.tags.get("emotion_intensity"), 0.0) or 0.0)
        persona_meta["emotion_arousal"] = float(_to_float(ctx.tags.get("emotion_arousal"), 0.0) or 0.0)
        persona_meta["same_calendar_day"] = _pick_value(ctx.tags.get("same_calendar_day"), ctx.meta.get("same_calendar_day"), False)
        persona_meta["minutes_since_previous"] = _pick_value(ctx.tags.get("minutes_since_previous"), ctx.meta.get("minutes_since_previous"), None)
        persona_meta["active_mode"] = str(
            _pick_value(ctx.state.get("active_mode"), ctx.meta.get("active_mode"), "") or ""
        ).strip().lower()
        persona_meta["is_technical"] = _to_bool(
            _pick_value(ctx.tags.get("is_technical"), ctx.meta.get("is_technical"), False),
            default=False,
        )
        
        persona_snapshot = self.persona_snapshot_builder.build(
            character_id=active_character_id,
            active_profile_snapshot=dict(ctx.state.get("active_profile_snapshot") or {}),
            identity_core_snapshot=dict(identity_core_payload),
            memory_context=self._build_persona_memory_context(dict(ctx.memory_context or {})),
            state=dict(ctx.state or {}),
            meta=persona_meta,
        )
        payload = {
            "mood": str(persona_snapshot.mood or "neutral").strip().lower() or "neutral",
            "relation_state": dict(persona_snapshot.relation_state or {}),
            "active_task": dict(persona_snapshot.active_task or {}),
            "relation_continuity": dict(persona_snapshot.relation_continuity or {}),
            "recent_user_state": dict(persona_snapshot.recent_user_state or {}),
            "user_addressing": dict(persona_snapshot.user_addressing or {}),
            "stable_traits": dict(persona_snapshot.stable_traits or {}),
            "response_bias": dict(persona_snapshot.response_bias or {}),
            "boundaries": dict(persona_snapshot.boundaries or {}),
            "emotional_handling": dict(persona_snapshot.emotional_handling or {}),
            "user_profile_hints": dict(persona_snapshot.user_profile_hints or {}),
            "active_mode": str(persona_snapshot.active_mode or "chatting").strip().lower() or "chatting",
            "debug": dict(persona_snapshot.debug or {}),
        }
        ctx.state["persona_snapshot"] = dict(payload)
        ctx.state["persona_snapshot_required"] = True
        prompt_state["persona_snapshot"] = dict(payload)
        prompt_state["persona_snapshot_required"] = True
        persona_sources = _as_dict(payload.get("debug", {}).get("sources"))
        ctx.meta["persona_snapshot_sources"] = dict(persona_sources)
        ctx.meta["persona_snapshot_required"] = True
        ctx.meta["persona_snapshot_built"] = {
            "character_id": active_character_id,
            "mood": payload["mood"],
            "active_mode": payload["active_mode"],
            "profile_keys": list(_as_list(payload.get("debug", {}).get("profile_keys"))),
            "relation_fields": sorted(dict(payload.get("relation_state") or {}).keys()),
            "active_task_present": bool(payload.get("active_task")),
            "relation_continuity_present": bool(payload.get("relation_continuity")),
            "recent_user_state_present": bool(payload.get("recent_user_state")),
            "stable_trait_fields": sorted(dict(payload.get("stable_traits") or {}).keys()),
            "boundary_fields": sorted(dict(payload.get("boundaries") or {}).keys()),
            "emotional_handling_fields": sorted(dict(payload.get("emotional_handling") or {}).keys()),
            "user_profile_hint_fields": sorted(dict(payload.get("user_profile_hints") or {}).keys()),
        }
        ctx.meta["persona_snapshot_applied"] = {
            "character_id": active_character_id,
            "relation_fields": sorted(dict(payload.get("relation_state") or {}).keys()),
            "active_task_fields": sorted(dict(payload.get("active_task") or {}).keys()),
            "relation_continuity_fields": sorted(dict(payload.get("relation_continuity") or {}).keys()),
            "recent_user_state_fields": sorted(dict(payload.get("recent_user_state") or {}).keys()),
            "applied_traits": sorted(dict(payload.get("stable_traits") or {}).keys()),
            "boundary_fields": sorted(dict(payload.get("boundaries") or {}).keys()),
            "emotional_handling_fields": sorted(dict(payload.get("emotional_handling") or {}).keys()),
            "applied_user_addressing_fields": sorted(
                [
                    key for key, value in dict(payload.get("user_addressing") or {}).items()
                    if not _is_empty_string_list_or_value(value)
                ]
            ),
            "persistent_fields": sorted(
                set(
                    list(_as_list(persona_sources.get("profile_hint_fields")))
                    + list(_as_list(persona_sources.get("relation_fields_from_persistent")))
                    + list(_as_list(persona_sources.get("relation_fields_from_identity_core")))
                    + list(_as_list(persona_sources.get("user_addressing_from_persistent")))
                    + list(_as_list(persona_sources.get("user_addressing_from_identity_core")))
                    + list(_as_list(persona_sources.get("stable_traits_from_persistent")))
                    + list(_as_list(persona_sources.get("stable_traits_from_identity_core")))
                    + list(_as_list(persona_sources.get("boundary_fields_from_identity_core")))
                    + list(_as_list(persona_sources.get("emotional_handling_fields_from_identity_core")))
                )
            ),
            "turn_local_fields": sorted(list(_as_list(persona_sources.get("turn_local_fields")))),
        }
        ctx.logs.append(
            "stage=prompt_build persona_snapshot "
            f"character={active_character_id} "
            f"profile_keys={len(list(_as_list(payload.get('debug', {}).get('profile_keys'))))}"
        )
        _emit_turn_summary(
            ctx,
            "persona_snapshot_sources",
            summary=(
                f"persistent={len(list(_as_list(persona_sources.get('persistent_profile_keys'))))} "
                f"turn_local={len(list(_as_list(persona_sources.get('turn_local_fields'))))}"
            ),
            character_id=active_character_id,
            persistent_profile_keys=sorted(list(_as_list(persona_sources.get("persistent_profile_keys")))),
            relation_fields_from_persistent=sorted(list(_as_list(persona_sources.get("relation_fields_from_persistent")))),
            relation_fields_from_identity_core=sorted(list(_as_list(persona_sources.get("relation_fields_from_identity_core")))),
            stable_traits_from_character_defaults=sorted(list(_as_list(persona_sources.get("stable_traits_from_character_defaults")))),
            stable_traits_from_persistent=sorted(list(_as_list(persona_sources.get("stable_traits_from_persistent")))),
            stable_traits_from_identity_core=sorted(list(_as_list(persona_sources.get("stable_traits_from_identity_core")))),
            stable_traits_ignored_runtime=sorted(list(_as_list(persona_sources.get("stable_traits_ignored_runtime")))),
            user_addressing_from_persistent=sorted(list(_as_list(persona_sources.get("user_addressing_from_persistent")))),
            user_addressing_from_runtime=sorted(list(_as_list(persona_sources.get("user_addressing_from_runtime")))),
            user_addressing_from_identity_core=sorted(list(_as_list(persona_sources.get("user_addressing_from_identity_core")))),
            boundary_fields_from_identity_core=sorted(list(_as_list(persona_sources.get("boundary_fields_from_identity_core")))),
            boundary_source=dict(_as_dict(persona_sources.get("boundary_source"))),
            emotional_handling_fields_from_identity_core=sorted(list(_as_list(persona_sources.get("emotional_handling_fields_from_identity_core")))),
            emotional_handling_source=dict(_as_dict(persona_sources.get("emotional_handling_source"))),
            trait_layer_priority=list(_as_list(persona_sources.get("trait_layer_priority"))),
            mood_source=str(persona_sources.get("mood_source") or ""),
            active_mode_source=str(persona_sources.get("active_mode_source") or ""),
            interaction_style_source=dict(_as_dict(persona_sources.get("interaction_style_source"))),
            response_bias_sources=sorted(list(_as_list(persona_sources.get("response_bias_sources")))),
            active_task_source=str(persona_sources.get("active_task_source") or ""),
            relation_continuity_sources=sorted(list(_as_list(persona_sources.get("relation_continuity_sources")))),
            recent_user_state_sources=sorted(list(_as_list(persona_sources.get("recent_user_state_sources")))),
            memory_input_mode=str(persona_sources.get("memory_input_mode") or ""),
            turn_local_fields=sorted(list(_as_list(persona_sources.get("turn_local_fields")))),
        )
        _emit_turn_summary(
            ctx,
            "persona_snapshot_built",
            summary=(
                f"traits={len(dict(payload.get('stable_traits') or {}))} "
                f"relation={len(dict(payload.get('relation_state') or {}))} "
                f"continuity={len(dict(payload.get('relation_continuity') or {}))} "
                f"user_state={len(dict(payload.get('recent_user_state') or {}))} "
                f"hints={len(dict(payload.get('user_profile_hints') or {}))} "
                f"boundaries={len(dict(payload.get('boundaries') or {}))} "
                f"emotional_handling={len(dict(payload.get('emotional_handling') or {}))}"
            ),
            character_id=active_character_id,
            mood=payload["mood"],
            active_mode=payload["active_mode"],
            profile_keys=sorted(list(_as_list(payload.get("debug", {}).get("profile_keys")))),
            relation_fields=sorted(dict(payload.get("relation_state") or {}).keys()),
            active_task_fields=sorted(dict(payload.get("active_task") or {}).keys()),
            relation_continuity_fields=sorted(dict(payload.get("relation_continuity") or {}).keys()),
            recent_user_state_fields=sorted(dict(payload.get("recent_user_state") or {}).keys()),
            stable_trait_fields=sorted(dict(payload.get("stable_traits") or {}).keys()),
            boundary_fields=sorted(dict(payload.get("boundaries") or {}).keys()),
            emotional_handling_fields=sorted(dict(payload.get("emotional_handling") or {}).keys()),
            user_profile_hint_fields=sorted(dict(payload.get("user_profile_hints") or {}).keys()),
        )
        _emit_turn_summary(
            ctx,
            "persona_snapshot_applied",
            summary=(
                f"traits={len(dict(payload.get('stable_traits') or {}))} "
                f"addressing={len(dict(payload.get('user_addressing') or {}))} "
                f"task={len(dict(payload.get('active_task') or {}))} "
                f"persistent={len(list(_as_list(persona_sources.get('profile_hint_fields'))))} "
                f"boundaries={len(dict(payload.get('boundaries') or {}))} "
                f"emotional_handling={len(dict(payload.get('emotional_handling') or {}))}"
            ),
            character_id=active_character_id,
            applied_traits=sorted(dict(payload.get("stable_traits") or {}).keys()),
            applied_user_addressing_fields=sorted(
                [
                    key for key, value in dict(payload.get("user_addressing") or {}).items()
                    if not _is_empty_string_list_or_value(value)
                ]
            ),
            relation_fields=sorted(dict(payload.get("relation_state") or {}).keys()),
            active_task_fields=sorted(dict(payload.get("active_task") or {}).keys()),
            relation_continuity_fields=sorted(dict(payload.get("relation_continuity") or {}).keys()),
            recent_user_state_fields=sorted(dict(payload.get("recent_user_state") or {}).keys()),
            boundary_fields=sorted(dict(payload.get("boundaries") or {}).keys()),
            emotional_handling_fields=sorted(dict(payload.get("emotional_handling") or {}).keys()),
            persistent_fields=sorted(
                set(
                    list(_as_list(persona_sources.get("profile_hint_fields")))
                    + list(_as_list(persona_sources.get("relation_fields_from_persistent")))
                    + list(_as_list(persona_sources.get("relation_fields_from_identity_core")))
                    + list(_as_list(persona_sources.get("user_addressing_from_persistent")))
                    + list(_as_list(persona_sources.get("user_addressing_from_identity_core")))
                    + list(_as_list(persona_sources.get("stable_traits_from_persistent")))
                    + list(_as_list(persona_sources.get("stable_traits_from_identity_core")))
                    + list(_as_list(persona_sources.get("boundary_fields_from_identity_core")))
                    + list(_as_list(persona_sources.get("emotional_handling_fields_from_identity_core")))
                )
            ),
            turn_local_fields=sorted(list(_as_list(persona_sources.get("turn_local_fields")))),
        )
        trace = _ensure_debug_trace(ctx)
        identity_sources = dict(_as_dict(dict(identity_core_payload).get("debug", {}).get("sources")))
        trace.identity_core = {
            "snapshot": dict(identity_core_payload),
            "memory_snapshot_input": dict(memory_identity_core_snapshot),
            "runtime_fallback_input": dict(stored_identity_core),
            "active_identity_keys": {
                "addressing": sorted(dict(identity_core_payload.get("addressing") or {}).keys()),
                "interaction_style": sorted(dict(identity_core_payload.get("interaction_style") or {}).keys()),
                "boundaries": sorted(dict(identity_core_payload.get("boundaries") or {}).keys()),
                "emotional_handling": sorted(dict(identity_core_payload.get("emotional_handling") or {}).keys()),
                "assistant_trait_baseline": sorted(dict(identity_core_payload.get("assistant_trait_baseline") or {}).keys()),
            },
            "trait_baselines": dict(identity_core_payload.get("assistant_trait_baseline") or {}),
            "sources": identity_sources,
            "source_priority": [
                "memory_identity_core",
                "runtime_identity_core",
                "active_profile",
            ],
        }
        trace.persona_snapshot = {
            **dict(payload),
            "sources": dict(persona_sources),
            "trait_overrides": dict(_as_dict(persona_sources.get("trait_overrides"))),
            "addressing_source": dict(_as_dict(persona_sources.get("addressing_source"))),
            "boundary_source": dict(_as_dict(persona_sources.get("boundary_source"))),
            "emotional_handling_source": dict(_as_dict(persona_sources.get("emotional_handling_source"))),
        }
        trace.active_profile = dict(_as_dict(ctx.state.get("active_profile_snapshot")))
        self._apply_persona_snapshot_prompt_policies(ctx=ctx, payload=payload)
        return payload

    def run(self, ctx: PipelineContext) -> PipelineContext:
        _ensure_debug_trace(ctx)
        if ctx.route != "chat":
            ctx.logs.append("stage=prompt_build skipped(route)")
            return ctx
        prompt_state = dict(ctx.state or {})
        merged_tags = _as_dict(prompt_state.get("context_tags"))
        merged_tags.update(dict(ctx.tags or {}))
        intent_alignment = _as_dict(ctx.meta.get("intent_alignment"))
        resolved_intent = str(
            _pick_value(
                intent_alignment.get("final_intent"),
                intent_alignment.get("corrected_intent"),
                ctx.meta.get("resolved_intent"),
                ctx.tags.get("resolved_intent"),
                ctx.tags.get("intent"),
                "",
            )
        ).strip()
        if resolved_intent:
            merged_tags["resolved_intent"] = resolved_intent
        if bool(intent_alignment.get("applied")):
            merged_tags["original_intent"] = str(intent_alignment.get("original_intent") or "")
            merged_tags["intent_alignment_reason"] = str(intent_alignment.get("reason") or "")
        for key in ("continuation_ref", "context_confidence", "query_effective"):
            value = ctx.meta.get(key)
            if value is None:
                continue
            text = str(value).strip()
            if text:
                merged_tags[key] = text
        prompt_state["context_tags"] = merged_tags
        if ctx.plan:
            prompt_state["plan"] = ctx.plan
        prompt_state["active_task"] = dict(_as_dict(ctx.state.get("active_task")))
        prompt_state["active_tasks"] = [dict(x) for x in list(_as_list(ctx.state.get("active_tasks"))) if isinstance(x, dict)]
        memory_context_for_prompt = _as_dict(ctx.memory_context)
        if memory_context_for_prompt:
            filtered_blocks = dict(_as_dict(memory_context_for_prompt.get("blocks")))
            if is_low_quality_session_summary(str(filtered_blocks.get("session_summary") or "").strip()):
                filtered_blocks.pop("session_summary", None)
            memory_context_for_prompt = dict(memory_context_for_prompt)
            memory_context_for_prompt["blocks"] = filtered_blocks
        memory_blocks = _as_dict(memory_context_for_prompt.get("blocks"))
        if memory_context_for_prompt:
            selected = list(_as_list(memory_context_for_prompt.get("selected")))
            if selected:
                ctx.retrieved_memories = selected
            _append_policy_rule(
                ctx.policies,
                "Retrieved memory is internal guidance, not user-facing wording. Do not quote recalled memory snippets literally unless the user explicitly asks what you remember.",
            )
            _append_policy_rule(
                ctx.policies,
                "Convert emotional or profile memory into tone adaptation, continuity, and response strategy instead of direct statements like 'you feel sad' or 'the user is frustrated'.",
            )
            _append_policy_rule(
                ctx.policies,
                "Never mention internal artifact text, retrieval metadata, memory labels, exposure modes, or system memory summaries in the final reply.",
            )
        self_facts = str(memory_blocks.get("self_facts") or "").strip()
        if self_facts:
            _append_policy_rule(
                ctx.policies,
                "If SELF_FACTS is present, treat it as the authoritative source for self-recall answers on this turn.",
            )
            _append_policy_rule(
                ctx.policies,
                "Do not override SELF_FACTS with weaker guesses from ordinary memory snippets.",
            )
        fact_expectation = _as_dict(memory_context_for_prompt.get("fact_expectation"))
        if bool(fact_expectation.get("exact_fact_required")):
            missing_predicates = [
                str(x).strip()
                for x in list(_as_list(fact_expectation.get("missing_predicates")))
                if str(x).strip()
            ]
            if missing_predicates:
                _append_policy_rule(
                    ctx.policies,
                    "If the user is asking to recall an exact environment fact and the exact semantic fact is missing from memory, say you do not see the exact fact in memory instead of guessing from similar messages.",
                )
                _append_policy_rule(
                    ctx.policies,
                    f"Missing exact memory facts for this turn: {', '.join(missing_predicates)}. Do not present them as remembered facts unless they are explicitly present in SELF_FACTS or FACT_EXPECTATION_CHECK.",
                )
        relevant_claims = str(memory_blocks.get("relevant_claims") or "").strip()
        if relevant_claims:
            _append_policy_rule(
                ctx.policies,
                "If RELEVANT_CLAIMS is present for a self-memory preference or usage question, answer from RELEVANT_CLAIMS before relying on session summary or working notes.",
            )
        if bool(ctx.meta.get("think", False)):
            rules = ctx.policies.get("rules")
            if not isinstance(rules, list):
                rules = [] if rules is None else [rules]
            rules.append(
                "Если включён thinking mode — пиши внутренние рассуждения ТОЛЬКО внутри тегов <think>...</think> "
                "и финальный ответ снаружи. Не упоминай эти теги пользователю."
            )
            ctx.policies["rules"] = rules

        web_intent = str(ctx.tags.get("web_query_intent") or "").strip().lower()
        web_category = str(
            _pick_value(
                ctx.meta.get("web_primary_category"),
                ctx.tags.get("web_primary_category"),
                _as_dict(ctx.meta.get("web_query_classification")).get("primary_category"),
                "",
            )
            or ""
        ).strip().lower()
        web_used = str(ctx.tags.get("web_used") or "").strip().lower() == "true"
        web_fresh_missing = str(ctx.tags.get("web_fresh_missing") or "").strip().lower() == "true"
        web_response_style = str(ctx.tags.get("web_response_style") or "").strip().lower()
        web_evidence_quality = _as_dict(ctx.meta.get("web_evidence_quality"))
        web_evidence_context = _as_dict(ctx.meta.get("web_evidence_context"))
        factual_response_mode = _resolve_web_factual_response_mode(
            web_intent=web_intent,
            web_category=web_category,
            web_evidence_context=web_evidence_context,
            quality=web_evidence_quality,
        )
        web_used_for_prompt = bool(web_used)
        factual_response_mode, web_used_for_prompt, self_fact_web_guard = _apply_self_fact_factual_mode_guard(
            factual_response_mode=factual_response_mode,
            web_used=web_used_for_prompt,
            memory_context=memory_context_for_prompt,
        )
        if self_fact_web_guard:
            ctx.meta["prompt_self_fact_guard"] = dict(self_fact_web_guard)
            ctx.meta["skip_factual_context_isolation"] = True
            ctx.logs.append("stage=prompt_build skip_factual_context_isolation(reason=exact_self_fact_present)")
        if factual_response_mode == _SELF_MEMORY_EXACT_MODE:
            _apply_self_memory_exact_turn_guard(ctx, prompt_state=prompt_state)
            web_used_for_prompt = False
            web_fresh_missing = False
            web_response_style = "default"
            web_evidence_quality = {}
            web_evidence_context = {}
        if factual_response_mode:
            ctx.meta["factual_response_mode"] = factual_response_mode
            ctx.tags["factual_response_mode"] = factual_response_mode
            if factual_response_mode == _SELF_MEMORY_EXACT_MODE:
                ctx.meta["web_skip_citations"] = True
                ctx.tags["self_memory_exact"] = "true"
            if web_evidence_context:
                web_evidence_context = dict(web_evidence_context)
                web_evidence_context.setdefault("factual_response_mode", factual_response_mode)
                ctx.meta["web_evidence_context"] = dict(web_evidence_context)
        else:
            ctx.meta.pop("factual_response_mode", None)
            ctx.tags.pop("factual_response_mode", None)
            ctx.tags.pop("self_memory_exact", None)
        if web_used_for_prompt and web_evidence_context:
            prompt_state["web_evidence_context"] = dict(web_evidence_context)
        else:
            prompt_state.pop("web_evidence_context", None)
        retrieved_for_prompt = list(_as_list(ctx.retrieved_memories))
        context_isolation_debug: dict[str, Any] = {}
        if web_used_for_prompt and factual_response_mode:
            memory_context_for_prompt, retrieved_for_prompt, context_isolation_debug = _isolate_factual_prompt_context(
                memory_context=memory_context_for_prompt,
                retrieved_memories=retrieved_for_prompt,
                web_intent=factual_response_mode,
            )
            if context_isolation_debug:
                ctx.meta["prompt_context_isolation"] = dict(context_isolation_debug)
                ctx.logs.append(
                    "stage=prompt_build factual_context_isolation "
                    f"intent={factual_response_mode} kept_blocks={len(list(_as_list(context_isolation_debug.get('included_memory_blocks'))))} "
                    f"dropped_blocks={len(list(_as_list(context_isolation_debug.get('dropped_memory_blocks'))))} "
                    f"kept_memories={int(_to_int(context_isolation_debug.get('included_retrieved_memories'), 0) or 0)} "
                    f"dropped_memories={len(list(_as_list(context_isolation_debug.get('dropped_retrieved_memories'))))}"
                )
                emitter = ctx.meta.get("emit_web_trace_event")
                if callable(emitter):
                    emitter("factual_context_isolation", dict(context_isolation_debug))
        elif factual_response_mode == _SELF_MEMORY_EXACT_MODE:
            memory_context_for_prompt, retrieved_for_prompt, context_isolation_debug = _isolate_self_memory_exact_context(
                memory_context=memory_context_for_prompt,
                retrieved_memories=retrieved_for_prompt,
            )
            if context_isolation_debug:
                ctx.meta["prompt_context_isolation"] = dict(context_isolation_debug)
                ctx.logs.append(
                    "stage=prompt_build self_memory_exact_context_isolation "
                    f"kept_blocks={len(list(_as_list(context_isolation_debug.get('included_memory_blocks'))))} "
                    f"dropped_blocks={len(list(_as_list(context_isolation_debug.get('dropped_memory_blocks'))))} "
                    f"kept_memories={int(_to_int(context_isolation_debug.get('included_retrieved_memories'), 0) or 0)} "
                    f"dropped_memories={len(list(_as_list(context_isolation_debug.get('dropped_retrieved_memories'))))}"
                )
        if memory_context_for_prompt:
            ctx.memory_context = dict(memory_context_for_prompt)
            ctx.state["memory_context"] = dict(memory_context_for_prompt)
            prompt_state["memory_context"] = dict(memory_context_for_prompt)
            memory_blocks = _as_dict(memory_context_for_prompt.get("blocks"))
            if str(memory_blocks.get("relevant_claims") or "").strip() and str(memory_blocks.get("session_summary") or "").strip():
                memory_blocks = dict(memory_blocks)
                memory_blocks.pop("session_summary", None)
                claim_guard = {
                    "applied": True,
                    "reason": "self_memory_claim_prefers_relevant_claims",
                }
                ctx.memory_context["blocks"] = dict(memory_blocks)
                ctx.memory_context["self_memory_claim_guard"] = dict(claim_guard)
                ctx.state["memory_context"] = dict(ctx.memory_context)
                prompt_state["memory_context"] = dict(ctx.memory_context)
                ctx.meta["self_memory_claim_guard"] = dict(claim_guard)
        else:
            ctx.memory_context = {}
            ctx.state.pop("memory_context", None)
            prompt_state.pop("memory_context", None)
            memory_blocks = {}
        prompt_state.pop("long_summary", None)
        prompt_state.pop("dialog_summary", None)
        prompt_state.pop("last_tool_result", None)
        
        # Summary больше не используется как источник continuity.
        # Это только fallback кеш для prompt_engine.
        # Не записываем summary_hint в prompt_state["long_summary"]/["dialog_summary"].
        # prompt_engine сам решит, нужен ли summary как fallback.
        summary_hint = sanitize_session_summary_text(memory_blocks.get("session_summary") or "")
        # Summary теперь только в memory_blocks["session_summary"] для prompt_engine
        # Но НЕ в prompt_state для continuity
        
        tool_hint = str(memory_blocks.get("active_tool_state") or "").strip()
        if tool_hint:
            prompt_state["last_tool_result"] = tool_hint
        if factual_response_mode == _SELF_MEMORY_EXACT_MODE:
            _append_policy_rule(
                ctx.policies,
                "This turn is a self_memory_exact recall request. Answer directly from SELF_FACTS.",
            )
            _append_policy_rule(
                ctx.policies,
                "Do not use WEB_EVIDENCE, external sources, web-only rules, or source-style citations for this turn.",
            )
            _append_policy_rule(
                ctx.policies,
                "If SELF_FACTS does not contain the exact requested detail, say you do not see the exact fact in memory instead of guessing.",
            )
        elif bool(intent_alignment.get("applied")) and factual_response_mode:
            _append_policy_rule(
                ctx.policies,
                f"This turn is a factual {factual_response_mode} request aligned from web classification, not open-ended chat. Keep the answer direct and evidence-led.",
            )

        if web_used_for_prompt and factual_response_mode:
            _append_policy_rule(
                ctx.policies,
                "For time-sensitive web answers, respond directly with facts and avoid rhetorical openers like 'Ах, ты опять...' or similar chatter.",
            )
            _append_policy_rule(
                ctx.policies,
                "Use fetched web evidence and include source domain and fetch/publish time; if sources disagree, report a range.",
            )
            _append_policy_rule(
                ctx.policies,
                f"For this {factual_response_mode} answer, ignore unrelated memory, tool fragments, and stale cross-category context. Use only category-matching WEB_EVIDENCE and matching factual context.",
            )
            _append_policy_rule(
                ctx.policies,
                "For web-backed factual answers, prefer compact evidence-led wording over conversational speculation or storytelling.",
            )

        if web_used_for_prompt and factual_response_mode in {"price", "historical_factual", "latest_factual", "weather"}:
            _append_policy_rule(
                ctx.policies,
                "For factual web-backed answers, state only the value, date, range, or conclusion that is directly supported by WEB_EVIDENCE, then name the source.",
            )
            _append_policy_rule(
                ctx.policies,
                "Do not fill evidence gaps from memory, priors, or style. If the exact factual claim is not reliably supported, say that it could not be reliably confirmed.",
            )

        if web_used_for_prompt and factual_response_mode == "fx_rate":
            _append_policy_rule(
                ctx.policies,
                "For FX answers, use a compact factual format: one short line per confirmed pair, then rate type, then source.",
            )
            _append_policy_rule(
                ctx.policies,
                "For FX answers, do not answer in free-form chatty prose. Do not guess, average, or merge incompatible rate types. Only state values directly supported by WEB_EVIDENCE.",
            )
            _append_policy_rule(
                ctx.policies,
                "If the exact FX value is not reliably confirmed, explicitly say that the exact value could not be reliably confirmed instead of inventing a number.",
            )

        if web_used_for_prompt:
            _append_policy_rule(
                ctx.policies,
                "When web evidence is used, include compact citations (domain + date/time) without turning the answer into a verbose report.",
            )
            _append_policy_rule(
                ctx.policies,
                "Live web lookup already executed for this turn. Do not claim lack of internet/web access or say that you cannot check the data; use WEB_EVIDENCE as the factual source.",
            )
        if web_used_for_prompt and factual_response_mode and bool(web_evidence_context.get("cautious_synthesis")):
            _append_policy_rule(
                ctx.policies,
                "WEB_EVIDENCE for this turn is weak or conflicting. Do not invent an exact number/date from memory or priors. If the precise value is not consistently supported, explicitly say that the exact value could not be reliably confirmed.",
            )
            _append_policy_rule(
                ctx.policies,
                "For numeric or date answers under cautious synthesis, prefer a careful confirmation-failure statement over a confident exact claim.",
            )

        if web_fresh_missing:
            _append_policy_rule(
                ctx.policies,
                "Time-sensitive data could not be confirmed from live web sources. Do not fabricate current numbers; explicitly state that verification failed and ask to retry.",
            )
            ctx.tags["web_guardrail"] = "fresh_missing_no_fabrication"
            tags_map = _as_dict(prompt_state.get("context_tags"))
            tags_map["web_guardrail"] = "fresh_missing_no_fabrication"
            prompt_state["context_tags"] = tags_map
        elif web_used_for_prompt and factual_response_mode in {"fx_rate", "weather"}:
            _append_policy_rule(
                ctx.policies,
                "When answering FX/weather requests, rely on fetched web evidence, include source domain and timestamp, and report a range if sources disagree.",
            )
            ctx.tags["web_guardrail"] = "cite_source_and_time"
            tags_map = _as_dict(prompt_state.get("context_tags"))
            tags_map["web_guardrail"] = "cite_source_and_time"
            prompt_state["context_tags"] = tags_map
        if factual_response_mode:
            tags_map = _as_dict(prompt_state.get("context_tags"))
            tags_map["factual_response_mode"] = factual_response_mode
            prompt_state["context_tags"] = tags_map
        if web_response_style:
            tags_map = _as_dict(prompt_state.get("context_tags"))
            tags_map["web_response_style"] = web_response_style
            prompt_state["context_tags"] = tags_map
        same_day = _to_bool(
            _pick_value(ctx.meta.get("same_calendar_day"), ctx.tags.get("same_calendar_day"), False),
            default=False,
        )
        minutes_since_previous = _to_float(
            _pick_value(ctx.meta.get("minutes_since_previous"), ctx.tags.get("minutes_since_previous"), None),
            None,
        )
        if same_day and minutes_since_previous is not None and float(minutes_since_previous) < 180.0:
            _append_policy_rule(
                ctx.policies,
                "If the previous user turn was within the same calendar day and within 180 minutes, do not phrase it as 'yesterday'/'day before yesterday'; use exact or neutral timing.",
            )
        # Раннее follow-up policy удалено — continuity определяется после построения persona snapshot
        ctx.retrieved_memories = list(retrieved_for_prompt)
        self._build_persona_snapshot_payload(ctx=ctx, prompt_state=prompt_state)
        ctx.prompt_pack = self.character_runtime.build(
            state=prompt_state,
            user_msg=ctx.clean_user_msg,
            retrieved_memories=retrieved_for_prompt,
            traits=ctx.traits,
            policies=ctx.policies,
        )
        if memory_context_for_prompt:
            ctx.prompt_pack = _apply_memory_context_to_prompt_pack(
                ctx.prompt_pack,
                memory_context=memory_context_for_prompt,
                selected_memories=retrieved_for_prompt,
            )
        ctx.prompt_pack = _apply_web_evidence_to_prompt_pack(
            ctx.prompt_pack,
            web_evidence_context=(web_evidence_context if web_used_for_prompt else {}),
        )
        emitter = ctx.meta.get("emit_web_trace_event")
        if callable(emitter):
            web_sources = [x for x in list(_as_list(web_evidence_context.get("sources"))) if isinstance(x, dict)]
            domains = sorted({str(_as_dict(x).get("domain") or "").strip().lower() for x in list(web_sources) if str(_as_dict(x).get("domain") or "").strip()})
            emitter(
                "web_context_injected",
                {
                    "retrieved_total": len(list(retrieved_for_prompt)),
                    "web_context_count": len(list(web_sources)),
                    "domains": domains,
                    "citations": len(list(_as_list(web_evidence_context.get("compact_citations")))),
                    "conflicting_sources": bool(web_evidence_context.get("conflicting_sources")),
                    "context_isolation": dict(context_isolation_debug or {}),
                },
            )
        ctx.state["prompt_persona_block"] = {
            "text": str(_as_dict(getattr(ctx.prompt_pack, "blocks", {})).get("persona") or "").strip(),
            "token_estimate": int(_to_int(_as_dict(getattr(ctx.prompt_pack, "token_usage", {})).get("persona"), 0) or 0),
            "active_character_id": str(
                _pick_value(
                    ctx.prompt_sections.get("active_character_id"),
                    prompt_state.get("active_character_id"),
                    prompt_state.get("active_personality_id"),
                    ctx.state.get("active_character_id"),
                    "",
                )
                or ""
            ).strip(),
        }
        ctx.logs.append("stage=prompt_build")
        return ctx


class PromptEngineStage(PipelineStage):
    name = "prompt_engine"

    def __init__(self, prompt_engine: PromptEngine):
        self.prompt_engine = prompt_engine

    def run(self, ctx: PipelineContext) -> PipelineContext:
        _ensure_debug_trace(ctx)
        if ctx.route != "chat":
            ctx.logs.append("stage=prompt_engine skipped(route)")
            return ctx
        if ctx.prompt_pack is None:
            ctx.logs.append("stage=prompt_engine skipped(no_pack)")
            return ctx

        engine_state = dict(ctx.state or {})
        engine_state["context_tags"] = dict(ctx.tags or {})
        if ctx.memory_context:
            engine_state["memory_context"] = dict(ctx.memory_context)
        current_web_context = _as_dict(ctx.meta.get("web_evidence_context"))
        current_web_used = _to_bool(_pick_value(ctx.tags.get("web_used"), ctx.meta.get("web_used"), False), default=False)
        if current_web_used and current_web_context:
            engine_state["web_evidence_context"] = dict(current_web_context)
        else:
            engine_state.pop("web_evidence_context", None)

        result = self.prompt_engine.compose(
            prompt_pack=ctx.prompt_pack,
            state=engine_state,
            traits=ctx.traits,
            policies=ctx.policies,
        )
        ctx.prompt_messages = list(result.messages)
        ctx.prompt_sections = dict(result.sections)
        memory_context = _as_dict(ctx.memory_context)
        memory_blocks = _as_dict(memory_context.get("blocks"))
        prompt_memory_text = self.prompt_engine._build_memory_retrieval_block(
            blocks=dict(getattr(ctx.prompt_pack, "blocks", {}) or {}),
            memory_blocks=memory_blocks,
            state_map=engine_state,
        )
        prompt_active_task_text = self.prompt_engine._render_active_task_block(
            _as_dict(ctx.state.get("active_task"))
        )
        prompt_persona_text = str(_as_dict(getattr(ctx.prompt_pack, "blocks", {})).get("persona") or "").strip()
        token_estimate = int(
            _to_int(
                _pick_value(
                    ctx.prompt_sections.get("budget_total_tokens"),
                    _as_dict(getattr(ctx.prompt_pack, "token_usage", {})).get("full_prompt"),
                    0,
                ),
                0,
            )
            or 0
        )
        ctx.state["prompt_memory_block"] = {
            "text": str(prompt_memory_text or "").strip(),
            "block_keys": sorted(str(x) for x in list(memory_blocks.keys()) if str(x).strip()),
            "selected_count": int(len(list(ctx.retrieved_memories or []))),
        }
        ctx.state["prompt_persona_block"] = {
            "text": prompt_persona_text,
            "active_character_id": str(ctx.prompt_sections.get("active_character_id") or "").strip(),
            "token_estimate": int(_to_int(_as_dict(getattr(ctx.prompt_pack, "token_usage", {})).get("persona"), 0) or 0),
        }
        ctx.state["prompt_active_task_block"] = {
            "text": str(prompt_active_task_text or "").strip(),
            "status": str(_as_dict(ctx.state.get("active_task")).get("status") or "").strip(),
        }
        ctx.state["prompt_token_estimate"] = token_estimate
        included_memory_blocks = _trace_prompt_block_titles(prompt_memory_text)
        omitted_memory_blocks = _trace_prompt_omitted_blocks(
            memory_blocks=memory_blocks,
            active_task_block=prompt_active_task_text,
            included_titles=included_memory_blocks,
            context_isolation=_as_dict(ctx.meta.get("prompt_context_isolation")),
        )
        ctx.state["prompt_memory_block"] = {
            **dict(ctx.state.get("prompt_memory_block") or {}),
            "included_blocks": list(included_memory_blocks or []),
            "omitted_blocks": [dict(x) for x in list(omitted_memory_blocks or []) if isinstance(x, dict)],
        }
        trace = _ensure_debug_trace(ctx)
        trace.prompt_pack = {
            "memory_block": dict(ctx.state.get("prompt_memory_block") or {}),
            "persona_block": dict(ctx.state.get("prompt_persona_block") or {}),
            "active_task_block": dict(ctx.state.get("prompt_active_task_block") or {}),
            "included_memory_blocks": list(included_memory_blocks or []),
            "omitted_memory_blocks": [dict(x) for x in list(omitted_memory_blocks or []) if isinstance(x, dict)],
            "token_estimate": token_estimate,
            "section_keys": sorted(str(x) for x in list(ctx.prompt_sections.keys()) if str(x).strip()),
        }
        ctx.logs.append(f"stage=prompt_engine messages={len(ctx.prompt_messages)}")
        section_names = [str(x).strip() for x in list(ctx.prompt_sections.keys()) if str(x).strip()]
        token_usage = dict(getattr(ctx.prompt_pack, "token_usage", {}) or {})
        web_context = _as_dict(ctx.meta.get("web_evidence_context"))
        _emit_turn_summary(
            ctx,
            "prompt_summary",
            summary=(
                f"messages={len(ctx.prompt_messages)} sections={len(section_names)} "
                f"memory_hits={len(_as_list(ctx.retrieved_memories))} "
                f"web_sources={len(_as_list(web_context.get('sources')))}"
            ),
            route=str(ctx.route or ""),
            message_count=int(len(ctx.prompt_messages)),
            section_names=section_names,
            prompt_tokens=int(_to_int(token_usage.get("prompt_tokens"), 0) or 0),
            total_tokens=int(_to_int(token_usage.get("total_tokens"), 0) or 0),
            memory_hits=int(len(_as_list(ctx.retrieved_memories))),
            web_sources=int(len(_as_list(web_context.get("sources")))),
            has_web_context=bool(web_context),
        )
        return ctx


class GenerateStage(PipelineStage):
    name = "generate"
    _build_persona_snapshot_payload = PromptBuildStage._build_persona_snapshot_payload
    _build_persona_memory_context = staticmethod(PromptBuildStage._build_persona_memory_context)
    _apply_persona_snapshot_prompt_policies = PromptBuildStage._apply_persona_snapshot_prompt_policies

    def __init__(
        self,
        provider: LLMProviderBase,
        character_runtime: CharacterRuntime,
        memory_core: Any | None = None,
        studio_generator: StudioGenerator | None = None,
    ):
        self.provider = provider
        self.character_runtime = character_runtime
        self.character_engine = character_runtime
        self.memory_core = memory_core
        self.persona_snapshot_builder = PersonaSnapshotBuilder()
        self.identity_core_builder = IdentityCoreBuilder()
        self.studio_generator = studio_generator or StudioGenerator()
        self._studio_sessions: dict[str, dict[str, Any]] = {}

    def _studio_key(self, ctx: PipelineContext) -> str:
        return str(
            _pick(
                ctx.meta.get("conversation_id"),
                ctx.state.get("conversation_id"),
                "default",
            )
        ).strip().lower() or "default"

    def _studio_state(self, ctx: PipelineContext) -> dict[str, Any]:
        local = dict(ctx.state.get(StudioGenerator.KEY) or {})
        if local:
            return local
        return dict(self._studio_sessions.get(self._studio_key(ctx), {}) or {})

    def _save_studio_state(self, ctx: PipelineContext, row: dict[str, Any]) -> None:
        payload = dict(row or {})
        ctx.state[StudioGenerator.KEY] = payload
        ctx.structured_output[StudioGenerator.KEY] = payload
        key = self._studio_key(ctx)
        if self.studio_generator.is_active({StudioGenerator.KEY: payload}):
            self._studio_sessions[key] = payload
        else:
            self._studio_sessions.pop(key, None)

    def is_studio_active(self, *, conversation_id: str = "", state: dict[str, Any] | None = None) -> bool:
        state_map = _as_dict(state)
        local = _as_dict(state_map.get(StudioGenerator.KEY))
        if self.studio_generator.is_active({StudioGenerator.KEY: local}):
            return True
        key = str(_pick(conversation_id, state_map.get("conversation_id"), "default")).strip().lower() or "default"
        cached = _as_dict(self._studio_sessions.get(key))
        return bool(self.studio_generator.is_active({StudioGenerator.KEY: cached}))

    @staticmethod
    def _scopes_map(ctx: PipelineContext) -> dict[str, Any]:
        row = dict(ctx.state.get("command_scopes") or {})
        out: dict[str, Any] = {}
        for key in ("chat", "studio"):
            out[key] = dict(row.get(key) or {})
        return out

    @staticmethod
    def _scope_get(ctx: PipelineContext, scope: str, key: str, default=None):
        scopes = dict(ctx.state.get("command_scopes") or {})
        scoped = dict(scopes.get(str(scope or "").strip().lower()) or {})
        return scoped.get(str(key or "").strip(), default)

    def _scope_set(self, ctx: PipelineContext, scope: str, **updates) -> None:
        scope_key = str(scope or "").strip().lower() or "chat"
        scopes = self._scopes_map(ctx)
        target = dict(scopes.get(scope_key) or {})
        target.update({str(k): v for k, v in dict(updates or {}).items() if str(k).strip()})
        scopes[scope_key] = target
        ctx.state["command_scopes"] = scopes
        ctx.structured_output["command_scopes"] = dict(scopes)
        ctx.memory_ops.append({"op": "state_scoped_settings", "value": dict(scopes)})

    @staticmethod
    def _apply_studio_telemetry(ctx: PipelineContext, studio_state: dict[str, Any]) -> None:
        root = _as_dict(studio_state)
        llm = _as_dict(root.get("last_llm"))
        thinking = str(llm.get("thinking") or "").strip()
        if thinking:
            ctx.thinking = thinking
        prompt_eval = _to_int(llm.get("prompt_eval_count"), None)
        eval_count = _to_int(llm.get("eval_count"), None)
        total_tokens = _to_int(llm.get("total_tokens"), None)
        if prompt_eval is None and eval_count is None and total_tokens is None:
            return
        stats = _as_dict(ctx.stats)
        if prompt_eval is not None:
            stats["prompt_eval_count"] = int(prompt_eval)
        if eval_count is not None:
            stats["eval_count"] = int(eval_count)
        if total_tokens is not None:
            stats["total_tokens"] = int(total_tokens)
        else:
            stats["total_tokens"] = int((stats.get("prompt_eval_count") or 0) + (stats.get("eval_count") or 0))
        model_name = str(llm.get("model") or "").strip()
        if model_name and not str(stats.get("served_model") or "").strip():
            stats["served_model"] = model_name
        ctx.stats = stats

    def run(self, ctx: PipelineContext) -> PipelineContext:
        if ctx.route == "system_event":
            self._handle_system_event(ctx)
            return ctx

        if ctx.route == "command":
            try:
                handled = self._handle_internal_command(ctx)
            except Exception as exc:
                ctx.text = f"Command failed: {type(exc).__name__}"
                ctx.logs.append(f"stage=generate command_error={type(exc).__name__}")
                ctx.errors.append(f"command:{type(exc).__name__}:{exc}")
                return ctx
            if handled:
                return ctx
            ctx.text = "Unknown command. Use /help"
            ctx.logs.append("stage=generate command=unknown")
            return ctx

        # Unified global studio generator (specs + character branches) in chat turns.
        studio_state = self._studio_state(ctx)
        if ctx.route == "chat" and self.studio_generator.is_active({StudioGenerator.KEY: studio_state}):
            reply = self.studio_generator.ingest(
                {StudioGenerator.KEY: studio_state},
                ctx.clean_user_msg or ctx.user_msg,
                provider=self.provider,
                model=_pick(ctx.meta.get("model"), ctx.state.get("model"), ctx.policies.get("model")),
                command_alias="",
            )
            self._save_studio_state(ctx, dict(reply.state or {}))
            self._apply_studio_telemetry(ctx, dict(reply.state or {}))
            ctx.text = str(reply.text or "")
            if reply.memory_ops:
                ctx.memory_ops.extend(list(reply.memory_ops or []))
            if reply.payload:
                ctx.structured_output["studio"] = dict(reply.payload)
            ctx.logs.append("stage=generate route=chat studio_generator=1")
            return ctx

        if ctx.route == "chat" and bool(ctx.meta.get("web_guardrail_local_reply")):
            intent = str(ctx.meta.get("web_query_intent") or ctx.tags.get("web_query_intent") or "").strip().lower()
            if intent == "fx_rate":
                ctx.text = (
                    "Не смогла подтвердить актуальный курс USD/UAH из веб-источников прямо сейчас. "
                    "Проверь ещё раз через /web курс доллара в Украине."
                )
            elif intent == "weather":
                ctx.text = (
                    "Не смогла подтвердить актуальную погоду из веб-источников прямо сейчас. "
                    "Проверь ещё раз через /web погода в Киеве."
                )
            else:
                ctx.text = (
                    "Не смогла подтвердить актуальные данные из веб-источников прямо сейчас. "
                    "Повтори запрос через /web."
                )
            ctx.logs.append("stage=generate route=chat web_guardrail=local_reply")
            return ctx

        if ctx.route == "chat":
            clarifying = str(ctx.meta.get("web_clarifying_question") or "").strip()
            if clarifying:
                ctx.text = clarifying
                ctx.logs.append("stage=generate route=chat web_guardrail=clarifying_question")
                return ctx

        req = self._build_request(ctx)
        use_agent_loop = bool(_to_bool(req.metadata.get("agent_loop"), default=False))
        stream_answer_cb = ctx.meta.get("stream_on_answer_chunk")
        stream_thinking_cb = ctx.meta.get("stream_on_thinking_chunk")

        # Streaming теперь работает ВСЕГДА, даже с agent_loop
        # agent_loop будет вызывать on_answer/on_thinking callbacks во время генерации
        use_stream = bool(callable(stream_answer_cb) or callable(stream_thinking_cb))

        if use_stream and not use_agent_loop:
            # Простой streaming без agent_loop
            stream_result = self._generate_stream(
                ctx,
                req,
                on_answer=stream_answer_cb,
                on_thinking=stream_thinking_cb,
            )
            if stream_result is not None:
                text_out, thinking_out, model_name, stream_stats = stream_result
                ctx.raw_output = text_out
                ctx.text = text_out
                ctx.thinking = thinking_out
                stats = {
                    "served_model": str(model_name or req.model or ""),
                    "answer_ms": float(_to_float(_as_dict(stream_stats).get("answer_ms"), 0.0) or 0.0),
                    "prompt_eval_count": 0,
                    "eval_count": 0,
                    "total_tokens": 0,
                    "streaming": True,
                    "verbose_enabled": bool(_to_bool(ctx.meta.get("verbose"), default=False)),
                }
                stats.update(_as_dict(stream_stats))
                ctx.stats = stats
                ctx.logs.append(
                    f"stage=generate route={ctx.route} model={ctx.stats.get('served_model')} stream=1"
                )
                return ctx

            # None здесь означает только одно:
            # provider.stream() вообще недоступен.
            ctx.logs.append("stage=generate stream=provider_missing fallback_to_generate")
        elif use_stream and use_agent_loop:
            ctx.logs.append("stage=generate stream=enabled with agent_loop")
        elif not use_stream:
            ctx.logs.append("stage=generate stream=disabled")

        agent_tool_calls = 0
        agent_passes = 1
        if use_agent_loop:
            resp, agent_tool_calls, agent_passes = self._generate_with_agent_loop(ctx, req)
        else:
            # Приоритет для основной модели — захватываем turn перед генерацией
            try:
                from llm.priority_manager import get_priority_manager, LLMPriorityManager
                manager = get_priority_manager()
                if manager:
                    # Основная модель имеет приоритет 0 (highest)
                    manager.acquire_turn(LLMPriorityManager.PRIORITY_MAIN, timeout=30.0)
            except Exception:
                pass  # Игнорируем ошибки priority manager
            
            resp = self.provider.generate(req)
            
            # Освобождаем приоритет после генерации
            try:
                from llm.priority_manager import get_priority_manager, LLMPriorityManager
                manager = get_priority_manager()
                if manager:
                    manager.release_turn(LLMPriorityManager.PRIORITY_MAIN)
            except Exception:
                pass  # Игнорируем ошибки priority manager
        ctx.raw_output = str(resp.text or "")
        ctx.text = ctx.raw_output
        ctx.thinking = str(getattr(resp, "thinking", "") or "")
        if resp.tool_calls and not use_agent_loop:
            ctx.tool_calls = [_tool_call_to_dict(x) for x in list(resp.tool_calls or [])]
        stats = {
            "served_model": str(resp.model or req.model or ""),
            "answer_ms": float(resp.timings.latency_ms or 0.0),
            "prompt_eval_count": int(resp.usage.prompt_tokens or 0),
            "eval_count": int(resp.usage.completion_tokens or 0),
            "total_tokens": int(resp.usage.total_tokens or 0),
            "agent_loop": bool(use_agent_loop),
            "agent_tool_calls": int(agent_tool_calls),
            "agent_passes": int(agent_passes),
            "verbose_enabled": bool(_to_bool(ctx.meta.get("verbose"), default=False)),
        }
        ctx.stats = self._merge_llm_stats(
            stats,
            usage=getattr(resp, "usage", None),
            timings=getattr(resp, "timings", None),
        )
        if use_agent_loop:
            ctx.logs.append(
                f"stage=generate route={ctx.route} model={ctx.stats.get('served_model')} "
                f"agent_loop=1 tool_calls={agent_tool_calls} passes={agent_passes}"
            )
        else:
            ctx.logs.append(f"stage=generate route={ctx.route} model={ctx.stats.get('served_model')}")
        return ctx

    def _generate_with_agent_loop(self, ctx: PipelineContext, req: LLMRequest) -> tuple[LLMResponse, int, int]:
        messages = list(req.messages or [])
        max_tool_calls = _agent_loop_tool_limit(ctx)
        executed_calls = 0
        pass_count = 0
        final_response: LLMResponse | None = None
        iteration_rows: list[dict[str, Any]] = []
        tool_rows: list[dict[str, Any]] = []
        
        # Streaming callbacks
        stream_answer_cb = ctx.meta.get("stream_on_answer_chunk")
        stream_thinking_cb = ctx.meta.get("stream_on_thinking_chunk")
        use_streaming = bool(callable(stream_answer_cb) or callable(stream_thinking_cb))

        # Memory gate: ЖЁСТКОЕ правило — если срабатывает, модель ОБЯЗАНА вызвать memory_retrieve
        # Добавляем жёсткую инструкцию в system message
        memory_gate_triggered = _memory_gate_should_trigger(ctx)
        if memory_gate_triggered:
            ctx.logs.append("memory_gate HARD trigger — memory_retrieve required before answer")
            # Жёсткое требование — прямой ответ запрещён до memory pass
            gate_instruction = (
                "HARD CONSTRAINT: This query depends on prior context or user-specific information. "
                "You MUST call memory_retrieve tool BEFORE providing any answer. "
                "Direct answer is FORBIDDEN until memory retrieval completes. "
                "If memory_retrieve returns no relevant information, explicitly state that you cannot answer without more context."
            )
            # Вставляем после первого system message
            insert_index = 0
            for i, msg in enumerate(messages):
                if msg.role == "system":
                    insert_index = i + 1
                else:
                    break
            messages.insert(insert_index, Message(role="system", content=gate_instruction))

        while True:
            pass_count += 1
            remaining = max(0, int(max_tool_calls) - int(executed_calls))
            active_tools = list(req.tools or []) if remaining > 0 else []
            loop_messages = list(messages)
            memory_reasoning_snapshot: dict[str, Any] = {}
            if executed_calls > 0:
                loop_messages, memory_reasoning_snapshot = self._inject_memory_reasoning_check(
                    ctx,
                    messages=loop_messages,
                )
            loop_req = replace(
                req,
                messages=list(loop_messages),
                tools=active_tools,
                metadata={
                    **dict(req.metadata or {}),
                    "agent_loop_iteration": int(pass_count),
                    "agent_loop_remaining_tools": int(remaining),
                    "reason_with_memory": bool(memory_reasoning_snapshot),
                    "memory_gate_triggered": memory_gate_triggered,
                },
            )
            iteration_row = {
                "iteration": int(pass_count),
                "remaining_tools": int(remaining),
                "tools_enabled": [str(tool.name or "") for tool in list(active_tools or []) if str(tool.name or "").strip()],
                "reason_with_memory": bool(memory_reasoning_snapshot),
                "memory_reasoning_sections": sorted(memory_reasoning_snapshot.keys()),
                "memory_gate_triggered": memory_gate_triggered,
            }
            if memory_reasoning_snapshot:
                ctx.logs.append(
                    "stage=generate agent_loop memory_reasoning=1 "
                    f"sections={','.join(sorted(memory_reasoning_snapshot.keys()))}"
                )
            
            # Streaming или обычный generate
            if use_streaming:
                # Streaming через provider.stream()
                response = self._generate_with_agent_loop_streaming(
                    ctx,
                    loop_req,
                    stream_answer_cb,
                    stream_thinking_cb,
                    emit_answer_live=not (memory_gate_triggered and pass_count == 1),
                )
            else:
                response = self.provider.generate(loop_req)
            if not response.tool_calls or not active_tools:
                iteration_row["tool_calls_requested"] = int(len(list(response.tool_calls or [])))
                iteration_row["executed_tool_calls"] = 0
                iteration_rows.append(iteration_row)
                
                # PROGRAMMATIC ENFORCEMENT: если memory gate сработал, но модель не вызвала tool,
                # принудительно вызываем memory_retrieve вместо ответа модели
                if memory_gate_triggered and pass_count == 1:
                    ctx.logs.append("memory_gate enforcement — model ignored tool call, forcing memory_retrieve")
                    # Принудительно вызываем memory_retrieve с дефолтным query
                    forced_tool_call = ToolCall(
                        id="forced_memory_retrieve",
                        name=_MEMORY_TOOL_NAME,
                        arguments={
                            "mode": "context",
                            "topic_hints": [str(ctx.clean_user_msg or "")[:50]],
                            "time_hint": "recent",
                            "sources": ["messages", "facts", "episodes"],
                            "top_k": 5,
                        },
                    )
                    # Выполняем принудительный tool call
                    tool_result_json, _ = self._execute_memory_retrieve_tool(ctx, forced_tool_call)
                    executed_calls += 1
                    tool_rows.append({"tool_call": _tool_call_to_dict(forced_tool_call), "result": tool_result_json})
                    # Добавляем результат в messages и продолжаем loop
                    messages.append(Message(role="assistant", content="", tool_calls=[forced_tool_call]))
                    messages.append(Message(role="tool", content=tool_result_json, tool_call_id=forced_tool_call.id))
                    # Перестраиваем memory_native_state с учётом новых данных
                    native_state = {
                        "hits": ctx.memory_context.get("hits", []) if ctx.memory_context else [],
                        "citations": ctx.memory_context.get("citations", []) if ctx.memory_context else [],
                    }
                    ctx.state["memory_native_state"] = native_state
                    ctx.logs.append("memory_native_state rebuilt after forced retrieval")
                    # Продолжаем loop для генерации ответа с memory context
                    continue
                
                final_response = response
                break

            tool_calls = list(response.tool_calls or [])[:remaining]
            if not tool_calls:
                iteration_row["tool_calls_requested"] = int(len(list(response.tool_calls or [])))
                iteration_row["executed_tool_calls"] = 0
                iteration_rows.append(iteration_row)
                final_response = response
                break

            if len(tool_calls) < len(list(response.tool_calls or [])):
                iteration_row["tool_calls_truncated"] = int(len(list(response.tool_calls or [])) - len(tool_calls))
                ctx.logs.append(
                    f"stage=generate agent_loop tool_calls_truncated={len(list(response.tool_calls or [])) - len(tool_calls)}"
                )
            messages.append(
                Message(
                    role="assistant",
                    content=str(response.text or ""),
                    tool_calls=list(tool_calls),
                )
            )
            executed_names: list[str] = []
            for call in tool_calls:
                content, is_error = self._execute_agent_tool_call(ctx, call)
                messages.append(
                    Message(
                        role="tool",
                        name=str(call.name or ""),
                        tool_call_id=str(call.id or ""),
                        content=content,
                    )
                )
                executed_names.append(str(call.name or ""))
                tool_rows.append(
                    {
                        "iteration": int(pass_count),
                        "tool": str(call.name or ""),
                        "call_id": str(call.id or ""),
                        "ok": bool(not is_error),
                        "result_preview": _preview_text(content, 180),
                    }
                )
                ctx.logs.append(
                    f"stage=generate agent_loop tool={call.name} ok={int(not is_error)} call_id={call.id}"
                )
            iteration_row["tool_calls_requested"] = int(len(list(response.tool_calls or [])))
            iteration_row["executed_tool_calls"] = int(len(tool_calls))
            iteration_row["executed_tools"] = executed_names
            iteration_rows.append(iteration_row)
            executed_calls += len(tool_calls)

        if final_response is None:
            raise RuntimeError("agent loop finished without final response")
        ctx.state["agent_loop_trace"] = {
            "agent_loop": True,
            "agent_tool_calls": int(executed_calls),
            "agent_passes": int(pass_count),
            "iterations": [dict(x) for x in list(iteration_rows or []) if isinstance(x, dict)],
            "executed_tools": [dict(x) for x in list(tool_rows or []) if isinstance(x, dict)],
        }
        return final_response, executed_calls, pass_count

    def _inject_memory_reasoning_check(
        self,
        ctx: PipelineContext,
        *,
        messages: list[Message],
    ) -> tuple[list[Message], dict[str, Any]]:
        snapshot = self._build_memory_reasoning_snapshot(ctx)
        if not snapshot:
            ctx.state.pop("memory_reasoning_snapshot", None)
            return (list(messages or []), {})

        ctx.state["memory_reasoning_snapshot"] = dict(snapshot)
        clean_messages = [
            message
            for message in list(messages or [])
            if not (
                str(getattr(message, "role", "") or "").strip().lower() == "system"
                and str(getattr(message, "content", "") or "").startswith(_MEMORY_REASONING_TAG)
            )
        ]
        insert_at = 0
        while insert_at < len(clean_messages) and str(clean_messages[insert_at].role or "") == "system":
            insert_at += 1
        clean_messages.insert(
            insert_at,
            Message(
                role="system",
                content=(
                    f"{_MEMORY_REASONING_TAG}\n"
                    "Use this compact memory snapshot only as a pre-answer consistency check.\n"
                    "Silently verify that the final answer does not contradict the profile facts, identity core, or active task.\n"
                    "If there is conflict or uncertainty, answer cautiously, avoid overclaiming, and prefer continuity over invention.\n"
                    f"{_compact_json(snapshot)}"
                ),
            ),
        )
        return (clean_messages, snapshot)

    def _build_memory_reasoning_snapshot(self, ctx: PipelineContext) -> dict[str, Any]:
        state = _as_dict(ctx.state)
        active_profile = self._flatten_memory_reasoning_profile(
            _as_dict(state.get("active_profile_snapshot"))
        )
        identity_core = self._compact_memory_reasoning_identity_core(
            _as_dict(state.get("identity_core"))
        )
        active_task = self._compact_memory_reasoning_active_task(
            _pick_value(
                state.get("active_task"),
                _as_dict(state.get("task_continuity")).get("active_task"),
                {},
            )
        )
        snapshot: dict[str, Any] = {}
        profile_facts = self._compact_memory_reasoning_profile_facts(active_profile)
        if not profile_facts:
            profile_facts = self._fallback_memory_reasoning_profile_facts(identity_core)
        if profile_facts:
            snapshot["profile_facts"] = profile_facts
        if identity_core:
            snapshot["identity_core"] = identity_core
        if active_task:
            snapshot["active_task"] = active_task
        return snapshot

    @staticmethod
    def _flatten_memory_reasoning_profile(snapshot: dict[str, Any] | None) -> dict[str, Any]:
        row = dict(snapshot or {})
        resolved = _as_dict(row.get("resolved_profile"))
        if resolved:
            return dict(resolved)
        # Упрощённая логика — возвращаем resolved_profile или пустой dict
        return {
            key: value
            for key, value in row.items()
            if key not in {
                "namespace",
                "active_facts",
                "conflicts",
                "persistent_traits",
                "volatile_preferences",
                "session_preferences",
                "resolved_profile",
                "debug",
                "updated_at",
            }
        }

    @staticmethod
    def _compact_memory_reasoning_profile_facts(profile: dict[str, Any] | None) -> dict[str, Any]:
        row = dict(profile or {})
        out: dict[str, Any] = {}
        for key in (
            "identity_name",
            "prefers_short_answers",
            "prefers_examples_on_user_code",
            "allow_light_teasing",
            "avoid_baby_talk",
        ):
            if key in row and not _is_empty_string_list_or_value(row.get(key)):
                out[key] = row.get(key)
        for key in (
            "assistant_warmth",
            "assistant_directness",
            "assistant_teasing",
            "relation_technical_collaboration",
        ):
            value = _to_float(row.get(key), None)
            if value is not None:
                out[key] = round(float(value), 3)
        if "preferred_answer_brevity" in row and "prefers_short_answers" not in out:
            value = _to_float(row.get("preferred_answer_brevity"), None)
            if value is not None:
                out["prefers_short_answers"] = round(float(value), 3)
        return out

    @staticmethod
    def _fallback_memory_reasoning_profile_facts(identity_core: dict[str, Any] | None) -> dict[str, Any]:
        row = dict(identity_core or {})
        out: dict[str, Any] = {}
        addressing = _as_dict(row.get("addressing"))
        interaction_style = _as_dict(row.get("interaction_style"))
        boundaries = _as_dict(row.get("boundaries"))
        assistant_trait_baseline = _as_dict(row.get("assistant_trait_baseline"))

        canonical_name = str(addressing.get("canonical_name") or "").strip()
        if canonical_name:
            out["identity_name"] = canonical_name
        for key in ("prefers_short_answers", "prefers_examples_on_user_code", "allows_light_teasing"):
            if key in interaction_style and not _is_empty_string_list_or_value(interaction_style.get(key)):
                mapped_key = "allow_light_teasing" if key == "allows_light_teasing" else key
                out[mapped_key] = interaction_style.get(key)
        if "avoid_baby_talk" in boundaries:
            out["avoid_baby_talk"] = bool(boundaries.get("avoid_baby_talk"))
        directness = _to_float(
            _pick_value(
                interaction_style.get("prefers_directness"),
                assistant_trait_baseline.get("directness_baseline"),
                None,
            ),
            None,
        )
        if directness is not None:
            out["assistant_directness"] = round(float(directness), 3)
        warmth = _to_float(assistant_trait_baseline.get("warmth_baseline"), None)
        if warmth is not None:
            out["assistant_warmth"] = round(float(warmth), 3)
        return out

    @staticmethod
    def _compact_memory_reasoning_identity_core(identity_core: dict[str, Any] | None) -> dict[str, Any]:
        row = dict(identity_core or {})
        out: dict[str, Any] = {}

        addressing = _as_dict(row.get("addressing"))
        if addressing:
            compact_addressing: dict[str, Any] = {}
            canonical_name = str(addressing.get("canonical_name") or "").strip()
            if canonical_name:
                compact_addressing["canonical_name"] = canonical_name
            allowed_forms = [str(x).strip() for x in list(addressing.get("allowed_forms") or []) if str(x).strip()]
            if allowed_forms:
                compact_addressing["allowed_forms"] = allowed_forms[:3]
            if "allow_diminutives" in addressing:
                compact_addressing["allow_diminutives"] = bool(addressing.get("allow_diminutives"))
            if compact_addressing:
                out["addressing"] = compact_addressing

        interaction_style = _as_dict(row.get("interaction_style"))
        if interaction_style:
            compact_style: dict[str, Any] = {}
            for key in ("prefers_directness", "prefers_short_answers"):
                value = _to_float(interaction_style.get(key), None)
                if value is not None:
                    compact_style[key] = round(float(value), 3)
            for key in ("prefers_examples_on_user_code", "allows_light_teasing", "technical_collaboration_style"):
                if key in interaction_style and not _is_empty_string_list_or_value(interaction_style.get(key)):
                    compact_style[key] = interaction_style.get(key)
            if compact_style:
                out["interaction_style"] = compact_style

        boundaries = _as_dict(row.get("boundaries"))
        if boundaries:
            compact_boundaries = {
                key: bool(boundaries.get(key))
                for key in ("avoid_baby_talk", "do_not_invent_user_facts", "avoid_overformal_tone")
                if key in boundaries
            }
            if compact_boundaries:
                out["boundaries"] = compact_boundaries

        emotional_handling = _as_dict(row.get("emotional_handling"))
        if emotional_handling:
            compact_emotional: dict[str, Any] = {}
            for key in ("deescalate_on_irritation", "treat_short_replies_as_low_bandwidth"):
                if key in emotional_handling:
                    compact_emotional[key] = bool(emotional_handling.get(key))
            for key in ("warmth_upshift_on_user_distress", "playfulness_downshift_on_user_distress"):
                value = _to_float(emotional_handling.get(key), None)
                if value is not None:
                    compact_emotional[key] = round(float(value), 3)
            if compact_emotional:
                out["emotional_handling"] = compact_emotional

        assistant_trait_baseline = _as_dict(row.get("assistant_trait_baseline"))
        if assistant_trait_baseline:
            compact_baseline: dict[str, Any] = {}
            for key in ("warmth_baseline", "directness_baseline", "sarcasm_ceiling"):
                value = _to_float(assistant_trait_baseline.get(key), None)
                if value is not None:
                    compact_baseline[key] = round(float(value), 3)
            if compact_baseline:
                out["assistant_trait_baseline"] = compact_baseline

        return out

    @staticmethod
    def _compact_memory_reasoning_active_task(value: Any) -> dict[str, Any]:
        row = _as_dict(value)
        if not row:
            return {}
        summary = str(
            _pick_value(
                row.get("current_goal"),
                row.get("summary_short"),
                row.get("topic"),
                "",
            )
            or ""
        ).strip()
        out = {
            "task_id": str(row.get("task_id") or "").strip(),
            "topic": str(row.get("topic") or "").strip(),
            "status": str(row.get("status") or "").strip().lower(),
            "summary": summary,
            "next_steps": [str(x).strip() for x in list(row.get("next_steps") or []) if str(x).strip()][:3],
            "open_questions": [str(x).strip() for x in list(row.get("open_questions") or []) if str(x).strip()][:3],
        }
        return {
            key: value
            for key, value in out.items()
            if not _is_empty_string_list_or_value(value)
        }

    def _execute_agent_tool_call(self, ctx: PipelineContext, call: ToolCall) -> tuple[str, bool]:
        name = str(call.name or "").strip().lower()
        
        # Memory retrieve (semantic search)
        if name == _MEMORY_TOOL_NAME:
            return self._execute_memory_retrieve_tool(ctx, call)
        
        # History tools (exact DB reads)
        if name in ("history_read_recent", "history_read_range", "history_search"):
            return self._execute_history_tool(ctx, call)

        # Topic tools (hidden topic threads)
        if name in _TOPIC_TOOL_NAMES:
            return self._execute_topic_tool(ctx, call)

        executor = ctx.meta.get("tool_executor")
        if not callable(executor):
            payload = {
                "tool": str(call.name or ""),
                "status": "error",
                "error": f"tool '{call.name}' is not available",
            }
            return _compact_json(payload), True
        try:
            result = executor(_tool_call_to_dict(call))
        except Exception as exc:
            payload = {
                "tool": str(call.name or ""),
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
            }
            return _compact_json(payload), True
        payload = {
            "tool": str(call.name or ""),
            "status": "ok",
            "result": result,
        }
        return _compact_json(payload), False

    def _generate_with_agent_loop_streaming(
        self,
        ctx: PipelineContext,
        req: LLMRequest,
        on_answer,
        on_thinking,
        *,
        emit_answer_live: bool = True,
    ) -> LLMResponse:
        """
        Streaming генерация внутри agent_loop.

        Использует provider.stream() вместо provider.generate() и вызывает
        callbacks для каждого chunk.
        """
        stream = getattr(self.provider, "stream", None)
        if not callable(stream):
            # Fallback на обычный generate если stream не доступен
            ctx.logs.append("stage=generate agent_loop_stream_fallback=no_provider_stream")
            return self.provider.generate(req)

        parser = _ThinkStreamParser()
        answer_parts: list[str] = []
        thinking_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        model_name = str(req.model or "")
        usage = Usage()
        timings = Timings()

        # Счётчики для диагностики streaming
        visible_answer_chunks = 0
        visible_thinking_chunks = 0

        def _handle_visible_answer(piece: str) -> None:
            nonlocal visible_answer_chunks
            if not piece:
                return
            visible_answer_chunks += 1
            if not emit_answer_live or not callable(on_answer):
                return
            if tool_calls:
                return
            try:
                on_answer(piece)
            except Exception:
                pass

        def _handle_visible_thinking(piece: str) -> None:
            nonlocal visible_thinking_chunks
            if not piece:
                return
            visible_thinking_chunks += 1
            if callable(on_thinking):
                try:
                    on_thinking(piece)
                except Exception:
                    pass

        try:
            for chunk in stream(req):
                chunk_model = str(getattr(chunk, "model", "") or "").strip()
                if chunk_model:
                    model_name = chunk_model

                tool_calls_delta = list(getattr(chunk, "tool_calls_delta", []) or [])
                if tool_calls_delta:
                    for call in tool_calls_delta:
                        if not isinstance(call, ToolCall):
                            continue
                        if any(str(existing.id or "") == str(call.id or "") for existing in tool_calls):
                            continue
                        tool_calls.append(call)

                # Thinking delta from provider
                thinking_delta = str(getattr(chunk, "thinking_delta", "") or "")
                if thinking_delta:
                    thinking_parts.append(thinking_delta)
                    _handle_visible_thinking(thinking_delta)

                # Answer delta with inline <think> support
                text_delta = str(getattr(chunk, "text_delta", "") or "")
                if text_delta:
                    visible, thinking_from_text = parser.feed(text_delta)
                    if thinking_from_text:
                        thinking_parts.append(thinking_from_text)
                        _handle_visible_thinking(thinking_from_text)
                    if visible:
                        answer_parts.append(visible)
                        _handle_visible_answer(visible)

                if bool(getattr(chunk, "done", False)):
                    usage = getattr(chunk, "usage", None) or usage
                    timings = getattr(chunk, "timings", None) or timings

            visible, thinking_from_text = parser.flush()
            if thinking_from_text:
                thinking_parts.append(thinking_from_text)
                _handle_visible_thinking(thinking_from_text)
            if visible:
                answer_parts.append(visible)
                _handle_visible_answer(visible)
            
            ctx.logs.append(
                f"stage=generate agent_loop_stream_visible answer_chunks={visible_answer_chunks} "
                f"thinking_chunks={visible_thinking_chunks} tool_calls={len(tool_calls)}"
            )
        except Exception as exc:
            ctx.logs.append(
                f"stage=generate agent_loop_stream_error={type(exc).__name__}:{exc}"
            )
            ctx.errors.append(f"agent_loop_stream:{type(exc).__name__}:{exc}")
            raise

        return LLMResponse(
            text="".join(answer_parts).strip(),
            tool_calls=list(tool_calls),
            thinking="".join(thinking_parts).strip(),
            model=model_name,
            usage=usage,
            timings=timings,
        )

    def _execute_memory_retrieve_tool(self, ctx: PipelineContext, call: ToolCall) -> tuple[str, bool]:
        """
        Выполнить memory_retrieve tool.

        Эмитит debug events если доступен stream_on_debug_event.
        """
        memory_core = ctx.meta.get("memory_core") or ctx.meta.get("memory_manager") or self.memory_core
        debug_event = ctx.meta.get("stream_on_debug_event")

        def emit(kind: str, payload: dict[str, Any]) -> None:
            if callable(debug_event):
                try:
                    debug_event(kind, payload)
                except Exception:
                    pass

        if memory_core is None:
            emit("memory_error", {"tool": _MEMORY_TOOL_NAME, "error": "memory_core_not_available"})
            payload = {
                "tool": _MEMORY_TOOL_NAME,
                "status": "error",
                "error": "memory core is not available",
            }
            return _compact_json(payload), True

        args = dict(call.arguments or {})
        if not any(key in args for key in ("mode", "topic_hints", "time_hint", "sources")):
            args = {
                "mode": "context",
                "topic_hints": [str(_pick_value(args.get("focus"), args.get("query"), ctx.clean_user_msg, "")).strip()],
                "time_hint": "recent",
                "sources": ["all"],
                "top_k": _to_int(args.get("top_k"), 6) or 6,
                "scopes": list(_as_list(args.get("scopes"))),
            }
        
        emit("memory_retrieval_start", {
            "tool": _MEMORY_TOOL_NAME,
            "mode": args.get("mode"),
            "topic_hints": args.get("topic_hints"),
            "time_hint": args.get("time_hint"),
        })

        # Упрощённая логика memory retrieval для memory_core
        base_query = str(_pick_value(ctx.clean_user_msg, ctx.user_msg, "")).strip()
        top_k = int(_to_int(args.get("top_k"), 6) or 6)

        try:
            # Выполняем retrieval через memory_core напрямую
            memory_core = ctx.meta.get("memory_core") or ctx.meta.get("memory_manager") or self.memory_core
            if memory_core is None:
                raise RuntimeError("memory_core not available")

            workspace_id = str(ctx.meta.get("workspace_id") or ctx.state.get("active_character_id") or "global")
            session_id = str(ctx.meta.get("conversation_id") or ctx.state.get("conversation_id") or "")
            topic_thread_id = str(ctx.state.get("topic_thread_id") or ctx.meta.get("topic_thread_id") or "")
            related_topic_ids = list(ctx.state.get("related_topic_thread_ids") or [])

            if hasattr(memory_core, "retrieve"):
                result = memory_core.retrieve(
                    query=base_query,
                    top_k=top_k,
                    workspace_id=workspace_id,
                    session_id=session_id,
                    topic_thread_id=topic_thread_id,
                    related_topic_ids=related_topic_ids,
                )
            elif hasattr(memory_core, "query"):
                result = memory_core.query(
                    text=base_query,
                    workspace_id=workspace_id,
                    session_id=session_id,
                    top_k=top_k,
                    include_citations=True,
                    topic_thread_id=topic_thread_id,
                    related_topic_ids=related_topic_ids,
                )
            else:
                request = {
                    "text": base_query,
                    "workspace_id": workspace_id,
                    "session_id": session_id,
                    "topic_thread_id": topic_thread_id,
                    "related_topic_ids": related_topic_ids,
                    "top_k": top_k,
                    "include_citations": True,
                }
                raw_result = memory_core.build_context(request)
                result = raw_result.to_dict() if hasattr(raw_result, "to_dict") else dict(raw_result or {})

            # Сохраняем результат в ctx.memory_context
            pack = {
                "selected": list(result.get("selected") or []),
                "blocks": dict(result.get("blocks") or {}),
                "recall_mode": str(result.get("recall_mode") or "semantic"),
                "tool_retrieval_plan": {"mode": "context", "top_k": top_k},
            }

            ctx.memory_context = dict(pack)
            ctx.state["memory_context"] = dict(pack)

            retrieved = list(_as_list(pack.get("selected")))
            if retrieved:
                ctx.retrieved_memories = retrieved

            trace = _ensure_debug_trace(ctx)
            trace.memory_retrieval = {
                "stage": "agent_memory_retrieve",
                "query": base_query,
                "selected_count": len(retrieved),
                "recall_mode": str(pack.get("recall_mode") or ""),
                "tool_mode": str(args.get("mode") or "context"),
            }

            selected_count = len(list(_as_list(pack.get("selected"))))
            emit("memory_retrieval_done", {
                "tool": _MEMORY_TOOL_NAME,
                "status": "ok",
                "selected_count": selected_count,
                "recall_mode": str(pack.get("recall_mode") or ""),
            })

        except Exception as exc:
            emit("memory_retrieval_error", {
                "tool": _MEMORY_TOOL_NAME,
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
            })
            payload = {
                "tool": _MEMORY_TOOL_NAME,
                "status": "error",
                "query": base_query,
                "error": f"{type(exc).__name__}: {exc}",
            }
            return _compact_json(payload), True

        # Упрощённый payload
        payload = {
            "tool": _MEMORY_TOOL_NAME,
            "status": "ok",
            "mode": str(args.get("mode") or "context"),
            "selected": [
                _trace_compact_selected_memory(row)
                for row in list(_as_list(ctx.memory_context.get("selected")))[:top_k]
                if isinstance(row, dict)
            ],
            "recall_mode": str(ctx.memory_context.get("recall_mode") or ""),
        }
        return _compact_json(payload), False

    def _execute_history_read_recent(
        self,
        ctx: PipelineContext,
        *,
        limit: int,
        role: str,
        order: str,
    ) -> HistoryReadResult:
        """
        Выполнить history_read_recent — чтение последних сообщений.
        
        Читает из state["history"], а не из vector storage.
        """
        history = list(ctx.state.get("history") or [])
        
        # Фильтр по role
        if role != "any":
            history = [h for h in history if str(h.get("role") or "").lower() == role]
        
        # Порядок
        if order == "newest_first":
            history = list(reversed(history))
        
        # Лимит
        total = len(history)
        history = history[:limit]
        
        return HistoryReadResult(
            messages=history,
            total_count=total,
            has_more=total > limit,
        )
    
    def _execute_history_search(
        self,
        ctx: PipelineContext,
        *,
        query: str,
        role: str,
        limit: int,
    ) -> HistoryReadResult:
        """
        Выполнить history_search — поиск по истории.
        
        Простой text search по state["history"].
        """
        history = list(ctx.state.get("history") or [])
        query_lower = query.lower()
        
        # Фильтр по role
        if role != "any":
            history = [h for h in history if str(h.get("role") or "").lower() == role]
        
        # Поиск
        results = [
            h for h in history
            if query_lower in str(h.get("content") or "").lower()
        ]
        
        # Лимит
        total = len(results)
        results = results[:limit]
        
        return HistoryReadResult(
            messages=results,
            total_count=total,
            has_more=total > limit,
        )

    def _execute_history_tool(self, ctx: PipelineContext, call: ToolCall) -> tuple[str, bool]:
        """
        Выполнить history tool (read_recent, read_range, search).

        Эти tools читают точную историю из state["history"], а не semantic search.
        """
        from memory_core.retrieval.history_tools import (
            _execute_history_read_recent,
            _execute_history_search,
            HistoryReadResult,
        )

        tool_name = str(call.name or "").strip().lower()
        args = dict(call.arguments or {})

        try:
            if tool_name == "history_read_recent":
                limit = _to_int(args.get("limit"), 10)
                role = str(args.get("role", "any")).strip().lower()
                order = str(args.get("order", "newest_first")).strip().lower()

                result: HistoryReadResult = _execute_history_read_recent(
                    ctx,
                    limit=limit,
                    role=role,
                    order=order,
                )

            elif tool_name == "history_search":
                query = str(args.get("query", "")).strip()
                limit = _to_int(args.get("limit"), 10)
                role = str(args.get("role", "any")).strip().lower()

                result = _execute_history_search(
                    ctx,
                    query=query,
                    role=role,
                    limit=limit,
                )

            else:
                payload = {
                    "tool": tool_name,
                    "status": "error",
                    "error": f"Unknown history tool: {tool_name}",
                }
                return _compact_json(payload), True
            
            payload = {
                "tool": tool_name,
                "status": "ok",
                "result": result.to_dict(),
                "message_count": result.total_count,
            }
            return _compact_json(payload), False
            
        except Exception as exc:
            payload = {
                "tool": tool_name,
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
            }
            return _compact_json(payload), True

    def _execute_topic_tool(self, ctx: PipelineContext, call: ToolCall) -> tuple[str, bool]:
        tool_name = str(call.name or "").strip().lower()
        args = dict(call.arguments or {})
        topic_store = _resolve_topic_store(ctx, self.memory_core)
        if topic_store is None:
            payload = {
                "tool": tool_name,
                "status": "error",
                "error": "topic store is not available",
            }
            return _compact_json(payload), True

        service = TopicToolService(topic_store)
        workspace_id = str(ctx.meta.get("workspace_id") or ctx.state.get("active_character_id") or "global")
        visible_chat_id = str(ctx.meta.get("conversation_id") or ctx.state.get("conversation_id") or "")
        session_id = str(ctx.meta.get("conversation_id") or ctx.state.get("conversation_id") or "")
        default_thread_id = str(ctx.state.get("topic_thread_id") or ctx.meta.get("topic_thread_id") or "").strip()

        try:
            if tool_name == "topic_read":
                thread_id = str(args.get("thread_id") or default_thread_id).strip()
                if not thread_id:
                    raise ValueError("thread_id is required")
                result = service.read_topic(
                    thread_id,
                    limit=_to_int(args.get("limit"), 12) or 12,
                    workspace_id=workspace_id,
                )
                if result is None:
                    raise LookupError(f"topic '{thread_id}' not found")
            elif tool_name == "topic_search":
                result = service.search_topics(
                    query=str(args.get("query") or args.get("topic") or "").strip(),
                    visible_chat_id=visible_chat_id,
                    workspace_id=workspace_id,
                    session_id=session_id,
                    status=str(args.get("status") or "").strip().lower() or None,
                    limit=_to_int(args.get("limit"), 6) or 6,
                )
            elif tool_name == "topic_related":
                thread_id = str(args.get("thread_id") or default_thread_id).strip()
                if not thread_id:
                    raise ValueError("thread_id is required")
                result = service.related_topics(
                    thread_id,
                    visible_chat_id=visible_chat_id,
                    workspace_id=workspace_id,
                    session_id=session_id,
                    limit=_to_int(args.get("limit"), 6) or 6,
                )
            else:
                payload = {
                    "tool": tool_name,
                    "status": "error",
                    "error": f"Unknown topic tool: {tool_name}",
                }
                return _compact_json(payload), True
        except Exception as exc:
            payload = {
                "tool": tool_name,
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
            }
            return _compact_json(payload), True

        payload = {
            "tool": tool_name,
            "status": "ok",
            "result": result,
        }
        return _compact_json(payload), False

    @staticmethod
    def _merge_llm_stats(
        base: dict[str, Any] | None,
        *,
        usage=None,
        timings=None,
    ) -> dict[str, Any]:
        stats = dict(base or {})
        if usage is not None:
            prompt_tokens = _to_int(getattr(usage, "prompt_tokens", None), None)
            completion_tokens = _to_int(getattr(usage, "completion_tokens", None), None)
            total_tokens = _to_int(getattr(usage, "total_tokens", None), None)
            if prompt_tokens is not None:
                stats["prompt_eval_count"] = int(prompt_tokens)
            if completion_tokens is not None:
                stats["eval_count"] = int(completion_tokens)
            if total_tokens is not None:
                stats["total_tokens"] = int(total_tokens)
            elif prompt_tokens is not None or completion_tokens is not None:
                stats["total_tokens"] = int((stats.get("prompt_eval_count") or 0) + (stats.get("eval_count") or 0))
        if timings is not None:
            answer_ms = _to_float(getattr(timings, "latency_ms", None), None)
            total_duration_ms = _to_float(getattr(timings, "total_duration_ms", None), None)
            load_duration_ms = _to_float(getattr(timings, "load_duration_ms", None), None)
            prompt_eval_duration_ms = _to_float(getattr(timings, "prompt_eval_duration_ms", None), None)
            eval_duration_ms = _to_float(getattr(timings, "eval_duration_ms", None), None)
            if answer_ms is not None:
                stats["answer_ms"] = float(answer_ms)
            if total_duration_ms is not None and total_duration_ms > 0:
                stats["total_duration_ms"] = float(total_duration_ms)
            if load_duration_ms is not None and load_duration_ms > 0:
                stats["load_duration_ms"] = float(load_duration_ms)
            if prompt_eval_duration_ms is not None and prompt_eval_duration_ms > 0:
                stats["prompt_eval_duration_ms"] = float(prompt_eval_duration_ms)
            if eval_duration_ms is not None and eval_duration_ms > 0:
                stats["eval_duration_ms"] = float(eval_duration_ms)
        eval_count = int(_to_int(stats.get("eval_count"), 0) or 0)
        eval_duration_ms = float(_to_float(stats.get("eval_duration_ms"), 0.0) or 0.0)
        if eval_count > 0 and eval_duration_ms > 0:
            stats["eval_tokens_per_sec"] = round(float(eval_count) / (eval_duration_ms / 1000.0), 2)
        prompt_count = int(_to_int(stats.get("prompt_eval_count"), 0) or 0)
        prompt_duration_ms = float(_to_float(stats.get("prompt_eval_duration_ms"), 0.0) or 0.0)
        if prompt_count > 0 and prompt_duration_ms > 0:
            stats["prompt_tokens_per_sec"] = round(float(prompt_count) / (prompt_duration_ms / 1000.0), 2)
        return stats

    def _generate_stream(self, ctx: PipelineContext, req: LLMRequest, *, on_answer, on_thinking) -> tuple[str, str, str, dict[str, Any]] | None:
        stream = getattr(self.provider, "stream", None)
        if not callable(stream):
            return None
        parser = _ThinkStreamParser()
        answer_parts: list[str] = []
        thinking_parts: list[str] = []
        model_name = str(req.model or "")
        stream_stats: dict[str, Any] = {}

        try:
            for chunk in stream(req):
                chunk_model = str(getattr(chunk, "model", "") or "").strip()
                if chunk_model:
                    model_name = chunk_model
                # 1) thinking из провайдера (Ollama отдаёт thinking_delta отдельно)
                thinking_delta = str(getattr(chunk, "thinking_delta", "") or "")
                if thinking_delta:
                    thinking_parts.append(thinking_delta)
                    if callable(on_thinking):
                        try:
                            on_thinking(thinking_delta)
                        except Exception:
                            pass

                # 2) обычный текст
                text_delta = str(getattr(chunk, "text_delta", "") or "")
                if text_delta:
                    # на всякий случай также поддерживаем <think>...</think> в самом тексте
                    visible, thinking_from_text = parser.feed(text_delta)
                    if thinking_from_text:
                        thinking_parts.append(thinking_from_text)
                        if callable(on_thinking):
                            try:
                                on_thinking(thinking_from_text)
                            except Exception:
                                pass
                    if visible:
                        answer_parts.append(visible)
                        if callable(on_answer):
                            try:
                                on_answer(visible)
                            except Exception:
                                pass
                if bool(getattr(chunk, "done", False)):
                    stream_stats = self._merge_llm_stats(
                        {
                            "served_model": str(model_name or req.model or ""),
                            "streaming": True,
                        },
                        usage=getattr(chunk, "usage", None),
                        timings=getattr(chunk, "timings", None),
                    )
            visible, thinking_from_text = parser.flush()
            if thinking_from_text:
                thinking_parts.append(thinking_from_text)
                if callable(on_thinking):
                    try:
                        on_thinking(thinking_from_text)
                    except Exception:
                        pass
            if visible:
                answer_parts.append(visible)
                if callable(on_answer):
                    try:
                        on_answer(visible)
                    except Exception:
                        pass
        except Exception as exc:
            ctx.logs.append(f"stage=generate stream_error={type(exc).__name__}:{exc}")
            ctx.errors.append(f"stream:{type(exc).__name__}:{exc}")
            raise

        return ("".join(answer_parts).strip(), "".join(thinking_parts).strip(), model_name, dict(stream_stats))

    @staticmethod
    def _handle_system_event(ctx: PipelineContext) -> None:
        event_name = str(ctx.meta.get("event") or ctx.meta.get("event_type") or ctx.clean_user_msg or "event").lower()
        if event_name in {"voice_start", "voice.start", "voice:started"}:
            ctx.text = "Voice mode started."
            ctx.ui_actions.append({"type": "voice_indicator", "state": "on"})
            ctx.memory_ops.append({"op": "state_mode", "value": "voice"})
        elif event_name in {"voice_stop", "voice.stop", "voice:stopped"}:
            ctx.text = "Voice mode stopped."
            ctx.ui_actions.append({"type": "voice_indicator", "state": "off"})
            ctx.memory_ops.append({"op": "state_mode", "value": "chat"})
        else:
            ctx.text = "System event processed."
            ctx.ui_actions.append({"type": "noop", "event": event_name})
        ctx.logs.append("stage=generate route=system_event")

    def _handle_internal_command(self, ctx: PipelineContext) -> bool:
        cmd = str(ctx.clean_user_msg or "").strip().lower()
        if cmd.startswith("/studio"):
            ctx.meta["force_output_format"] = True
        studio_state = self._studio_state(ctx)
        studio_active = self.studio_generator.is_active({StudioGenerator.KEY: studio_state})
        scope = "studio" if studio_active else "chat"

        if cmd.startswith("/specs"):
            ctx.text = "Команда удалена. Используй /studio ..."
            ctx.logs.append("stage=generate command=specs:removed")
            return True

        if cmd.startswith("/studio mode"):
            ctx.text = "Команда удалена. Используй /studio ..."
            ctx.logs.append("stage=generate command=studio:mode_removed")
            return True

        model = _pick(ctx.meta.get("model"), ctx.state.get("model"), ctx.policies.get("model"))

        if cmd in {"/apply", "/cancel"} and studio_active:
            if cmd == "/cancel":
                reply = self.studio_generator.cancel({StudioGenerator.KEY: studio_state}, command_alias="studio")
                self._save_studio_state(ctx, dict(reply.state or {}))
                self._apply_studio_telemetry(ctx, dict(reply.state or {}))
                ctx.text = str(reply.text or "")
                if reply.memory_ops:
                    ctx.memory_ops.extend(list(reply.memory_ops or []))
                if reply.payload:
                    ctx.structured_output["studio"] = dict(reply.payload)
                ctx.logs.append("stage=generate command=studio:cancel_local")
                return True
            reply = self.studio_generator.apply(
                {StudioGenerator.KEY: studio_state},
                provider=self.provider,
                model=model,
                command_alias="studio",
            )
            self._save_studio_state(ctx, dict(reply.state or {}))
            self._apply_studio_telemetry(ctx, dict(reply.state or {}))
            if bool(getattr(reply, "done", False)) and not str(getattr(reply, "error", "") or "").strip():
                invalidate_spec_cache()
            ctx.text = str(reply.text or "")
            if reply.memory_ops:
                ctx.memory_ops.extend(list(reply.memory_ops or []))
            if reply.payload:
                ctx.structured_output["studio"] = dict(reply.payload)
            ctx.logs.append("stage=generate command=studio:apply_local")
            return True

        if cmd.startswith("/studio"):
            studio_row = self._studio_state(ctx)

            if cmd in {"/studio", "/studio help", "/studio status"}:
                if cmd == "/studio" and not self.studio_generator.is_active({StudioGenerator.KEY: studio_row}):
                    reply = self.studio_generator.start(
                        {StudioGenerator.KEY: studio_row},
                        seed="",
                        provider=self.provider,
                        model=model,
                        command_alias="studio",
                    )
                else:
                    reply = self.studio_generator.status(
                        {StudioGenerator.KEY: studio_row},
                        provider=self.provider,
                        model=model,
                        command_alias="studio",
                    )
                self._save_studio_state(ctx, dict(reply.state or {}))
                self._apply_studio_telemetry(ctx, dict(reply.state or {}))
                ctx.text = str(reply.text or "")
                if reply.memory_ops:
                    ctx.memory_ops.extend(list(reply.memory_ops or []))
                if reply.payload:
                    ctx.structured_output["studio"] = dict(reply.payload)
                ctx.logs.append("stage=generate command=studio:status")
                return True

            if cmd.startswith("/studio start"):
                seed = ""
                parts = str(ctx.clean_user_msg or "").strip().split(" ", 2)
                if len(parts) >= 3:
                    seed = str(parts[2] or "").strip()
                reply = self.studio_generator.start(
                    {StudioGenerator.KEY: studio_row},
                    seed=seed,
                    provider=self.provider,
                    model=model,
                    command_alias="studio",
                )
                self._save_studio_state(ctx, dict(reply.state or {}))
                self._apply_studio_telemetry(ctx, dict(reply.state or {}))
                ctx.state["quality_profile"] = PROFILE_AUTONOMOUS
                ctx.text = str(reply.text or "")
                ctx.logs.append("stage=generate command=studio:start")
                return True

            if cmd == "/studio cancel":
                reply = self.studio_generator.cancel({StudioGenerator.KEY: studio_row}, command_alias="studio")
                self._save_studio_state(ctx, dict(reply.state or {}))
                self._apply_studio_telemetry(ctx, dict(reply.state or {}))
                ctx.text = str(reply.text or "")
                if reply.memory_ops:
                    ctx.memory_ops.extend(list(reply.memory_ops or []))
                if reply.payload:
                    ctx.structured_output["studio"] = dict(reply.payload)
                # local short-circuit: command is handled inside studio and never dispatched to main generation request
                ctx.logs.append("stage=generate command=studio:cancel_local")
                return True

            if cmd == "/studio apply":
                reply = self.studio_generator.apply(
                    {StudioGenerator.KEY: studio_row},
                    provider=self.provider,
                    model=model,
                    command_alias="studio",
                )
                self._save_studio_state(ctx, dict(reply.state or {}))
                self._apply_studio_telemetry(ctx, dict(reply.state or {}))
                if bool(getattr(reply, "done", False)) and not str(getattr(reply, "error", "") or "").strip():
                    invalidate_spec_cache()
                ctx.text = str(reply.text or "")
                if reply.memory_ops:
                    ctx.memory_ops.extend(list(reply.memory_ops or []))
                if reply.payload:
                    ctx.structured_output["studio"] = dict(reply.payload)
                # local short-circuit: command is handled inside studio and never dispatched to main generation request
                ctx.logs.append("stage=generate command=studio:apply_local")
                return True

            if cmd.startswith("/studio answer "):
                answer = str(ctx.clean_user_msg or "").split(" ", 2)[2] if len(str(ctx.clean_user_msg or "").split(" ", 2)) >= 3 else ""
                reply = self.studio_generator.ingest(
                    {StudioGenerator.KEY: studio_row},
                    answer,
                    provider=self.provider,
                    model=model,
                    command_alias="studio",
                )
                self._save_studio_state(ctx, dict(reply.state or {}))
                self._apply_studio_telemetry(ctx, dict(reply.state or {}))
                ctx.text = str(reply.text or "")
                if reply.memory_ops:
                    ctx.memory_ops.extend(list(reply.memory_ops or []))
                if reply.payload:
                    ctx.structured_output["studio"] = dict(reply.payload)
                ctx.logs.append("stage=generate command=studio:answer")
                return True

            ctx.text = "Неизвестная команда studio. Используй /studio status|start|apply|cancel"
            ctx.logs.append("stage=generate command=studio:unknown")
            return True

        if cmd in {"/characters", "/character list"}:
            ids = self.character_engine.list_ids()
            current = str(
                ctx.state.get("active_character_id")
                or ctx.state.get("active_personality_id")
                or (ids[0] if ids else "asya")
            ).strip().lower()
            if not ids:
                ctx.text = "No characters available."
            else:
                rows = [f"{'*' if x == current else ' '} {x}" for x in ids]
                ctx.text = "Characters:\n" + "\n".join(rows)
            ctx.logs.append("stage=generate command=characters:list")
            return True
        
        if cmd in {"/character", "/character current"}:
            ids = self.character_engine.list_ids()
            current = str(
                ctx.state.get("active_character_id")
                or ctx.state.get("active_personality_id")
                or (ids[0] if ids else "asya")
            ).strip().lower()
            locked = bool(ctx.state.get("character_locked", False))
            ctx.text = f"Character: {current} ({'locked' if locked else 'auto'})"
            ctx.logs.append("stage=generate command=character:show")
            return True
        
        if cmd.startswith("/character "):
            parts = [x for x in str(ctx.clean_user_msg or "").strip().split(" ") if x]
            if len(parts) >= 2 and str(parts[1]).strip().lower() in {"delete", "remove", "del", "rm"}:
                if len(parts) < 3:
                    ctx.text = "Usage: /character delete <character_id>"
                    ctx.logs.append("stage=generate command=character:delete:usage")
                    return True
                return self._handle_character_delete_command(ctx, str(parts[2]).strip().lower())

            target = cmd.split(" ", 1)[1].strip().lower()
            if not target:
                return False
            current = str(
                ctx.state.get("active_character_id")
                or ctx.state.get("active_personality_id")
                or "asya"
            ).strip().lower()
            if target == "auto":
                ctx.text = "Character auto mode enabled."
                ctx.memory_ops.append(
                    {
                        "op": "state_character",
                        "value": current,
                        "locked": False,
                        "ts": now_local_ts(),
                        "reason": "manual_auto",
                    }
                )
                ctx.logs.append("stage=generate command=character:auto")
                return True
            known = set(self.character_engine.list_ids())
            if target not in known:
                ctx.text = f"Unknown character '{target}'. Available: {', '.join(sorted(known))}"
                ctx.logs.append(f"stage=generate command=character:unknown:{target}")
                return True
            try:
                self.character_engine.set_active_character(target)
            except Exception:
                # Runtime state update below is still the source of truth for current turn.
                pass
            ctx.text = f"Character switched to: {target}"
            ctx.memory_ops.append(
                {
                    "op": "state_character",
                    "value": target,
                    "locked": True,
                    "ts": now_local_ts(),
                    "reason": "manual",
                    "confidence": 1.0,
                }
            )
            ctx.memory_ops.append(
                {
                    "op": "state_personality",
                    "value": target,
                    "locked": True,
                    "blend": {
                        "active": bool(current and current != target),
                        "from": current or target,
                        "to": target,
                        "step": 1 if current and current != target else 0,
                        "steps": 4,
                        "old_weight": 0.75 if current and current != target else 0.0,
                        "new_weight": 0.25 if current and current != target else 1.0,
                    },
                    "ts": now_local_ts(),
                    "reason": "manual_character_mirror",
                    "confidence": 1.0,
                }
            )
            ctx.logs.append(f"stage=generate command=character:{target}")
            return True
        if cmd.startswith("/trait "):
            return self._handle_trait_command(ctx, cmd)
        
        if cmd == "/think":
            if scope == "chat":
                ctx.text = "Thinking mode enabled."
                ctx.memory_ops.append({"op": "state_think", "value": True})
            else:
                self._scope_set(ctx, scope, think=True)
                ctx.text = f"Thinking mode enabled for scope: {scope}."
            ctx.ui_actions.append({"type": "toggle_think", "enabled": True})
            ctx.logs.append("stage=generate command=think")
            return True
    
        if cmd == "/nothink":
            if scope == "chat":
                ctx.text = "Thinking mode disabled."
                ctx.memory_ops.append({"op": "state_think", "value": False})
            else:
                self._scope_set(ctx, scope, think=False)
                ctx.text = f"Thinking mode disabled for scope: {scope}."
            ctx.ui_actions.append({"type": "toggle_think", "enabled": False})
            ctx.logs.append("stage=generate command=nothink")
            return True

        if cmd == "/verbose":
            if scope == "chat":
                ctx.text = "Verbose stats enabled."
                ctx.memory_ops.append({"op": "state_verbose", "value": True})
            else:
                self._scope_set(ctx, scope, verbose=True)
                ctx.text = f"Verbose stats enabled for scope: {scope}."
            ctx.ui_actions.append({"type": "toggle_verbose", "enabled": True})
            ctx.logs.append("stage=generate command=verbose")
            return True

        if cmd == "/quiet":
            if scope == "chat":
                ctx.text = "Verbose stats disabled."
                ctx.memory_ops.append({"op": "state_verbose", "value": False})
            else:
                self._scope_set(ctx, scope, verbose=False)
                ctx.text = f"Verbose stats disabled for scope: {scope}."
            ctx.ui_actions.append({"type": "toggle_verbose", "enabled": False})
            ctx.logs.append("stage=generate command=quiet")
            return True
        
        if cmd == "/web":
            if scope == "chat":
                ctx.text = "Web mode enabled."
                ctx.memory_ops.append({"op": "state_web_mode", "value": "on"})
            else:
                self._scope_set(ctx, scope, web_mode="on")
                ctx.text = f"Web mode enabled for scope: {scope}."
            ctx.ui_actions.append({"type": "set_web_mode", "mode": "on"})
            ctx.logs.append("stage=generate command=web mode=on")
            return True
        
        if cmd == "/no-web":
            if scope == "chat":
                ctx.text = "Web mode disabled."
                ctx.memory_ops.append({"op": "state_web_mode", "value": "off"})
            else:
                self._scope_set(ctx, scope, web_mode="off")
                ctx.text = f"Web mode disabled for scope: {scope}."
            ctx.ui_actions.append({"type": "set_web_mode", "mode": "off"})
            ctx.logs.append("stage=generate command=no-web mode=off")
            return True

        if cmd.startswith("/web-auto") or cmd.startswith("/web_auto") or cmd.startswith("/auto-web"):
            ctx.text = "Web auto profiles are removed. Use /web to enable web mode or /no-web to disable it."
            ctx.logs.append(f"stage=generate command=web-auto:deprecated scope={scope}")
            return True

        if cmd in {"/output", "/output status"}:
            output_spec = load_spec("output", required=False)
            mode = normalize_mode_name(
                _pick(
                    ctx.state.get("active_mode"),
                    ctx.meta.get("active_mode"),
                    ctx.state.get("mode"),
                    "chatting",
                ),
                allow_custom=True,
            )
            strict_modes = {
                str(x).strip().lower()
                for x in list(output_spec.get("default_enabled_modes") or ["engineer", "debugger", "planner"])
                if str(x).strip()
            }
            strict_default = mode in strict_modes
            scoped_of = self._scope_get(ctx, scope, "output_format", None) if scope != "chat" else None
            base_of = scoped_of if isinstance(scoped_of, dict) else _pick_value(ctx.state.get("output_format"), ctx.meta.get("output_format"), {})
            of = _coerce_output_format_state(base_of)
            manual_params = of.get("show_parameters")
            manual_summary = of.get("show_summary")
            eff_params = bool(manual_params) if isinstance(manual_params, bool) else bool(strict_default)
            summary_cfg = _as_dict(output_spec.get("summary"))
            required_modes = {
                str(x).strip().lower()
                for x in list(summary_cfg.get("required_modes") or [])
                if str(x).strip()
            }
            if isinstance(manual_summary, bool):
                eff_summary = bool(manual_summary)
            elif required_modes:
                eff_summary = mode in required_modes
            else:
                eff_summary = bool(strict_default)
            manual_params_text = ("on" if manual_params else "off") if isinstance(manual_params, bool) else "auto"
            manual_summary_text = ("on" if manual_summary else "off") if isinstance(manual_summary, bool) else "auto"
            ctx.text = (
                "Output format:\n"
                f"- mode: {mode}\n"
                f"- parameters: {'on' if eff_params else 'off'} (manual={manual_params_text})\n"
                f"- summary: {'on' if eff_summary else 'off'} (manual={manual_summary_text})"
            )
            ctx.logs.append("stage=generate command=output:status")
            return True

        if cmd.startswith("/output "):
            tail = cmd.replace("/output", "", 1).strip().lower()
            parts = [x for x in tail.split(" ") if x]
            if len(parts) != 2:
                ctx.text = "Usage: /output parameters on|off OR /output summary on|off"
                ctx.logs.append("stage=generate command=output:usage")
                return True
            field, raw_value = parts
            enabled = raw_value in {"on", "true", "1", "yes"}
            if raw_value not in {"on", "off", "true", "false", "1", "0", "yes", "no"}:
                ctx.text = "Usage: /output parameters on|off OR /output summary on|off"
                ctx.logs.append("stage=generate command=output:usage")
                return True
            if field not in {"parameters", "summary"}:
                ctx.text = "Usage: /output parameters on|off OR /output summary on|off"
                ctx.logs.append(f"stage=generate command=output:unknown:{field}")
                return True
            value: dict[str, Any]
            if field == "parameters":
                value = {"show_parameters": bool(enabled)}
            else:
                value = {"show_summary": bool(enabled)}
            if scope == "chat":
                ctx.memory_ops.append({"op": "state_output_format", "value": value})
            else:
                current = _coerce_output_format_state(self._scope_get(ctx, scope, "output_format", {}))
                current.update(dict(value))
                self._scope_set(ctx, scope, output_format=current)
            ctx.text = f"Output {field} {'enabled' if enabled else 'disabled'}."
            ctx.logs.append(f"stage=generate command=output:{field}:{'on' if enabled else 'off'}")
            return True
        
        if cmd in {"/cache", "/cache stats"}:
            stats = _cache_stats()
            namespaces = list(stats.get("namespaces") or [])
            if not namespaces:
                ctx.text = (
                    "Cache stats:\n"
                    f"- root: {stats.get('root')}\n"
                    "- namespaces: 0\n"
                    "- files: 0\n"
                    "- size_bytes: 0"
                )
            else:
                rows = [
                    (
                        f"- {row.get('namespace')}: files={row.get('files')} "
                        f"size_bytes={row.get('size_bytes')}"
                    )
                    for row in namespaces
                ]
                ctx.text = (
                    "Cache stats:\n"
                    f"- root: {stats.get('root')}\n"
                    f"- namespaces: {stats.get('namespace_count')}\n"
                    f"- files: {stats.get('total_files')}\n"
                    f"- size_bytes: {stats.get('total_size_bytes')}\n"
                    + "\n".join(rows)
                )
            ctx.logs.append("stage=generate command=cache:stats")
            return True
        
        if cmd in {"/cache clear", "/cache clean"}:
            summary = _cache_clear()
            ctx.text = (
                "Cache cleared:\n"
                f"- root: {summary.get('root')}\n"
                f"- removed_files: {summary.get('removed_files')}\n"
                f"- removed_dirs: {summary.get('removed_dirs')}\n"
                f"- errors: {summary.get('errors')}"
            )
            ctx.logs.append("stage=generate command=cache:clear")
            return True
        
        if cmd in {"/mode", "/mode current"}:
            if scope == "chat":
                current_mode = normalize_mode_name(
                    _pick(
                        ctx.state.get("active_mode"),
                        ctx.meta.get("active_mode"),
                        ctx.state.get("mode"),
                        "chatting",
                    ),
                    allow_custom=True,
                )
                locked = bool(_pick_value(ctx.state.get("mode_lock"), ctx.meta.get("mode_lock"), False))
            else:
                current_mode = normalize_mode_name(
                    str(self._scope_get(ctx, scope, "active_mode", "chatting") or "chatting"),
                    allow_custom=True,
                )
                locked = bool(self._scope_get(ctx, scope, "mode_lock", False))
            ctx.text = f"Mode: {current_mode} ({'locked' if locked else 'auto'})"
            ctx.logs.append("stage=generate command=mode:show")
            return True

        if cmd == "/modes" or cmd.startswith("/modes "):
            target_character = str(
                ctx.state.get("active_character_id")
                or ctx.state.get("active_personality_id")
                or "asya"
            ).strip().lower() or "asya"
            if cmd.startswith("/modes "):
                raw_target = cmd.split(" ", 1)[1].strip().lower()
                if raw_target:
                    known = set(self.character_engine.list_ids()) if hasattr(self.character_engine, "list_ids") else set()
                    if known and raw_target not in known:
                        ctx.text = f"Unknown character '{raw_target}'. Available: {', '.join(sorted(known))}"
                        ctx.logs.append(f"stage=generate command=modes:unknown_character:{raw_target}")
                        return True
                    target_character = raw_target

            persona_spec = {}
            try:
                persona_spec = load_character_spec(target_character, "persona_spec", required=False)
            except Exception:
                persona_spec = {}
            persona_modes = dict(persona_spec.get("modes") or {})

            if persona_modes:
                modes = sorted(str(x).strip().lower() for x in persona_modes.keys() if str(x).strip())
                source = f"characters/{target_character}/persona_spec.json"
            else:
                modes_spec = load_spec("modes", required=False)
                modes_map = dict(modes_spec.get("modes") or {})
                modes = sorted(str(x).strip().lower() for x in modes_map.keys() if str(x).strip())
                source = "modes_spec.json"

            current_mode = normalize_mode_name(
                _pick(
                    ctx.state.get("active_mode"),
                    ctx.meta.get("active_mode"),
                    ctx.state.get("mode"),
                    "chatting",
                ),
                allow_custom=True,
            )
            if not modes:
                ctx.text = f"No modes configured for '{target_character}'."
                ctx.logs.append("stage=generate command=modes:empty")
                return True
            rows = [f"{'*' if m == current_mode else ' '} {m}" for m in modes]
            ctx.text = (
                f"Modes for {target_character}:\n"
                + "\n".join(rows)
                + f"\n(source: {source})"
            )
            ctx.logs.append(f"stage=generate command=modes:list character={target_character}")
            return True

        if cmd.startswith("/mode "):
            target_raw = cmd.replace("/mode", "", 1).strip().lower()
            if target_raw:
                if target_raw in {"off", "auto"}:
                    ctx.text = "Mode auto enabled (lock off)."
                    if scope == "chat":
                        ctx.memory_ops.append({"op": "state_mode_lock", "value": False})
                    else:
                        self._scope_set(ctx, scope, mode_lock=False)
                    ctx.logs.append("stage=generate command=mode:auto")
                    return True
                safe_raw = "".join(ch for ch in str(target_raw or "") if ch.isalnum() or ch in {"_", "-", " "}).strip()
                if not safe_raw:
                    allowed = ", ".join(list_runtime_modes())
                    ctx.text = f"Unknown mode '{target_raw}'. Available: {allowed}"
                    ctx.logs.append(f"stage=generate command=mode:unknown_token:{target_raw}")
                    return True
                target = normalize_mode_name(target_raw, allow_custom=True)
                if not str(target or "").strip():
                    allowed = ", ".join(list_runtime_modes())
                    ctx.text = f"Unknown mode '{target_raw}'. Available: {allowed}"
                    ctx.logs.append(f"stage=generate command=mode:unknown_empty:{target_raw}")
                    return True
                ctx.text = f"Mode switched to: {target} (lock on)."
                if scope == "chat":
                    ctx.memory_ops.append(
                        {
                            "op": "state_mode",
                            "value": target,
                            "reason": "manual_mode_command",
                            "confidence": 1.0,
                        }
                    )
                    ctx.memory_ops.append({"op": "state_mode_lock", "value": True})
                else:
                    self._scope_set(ctx, scope, active_mode=target, mode_lock=True)
                ctx.logs.append(f"stage=generate command=mode:{target}")
                return True

        if cmd.startswith("/mode_lock "):
            raw_value = cmd.replace("/mode_lock", "", 1).strip().lower()
            enabled = raw_value in {"on", "true", "1", "yes"}
            if raw_value not in {"on", "off", "true", "false", "1", "0", "yes", "no"}:
                ctx.text = "Usage: /mode_lock on|off"
                ctx.logs.append("stage=generate command=mode_lock:usage")
                return True
            ctx.text = f"Mode lock {'enabled' if enabled else 'disabled'}."
            if scope == "chat":
                ctx.memory_ops.append({"op": "state_mode_lock", "value": bool(enabled)})
            else:
                self._scope_set(ctx, scope, mode_lock=bool(enabled))
            ctx.logs.append(f"stage=generate command=mode_lock:{'on' if enabled else 'off'}")
            return True

        if cmd in {"/mode_lock", "/mode_lock status"}:
            if scope == "chat":
                locked = bool(_pick_value(ctx.state.get("mode_lock"), ctx.meta.get("mode_lock"), False))
            else:
                locked = bool(self._scope_get(ctx, scope, "mode_lock", False))
            ctx.text = f"Mode lock: {'on' if locked else 'off'}"
            ctx.logs.append("stage=generate command=mode_lock:show")
            return True

        if cmd in {"/persona_debug", "/persona debug"} or cmd.startswith("/persona_debug "):
            payload: dict[str, Any]
            if hasattr(self.character_engine, "get_persona_debug"):
                try:
                    payload = dict(self.character_engine.get_persona_debug(state=ctx.state, max_deltas=3) or {})
                except Exception:
                    payload = {}
            else:
                payload = {}
            if not payload:
                payload = {"active_character_id": str(ctx.state.get("active_character_id") or "default")}
            ctx.text = _format_persona_debug(payload)
            ctx.logs.append("stage=generate command=persona_debug")
            return True

        if cmd in {"/brain_debug", "/brain debug"} or cmd.startswith("/brain_debug "):
            payload: dict[str, Any]
            if hasattr(self.character_engine, "get_brain_debug"):
                try:
                    payload = dict(self.character_engine.get_brain_debug(state=ctx.state, max_actions=3) or {})
                except Exception:
                    payload = {}
            else:
                payload = {}
            if not payload:
                payload = {
                    "active_mode": str(ctx.state.get("active_mode") or ctx.state.get("mode") or "chatting"),
                    "mode_lock": bool(ctx.state.get("mode_lock", False)),
                }
            ctx.text = _format_brain_debug(payload)
            ctx.logs.append("stage=generate command=brain_debug")
            return True
            
        if cmd in {"/persona", "/persona current", "/personality"}:
            current = str(
                ctx.state.get("active_character_id")
                or ctx.state.get("active_personality_id")
                or "default"
            ).strip().lower() or "default"
            locked = bool(ctx.state.get("character_locked", False))
            ctx.text = f"Character style source: {current} ({'locked' if locked else 'auto'})"
            ctx.logs.append("stage=generate command=persona:show_alias")
            return True
        
        if cmd in {"/persona auto", "/personality auto"}:
            current = str(
                ctx.state.get("active_character_id")
                or ctx.state.get("active_personality_id")
                or "default"
            ).strip().lower() or "default"
            ctx.text = "Character auto mode enabled."
            ctx.memory_ops.append(
                {
                    "op": "state_character",
                    "value": current,
                    "locked": False,
                    "ts": now_local_ts(),
                    "reason": "manual_auto_alias_persona",
                }
            )
            ctx.memory_ops.append(
                {
                    "op": "state_personality",
                    "value": current,
                    "locked": False,
                    "blend": {
                        "active": False,
                        "from": current,
                        "to": current,
                        "step": 0,
                        "steps": 4,
                        "old_weight": 0.0,
                        "new_weight": 1.0,
                    },
                    "ts": now_local_ts(),
                    "reason": "manual_auto_alias_persona",
                }
            )
            ctx.logs.append("stage=generate command=persona:auto_alias")
            return True
        
        if cmd.startswith("/persona ") or cmd.startswith("/personality "):
            target = cmd.split(" ", 1)[1].strip().lower()
            if not target:
                return False
            known_char = set(self.character_engine.list_ids())
            if target not in known_char:
                ctx.text = f"Unknown character '{target}'. Available: {', '.join(sorted(known_char))}"
                ctx.logs.append(f"stage=generate command=persona:unknown:{target}")
                return True
            current = str(ctx.state.get("active_personality_id") or "default").strip().lower() or "default"
            try:
                self.character_engine.set_active_character(target)
            except Exception:
                # Runtime state update below is still the source of truth for current turn.
                pass
            ctx.text = f"Character switched to: {target}"
            ctx.memory_ops.append(
                {
                    "op": "state_personality",
                    "value": target,
                    "locked": True,
                    "blend": {
                        "active": False,
                        "from": target,
                        "to": target,
                        "step": 0,
                        "steps": 1,
                        "old_weight": 0.0,
                        "new_weight": 1.0,
                    },
                    "ts": now_local_ts(),
                    "reason": "manual_alias_persona_to_character",
                    "confidence": 1.0,
                }
            )
            ctx.memory_ops.append(
                {
                    "op": "state_character",
                    "value": target,
                    "locked": True,
                    "ts": now_local_ts(),
                    "reason": "manual_alias_persona_to_character",
                    "confidence": 1.0,
                }
            )
            ctx.logs.append(f"stage=generate command=persona_alias:{target}")
            return True
        return False

    def _handle_trait_command(self, ctx: PipelineContext, cmd: str) -> bool:
        parts = [x for x in cmd.split(" ") if x]
        if len(parts) < 2:
            ctx.text = "Usage: /trait list | /trait set <name> <value> | /trait remove <name>"
            return True
        action = str(parts[1]).strip().lower()
        current = str(
            ctx.state.get("active_character_id")
            or "default"
        ).strip().lower()

        if action == "list":
            traits = self.character_engine.list_traits(current)
            if not traits:
                ctx.text = f"Traits for {current}: (empty)"
                return True
            rows = []
            for key in sorted(traits.keys()):
                row = dict(traits.get(key) or {})
                rows.append(f"- {key}: {row.get('value')} (conf={row.get('confidence')})")
            ctx.text = f"Traits for {current}:\n" + "\n".join(rows)
            return True

        if action in {"remove", "del", "delete"} and len(parts) >= 3:
            name = str(parts[2]).strip().lower()
            ok = self.character_engine.remove_trait(current, name)
            ctx.text = f"Trait removed: {name}" if ok else f"Trait not found: {name}"
            ctx.logs.append(f"stage=generate command=trait:remove:{name}")
            return True

        if action == "set" and len(parts) >= 4:
            name = str(parts[2]).strip().lower()
            raw_value = str(parts[3]).strip().lower()
            if raw_value in {"true", "on", "yes", "1"}:
                value = True
            elif raw_value in {"false", "off", "no", "0"}:
                value = False
            else:
                try:
                    value = float(raw_value)
                except Exception:
                    value = raw_value
            updated = self.character_engine.set_trait(current, name, value=value, confidence=0.85)
            ctx.text = f"Trait updated: {name}={updated.get('value')} (conf={updated.get('confidence')})"
            ctx.logs.append(f"stage=generate command=trait:set:{name}")
            return True

        ctx.text = "Usage: /trait list | /trait set <name> <value> | /trait remove <name>"
        return True

    def _handle_character_delete_command(self, ctx: PipelineContext, target: str) -> bool:
        character_id = str(target or "").strip().lower()
        if not character_id:
            ctx.text = "Usage: /character delete <character_id>"
            ctx.logs.append("stage=generate command=character:delete:usage")
            return True

        known = set(self.character_engine.list_ids())
        if character_id not in known:
            ctx.text = f"Unknown character '{character_id}'. Available: {', '.join(sorted(known))}"
            ctx.logs.append(f"stage=generate command=character:delete:unknown:{character_id}")
            return True
        if character_id == "default":
            ctx.text = "Character 'default' cannot be deleted."
            ctx.logs.append("stage=generate command=character:delete:blocked_default")
            return True
        if len(known) <= 1:
            ctx.text = "Cannot delete the last remaining character."
            ctx.logs.append("stage=generate command=character:delete:blocked_last")
            return True

        current = str(
            _pick(
                ctx.state.get("active_character_id"),
                ctx.state.get("active_personality_id"),
            )
        ).strip().lower()
        if not current and hasattr(self.character_engine, "get_active_character_id"):
            try:
                current = str(self.character_engine.get_active_character_id(ctx.state) or "").strip().lower()
            except Exception:
                current = ""

        switched_to = ""
        if current == character_id:
            fallback_candidates = [x for x in sorted(known) if x != character_id]
            if not fallback_candidates:
                ctx.text = "Cannot delete the last remaining character."
                ctx.logs.append("stage=generate command=character:delete:blocked_last")
                return True
            switched_to = str(fallback_candidates[0]).strip().lower()
            try:
                try:
                    self.character_engine.set_active_character(switched_to, locked=False, switch_ts=now_local_ts())
                except TypeError:
                    self.character_engine.set_active_character(switched_to)
            except Exception:
                # Memory ops below still keep current turn state consistent.
                pass
            ctx.state["active_character_id"] = switched_to
            ctx.state["active_personality_id"] = switched_to
            ctx.memory_ops.append(
                {
                    "op": "state_character",
                    "value": switched_to,
                    "locked": False,
                    "ts": now_local_ts(),
                    "reason": "character_delete_fallback",
                }
            )
            ctx.memory_ops.append(
                {
                    "op": "state_personality",
                    "value": switched_to,
                    "locked": False,
                    "ts": now_local_ts(),
                    "reason": "character_delete_fallback",
                }
            )

        removed = self._delete_character_data(character_id)
        for cache_name in ("_meta_cache", "_profiles_cache"):
            cache = getattr(self.character_engine, cache_name, None)
            if isinstance(cache, dict):
                cache.pop(character_id, None)
        try:
            storage = getattr(self.character_engine, "storage", None)
            if storage is not None and hasattr(storage, "load_manifest") and hasattr(storage, "save_manifest"):
                manifest = dict(storage.load_manifest() or {})
                rows = [x for x in list(manifest.get("characters") or []) if isinstance(x, dict)]
                rows = [x for x in rows if str(x.get("id") or "").strip().lower() != character_id]
                manifest["characters"] = rows
                active_id = str(manifest.get("active_character_id") or "").strip().lower()
                if active_id == character_id:
                    if switched_to:
                        manifest["active_character_id"] = switched_to
                    elif rows:
                        manifest["active_character_id"] = str(rows[0].get("id") or "default").strip().lower() or "default"
                    else:
                        manifest["active_character_id"] = "default"
                storage.save_manifest(manifest)
            if storage is not None and hasattr(storage, "sync_manifest"):
                storage.sync_manifest()
            if storage is not None and hasattr(storage, "append_manifest_event"):
                storage.append_manifest_event(
                    {
                        "type": "character_deleted",
                        "character_id": character_id,
                        "switched_to": switched_to,
                        "removed_files": int(removed.get("removed_files", 0) or 0),
                        "removed_dirs": int(removed.get("removed_dirs", 0) or 0),
                        "errors": int(removed.get("errors", 0) or 0),
                    }
                )
        except Exception:
            pass

        current_ids = set(self.character_engine.list_ids())
        if character_id in current_ids:
            ctx.text = f"Character delete failed: {character_id}"
            ctx.logs.append(f"stage=generate command=character:delete:failed:{character_id}")
            return True

        details = f"Character deleted: {character_id}"
        if switched_to:
            details += f". Active character switched to: {switched_to}"
        ctx.text = details
        ctx.logs.append(f"stage=generate command=character:delete:{character_id}")
        return True

    def _delete_character_data(self, character_id: str) -> dict[str, int]:
        summary = {
            "removed_files": 0,
            "removed_dirs": 0,
            "errors": 0,
        }
        storage = getattr(self.character_engine, "storage", None)
        if storage is None:
            summary["errors"] = 1
            return summary

        targets: list[Path] = []
        for attr in ("root", "spec_root", "character_logs_root"):
            base = getattr(storage, attr, None)
            if base is None:
                continue
            try:
                base_path = Path(base).expanduser().resolve()
                target = (base_path / character_id).resolve()
                target.relative_to(base_path)
            except Exception:
                summary["errors"] += 1
                continue
            targets.append(target)

        for target in targets:
            if not target.exists():
                continue
            for row in sorted(target.rglob("*"), key=lambda p: len(p.parts), reverse=True):
                try:
                    if row.is_file():
                        row.unlink()
                        summary["removed_files"] += 1
                    elif row.is_dir():
                        os.rmdir(row)
                        summary["removed_dirs"] += 1
                except Exception:
                    summary["errors"] += 1
            try:
                if target.exists():
                    os.rmdir(target)
                    summary["removed_dirs"] += 1
            except Exception:
                if target.exists():
                    summary["errors"] += 1
        return summary

    def _build_request(self, ctx: PipelineContext) -> LLMRequest:
        scope = "chat"
        studio_row = _as_dict(_as_dict(ctx.state).get(StudioGenerator.KEY))
        if self.studio_generator.is_active({StudioGenerator.KEY: studio_row}):
            scope = "studio"
        scoped_think = self._scope_get(ctx, scope, "think", None)
        if isinstance(scoped_think, bool):
            ctx.meta["think"] = bool(scoped_think)
        scoped_verbose = self._scope_get(ctx, scope, "verbose", None)
        if isinstance(scoped_verbose, bool):
            ctx.meta["verbose"] = bool(scoped_verbose)
        elif isinstance(ctx.state.get("verbose_enabled"), bool) and "verbose" not in ctx.meta:
            ctx.meta["verbose"] = bool(ctx.state.get("verbose_enabled"))
        scoped_web_mode = str(self._scope_get(ctx, scope, "web_mode", "") or "").strip().lower()
        if scoped_web_mode in {"on", "off", "auto"}:
            ctx.meta["web_mode"] = scoped_web_mode
        scoped_output = self._scope_get(ctx, scope, "output_format", None)
        if isinstance(scoped_output, dict):
            ctx.state["output_format"] = _coerce_output_format_state(scoped_output)
        if str(ctx.profile or "").strip().upper() == PROFILE_AUTONOMOUS:
            ctx.meta["think"] = True
        if ctx.route == "chat":
            if ctx.prompt_messages:
                messages = list(ctx.prompt_messages)
            else:
                prompt_state = dict(ctx.state or {})
                merged_tags = _as_dict(prompt_state.get("context_tags"))
                merged_tags.update(dict(ctx.tags or {}))
                prompt_state["context_tags"] = merged_tags
                if ctx.plan:
                    prompt_state["plan"] = ctx.plan
                prompt_state["active_task"] = dict(_as_dict(ctx.state.get("active_task")))
                prompt_state["active_tasks"] = [dict(x) for x in list(_as_list(ctx.state.get("active_tasks"))) if isinstance(x, dict)]
                web_intent = str(ctx.tags.get("web_query_intent") or "").strip().lower()
                web_category = str(
                    _pick_value(
                        ctx.meta.get("web_primary_category"),
                        ctx.tags.get("web_primary_category"),
                        _as_dict(ctx.meta.get("web_query_classification")).get("primary_category"),
                        "",
                    )
                    or ""
                ).strip().lower()
                memory_context_for_prompt = _as_dict(ctx.memory_context)
                retrieved_for_prompt = list(_as_list(ctx.retrieved_memories))
                web_evidence_quality = _as_dict(ctx.meta.get("web_evidence_quality"))
                web_evidence_context = _as_dict(ctx.meta.get("web_evidence_context"))
                web_used_for_prompt = False
                if web_evidence_context and _to_bool(
                    _pick_value(ctx.tags.get("web_used"), ctx.meta.get("web_used"), False),
                    default=False,
                ):
                    web_used_for_prompt = True
                    factual_response_mode = _resolve_web_factual_response_mode(
                        web_intent=web_intent,
                        web_category=web_category,
                        web_evidence_context=web_evidence_context,
                        quality=web_evidence_quality,
                    )
                    factual_response_mode, web_used_for_prompt, self_fact_web_guard = _apply_self_fact_factual_mode_guard(
                        factual_response_mode=factual_response_mode,
                        web_used=web_used_for_prompt,
                        memory_context=memory_context_for_prompt,
                    )
                    if self_fact_web_guard:
                        ctx.meta["prompt_self_fact_guard"] = dict(self_fact_web_guard)
                        ctx.meta["skip_factual_context_isolation"] = True
                    if factual_response_mode == _SELF_MEMORY_EXACT_MODE:
                        _apply_self_memory_exact_turn_guard(ctx, prompt_state=prompt_state)
                        web_used_for_prompt = False
                        web_evidence_quality = {}
                        web_evidence_context = {}
                    if factual_response_mode:
                        ctx.meta["factual_response_mode"] = factual_response_mode
                        ctx.tags["factual_response_mode"] = factual_response_mode
                        if factual_response_mode == _SELF_MEMORY_EXACT_MODE:
                            ctx.meta["web_skip_citations"] = True
                            ctx.tags["self_memory_exact"] = "true"
                        web_evidence_context = dict(web_evidence_context)
                        web_evidence_context.setdefault("factual_response_mode", factual_response_mode)
                        ctx.meta["web_evidence_context"] = dict(web_evidence_context)
                    else:
                        ctx.meta.pop("factual_response_mode", None)
                        ctx.tags.pop("factual_response_mode", None)
                        ctx.tags.pop("self_memory_exact", None)
                    if web_used_for_prompt:
                        prompt_state["web_evidence_context"] = dict(web_evidence_context)
                    else:
                        prompt_state.pop("web_evidence_context", None)
                    if web_used_for_prompt and factual_response_mode:
                        memory_context_for_prompt, retrieved_for_prompt, context_isolation_debug = _isolate_factual_prompt_context(
                            memory_context=memory_context_for_prompt,
                            retrieved_memories=retrieved_for_prompt,
                            web_intent=factual_response_mode,
                        )
                        if context_isolation_debug:
                            ctx.meta["prompt_context_isolation"] = dict(context_isolation_debug)
                if memory_context_for_prompt:
                    ctx.memory_context = dict(memory_context_for_prompt)
                    ctx.state["memory_context"] = dict(memory_context_for_prompt)
                    prompt_state["memory_context"] = dict(memory_context_for_prompt)
                    context_blocks = _as_dict(memory_context_for_prompt.get("blocks"))
                    # Summary больше не используется как источник continuity.
                    # Это только fallback кеш для prompt_engine.
                    # Не записываем summary_hint в prompt_state["long_summary"]/["dialog_summary"].
                    summary_hint = sanitize_session_summary_text(context_blocks.get("session_summary") or "")
                    # Summary теперь только в context_blocks["session_summary"] для prompt_engine
                else:
                    ctx.memory_context = {}
                    ctx.state.pop("memory_context", None)
                ctx.retrieved_memories = list(retrieved_for_prompt)
                self._build_persona_snapshot_payload(ctx=ctx, prompt_state=prompt_state)
                ctx.prompt_pack = self.character_runtime.build(
                    state=prompt_state,
                    user_msg=ctx.clean_user_msg,
                    retrieved_memories=retrieved_for_prompt,
                    traits=ctx.traits,
                    policies=ctx.policies,
                )
                if memory_context_for_prompt:
                    ctx.prompt_pack = _apply_memory_context_to_prompt_pack(
                        ctx.prompt_pack,
                        memory_context=memory_context_for_prompt,
                        selected_memories=retrieved_for_prompt,
                    )
                ctx.prompt_pack = _apply_web_evidence_to_prompt_pack(
                    ctx.prompt_pack,
                    web_evidence_context=(
                        web_evidence_context
                        if web_used_for_prompt
                        else {}
                    ),
                )
                messages = _messages_from_prompt_pack(ctx.prompt_pack)
        else:
            messages = [
                Message(role="system", content="You are MMis assistant. Reply concisely."),
                Message(role="user", content=ctx.clean_user_msg),
            ]

        tools = _parse_tools(_pick_value(ctx.meta.get("tools"), ctx.policies.get("tools"), ctx.state.get("tools")))
        agent_loop_enabled = _should_enable_agent_loop(ctx)
        if agent_loop_enabled:
            tools = _merge_tool_specs(tools, _agent_loop_tools(ctx))
            agent_loop_enabled = bool(tools)
            if agent_loop_enabled:
                messages = _inject_agent_loop_messages(messages, tools)

        model = _pick(
            ctx.meta.get("model"),
            ctx.state.get("model"),
            ctx.policies.get("model"),
        )
        response_format = _pick_value(
            ctx.meta.get("response_format"),
            ctx.policies.get("response_format"),
            None,
        )
        req_max_tokens = _to_int(_pick_value(ctx.meta.get("max_tokens"), ctx.policies.get("max_tokens"), None), None)
        if req_max_tokens is None and str(ctx.profile or "").strip().upper() == PROFILE_AUTONOMOUS:
            provider_name = str(type(self.provider).__name__ or "").strip().lower()
            req_max_tokens = -1 if "ollama" in provider_name else None
        if req_max_tokens is None:
            verbosity = _to_float(
                _pick_value(
                    _as_dict(ctx.meta.get("dialog_mode")).get("verbosity_level"),
                    ctx.meta.get("verbosity_level"),
                    ctx.tags.get("dialog_verbosity_level"),
                ),
                None,
            )
            if verbosity is not None:
                req_max_tokens = _verbosity_to_max_tokens(verbosity)
        req_metadata = _request_metadata(ctx)
        if agent_loop_enabled:
            req_metadata["agent_loop"] = True
            req_metadata["agent_loop_tool_limit"] = _agent_loop_tool_limit(ctx)
        return LLMRequest(
            model=model,
            messages=messages,
            temperature=_to_float(_pick_value(ctx.meta.get("temperature"), ctx.policies.get("temperature"), None), None),
            top_p=_to_float(_pick_value(ctx.meta.get("top_p"), ctx.policies.get("top_p"), None), None),
            repeat_penalty=_to_float(_pick_value(ctx.meta.get("repeat_penalty"), ctx.policies.get("repeat_penalty"), None), None),
            seed=_to_int(_pick_value(ctx.meta.get("seed"), ctx.policies.get("seed"), None), None),
            max_tokens=req_max_tokens,
            stop=[str(x) for x in _as_list(_pick_value(ctx.meta.get("stop"), ctx.policies.get("stop"), [])) if str(x)],
            json_mode=bool(_pick_value(ctx.meta.get("json_mode"), ctx.policies.get("json_mode"), False)),
            response_format=(dict(response_format) if isinstance(response_format, dict) else None),
            tools=tools,
            metadata=req_metadata,
        )


class PostprocessStage(PipelineStage):
    name = "postprocess"

    def __init__(self, evaluator: ResponseConstraintEvaluator | None = None):
        self._evaluator = evaluator or ResponseConstraintEvaluator()

    def run(self, ctx: PipelineContext) -> PipelineContext:
        text = _normalize_text(ctx.text)
        if not text:
            ctx.text = text
            ctx.logs.append("stage=postprocess empty")
            return ctx

        verify_cfg = _as_dict(ctx.policies.get("verify"))
        require_json = bool(verify_cfg.get("require_json", False))
        json_mode = bool(ctx.meta.get("json_mode", False))
        response_format = _pick_value(ctx.meta.get("response_format"), ctx.policies.get("response_format"), None)
        preserve_json = bool(require_json or json_mode or isinstance(response_format, dict))
        text, unwrapped = _unwrap_safety_output_json(text, preserve_json=preserve_json)
        if unwrapped:
            ctx.logs.append("stage=postprocess unwrap=safety_output")

        if not _looks_like_json(text):
            before_hygiene = text
            text = _enforce_response_hygiene(text)
            if text != before_hygiene:
                ctx.logs.append("stage=postprocess hygiene=applied")
            text = _normalize_text(text)
            text = _enforce_response_hygiene(text)
            dialog_mode = _resolve_dialog_mode(ctx)
            address_terms_policy = _resolve_address_terms_policy(ctx)
            text, hard_applied = self._evaluator.enforce(
                text,
                user_text=ctx.clean_user_msg,
                dialog_mode=dialog_mode,
                metadata=ctx.meta,
                address_terms_policy=address_terms_policy,
                user_addressing=_resolve_user_addressing(ctx),
            )
            if hard_applied:
                ctx.logs.append("stage=postprocess hard_constraints=" + ",".join(hard_applied))
            if any(str(x).startswith("terms_") for x in hard_applied):
                removed = [str(x) for x in hard_applied if "removed" in str(x)]
                kept = [str(x) for x in hard_applied if "kept" in str(x)]
                if removed:
                    ctx.logs.append("stage=postprocess terms_removed=" + ",".join(removed))
                if kept:
                    ctx.logs.append("stage=postprocess terms_kept=" + ",".join(kept))

            factual_response_mode = str(
                _pick_value(
                    ctx.meta.get("factual_response_mode"),
                    ctx.tags.get("factual_response_mode"),
                    "",
                )
                or ""
            ).strip().lower()
            self_memory_exact = factual_response_mode == _SELF_MEMORY_EXACT_MODE
            web_intent = "" if self_memory_exact else str(ctx.tags.get("web_query_intent") or "").strip().lower()
            web_style = "" if self_memory_exact else str(ctx.tags.get("web_response_style") or "").strip().lower()
            if not self_memory_exact:
                web_fixed = _apply_time_sensitive_web_failsafe(
                    text,
                    web_intent=web_intent,
                    web_response_style=web_style,
                )
                if web_fixed != text:
                    text = web_fixed
                    ctx.logs.append(f"stage=postprocess web_failsafe=applied intent={web_intent}")

                web_sync_fixed, web_sync_changed = _apply_web_tool_sync_guard(
                    text,
                    meta=_as_dict(ctx.meta),
                    web_evidence_context=_as_dict(ctx.meta.get("web_evidence_context")),
                    web_intent=web_intent,
                )
                if web_sync_changed:
                    text = web_sync_fixed
                    ctx.logs.append(f"stage=postprocess web_tool_sync=applied intent={web_intent}")

                web_cautious_fixed, web_cautious_changed = _apply_web_factual_caution_guard(
                    text,
                    meta=_as_dict(ctx.meta),
                    web_evidence_context=_as_dict(ctx.meta.get("web_evidence_context")),
                    web_intent=web_intent,
                )
                if web_cautious_changed:
                    text = web_cautious_fixed
                    ctx.logs.append(f"stage=postprocess web_factual_caution=applied intent={web_intent}")

                web_fx_fixed, web_fx_changed, web_fx_debug = _apply_web_fx_response_guard(
                    text,
                    meta=_as_dict(ctx.meta),
                    web_evidence_context=_as_dict(ctx.meta.get("web_evidence_context")),
                    web_intent=web_intent,
                )
                if web_fx_debug:
                    ctx.meta["fx_response_format_debug"] = dict(web_fx_debug)
                if web_fx_changed:
                    text = web_fx_fixed
                    ctx.meta["web_skip_citations"] = True
                    ctx.logs.append(
                        "stage=postprocess web_fx_format=applied "
                        f"pairs={int(web_fx_debug.get('pair_count', 0) or 0)} "
                        f"cautious={str(bool(web_fx_debug.get('cautious'))).lower()}"
                    )

            temporal_fixed, temporal_changed = _apply_temporal_consistency_guard(
                text=text,
                user_text=ctx.clean_user_msg,
                meta=ctx.meta,
            )
            if temporal_changed:
                text = temporal_fixed
                ctx.logs.append("stage=postprocess temporal_grounding=applied")

            if _should_apply_echo_guard(ctx.clean_user_msg) and _looks_like_echo_response(answer=text, user_msg=ctx.clean_user_msg):
                text = _echo_fallback_text(ctx.clean_user_msg)
                ctx.logs.append("stage=postprocess echo_guard=applied")

            add_emoji = bool(_as_dict(ctx.policies.get("postprocess")).get("add_emoji", False))
            if add_emoji:
                mood = str(ctx.tags.get("mood") or "neutral")
                emoji = {"happy": " :)", "angry": " !!", "neutral": ""}.get(mood, "")
                if emoji and not text.endswith(emoji.strip()):
                    text = f"{text}{emoji}"

            # Citations are mandatory for web-backed answers, but remain compact and adaptive.
            web_used = _to_bool(_pick_value(ctx.tags.get("web_used"), ctx.meta.get("web_used"), False), default=False)
            if self_memory_exact:
                web_used = False
                ctx.meta["web_skip_citations"] = True
            skip_citations = _to_bool(ctx.meta.get("web_skip_citations"), default=False)
            compact_citations = [str(x) for x in _as_list(_as_dict(ctx.meta).get("web_citations")) if str(x).strip()]
            if web_used and compact_citations and not skip_citations:
                classification = _as_dict(_as_dict(ctx.meta).get("web_query_classification"))
                text = format_postprocess_citation_suffix(
                    text=text,
                    compact_citations=compact_citations,
                    classification=SimpleNamespace(query_type=str(classification.get("query_type") or "")),
                )
                ctx.logs.append(f"stage=postprocess citations=applied count={len(compact_citations)}")
            if web_used and factual_response_mode:
                quality = _as_dict(
                    _pick_value(
                        ctx.meta.get("web_evidence_quality"),
                        ctx.state.get("web_evidence_quality"),
                        {},
                    )
                )
                cautious_answer = bool(
                    _pick_value(
                        ctx.meta.get("web_cautious_synthesis"),
                        quality.get("cautious_synthesis"),
                        False,
                    )
                )
                _emit_turn_summary(
                    ctx,
                    "factual_response_summary",
                    summary=(
                        f"mode={factual_response_mode} web_used={str(bool(web_used)).lower()} "
                        f"quality={float(_to_float(quality.get('score'), 0.0) or 0.0):.3f} "
                        f"conflict={float(_to_float(quality.get('conflict_severity'), 0.0) or 0.0):.3f} "
                        f"confidence={float(_to_float(quality.get('final_factual_confidence'), 0.0) or 0.0):.3f} "
                        f"cautious={str(cautious_answer).lower()}"
                    ),
                    factual_mode=str(factual_response_mode),
                    web_used=bool(web_used),
                    evidence_quality=float(_to_float(quality.get("score"), 0.0) or 0.0),
                    conflict_severity=float(_to_float(quality.get("conflict_severity"), 0.0) or 0.0),
                    final_factual_confidence=float(_to_float(quality.get("final_factual_confidence"), 0.0) or 0.0),
                    cautious_answer=bool(cautious_answer),
                    context_isolation_applied=bool(ctx.meta.get("prompt_context_isolation")),
                )

        updated_address_terms = _update_address_terms_after_response(ctx=ctx, text=text)
        if updated_address_terms is not None:
            ctx.state["address_terms"] = updated_address_terms
            ctx.memory_ops.append({"op": "state_address_terms", "value": dict(updated_address_terms)})
        ctx.text = text
        ctx.logs.append("stage=postprocess")
        return ctx


class ToolRouterStage(PipelineStage):
    name = "tool_router"

    def run(self, ctx: PipelineContext) -> PipelineContext:
        calls = list(ctx.tool_calls or [])
        if not calls:
            calls = _extract_tool_calls(ctx.text)
        ctx.tool_calls = calls
        if not calls:
            ctx.logs.append("stage=tool_router calls=0")
            return ctx

        auto_exec = bool(ctx.meta.get("auto_execute_tools", True))
        executor = ctx.meta.get("tool_executor")
        if auto_exec and callable(executor):
            for call in calls:
                try:
                    result = executor(call)
                    ctx.tool_results.append({"tool": call.get("tool"), "ok": True, "result": result})
                except Exception as exc:
                    ctx.tool_results.append({"tool": call.get("tool"), "ok": False, "error": str(exc)})
        else:
            for call in calls:
                ctx.ui_actions.append({"type": "tool_call", "payload": call, "execute": False})

        if ctx.tool_results:
            ctx.memory_ops.append({"op": "tool_results", "items": list(ctx.tool_results)})
            ok_count = sum(1 for x in ctx.tool_results if bool(x.get("ok")))
            ctx.logs.append(f"stage=tool_router executed={ok_count}/{len(ctx.tool_results)}")
        else:
            ctx.logs.append(f"stage=tool_router calls={len(calls)} no_exec")
        return ctx


class VerifyStage(PipelineStage):
    name = "verify"

    def run(self, ctx: PipelineContext) -> PipelineContext:
        text = str(ctx.text or "")
        verify_cfg = _as_dict(ctx.policies.get("verify"))
        require_json = bool(verify_cfg.get("require_json", False))
        forbidden = [str(x).strip().lower() for x in _as_list(verify_cfg.get("forbidden_terms")) if str(x).strip()]

        if require_json and not _looks_like_json(text):
            ctx.errors.append("verify:json_required")
            ctx.text = '{"error":"invalid_json_output"}'

        lowered = text.lower()
        if forbidden and any(term and term in lowered for term in forbidden):
            ctx.errors.append("verify:forbidden_content")
            ctx.text = "I cannot answer in this form. Please rephrase."

        if _looks_like_json(ctx.text):
            try:
                json.loads(ctx.text)
            except Exception:
                ctx.errors.append("verify:json_parse_error")
                if require_json:
                    ctx.text = '{"error":"json_parse_error"}'

        ctx.logs.append("stage=verify ok" if not ctx.errors else f"stage=verify errors={len(ctx.errors)}")
        return ctx


class OutputFormatStage(PipelineStage):
    name = "output_format"

    def __init__(self, provider: LLMProviderBase):
        self._provider = provider
        self._output_spec = load_spec("output", required=False)

    def run(self, ctx: PipelineContext) -> PipelineContext:
        text = _normalize_text(ctx.text)
        force_format = bool(ctx.meta.get("force_output_format", False))
        if ctx.route != "chat" and not force_format:
            merged_output = dict(ctx.structured_output or {})
            merged_output.update({
                "parameters": None,
                "summary": None,
                "text": text,
                "formatted": False,
            })
            ctx.structured_output = merged_output
            ctx.logs.append("stage=output_format skipped(route)")
            return ctx

        verify_cfg = _as_dict(ctx.policies.get("verify"))
        require_json = bool(verify_cfg.get("require_json", False))
        json_mode = bool(ctx.meta.get("json_mode", False))
        response_format = _pick_value(ctx.meta.get("response_format"), ctx.policies.get("response_format"), None)
        if require_json or json_mode or isinstance(response_format, dict):
            merged_output = dict(ctx.structured_output or {})
            merged_output.update({
                "parameters": None,
                "summary": None,
                "text": text,
                "formatted": False,
            })
            ctx.structured_output = merged_output
            ctx.logs.append("stage=output_format skipped(json_mode)")
            return ctx

        self._output_spec = load_spec("output", required=False)
        mode = normalize_mode_name(
            _pick(
                ctx.state.get("active_mode"),
                ctx.meta.get("active_mode"),
                ctx.state.get("mode"),
                "chatting",
            ),
            allow_custom=True,
        )
        strict_modes = {
            str(x).strip().lower()
            for x in list(self._output_spec.get("default_enabled_modes") or ["engineer", "debugger", "planner"])
            if str(x).strip()
        }
        strict_default = mode in strict_modes
        output_format = _coerce_output_format_state(
            _pick_value(ctx.state.get("output_format"), ctx.meta.get("output_format"), {})
        )
        manual_params = output_format.get("show_parameters")
        manual_summary = output_format.get("show_summary")
        show_parameters = bool(manual_params) if isinstance(manual_params, bool) else bool(strict_default and self._spec_default_parameters())
        show_summary = bool(manual_summary) if isinstance(manual_summary, bool) else bool(self._summary_enabled_for_mode(mode, strict_default))

        parameters, parameters_lines = self._build_parameters(ctx, mode=mode)
        summary = ""
        if show_summary:
            summary = self._generate_summary(ctx, text)
            if not summary:
                summary = "-"

        formatted = bool(show_parameters or show_summary)
        if formatted:
            ctx.text = _render_output_blocks(
                parameters_lines=parameters_lines,
                summary=summary,
                text=text,
                include_parameters=bool(show_parameters),
                include_summary=bool(show_summary),
                order=self._output_order(),
            )
        else:
            ctx.text = text

        merged_output = dict(ctx.structured_output or {})
        merged_output.update({
            "parameters": (dict(parameters) if show_parameters else None),
            "summary": (str(summary) if show_summary else None),
            "text": str(text),
            "formatted": bool(formatted),
        })
        ctx.structured_output = merged_output
        ctx.logs.append(
            "stage=output_format "
            f"mode={mode} formatted={int(formatted)} "
            f"parameters={int(bool(show_parameters))} summary={int(bool(show_summary))}"
        )
        return ctx

    def _build_parameters(self, ctx: PipelineContext, *, mode: str) -> tuple[dict[str, Any], list[str]]:
        mood = str(
            _pick(
                ctx.state.get("mood"),
                ctx.tags.get("mood"),
                _as_dict(ctx.state.get("context_tags")).get("mood"),
                "neutral",
            )
        ).strip().lower() or "neutral"
        intent = str(ctx.tags.get("intent") or "").strip().lower()
        emotion = str(ctx.tags.get("mood") or "").strip().lower()
        topics = _extract_topics_from_tags(ctx.tags)
        traits = _extract_output_traits(ctx.state, ctx.traits)
        tags = [str(x).strip().lower() for x in _as_list(ctx.tags.get("metadata_tags")) if str(x).strip()]
        completion_tokens = max(0, int(_to_int(_as_dict(ctx.stats).get("eval_count"), 0) or 0))
        answer_estimated = max(0, int(estimate_tokens(str(ctx.text or "")) or 0))
        thinking_tokens = max(0, int(estimate_tokens(str(ctx.thinking or "")) or 0))

        if completion_tokens > 0:
            if thinking_tokens > 0:
                # If provider reports completion tokens, keep totals consistent.
                answer_tokens = max(0, int(completion_tokens - thinking_tokens))
                if answer_tokens <= 0:
                    answer_tokens = answer_estimated
            else:
                # Thinking can be hidden/non-rendered; infer from completion budget.
                answer_tokens = answer_estimated or completion_tokens
                inferred_thinking = max(0, int(completion_tokens - answer_tokens))
                if inferred_thinking > 0:
                    thinking_tokens = inferred_thinking
        else:
            answer_tokens = answer_estimated

        total_tokens = max(0, int(answer_tokens + thinking_tokens))
        character_id = str(
            _pick(
                ctx.state.get("active_character_id"),
                ctx.state.get("active_personality_id"),
                ctx.meta.get("character_id"),
                "default",
            )
        ).strip().lower() or "default"
        base = {
            "character_id": character_id,
            "mode": mode,
            "mood": mood,
            "traits": traits,
            "intent": intent,
            "emotion": emotion,
            "topics": topics,
            "tags": tags,
            "thinking_tokens": thinking_tokens,
            "answer_tokens": answer_tokens,
            "total_tokens": total_tokens,
        }
        fields = [str(x).strip() for x in list(self._output_spec.get("parameters_fields") or []) if str(x).strip()]
        selected = _select_parameter_fields(base, fields=fields)
        rendered_lines = _render_parameters_lines(selected, field_order=(fields or None))
        return selected, rendered_lines

    def _summary_enabled_for_mode(self, mode: str, strict_default: bool) -> bool:
        summary_cfg = _as_dict(self._output_spec.get("summary"))
        if not bool(summary_cfg.get("enabled", self._output_spec.get("show_summary", True))):
            return False
        required_modes = {
            str(x).strip().lower()
            for x in list(summary_cfg.get("required_modes") or [])
            if str(x).strip()
        }
        if required_modes:
            return mode in required_modes
        return bool(strict_default)

    def _spec_default_parameters(self) -> bool:
        return bool(self._output_spec.get("show_parameters", True))

    def _output_order(self) -> list[str]:
        raw = [str(x).strip().lower() for x in list(self._output_spec.get("order") or []) if str(x).strip()]
        allowed = {"parameters", "summary", "response"}
        out = [x for x in raw if x in allowed]
        if "response" not in out:
            out.append("response")
        return out

    def _generate_summary(self, ctx: PipelineContext, text: str) -> str:
        source = _normalize_text(text)
        if not source:
            return ""
        summary_cfg = _as_dict(self._output_spec.get("summary"))
        strategy = str(summary_cfg.get("strategy") or "mini_pass").strip().lower()
        max_sentences = max(1, int(summary_cfg.get("max_sentences") or 2))
        if strategy in {"mini_pass", "mini_pass_fallback", "llm"}:
            mini = self._summary_mini_pass(ctx, source)
            if mini:
                return _limit_summary_sentences(mini, max_sentences=max_sentences)
            return _fallback_summary_from_text(source, max_sentences=max_sentences)
        return _fallback_summary_from_text(source, max_sentences=max_sentences)

    def _summary_mini_pass(self, ctx: PipelineContext, text: str) -> str:
        model = str(ctx.stats.get("served_model") or ctx.meta.get("model") or "").strip()
        system_prompt = "Summarize assistant response in 1-2 concise sentences. Keep key action points. No bullet list."
        user_prompt = f"Response:\n{text}\n\nShort summary:"
        try:
            result = run_task_model(
                "summary_mini_pass",
                user_prompt,
                system_prompt=system_prompt,
                metadata={
                    "trace_id": str(ctx.meta.get("trace_id") or f"summary_{int(time.time() * 1000)}"),
                    "summary_mini_pass": True,
                },
                max_output_chars=240,
                max_retries=1,
                context=_turn_log_context(ctx),
            )
            summary = _normalize_text(str(result.text or ""))
            if summary and not _looks_like_json(summary):
                ctx.logs.append("stage=output_format summary_mini_pass=task_model")
                return _squeeze_summary(summary)
        except Exception as exc:
            ctx.logs.append(f"stage=output_format summary_mini_pass_task_model_error={type(exc).__name__}")
        request = LLMRequest(
            model=model,
            messages=[
                Message(
                    role="system",
                    content=system_prompt,
                ),
                Message(
                    role="user",
                    content=user_prompt,
                ),
            ],
            temperature=0.2,
            max_tokens=96,
            metadata={
                "trace_id": str(ctx.meta.get("trace_id") or f"summary_{int(time.time() * 1000)}"),
                "summary_mini_pass": True,
            },
        )
        try:
            response = self._provider.generate(request)
        except Exception as exc:
            ctx.logs.append(f"stage=output_format summary_mini_pass_error={type(exc).__name__}")
            return ""
        ctx.logs.append("stage=output_format summary_mini_pass=legacy")
        summary = _normalize_text(str(response.text or ""))
        if not summary:
            return ""
        if _looks_like_json(summary):
            return ""
        return _squeeze_summary(summary)


class MemoryWriteStage(PipelineStage):
    name = "memory_write"

    def __init__(self, memory_core: MemoryCoreAdapter | None = None, memory_manager: Any | None = None):
        self.memory_core = memory_core or memory_manager

    def run(self, ctx: PipelineContext) -> PipelineContext:
        studio_active = bool(_as_dict(_as_dict(ctx.state).get(StudioGenerator.KEY)).get("active", False))
        if studio_active:
            ctx.logs.append("stage=memory_write skipped(scoped_dialog)")
            return ctx
        store_turn = bool(ctx.meta.get("store_turn", True))
        if not store_turn:
            ctx.logs.append("stage=memory_write skipped")
            return ctx
        if ctx.route == "command" and not bool(ctx.meta.get("store_command_turns", False)):
            ctx.logs.append("stage=memory_write skipped(command)")
            return ctx

        context = _turn_log_context(ctx)
        self._update_task_continuity_from_assistant_reply(ctx)
        if ctx.route in {"chat", "command"} and ctx.clean_user_msg:
            turn_tags = self._build_turn_tags(ctx)
            ctx.memory_ops.append(
                {
                    "op": "turn_user",
                    "text": ctx.clean_user_msg,
                    "tags": turn_tags,
                    "ts": now_local_ts(),
                    "trace_id": context.get("trace_id"),
                    "request_id": context.get("request_id"),
                    "turn_id": context.get("turn_id"),
                    "conversation_id": context.get("conversation_id"),
                }
            )
        if ctx.route in {"chat", "command"} and ctx.text:
            turn_tags = self._build_turn_tags(ctx)
            ctx.memory_ops.append(
                {
                    "op": "turn_assistant",
                    "text": str(ctx.text),
                    "thinking": str(ctx.thinking or ""),
                    "tags": turn_tags,
                    "ts": now_local_ts(),
                    "trace_id": context.get("trace_id"),
                    "request_id": context.get("request_id"),
                    "turn_id": context.get("turn_id"),
                    "conversation_id": context.get("conversation_id"),
                }
            )
        if ctx.prompt_pack is not None:
            ctx.memory_ops.append(
                {
                    "op": "prompt_stats",
                    "tokens": dict(ctx.prompt_pack.token_usage),
                    "cuts": dict(ctx.prompt_pack.cut_info),
                }
            )
            # Разрываем summary-петлю: не записываем conversation_summary,
            # если long_summary пришёл из dialog_summary/rolling_summary.
            # Иначе summary становится самоподдерживающимся кешем, а не живым слоем памяти.
            long_summary = str(ctx.prompt_pack.blocks.get("long_summary") or "").strip()
            if long_summary and not is_low_quality_session_summary(long_summary):
                # Проверяем, не является ли long_summary просто копией dialog_summary
                dialog_summary = str(ctx.state.get("dialog_summary") or "").strip()
                rolling_summary = str(ctx.state.get("rolling_summary") or "").strip()
                long_summary_from_existing_summary = (
                    dialog_summary and long_summary == dialog_summary
                ) or (
                    rolling_summary and long_summary == rolling_summary
                )
                
                if not long_summary_from_existing_summary:
                    # Это новый summary, можно записывать
                    ctx.memory_ops.append(
                        {
                            "op": "conversation_summary",
                            "text": long_summary,
                            "topic_thread_id": str(ctx.state.get("topic_thread_id") or ctx.meta.get("topic_thread_id") or ""),
                            "topic_key": str(ctx.state.get("topic_key") or ctx.meta.get("topic_key") or ""),
                            "topic_title": str(
                                ctx.state.get("topic_thread_title")
                                or ctx.meta.get("topic_title")
                                or ctx.meta.get("topic_thread_title")
                                or ""
                            ),
                            "related_topic_thread_ids": list(ctx.state.get("related_topic_thread_ids") or []),
                            "ts": now_local_ts(),
                            "trace_id": context.get("trace_id"),
                            "request_id": context.get("request_id"),
                            "turn_id": context.get("turn_id"),
                            "conversation_id": context.get("conversation_id"),
                        }
                    )
                else:
                    # Summary не изменился — это петля, пропускаем запись
                    ctx.logs.append("stage=memory_write conversation_summary skipped(loop_detected)")
        ctx.logs.append(f"stage=memory_write ops={len(ctx.memory_ops)}")
        op_summary = _queued_memory_ops_summary(ctx.memory_ops)
        _emit_turn_summary(
            ctx,
            "memory_write_summary",
            summary=(
                f"queued={op_summary['queued']} turn_user={op_summary['turn_user']} "
                f"turn_assistant={op_summary['turn_assistant']} "
                f"web_items={op_summary['web_memory_items']} "
                f"summaries={op_summary['conversation_summary']}"
            ),
            route=str(ctx.route or ""),
            queue_only=True,
            queued_ops=int(op_summary["queued"]),
            queued_turn_user=int(op_summary["turn_user"]),
            queued_turn_assistant=int(op_summary["turn_assistant"]),
            queued_web_memory_writes=int(op_summary["web_memory_write"]),
            queued_web_memory_items=int(op_summary["web_memory_items"]),
            queued_conversation_summaries=int(op_summary["conversation_summary"]),
            queued_prompt_stats=int(op_summary["prompt_stats"]),
        )
        return ctx

    def _build_turn_tags(self, ctx: PipelineContext) -> dict[str, Any]:
        turn_tags = dict(ctx.tags)
        turn_tags["active_mode"] = normalize_mode_name(
            _pick(ctx.state.get("active_mode"), ctx.meta.get("active_mode"), ctx.state.get("mode"), "chatting"),
            allow_custom=True,
        )
        personality_id = str(
            ctx.state.get("active_character_id")
            or ctx.state.get("active_personality_id")
            or ""
        ).strip().lower()
        if personality_id:
            turn_tags["personality_id"] = personality_id
        turn_tags["visible_chat_id"] = str(
            _pick(ctx.meta.get("conversation_id"), ctx.state.get("conversation_id"), "")
        ).strip()
        turn_tags["topic_thread_id"] = str(ctx.state.get("topic_thread_id") or ctx.meta.get("topic_thread_id") or "").strip()
        turn_tags["topic_key"] = str(ctx.state.get("topic_key") or ctx.meta.get("topic_key") or "").strip()
        turn_tags["topic_title"] = str(
            ctx.state.get("topic_thread_title")
            or ctx.meta.get("topic_title")
            or ctx.meta.get("topic_thread_title")
            or ""
        ).strip()
        turn_tags["topic_route_reason"] = str(
            ctx.state.get("topic_route_reason")
            or ctx.meta.get("topic_route_reason")
            or ""
        ).strip()
        turn_tags["topic_route_score"] = _to_float(
            _pick_value(ctx.state.get("topic_route_score"), ctx.meta.get("topic_route_score"), 0.0),
            0.0,
        ) or 0.0
        turn_tags["related_topic_thread_ids"] = [
            str(item).strip()
            for item in list(ctx.state.get("related_topic_thread_ids") or ctx.meta.get("related_topic_thread_ids") or [])
            if str(item).strip()
        ]
        return turn_tags

    def _update_task_continuity_from_assistant_reply(self, ctx: PipelineContext) -> None:
        active_task = dict(ctx.state.get("active_task") or {})
        if not active_task or not str(ctx.text or "").strip():
            return

        updated_task = dict(active_task)
        updated_task["status"] = "active"
        updated_task["planner_source"] = "assistant_reply"
        next_steps = [
            str(match.group(1) or "").strip().rstrip(".")
            for match in re.finditer(r"(?m)^\s*\d+\.\s+(.+?)\s*$", str(ctx.text or ""))
            if str(match.group(1) or "").strip()
        ]
        if next_steps:
            updated_task["next_steps"] = next_steps
        updated_task["open_questions"] = []
        ctx.state["active_task"] = updated_task
        ctx.state["open_questions"] = []
        if list(updated_task.get("decisions") or []):
            ctx.state["current_decisions"] = list(updated_task.get("decisions") or [])
            ctx.state["active_tasks"] = [dict(updated_task)]
        ctx.state["active_goal"] = str(
            updated_task.get("current_goal") or updated_task.get("summary_short") or ctx.state.get("active_goal") or ""
        ).strip()

        memory_core = ctx.meta.get("memory_core") or ctx.meta.get("memory_manager") or self.memory_core
        continuity_snapshot = {}
        if memory_core is not None and hasattr(memory_core, "update_task_continuity"):
            try:
                continuity_snapshot = dict(
                    memory_core.update_task_continuity(
                        namespace=str(ctx.meta.get("conversation_id") or ctx.state.get("conversation_id") or "default"),
                        active_task=dict(updated_task),
                        previous_active_task=dict(active_task),
                        source="assistant_reply",
                        now_ts=_pick_value(ctx.meta.get("now_ts"), now_local_ts()),
                    )
                    or {}
                )
            except Exception:
                continuity_snapshot = {}
        if continuity_snapshot:
            ctx.state["task_continuity"] = dict(continuity_snapshot)
        ctx.memory_ops.append(
            {
                "op": "task_continuity",
                "source": "assistant_reply",
                "active_task": dict(updated_task),
            }
        )

    def _update_working_state_from_memory(self, ctx: PipelineContext, memory_result: dict[str, Any]) -> None:
        """
        Обновляет working state из результата memory retrieval.
        
        Это связывает explicit recall с always-on memory state:
        - извлечённые факты -> identity_core
        - извлечённые задачи -> active_tasks, open_questions
        - извлечённые решения -> current_decisions
        - извлечённые темы -> recent_topics
        
        Так память становится 'внутренней': модель видит результат retrieval
        и обновляет своё working state для будущих turn'ов.
        """
        if not isinstance(memory_result, dict):
            return
        
        # Извлекаем факты для identity_core
        facts = memory_result.get("facts") or memory_result.get("identity_facts") or []
        if facts:
            addressing = dict(ctx.state.get("addressing") or ctx.state.get("user_addressing") or {})
            for fact in list(facts[:5])[:3]:  # Берём только топ-3 факта
                if isinstance(fact, dict):
                    key = str(fact.get("key") or fact.get("predicate") or "")
                    value = fact.get("value")
                    if key and value is not None:
                        if "name" in key.lower() or "addressing" in key.lower():
                            if "canonical" in key.lower():
                                addressing["canonical_name"] = str(value)
                            elif "allowed" in key.lower():
                                addressing["allowed_forms"] = value if isinstance(value, list) else [value]
                            elif "forbidden" in key.lower():
                                addressing["forbidden_forms"] = value if isinstance(value, list) else [value]
            
            if addressing:
                ctx.state["addressing"] = addressing
                ctx.state["user_addressing"] = addressing
        
        # Извлекаем задачи для active_tasks и open_questions
        tasks = memory_result.get("tasks") or memory_result.get("active_tasks") or []
        if tasks:
            ctx.state["active_tasks"] = list(tasks[:5])
            open_questions = []
            current_decisions = []
            for task in list(tasks[:5]):
                if isinstance(task, dict):
                    if task.get("open_questions"):
                        open_questions.extend(list(task.get("open_questions", [])))
                    if task.get("decisions"):
                        current_decisions.extend(list(task.get("decisions", [])))
            
            if open_questions:
                ctx.state["open_questions"] = list(set(open_questions))[:5]
            if current_decisions:
                ctx.state["current_decisions"] = list(set(current_decisions))[:5]
        
        # Извлекаем темы для recent_topics
        topics = memory_result.get("topics") or memory_result.get("recent_topics") or []
        if topics:
            ctx.state["recent_topics"] = list(topics[:5])
        
        # Извлекаем relation state
        relation = memory_result.get("relation_state") or memory_result.get("user_preferences") or {}
        if relation:
            existing_relation = dict(ctx.state.get("relation_state") or {})
            existing_relation.update(dict(relation))
            ctx.state["relation_state"] = existing_relation


class ResponsePipeline:
    def __init__(
        self,
        provider: LLMProviderBase,
        character_runtime: CharacterRuntime | None = None,
        metadata_extractor: MetadataExtractor | None = None,
        prompt_engine: PromptEngine | None = None,
        memory_core: MemoryCoreAdapter | None = None,
        memory_manager: Any | None = None,
        studio_generator: StudioGenerator | None = None,
    ):
        self.provider = provider
        self.character_engine = character_runtime or CharacterRuntime()
        self.character_runtime = self.character_engine
        self.memory_core = memory_core or memory_manager or MemoryCoreAdapter()
        artifact_store = getattr(getattr(self.memory_core, "service", None), "artifact_store", None)
        self.topic_store = TopicStore(artifact_store) if artifact_store is not None else None
        self.topic_router = (
            TopicRouter(
                self.topic_store,
                hint_extractor=EpisodeProcessor().detect_topic_hints,
            )
            if self.topic_store is not None
            else None
        )
        self.prompt_engine = prompt_engine or PromptEngine(
            character_runtime=self.character_engine,
        )
        self._stages: dict[str, PipelineStage] = {
            "preprocess": PreprocessStage(metadata_extractor=metadata_extractor),
            "mode_select": ModeSelectStage(),
            "plan": PlanStage(),
            "personality": PersonalityStage(
                character_runtime=self.character_engine,
            ),
            "topic_routing": TopicRoutingStage(topic_router=self.topic_router),
            "memory_native_state": MemoryNativeStateStage(memory_core=self.memory_core),
            "memory_retrieve": MemoryRetrieveStage(memory_core=self.memory_core),
            "episode_continuity": EpisodeContinuityStage(memory_core=self.memory_core),
            "web_retrieve": WebStageV2(),
            "prompt_build": PromptBuildStage(character_runtime=self.character_runtime),
            "prompt_engine": PromptEngineStage(prompt_engine=self.prompt_engine),
            "generate": GenerateStage(
                provider=self.provider,
                character_runtime=self.character_runtime,
                memory_core=self.memory_core,
                studio_generator=studio_generator,
            ),
            "postprocess": PostprocessStage(),
            "tool_router": ToolRouterStage(),
            "verify": VerifyStage(),
            "output_format": OutputFormatStage(provider=self.provider),
            "memory_write": MemoryWriteStage(memory_core=self.memory_core),
        }
        # Memory retrieval теперь работает ТОЛЬКО через agent_loop + memory_retrieve tool.
        # MemoryNativeStateStage строит always-on memory layer для всех профилей.
        # MemoryRetrieveStage остаётся только как fallback для экстренных случаев.
        self._profiles: dict[str, tuple[str, ...]] = {
            PROFILE_FAST: (
                "preprocess",
                "mode_select",
                "personality",
                "topic_routing",
                "memory_native_state",
                "episode_continuity",
                "prompt_build",
                "prompt_engine",
                "generate",
                "postprocess",
                "verify",
                "output_format",
                "memory_write",
            ),
            PROFILE_BALANCED: (
                "preprocess",
                "mode_select",
                "plan",
                "personality",
                "topic_routing",
                "memory_native_state",
                "episode_continuity",
                "web_retrieve",
                "prompt_build",
                "prompt_engine",
                "generate",
                "postprocess",
                "tool_router",
                "verify",
                "output_format",
                "memory_write",
            ),
            PROFILE_QUALITY: (
                "preprocess",
                "mode_select",
                "plan",
                "personality",
                "topic_routing",
                "memory_native_state",
                "episode_continuity",
                "web_retrieve",
                "prompt_build",
                "prompt_engine",
                "generate",
                "tool_router",
                "verify",
                "postprocess",
                "output_format",
                "memory_write",
            ),
            PROFILE_AUTONOMOUS: (
                "preprocess",
                "mode_select",
                "plan",
                "personality",
                "topic_routing",
                "episode_continuity",
                "web_retrieve",
                "prompt_build",
                "prompt_engine",
                "generate",
                "verify",
                "postprocess",
                "output_format",
                "memory_write",
            ),
        }

    def run(
        self,
        *,
        route: str,
        user_msg: str,
        state: dict[str, Any],
        meta: dict[str, Any],
        retrieved_memories,
        traits,
        policies,
    ) -> PipelineResult:
        ctx = PipelineContext(
            route=str(route or "chat").strip().lower() or "chat",
            user_msg=str(user_msg or ""),
            state=_as_dict(state),
            meta=_as_dict(meta),
            retrieved_memories=list(_as_list(retrieved_memories)),
            traits=_as_dict(traits),
            policies=_as_dict(policies),
            profile=self._resolve_profile(meta=meta, state=state, policies=policies),
        )
        _inject_temporal_grounding(ctx)
        ctx.meta.setdefault("memory_core", self.memory_core)
        if "memory_manager" not in ctx.meta and self.memory_core is not None:
            ctx.meta["memory_manager"] = self.memory_core
        if self.topic_store is not None:
            ctx.meta.setdefault("topic_store", self.topic_store)
        ctx.meta.setdefault("pipeline", self)
        ctx.meta.setdefault("conversation_id", ctx.state.get("conversation_id"))
        ctx.meta.setdefault("turn_id", ctx.state.get("turn_id"))
        turn_token = int(_to_int(_pick_value(ctx.meta.get("turn_id"), ctx.state.get("turn_id"), 0), 0) or 0)
        if not str(ctx.meta.get("request_id") or "").strip():
            ctx.meta["request_id"] = f"req_{int(time.time() * 1000)}_{turn_token}"
        if not str(ctx.meta.get("trace_id") or "").strip():
            ctx.meta["trace_id"] = f"trace_{int(time.time() * 1000)}_{int(_to_int(ctx.state.get('turn_id'), 0) or 0)}"
        _ensure_debug_trace(ctx)
        trace_id = str(_pick(ctx.meta.get("trace_id"), ctx.state.get("conversation_id"), "-")).strip() or "-"
        request_id = str(_pick(ctx.meta.get("request_id"), ctx.meta.get("trace_id"), "")).strip()
        web_mode = _resolve_web_mode(ctx.meta, ctx.state)
        ctx.meta.setdefault("web_mode", web_mode)
        force_web = _is_forced_web_request(ctx.user_msg)
        input_preview = _text_preview(_loggable_user_input(ctx.user_msg, route=ctx.route), 120)

        stage_profile = self._resolve_stage_profile(ctx.profile)
        ctx.meta["stage_profile"] = stage_profile
        ctx.meta["_web_trace_events"] = []
        ctx.meta["web_trace_started_at"] = _utc_now_iso()
        ctx.meta["emit_web_trace_event"] = lambda event, payload=None: self._emit_web_trace_event(
            ctx,
            event=event,
            payload=payload,
        )
        stage_names = self._resolve_stage_names(stage_profile, ctx.meta, ctx.policies)
        ctx.logs.append(f"profile={ctx.profile}")
        ctx.logs.append(f"stage_profile={stage_profile}")
        ctx.logs.append(f"stages={','.join(stage_names)}")
        ctx.logs.append(
            "stage=web_trace start "
            f"trace={trace_id} request={request_id or '-'} route={ctx.route} web_mode={web_mode} "
            f"force={int(bool(force_web))} input_len={len(str(_loggable_user_input(ctx.user_msg, route=ctx.route) or '').strip())} "
            f"input_preview={input_preview}"
        )
        self._emit_web_trace_event(
            ctx,
            event="web_trace_start",
            payload={
                "force_web": bool(force_web),
                "input_len": len(str(_loggable_user_input(ctx.user_msg, route=ctx.route) or "").strip()),
                "input_preview": input_preview,
                "active_profile": str(ctx.profile or ""),
                "mode": str(web_mode or "auto"),
            },
        )

        for name in stage_names:
            stage = self._stages.get(name)
            if stage is None:
                ctx.logs.append(f"stage={name} missing")
                continue
            try:
                ctx = stage.run(ctx)
            except Exception as exc:
                ctx.errors.append(f"{name}:{type(exc).__name__}:{exc}")
                ctx.logs.append(f"stage={name} error={type(exc).__name__}:{exc}")
                if name == "generate":
                    if ctx.route == "command":
                        ctx.text = "Command processing failed. Check logs and command syntax."
                    else:
                        ctx.text = "I got stuck during generation. Please try again."
                    ctx.stop = True
            if ctx.stop:
                break

        output_preview = _text_preview(ctx.text, 160)
        output_len = len(str(ctx.text or ""))
        web_used = str(ctx.tags.get("web_used") or "false").strip().lower() or "false"
        web_query = str(ctx.meta.get("web_query") or "").strip()
        web_fetched = int(_to_int(ctx.meta.get("web_fetched"), 0) or 0)
        web_result_count = int(_to_int(ctx.meta.get("web_result_count"), 0) or 0)
        if bool(ctx.meta.get("web_guardrail_local_reply")):
            _append_turn_warning(ctx, "web_guardrail_local_reply")
        if _to_bool(_pick(ctx.meta.get("web_fresh_missing"), ctx.tags.get("web_fresh_missing"), False), default=False):
            _append_turn_warning(ctx, "web_fresh_missing")
        if str(ctx.meta.get("web_clarify_reason") or "").strip():
            _append_turn_warning(ctx, f"web_clarify:{str(ctx.meta.get('web_clarify_reason') or '').strip()}")
        if list(ctx.errors or []):
            _append_turn_warning(ctx, f"pipeline_errors:{len(list(ctx.errors or []))}")
        ctx.logs.append(
            "stage=web_trace end "
            f"trace={trace_id} request={request_id or '-'} route={ctx.route} web_mode={web_mode} web_used={web_used} "
            f"query_len={len(web_query)} results={web_result_count} fetched={web_fetched} "
            f"output_len={output_len} output_preview={output_preview}"
        )
        self._emit_web_trace_event(
            ctx,
            event="web_trace_end",
            payload={
                "web_used": (web_used == "true"),
                "query_len": len(web_query),
                "results": web_result_count,
                "fetched": web_fetched,
                "fresh_missing": str(ctx.tags.get("web_fresh_missing") or "").strip().lower() == "true",
                "guardrail_applied": bool(ctx.meta.get("web_guardrail_local_reply")),
                "output_len": output_len,
                "output_preview": output_preview,
                "errors": len(list(ctx.errors or [])),
            },
        )
        trace_file = self._flush_web_trace_file(ctx)
        if trace_file:
            ctx.logs.append(f"stage=web_trace file={trace_file}")
        detail_trace_file = str(ctx.meta.get("web_trace_detail_file_path") or "").strip()
        if detail_trace_file:
            ctx.logs.append(f"stage=web_trace detail_file={detail_trace_file}")
        compact_trace_file = str(ctx.meta.get("web_trace_compact_file_path") or "").strip()
        if compact_trace_file:
            ctx.logs.append(f"stage=web_trace compact_file={compact_trace_file}")
        trace = _ensure_debug_trace(ctx)
        memory_reasoning_snapshot = _as_dict(ctx.state.get("memory_reasoning_snapshot"))
        agent_loop_trace = _as_dict(ctx.state.get("agent_loop_trace"))
        trace.final_answer_meta = {
            "route": str(ctx.route or ""),
            "request_id": str(request_id or ""),
            "trace_id": str(trace_id or ""),
            "stage_profile": str(stage_profile or ""),
            "web_mode": str(web_mode or ""),
            "web_used": bool(web_used == "true"),
            "factual_response_mode": str(ctx.meta.get("factual_response_mode") or ""),
            "served_model": str(_as_dict(ctx.stats).get("served_model") or ""),
            "verbose_enabled": bool(_to_bool(_as_dict(ctx.stats).get("verbose_enabled"), default=False)),
            "output_len": int(output_len),
            "output_preview": str(output_preview or ""),
            "agent_loop": bool(_to_bool(_as_dict(ctx.stats).get("agent_loop"), default=False)),
            "agent_tool_calls": int(_to_int(_as_dict(ctx.stats).get("agent_tool_calls"), 0) or 0),
            "agent_passes": int(_to_int(_as_dict(ctx.stats).get("agent_passes"), 0) or 0),
            "tool_loop": dict(agent_loop_trace),
            "memory_reasoning_used": bool(memory_reasoning_snapshot),
            "memory_reasoning_sections": sorted(memory_reasoning_snapshot.keys()),
            "memory_reasoning_snapshot": dict(memory_reasoning_snapshot),
            "errors": [str(x).strip() for x in list(ctx.errors or []) if str(x).strip()],
            "warnings": [str(x).strip() for x in list(_as_list(ctx.meta.get("turn_log_warnings"))) if str(x).strip()],
        }
        # Упрощённый memory_debug_snapshot
        memory_debug_snapshot = {
            "user_text": str(ctx.clean_user_msg or ctx.user_msg or ""),
            "memory_context": dict(ctx.memory_context or {}),
            "memory_native_state": dict(ctx.state.get("memory_native_state") or {}),
            "retrieved_count": len(list(ctx.retrieved_memories or [])),
            "tool_loop": dict(_as_dict(ctx.state.get("agent_loop_trace"))),
        }
        ctx.state["memory_debug_snapshot"] = dict(memory_debug_snapshot or {})
        
        # Сохраняем trace в trace_store если есть
        trace_store = ctx.meta.get("memory_trace_store")
        if trace_store is not None:
            row = {
                "trace_id": str(ctx.meta.get("trace_id") or ""),
                "request_id": str(ctx.meta.get("request_id") or ""),
                "conversation_id": str(_pick(ctx.meta.get("conversation_id"), ctx.state.get("conversation_id"), "")),
                "turn_id": _pick_value(ctx.meta.get("turn_id"), ctx.state.get("turn_id"), 0),
                "created_at": str(ctx.meta.get("web_trace_started_at") or _utc_now_iso()),
                "route": str(ctx.route or ""),
                "user_text": str(ctx.clean_user_msg or ctx.user_msg or ""),
                "pipeline": {
                    **trace.to_dict(),
                    "turn_log_summaries": _as_dict(ctx.meta.get("turn_log_summaries")),
                    "turn_log_warnings": [str(x).strip() for x in list(_as_list(ctx.meta.get("turn_log_warnings"))) if str(x).strip()],
                },
                "memory": {
                    "event_ids": [],
                    "job_ids": [],
                    "artifact_ids_created": [],
                    "artifact_ids_updated": [],
                    "indexed_ids": [],
                },
                "links": {
                    "web_trace_detail_file_path": str(detail_trace_file),
                    "web_trace_compact_file_path": str(compact_trace_file),
                },
            }
            trace_store.append_turn_trace(row)
        
        if isinstance(meta, dict):
            meta["trace_id"] = str(ctx.meta.get("trace_id") or "")
            meta["request_id"] = str(ctx.meta.get("request_id") or "")
            meta["conversation_id"] = str(_pick(ctx.meta.get("conversation_id"), ctx.state.get("conversation_id"), ""))
            meta["turn_id"] = _pick_value(ctx.meta.get("turn_id"), ctx.state.get("turn_id"), 0)
            meta["turn_log_summaries"] = _as_dict(ctx.meta.get("turn_log_summaries"))
            meta["turn_log_warnings"] = [str(x).strip() for x in list(_as_list(ctx.meta.get("turn_log_warnings"))) if str(x).strip()]
            meta["web_trace_detail_file_path"] = detail_trace_file
            meta["web_trace_compact_file_path"] = compact_trace_file
            meta["debug_trace"] = trace.to_dict()
            meta["memory_debug_snapshot"] = dict(memory_debug_snapshot or {})

        return PipelineResult(
            text=str(ctx.text or ""),
            thinking=str(ctx.thinking or ""),
            structured_output=dict(ctx.structured_output or {}),
            tool_calls=list(ctx.tool_calls or []),
            memory_ops=list(ctx.memory_ops or []),
            ui_actions=list(ctx.ui_actions or []),
            logs=list(ctx.logs or []),
            stats=dict(ctx.stats or {}),
            debug_trace=dict(meta.get("debug_trace") or {}),
            memory_debug_snapshot=dict(meta.get("memory_debug_snapshot") or {}),
        )

    def _emit_web_trace_event(self, ctx: PipelineContext, *, event: str, payload: dict[str, Any] | None = None) -> None:
        trace_id = str(_pick(ctx.meta.get("trace_id"), ctx.state.get("conversation_id"), "-")).strip() or "-"
        row = {
            "event": str(event or "").strip(),
            "trace_id": trace_id,
            "request_id": str(_pick(ctx.meta.get("request_id"), ctx.meta.get("trace_id"), "")),
            "ts": _utc_now_iso(),
            "conversation_id": str(_pick(ctx.meta.get("conversation_id"), ctx.state.get("conversation_id"), "")),
            "turn_id": str(_pick(ctx.meta.get("turn_id"), ctx.state.get("turn_id"), "")),
            "route": str(ctx.route or ""),
            "profile": str(ctx.profile or ""),
            "stage_profile": str(ctx.meta.get("stage_profile") or ""),
            "web_mode": str(ctx.meta.get("web_mode") or ctx.state.get("web_mode") or "auto"),
            "web_query_intent": str(_pick(ctx.meta.get("web_query_intent"), ctx.tags.get("web_query_intent"), "generic")),
            "web_fresh_required": _to_bool(_pick(ctx.meta.get("web_fresh_required"), ctx.tags.get("web_fresh_required"), False), default=False),
            "web_fresh_missing": _to_bool(_pick(ctx.meta.get("web_fresh_missing"), ctx.tags.get("web_fresh_missing"), False), default=False),
            "payload": dict(payload or {}),
        }
        events = _as_list(ctx.meta.get("_web_trace_events"))
        if isinstance(ctx.meta, dict):
            events = list(events)
            events.append(dict(row))
            ctx.meta["_web_trace_events"] = events
        log_json(
            WEB_TRACE_LOGGER,
            "web_trace_event",
            context={"trace_id": trace_id, "request_id": str(row.get("request_id") or ""), "turn_id": str(row.get("turn_id") or ""), "conversation_id": str(row.get("conversation_id") or "")},
            trace_id=trace_id,
            trace_event=str(event or "").strip(),
            route=str(ctx.route or ""),
            web_mode=str(ctx.meta.get("web_mode") or ctx.state.get("web_mode") or "auto"),
        )

    def _flush_web_trace_file(self, ctx: PipelineContext) -> str:
        events = [x for x in _as_list(ctx.meta.get("_web_trace_events")) if isinstance(x, dict)]
        if not events:
            return ""
        cfg = load_config()
        trace_id = str(_pick(ctx.meta.get("trace_id"), ctx.state.get("conversation_id"), "-")).strip() or "-"
        safe_trace_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", trace_id).strip("._") or "trace"
        day = dt.datetime.now(dt.timezone.utc).date().isoformat()
        log_root = Path(str(cfg.log_dir or "")).expanduser().resolve()
        root = log_root / "web_trace" / day
        root.mkdir(parents=True, exist_ok=True)
        jsonl_path = log_root / "web_trace.jsonl"
        path = root / f"{safe_trace_id}.json"
        compact_path = root / f"{safe_trace_id}.compact.json"

        by_event: dict[str, dict[str, Any]] = {}
        for row in events:
            name = str(row.get("event") or "").strip()
            if not name:
                continue
            by_event[name] = dict(row)

        trace_context = self._web_trace_context(ctx)
        query_plan = _as_dict(_as_dict(by_event.get("web_query_plan")).get("payload"))
        search_results = _as_dict(_as_dict(by_event.get("web_search_results")).get("payload"))
        policy_payload = _as_dict(_as_dict(by_event.get("web_policy_decision")).get("payload"))
        start_payload = _as_dict(_as_dict(by_event.get("web_trace_start")).get("payload"))
        end_payload = _as_dict(_as_dict(by_event.get("web_trace_end")).get("payload"))
        evidence_quality = _as_dict(_as_dict(by_event.get("web_evidence_quality")).get("payload"))
        if not evidence_quality:
            evidence_quality = _as_dict(ctx.meta.get("web_evidence_quality"))
        compact_events = [self._compact_web_trace_event(row, trace_context=trace_context) for row in events]
        compact_summary = self._build_compact_web_trace_summary_doc(
            ctx,
            by_event=by_event,
            trace_context=trace_context,
        )

        trace_doc = {
            "trace_id": trace_id,
            "created_at": str(ctx.meta.get("web_trace_started_at") or _utc_now_iso()),
            "trace_context": dict(trace_context),
            "compact_summary_path": str(compact_path),
            "input": {
                "raw_text": _loggable_user_input(ctx.user_msg, route=ctx.route),
                "clean_text": str(ctx.clean_user_msg or ""),
                "web_mode": str(ctx.meta.get("web_mode") or ctx.state.get("web_mode") or "auto"),
                "start": dict(start_payload),
            },
            "context_link": {
                "continuation_ref": str(ctx.meta.get("continuation_ref") or ""),
                "context_confidence": float(_to_float(ctx.meta.get("context_confidence"), 0.0) or 0.0),
                "active_task": _as_dict(_as_dict(ctx.state).get("web_active_task")),
                "minutes_since_previous": str(ctx.meta.get("minutes_since_previous") or ""),
                "same_calendar_day": _to_bool(ctx.meta.get("same_calendar_day"), default=False),
            },
            "policy_decision_breakdown": {
                "mode": str(policy_payload.get("mode") or ""),
                "should_search": bool(policy_payload.get("should_search")),
                "web_need_score": float(_to_float(policy_payload.get("web_need_score"), 0.0) or 0.0),
                "reason": str(policy_payload.get("reason") or ""),
                "decision_breakdown": dict(policy_payload.get("decision_breakdown") or {}),
                "category_penalty_overridden": bool(policy_payload.get("category_penalty_overridden")),
            },
            "queries": {
                "query_effective": str(ctx.meta.get("web_query_effective") or ctx.meta.get("query_effective") or ""),
                "query_plan": dict(query_plan),
                "queries_used": list(search_results.get("queries_used") or []),
                "scout_queries_used": list(search_results.get("scout_queries_used") or []),
                "focused_queries_used": list(search_results.get("focused_queries_used") or []),
            },
            "ranked_sources": list(search_results.get("top_results") or []),
            "consensus": dict(_as_dict(ctx.meta.get("web_consensus"))),
            "evidence_quality": dict(evidence_quality),
            "final_outcome": {
                "web_used": bool(end_payload.get("web_used")),
                "fresh_missing": bool(end_payload.get("fresh_missing")),
                "guardrail_applied": bool(end_payload.get("guardrail_applied")),
                "clarify_needed": bool(str(ctx.meta.get("web_clarifying_question") or "").strip()),
                "clarifying_question": str(ctx.meta.get("web_clarifying_question") or ""),
                "end": dict(end_payload),
            },
            "compact_summary": dict(compact_summary),
            "raw_event_count": int(len(compact_events)),
            "events": compact_events,
        }
        path.write_text(json.dumps(trace_doc, ensure_ascii=False, indent=2), encoding="utf-8")
        compact_path.write_text(
            json.dumps(
                {
                    "trace_id": trace_id,
                    "created_at": str(ctx.meta.get("web_trace_started_at") or _utc_now_iso()),
                    "trace_context": dict(trace_context),
                    "summary": dict(compact_summary),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        jsonl_row = self._build_web_trace_jsonl_row(
            ctx,
            trace_context=trace_context,
            compact_summary=compact_summary,
            compact_events=compact_events,
            detail_path=path,
            compact_path=compact_path,
        )
        self._append_jsonl_row(jsonl_path, jsonl_row)
        ctx.meta["web_trace_file_path"] = str(jsonl_path)
        ctx.meta["web_trace_detail_file_path"] = str(path)
        ctx.meta["web_trace_compact_file_path"] = str(compact_path)
        log_json(
            WEB_TRACE_LOGGER,
            "web_trace_summary",
            context=trace_context,
            trace_id=trace_id,
            route=str(ctx.route or ""),
            web_mode=str(ctx.meta.get("web_mode") or ctx.state.get("web_mode") or "auto"),
            web_used=bool(end_payload.get("web_used")),
            file=str(jsonl_path),
            detail_file=str(path),
            compact_file=str(compact_path),
            raw_event_count=int(len(compact_events)),
        )
        return str(jsonl_path)

    @staticmethod
    def _append_jsonl_row(path: Path, row: dict[str, Any]) -> None:
        payload = json.dumps(dict(row or {}), ensure_ascii=False)
        with _WEB_TRACE_JSONL_LOCK:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(payload + "\n")

    def _build_web_trace_jsonl_row(
        self,
        ctx: PipelineContext,
        *,
        trace_context: dict[str, Any],
        compact_summary: dict[str, Any],
        compact_events: list[dict[str, Any]],
        detail_path: Path,
        compact_path: Path,
    ) -> dict[str, Any]:
        summary = dict(compact_summary or {})
        return {
            "event": "web_trace_turn",
            "trace_id": str(trace_context.get("trace_id") or ""),
            "request_id": str(trace_context.get("request_id") or ""),
            "conversation_id": str(trace_context.get("conversation_id") or ""),
            "turn_id": str(trace_context.get("turn_id") or ""),
            "route": str(trace_context.get("route") or ctx.route or ""),
            "profile": str(trace_context.get("profile") or ""),
            "stage_profile": str(trace_context.get("stage_profile") or ""),
            "created_at": str(ctx.meta.get("web_trace_started_at") or _utc_now_iso()),
            "query": str(summary.get("query") or ""),
            "status": str(summary.get("status") or ""),
            "web_used": bool(summary.get("web_used")),
            "mode": str(summary.get("mode") or ""),
            "summary": summary,
            "timeline": self._compact_jsonl_timeline(compact_events),
            "detail_file": str(detail_path),
            "compact_file": str(compact_path),
        }

    @staticmethod
    def _compact_jsonl_timeline(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for row in list(rows or []):
            item = dict(row or {})
            out.append(
                {
                    "ts": str(item.get("ts") or ""),
                    "event": str(item.get("event") or ""),
                    "summary": ResponsePipeline._timeline_summary(_as_dict(item.get("payload"))),
                }
            )
        return out

    @staticmethod
    def _timeline_summary(payload: dict[str, Any]) -> str:
        parts: list[str] = []
        for key in (
            "mode",
            "reason",
            "query",
            "effective_query",
            "should_search",
            "decision_score",
            "result_count",
            "fetched",
            "web_used",
            "error",
        ):
            value = payload.get(key)
            if value is None:
                continue
            if isinstance(value, bool):
                parts.append(f"{key}={str(value).lower()}")
                continue
            text = str(value).strip()
            if not text:
                continue
            if len(text) > 96:
                text = text[:93].rstrip() + "..."
            parts.append(f"{key}={text}")
        return " ".join(parts)

    def _web_trace_context(self, ctx: PipelineContext) -> dict[str, Any]:
        return {
            "trace_id": str(_pick(ctx.meta.get("trace_id"), ctx.state.get("conversation_id"), "-")).strip() or "-",
            "request_id": str(_pick(ctx.meta.get("request_id"), ctx.meta.get("trace_id"), "")).strip(),
            "conversation_id": str(_pick(ctx.meta.get("conversation_id"), ctx.state.get("conversation_id"), "")),
            "turn_id": str(_pick(ctx.meta.get("turn_id"), ctx.state.get("turn_id"), "")),
            "route": str(ctx.route or ""),
            "profile": str(ctx.profile or ""),
            "stage_profile": str(ctx.meta.get("stage_profile") or ""),
            "web_mode": str(ctx.meta.get("web_mode") or ctx.state.get("web_mode") or "auto"),
            "web_query_intent": str(_pick(ctx.meta.get("web_query_intent"), ctx.tags.get("web_query_intent"), "generic")),
            "web_fresh_required": _to_bool(_pick(ctx.meta.get("web_fresh_required"), ctx.tags.get("web_fresh_required"), False), default=False),
            "web_fresh_missing": _to_bool(_pick(ctx.meta.get("web_fresh_missing"), ctx.tags.get("web_fresh_missing"), False), default=False),
        }

    @staticmethod
    def _compact_web_trace_event(row: dict[str, Any], *, trace_context: dict[str, Any]) -> dict[str, Any]:
        item = {
            "ts": str(row.get("ts") or ""),
            "event": str(row.get("event") or ""),
            "payload": dict(row.get("payload") or {}),
        }
        for key in ("web_mode", "web_query_intent", "web_fresh_required", "web_fresh_missing"):
            if row.get(key) != trace_context.get(key):
                item[key] = row.get(key)
        return item

    def _build_compact_web_trace_summary_doc(
        self,
        ctx: PipelineContext,
        *,
        by_event: dict[str, dict[str, Any]],
        trace_context: dict[str, Any],
    ) -> dict[str, Any]:
        compact = _as_dict(ctx.meta.get("web_trace_compact_summary"))
        if not compact:
            compact = _as_dict(_as_dict(by_event.get("web_summary")).get("payload"))

        policy_payload = _as_dict(_as_dict(by_event.get("web_policy_decision")).get("payload"))
        query_plan = _as_dict(_as_dict(by_event.get("web_query_plan")).get("payload"))
        search_results = _as_dict(_as_dict(by_event.get("web_search_results")).get("payload"))
        budget_update = _as_dict(_as_dict(by_event.get("web_budget_update")).get("payload"))
        evidence_pack = _as_dict(_as_dict(by_event.get("web_evidence_pack")).get("payload"))
        end_payload = _as_dict(_as_dict(by_event.get("web_trace_end")).get("payload"))

        out = dict(compact or {})
        out.setdefault("status", "ok" if bool(end_payload.get("web_used")) else ("skipped" if not search_results else "ok"))
        out.setdefault(
            "policy",
            {
                "mode": str(policy_payload.get("mode") or trace_context.get("web_mode") or ""),
                "reason": str(policy_payload.get("reason") or ""),
                "should_search": bool(policy_payload.get("should_search")),
                "decision_score": float(_to_float(policy_payload.get("web_need_score"), 0.0) or 0.0),
            },
        )
        out.setdefault(
            "queries",
            {
                "planned_total": int(len(list(_as_list(_as_dict(query_plan).get("scout_queries"))) + list(_as_list(_as_dict(query_plan).get("focused_queries"))) + list(_as_list(_as_dict(query_plan).get("fallback_queries"))))),
                "used_total": int(len(list(search_results.get("queries_used") or []))),
                "used": [str(x or "").strip() for x in list(search_results.get("queries_used") or []) if str(x or "").strip()][:8],
                "roles": dict(_as_dict(query_plan.get("query_roles"))),
                "runs": [dict(x or {}) for x in list(search_results.get("query_runs") or [])[:8] if isinstance(x, dict)],
            },
        )
        out.setdefault(
            "budget",
            {
                "limit": dict(_as_dict(budget_update.get("budget"))),
                "used": dict(_as_dict(budget_update.get("used"))),
                "cooldown_applied": bool(budget_update.get("cooldown_applied")),
                "retries_used": int(_to_int(budget_update.get("retries_used"), 0) or 0),
            },
        )
        out.setdefault(
            "sources",
            {
                "scanned": int(_to_int(_as_dict(_as_dict(evidence_pack.get("selection_summary"))).get("sources_scanned"), 0) or 0),
                "selected": int(_to_int(_as_dict(_as_dict(evidence_pack.get("selection_summary"))).get("selected_sources"), 0) or 0),
                "filtered": int(_to_int(_as_dict(_as_dict(evidence_pack.get("selection_summary"))).get("filtered_sources"), 0) or 0),
            },
        )
        out.setdefault(
            "evidence",
            {
                "count": int(_to_int(evidence_pack.get("items"), 0) or 0),
                "citations": int(_to_int(evidence_pack.get("citations"), 0) or 0),
                "key_facts": int(_to_int(evidence_pack.get("key_facts"), 0) or 0),
                "conflicting_sources": bool(evidence_pack.get("conflicting_sources")),
            },
        )
        issues = [str(x).strip() for x in list(_as_list(out.get("issues"))) if str(x).strip()]
        warnings = [str(x).strip() for x in list(_as_list(out.get("warnings"))) if str(x).strip()]
        if bool(end_payload.get("fresh_missing")) and "fresh_data_missing" not in issues:
            issues.append("fresh_data_missing")
        if bool(end_payload.get("guardrail_applied")) and "guardrail_applied" not in warnings:
            warnings.append("guardrail_applied")
        out["issues"] = issues
        out["warnings"] = warnings
        out["outcome"] = {
            "web_used": bool(end_payload.get("web_used")),
            "results": int(_to_int(end_payload.get("results"), 0) or 0),
            "fetched": int(_to_int(end_payload.get("fetched"), 0) or 0),
            "fresh_missing": bool(end_payload.get("fresh_missing")),
            "guardrail_applied": bool(end_payload.get("guardrail_applied")),
        }
        return out

    def is_studio_active(self, *, conversation_id: str = "", state: dict[str, Any] | None = None) -> bool:
        state_map = _as_dict(state)
        local = _as_dict(state_map.get(StudioGenerator.KEY))
        if bool(local.get("active", False)):
            return True
        stage = self._stages.get("generate")
        if isinstance(stage, GenerateStage):
            return bool(stage.is_studio_active(conversation_id=conversation_id, state=state_map))
        return False

    def _resolve_profile(self, *, meta, state, policies) -> str:
        studio_gen = _as_dict(_as_dict(state).get(StudioGenerator.KEY))
        if bool(studio_gen.get("active", False)):
            return PROFILE_AUTONOMOUS
        meta_map = _as_dict(meta)
        state_map = _as_dict(state)
        policies_map = _as_dict(policies)
        preferred = _pick(
            meta_map.get("profile"),
            meta_map.get("personality_llm_profile"),
            meta_map.get("llm_profile"),
            state_map.get("personality_llm_profile"),
            state_map.get("llm_profile"),
            meta_map.get("quality_profile"),
            state_map.get("quality_profile"),
            policies_map.get("profile"),
            policies_map.get("quality_profile"),
            PROFILE_BALANCED,
        )
        norm = str(preferred or PROFILE_BALANCED).strip().upper()
        valid_profiles = set(self._profiles.keys()) | {PROFILE_ECONOM, PROFILE_ASYA}
        return norm if norm in valid_profiles else PROFILE_BALANCED

    def _resolve_stage_profile(self, profile: str) -> str:
        norm = str(profile or "").strip().upper()
        if norm == PROFILE_ASYA:
            return PROFILE_BALANCED
        if norm == PROFILE_ECONOM:
            return PROFILE_FAST
        if norm in self._profiles:
            return norm
        return PROFILE_BALANCED

    def _resolve_stage_names(self, profile: str, meta: dict[str, Any], policies: dict[str, Any]) -> list[str]:
        base = list(self._profiles.get(profile, self._profiles[PROFILE_BALANCED]))
        disable = {str(x).strip() for x in _as_list(meta.get("disable_stages")) + _as_list(policies.get("disable_stages"))}
        enable = [str(x).strip() for x in _as_list(meta.get("enable_stages")) + _as_list(policies.get("enable_stages"))]
        active = [name for name in base if name and name not in disable]
        for name in enable:
            if name and name in self._stages and name not in active:
                active.append(name)
        return active


class _ThinkStreamParser:
    """Split model stream into visible answer and hidden thinking blocks."""

    def __init__(self):
        self._in_think = False
        self._open_tags = tuple(str(tag).lower() for tag in ("<think>", "<thinking>", "<reasoning>"))
        self._close_tags = tuple(str(tag).lower() for tag in ("</think>", "</thinking>", "</reasoning>"))
        self._pending_tag = ""

    def feed(self, chunk: str) -> tuple[str, str]:
        visible_parts: list[str] = []
        thinking_parts: list[str] = []
        for ch in str(chunk or ""):
            visible, thinking = self._feed_char(ch)
            if visible:
                visible_parts.append(visible)
            if thinking:
                thinking_parts.append(thinking)
        return "".join(visible_parts), "".join(thinking_parts)

    def flush(self) -> tuple[str, str]:
        if not self._pending_tag:
            return "", ""
        out = self._emit_literal(self._pending_tag)
        self._pending_tag = ""
        return out

    def _feed_char(self, ch: str) -> tuple[str, str]:
        if self._pending_tag:
            return self._continue_pending_tag(ch)
        if ch == "<":
            self._pending_tag = "<"
            return "", ""
        return self._emit_literal(ch)

    def _continue_pending_tag(self, ch: str) -> tuple[str, str]:
        tags = self._close_tags if self._in_think else self._open_tags
        candidate = self._pending_tag + str(ch or "")
        candidate_low = candidate.lower()
        if any(tag.startswith(candidate_low) for tag in tags):
            self._pending_tag = candidate
            if candidate_low in tags:
                self._pending_tag = ""
                self._in_think = not self._in_think
            return "", ""

        literal = self._pending_tag
        self._pending_tag = ""
        visible, thinking = self._emit_literal(literal)
        extra_visible, extra_thinking = self._feed_char(ch)
        return visible + extra_visible, thinking + extra_thinking

    def _emit_literal(self, text: str) -> tuple[str, str]:
        if not text:
            return "", ""
        if self._in_think:
            return "", text
        return text, ""


def _tool_calls_from_row(row: dict[str, Any]) -> list[ToolCall]:
    out: list[ToolCall] = []
    for idx, item in enumerate(list(row.get("tool_calls") or [])):
        if not isinstance(item, dict):
            continue
        fn = dict(item.get("function") or {})
        name = str(fn.get("name") or item.get("name") or "").strip()
        if not name:
            continue
        args_raw = fn.get("arguments", item.get("arguments"))
        raw_text = ""
        args: dict[str, Any] = {}
        if isinstance(args_raw, dict):
            args = dict(args_raw)
            raw_text = json.dumps(args_raw, ensure_ascii=False)
        else:
            raw_text = str(args_raw or "")
            if raw_text:
                try:
                    parsed = json.loads(raw_text)
                    if isinstance(parsed, dict):
                        args = parsed
                except Exception:
                    args = {}
        out.append(
            ToolCall(
                id=str(item.get("id") or f"tool_{idx+1}"),
                name=name,
                arguments=args,
                raw_arguments=raw_text,
            )
        )
    return out


def _messages_from_prompt_pack(pack: PromptPack) -> list[Message]:
    out: list[Message] = []
    for row in list(pack.messages or []):
        if not isinstance(row, dict):
            continue
        role_raw = str(row.get("role") or "user").strip().lower()
        role = role_raw if role_raw in {"system", "user", "assistant", "tool"} else "user"
        out.append(
            Message(
                role=role,  # type: ignore[arg-type]
                content=str(row.get("content") or ""),
                name=str(row.get("name") or ""),
                tool_call_id=str(row.get("tool_call_id") or ""),
                tool_calls=_tool_calls_from_row(row),
            )
        )
    return out


_SELF_MEMORY_CLAIM_HINTS: tuple[str, ...] = (
    "\u0443 \u043c\u0435\u043d\u044f",
    "\u043c\u043d\u0435",
    "\u043c\u0435\u043d\u044f",
    "\u043c\u043e\u0439",
    "\u043c\u043e\u044f",
    "\u043c\u043e\u044e",
    "\u043c\u043e\u0451",
    "my",
    "me",
    "what do i ",
    "what's my",
    "what is my",
)


def _contains_any_fragment(text: str, fragments: tuple[str, ...]) -> bool:
    low = unicodedata.normalize("NFKC", str(text or "")).strip().lower()
    if not low:
        return False
    return any(str(fragment).strip().lower() in low for fragment in fragments if str(fragment).strip())


def _apply_memory_context_to_prompt_pack(
    pack: PromptPack,
    *,
    memory_context: dict[str, Any],
    selected_memories: list[Any] | None = None,
) -> PromptPack:
    row = _as_dict(memory_context)
    blocks = _as_dict(row.get("blocks"))
    if not blocks:
        return pack

    merged = dict(pack.blocks or {})
    claim_guard = _as_dict(row.get("self_memory_claim_guard"))
    merged["retrieved_memories"] = _render_memory_context_for_prompt(blocks)
    conversation_tail_hint = str(blocks.get("conversation_tail") or "").strip()
    if conversation_tail_hint:
        merged["conversation_tail"] = conversation_tail_hint

    if bool(claim_guard.get("applied")):
        merged.pop("long_summary", None)
    summary = sanitize_session_summary_text(blocks.get("session_summary") or "")
    if summary and not str(blocks.get("relevant_claims") or "").strip():
        merged["long_summary"] = summary
    user_msg = str(blocks.get("user_message") or "").strip()
    if user_msg:
        merged["user_message"] = user_msg
    system_core = str(blocks.get("system_core") or "").strip()
    if system_core:
        merged["system_role"] = system_core

    token_usage = dict(pack.token_usage or {})
    for key in ("system_role", "user_message", "retrieved_memories", "long_summary", "conversation_tail"):
        token_usage[key] = int(estimate_tokens(str(merged.get(key) or "")))
    token_usage["full_prompt"] = int(estimate_tokens(_render_prompt_preview_from_blocks(merged)))

    cut_info = dict(pack.cut_info or {})
    truncation_log = [dict(x) for x in list(_as_list(row.get("truncation_log"))) if isinstance(x, dict)]
    cut_info["memory_context_selected"] = len(list(_as_list(row.get("selected"))))
    cut_info["memory_context_dropped"] = len(list(_as_list(row.get("dropped"))))
    cut_info["memory_context_compression_steps"] = len(truncation_log)
    if bool(claim_guard.get("applied")):
        cut_info["self_memory_claim_guard"] = dict(claim_guard)
    if truncation_log:
        cut_info["memory_context_truncation_log"] = truncation_log[:24]

    selected_rows = [dict(x) for x in list(_as_list(selected_memories)) if isinstance(x, dict)]
    return replace(
        pack,
        blocks=merged,
        token_usage=token_usage,
        cut_info=cut_info,
        selected_memories=(selected_rows if selected_rows else list(pack.selected_memories or [])),
    )


def _apply_web_evidence_to_prompt_pack(
    pack: PromptPack,
    *,
    web_evidence_context: dict[str, Any] | None,
) -> PromptPack:
    context = _as_dict(web_evidence_context)
    if not context:
        return pack
    blocks = _as_dict(pack.blocks)
    if str(blocks.get("self_facts") or "").strip():
        cut_info = dict(pack.cut_info or {})
        cut_info["web_evidence_skipped_for_self_facts"] = True
        return replace(pack, cut_info=cut_info)
    web_block = _render_web_evidence_for_prompt(context)
    if not web_block:
        return pack

    merged = dict(blocks or {})
    merged["web_evidence"] = web_block

    token_usage = dict(pack.token_usage or {})
    token_usage["web_evidence"] = int(estimate_tokens(str(web_block or "")))
    token_usage["full_prompt"] = int(estimate_tokens(_render_prompt_preview_from_blocks(merged)))

    cut_info = dict(pack.cut_info or {})
    cut_info["web_evidence_injected"] = True
    cut_info["web_evidence_sources"] = len(list(_as_list(context.get("sources"))))
    cut_info["web_evidence_conflicting_sources"] = bool(context.get("conflicting_sources"))
    cut_info["web_evidence_citations"] = len(list(_as_list(context.get("compact_citations"))))

    return replace(
        pack,
        blocks=merged,
        token_usage=token_usage,
        cut_info=cut_info,
    )


def _render_web_evidence_for_prompt(context: dict[str, Any]) -> str:
    direct = str(context.get("prompt_block") or "").strip()
    if direct:
        if "[WEB_TOOL_STATUS]" in direct:
            return direct
        status_block = (
            "[WEB_TOOL_STATUS]\n"
            "- live_web_lookup: already_executed_for_this_turn\n"
            "- response_rule: do not say that you cannot browse/check the internet or access live data for this turn.\n"
            "- response_rule: answer from the WEB_EVIDENCE facts below."
        )
        return _join_non_empty([status_block, direct])

    summary = str(context.get("summary") or "").strip()
    freshness = str(context.get("freshness_summary") or "").strip()
    key_facts = [str(x).strip() for x in _as_list(context.get("key_facts")) if str(x).strip()]
    citations = [str(x).strip() for x in _as_list(context.get("compact_citations")) if str(x).strip()]
    conflict_notes = [str(x).strip() for x in _as_list(context.get("conflict_notes")) if str(x).strip()]
    sources = [x for x in _as_list(context.get("sources")) if isinstance(x, dict)]
    conflicting_sources = bool(context.get("conflicting_sources"))

    lines: list[str] = []
    if summary:
        lines.append(f"- summary: {summary}")
    if freshness:
        lines.append(f"- freshness: {freshness}")
    if key_facts:
        lines.append("- key_facts:")
        for fact in list(key_facts)[:8]:
            lines.append(f"  - {fact}")
    if citations:
        lines.append("- citations:")
        for citation in list(citations)[:4]:
            lines.append(f"  - {citation}")
    lines.append(f"- source_conflicts: {'yes' if conflicting_sources else 'no'}")
    if conflicting_sources and conflict_notes:
        lines.append("- conflict_notes:")
        for note in list(conflict_notes)[:4]:
            lines.append(f"  - {note}")
    if sources:
        lines.append("- sources:")
        for row in list(sources)[:5]:
            item = _as_dict(row)
            domain = str(item.get("domain") or "").strip() or "unknown"
            url = str(item.get("url") or "").strip()
            published = str(item.get("published_at") or "").strip() or "-"
            fetched = str(item.get("fetched_at") or "").strip() or "-"
            lines.append(f"  - {domain} | published={published} | fetched={fetched} | {url}")

    if not lines:
        return ""
    status_block = (
        "[WEB_TOOL_STATUS]\n"
        "- live_web_lookup: already_executed_for_this_turn\n"
        "- response_rule: do not say that you cannot browse/check the internet or access live data for this turn.\n"
        "- response_rule: answer from the WEB_EVIDENCE facts below."
    )
    return _join_non_empty([status_block, "[WEB_EVIDENCE]\n" + "\n".join(lines).strip()])


def _render_memory_context_for_prompt(blocks: dict[str, Any]) -> str:
    order = [
        ("MEMORY_RECALL_MODE", "memory_recall_mode"),
        ("SELF_FACTS", "self_facts"),
        ("FACT_EXPECTATION_CHECK", "fact_expectation_check"),
        ("RELEVANT_CLAIMS", "relevant_claims"),
        ("EXACT_RECALL", "exact_recall"),
        ("ANSWER_SUPPORT", "answer_support"),
        ("CONTINUITY_HINTS", "continuity_hints"),
        ("TONE_HINTS", "tone_hints"),
        ("RECALLED_DIALOG", "recalled_dialog"),
        ("DOCUMENT_EVIDENCE", "document_evidence"),
        ("EXACT_FACT_EVIDENCE", "exact_fact_evidence"),
        ("SUPPORTING_MESSAGES", "supporting_messages"),
        ("SUPPORTING_MESSAGE", "supporting_message"),
        ("WORKING_MEMORY", "working_memory"),
        ("SESSION_SUMMARY", "session_summary"),
        ("SEMANTIC_FACTS", "retrieved_semantic"),
        ("EPISODIC_MEMORIES", "retrieved_episodic"),
        ("DOCUMENT_SNIPPETS", "retrieved_docs"),
        ("TASK_TOOL_STATE", "active_tool_state"),
        ("UNRESOLVED_ITEMS", "unresolved_items"),
    ]
    chunks: list[str] = []
    suppress_raw_episodic = bool(str(blocks.get("recalled_dialog") or "").strip())
    for title, key in order:
        if key == "retrieved_episodic" and suppress_raw_episodic:
            continue
        value = str(blocks.get(key) or "").strip()
        if not value:
            continue
        chunks.append(f"[{title}]\n{value}")
    return "\n\n".join(chunks).strip()


def _has_exact_self_facts(memory_context: dict[str, Any] | None) -> bool:
    row = _as_dict(memory_context)
    if not row:
        return False
    self_facts_context = _as_dict(row.get("self_facts_context"))
    found_predicates = [
        str(x).strip()
        for x in list(_as_list(self_facts_context.get("found_predicates")))
        if str(x).strip()
    ]
    if found_predicates:
        return True
    fact_expectation = _as_dict(row.get("fact_expectation"))
    expectation_found = [str(x).strip() for x in list(_as_list(fact_expectation.get("found_predicates"))) if str(x).strip()]
    if expectation_found:
        return True
    blocks = _as_dict(row.get("blocks"))
    return bool(str(blocks.get("self_facts") or "").strip())


def _apply_self_fact_factual_mode_guard(
    *,
    factual_response_mode: str,
    web_used: bool,
    memory_context: dict[str, Any] | None,
) -> tuple[str, bool, dict[str, Any]]:
    mode = str(factual_response_mode or "").strip()
    used = bool(web_used)
    if not _has_exact_self_facts(memory_context):
        return mode, used, {}
    return _SELF_MEMORY_EXACT_MODE, False, {
        "reason": "exact_self_fact_present",
        "skip_factual_context_isolation": True,
        "factual_response_mode": _SELF_MEMORY_EXACT_MODE,
    }


def _apply_self_memory_exact_turn_guard(ctx: PipelineContext, *, prompt_state: dict[str, Any] | None = None) -> None:
    ctx.meta["factual_response_mode"] = _SELF_MEMORY_EXACT_MODE
    ctx.tags["factual_response_mode"] = _SELF_MEMORY_EXACT_MODE
    ctx.tags["self_memory_exact"] = "true"
    ctx.meta["web_skip_citations"] = True
    ctx.meta["web_used"] = False
    ctx.meta["web_fresh_missing"] = False
    ctx.meta["web_guardrail_local_reply"] = False
    ctx.meta["web_response_style"] = "default"
    ctx.tags["web_used"] = "false"
    ctx.tags["web_fresh_missing"] = "false"
    ctx.tags["web_response_style"] = "default"
    for key in (
        "web_evidence_context",
        "web_evidence_quality",
        "web_clarifying_question",
        "web_clarify_reason",
        "web_guardrail",
    ):
        ctx.meta.pop(key, None)
    if isinstance(ctx.state, dict):
        ctx.state.pop("web_evidence_context", None)
        ctx.state.pop("web_evidence_quality", None)
    if isinstance(prompt_state, dict):
        prompt_state.pop("web_evidence_context", None)
        tags_map = _as_dict(prompt_state.get("context_tags"))
        tags_map["factual_response_mode"] = _SELF_MEMORY_EXACT_MODE
        tags_map["self_memory_exact"] = "true"
        tags_map["web_used"] = "false"
        tags_map["web_fresh_missing"] = "false"
        tags_map["web_response_style"] = "default"
        tags_map.pop("web_guardrail", None)
        prompt_state["context_tags"] = tags_map


def _render_prompt_preview_from_blocks(blocks: dict[str, Any]) -> str:
    parts = [
        ("SYSTEM", str(blocks.get("system_role") or "").strip()),
        ("MEMORY", str(blocks.get("retrieved_memories") or "").strip()),
        ("WEB_EVIDENCE", str(blocks.get("web_evidence") or "").strip()),
        ("SUMMARY", str(blocks.get("long_summary") or "").strip()),
        ("USER", str(blocks.get("user_message") or "").strip()),
    ]
    out: list[str] = []
    for name, text in parts:
        if not text:
            continue
        out.append(f"[{name}]\n{text}")
    return "\n\n".join(out).strip()


def _memory_retrieve_tool_spec() -> ToolSpec:
    return ToolSpec(
        name=_MEMORY_TOOL_NAME,
        description=(
            "Retrieve memory candidates and context blocks for the current conversation. "
            "Call it when you need past facts, episodes, tasks, or profile context and answer cannot be grounded from the current prompt alone."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "mode": {
                    "type": "string",
                    "description": "Retrieval mode such as profile, fact, episode, task, document, or context.",
                },
                "topic_hints": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Short topic anchors for deterministic memory fan-out.",
                },
                "time_hint": {
                    "type": "string",
                    "description": "Temporal bias such as recent, session, historical, persistent, or any.",
                },
                "scopes": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": [
                            "conversation",
                            "session",
                            "project",
                            "global_user",
                            "character",
                            "temporary",
                        ],
                    },
                    "description": "Optional memory scopes to search.",
                },
                "sources": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Preferred memory sources such as facts, episodes, tasks, profile, documents, or messages.",
                },
                "top_k": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 12,
                    "description": "How many memory candidates to retrieve.",
                },
            },
            "required": ["mode", "topic_hints", "time_hint", "sources", "top_k"],
            "additionalProperties": False,
        },
    )


def _resolve_topic_store(ctx: PipelineContext, fallback_memory_core: Any | None = None) -> TopicStore | None:
    existing = ctx.meta.get("topic_store")
    if isinstance(existing, TopicStore):
        return existing

    memory_core = (
        ctx.meta.get("memory_core")
        or ctx.meta.get("memory_manager")
        or getattr(ctx.meta.get("pipeline"), "memory_core", None)
        or fallback_memory_core
    )
    artifact_store = getattr(getattr(memory_core, "service", None), "artifact_store", None)
    if artifact_store is None:
        return None
    store = TopicStore(artifact_store)
    ctx.meta["topic_store"] = store
    return store


def _augment_memory_context_with_topic_summary(
    ctx: PipelineContext,
    *,
    workspace_id: str,
    memory_core: Any | None = None,
) -> None:
    topic_thread_id = str(ctx.state.get("topic_thread_id") or ctx.meta.get("topic_thread_id") or "").strip()
    if not topic_thread_id:
        return
    if not isinstance(ctx.memory_context, dict):
        ctx.memory_context = {}
    topic_store = _resolve_topic_store(ctx, memory_core)
    if topic_store is None:
        return
    details = TopicToolService(topic_store).read_topic(
        topic_thread_id,
        workspace_id=str(workspace_id or "").strip(),
        limit=8,
    )
    if not isinstance(details, dict) or not details:
        return
    blocks = dict(ctx.memory_context.get("blocks") or {})
    prompt_blocks = _build_topic_prompt_blocks(details)
    blocks.update(prompt_blocks)
    ctx.memory_context["blocks"] = blocks
    ctx.memory_context["topic_summary"] = dict(details)
    ctx.memory_context["open_questions"] = _merge_compact_strings(
        ctx.memory_context.get("open_questions"),
        details.get("open_questions"),
        limit=10,
    )
    ctx.memory_context["current_decisions"] = _merge_compact_strings(
        ctx.memory_context.get("current_decisions"),
        details.get("current_decisions"),
        limit=10,
    )


def _build_topic_prompt_blocks(details: dict[str, Any]) -> dict[str, str]:
    thread = dict(details.get("thread") or {})
    title = str(thread.get("title") or thread.get("topic_key") or thread.get("thread_id") or "").strip()
    status = str(thread.get("status") or "active").strip()
    summary = str(details.get("summary") or thread.get("summary") or "").strip()
    current_topic_lines: list[str] = []
    if title:
        current_topic_lines.append(f"- title: {title}")
    topic_key = str(thread.get("topic_key") or "").strip()
    if topic_key:
        current_topic_lines.append(f"- key: {topic_key}")
    if status:
        current_topic_lines.append(f"- status: {status}")
    if summary:
        current_topic_lines.append(f"- summary: {summary}")
    related_topics = [
        str(dict(row or {}).get("title") or dict(row or {}).get("topic_key") or dict(row or {}).get("thread_id") or "").strip()
        for row in list(details.get("related_topics") or [])
        if str(dict(row or {}).get("title") or dict(row or {}).get("topic_key") or dict(row or {}).get("thread_id") or "").strip()
    ]
    return {
        "current_topic": "\n".join(current_topic_lines).strip(),
        "topic_open_questions": _render_topic_list_block(details.get("open_questions"), prefix="- "),
        "topic_decisions": _render_topic_list_block(details.get("current_decisions"), prefix="- "),
        "related_topics": _render_topic_list_block(related_topics, prefix="- "),
    }


def _render_topic_list_block(values: Any, *, prefix: str = "- ") -> str:
    items = _merge_compact_strings(values, limit=8)
    if not items:
        return ""
    return "\n".join(f"{prefix}{item}" for item in items)


def _merge_compact_strings(*groups: Any, limit: int = 10) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for group in groups:
        for value in list(group or []):
            clean = " ".join(str(value or "").strip().split())
            if not clean or clean in seen:
                continue
            seen.add(clean)
            result.append(clean)
            if len(result) >= max(1, int(limit or 10)):
                return result
    return result


def _should_enable_agent_loop(ctx: PipelineContext) -> bool:
    """
    Определяет, нужно ли включать agent loop для текущего запроса.
    
    Agent loop теперь включён для ВСЕХ chat запросов по умолчанию.
    Это делает память 'внутренней' для модели, а не внешним сервисом.
    
    Отключить можно через meta['agent_loop'] = False.
    """
    if str(ctx.route or "").strip().lower() != "chat":
        return False
    
    # Явный override имеет приоритет
    override = _pick_value(ctx.meta.get("agent_loop"), ctx.policies.get("agent_loop"), None)
    if override is not None:
        return _to_bool(override, default=True)
    
    # По умолчанию agent loop включён для всех chat запросов
    return True


def _should_enable_memory_retrieve_stage(ctx: PipelineContext) -> bool:
    """
    Определяет, нужно ли включать MemoryRetrieveStage.

    MemoryRetrieveStage используется:
    1. Для моделей без tool support (deepseek-r1 и т.д.)
    2. Как fallback, когда agent_loop не сработал

    Это основной путь retrieval для моделей без function calling.
    """
    if str(ctx.route or "").strip().lower() != "chat":
        return False

    # Если agent_loop отключён из-за модели — используем stage
    if not _should_enable_agent_loop(ctx):
        return True

    # Если agent_loop включён — stage не нужен (будет tool call)
    return False


def _memory_gate_should_trigger(ctx: PipelineContext) -> bool:
    """
    Memory gate: ЖЁСТКОЕ правило для memory-dependent вопросов.

    Если gate срабатывает, модель ОБЯЗАНА вызвать memory_retrieve перед ответом.
    Прямой ответ запрещён до completion memory pass.

    Срабатывает, когда:
    - вопрос зависит от прошлых фактов
    - есть местоимения типа "это", "тогда", "оно", "тот модуль"
    - затронуты user-specific preferences / project continuity
    - есть unresolved questions из прошлых.turns
    - вопрос о личных фактах пользователя
    - вопрос о прошлых разговорах/сообщениях/решениях

    НЕ срабатывает на:
    - smalltalk / greetings
    - общие вопросы без контекста
    - факты общего знания
    """
    query = str(ctx.clean_user_msg or ctx.user_msg or "").strip().lower()
    if not query:
        return False

    state = ctx.state or {}

    # === БЫСТРЫЙ ОТКАЗ: smalltalk и общие вопросы ===
    # Эти темы НЕ требуют memory retrieval
    import re
    
    # Сначала проверяем на personal possessive — если есть, НЕ блокируем
    has_personal_possessive = bool(re.search(r"\b(my|мой|моё|моя|мои|меня|мне)\b", query))
    
    if not has_personal_possessive:
        smalltalk_patterns = [
            # Приветствия / прощания
            r"^\s*(привет|здравствуй|hello|hi|hey|добрый)\b",
            r"^\s*(пока|до свидания|goodbye|bye)\b",
            
            # "Как дела" без контекста
            r"^\s*(как дела|как жизнь|как оно|how are you|how's it going)\s*[?!]?\s*$",
            
            # Общие вопросы без личного контекста
            r"^\s*(что нового|what's new)\s*[?!]?\s*$",
            
            # Благодарности
            r"^\s*(спасибо|благодарю|thank you|thanks)\b",
            
            # Извинения
            r"^\s*(извини|прости|sorry)\b",
            
            # Подтверждения
            r"^\s*(да|нет|ок|ok|хорошо|well|yes|no)\s*[!?.]?\s*$",
            
            # Погода (общий вопрос)
            r"^\s*(какая погода|weather)\b",
        ]
        
        for pattern in smalltalk_patterns:
            if re.search(pattern, query):
                return False

    # === Memory gate логика ===
    
    # 1. Местоимения и ссылки на предыдущий контекст (расширенный список)
    continuity_markers = [
        # Русские местоимения и указатели
        "это ", "этот ", "эта ", "эти ", "этом ", "этому ", "этим ", "этой ",
        "тогда ", "тот ", "та ", "то ", "те ", "тем ", "тому ",
        "оно ", "она ", "они ", "ним ", "ней ", "них ", "нее ",
        "такой ", "такая ", "такое ", "такие ",
        "выше ", "ниже ", "ранее ", "прежде ",
        # Указатели на прошлое
        "напомни ", "вспомина ", "помниш ", "помню ",
        "ещё раз ", "снова ", "опять ", "вернись ",
        # English
        "this ", "that ", "these ", "those ",
        "it ", "they ", "them ", "such ",
        "above ", "below ", "earlier ", "before ",
        "remember ", "remind ", "again ", "back ",
    ]
    has_continuity_marker = any(marker in query for marker in continuity_markers)

    # 2. Вопросы, которые явно зависят от контекста (максимально полный список)
    context_dependent_patterns = [
        # === Вопросы о предыдущих решениях/планах/договорённостях ===
        r"(мы\s+.+\s+решили|мы\s+.+\s+договорились|мы\s+.+\s+планировали)",
        r"(мы\s+решили|мы\s+договорились|мы\s+планировали|как\s+договаривались)",
        r"(we\s+.+\s+decided|we\s+.+\s+agreed|we\s+.+\s+planned)",
        r"(we\s+decided|we\s+agreed|we\s+planned|as\s+discussed)",
        
        # === Вопросы о проекте/коде/файлах ===
        r"(мой\s+.+\s+проект|мой\s+.+\s+код|этот\s+.+\s+модуль|этот\s+.+\s+файл)",
        r"(мой\s+проект|мой\s+код|этот\s+модуль|этот\s+файл)",
        r"(my\s+.+\s+project|my\s+.+\s+code|this\s+.+\s+module|this\s+.+\s+file)",
        r"(my\s+project|my\s+code|this\s+module|this\s+file)",
        
        # === Вопросы о предпочтениях ===
        r"(я\s+.+\s+предпочитаю|мне\s+.+\s+нравится|как\s+я\s+.+\s+люблю)",
        r"(я\s+предпочитаю|мне\s+нравится|как\s+я\s+люблю)",
        r"(i\s+.+\s+prefer|i\s+.+\s+like|how\s+i\s+.+\s+like)",
        r"(i\s+prefer|i\s+like|how\s+i\s+like)",
        
        # === Вопросы о прошлых событиях/разговорах (гибкие паттерны) ===
        r"(мы\s+.+\s+обсуждали|мы\s+.+\s+говорили|ты\s+помнишь|помнишь\s+как)",
        r"(мы\s+обсуждали|мы\s+говорили|ты\s+помнишь|помнишь\s+как)",
        r"(we\s+.+\s+discussed|we\s+.+\s+talked|do\s+you\s+remember)",
        r"(we\s+discussed|we\s+talked|do\s+you\s+remember)",
        
        # === Прямые вопросы о прошлых сообщениях/разговорах ===
        r"(о\s+чём\s+мы\s+говорили|что\s+мы\s+обсуждали|последн[иеыхих]*\s*сообщен)",
        r"(какие\s+были\s+темы|какие\s+были\s+идеи|какие\s+были\s+планы)",
        r"(what\s+did\s+we\s+talk|last\s+messages|previous\s+conversation)",
        r"(what\s+were\s+the\s+topics|what\s+were\s+the\s+ideas|what\s+were\s+the\s+plans)",
        
        # === Вопросы о продолжении/возврате к теме ===
        r"(вернись\s+к|возвращаясь\s+к|продолжим\s+про|давай\s+ещё\s+раз\s+про)",
        r"(continue\s+about|back\s+to|let'?s\s+continue\s+on|more\s+about)",
        
        # === Вопросы "что ты помнишь" ===
        r"(что\s+ты\s+помнишь|что\s+помнишь\s+про|что\s+помнишь\s+о)",
        r"(what\s+do\s+you\s+remember|what\s+remember\s+about)",
        
        # === Вопросы "расскажи ещё раз" ===
        r"(расскажи\s+ещё|расскажи\s+снова|повтори\s+ещё|повтори\s+снова)",
        r"(tell\s+again|repeat\s+again|once\s+more)",
    ]
    import re
    has_context_dependent_pattern = any(
        re.search(pattern, query) for pattern in context_dependent_patterns
    )

    # 3. Есть ли unresolved questions в state
    open_questions = state.get("open_questions") or state.get("unresolved_questions")
    has_open_questions = bool(open_questions and (
        (isinstance(open_questions, list) and len(open_questions) > 0)
        or (isinstance(open_questions, str) and open_questions.strip())
    ))

    # 4. Есть ли active task / goal
    active_task = state.get("active_goal") or state.get("current_task") or state.get("task")
    has_active_task = bool(active_task and str(active_task).strip())

    # 5. User-specific preferences в identity_core
    addressing = _as_dict(state.get("addressing") or state.get("user_addressing") or {})
    has_identity_core = bool(addressing and (
        addressing.get("canonical_name")
        or addressing.get("allowed_forms")
        or addressing.get("forbidden_forms")
    ))

    # 6. Вопросы о пользователе (Таня, Ася, имя, факты о себе)
    personal_questions = [
        r"\b(таня|ася|ты\s+о\s+себе|ты\s+помнишь\s+себя)\b",
        r"\b(who\s+are\s+you|what\s+do\s+you\s+remember|about\s+you)\b",
        # Вопросы о личных фактах (имя, возраст и т.д.)
        r"\b(what\s+is\s+my|как\s+меня|моё\s+имя|my\s+name)\b",
    ]
    has_personal_question = any(
        re.search(pattern, query) for pattern in personal_questions
    )

    # 7. Дополнительные эвристики для memory-dependent вопросов
    # Вопросы с "какой"/"какие" о прошлых решениях
    has_which_question = bool(re.search(r"(какой\s+мы|какие\s+мы|какой\s+ты|какие\s+ты)", query))
    
    # Вопросы с "почему" о прошлых действиях
    has_why_question = bool(re.search(r"(почему\s+мы|почему\s+ты|why\s+we|why\s+you)", query))
    
    # Вопросы с "когда" о прошлых событиях
    has_when_question = bool(re.search(r"(когда\s+мы|когда\s+ты|when\s+we|when\s+you)", query))

    # Комбинируем сигналы — жёсткий gate
    score = 0
    if has_continuity_marker:
        score += 2
    if has_context_dependent_pattern:
        score += 3
    if has_open_questions:
        score += 2
    if has_active_task:
        score += 1
    if has_identity_core and any(query.count(word) for word in ["я", "мне", "меня", "мной", "i ", "i'", "me ", "my "]):
        # Личные местоимения + есть identity_core = возможна персонализация
        score += 1
    if has_personal_question:
        score += 3  # Вопросы о себе — всегда memory-dependent
    if has_which_question or has_why_question or has_when_question:
        score += 2  # Вопросы о прошлых событиях

    # Порог срабатывания снижен для жёсткого gate
    return score >= 2


def _agent_loop_tool_limit(ctx: PipelineContext) -> int:
    raw = _pick_value(ctx.meta.get("agent_tool_limit"), ctx.policies.get("agent_tool_limit"), _AGENT_LOOP_MAX_TOOL_CALLS)
    try:
        value = int(raw)
    except Exception:
        value = _AGENT_LOOP_MAX_TOOL_CALLS
    return max(_AGENT_LOOP_MIN_TOOL_CALLS, min(_AGENT_LOOP_MAX_TOOL_CALLS, value))


def _agent_loop_tools(ctx: PipelineContext) -> list[ToolSpec]:
    """
    Вернуть список tools для agent loop.

    Включает:
    - memory_retrieve (semantic search)
    - history_read_recent (точные последние сообщения)
    - history_search (поиск по истории)
    """
    memory_core = (
        ctx.meta.get("memory_core")
        or ctx.meta.get("memory_manager")
        or getattr(ctx.meta.get("pipeline"), "memory_core", None)
    )
    tools: list[ToolSpec] = []

    # Memory retrieve (semantic search) — всегда добавляем, если memory_core доступен
    if memory_core is not None:
        tools.append(_memory_retrieve_tool_spec())

    topic_store = _resolve_topic_store(ctx)
    if topic_store is not None:
        tools.extend(topic_tools_list())

    # History tools (exact DB reads)
    tools.extend(history_tools_list())

    return tools


def _merge_tool_specs(current: list[ToolSpec], extra: list[ToolSpec]) -> list[ToolSpec]:
    out: list[ToolSpec] = []
    seen: set[str] = set()
    for row in list(current or []) + list(extra or []):
        name = str(getattr(row, "name", "") or "").strip()
        key = name.lower()
        if not name or key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def _inject_agent_loop_messages(messages: list[Message], tools: list[ToolSpec]) -> list[Message]:
    tool_names = {str(row.name or "").strip().lower() for row in list(tools or []) if str(row.name or "").strip()}
    has_memory_tool = _MEMORY_TOOL_NAME in tool_names
    has_topic_tools = bool(tool_names & _TOPIC_TOOL_NAMES)
    if not has_memory_tool and not has_topic_tools:
        return list(messages or [])
    lines = ["Agent loop rules:"]
    if has_memory_tool:
        lines.extend(
            [
                "- If user-specific memory, prior dialogue, identity, preferences, plans, or unresolved context may matter, call memory_retrieve before answering.",
                "- Call memory_retrieve with a structured retrieval plan: mode, topic_hints, time_hint, sources, top_k.",
                "- Do not send a long natural-language query inside the tool call. The code will build fan-out retrieval queries for you.",
                "- memory_retrieve returns memory candidates and context blocks, not a final answer.",
                "- Use at most one memory_retrieve call unless a second pass is truly necessary.",
            ]
        )
    if has_topic_tools:
        lines.extend(
            [
                "- Use topic_search to find hidden topic threads inside the current visible chat by subject or keyword.",
                "- Use topic_read to inspect one topic thread in detail: summary, open questions, recent episodes, linked tasks, and recent artifacts.",
                "- Use topic_related to expand a topic into nearby related threads before answering cross-topic questions.",
                "- Prefer topic tools for navigating hidden project threads; prefer memory_retrieve for profile, fact, and continuity grounding.",
            ]
        )
    lines.append("- After tool results arrive, answer the user directly and do not mention the tool protocol.")
    instruction = "\n".join(lines)
    return [Message(role="system", content=instruction), *list(messages or [])]


def _parse_tools(value) -> list[ToolSpec]:
    out: list[ToolSpec] = []
    for row in _as_list(value):
        if not isinstance(row, dict):
            continue
        name = str(row.get("name") or row.get("tool") or "").strip()
        if not name:
            continue
        out.append(
            ToolSpec(
                name=name,
                description=str(row.get("description") or ""),
                input_schema=dict(row.get("input_schema") or row.get("parameters") or {}),
            )
        )
    return out


def _append_policy_rule(policies: dict[str, Any], rule: str) -> None:
    text = str(rule or "").strip()
    if not text:
        return
    rules = policies.get("rules")
    if not isinstance(rules, list):
        rules = [] if rules is None else [rules]
    low = {str(x).strip().lower() for x in rules if str(x).strip()}
    if text.lower() in low:
        return
    rules.append(text)
    policies["rules"] = rules


def _request_metadata(ctx: PipelineContext) -> dict[str, Any]:
    tags = dict(ctx.tags or {})
    meta = dict(ctx.meta or {})
    
    # Загружаем performance profile и применяем настройки
    profile_name = str(ctx.state.get("quality_profile") or ctx.meta.get("quality_profile") or "BALANCED").strip().upper()
    ollama_options = _load_ollama_options_from_performance_profile(profile_name)
    if ollama_options:
        # Применяем настройки из profile (если не переопределены в meta)
        for key, value in ollama_options.items():
            if key not in meta and value is not None:
                meta[key] = value
    
    out = {
        "trace_id": str(meta.get("trace_id") or f"trace_{int(time.time() * 1000)}"),
        "user_id": str(meta.get("user_id") or "anonymous"),
        "tags": tags,
    }
    personality_id = str(
        ctx.state.get("active_character_id")
        or ctx.state.get("active_personality_id")
        or ""
    ).strip().lower()
    if personality_id:
        out["personality_id"] = personality_id
        out["character_id"] = personality_id
    for key in (
        "num_ctx",
        "num_thread",
        "num_gpu",
        "num_batch",
        "keep_alive",
        "think",
        "verbose",
        "user_greeting",
        "allow_greeting",
        "new_session",
        "greeted_today",
        "conversation_state",
        "smalltalk_allowed",
        "should_ask_back",
        "local_date",
        "local_region",
        "dialog_mode",
        "address_terms_policy",
        "allowed_term",
        "use_term_now",
        "verbosity_level",
        "strictness_level",
        "warmth_level",
        "sarcasm_level",
        "is_technical",
        "active_mode",
        "mode_lock",
        "now_iso",
        "timezone",
        "previous_user_at",
        "minutes_since_previous",
        "same_calendar_day",
        "continuation_ref",
        "context_confidence",
    ):
        if key in meta:
            out[key] = meta.get(key)
    return out


def _load_ollama_options_from_performance_profile(profile_name: str) -> dict[str, Any]:
    """
    Загрузить Ollama настройки из performance profile.
    
    Performance profiles хранятся в data/specs/performance_profiles.json
    """
    try:
        from pathlib import Path
        import json
        
        specs_file = Path(__file__).parent.parent / "data" / "specs" / "performance_profiles.json"
        if not specs_file.exists():
            return {}
        
        with open(specs_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        profiles = data.get("profiles", {})
        profile = profiles.get(profile_name, {})
        ollama = profile.get("ollama", {})
        
        # Возвращаем только Ollama настройки
        options = {}
        for key in ("num_ctx", "num_thread", "num_gpu", "num_batch", "keep_alive"):
            if key in ollama and ollama[key] is not None:
                options[key] = ollama[key]
        
        return options
    except Exception:
        return {}


def _inject_temporal_grounding(ctx: PipelineContext) -> None:
    meta = _as_dict(ctx.meta)
    state = _as_dict(ctx.state)
    cooldowns = _as_dict(state.get("cooldowns"))
    context_tags = _as_dict(state.get("context_tags"))

    timezone_name = str(
        _pick(
            meta.get("timezone"),
            context_tags.get("timezone"),
            meta.get("user_timezone"),
            "Europe/Kiev",
        )
    ).strip() or "Europe/Kiev"
    tzinfo = _zoneinfo_or_utc(timezone_name)
    now_dt = dt.datetime.now(tzinfo)

    previous_user_at = str(cooldowns.get("prev_user_ts") or cooldowns.get("last_user_ts") or "").strip()
    minutes_since_previous: int | None = None
    same_calendar_day = False
    if previous_user_at:
        prev_ts = parse_time_to_epoch(previous_user_at, 0.0)
        if float(prev_ts) > 0:
            prev_dt = dt.datetime.fromtimestamp(float(prev_ts), tz=tzinfo)
            delta_minutes = max(0.0, (now_dt - prev_dt).total_seconds() / 60.0)
            minutes_since_previous = int(round(delta_minutes))
            same_calendar_day = bool(prev_dt.date() == now_dt.date())

    meta["now_iso"] = now_dt.isoformat()
    meta["timezone"] = timezone_name
    meta["previous_user_at"] = previous_user_at
    meta["minutes_since_previous"] = (
        "" if minutes_since_previous is None else str(max(0, int(minutes_since_previous)))
    )
    meta["same_calendar_day"] = "true" if same_calendar_day else "false"
    ctx.meta = meta


def _apply_temporal_consistency_guard(*, text: str, user_text: str, meta: dict[str, Any]) -> tuple[str, bool]:
    source = str(text or "").strip()
    if not source:
        return ("", False)

    same_day = _to_bool(meta.get("same_calendar_day"), default=False)
    minutes = _to_int(meta.get("minutes_since_previous"), None)
    if not same_day or minutes is None or int(minutes) >= 180:
        return (source, False)

    user_low = str(user_text or "").strip().lower()
    if any(token in user_low for token in ("вчера", "позавчера", "yesterday", "day before yesterday")):
        return (source, False)

    out = source
    replacement_ru = _relative_time_ru(int(minutes))
    replacement_en = _relative_time_en(int(minutes))

    out = re.sub(r"(?i)\bпозавчера\b", replacement_ru, out)
    out = re.sub(r"(?i)\bвчера\b", replacement_ru, out)
    out = re.sub(r"(?i)\bday before yesterday\b", replacement_en, out)
    out = re.sub(r"(?i)\byesterday\b", replacement_en, out)

    changed = out != source
    return (out, changed)


def _relative_time_ru(minutes: int) -> str:
    mins = max(1, int(minutes))
    if mins < 60:
        return f"{mins} минут назад"
    hours = max(1, int(round(float(mins) / 60.0)))
    return f"{hours} часов назад"


def _relative_time_en(minutes: int) -> str:
    mins = max(1, int(minutes))
    if mins < 60:
        return f"{mins} minutes ago"
    hours = max(1, int(round(float(mins) / 60.0)))
    return f"{hours} hours ago"


def _zoneinfo_or_utc(name: str) -> dt.tzinfo:
    token = str(name or "").strip()
    if not token:
        return dt.timezone.utc
    try:
        from zoneinfo import ZoneInfo

        return ZoneInfo(token)
    except Exception:
        return dt.timezone.utc


def _tool_call_to_dict(row: ToolCall) -> dict[str, Any]:
    return {
        "id": str(row.id or ""),
        "tool": str(row.name or ""),
        "args": dict(row.arguments or {}),
        "raw_arguments": str(row.raw_arguments or ""),
    }


def _extract_tool_calls(text: str) -> list[dict[str, Any]]:
    raw = str(text or "").strip()
    if not raw:
        return []
    try:
        payload = json.loads(raw)
    except Exception:
        return []
    if isinstance(payload, dict) and payload.get("tool"):
        return [
            {
                "id": str(payload.get("id") or "tool_1"),
                "tool": str(payload.get("tool") or ""),
                "args": dict(payload.get("args") or {}),
                "raw_arguments": json.dumps(payload.get("args") or {}, ensure_ascii=False),
            }
        ]
    if isinstance(payload, list):
        out: list[dict[str, Any]] = []
        for idx, row in enumerate(payload):
            if not isinstance(row, dict) or not row.get("tool"):
                continue
            out.append(
                {
                    "id": str(row.get("id") or f"tool_{idx+1}"),
                    "tool": str(row.get("tool") or ""),
                    "args": dict(row.get("args") or {}),
                    "raw_arguments": json.dumps(row.get("args") or {}, ensure_ascii=False),
                }
            )
        return out
    return []


def _looks_like_json(value: str) -> bool:
    raw = str(value or "").strip()
    return (raw.startswith("{") and raw.endswith("}")) or (raw.startswith("[") and raw.endswith("]"))


_THINK_BLOCK_RE = re.compile(
    r"<(?:think|thinking|reasoning)>(.*?)</(?:think|thinking|reasoning)>",
    flags=re.IGNORECASE | re.DOTALL,
)


def _split_visible_and_thinking_blocks(text: str) -> tuple[str, str]:
    raw = str(text or "")
    if not raw:
        return "", ""

    thoughts: list[str] = []

    def _collect(match: re.Match) -> str:
        part = str(match.group(1) or "").strip()
        if part:
            thoughts.append(part)
        return ""

    visible = _THINK_BLOCK_RE.sub(_collect, raw).strip()
    thinking = "\n\n".join(thoughts).strip()
    return visible, thinking


def _unwrap_safety_output_json(text: str, *, preserve_json: bool) -> tuple[str, bool]:
    raw = str(text or "").strip()
    if not raw or preserve_json:
        return raw, False

    visible, thinking = _split_visible_and_thinking_blocks(raw)
    candidate = visible if visible else raw
    if not _looks_like_json(candidate):
        return raw, False

    try:
        payload = json.loads(candidate)
    except Exception:
        return raw, False
    if not isinstance(payload, dict):
        return raw, False
    if "tool" in payload and "args" in payload:
        return raw, False
    if "output" not in payload:
        return raw, False
    if not any(key in payload for key in ("safe", "reason", "output")):
        return raw, False

    output = payload.get("output")
    unwrapped = ""
    if isinstance(output, str):
        text_out = output.strip()
        if text_out:
            unwrapped = text_out
    elif isinstance(output, (dict, list)):
        unwrapped = json.dumps(output, ensure_ascii=False)

    if not unwrapped:
        reason = str(payload.get("reason") or "").strip()
        if reason:
            unwrapped = reason
    if not unwrapped:
        return raw, False

    if thinking:
        return f"{unwrapped}\n<think>{thinking}</think>", True
    return unwrapped, True


_DROP_ROLE_LINE_RE = re.compile(r"^\s*(?:thinking|you|user|system)\s*>\s*", flags=re.IGNORECASE)
_ASSISTANT_LINE_PREFIX_RE = re.compile(r"^\s*assistant\s*>\s*", flags=re.IGNORECASE)
_MODEL_LINE_RE = re.compile(r"^\s*\[model:[^\]]+\]\s*$", flags=re.IGNORECASE)
_THINKING_HEADER_RE = re.compile(r"^\s*\[thinking\].*$", flags=re.IGNORECASE)
_WEB_NOISY_MEMORY_MARKERS = ("[parameters]", "[summary]", "[response]")
_WEB_NOISY_CHATTER_RE = re.compile(
    r"^\s*(?:ах|ой|ну)\b.*\b(?:опять|снова|шутк|новост|что посмотреть)\b",
    flags=re.IGNORECASE,
)
_FACTUAL_CONTEXT_MARKERS = {
    "fx_rate": (
        "курс",
        "exchange rate",
        "currency",
        "forex",
        "usd",
        "eur",
        "uah",
        "gbp",
        "доллар",
        "евро",
        "гривн",
        "грн",
        "валют",
        "nbu",
        "cash",
        "банк",
        "котиров",
        "обмен",
    ),
    "price": (
        "price",
        "pricing",
        "cost",
        "costs",
        "стоимость",
        "цена",
        "стоит",
        "sale price",
        "retail",
        "shop",
        "магазин",
        "товар",
        "product",
        "buy now",
        "auction",
    ),
    "historical_factual": (
        "historical",
        "history",
        "archive",
        "архив",
        "истор",
        "в 20",
        "в 19",
        "год",
        "году",
        "earlier",
        "previously",
        "раньше",
        "ранее",
        "дата",
    ),
    "latest_factual": (
        "latest",
        "current",
        "recent",
        "today",
        "на сегодня",
        "сегодня",
        "точная дата",
        "точное значение",
        "confirmed",
        "mention",
        "mentioned",
        "упомин",
        "последн",
        "когда",
        "дата",
        "событие",
    ),
    "weather": (
        "weather",
        "forecast",
        "temperature",
        "rain",
        "snow",
        "wind",
        "umbrella",
        "погод",
        "прогноз",
        "температур",
        "дожд",
        "снег",
        "ветер",
        "зонтик",
        "градус",
    ),
    "news_release": (
        "news",
        "release",
        "version",
        "changelog",
        "docs",
        "documentation",
        "mentioned",
        "latest",
        "новост",
        "релиз",
        "верси",
        "документац",
        "упомин",
        "дата",
    ),
}
_FACTUAL_CONTEXT_BLOCK_KEYS = {
    "relevant_claims",
    "exact_recall",
    "answer_support",
    "recalled_dialog",
    "document_evidence",
    "supporting_messages",
    "working_memory",
    "session_summary",
    "retrieved_semantic",
    "retrieved_episodic",
    "retrieved_docs",
    "active_tool_state",
    "unresolved_items",
    "conversation_tail",
}
_FX_VALUE_RE = re.compile(r"\b\d+(?:[\.,]\d+)?\b")
_FX_PAIR_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bUSD\s*/\s*UAH\b", re.IGNORECASE), "USD/UAH"),
    (re.compile(r"\bEUR\s*/\s*UAH\b", re.IGNORECASE), "EUR/UAH"),
    (re.compile(r"\bGBP\s*/\s*UAH\b", re.IGNORECASE), "GBP/UAH"),
    (re.compile(r"\bUSD\b.*\bUAH\b|\bUAH\b.*\bUSD\b", re.IGNORECASE), "USD/UAH"),
    (re.compile(r"\bEUR\b.*\bUAH\b|\bUAH\b.*\bEUR\b", re.IGNORECASE), "EUR/UAH"),
    (re.compile(r"\bGBP\b.*\bUAH\b|\bUAH\b.*\bGBP\b", re.IGNORECASE), "GBP/UAH"),
    (re.compile("\u0434\u043e\u043b\u043b\u0430\u0440.*(?:\u0433\u0440\u043d|\u0433\u0440\u0438\u0432)|(?:\u0433\u0440\u043d|\u0433\u0440\u0438\u0432).*\u0434\u043e\u043b\u043b\u0430\u0440", re.IGNORECASE), "USD/UAH"),
    (re.compile("\u0435\u0432\u0440\u043e.*(?:\u0433\u0440\u043d|\u0433\u0440\u0438\u0432)|(?:\u0433\u0440\u043d|\u0433\u0440\u0438\u0432).*\u0435\u0432\u0440\u043e", re.IGNORECASE), "EUR/UAH"),
    (re.compile("\u0444\u0443\u043d\u0442.*(?:\u0433\u0440\u043d|\u0433\u0440\u0438\u0432)|(?:\u0433\u0440\u043d|\u0433\u0440\u0438\u0432).*\u0444\u0443\u043d\u0442", re.IGNORECASE), "GBP/UAH"),
    (re.compile(r"\b(?:dollar|usd)\b", re.IGNORECASE), "USD/UAH"),
    (re.compile(r"\b(?:euro|eur)\b", re.IGNORECASE), "EUR/UAH"),
    (re.compile(r"\b(?:pound|gbp)\b", re.IGNORECASE), "GBP/UAH"),
)
_WEB_STYLE_REPLACEMENTS = (
    re.compile(r"^\s*ах,\s*ты\s+(?:опять|снова)\b[^.!?\n]*[.!?]\s*", flags=re.IGNORECASE),
    re.compile(r"^\s*ах,\s*ты\b[^.!?\n]*[.!?]\s*", flags=re.IGNORECASE),
    re.compile(r"^\s*(?:ах|ой|ну)\b[^.!?\n]{0,180}[.!?]\s*", flags=re.IGNORECASE),
)
_WEB_FALSE_LIMITATION_RE = (
    re.compile(
        r"(?:^|[\s\n])(?:я|мы)\s+(?:не\s+могу|не\s+можем|не\s+умею|не\s+имею\s+доступа|не\s+могу\s+сейчас)\b"
        r"[^.!?\n]{0,180}\b(?:проверить|посмотреть|найти|искать|погоду|курс|новости|данные|интернет|веб|сайт)\b[^.!?\n]*[.!?]?\s*",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"(?:^|[\s\n])(?:у\s+меня\s+нет|нет)\s+доступа\s+к\s+(?:интернету|вебу|сети)\b[^.!?\n]*[.!?]?\s*",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"(?:^|[\s\n])i\s+(?:can't|cannot|do\s+not\s+have\s+access)\b[^.!?\n]{0,180}\b(?:browse|check|look\s+up|access|internet|web|live\s+data|weather|rates|news)\b[^.!?\n]*[.!?]?\s*",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"(?:^|[\s\n])(?:но|but)?\s*без\s+прямого\s+доступа\s+к\s+внешним\s+ресурсам\b[^.!?\n]{0,260}"
        r"(?:не\s+могу|не\s+можем)\s+(?:предоставить|дать|сообщить|подтвердить)\b[^.!?\n]{0,120}"
        r"(?:актуальн\w*\s+данн\w*|данн\w*|курс|погод\w*|новост\w*)[^.!?\n]*[.!?]?\s*",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"(?:^|[\s\n])(?:я|мы)\s+понима\w+[^.!?\n]{0,180}(?:но|but)\b[^.!?\n]{0,220}"
        r"(?:не\s+могу|не\s+можем)\s+(?:предоставить|дать|сообщить|подтвердить)\b[^.!?\n]{0,120}"
        r"(?:актуальн\w*\s+данн\w*|данн\w*|курс|погод\w*|новост\w*)[^.!?\n]*[.!?]?\s*",
        flags=re.IGNORECASE,
    ),
)
_WEB_EMPTYISH_RE = re.compile(r"^(?:[^\w\u0400-\u04FF]*|(?:but|но|однако|however)\b[^\w\u0400-\u04FF]*)*$", flags=re.IGNORECASE)
_WEB_FALSE_LIMITATION_RESIDUAL_RE = (
    re.compile(r"(?:^|[\s\n])(?:но|but)?\s*без\s+прямого\s+доступа\s+к\s+внешним\s+ресурсам\b[.!?]?\s*", flags=re.IGNORECASE),
    re.compile(r"(?:^|[\s\n])(?:но|but)?\s*(?:without|no)\s+direct\s+access\s+to\s+external\s+resources\b[.!?]?\s*", flags=re.IGNORECASE),
)


def _enforce_response_hygiene(text: str) -> str:
    src = _normalize_text(text)
    if not src:
        return ""
    cleaned = _strip_service_markers(src)
    cleaned = _dedupe_adjacent_blocks(cleaned)
    return _normalize_text(cleaned)


def _filter_retrieved_memories_for_time_sensitive_web(
    items: list[Any],
    *,
    target_category: str = "",
) -> tuple[list[Any], int, list[dict[str, Any]]]:
    if not items:
        return [], 0, []
    kept: list[Any] = []
    dropped = 0
    dropped_rows: list[dict[str, Any]] = []
    target = str(target_category or "").strip().lower()
    for item in list(items):
        row = item if isinstance(item, dict) else {}
        metadata = _as_dict(row.get("metadata"))
        source = str(_pick_value(row.get("source"), metadata.get("source"), "") or "").strip().lower()
        topic = str(_pick_value(row.get("topic"), metadata.get("topic"), "") or "").strip().lower()
        text = _normalize_text(_pick_value(row.get("text"), row.get("content"), row.get("summary"), ""))
        title = _normalize_text(_pick_value(row.get("title"), metadata.get("title"), ""))
        compound = _normalize_text(" ".join(part for part in (topic, source, title, text) if part))
        detected_category = _detect_factual_context_category(compound)
        low = compound.lower()
        is_web_evidence = bool(
            topic.startswith("web:")
            or source in {"web", "search", "internet"}
            or "source_url:" in low
            or low.startswith("[web]")
            or low.startswith("[web_search]")
        )
        is_noisy_block = any(marker in low for marker in _WEB_NOISY_MEMORY_MARKERS)
        is_noisy_chatter = bool(_WEB_NOISY_CHATTER_RE.search(low))
        is_chat_like_source = source in {"message", "short", "summary", "chat", "history"}

        drop_reason = ""
        if not is_web_evidence and (is_noisy_block or (is_chat_like_source and is_noisy_chatter)):
            drop_reason = "noisy_memory_fragment"
        elif target and detected_category and detected_category != target:
            drop_reason = f"cross_category:{detected_category}"
        elif target and not detected_category and not is_web_evidence:
            drop_reason = "unmatched_context_fragment"

        if drop_reason:
            dropped += 1
            dropped_rows.append(
                {
                    "source": source,
                    "topic": topic,
                    "detected_category": detected_category,
                    "reason": drop_reason,
                    "preview": _preview_text(text or title or compound, 140),
                }
            )
            continue
        kept.append(item)
    return kept, dropped, dropped_rows


def _isolate_factual_prompt_context(
    *,
    memory_context: dict[str, Any] | None,
    retrieved_memories: list[Any],
    web_intent: str,
) -> tuple[dict[str, Any], list[Any], dict[str, Any]]:
    target = _normalize_factual_context_target(web_intent)
    row = dict(_as_dict(memory_context))
    if target not in {"fx_rate", "weather", "news_release", "price", "historical_factual", "latest_factual"}:
        return row, list(retrieved_memories or []), {}

    filtered_memories, dropped_count, dropped_memories = _filter_retrieved_memories_for_time_sensitive_web(
        list(retrieved_memories or []),
        target_category=target,
    )
    blocks = _as_dict(row.get("blocks"))
    filtered_blocks: dict[str, Any] = {}
    included_blocks: list[str] = []
    dropped_blocks: list[dict[str, Any]] = []
    for key, value in dict(blocks or {}).items():
        name = str(key or "").strip()
        text = _normalize_text(value)
        if not name or not text:
            continue
        if name in {"system_core", "user_message"}:
            filtered_blocks[name] = value
            included_blocks.append(name)
            continue
        if name == "active_tool_state" or (name == "working_memory" and target == "fx_rate"):
            dropped_blocks.append(
                {
                    "block": name,
                    "detected_category": _detect_factual_context_category(text),
                    "reason": "runtime_state_not_allowed_in_factual_context",
                    "preview": _preview_text(text, 160),
                }
            )
            continue
        if name not in _FACTUAL_CONTEXT_BLOCK_KEYS:
            filtered_blocks[name] = value
            included_blocks.append(name)
            continue
        detected_category = _detect_factual_context_category(text)
        if detected_category == target:
            filtered_blocks[name] = value
            included_blocks.append(name)
            continue
        dropped_blocks.append(
            {
                "block": name,
                "detected_category": detected_category,
                "reason": "cross_category_contamination" if detected_category else "unmatched_context_fragment",
                "preview": _preview_text(text, 160),
            }
        )

    row["blocks"] = filtered_blocks
    row["selected"] = [dict(x) for x in list(filtered_memories) if isinstance(x, dict)]
    debug = {
        "target_category": target,
        "included_evidence_blocks": ["WEB_EVIDENCE"] + [f"MEMORY:{name}" for name in included_blocks if name in _FACTUAL_CONTEXT_BLOCK_KEYS],
        "included_memory_blocks": included_blocks,
        "dropped_memory_blocks": dropped_blocks,
        "included_retrieved_memories": len(list(filtered_memories or [])),
        "dropped_retrieved_memories": dropped_memories,
        "strict_context_isolation": True,
    }
    return row, filtered_memories, debug


def _isolate_self_memory_exact_context(
    *,
    memory_context: dict[str, Any] | None,
    retrieved_memories: list[Any],
) -> tuple[dict[str, Any], list[Any], dict[str, Any]]:
    row = dict(_as_dict(memory_context))
    blocks = _as_dict(row.get("blocks"))
    kept_blocks: dict[str, Any] = {}
    included_blocks: list[str] = []
    dropped_blocks: list[dict[str, Any]] = []
    for key in ("memory_recall_mode", "self_facts", "fact_expectation_check"):
        text = _normalize_text(blocks.get(key))
        if text:
            kept_blocks[key] = text
            included_blocks.append(key)

    exact_fact_rows: list[dict[str, Any]] = []
    supporting_messages: list[dict[str, Any]] = []
    dropped_memories: list[dict[str, Any]] = []
    for item in list(retrieved_memories or []):
        row_item = _as_dict(item)
        mem_type = str(row_item.get("memory_type") or "").strip().lower()
        level = str(row_item.get("level") or "").strip().lower()
        status = str(row_item.get("status") or "").strip().lower()
        text = _normalize_text(row_item.get("text"))
        meta = _as_dict(row_item.get("metadata"))
        fact = _as_dict(meta.get("fact"))
        source_kind = str(meta.get("source_kind") or row_item.get("source_kind") or "").strip().lower()
        subject = str(fact.get("subject") or "").strip().lower()

        keep_reason = ""
        drop_reason = ""
        # Exact self-recall should never be influenced by old assistant replies.
        if source_kind == "assistant_reply":
            drop_reason = "assistant_reply_not_allowed_in_self_memory_exact"
        if (
            not drop_reason
            and mem_type == "fact"
            and level == "l3_semantic"
            and status == "active"
            and subject == "user"
            and text
            and not exact_fact_rows
        ):
            exact_fact_rows.append(row_item)
            keep_reason = "exact_active_user_fact"
        elif (
            not drop_reason
            and
            mem_type == "message"
            and source_kind == "user"
            and text
            and not supporting_messages
        ):
            supporting_messages.append(row_item)
            keep_reason = "single_supporting_user_message"

        if keep_reason:
            continue
        dropped_memories.append(
            {
                "id": str(row_item.get("id") or ""),
                "memory_type": mem_type or "unknown",
                "reason": drop_reason or "self_memory_exact_noise",
                "preview": _preview_text(text, 140),
            }
        )

    if exact_fact_rows:
        row_item = exact_fact_rows[0]
        kept_blocks["exact_fact_evidence"] = (
            f"- score={float(_to_float(row_item.get('score'), 0.0) or 0.0):.3f} "
            f"level={str(row_item.get('level') or '').strip()} "
            f"scope={str(row_item.get('scope') or '').strip()} "
            f"{str(row_item.get('text') or '').strip()}"
        ).strip()
        included_blocks.append("exact_fact_evidence")
    if supporting_messages:
        row_item = supporting_messages[0]
        kept_blocks["supporting_message"] = f"- {str(row_item.get('text') or '').strip()}".strip()
        included_blocks.append("supporting_message")

    for key, value in dict(blocks or {}).items():
        name = str(key or "").strip()
        if not name:
            continue
        if name in kept_blocks:
            continue
        text = _normalize_text(value)
        if not text:
            continue
        dropped_blocks.append(
            {
                "block": name,
                "reason": "self_memory_exact_noise",
                "preview": _preview_text(text, 160),
            }
        )

    filtered_selected = list(exact_fact_rows) + list(supporting_messages)
    row["blocks"] = kept_blocks
    row["selected"] = [dict(x) for x in filtered_selected]
    debug = {
        "target_category": _SELF_MEMORY_EXACT_MODE,
        "included_memory_blocks": included_blocks,
        "dropped_memory_blocks": dropped_blocks,
        "included_retrieved_memories": len(filtered_selected),
        "dropped_retrieved_memories": dropped_memories,
        "strict_context_isolation": True,
    }
    return row, filtered_selected, debug


def _normalize_factual_context_target(value: str) -> str:
    token = str(value or "").strip().lower()
    if token in {"fx_rate", "finance"}:
        return "fx_rate"
    if token in {"weather", "weather_factual"}:
        return "weather"
    if token in {"news_release", "version", "release"}:
        return "news_release"
    if token in {"price", "pricing"}:
        return "price"
    if token in {"historical", "historical_factual", "history"}:
        return "historical_factual"
    if token in {"latest", "latest_factual", "external", "news", "generic_factual"}:
        return "latest_factual"
    return ""


def _resolve_web_factual_response_mode(
    *,
    web_intent: str = "",
    web_category: str = "",
    web_evidence_context: dict[str, Any] | None = None,
    quality: dict[str, Any] | None = None,
) -> str:
    intent = str(web_intent or "").strip().lower()
    category = str(web_category or "").strip().lower()
    context = _as_dict(web_evidence_context)
    quality_row = _as_dict(quality)
    numeric_profile = str(
        _pick_value(
            context.get("numeric_profile"),
            quality_row.get("numeric_profile"),
            "",
        )
        or ""
    ).strip().lower()

    if intent == "fx_rate" or category == "finance" or numeric_profile == "fx_rate":
        return "fx_rate"
    if intent == "weather" or category == "weather" or numeric_profile == "weather":
        return "weather"
    if category == "price" or numeric_profile == "price":
        return "price"
    if numeric_profile == "historical":
        return "historical_factual"
    if category == "version" or intent == "news_release":
        return "latest_factual"
    if category in {"external", "news"}:
        return "latest_factual"
    return ""


def _detect_factual_context_category(text: str) -> str:
    low = _normalize_text(text).lower()
    if not low:
        return ""
    scores: dict[str, int] = {}
    for category, markers in dict(_FACTUAL_CONTEXT_MARKERS).items():
        score = 0
        for marker in list(markers or []):
            token = str(marker or "").strip().lower()
            if token and token in low:
                score += 1
        scores[category] = score
    best_category = ""
    best_score = 0
    for category, score in scores.items():
        if score > best_score:
            best_category = category
            best_score = score
    if best_score <= 0:
        return ""
    competing = [cat for cat, score in scores.items() if cat != best_category and score >= best_score]
    if competing:
        competing = [
            cat
            for cat in competing
            if not (cat == "latest_factual" and best_category in {"fx_rate", "price", "weather", "historical_factual"})
        ]
    if competing:
        return ""
    return best_category


def _preview_text(value: Any, limit: int = 140) -> str:
    text = _normalize_text(value)
    if len(text) <= max(1, int(limit)):
        return text
    return text[: max(1, int(limit)) - 1].rstrip() + "…"


def _apply_time_sensitive_web_failsafe(
    text: str,
    *,
    web_intent: str = "",
    web_response_style: str = "",
) -> str:
    src = _normalize_text(text)
    if not src:
        return ""
    intent = _normalize_factual_context_target(web_intent)
    style = str(web_response_style or "").strip().lower()
    if intent not in {"fx_rate", "weather", "news_release", "price", "historical_factual", "latest_factual"}:
        return src
    if style != "factual_direct":
        return src
    cleaned = src
    for _ in range(3):
        changed = False
        for rx in _WEB_STYLE_REPLACEMENTS:
            nxt = rx.sub("", cleaned, count=1)
            if nxt != cleaned:
                cleaned = nxt
                changed = True
        if not changed:
            break
    cleaned = _normalize_text(cleaned)
    return cleaned or src


def _apply_web_tool_sync_guard(
    text: str,
    *,
    meta: dict[str, Any] | None = None,
    web_evidence_context: dict[str, Any] | None = None,
    web_intent: str = "",
) -> tuple[str, bool]:
    src = _normalize_text(text)
    meta_map = _as_dict(meta)
    context = _as_dict(web_evidence_context)
    web_used = _to_bool(_pick_value(meta_map.get("web_used"), False), default=False)
    if not src or not web_used or not context:
        return src, False

    cleaned = src
    changed = False
    for rx in _WEB_FALSE_LIMITATION_RE:
        nxt = rx.sub(" ", cleaned)
        if nxt != cleaned:
            cleaned = nxt
            changed = True
    for rx in _WEB_FALSE_LIMITATION_RESIDUAL_RE:
        nxt = rx.sub(" ", cleaned)
        if nxt != cleaned:
            cleaned = nxt
            changed = True
    cleaned = _normalize_text(cleaned)
    if not changed and _looks_like_false_web_limitation(src):
        fallback = _build_web_tool_sync_fallback(context=context, web_intent=web_intent)
        if fallback:
            return fallback, True
        return src, False
    if not changed:
        return src, False

    if not cleaned or _WEB_EMPTYISH_RE.match(cleaned or ""):
        fallback = _build_web_tool_sync_fallback(context=context, web_intent=web_intent)
        if fallback:
            return fallback, True
        return src, False
    return cleaned, True


def _apply_web_factual_caution_guard(
    text: str,
    *,
    meta: dict[str, Any] | None = None,
    web_evidence_context: dict[str, Any] | None = None,
    web_intent: str = "",
) -> tuple[str, bool]:
    src = _normalize_text(text)
    meta_map = _as_dict(meta)
    context = _as_dict(web_evidence_context)
    if not src or not _to_bool(meta_map.get("web_used"), default=False) or not context:
        return src, False

    quality = _as_dict(meta_map.get("web_evidence_quality"))
    web_category = str(
        _pick_value(
            meta_map.get("web_primary_category"),
            meta_map.get("web_category"),
            "",
        )
        or ""
    ).strip().lower()
    numeric_profile = str(
        _pick_value(
            context.get("numeric_profile"),
            quality.get("numeric_profile"),
            "",
        )
        or ""
    ).strip().lower()
    factual_mode = _resolve_web_factual_response_mode(
        web_intent=_pick_value(
            meta_map.get("factual_response_mode"),
            context.get("factual_response_mode"),
            web_intent,
        ),
        web_category=web_category,
        web_evidence_context=context,
        quality=quality,
    )
    final_confidence = float(
        _to_float(
            _pick_value(
                context.get("final_factual_confidence"),
                quality.get("final_factual_confidence"),
                0.0,
            ),
            0.0,
        )
        or 0.0
    )
    conflict_severity = float(
        _to_float(
            _pick_value(
                context.get("conflict_severity"),
                quality.get("conflict_severity"),
                0.0,
            ),
            0.0,
        )
        or 0.0
    )
    cautious = bool(
        _pick_value(
            context.get("cautious_synthesis"),
            quality.get("cautious_synthesis"),
            False,
        )
    )
    selected_page_type = str(
        _pick_value(
            context.get("selected_result_factual_page_type"),
            quality.get("selected_result_factual_page_type"),
            "",
        )
        or ""
    ).strip().lower()
    unresolved_type_mismatch = bool(
        list(_as_list(context.get("type_mismatch_notes")))
        and not selected_page_type
    )
    strict_numeric = bool(
        numeric_profile in {"fx_rate", "price", "historical"}
        or factual_mode in {"fx_rate", "price", "historical_factual", "latest_factual", "weather"}
        or str(web_intent or "").strip().lower() in {"fx_rate", "weather"}
        or web_category in {"finance", "price", "external", "weather", "news"}
    )
    if not strict_numeric:
        return src, False
    if not cautious and not unresolved_type_mismatch and conflict_severity < 0.55 and final_confidence >= 0.64:
        return src, False
    if not _looks_overconfident_numeric_claim(src):
        return src, False

    fallback = _build_web_cautious_fallback(
        context=context,
        conflict_severity=conflict_severity,
        final_confidence=final_confidence,
    )
    if fallback:
        return fallback, True
    return src, False


def _apply_web_fx_response_guard(
    text: str,
    *,
    meta: dict[str, Any] | None = None,
    web_evidence_context: dict[str, Any] | None = None,
    web_intent: str = "",
) -> tuple[str, bool, dict[str, Any]]:
    src = _normalize_text(text)
    meta_map = _as_dict(meta)
    context = _as_dict(web_evidence_context)
    if not src or not _to_bool(meta_map.get("web_used"), default=False):
        return src, False, {}
    if str(web_intent or "").strip().lower() != "fx_rate":
        return src, False, {}
    if not context:
        return src, False, {}

    formatted, debug = _build_web_fx_compact_response(context=context)
    if not formatted:
        return src, False, debug
    if _normalize_text(formatted) == src:
        return src, False, debug
    return formatted, True, debug


def _build_web_fx_compact_response(*, context: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    query = str(context.get("query") or "").strip()
    key_facts = [str(x).strip() for x in _as_list(context.get("key_facts")) if str(x).strip()]
    summary = str(context.get("summary") or "").strip()
    citations = [str(x).strip() for x in _as_list(context.get("compact_citations")) if str(x).strip()]
    selected_rate_type = str(context.get("selected_rate_type") or context.get("requested_rate_type") or "").strip().lower()
    cautious = _to_bool(context.get("cautious_synthesis"), default=False)
    confidence = float(_to_float(context.get("final_factual_confidence"), 0.0) or 0.0)
    conflict_reason = str(context.get("conflict_reason") or "").strip().lower()

    fact_rows = _extract_fx_fact_rows(query=query, key_facts=key_facts)
    requested_pairs = _fx_requested_pairs(query)
    if not fact_rows and summary:
        fact_rows = _extract_fx_fact_rows(query=query, key_facts=[summary])
    source_label = citations[0] if citations else str(context.get("selected_result_domain") or "").strip()
    type_label = _humanize_fx_rate_type(selected_rate_type)
    cautious_needed = bool(cautious or confidence < 0.64 or conflict_reason in {"true_numeric_conflict", "rate_type_mismatch"})

    lines: list[str] = []
    if fact_rows:
        for row in fact_rows[:2]:
            pair = str(row.get("pair") or "").strip() or "\u041a\u0443\u0440\u0441"
            value = str(row.get("value") or "").strip()
            if cautious_needed:
                note = "\u0442\u043e\u0447\u043d\u043e\u0435 \u0437\u043d\u0430\u0447\u0435\u043d\u0438\u0435 \u043d\u0430\u0434\u0435\u0436\u043d\u043e \u043d\u0435 \u043f\u043e\u0434\u0442\u0432\u0435\u0440\u0436\u0434\u0435\u043d\u043e"
                if conflict_reason == "true_numeric_conflict":
                    note = "\u0442\u043e\u0447\u043d\u043e\u0435 \u0437\u043d\u0430\u0447\u0435\u043d\u0438\u0435 \u043d\u0430\u0434\u0435\u0436\u043d\u043e \u043d\u0435 \u043f\u043e\u0434\u0442\u0432\u0435\u0440\u0436\u0434\u0435\u043d\u043e; \u0438\u0441\u0442\u043e\u0447\u043d\u0438\u043a\u0438 \u0440\u0430\u0441\u0445\u043e\u0434\u044f\u0442\u0441\u044f"
                lines.append(f"{pair}: {note}")
            elif value:
                lines.append(f"{pair}: {value}")
    elif cautious_needed and requested_pairs:
        for pair in requested_pairs[:2]:
            note = "\u0442\u043e\u0447\u043d\u043e\u0435 \u0437\u043d\u0430\u0447\u0435\u043d\u0438\u0435 \u043d\u0430\u0434\u0435\u0436\u043d\u043e \u043d\u0435 \u043f\u043e\u0434\u0442\u0432\u0435\u0440\u0436\u0434\u0435\u043d\u043e"
            if conflict_reason == "true_numeric_conflict":
                note = "\u0442\u043e\u0447\u043d\u043e\u0435 \u0437\u043d\u0430\u0447\u0435\u043d\u0438\u0435 \u043d\u0430\u0434\u0435\u0436\u043d\u043e \u043d\u0435 \u043f\u043e\u0434\u0442\u0432\u0435\u0440\u0436\u0434\u0435\u043d\u043e; \u0438\u0441\u0442\u043e\u0447\u043d\u0438\u043a\u0438 \u0440\u0430\u0441\u0445\u043e\u0434\u044f\u0442\u0441\u044f"
            lines.append(f"{pair}: {note}")
    else:
        if cautious_needed:
            lines.append("\u0422\u043e\u0447\u043d\u043e\u0435 \u0437\u043d\u0430\u0447\u0435\u043d\u0438\u0435 \u043a\u0443\u0440\u0441\u0430 \u043d\u0430\u0434\u0435\u0436\u043d\u043e \u043d\u0435 \u043f\u043e\u0434\u0442\u0432\u0435\u0440\u0436\u0434\u0435\u043d\u043e.")
        elif summary:
            lines.append(summary)

    if not lines:
        return "", {
            "applied": False,
            "reason": "no_fx_fact_basis",
        }

    if type_label:
        lines.append(f"\u0422\u0438\u043f \u043a\u0443\u0440\u0441\u0430: {type_label}")
    if source_label:
        lines.append(f"\u0418\u0441\u0442\u043e\u0447\u043d\u0438\u043a: {source_label}")

    debug = {
        "applied": True,
        "reason": "fx_compact_format",
        "pair_count": int(len(fact_rows)),
        "pairs": [str(row.get("pair") or "").strip() for row in fact_rows[:2] if str(row.get("pair") or "").strip()],
        "cautious": bool(cautious_needed),
        "selected_rate_type": str(selected_rate_type or ""),
        "source": str(source_label or ""),
        "confidence": round(float(confidence), 4),
    }
    return "\n".join([line for line in lines if str(line).strip()]), debug


def _extract_fx_fact_rows(*, query: str, key_facts: list[str]) -> list[dict[str, str]]:
    requested_pairs = _fx_requested_pairs(query)
    rows: list[dict[str, str]] = []
    seen_pairs: set[str] = set()
    for fact in list(key_facts or []):
        pair = _detect_fx_pair(fact)
        if not pair:
            continue
        if requested_pairs and pair not in requested_pairs:
            continue
        if pair in seen_pairs:
            continue
        value = _extract_fx_value_text(fact)
        if not value:
            continue
        rows.append({"pair": pair, "value": value, "fact": fact})
        seen_pairs.add(pair)
    if rows:
        return rows
    for fact in list(key_facts or []):
        pair = _detect_fx_pair(fact) or (requested_pairs[0] if requested_pairs else "")
        value = _extract_fx_value_text(fact)
        if not value:
            continue
        if pair and pair not in seen_pairs:
            rows.append({"pair": pair, "value": value, "fact": fact})
            seen_pairs.add(pair)
        if len(rows) >= 2:
            break
    return rows


def _fx_requested_pairs(query: str) -> list[str]:
    low = str(query or "").strip().lower()
    pairs: list[str] = []
    if any(token in low for token in ("usd", "dollar", "\u0434\u043e\u043b\u043b\u0430\u0440", "\u0434\u043e\u043b\u043b\u0430\u0440\u0430", "\u0431\u0430\u043a\u0441")):
        pairs.append("USD/UAH")
    if any(token in low for token in ("eur", "euro", "\u0435\u0432\u0440\u043e")):
        pairs.append("EUR/UAH")
    if any(token in low for token in ("gbp", "pound", "\u0444\u0443\u043d\u0442")):
        pairs.append("GBP/UAH")
    out: list[str] = []
    seen: set[str] = set()
    for pair in pairs:
        if pair not in seen:
            out.append(pair)
            seen.add(pair)
    return out


def _detect_fx_pair(text: str) -> str:
    src = str(text or "").strip()
    if not src:
        return ""
    for rx, label in _FX_PAIR_PATTERNS:
        if rx.search(src):
            return label
    return ""


def _extract_fx_value_text(text: str) -> str:
    src = str(text or "").strip()
    if not src:
        return ""
    matches = list(_FX_VALUE_RE.finditer(src))
    if not matches:
        return ""
    preferred: list[str] = []
    fallback: list[str] = []
    low = src.lower()
    for match in matches:
        raw = str(match.group(0) or "").strip()
        normalized = raw.replace(",", ".").strip()
        if not normalized:
            continue
        try:
            numeric_value = float(normalized)
        except Exception:
            continue
        if len(raw) == 4 and 1900.0 <= numeric_value <= 2100.0:
            continue
        window_start = max(0, match.start() - 12)
        window_end = min(len(low), match.end() + 12)
        window = low[window_start:window_end]
        if any(token in window for token in ("uah", "\u0433\u0440\u043d", "\u0433\u0440\u0438\u0432")):
            preferred.append(normalized)
            continue
        if "." in normalized and numeric_value >= 1.0:
            preferred.append(normalized)
            continue
        if numeric_value >= 1.0:
            fallback.append(normalized)
    value = preferred[0] if preferred else (fallback[0] if fallback else "")
    if not value:
        return ""
    unit = "UAH"
    if "\u0433\u0440\u043d" in low or "uah" in low:
        unit = "UAH"
    return f"{value} {unit}"


def _humanize_fx_rate_type(rate_type: str) -> str:
    token = str(rate_type or "").strip().lower()
    mapping = {
        "cash_rate": "\u043d\u0430\u043b\u0438\u0447\u043d\u044b\u0439",
        "nbu_rate": "\u041d\u0411\u0423",
        "bank_rate": "\u0431\u0430\u043d\u043a\u043e\u0432\u0441\u043a\u0438\u0439",
        "currency_overview": "\u0440\u044b\u043d\u043e\u0447\u043d\u044b\u0439 \u043e\u0431\u0437\u043e\u0440",
        "currency_index": "\u0432\u0430\u043b\u044e\u0442\u043d\u044b\u0439 \u0438\u043d\u0434\u0435\u043a\u0441",
        "historical_rate": "\u0438\u0441\u0442\u043e\u0440\u0438\u0447\u0435\u0441\u043a\u0438\u0439",
    }
    return str(mapping.get(token, token.replace("_", " ")) or "").strip()


def _build_web_cautious_fallback(
    *,
    context: dict[str, Any],
    conflict_severity: float,
    final_confidence: float,
) -> str:
    citations = [str(x).strip() for x in _as_list(context.get("compact_citations")) if str(x).strip()]
    conflict_notes = [str(x).strip() for x in _as_list(context.get("conflict_notes")) if str(x).strip()]
    if conflict_severity >= 0.55 or conflict_notes:
        reason = "источники расходятся"
    elif final_confidence < 0.50:
        reason = "доступные источники слишком слабые или неполные"
    else:
        reason = "точное значение не подтверждается достаточно надежно"
    if citations:
        return (
            f"Не удалось надежно подтвердить точное значение по текущим веб-источникам: {reason}. "
            f"Лучше перепроверить по {citations[0]}."
        )
    return f"Не удалось надежно подтвердить точное значение по текущим веб-источникам: {reason}."


def _build_web_tool_sync_fallback(*, context: dict[str, Any], web_intent: str = "") -> str:
    summary = str(context.get("summary") or "").strip()
    if summary:
        return summary

    key_facts = [str(x).strip() for x in _as_list(context.get("key_facts")) if str(x).strip()]
    if key_facts:
        if str(web_intent or "").strip().lower() == "weather" and len(key_facts) >= 2:
            return "; ".join(key_facts[:2])
        return key_facts[0]

    citations = [str(x).strip() for x in _as_list(context.get("compact_citations")) if str(x).strip()]
    if citations:
        return f"По веб-источникам есть подтверждение: {citations[0]}"
    return ""


def _looks_like_false_web_limitation(text: str) -> bool:
    low = _normalize_text(text).lower()
    if not low:
        return False
    deny_markers = (
        "не могу проверить",
        "не могу предоставить",
        "не могу дать",
        "не могу сообщить",
        "не могу подтвердить",
        "не имею доступа",
        "нет доступа к интернету",
        "без прямого доступа к внешним ресурсам",
        "can't browse",
        "cannot browse",
        "cannot check",
        "don't have access",
    )
    web_markers = (
        "интернет",
        "веб",
        "внешним ресурсам",
        "актуальн",
        "данн",
        "курс",
        "погод",
        "новост",
        "weather",
        "rate",
        "rates",
        "news",
    )
    return any(marker in low for marker in deny_markers) and any(marker in low for marker in web_markers)


def _looks_overconfident_numeric_claim(text: str) -> bool:
    low = _normalize_text(text).lower()
    if not low:
        return False
    if not re.search(r"\b\d+(?:[.,]\d+)?\b", low) and not re.search(r"\b(?:19|20)\d{2}\b", low):
        return False
    uncertainty_markers = (
        "не удалось",
        "не могу надежно",
        "не получилось подтвердить",
        "источники расходятся",
        "примерно",
        "около",
        "похоже",
        "возможно",
        "диапазон",
        "range",
        "could not reliably confirm",
        "sources disagree",
        "not reliably confirmed",
    )
    if any(marker in low for marker in uncertainty_markers):
        return False
    return True


def _strip_service_markers(text: str) -> str:
    out_lines: list[str] = []
    for raw_line in str(text or "").split("\n"):
        line = str(raw_line or "")
        stripped = line.strip()
        if not stripped:
            out_lines.append("")
            continue
        if _MODEL_LINE_RE.match(stripped):
            continue
        if _THINKING_HEADER_RE.match(stripped):
            continue
        if _DROP_ROLE_LINE_RE.match(line):
            continue
        if _ASSISTANT_LINE_PREFIX_RE.match(line):
            line = _ASSISTANT_LINE_PREFIX_RE.sub("", line, count=1)
            if not line.strip():
                continue
        out_lines.append(line.rstrip())
    return "\n".join(out_lines)


def _dedupe_adjacent_blocks(text: str) -> str:
    src = str(text or "").strip()
    if not src:
        return ""

    paragraphs = re.split(r"\n\s*\n+", src)
    kept: list[str] = []
    prev_key = ""
    for para in paragraphs:
        p = _dedupe_adjacent_lines(para)
        if not p:
            continue
        key = _dedupe_key(p)
        if key and key == prev_key:
            continue
        kept.append(p)
        prev_key = key

    if not kept:
        return ""
    if len(kept) == 1:
        return kept[0]
    return "\n\n".join(kept)


def _dedupe_adjacent_lines(text: str) -> str:
    lines = str(text or "").split("\n")
    kept: list[str] = []
    prev_key = ""
    for raw in lines:
        line = str(raw or "").strip()
        if not line:
            continue
        key = _dedupe_key(line)
        if key and key == prev_key:
            continue
        kept.append(line)
        prev_key = key
    return _normalize_text("\n".join(kept))


def _dedupe_key(text: str) -> str:
    src = _normalize_text(text).lower()
    if not src:
        return ""
    src = re.sub(r"[\"'`Р В РІР‚в„ўР вЂ™Р’В«Р В РІР‚в„ўР вЂ™Р’В»Р В Р вЂ Р В РІР‚С™Р РЋРІР‚С”Р В Р вЂ Р В РІР‚С™Р РЋРЎв„ўР В Р вЂ Р В РІР‚С™Р РЋРЎС™]", "", src)
    src = re.sub(r"[\s\.,;:!?()\[\]{}\-_/\\]+", " ", src)
    return src.strip()


def _compute_greeting_flags(
    *,
    text: str,
    state: dict[str, Any],
    meta: dict[str, Any],
    policies: dict[str, Any],
) -> dict[str, Any]:
    src = _normalize_text(text)
    state_map = _as_dict(state)
    meta_map = _as_dict(meta)
    policy_map = _as_dict(policies)
    cooldowns = _as_dict(state_map.get("cooldowns"))
    context = _as_dict(state_map.get("context_tags"))

    policy_state = {
        "greeted_on_date": str(cooldowns.get("greeting_date_local") or ""),
        "last_turn_ts": cooldowns.get("prev_user_ts"),
        "session_id": _pick_value(meta_map.get("session_id"), state_map.get("conversation_id"), cooldowns.get("session_id"), ""),
        "prev_session_id": cooldowns.get("prev_session_id"),
        "conversation_state": _pick_value(meta_map.get("conversation_state"), context.get("conversation_state"), state_map.get("conversation_state"), ""),
        "conversation_id": _pick_value(meta_map.get("conversation_id"), state_map.get("conversation_id"), cooldowns.get("session_id"), ""),
        "turn_id": _pick_value(meta_map.get("turn_id"), state_map.get("turn_id"), 0),
        "address_terms": _as_dict(state_map.get("address_terms")),
    }
    policy_cfg = {
        "new_session_after_min": _pick_value(
            meta_map.get("session_timeout_sec"),
            meta_map.get("greeting_session_timeout_sec"),
            policy_map.get("session_timeout_sec"),
            policy_map.get("greeting_session_timeout_sec"),
            6 * 3600,
        ),
        "greeting_max_words": _pick_value(meta_map.get("greeting_max_words"), policy_map.get("greeting_max_words")),
        "greeting_max_chars": _pick_value(meta_map.get("greeting_max_chars"), policy_map.get("greeting_max_chars")),
        "greetings": _pick_value(meta_map.get("dialog_greetings"), policy_map.get("dialog_greetings")),
        "greeting_exclusions": _pick_value(
            meta_map.get("dialog_greeting_exclusions"),
            policy_map.get("dialog_greeting_exclusions"),
        ),
        "terms_enabled": _pick_value(meta_map.get("terms_enabled"), policy_map.get("terms_enabled")),
        "terms_list": _pick_value(meta_map.get("terms_list"), policy_map.get("terms_list")),
        "terms_cooldown_turns": _pick_value(meta_map.get("terms_cooldown_turns"), policy_map.get("terms_cooldown_turns")),
        "terms_cooldown_seconds": _pick_value(meta_map.get("terms_cooldown_seconds"), policy_map.get("terms_cooldown_seconds")),
        "terms_max_per_session": _pick_value(meta_map.get("terms_max_per_session"), policy_map.get("terms_max_per_session")),
        "terms_insert_probability": _pick_value(meta_map.get("terms_insert_probability"), policy_map.get("terms_insert_probability")),
        "terms_ban_scope": _pick_value(meta_map.get("terms_ban_scope"), policy_map.get("terms_ban_scope")),
        "terms_disable_patterns": _pick_value(meta_map.get("terms_disable_patterns"), policy_map.get("terms_disable_patterns")),
        "terms_enable_patterns": _pick_value(meta_map.get("terms_enable_patterns"), policy_map.get("terms_enable_patterns")),
    }
    try:
        sec = float(policy_cfg["new_session_after_min"] or 6 * 3600)
    except Exception:
        sec = float(6 * 3600)
    policy_cfg["new_session_after_min"] = max(1, int(sec / 60.0))

    policy_meta = {
        "intent": _pick_value(meta_map.get("intent"), context.get("intent"), state_map.get("intent"), ""),
        "emotion": _pick_value(meta_map.get("emotion"), meta_map.get("mood"), context.get("mood"), state_map.get("mood"), ""),
        "mood": _pick_value(meta_map.get("mood"), context.get("mood"), state_map.get("mood"), ""),
        "active_personality_profile": _pick_value(
            meta_map.get("active_personality_profile"),
            meta_map.get("personality"),
            state_map.get("active_character_id"),
            state_map.get("active_personality_id"),
            "",
        ),
        "safety_mode": _pick_value(meta_map.get("safety_mode"), policy_map.get("safety_mode"), ""),
        "safety_lock": _to_bool(_pick_value(meta_map.get("safety_lock"), state_map.get("safety_lock"), False), default=False),
        "conversation_id": _pick_value(meta_map.get("conversation_id"), state_map.get("conversation_id"), ""),
        "turn_id": _pick_value(meta_map.get("turn_id"), state_map.get("turn_id"), 0),
    }
    recent_disable_directive = _recent_user_disable_directive(
        state_map=state_map,
        terms_list=_as_list(_pick_value(policy_cfg.get("terms_list"), ["Р В РЎВР В РЎвЂР В Р’В»Р В Р’В°Р РЋРІвЂљВ¬Р В РЎвЂќР В Р’В°"])),
        disable_patterns=_as_list(_pick_value(policy_cfg.get("terms_disable_patterns"), [])),
    )
    if recent_disable_directive:
        policy_meta["recent_disable_directive"] = True
    flags = dict(dialog_compute_dialog_flags(src, time.time(), policy_state, config=policy_cfg, metadata=policy_meta))

    if meta_map.get("greeted_today") is not None:
        flags["greeted_today"] = _to_bool(meta_map.get("greeted_today"), default=bool(flags.get("greeted_today")))
    if meta_map.get("user_greeting") is not None:
        flags["user_greeting"] = _to_bool(meta_map.get("user_greeting"), default=bool(flags.get("user_greeting")))
    if meta_map.get("new_session") is not None:
        flags["new_session"] = _to_bool(meta_map.get("new_session"), default=bool(flags.get("new_session")))
    if meta_map.get("allow_greeting") is not None:
        flags["allow_greeting"] = _to_bool(meta_map.get("allow_greeting"), default=bool(flags.get("allow_greeting")))
    if meta_map.get("smalltalk_allowed") is not None:
        flags["smalltalk_allowed"] = _to_bool(meta_map.get("smalltalk_allowed"), default=bool(flags.get("smalltalk_allowed")))

    flags.setdefault("conversation_state", str(policy_state.get("conversation_state") or ""))
    dialog_mode = dict(flags.get("dialog_mode") or {})
    if meta_map.get("allow_greeting") is not None:
        dialog_mode["greeting_allowed"] = _to_bool(meta_map.get("allow_greeting"), default=bool(dialog_mode.get("greeting_allowed")))
    if meta_map.get("smalltalk_allowed") is not None:
        dialog_mode["smalltalk_allowed"] = _to_bool(meta_map.get("smalltalk_allowed"), default=bool(dialog_mode.get("smalltalk_allowed")))
    for key in ("greeting_allowed", "smalltalk_allowed"):
        if key in meta_map:
            dialog_mode[key] = _to_bool(meta_map.get(key), default=bool(dialog_mode.get(key)))
    for key in ("sarcasm_level", "warmth_level", "strictness_level", "verbosity_level"):
        if key in meta_map:
            value = _to_float(meta_map.get(key), None)
            if value is not None:
                dialog_mode[key] = max(0.0, min(1.0, float(value)))
    if dialog_mode:
        flags["dialog_mode"] = dialog_mode
        flags["allow_greeting"] = bool(dialog_mode.get("greeting_allowed", flags.get("allow_greeting")))
        flags["smalltalk_allowed"] = bool(dialog_mode.get("smalltalk_allowed", flags.get("smalltalk_allowed")))

    address_terms_policy = _as_dict(flags.get("address_terms_policy"))
    if recent_disable_directive:
        address_terms_policy["use_term_now"] = False
        address_terms_policy["recent_disable_directive"] = True
    if meta_map.get("use_term_now") is not None:
        address_terms_policy["use_term_now"] = _to_bool(meta_map.get("use_term_now"), default=_to_bool(address_terms_policy.get("use_term_now"), default=False))
    if meta_map.get("allowed_term") is not None:
        address_terms_policy["allowed_term"] = str(meta_map.get("allowed_term") or "").strip().lower()
    flags["address_terms_policy"] = address_terms_policy
    flags["allowed_term"] = str(address_terms_policy.get("allowed_term") or flags.get("allowed_term") or "")
    flags["use_term_now"] = _to_bool(address_terms_policy.get("use_term_now"), default=_to_bool(flags.get("use_term_now"), default=False))

    return {
        "user_greeting": bool(flags.get("user_greeting")),
        "new_session": bool(flags.get("new_session")),
        "greeted_today": bool(flags.get("greeted_today")),
        "allow_greeting": bool(flags.get("allow_greeting")),
        "smalltalk_allowed": bool(flags.get("smalltalk_allowed", True)),
        "should_ask_back": bool(flags.get("should_ask_back", False)),
        "conversation_state": str(flags.get("conversation_state") or ""),
        "local_date": str(flags.get("local_date") or _local_day_kyiv()),
        "local_region": str(flags.get("local_region") or _local_region_name()),
        "dialog_mode": dict(flags.get("dialog_mode") or {}),
        "address_terms_policy": dict(address_terms_policy),
        "allowed_term": str(flags.get("allowed_term") or ""),
        "use_term_now": bool(flags.get("use_term_now", False)),
        "is_technical": bool(flags.get("is_technical", False)),
        "decision_path": list(_as_list(flags.get("decision_path"))),
        "intent": str(flags.get("intent") or ""),
        "emotion": str(flags.get("emotion") or ""),
        "user_tone": str(flags.get("user_tone") or ""),
    }


def _resolve_dialog_mode(ctx: PipelineContext) -> dict[str, Any]:
    mode = {}
    for src in (
        _as_dict(ctx.meta.get("dialog_mode")),
        _as_dict(ctx.state.get("dialog_mode")),
        _as_dict(_as_dict(ctx.state).get("context_tags")),
    ):
        if src:
            mode.update(src)

    mode["greeting_allowed"] = _to_bool(
        _pick_value(ctx.meta.get("allow_greeting"), mode.get("greeting_allowed"), mode.get("allow_greeting"), ctx.tags.get("allow_greeting"), True),
        default=True,
    )
    mode["smalltalk_allowed"] = _to_bool(
        _pick_value(ctx.meta.get("smalltalk_allowed"), mode.get("smalltalk_allowed"), ctx.tags.get("smalltalk_allowed"), True),
        default=True,
    )
    mode["sarcasm_level"] = max(0.0, min(1.0, _to_float(_pick_value(mode.get("sarcasm_level"), ctx.meta.get("sarcasm_level"), 0.2), 0.2) or 0.2))
    mode["warmth_level"] = max(0.0, min(1.0, _to_float(_pick_value(mode.get("warmth_level"), ctx.meta.get("warmth_level"), 0.6), 0.6) or 0.6))
    mode["strictness_level"] = max(0.0, min(1.0, _to_float(_pick_value(mode.get("strictness_level"), ctx.meta.get("strictness_level"), 0.55), 0.55) or 0.55))
    mode["verbosity_level"] = max(0.0, min(1.0, _to_float(_pick_value(mode.get("verbosity_level"), ctx.meta.get("verbosity_level"), 0.46), 0.46) or 0.46))
    return mode


def _resolve_address_terms_policy(ctx: PipelineContext) -> dict[str, Any]:
    policy = {}
    for src in (
        _as_dict(ctx.meta.get("address_terms_policy")),
        _as_dict(ctx.state.get("address_terms_policy")),
    ):
        if src:
            policy.update(src)
    policy["allowed_term"] = str(
        _pick_value(
            ctx.meta.get("allowed_term"),
            policy.get("allowed_term"),
            ctx.state.get("allowed_term"),
            _as_dict(ctx.state.get("context_tags")).get("allowed_term"),
            "",
        )
        or ""
    ).strip().lower()
    policy["use_term_now"] = _to_bool(
        _pick_value(
            ctx.meta.get("use_term_now"),
            policy.get("use_term_now"),
            ctx.state.get("use_term_now"),
            _as_dict(ctx.state.get("context_tags")).get("use_term_now"),
            False,
        ),
        default=False,
    )
    return policy


def _resolve_user_addressing(ctx: PipelineContext) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for src in (
        _as_dict(ctx.meta.get("user_addressing")),
        _as_dict(ctx.state.get("user_addressing")),
        _as_dict(_as_dict(ctx.state.get("character_state")).get("user_addressing")),
    ):
        if src:
            out.update(src)
    return out


def _apply_address_term_updates(
    current_state,
    *,
    address_terms_policy: dict[str, Any],
    conversation_id: str,
) -> dict[str, Any]:
    out = _coerce_address_terms_state(current_state, conversation_id=conversation_id)
    policy = _as_dict(address_terms_policy)
    if not policy:
        return out

    ban_updates = _as_dict(policy.get("ban_updates"))
    unban_updates = _as_dict(policy.get("unban_updates"))

    for term in _as_list(ban_updates.get("add_terms")):
        item = str(term or "").strip().lower()
        if not item:
            continue
        if item not in out["banned_terms"]:
            out["banned_terms"].append(item)
    for term, marker in _as_dict(ban_updates.get("banned_terms_until")).items():
        item = str(term or "").strip().lower()
        if not item:
            continue
        out["banned_terms_until"][item] = marker
    if ban_updates.get("last_disable_directive_ts") is not None:
        out["last_disable_directive_ts"] = _to_float(ban_updates.get("last_disable_directive_ts"), out["last_disable_directive_ts"]) or out["last_disable_directive_ts"]

    remove_terms = [str(x or "").strip().lower() for x in _as_list(unban_updates.get("remove_terms")) if str(x or "").strip()]
    remove_until = [str(x or "").strip().lower() for x in _as_list(unban_updates.get("remove_banned_terms_until")) if str(x or "").strip()]
    if remove_terms:
        out["banned_terms"] = [x for x in out["banned_terms"] if x not in set(remove_terms)]
    for term in remove_until:
        out["banned_terms_until"].pop(term, None)
    if unban_updates.get("last_enable_directive_ts") is not None:
        out["last_enable_directive_ts"] = _to_float(unban_updates.get("last_enable_directive_ts"), out["last_enable_directive_ts"]) or out["last_enable_directive_ts"]
    return _coerce_address_terms_state(out, conversation_id=conversation_id)


def _update_address_terms_after_response(ctx: PipelineContext, text: str) -> dict[str, Any] | None:
    policy = _resolve_address_terms_policy(ctx)
    allowed_term = str(policy.get("allowed_term") or "").strip().lower()
    if not allowed_term:
        return None

    conversation_id = str(ctx.state.get("conversation_id") or ctx.meta.get("conversation_id") or "").strip()
    state_address = _coerce_address_terms_state(ctx.state.get("address_terms"), conversation_id=conversation_id)
    term_used = _text_has_term(text, allowed_term)
    if not term_used:
        return state_address

    now_ts = time.time()
    turn_id = _to_int(_pick_value(ctx.meta.get("turn_id"), ctx.state.get("turn_id"), 0), 0) or 0
    state_address["last_term_used_at"] = now_ts
    if turn_id > 0:
        state_address["term_used_turn_index"] = turn_id
    state_address["terms_used_count_session"] = max(0, int(state_address.get("terms_used_count_session") or 0) + 1)
    return _coerce_address_terms_state(state_address, conversation_id=conversation_id)


def _coerce_address_terms_state(value, *, conversation_id: str) -> dict[str, Any]:
    row = _as_dict(value)
    out = {
        "last_term_used_at": 0.0,
        "term_used_turn_index": 0,
        "terms_used_count_session": 0,
        "session_id_snapshot": "",
        "banned_terms": [],
        "banned_terms_until": {},
        "last_disable_directive_ts": 0.0,
        "last_enable_directive_ts": 0.0,
    }
    for key in out:
        if key in row:
            out[key] = row.get(key)

    out["last_term_used_at"] = _to_float(out.get("last_term_used_at"), 0.0) or 0.0
    out["term_used_turn_index"] = _to_int(out.get("term_used_turn_index"), 0) or 0
    out["terms_used_count_session"] = max(0, _to_int(out.get("terms_used_count_session"), 0) or 0)
    out["last_disable_directive_ts"] = _to_float(out.get("last_disable_directive_ts"), 0.0) or 0.0
    out["last_enable_directive_ts"] = _to_float(out.get("last_enable_directive_ts"), 0.0) or 0.0

    banned_terms: list[str] = []
    seen: set[str] = set()
    for term in _as_list(out.get("banned_terms")):
        item = str(term or "").strip().lower()
        if not item or item in seen:
            continue
        seen.add(item)
        banned_terms.append(item)
    out["banned_terms"] = banned_terms

    banned_until: dict[str, Any] = {}
    for term, marker in _as_dict(out.get("banned_terms_until")).items():
        key = str(term or "").strip().lower()
        if not key:
            continue
        banned_until[key] = marker
    out["banned_terms_until"] = banned_until

    prev_session = str(out.get("session_id_snapshot") or "").strip()
    current_session = str(conversation_id or "").strip()
    if current_session and prev_session and current_session != prev_session:
        out["terms_used_count_session"] = 0
        keep_terms: list[str] = []
        keep_until: dict[str, Any] = {}
        for term in list(out["banned_terms"]):
            marker = str(out["banned_terms_until"].get(term) or "").strip()
            if marker.startswith("session:"):
                continue
            keep_terms.append(term)
            if marker:
                keep_until[term] = marker
        out["banned_terms"] = keep_terms
        out["banned_terms_until"] = keep_until
    if current_session:
        out["session_id_snapshot"] = current_session
    else:
        out["session_id_snapshot"] = prev_session
    return out


def _recent_user_disable_directive(
    *,
    state_map: dict[str, Any],
    terms_list: list[Any],
    disable_patterns: list[Any],
    lookback_messages: int = 4,
) -> bool:
    terms = [str(x or "").strip().lower() for x in list(terms_list or []) if str(x or "").strip()]
    if not terms:
        terms = ["Р В РЎВР В РЎвЂР В Р’В»Р В Р’В°Р РЋРІвЂљВ¬Р В РЎвЂќР В Р’В°"]
    patterns = [str(x or "").strip().lower() for x in list(disable_patterns or []) if str(x or "").strip()]
    if not patterns:
        patterns = ["Р В Р вЂ¦Р В Р’Вµ Р В Р вЂ¦Р В Р’В°Р В Р’В·Р РЋРІР‚в„–Р В Р вЂ Р В Р’В°Р В РІвЂћвЂ“", "Р В РЎвЂ”Р РЋР вЂљР В Р’ВµР В РЎвЂќР РЋР вЂљР В Р’В°Р РЋРІР‚С™Р В РЎвЂ Р В Р вЂ¦Р В Р’В°Р В Р’В·Р РЋРІР‚в„–Р В Р вЂ Р В Р’В°Р РЋРІР‚С™Р РЋР Р‰", "Р В Р вЂ¦Р В Р’Вµ Р В Р’В·Р В РЎвЂўР В Р вЂ Р В РЎвЂ", "Р В РЎвЂ”Р РЋР вЂљР В Р’ВµР В РЎвЂќР РЋР вЂљР В Р’В°Р РЋРІР‚С™Р В РЎвЂ Р В Р’В·Р В Р вЂ Р В Р’В°Р РЋРІР‚С™Р РЋР Р‰"]

    history = list(_as_list(state_map.get("history")))
    user_rows = [row for row in history if str(_as_dict(row).get("role") or "").strip().lower() == "user"]
    tail = user_rows[-max(1, int(lookback_messages)) :]
    for row in reversed(tail):
        text = str(_as_dict(row).get("content") or "").strip()
        if not text:
            continue
        directive = dialog_extract_term_directive(text, terms, disable_patterns=patterns, enable_patterns=[])
        if bool(directive.get("is_disable")):
            return True
    return False


def _text_has_term(text: str, term: str) -> bool:
    src = str(text or "")
    item = str(term or "").strip()
    if not src or not item:
        return False
    pattern = re.compile(rf"(?<!\w){re.escape(item)}(?!\w)", flags=re.IGNORECASE | re.UNICODE)
    return bool(pattern.search(src))


def _compact_json(value) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    except Exception:
        return str(value or "")


def _strip_leading_greeting(text: str) -> str:
    return str(dialog_trim_leading_greeting(_normalize_text(text)))


def _cache_root() -> Path:
    try:
        cfg = load_config()
        return Path(cfg.cache_dir).expanduser().resolve()
    except Exception:
        return (Path(__file__).resolve().parents[1] / "data" / "cache").resolve()


def _cache_stats() -> dict[str, Any]:
    root = _cache_root()
    root.mkdir(parents=True, exist_ok=True)
    namespaces: list[dict[str, Any]] = []
    total_files = 0
    total_size = 0

    for path in sorted(root.iterdir(), key=lambda p: p.name.lower()):
        if not path.is_dir():
            continue
        files = 0
        size = 0
        for row in path.rglob("*"):
            if not row.is_file():
                continue
            files += 1
            try:
                size += int(row.stat().st_size)
            except Exception:
                continue
        total_files += files
        total_size += size
        namespaces.append({"namespace": path.name, "files": files, "size_bytes": size})

    return {
        "root": str(root),
        "namespace_count": len(namespaces),
        "total_files": total_files,
        "total_size_bytes": total_size,
        "namespaces": namespaces,
    }


def _cache_clear() -> dict[str, Any]:
    root = _cache_root()
    root.mkdir(parents=True, exist_ok=True)
    removed_files = 0
    removed_dirs = 0
    errors = 0

    for row in sorted(root.rglob("*"), key=lambda p: len(p.parts), reverse=True):
        try:
            if row.is_file():
                row.unlink()
                removed_files += 1
            elif row.is_dir():
                os.rmdir(row)
                removed_dirs += 1
        except Exception:
            errors += 1

    return {
        "root": str(root),
        "removed_files": removed_files,
        "removed_dirs": removed_dirs,
        "errors": errors,
    }


def _local_day_kyiv() -> str:
    return str(dialog_local_date_kyiv(time.time()))


def _local_region_name() -> str:
    return str(dialog_local_region_name(time.time()))


def _bool_to_text(value: bool) -> str:
    return "true" if bool(value) else "false"


def _format_persona_debug(payload: dict[str, Any]) -> str:
    row = dict(payload or {})
    character = str(row.get("active_character_id") or row.get("character_id") or "default").strip().lower() or "default"
    mode = str(row.get("active_mode") or "chatting").strip().lower() or "chatting"
    locked = bool(row.get("mode_lock", False))
    mood = str(row.get("mood") or "neutral").strip().lower() or "neutral"
    traits = dict(row.get("traits") or {})
    locks = dict(row.get("locks") or {})
    bans = [str(x).strip() for x in list(row.get("bans") or []) if str(x).strip()]
    deltas = [dict(x) for x in list(row.get("last_deltas") or []) if isinstance(x, dict)]
    feedback = [str(x).strip() for x in list(row.get("feedback") or []) if str(x).strip()]
    event_changes = [dict(x) for x in list(row.get("last_event_changes") or []) if isinstance(x, dict)]

    lines = ["persona_debug:"]
    lines.append(f"- character={character}")
    lines.append(f"- mode={mode} ({'locked' if locked else 'auto'})")
    lines.append(f"- mood={mood}")
    if traits:
        ordered = []
        for key in ("warmth", "sarcasm", "teasing", "strictness", "verbosity", "empathy"):
            if key not in traits:
                continue
            try:
                ordered.append(f"{key}={float(traits.get(key)):.2f}")
            except Exception:
                continue
        if ordered:
            lines.append("- traits: " + " ".join(ordered))
    if locks:
        lock_rows = [f"{k}={_bool_to_text(bool(v))}" for k, v in sorted(locks.items(), key=lambda x: str(x[0]))]
        if lock_rows:
            lines.append("- locks: " + ", ".join(lock_rows))
    lines.append("- bans: " + (", ".join(bans) if bans else "(none)"))
    if feedback:
        lines.append("- feedback: " + ", ".join(feedback[:8]))

    if deltas:
        rendered: list[str] = []
        for diff in deltas[:3]:
            pairs: list[str] = []
            for key, values in sorted(diff.items(), key=lambda x: str(x[0])):
                if not isinstance(values, dict):
                    continue
                old = values.get("old")
                new = values.get("new")
                if old == new:
                    continue
                pairs.append(f"{key}:{old}->{new}")
            if pairs:
                rendered.append("; ".join(pairs[:4]))
        if rendered:
            lines.append("- last_deltas: " + " | ".join(rendered[:3]))

    if event_changes:
        compact: list[str] = []
        for item in event_changes[:3]:
            kind = str(item.get("kind") or "").strip().lower()
            if not kind:
                continue
            if kind == "persona_drift":
                td = dict(item.get("trait_deltas") or {})
                if td:
                    bits = [f"{k}:{v}" for k, v in sorted(td.items(), key=lambda x: str(x[0]))]
                    compact.append(f"{kind}({', '.join(bits[:4])})")
                else:
                    compact.append(kind)
            elif kind == "user_feedback":
                fb = [str(x).strip() for x in list(item.get("feedback") or []) if str(x).strip()]
                compact.append(f"{kind}({', '.join(fb[:3])})" if fb else kind)
            else:
                compact.append(kind)
        if compact:
            lines.append("- last_update: " + " | ".join(compact))

    return "\n".join(lines)


def _format_brain_debug(payload: dict[str, Any]) -> str:
    row = dict(payload or {})
    lines = ["brain_debug:"]
    mode = str(row.get("active_mode") or "chatting").strip().lower() or "chatting"
    mode_lock = bool(row.get("mode_lock", False))
    lines.append(f"- active_mode={mode} ({'locked' if mode_lock else 'auto'})")
    lines.append(f"- web_mode={str(row.get('web_mode') or 'auto').strip().lower() or 'auto'}")
    lines.append(f"- thinking_enabled={_bool_to_text(bool(row.get('thinking_enabled', False)))}")
    lines.append(f"- quality_profile={str(row.get('quality_profile') or 'BALANCED').strip().upper() or 'BALANCED'}")
    output_format = _coerce_output_format_state(row.get("output_format"))
    p_manual = output_format.get("show_parameters")
    s_manual = output_format.get("show_summary")
    p_text = ("on" if p_manual else "off") if isinstance(p_manual, bool) else "auto"
    s_text = ("on" if s_manual else "off") if isinstance(s_manual, bool) else "auto"
    lines.append(f"- output_parameters={p_text}")
    lines.append(f"- output_summary={s_text}")
    goal = str(row.get("active_goal") or "").strip()
    lines.append(f"- active_goal={goal or '(none)'}")

    signals = dict(row.get("last_signals") or {})
    if signals:
        parts = []
        for k, v in sorted(signals.items(), key=lambda x: str(x[0])):
            if v is None or v == "" or v == []:
                continue
            parts.append(f"{k}={v}")
        lines.append("- last_signals: " + (", ".join(parts[:10]) if parts else "(none)"))
    else:
        lines.append("- last_signals: (none)")
    feedback = [str(x).strip() for x in list(row.get("feedback") or []) if str(x).strip()]
    if feedback:
        lines.append("- feedback: " + ", ".join(feedback[:8]))

    tasks = [dict(x) for x in list(row.get("active_tasks") or []) if isinstance(x, dict)]
    if tasks:
        out = []
        for item in tasks[:4]:
            label = str(item.get("title") or item.get("name") or item.get("text") or item.get("id") or "").strip()
            if label:
                out.append(label)
        lines.append("- active_tasks: " + (", ".join(out) if out else "(none)"))
    else:
        lines.append("- active_tasks: (none)")

    actions = [dict(x) for x in list(row.get("last_actions") or []) if isinstance(x, dict)]
    if actions:
        chunks = []
        for item in actions[-3:]:
            atype = str(item.get("type") or "EVENT").strip().upper() or "EVENT"
            ts = str(item.get("ts") or "").strip()
            diff = dict(item.get("state_diff") or {})
            diff_bits = []
            for key, values in sorted(diff.items(), key=lambda x: str(x[0])):
                if not isinstance(values, dict):
                    continue
                old = values.get("old")
                new = values.get("new")
                if old == new:
                    continue
                diff_bits.append(f"{key}:{old}->{new}")
            label = f"{atype}@{ts}" if ts else atype
            if atype == "FEEDBACK_RECEIVED":
                fb = [str(x).strip() for x in list(item.get("feedback") or []) if str(x).strip()]
                if fb:
                    label += "(" + ", ".join(fb[:3]) + ")"
            if diff_bits:
                label += " [" + ", ".join(diff_bits[:4]) + "]"
            chunks.append(label)
        lines.append("- last_actions: " + " | ".join(chunks))
    else:
        lines.append("- last_actions: (none)")

    return "\n".join(lines)


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


def _coerce_output_format_state(value) -> dict[str, Any]:
    row = _as_dict(value)
    return {
        "show_parameters": _coerce_nullable_bool(row.get("show_parameters")),
        "show_summary": _coerce_nullable_bool(row.get("show_summary")),
    }


def _extract_topics_from_tags(tags: dict[str, Any]) -> list[str]:
    row = _as_dict(tags)
    found: list[str] = []
    seen: set[str] = set()
    explicit_topic = str(row.get("topic") or "").strip().lower()
    if explicit_topic:
        seen.add(explicit_topic)
        found.append(explicit_topic)
    for tag in list(_as_list(row.get("metadata_tags"))):
        token = str(tag or "").strip().lower()
        if not token.startswith("topic_"):
            continue
        item = token.replace("topic_", "", 1).strip()
        if not item or item in seen:
            continue
        seen.add(item)
        found.append(item)
    return found


def _extract_output_traits(state: dict[str, Any], traits: dict[str, Any]) -> dict[str, float]:
    order = ("warmth", "sarcasm", "teasing", "strictness", "verbosity", "empathy")
    state_map = _as_dict(state)
    traits_map = _as_dict(traits)
    persona_traits = {}
    active_character = str(state_map.get("active_character_id") or "").strip().lower()
    characters = _as_dict(state_map.get("characters"))
    if active_character:
        entry = _as_dict(characters.get(active_character))
        persona = _as_dict(entry.get("persona"))
        persona_traits = _as_dict(persona.get("traits"))
    out: dict[str, float] = {}
    for key in order:
        raw = None
        if key in persona_traits:
            raw = persona_traits.get(key)
        elif key in traits_map:
            raw = traits_map.get(key)
        if raw is None:
            continue
        try:
            out[key] = max(0.0, min(1.0, float(raw)))
        except Exception:
            continue
    return out


def _fallback_summary_from_text(text: str, *, max_sentences: int = 2) -> str:
    src = _normalize_text(text)
    if not src:
        return ""
    parts = re.split(r"(?<=[.!?])\s+", src)
    if not parts:
        return ""
    head = [x.strip() for x in parts[: max(1, int(max_sentences))] if str(x).strip()]
    return _squeeze_summary(" ".join(head))


def _limit_summary_sentences(text: str, *, max_sentences: int = 2) -> str:
    src = _normalize_text(text)
    if not src:
        return ""
    parts = [x.strip() for x in re.split(r"(?<=[.!?])\s+", src) if str(x).strip()]
    if not parts:
        return src
    limited = " ".join(parts[: max(1, int(max_sentences))]).strip()
    return _squeeze_summary(limited)


def _squeeze_summary(text: str, *, max_chars: int = 300) -> str:
    src = _normalize_text(text)
    if not src:
        return ""
    if len(src) <= max(24, int(max_chars)):
        return src
    clipped = src[: max(24, int(max_chars))]
    cut = clipped.rsplit(" ", 1)[0].strip()
    return (cut if cut else clipped.strip()).rstrip(".") + "."


def _render_output_blocks(
    *,
    parameters_lines: list[str],
    summary: str,
    text: str,
    include_parameters: bool,
    include_summary: bool,
    order: list[str] | None = None,
) -> str:
    chunks: list[str] = []
    queue = [str(x).strip().lower() for x in list(order or ["parameters", "summary", "response"]) if str(x).strip()]
    if "response" not in queue:
        queue.append("response")
    for section in queue:
        if section == "parameters":
            if include_parameters:
                chunks.append(_render_parameters_block(parameters_lines))
            continue
        if section == "summary":
            if include_summary:
                summary_text = _normalize_text(summary) or "-"
                chunks.append("[SUMMARY]\n" + summary_text)
            continue
        if section == "response":
            chunks.append("[RESPONSE]\n" + (_normalize_text(text) or "-"))
    return "\n\n".join(chunks).strip()


def _render_parameters_block(lines: list[str]) -> str:
    row = [str(x).strip() for x in list(lines or []) if str(x).strip()]
    if not row:
        row = ["-"]
    return "[PARAMETERS]\n" + "\n".join(row)


def _render_parameters_lines(parameters: dict[str, Any], *, field_order: list[str] | None = None) -> list[str]:
    flat = _flatten_parameter_map(parameters)
    if not flat:
        return []
    order = [str(x).strip() for x in list(field_order or []) if str(x).strip()]
    lines: list[str] = []
    if order:
        for field in order:
            if field not in flat:
                continue
            lines.append(_render_kv_line(field, flat.get(field)))
    else:
        for key in sorted(flat.keys()):
            lines.append(_render_kv_line(key, flat.get(key)))
    return lines


def _render_kv_line(key: str, value: Any) -> str:
    if isinstance(value, float):
        return f"{key}={max(0.0, min(1.0, value)):.2f}"
    if isinstance(value, list):
        compact = [str(x).strip().lower() for x in value if str(x).strip()]
        return f"{key}={', '.join(compact)}"
    return f"{key}={str(value).strip()}"


def _select_parameter_fields(payload: dict[str, Any], *, fields: list[str]) -> dict[str, Any]:
    base = _as_dict(payload)
    if not fields:
        return base
    out: dict[str, Any] = {}
    for raw in list(fields or []):
        field = str(raw or "").strip()
        if not field:
            continue
        value, exists = _get_nested(base, field)
        if not exists:
            continue
        _set_nested(out, field, value)
    return out


def _flatten_parameter_map(payload: dict[str, Any], *, prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    row = _as_dict(payload)
    for key, value in row.items():
        item_key = str(key or "").strip()
        if not item_key:
            continue
        full_key = f"{prefix}.{item_key}" if prefix else item_key
        if isinstance(value, dict):
            out.update(_flatten_parameter_map(value, prefix=full_key))
            continue
        out[full_key] = value
    return out


def _get_nested(payload: dict[str, Any], field: str) -> tuple[Any, bool]:
    cur: Any = payload
    for part in str(field).split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None, False
        cur = cur.get(part)
    return cur, True


def _set_nested(payload: dict[str, Any], field: str, value: Any) -> None:
    parts = [str(x).strip() for x in str(field).split(".") if str(x).strip()]
    if not parts:
        return
    cur = payload
    for part in parts[:-1]:
        child = cur.get(part)
        if not isinstance(child, dict):
            child = {}
            cur[part] = child
        cur = child
    cur[parts[-1]] = value


_CHECKIN_RE = re.compile(
    r"(как\s+(?:у\s+тебя\s+)?дела|как\s+ты|как\s+сам|что\s+нового|как\s+настроение|how\s+are\s+you)",
    flags=re.IGNORECASE,
)
_QUESTION_START_RE = re.compile(
    r"^\s*(как|что|почему|зачем|когда|где|кто|чем|какой|какая|какие|сколько|how|what|why|where|when)\b",
    flags=re.IGNORECASE,
)


def _should_apply_echo_guard(user_msg: str) -> bool:
    src = _normalize_text(user_msg)
    if not src:
        return False
    if "?" in src:
        return True
    return bool(_QUESTION_START_RE.search(src))


def _normalize_for_echo(text: str) -> str:
    src = _normalize_text(text).lower()
    if not src:
        return ""
    src = re.sub(r"<[^>]+>", " ", src)
    src = re.sub(r"[^\w\s]", " ", src, flags=re.UNICODE)
    src = re.sub(r"\s+", " ", src)
    return src.strip()


def _looks_like_echo_response(*, answer: str, user_msg: str) -> bool:
    ans = _normalize_for_echo(answer)
    usr = _normalize_for_echo(user_msg)
    if not ans or not usr:
        return False

    ans_tokens = ans.split(" ")
    usr_tokens = usr.split(" ")
    if len(ans_tokens) > 10 or len(usr_tokens) > 10:
        return False

    if ans == usr:
        return True
    if ans in usr and len(ans) >= max(4, int(len(usr) * 0.75)):
        return True
    if usr in ans and len(usr) >= max(4, int(len(ans) * 0.75)):
        return True
    return False


def _echo_fallback_text(user_msg: str) -> str:
    src = _normalize_text(user_msg)
    if _CHECKIN_RE.search(src):
        return "У меня все нормально, спасибо. Как ты?"
    return "Поняла. Я на связи и готова помочь. Уточни, что именно нужно."


def _as_dict(value) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if value is None:
        return {}
    try:
        return dict(vars(value))
    except Exception:
        return {}


def _as_list(value) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _pick(*values) -> str:
    for value in values:
        text = _normalize_text(value)
        if text:
            return text
    return ""


def _pick_value(*values):
    for value in values:
        if value is None:
            continue
        return value
    return None


def _strip_low_quality_session_summary_from_pack(pack: dict[str, Any] | None) -> dict[str, Any]:
    payload = dict(pack or {})
    blocks = _as_dict(payload.get("blocks"))
    session_summary = str(blocks.get("session_summary") or "").strip()
    if not is_low_quality_session_summary(session_summary):
        return payload
    next_blocks = dict(blocks)
    next_blocks.pop("session_summary", None)
    payload["blocks"] = next_blocks
    return payload


def _ensure_debug_trace(ctx: PipelineContext) -> DebugTrace:
    request_id = str(_pick(ctx.meta.get("request_id"), ctx.meta.get("trace_id"), "")).strip()
    user_text = str(ctx.clean_user_msg or ctx.user_msg or "")
    existing = ctx.state.get("debug_trace")
    if isinstance(existing, DebugTrace):
        existing.request_id = request_id
        existing.user_text = user_text
        return existing
    trace = DebugTrace(
        request_id=request_id,
        user_text=user_text,
    )
    ctx.state["debug_trace"] = trace
    return trace


def _trace_compact_selected_memory(row: dict[str, Any] | None) -> dict[str, Any]:
    item = dict(row or {})
    meta = _as_dict(item.get("metadata"))
    fact = _as_dict(meta.get("fact"))
    claim = _as_dict(meta.get("claim"))
    breakdown = _as_dict(item.get("score_breakdown"))
    out = {
        "id": str(item.get("id") or ""),
        "memory_type": str(item.get("memory_type") or ""),
        "score": float(_to_float(item.get("score"), 0.0) or 0.0),
        "level": str(item.get("level") or ""),
        "scope": str(item.get("scope") or ""),
        "status": str(item.get("status") or ""),
    }
    reason = str(
        _pick_value(
            item.get("reason"),
            item.get("match_reason"),
            item.get("selection_reason"),
            meta.get("reason"),
            "",
        )
        or ""
    ).strip()
    source = str(
        _pick_value(
            item.get("source"),
            item.get("source_kind"),
            meta.get("source"),
            meta.get("source_kind"),
            "",
        )
        or ""
    ).strip()
    if reason:
        out["reason"] = reason
    if source:
        out["source"] = source
    why_selected = _trace_why_selected_from_breakdown(breakdown)
    if why_selected:
        out["why_selected"] = why_selected
    if breakdown:
        out["score_breakdown"] = {
            key: round(float(value), 4)
            for key, value in breakdown.items()
            if _to_float(value, None) is not None and float(_to_float(value, 0.0) or 0.0) > 0.0
        }
    if fact:
        out["predicate"] = str(fact.get("predicate") or "")
        out["value"] = fact.get("value")
        out["subject"] = str(fact.get("subject") or "")
    elif claim:
        out["predicate"] = str(claim.get("predicate") or "")
        out["value"] = claim.get("obj")
        out["subject"] = str(claim.get("subject") or "")
    else:
        text = str(item.get("text") or "").strip()
        if text:
            out["text_preview"] = _text_preview(text, 120)
    return out


def _trace_compact_episode_hit(row: dict[str, Any] | None) -> dict[str, Any]:
    item = dict(row or {})
    episode = _as_dict(item.get("episode"))
    summary_short = str(_pick_value(item.get("summary_short"), episode.get("summary_short"), "") or "").strip()
    summary_reasoning = str(_pick_value(item.get("summary_reasoning"), episode.get("summary_reasoning"), "") or "").strip()
    episode_id = str(_pick_value(item.get("record_id"), episode.get("id"), "") or "")
    return {
        "record_id": episode_id,
        "episode_id": episode_id,
        "score": float(_to_float(item.get("score"), 0.0) or 0.0),
        "topic": str(episode.get("topic") or ""),
        "summary_short": summary_short,
        "summary_reasoning": summary_reasoning,
        "status": str(episode.get("status") or ""),
        "decisions": [str(x).strip() for x in list(_pick_value(item.get("decisions"), episode.get("decisions"), []) or []) if str(x).strip()],
        "open_questions": [str(x).strip() for x in list(episode.get("open_questions") or []) if str(x).strip()],
    }


def _trace_compact_filtered_item(row: dict[str, Any] | None) -> dict[str, Any]:
    item = dict(row or {})
    return {
        "id": str(_pick_value(item.get("id"), item.get("record_id"), "") or ""),
        "reason": str(item.get("reason") or item.get("drop_reason") or ""),
        "memory_type": str(item.get("memory_type") or ""),
        "score": float(_to_float(item.get("score"), 0.0) or 0.0),
    }


def _trace_why_selected_from_breakdown(breakdown: dict[str, Any] | None) -> list[str]:
    row = dict(breakdown or {})
    weights = {
        "semantic_similarity": float(_to_float(row.get("semantic_similarity"), 0.0) or 0.0),
        "lexical_score": float(_to_float(row.get("lexical_score"), 0.0) or 0.0),
        "recency_score": float(_to_float(row.get("recency_score"), 0.0) or 0.0),
        "importance_score": float(_to_float(row.get("importance_score"), 0.0) or 0.0),
        "confidence_score": float(_to_float(row.get("confidence_score"), 0.0) or 0.0),
        "entity_overlap_score": float(_to_float(row.get("entity_overlap_score"), 0.0) or 0.0),
        "numeric_overlap_score": float(_to_float(row.get("numeric_overlap_score"), 0.0) or 0.0),
        "exact_match_boost": float(_to_float(row.get("exact_match_boost"), 0.0) or 0.0),
        "scope_match_score": float(_to_float(row.get("scope_match_score"), 0.0) or 0.0),
    }
    ranked = sorted(weights.items(), key=lambda item: float(item[1]), reverse=True)
    return [str(name) for name, value in ranked if float(value) > 0.0][:3]


def _trace_prompt_block_titles(text: str) -> list[str]:
    src = str(text or "")
    return [
        str(x).strip()
        for x in re.findall(r"^\[([A-Z0-9_]+)\]\s*$", src, flags=re.M)
        if str(x).strip()
    ]


def _trace_prompt_omitted_blocks(
    *,
    memory_blocks: dict[str, Any] | None,
    active_task_block: str,
    included_titles: list[str] | None,
    context_isolation: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    order = [
        ("MEMORY_RECALL_MODE", "memory_recall_mode"),
        ("ACTIVE_TASK", "__active_task__"),
        ("SELF_FACTS", "self_facts"),
        ("FACT_EXPECTATION_CHECK", "fact_expectation_check"),
        ("RELEVANT_CLAIMS", "relevant_claims"),
        ("RECALLED_DIALOG", "recalled_dialog"),
        ("DOCUMENT_EVIDENCE", "document_evidence"),
        ("EXACT_FACT_EVIDENCE", "exact_fact_evidence"),
        ("SUPPORTING_MESSAGES", "supporting_messages"),
        ("SUPPORTING_MESSAGE", "supporting_message"),
        ("WORKING_MEMORY", "working_memory"),
        ("SESSION_SUMMARY", "session_summary"),
        ("SEMANTIC_FACTS", "retrieved_semantic"),
        ("EPISODIC_MEMORIES", "retrieved_episodic"),
        ("DOCUMENT_SNIPPETS", "retrieved_docs"),
        ("TASK_TOOL_STATE", "active_tool_state"),
        ("UNRESOLVED_ITEMS", "unresolved_items"),
    ]
    key_to_title = {key: title for title, key in order}
    included = {str(x).strip().upper() for x in list(included_titles or []) if str(x).strip()}
    isolation_rows = [
        dict(x)
        for x in list(_as_dict(context_isolation).get("dropped_memory_blocks") or [])
        if isinstance(x, dict)
    ]
    dropped_by_title: dict[str, dict[str, Any]] = {}
    for row in isolation_rows:
        block_key = str(row.get("block") or "").strip()
        title = key_to_title.get(block_key)
        if not title:
            continue
        dropped_by_title[title] = {
            "reason": str(row.get("reason") or ""),
            "preview": str(row.get("preview") or ""),
            "detected_category": str(row.get("detected_category") or ""),
        }

    out: list[dict[str, Any]] = []
    for title, key in order:
        if title in included:
            continue
        if key == "__active_task__":
            text = str(active_task_block or "").strip()
        else:
            text = str(_as_dict(memory_blocks).get(key) or "").strip()
        dropped = dict(dropped_by_title.get(title) or {})
        if not text and not dropped:
            continue
        row = {
            "block": title,
            "reason": str(dropped.get("reason") or "not_rendered"),
        }
        preview = str(dropped.get("preview") or "").strip() or _preview_text(text, 160)
        if preview:
            row["preview"] = preview
        detected_category = str(dropped.get("detected_category") or "").strip()
        if detected_category:
            row["detected_category"] = detected_category
        out.append(row)
    return out


def _is_empty_string_list_or_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not bool(value.strip())
    if isinstance(value, (list, tuple, set)):
        return not any(str(x).strip() for x in list(value or []))
    if isinstance(value, dict):
        return not bool(value)
    return False


def _to_bool(value, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return bool(default)
    if isinstance(value, (int, float)):
        return bool(value)
    src = str(value).strip().lower()
    if src in {"1", "true", "yes", "on", "y"}:
        return True
    if src in {"0", "false", "no", "off", "n", ""}:
        return False
    return bool(default)


def _to_float(value, default: float | None) -> float | None:
    if value is None:
        return default
    try:
        return float(value)
    except Exception:
        if default is None:
            return None
        return parse_time_to_epoch(value, default)


def _to_int(value, default: int | None) -> int | None:
    if value is None:
        return default
    try:
        return int(value)
    except Exception:
        return default


def _verbosity_to_max_tokens(level: float) -> int:
    value = max(0.0, min(1.0, float(level)))
    from config.settings import load_config
    cfg = load_config()
    lower = cfg.llm_max_tokens_lower_bound
    upper = cfg.llm_max_tokens_upper_bound
    range_val = max(0, upper - lower)
    return int(round(lower + (value * range_val)))


def _normalize_text(value) -> str:
    text = _repair_mojibake(str(value or ""))
    text = text.replace("\ufeff", "").replace("Р В РІР‚СљР В РІР‚РЋР В РІР‚в„ўР вЂ™Р’В»Р В РІР‚в„ўР РЋРІР‚вЂќ", "")
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\xa0", " ")
    text = re.sub(r"[\u200B-\u200F\u2060\uFEFF]", "", text)
    text = re.sub(r"[ \t\f\v]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _repair_mojibake(text: str) -> str:
    src = str(text or "")
    if not src:
        return ""
    if _contains_cyrillic(src):
        return src
    if "Р вЂњРЎвЂ™" not in src and "Р вЂњРІР‚В" not in src:
        return src
    try:
        repaired = src.encode("latin1", errors="ignore").decode("utf-8", errors="ignore")
    except Exception:
        return src
    return repaired if _contains_cyrillic(repaired) else src


def _contains_cyrillic(text: str) -> bool:
    return bool(re.search(r"[\u0400-\u04FF]", str(text or "")))


def _resolve_web_mode(meta: dict[str, Any], state: dict[str, Any]) -> str:
    raw = _pick(
        _as_dict(meta).get("web_mode"),
        _as_dict(state).get("web_mode"),
        "auto",
    ).lower()
    if raw in {"on", "off", "auto"}:
        return raw
    return "auto"


def _utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _is_web_memory_item(item: Any) -> bool:
    row = _as_dict(item)
    text = _normalize_text(_pick_value(row.get("text"), row.get("content"), "")).lower()
    source = str(_pick_value(row.get("source"), _as_dict(row.get("metadata")).get("source"), "") or "").strip().lower()
    topic = str(_pick_value(row.get("topic"), _as_dict(row.get("metadata")).get("topic"), "") or "").strip().lower()
    return bool(
        topic.startswith("web:")
        or source in {"web", "search", "internet"}
        or "source_url:" in text
        or text.startswith("[web]")
        or text.startswith("[web_search]")
    )


def _web_memory_domain(item: Any) -> str:
    row = _as_dict(item)
    direct = str(_pick_value(row.get("source_domain"), _as_dict(row.get("metadata")).get("source_domain"), "") or "").strip().lower()
    if direct:
        return direct
    source = str(_pick_value(row.get("source"), _as_dict(row.get("metadata")).get("source"), "") or "").strip().lower()
    if source and "." in source:
        return source
    text = _normalize_text(_pick_value(row.get("text"), row.get("content"), ""))
    m = re.search(r"source_domain:\s*([^\s]+)", text, flags=re.I)
    if m:
        return str(m.group(1) or "").strip().lower()
    return ""


def _is_forced_web_request(text: str) -> bool:
    src = str(text or "").strip().lower()
    return src == "/web" or src.startswith("/web ")


def _text_preview(value: str, max_chars: int = 120) -> str:
    src = _normalize_text(value).replace("\n", " ")
    if not src:
        return "-"
    n = max(12, int(max_chars or 120))
    if len(src) <= n:
        return src
    return src[: n - 3].rstrip() + "..."


def _loggable_user_input(value: Any, *, route: str) -> str:
    text = str(value or "")
    if str(route or "").strip().lower() != "command":
        return text
    cleaned = strip_service_command_prefix(text)
    return str(cleaned or "").strip()


def _turn_log_context(ctx: PipelineContext) -> dict[str, Any]:
    return {
        "trace_id": str(_pick(ctx.meta.get("trace_id"), ctx.state.get("conversation_id"), "-")).strip() or "-",
        "request_id": str(_pick(ctx.meta.get("request_id"), ctx.meta.get("trace_id"), "")).strip(),
        "turn_id": str(_pick(ctx.meta.get("turn_id"), ctx.state.get("turn_id"), "")).strip(),
        "conversation_id": str(_pick(ctx.meta.get("conversation_id"), ctx.state.get("conversation_id"), "")).strip(),
    }


def _update_topic_stack(
    stack: Any,
    *,
    thread_id: str,
    title: str,
    topic_key: str,
    limit: int = 8,
) -> list[dict[str, Any]]:
    target = str(thread_id or "").strip()
    if not target:
        return list(_as_list(stack))
    items: list[dict[str, Any]] = []
    seen: set[str] = set()
    now_value = now_local_ts()
    new_entry = {
        "thread_id": target,
        "title": str(title or "").strip(),
        "topic_key": str(topic_key or "").strip(),
        "updated_at": now_value,
    }
    for row in [new_entry, *list(_as_list(stack))]:
        item = _as_dict(row)
        item_thread_id = str(item.get("thread_id") or "").strip()
        if not item_thread_id or item_thread_id in seen:
            continue
        seen.add(item_thread_id)
        items.append(
            {
                "thread_id": item_thread_id,
                "title": str(item.get("title") or "").strip(),
                "topic_key": str(item.get("topic_key") or "").strip(),
                "updated_at": item.get("updated_at") or now_value,
            }
        )
        if len(items) >= limit:
            break
    return items


def _remember_turn_summary(ctx: PipelineContext, event: str, payload: dict[str, Any]) -> None:
    if not isinstance(ctx.meta, dict):
        return
    summaries = _as_dict(ctx.meta.get("turn_log_summaries"))
    summaries[str(event or "").strip()] = dict(payload or {})
    ctx.meta["turn_log_summaries"] = summaries


def _append_turn_warning(ctx: PipelineContext, warning: str) -> None:
    text = str(warning or "").strip()
    if not text or not isinstance(ctx.meta, dict):
        return
    warnings = [str(x).strip() for x in list(_as_list(ctx.meta.get("turn_log_warnings"))) if str(x).strip()]
    if text not in warnings:
        warnings.append(text)
    ctx.meta["turn_log_warnings"] = warnings


def _emit_turn_summary(ctx: PipelineContext, event: str, *, summary: str, **payload) -> None:
    context = _turn_log_context(ctx)
    compact = str(summary or "").strip()
    ctx.logs.append(
        "summary="
        f"{str(event or '').strip()} "
        f"trace={context.get('trace_id') or '-'} "
        f"request={context.get('request_id') or '-'} "
        f"turn={context.get('turn_id') or '-'} "
        f"conversation={context.get('conversation_id') or '-'} "
        f"{compact}"
    )
    row = dict(payload or {})
    _remember_turn_summary(ctx, str(event or "").strip(), row)
    log_json(LOGGER, str(event or "").strip(), summary=compact, context=context, **row)


def _queued_memory_ops_summary(ops: list[dict[str, Any]]) -> dict[str, int]:
    rows = [dict(x) for x in list(ops or []) if isinstance(x, dict)]
    counts: dict[str, int] = {
        "queued": int(len(rows)),
        "turn_user": 0,
        "turn_assistant": 0,
        "web_memory_write": 0,
        "web_memory_items": 0,
        "conversation_summary": 0,
        "prompt_stats": 0,
    }
    for row in rows:
        key = str(row.get("op") or "").strip().lower()
        if key in counts:
            counts[key] += 1
        if key == "web_memory_write":
            counts["web_memory_items"] += len([x for x in list(row.get("items") or []) if isinstance(x, dict)])
    return counts


def run_response_pipeline(text: str) -> str:
    return str(text or "").strip()
