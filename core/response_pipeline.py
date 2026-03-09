from __future__ import annotations

import json
import os
import re
import time
import unicodedata
import datetime as dt
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
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
from core.character_runtime import CharacterRuntime
from core.character_runtime import PromptPack
from core.mode_selector import ModeSelector, list_runtime_modes, normalize_mode_name
from llm.provider_base import LLMProviderBase, LLMRequest, Message, ToolCall, ToolSpec
from llm.tokenizer import estimate_tokens
from memory.memory_models import ContextBuildRequest, MemoryScope
from metadata.metadata_extractor import MetadataExtractor
from modules.character.evaluator import ResponseConstraintEvaluator
from modules.studio.studio_generator import StudioGenerator
from prompt_engine import PromptEngine
from utils.datetime_local import now_local_ts, parse_time_to_epoch
from utils.logger import get_logger, log_json
from core.web_rag_stage import WebRetrieveStage


PROFILE_FAST = "FAST"
PROFILE_BALANCED = "BALANCED"
PROFILE_QUALITY = "QUALITY"
PROFILE_ECONOM = "ECONOM"
PROFILE_ASYA = "ASYA"
PROFILE_AUTONOMOUS = "AUTONOMOUS"
WEB_TRACE_LOGGER = get_logger("web.trace")


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
            "intent": metadata.intent.label,
            "topic": topic or "",
            "intent_conf": float(metadata.intent.conf),
            "emotion_intensity": float(metadata.emotion.intensity),
            "metadata_tags": list(metadata.tags),
        }
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


class MemoryRetrieveStage(PipelineStage):
    name = "memory_retrieve"

    def __init__(self, memory_manager=None):
        self.memory_manager = memory_manager

    def run(self, ctx: PipelineContext) -> PipelineContext:
        if ctx.route not in {"chat", "command"}:
            ctx.logs.append("stage=memory_retrieve skipped(route)")
            return ctx
        query = str(ctx.clean_user_msg or ctx.user_msg or "").strip()
        if not query:
            ctx.logs.append("stage=memory_retrieve skipped(empty)")
            return ctx

        manager = ctx.meta.get("memory_manager") or self.memory_manager
        if manager is None or not hasattr(manager, "build_context"):
            ctx.logs.append("stage=memory_retrieve skipped(no_manager)")
            return ctx

        try:
            k = max(1, int(_pick_value(ctx.meta.get("memory_k"), ctx.policies.get("memory_k"), 8) or 8))
        except Exception:
            k = 8

        try:
            scope_names = list(_as_list(_pick_value(ctx.meta.get("memory_scopes"), ctx.policies.get("memory_scopes"), [])))
            scopes = [_scope_from_name(x) for x in scope_names]
            scopes = [x for x in scopes if x is not None]
            if not scopes:
                scopes = [
                    MemoryScope.CONVERSATION,
                    MemoryScope.SESSION,
                    MemoryScope.PROJECT,
                    MemoryScope.GLOBAL_USER,
                    MemoryScope.CHARACTER,
                    MemoryScope.TEMPORARY,
                ]
            context_request = ContextBuildRequest(
                system_prompt=str(_pick_value(ctx.state.get("system_prompt"), "")),
                user_message=query,
                namespace=str(
                    _pick_value(
                        ctx.meta.get("memory_namespace"),
                        ctx.meta.get("conversation_id"),
                        ctx.state.get("conversation_id"),
                        "default",
                    )
                ),
                scopes=list(scopes),
                top_k=max(1, int(k)),
                session_summary=str(_pick_value(ctx.state.get("dialog_summary"), "")),
                tool_state=_as_dict(ctx.state.get("last_tool_result")),
                unresolved_items=[str(x) for x in list(_as_list(ctx.state.get("open_questions"))) if str(x).strip()],
                context_budget_total=int(_pick_value(ctx.meta.get("context_budget_total"), 2200) or 2200),
                context_budget_memory=int(_pick_value(ctx.meta.get("context_budget_memory"), 700) or 700),
                context_budget_docs=int(_pick_value(ctx.meta.get("context_budget_docs"), 600) or 600),
                context_budget_tools=int(_pick_value(ctx.meta.get("context_budget_tools"), 220) or 220),
                context_budget_response_reserve=int(
                    _pick_value(ctx.meta.get("context_budget_response_reserve"), 260) or 260
                ),
            )
            result = manager.build_context(context_request)
            pack = result.to_dict() if hasattr(result, "to_dict") else dict(result or {})
        except Exception as exc:
            ctx.logs.append(f"stage=memory_retrieve error={type(exc).__name__}")
            return ctx

        if not isinstance(pack, dict):
            ctx.logs.append("stage=memory_retrieve skipped(invalid_pack)")
            return ctx

        ctx.memory_context = dict(pack)
        retrieved = list(_as_list(pack.get("selected")))
        if retrieved:
            ctx.retrieved_memories = retrieved

        blocks = _as_dict(pack.get("blocks"))
        session_summary = str(blocks.get("session_summary") or "").strip()
        if session_summary:
            ctx.state["dialog_summary"] = session_summary

        ctx.logs.append(
            f"stage=memory_retrieve retrieved={len(ctx.retrieved_memories)} tail={len(_as_list(ctx.state.get('history')))}"
        )
        return ctx


class PromptBuildStage(PipelineStage):
    name = "prompt_build"

    def __init__(self, character_runtime: CharacterRuntime):
        self.character_runtime = character_runtime

    def run(self, ctx: PipelineContext) -> PipelineContext:
        if ctx.route != "chat":
            ctx.logs.append("stage=prompt_build skipped(route)")
            return ctx
        prompt_state = dict(ctx.state or {})
        merged_tags = _as_dict(prompt_state.get("context_tags"))
        merged_tags.update(dict(ctx.tags or {}))
        prompt_state["context_tags"] = merged_tags
        if ctx.plan:
            prompt_state["plan"] = ctx.plan
        if ctx.memory_context:
            prompt_state.setdefault("memory_context", dict(ctx.memory_context))
            blocks = _as_dict(ctx.memory_context.get("blocks"))
            prompt_state.setdefault(
                "long_summary",
                str(_pick_value(blocks.get("session_summary"), blocks.get("working_memory"), "")),
            )
        if bool(ctx.meta.get("think", False)):
            rules = ctx.policies.get("rules")
            if not isinstance(rules, list):
                rules = [] if rules is None else [rules]
            rules.append(
                "Р•СЃР»Рё РІРєР»СЋС‡С‘РЅ thinking mode вЂ” РїРёС€Рё РІРЅСѓС‚СЂРµРЅРЅРёРµ СЂР°СЃСЃСѓР¶РґРµРЅРёСЏ РўРћР›Р¬РљРћ РІРЅСѓС‚СЂРё С‚РµРіРѕРІ <think>...</think> "
                "Рё С„РёРЅР°Р»СЊРЅС‹Р№ РѕС‚РІРµС‚ СЃРЅР°СЂСѓР¶Рё. РќРµ СѓРїРѕРјРёРЅР°Р№ СЌС‚Рё С‚РµРіРё РїРѕР»СЊР·РѕРІР°С‚РµР»СЋ."
            )
            ctx.policies["rules"] = rules

        web_intent = str(ctx.tags.get("web_query_intent") or "").strip().lower()
        web_used = str(ctx.tags.get("web_used") or "").strip().lower() == "true"
        web_fresh_missing = str(ctx.tags.get("web_fresh_missing") or "").strip().lower() == "true"
        web_response_style = str(ctx.tags.get("web_response_style") or "").strip().lower()
        retrieved_for_prompt = list(_as_list(ctx.retrieved_memories))

        if web_used and web_intent in {"fx_rate", "weather", "news_release"}:
            filtered, dropped = _filter_retrieved_memories_for_time_sensitive_web(retrieved_for_prompt)
            retrieved_for_prompt = filtered
            if dropped > 0:
                ctx.logs.append(
                    f"stage=prompt_build web_memory_filter=applied intent={web_intent} dropped={dropped} kept={len(filtered)}"
                )
            _append_policy_rule(
                ctx.policies,
                "For time-sensitive web answers, respond directly with facts and avoid rhetorical openers like 'Ах, ты опять...' or similar chatter.",
            )
            _append_policy_rule(
                ctx.policies,
                "Use fetched web evidence and include source domain and fetch/publish time; if sources disagree, report a range.",
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
        elif web_used and web_intent in {"fx_rate", "weather"}:
            _append_policy_rule(
                ctx.policies,
                "When answering FX/weather requests, rely on fetched web evidence, include source domain and timestamp, and report a range if sources disagree.",
            )
            ctx.tags["web_guardrail"] = "cite_source_and_time"
            tags_map = _as_dict(prompt_state.get("context_tags"))
            tags_map["web_guardrail"] = "cite_source_and_time"
            prompt_state["context_tags"] = tags_map
        if web_response_style:
            tags_map = _as_dict(prompt_state.get("context_tags"))
            tags_map["web_response_style"] = web_response_style
            prompt_state["context_tags"] = tags_map
        ctx.prompt_pack = self.character_runtime.build(
            state=prompt_state,
            user_msg=ctx.clean_user_msg,
            retrieved_memories=retrieved_for_prompt,
            traits=ctx.traits,
            policies=ctx.policies,
        )
        emitter = ctx.meta.get("emit_web_trace_event")
        if callable(emitter):
            web_rows = [x for x in list(retrieved_for_prompt) if _is_web_memory_item(x)]
            domains = sorted(
                {
                    _web_memory_domain(x)
                    for x in list(web_rows)
                    if _web_memory_domain(x)
                }
            )
            emitter(
                "web_context_injected",
                {
                    "retrieved_total": len(list(retrieved_for_prompt)),
                    "web_context_count": len(list(web_rows)),
                    "domains": domains,
                    "truncated_candidates": sum(
                        1
                        for x in list(web_rows)
                        if str(_pick_value(_as_dict(x).get("text"), _as_dict(x).get("content"), "")).rstrip().endswith("...")
                    ),
                },
            )
        ctx.logs.append("stage=prompt_build")
        return ctx


class PromptEngineStage(PipelineStage):
    name = "prompt_engine"

    def __init__(self, prompt_engine: PromptEngine):
        self.prompt_engine = prompt_engine

    def run(self, ctx: PipelineContext) -> PipelineContext:
        if ctx.route != "chat":
            ctx.logs.append("stage=prompt_engine skipped(route)")
            return ctx
        if ctx.prompt_pack is None:
            ctx.logs.append("stage=prompt_engine skipped(no_pack)")
            return ctx

        result = self.prompt_engine.compose(
            prompt_pack=ctx.prompt_pack,
            state=ctx.state,
            traits=ctx.traits,
            policies=ctx.policies,
        )
        ctx.prompt_messages = list(result.messages)
        ctx.prompt_sections = dict(result.sections)
        ctx.logs.append(f"stage=prompt_engine messages={len(ctx.prompt_messages)}")
        return ctx


class GenerateStage(PipelineStage):
    name = "generate"

    def __init__(
        self,
        provider: LLMProviderBase,
        character_runtime: CharacterRuntime,
        studio_generator: StudioGenerator | None = None,
    ):
        self.provider = provider
        self.character_runtime = character_runtime
        self.character_engine = character_runtime
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

        req = self._build_request(ctx)
        stream_answer_cb = ctx.meta.get("stream_on_answer_chunk")
        stream_thinking_cb = ctx.meta.get("stream_on_thinking_chunk")
        use_stream = callable(stream_answer_cb) or callable(stream_thinking_cb)

        if use_stream:
            stream_result = self._generate_stream(ctx, req, on_answer=stream_answer_cb, on_thinking=stream_thinking_cb)
            if stream_result is not None:
                text_out, thinking_out, model_name = stream_result
                ctx.raw_output = text_out
                ctx.text = text_out
                ctx.thinking = thinking_out
                ctx.stats = {
                    "served_model": str(model_name or req.model or ""),
                    "answer_ms": 0.0,
                    "prompt_eval_count": 0,
                    "eval_count": 0,
                    "total_tokens": 0,
                    "streaming": True,
                }
                ctx.logs.append(f"stage=generate route={ctx.route} model={ctx.stats.get('served_model')} stream=1")
                return ctx

        resp = self.provider.generate(req)
        ctx.raw_output = str(resp.text or "")
        ctx.text = ctx.raw_output
        ctx.thinking = str(getattr(resp, "thinking", "") or "")
        if resp.tool_calls:
            ctx.tool_calls = [_tool_call_to_dict(x) for x in list(resp.tool_calls or [])]
        ctx.stats = {
            "served_model": str(resp.model or req.model or ""),
            "answer_ms": float(resp.timings.latency_ms or 0.0),
            "prompt_eval_count": int(resp.usage.prompt_tokens or 0),
            "eval_count": int(resp.usage.completion_tokens or 0),
            "total_tokens": int(resp.usage.total_tokens or 0),
        }
        ctx.logs.append(f"stage=generate route={ctx.route} model={ctx.stats.get('served_model')}")
        return ctx

    def _generate_stream(self, ctx: PipelineContext, req: LLMRequest, *, on_answer, on_thinking) -> tuple[str, str, str] | None:
        stream = getattr(self.provider, "stream", None)
        if not callable(stream):
            return None
        parser = _ThinkStreamParser()
        answer_parts: list[str] = []
        thinking_parts: list[str] = []
        model_name = str(req.model or "")

        try:
            for chunk in stream(req):
                # 1) thinking РёР· РїСЂРѕРІР°Р№РґРµСЂР° (Ollama РѕС‚РґР°С‘С‚ thinking_delta РѕС‚РґРµР»СЊРЅРѕ)
                thinking_delta = str(getattr(chunk, "thinking_delta", "") or "")
                if thinking_delta:
                    thinking_parts.append(thinking_delta)
                    if callable(on_thinking):
                        try:
                            on_thinking(thinking_delta)
                        except Exception:
                            pass

                # 2) РѕР±С‹С‡РЅС‹Р№ С‚РµРєСЃС‚
                text_delta = str(getattr(chunk, "text_delta", "") or "")
                if text_delta:
                    # РЅР° РІСЃСЏРєРёР№ СЃР»СѓС‡Р°Р№ С‚Р°РєР¶Рµ РїРѕРґРґРµСЂР¶РёРІР°РµРј <think>...</think> РІ СЃР°РјРѕРј С‚РµРєСЃС‚Рµ
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
        except Exception:
            return None

        return ("".join(answer_parts).strip(), "".join(thinking_parts).strip(), model_name)

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
            raw_parts = [x for x in str(ctx.clean_user_msg or "").strip().split(" ") if x]
            arg = str(raw_parts[1] or "").strip().lower() if len(raw_parts) >= 2 else ""
            if arg and arg not in {"balanced", "aggressive", "status"}:
                ctx.text = "Usage: /web-auto [balanced|aggressive|status]"
                ctx.logs.append(f"stage=generate command=web-auto:usage arg={arg}")
                return True

            if arg == "status":
                if scope == "chat":
                    mode_value = str(ctx.state.get("web_mode") or "auto").strip().lower()
                    profile_value = str(ctx.state.get("web_auto_profile") or "balanced").strip().lower()
                else:
                    mode_value = str(self._scope_get(ctx, scope, "web_mode", "auto") or "auto").strip().lower()
                    profile_value = str(self._scope_get(ctx, scope, "web_auto_profile", "balanced") or "balanced").strip().lower()
                if mode_value not in {"on", "off", "auto"}:
                    mode_value = "auto"
                if profile_value not in {"balanced", "aggressive"}:
                    profile_value = "balanced"
                ctx.text = (
                    "Web auto status:\n"
                    f"- scope: {scope}\n"
                    f"- web_mode: {mode_value}\n"
                    f"- web_auto_profile: {profile_value}"
                )
                ctx.logs.append(f"stage=generate command=web-auto:status scope={scope}")
                return True

            profile_set = arg if arg in {"balanced", "aggressive"} else ""
            if scope == "chat":
                ctx.memory_ops.append({"op": "state_web_mode", "value": "auto"})
                if profile_set:
                    ctx.memory_ops.append({"op": "state_web_auto_profile", "value": profile_set})
            else:
                updates: dict[str, Any] = {"web_mode": "auto"}
                if profile_set:
                    updates["web_auto_profile"] = profile_set
                self._scope_set(ctx, scope, **updates)

            if profile_set:
                ctx.text = f"Web mode set to auto ({profile_set})" + (f" for scope: {scope}." if scope != "chat" else ".")
            else:
                ctx.text = "Web mode set to auto." if scope == "chat" else f"Web mode set to auto for scope: {scope}."
            ctx.ui_actions.append({"type": "set_web_mode", "mode": "auto"})
            ctx.logs.append(
                "stage=generate command=web-auto mode=auto "
                f"profile={profile_set or '-'} scope={scope}"
            )
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
        scoped_web_mode = str(self._scope_get(ctx, scope, "web_mode", "") or "").strip().lower()
        if scoped_web_mode in {"on", "off", "auto"}:
            ctx.meta["web_mode"] = scoped_web_mode
        scoped_web_auto_profile = str(self._scope_get(ctx, scope, "web_auto_profile", "") or "").strip().lower()
        if scoped_web_auto_profile in {"balanced", "aggressive"}:
            ctx.meta["web_auto_profile"] = scoped_web_auto_profile
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
                ctx.prompt_pack = self.character_runtime.build(
                    state=prompt_state,
                    user_msg=ctx.clean_user_msg,
                    retrieved_memories=ctx.retrieved_memories,
                    traits=ctx.traits,
                    policies=ctx.policies,
                )
                messages = _messages_from_prompt_pack(ctx.prompt_pack)
        else:
            messages = [
                Message(role="system", content="You are MMis assistant. Reply concisely."),
                Message(role="user", content=ctx.clean_user_msg),
            ]

        model = _pick(
            ctx.meta.get("model"),
            ctx.state.get("model"),
            ctx.policies.get("model"),
        )
        tools = _parse_tools(_pick_value(ctx.meta.get("tools"), ctx.policies.get("tools"), ctx.state.get("tools")))
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

            web_intent = str(ctx.tags.get("web_query_intent") or "").strip().lower()
            web_style = str(ctx.tags.get("web_response_style") or "").strip().lower()
            web_fixed = _apply_time_sensitive_web_failsafe(
                text,
                web_intent=web_intent,
                web_response_style=web_style,
            )
            if web_fixed != text:
                text = web_fixed
                ctx.logs.append(f"stage=postprocess web_failsafe=applied intent={web_intent}")

            if _should_apply_echo_guard(ctx.clean_user_msg) and _looks_like_echo_response(answer=text, user_msg=ctx.clean_user_msg):
                text = _echo_fallback_text(ctx.clean_user_msg)
                ctx.logs.append("stage=postprocess echo_guard=applied")

            add_emoji = bool(_as_dict(ctx.policies.get("postprocess")).get("add_emoji", False))
            if add_emoji:
                mood = str(ctx.tags.get("mood") or "neutral")
                emoji = {"happy": " :)", "angry": " !!", "neutral": ""}.get(mood, "")
                if emoji and not text.endswith(emoji.strip()):
                    text = f"{text}{emoji}"

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
            "web_trace_id": str(_pick(ctx.stats.get("web_trace_id"), ctx.meta.get("web_trace_id"), "")),
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
        request = LLMRequest(
            model=model,
            messages=[
                Message(
                    role="system",
                    content=(
                        "Summarize assistant response in 1-2 concise sentences. "
                        "Keep key action points. No bullet list."
                    ),
                ),
                Message(
                    role="user",
                    content=f"Response:\n{text}\n\nShort summary:",
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
        summary = _normalize_text(str(response.text or ""))
        if not summary:
            return ""
        if _looks_like_json(summary):
            return ""
        return _squeeze_summary(summary)


class WebSecondPassStage(PipelineStage):
    name = "web_second_pass"

    def __init__(self, *, stages: dict[str, PipelineStage] | None = None):
        self._stages = stages if isinstance(stages, dict) else {}

    def run(self, ctx: PipelineContext) -> PipelineContext:
        if ctx.route != "chat":
            ctx.logs.append("stage=web_second_pass skipped(route)")
            return ctx
        web_mode = _resolve_web_mode(ctx.meta, ctx.state)
        web_auto_profile = _resolve_web_auto_profile(ctx.meta, ctx.state)
        if web_mode != "auto":
            ctx.logs.append("stage=web_second_pass skipped(mode)")
            return ctx
        if web_auto_profile != "aggressive":
            ctx.logs.append("stage=web_second_pass skipped(profile)")
            return ctx
        if bool(ctx.meta.get("web_second_pass_done", False)):
            ctx.logs.append("stage=web_second_pass skipped(done)")
            return ctx
        if str(ctx.tags.get("web_used") or "").strip().lower() == "true":
            ctx.logs.append("stage=web_second_pass skipped(web_already_used)")
            return ctx
        if not _should_trigger_web_second_pass(ctx):
            ctx.logs.append("stage=web_second_pass skipped(confident)")
            return ctx

        ctx.meta["web_second_pass_done"] = True
        web_stage = self._stages.get("web_retrieve")
        if web_stage is None:
            ctx.logs.append("stage=web_second_pass skipped(no_web_stage)")
            return ctx

        prev_use_web = ctx.meta.get("use_web", None)
        prev_second_pass = ctx.meta.get("web_second_pass", None)
        ctx.meta["use_web"] = True
        ctx.meta["web_second_pass"] = True
        try:
            ctx = web_stage.run(ctx)
        except Exception as exc:
            ctx.errors.append(f"web_second_pass:web_retrieve:{type(exc).__name__}")
            ctx.logs.append(f"stage=web_second_pass web_retrieve_error={type(exc).__name__}")
            return ctx
        finally:
            if prev_use_web is None:
                ctx.meta.pop("use_web", None)
            else:
                ctx.meta["use_web"] = prev_use_web
            if prev_second_pass is None:
                ctx.meta.pop("web_second_pass", None)
            else:
                ctx.meta["web_second_pass"] = prev_second_pass

        if str(ctx.tags.get("web_used") or "").strip().lower() != "true":
            ctx.logs.append("stage=web_second_pass skipped(no_web_after_retry)")
            return ctx

        rerun = ("prompt_build", "prompt_engine", "generate", "verify", "postprocess", "output_format")
        for stage_name in rerun:
            stage = self._stages.get(stage_name)
            if stage is None:
                continue
            try:
                ctx = stage.run(ctx)
            except Exception as exc:
                ctx.errors.append(f"web_second_pass:{stage_name}:{type(exc).__name__}")
                ctx.logs.append(f"stage=web_second_pass rerun_stage={stage_name} error={type(exc).__name__}")
                break
            if ctx.stop:
                break
        ctx.logs.append("stage=web_second_pass applied")
        return ctx


class MemoryWriteStage(PipelineStage):
    name = "memory_write"

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

        if ctx.route in {"chat", "command"} and ctx.clean_user_msg:
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
            ctx.memory_ops.append(
                {
                    "op": "turn_user",
                    "text": ctx.clean_user_msg,
                    "tags": turn_tags,
                    "ts": now_local_ts(),
                }
            )
        if ctx.route in {"chat", "command"} and ctx.text:
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
            ctx.memory_ops.append(
                {
                    "op": "turn_assistant",
                    "text": str(ctx.text),
                    "thinking": str(ctx.thinking or ""),
                    "tags": turn_tags,
                    "ts": now_local_ts(),
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
            long_summary = str(ctx.prompt_pack.blocks.get("long_summary") or "").strip()
            if long_summary and long_summary != "- none":
                ctx.memory_ops.append(
                    {
                        "op": "conversation_summary",
                        "text": long_summary,
                        "ts": now_local_ts(),
                    }
                )
        ctx.logs.append(f"stage=memory_write ops={len(ctx.memory_ops)}")
        return ctx


class ResponsePipeline:
    def __init__(
        self,
        provider: LLMProviderBase,
        character_runtime: CharacterRuntime | None = None,
        metadata_extractor: MetadataExtractor | None = None,
        prompt_engine: PromptEngine | None = None,
        memory_manager=None,
        studio_generator: StudioGenerator | None = None,
    ):
        self.provider = provider
        self.character_engine = character_runtime or CharacterRuntime()
        self.character_runtime = self.character_engine
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
            "memory_retrieve": MemoryRetrieveStage(memory_manager=memory_manager),
            "web_retrieve": WebRetrieveStage(),
            "prompt_build": PromptBuildStage(character_runtime=self.character_runtime),
            "prompt_engine": PromptEngineStage(prompt_engine=self.prompt_engine),
            "generate": GenerateStage(
                provider=self.provider,
                character_runtime=self.character_runtime,
                studio_generator=studio_generator,
            ),
            "postprocess": PostprocessStage(),
            "tool_router": ToolRouterStage(),
            "verify": VerifyStage(),
            "output_format": OutputFormatStage(provider=self.provider),
            "memory_write": MemoryWriteStage(),
        }
        self._stages["web_second_pass"] = WebSecondPassStage(stages=self._stages)
        self._profiles: dict[str, tuple[str, ...]] = {
            PROFILE_FAST: (
                "preprocess",
                "mode_select",
                "personality",
                "memory_retrieve",
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
                "memory_retrieve",
                "web_retrieve",
                "prompt_build",
                "prompt_engine",
                "generate",
                "postprocess",
                "tool_router",
                "verify",
                "output_format",
                "web_second_pass",
                "memory_write",
            ),
            PROFILE_QUALITY: (
                "preprocess",
                "mode_select",
                "plan",
                "personality",
                "memory_retrieve",
                "web_retrieve",
                "prompt_build",
                "prompt_engine",
                "generate",
                "tool_router",
                "verify",
                "postprocess",
                "output_format",
                "web_second_pass",
                "memory_write",
            ),
            PROFILE_AUTONOMOUS: (
                "preprocess",
                "generate",
                "postprocess",
                "output_format",
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
        web_trace_id = _resolve_web_trace_id(ctx.meta, ctx.user_msg)
        ctx.meta["web_trace_id"] = web_trace_id
        web_mode = _resolve_web_mode(ctx.meta, ctx.state)
        web_auto_profile = _resolve_web_auto_profile(ctx.meta, ctx.state)
        ctx.meta.setdefault("web_mode", web_mode)
        ctx.meta.setdefault("web_auto_profile", web_auto_profile)
        force_web = _is_forced_web_request(ctx.user_msg)
        input_preview = _text_preview(ctx.user_msg, 120)

        stage_profile = self._resolve_stage_profile(ctx.profile)
        ctx.meta["stage_profile"] = stage_profile
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
            f"trace={web_trace_id} route={ctx.route} web_mode={web_mode} "
            f"web_auto_profile={web_auto_profile} "
            f"force={int(bool(force_web))} input_len={len(str(ctx.user_msg or '').strip())} "
            f"input_preview={input_preview}"
        )
        self._emit_web_trace_event(
            ctx,
            event="web_trace_start",
            payload={
                "force_web": bool(force_web),
                "input_len": len(str(ctx.user_msg or "").strip()),
                "input_preview": input_preview,
                "active_profile": str(ctx.profile or ""),
                "mode": str(web_mode or "auto"),
                "auto_profile": str(web_auto_profile or "balanced"),
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
                ctx.errors.append(f"{name}:{exc}")
                ctx.logs.append(f"stage={name} error={type(exc).__name__}")
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
        ctx.logs.append(
            "stage=web_trace end "
            f"trace={web_trace_id} route={ctx.route} web_mode={web_mode} web_used={web_used} "
            f"web_auto_profile={web_auto_profile} "
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

        ctx.stats["web_trace_id"] = str(web_trace_id)
        meta_payload = _as_dict(ctx.structured_output.get("meta"))
        meta_payload["web_trace_id"] = str(web_trace_id)
        ctx.structured_output["meta"] = meta_payload

        return PipelineResult(
            text=str(ctx.text or ""),
            thinking=str(ctx.thinking or ""),
            structured_output=dict(ctx.structured_output or {}),
            tool_calls=list(ctx.tool_calls or []),
            memory_ops=list(ctx.memory_ops or []),
            ui_actions=list(ctx.ui_actions or []),
            logs=list(ctx.logs or []),
            stats=dict(ctx.stats or {}),
        )

    def _emit_web_trace_event(self, ctx: PipelineContext, *, event: str, payload: dict[str, Any] | None = None) -> None:
        web_trace_id = str(ctx.meta.get("web_trace_id") or "").strip() or "-"
        row = {
            "trace_id": web_trace_id,
            "ts": _utc_now_iso(),
            "conversation_id": str(_pick(ctx.meta.get("conversation_id"), ctx.state.get("conversation_id"), "")),
            "turn_id": str(_pick(ctx.meta.get("turn_id"), ctx.state.get("turn_id"), "")),
            "route": str(ctx.route or ""),
            "profile": str(ctx.profile or ""),
            "stage_profile": str(ctx.meta.get("stage_profile") or ""),
            "web_mode": str(ctx.meta.get("web_mode") or ctx.state.get("web_mode") or "auto"),
            "web_auto_profile": str(
                ctx.meta.get("web_auto_profile") or ctx.state.get("web_auto_profile") or "balanced"
            ),
            "web_query_intent": str(_pick(ctx.meta.get("web_query_intent"), ctx.tags.get("web_query_intent"), "generic")),
            "web_fresh_required": _to_bool(_pick(ctx.meta.get("web_fresh_required"), ctx.tags.get("web_fresh_required"), False), default=False),
            "web_fresh_missing": _to_bool(_pick(ctx.meta.get("web_fresh_missing"), ctx.tags.get("web_fresh_missing"), False), default=False),
            "payload": dict(payload or {}),
        }
        log_json(WEB_TRACE_LOGGER, str(event or "").strip(), **row)

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
        self._buf = ""
        self._in_think = False
        self._open_tags = ("<think>", "<thinking>", "<reasoning>")
        self._close_tags = ("</think>", "</thinking>", "</reasoning>")

    def feed(self, chunk: str) -> tuple[str, str]:
        self._buf += str(chunk or "")
        return self._drain()

    def flush(self) -> tuple[str, str]:
        if not self._buf:
            return "", ""
        if self._in_think:
            out = ("", self._buf)
        else:
            out = (self._buf, "")
        self._buf = ""
        return out

    def _drain(self) -> tuple[str, str]:
        visible_parts: list[str] = []
        thinking_parts: list[str] = []

        while self._buf:
            if self._in_think:
                close_idx, close_tag = self._find_first(self._close_tags)
                if close_idx < 0:
                    keep = max(len(x) for x in self._close_tags) - 1
                    if len(self._buf) > keep:
                        thinking_parts.append(self._buf[:-keep])
                        self._buf = self._buf[-keep:]
                    break
                if close_idx > 0:
                    thinking_parts.append(self._buf[:close_idx])
                self._buf = self._buf[close_idx + len(close_tag) :]
                self._in_think = False
                continue

            open_idx, open_tag = self._find_first(self._open_tags)
            if open_idx < 0:
                keep = max(len(x) for x in self._open_tags) - 1
                if len(self._buf) > keep:
                    visible_parts.append(self._buf[:-keep])
                    self._buf = self._buf[-keep:]
                break
            if open_idx > 0:
                visible_parts.append(self._buf[:open_idx])
            self._buf = self._buf[open_idx + len(open_tag) :]
            self._in_think = True

        return "".join(visible_parts), "".join(thinking_parts)

    def _find_first(self, tags: tuple[str, ...]) -> tuple[int, str]:
        src = self._buf.lower()
        best_idx = -1
        best_tag = ""
        for tag in tags:
            idx = src.find(tag)
            if idx < 0:
                continue
            if best_idx < 0 or idx < best_idx:
                best_idx = idx
                best_tag = tag
        return best_idx, best_tag


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
            )
        )
    return out


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
    ):
        if key in meta:
            out[key] = meta.get(key)
    return out


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
_WEB_STYLE_REPLACEMENTS = (
    re.compile(r"^\s*ах,\s*ты\s+(?:опять|снова)\b[^.!?\n]*[.!?]\s*", flags=re.IGNORECASE),
    re.compile(r"^\s*ах,\s*ты\b[^.!?\n]*[.!?]\s*", flags=re.IGNORECASE),
    re.compile(r"^\s*(?:ах|ой|ну)\b[^.!?\n]{0,180}[.!?]\s*", flags=re.IGNORECASE),
)


def _enforce_response_hygiene(text: str) -> str:
    src = _normalize_text(text)
    if not src:
        return ""
    cleaned = _strip_service_markers(src)
    cleaned = _dedupe_adjacent_blocks(cleaned)
    return _normalize_text(cleaned)


def _filter_retrieved_memories_for_time_sensitive_web(items: list[Any]) -> tuple[list[Any], int]:
    if not items:
        return [], 0
    kept: list[Any] = []
    dropped = 0
    for item in list(items):
        row = item if isinstance(item, dict) else {}
        text = _normalize_text(_pick_value(row.get("text"), row.get("content"), ""))
        source = str(_pick_value(row.get("source"), _as_dict(row.get("metadata")).get("source"), "") or "").strip().lower()
        topic = str(_pick_value(row.get("topic"), _as_dict(row.get("metadata")).get("topic"), "") or "").strip().lower()
        low = text.lower()
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
        if not is_web_evidence and (is_noisy_block or (is_chat_like_source and is_noisy_chatter)):
            dropped += 1
            continue
        kept.append(item)
    return kept, dropped


def _apply_time_sensitive_web_failsafe(
    text: str,
    *,
    web_intent: str = "",
    web_response_style: str = "",
) -> str:
    src = _normalize_text(text)
    if not src:
        return ""
    intent = str(web_intent or "").strip().lower()
    style = str(web_response_style or "").strip().lower()
    if intent not in {"fx_rate", "weather", "news_release"}:
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


def _scope_from_name(value) -> MemoryScope | None:
    raw = str(value or "").strip().lower()
    if not raw:
        return None
    for scope in MemoryScope:
        if raw == scope.value:
            return scope
    return None


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


def _resolve_web_auto_profile(meta: dict[str, Any], state: dict[str, Any]) -> str:
    raw = str(
        _pick(
            _as_dict(meta).get("web_auto_profile"),
            _as_dict(state).get("web_auto_profile"),
            "balanced",
        )
        or "balanced"
    ).strip().lower()
    return raw if raw in {"balanced", "aggressive"} else "balanced"


def _has_precision_markers(text: str) -> bool:
    low = str(text or "").strip().lower()
    if not low:
        return False
    markers = (
        "найди",
        "точн",
        "проверь",
        "актуал",
        "источник",
        "ссылка",
        "рецепт",
        "latest",
        "exact",
        "verify",
        "source",
    )
    return any(token in low for token in markers)


def _should_trigger_web_second_pass(ctx: PipelineContext) -> bool:
    user_text = str(ctx.clean_user_msg or ctx.user_msg or "").strip()
    answer = str(ctx.text or "").strip()
    if not user_text:
        return False
    intent = str(ctx.tags.get("intent") or "").strip().lower()
    smalltalk_re = re.compile(
        r"\b(\u043a\u0430\u043a \u0434\u0435\u043b\u0430|\u0447\u0442\u043e \u043d\u043e\u0432\u043e\u0433\u043e|\u043f\u0440\u0438\u0432\u0435\u0442|hello|hi|how are you)\b",
        flags=re.I,
    )
    if bool(smalltalk_re.search(user_text)):
        return False
    if not answer:
        return True
    answer_low = answer.lower()
    uncertain_markers = (
        "не уверен",
        "не знаю",
        "не могу",
        "не удалось",
        "нет данных",
        "может быть",
        "possibly",
        "probably",
        "not sure",
        "cannot verify",
        "can't verify",
    )
    if any(token in answer_low for token in uncertain_markers):
        return True
    question_like = ("?" in user_text) or (intent in {"question", "implementation", "action_request"})
    if _has_precision_markers(user_text) and len(answer) < 280:
        return True
    if question_like and len(answer) < 140:
        return True
    return False


def _resolve_web_trace_id(meta: dict[str, Any], user_msg: str) -> str:
    source = _pick(_as_dict(meta).get("web_trace_id"), _as_dict(meta).get("trace_id"))
    if source:
        return source
    ts = int(time.time() * 1000)
    msg_len = len(str(user_msg or "").strip())
    return f"web-{ts}-{msg_len}"


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


def run_response_pipeline(text: str) -> str:
    return str(text or "").strip()
