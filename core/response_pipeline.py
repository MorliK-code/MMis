from __future__ import annotations

import json
import os
import re
import time
import unicodedata
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from config.settings import load_config
from modules.character.dialog_policies import (
    compute_dialog_flags as dialog_compute_dialog_flags,
    extract_term_directive as dialog_extract_term_directive,
    local_date_kyiv as dialog_local_date_kyiv,
    local_region_name as dialog_local_region_name,
    trim_leading_greeting as dialog_trim_leading_greeting,
)
from core.prompt_builder import PromptBuilder, PromptPack
from llm.provider_base import LLMProviderBase, LLMRequest, Message, ToolCall, ToolSpec
from metadata.metadata_extractor import MetadataExtractor
from modules.character.engine import CharacterEngine
from modules.character.evaluator import ResponseConstraintEvaluator
from prompt_engine import PromptEngine
from utils.datetime_local import now_local_ts, parse_time_to_epoch
from core.web_rag_stage import WebRetrieveStage


PROFILE_FAST = "FAST"
PROFILE_BALANCED = "BALANCED"
PROFILE_QUALITY = "QUALITY"


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


class PlanStage(PipelineStage):
    name = "plan"

    def run(self, ctx: PipelineContext) -> PipelineContext:
        mode = str(ctx.state.get("mode") or "chat")
        goal = _pick(
            ctx.state.get("active_goal"),
            ctx.state.get("current_task"),
            ctx.state.get("task"),
        )
        intent = str(ctx.tags.get("intent") or "chat")
        if mode == "task" and goal:
            ctx.plan = f"goal={goal}; intent={intent}; style=step-by-step"
        elif intent in {"question", "implementation", "action_request"}:
            ctx.plan = f"intent={intent}; provide concise actionable answer"
        else:
            ctx.plan = f"intent={intent}; maintain conversational flow"
        ctx.logs.append("stage=plan")
        return ctx


class PersonalityStage(PipelineStage):
    name = "personality"

    def __init__(self, character_engine: CharacterEngine | None = None):
        self._characters = character_engine or CharacterEngine()

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

        update = self._characters.update(
            text=str(ctx.clean_user_msg or ctx.user_msg or ""),
            meta={
                "intent": str(ctx.tags.get("intent") or ""),
                "mood": str(ctx.tags.get("mood") or ""),
                "mode": str(ctx.state.get("mode") or "chat"),
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
        ctx.state["character_prompt_block"] = str(update.prompt_block or "")
        ctx.state["mood"] = str(update.mood or "")
        if update.llm_profile:
            if not str(ctx.state.get("quality_profile") or "").strip():
                ctx.state["quality_profile"] = update.llm_profile
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
        if manager is None or not hasattr(manager, "build_context_pack"):
            ctx.logs.append("stage=memory_retrieve skipped(no_manager)")
            return ctx

        try:
            k = max(1, int(_pick_value(ctx.meta.get("memory_k"), ctx.policies.get("memory_k"), 8) or 8))
        except Exception:
            k = 8
        try:
            tail_n = max(1, int(_pick_value(ctx.meta.get("memory_tail_n"), ctx.policies.get("memory_tail_n"), 10) or 10))
        except Exception:
            tail_n = 10
        filters = _pick_value(ctx.meta.get("memory_filters"), ctx.policies.get("memory_filters"), None)
        filters = dict(filters) if isinstance(filters, dict) else None

        try:
            pack = manager.build_context_pack(query=query, k=k, filters=filters, tail_n=tail_n)
        except Exception as exc:
            ctx.logs.append(f"stage=memory_retrieve error={type(exc).__name__}")
            return ctx

        if not isinstance(pack, dict):
            ctx.logs.append("stage=memory_retrieve skipped(invalid_pack)")
            return ctx

        ctx.memory_context = dict(pack)
        retrieved = list(_as_list(pack.get("retrieved")))
        if retrieved:
            ctx.retrieved_memories = retrieved

        tail = list(_as_list(pack.get("tail")))
        if tail:
            history = list(_as_list(ctx.state.get("history")))
            merged = history if history else tail
            if history and bool(ctx.meta.get("merge_memory_tail", True)):
                merged = (history + tail)[-max(len(history), tail_n) :]
            ctx.state["history"] = merged

        short_summary = str(pack.get("short_summary") or "").strip()
        if short_summary and not str(ctx.state.get("dialog_summary") or "").strip():
            ctx.state["dialog_summary"] = short_summary

        profile_summary = pack.get("profile_summary")
        if isinstance(profile_summary, dict):
            ctx.state["profile_summary"] = dict(profile_summary)

        ctx.logs.append(
            f"stage=memory_retrieve retrieved={len(ctx.retrieved_memories)} tail={len(_as_list(ctx.state.get('history')))}"
        )
        return ctx


class PromptBuildStage(PipelineStage):
    name = "prompt_build"

    def __init__(self, prompt_builder: PromptBuilder):
        self.prompt_builder = prompt_builder

    def run(self, ctx: PipelineContext) -> PipelineContext:
        if ctx.route != "chat":
            ctx.logs.append("stage=prompt_build skipped(route)")
            return ctx
        prompt_state = dict(ctx.state or {})
        prompt_state.setdefault("context_tags", dict(ctx.tags))
        if ctx.plan:
            prompt_state["plan"] = ctx.plan
        if ctx.memory_context:
            prompt_state.setdefault("memory_context", dict(ctx.memory_context))
            prompt_state.setdefault("long_summary", str(ctx.memory_context.get("short_summary") or ""))
        ctx.prompt_pack = self.prompt_builder.build(
            state=prompt_state,
            user_msg=ctx.clean_user_msg,
            retrieved_memories=ctx.retrieved_memories,
            traits=ctx.traits,
            policies=ctx.policies,
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
        prompt_builder: PromptBuilder,
        character_engine: CharacterEngine | None = None,
    ):
        self.provider = provider
        self.prompt_builder = prompt_builder
        self.character_engine = character_engine or CharacterEngine()

    def run(self, ctx: PipelineContext) -> PipelineContext:
        if ctx.route == "system_event":
            self._handle_system_event(ctx)
            return ctx

        if ctx.route == "command":
            handled = self._handle_internal_command(ctx)
            if handled:
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
                text_delta = str(getattr(chunk, "text_delta", "") or "")
                if not text_delta:
                    continue
                visible, thinking = parser.feed(text_delta)
                if thinking:
                    thinking_parts.append(thinking)
                    if callable(on_thinking):
                        try:
                            on_thinking(thinking)
                        except Exception:
                            pass
                if visible:
                    answer_parts.append(visible)
                    if callable(on_answer):
                        try:
                            on_answer(visible)
                        except Exception:
                            pass
            tail_visible, tail_thinking = parser.flush()
            if tail_visible:
                answer_parts.append(tail_visible)
                if callable(on_answer):
                    try:
                        on_answer(tail_visible)
                    except Exception:
                        pass
            if tail_thinking:
                thinking_parts.append(tail_thinking)
                if callable(on_thinking):
                    try:
                        on_thinking(tail_thinking)
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
            ctx.text = "Thinking mode enabled."
            ctx.memory_ops.append({"op": "state_think", "value": True})
            ctx.ui_actions.append({"type": "toggle_think", "enabled": True})
            ctx.logs.append("stage=generate command=think")
            return True
    
        if cmd == "/nothink":
            ctx.text = "Thinking mode disabled."
            ctx.memory_ops.append({"op": "state_think", "value": False})
            ctx.ui_actions.append({"type": "toggle_think", "enabled": False})
            ctx.logs.append("stage=generate command=nothink")
            return True
        
        if cmd == "/web":
            ctx.text = "Web mode enabled."
            ctx.memory_ops.append({"op": "state_web_mode", "value": "on"})
            ctx.ui_actions.append({"type": "set_web_mode", "mode": "on"})
            ctx.logs.append("stage=generate command=web mode=on")
        return True
        
        if cmd == "/no-web":
            ctx.text = "Web mode disabled."
            ctx.memory_ops.append({"op": "state_web_mode", "value": "off"})
            ctx.ui_actions.append({"type": "set_web_mode", "mode": "off"})
            ctx.logs.append("stage=generate command=no-web mode=off")
        return True

        if cmd in {"/web-auto", "/web_auto", "/auto-web"}:
            ctx.text = "Web mode set to auto."
            ctx.memory_ops.append({"op": "state_web_mode", "value": "auto"})
            ctx.ui_actions.append({"type": "set_web_mode", "mode": "auto"})
            ctx.logs.append("stage=generate command=web-auto mode=auto")
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
        
        if cmd.startswith("/mode "):
            target = cmd.replace("/mode", "", 1).strip()
            if target:
                ctx.text = f"Mode switched to: {target}"
                ctx.memory_ops.append({"op": "state_mode", "value": target})
                ctx.logs.append(f"stage=generate command=mode:{target}")
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

    def _build_request(self, ctx: PipelineContext) -> LLMRequest:
        if ctx.route == "chat":
            if ctx.prompt_messages:
                messages = list(ctx.prompt_messages)
            else:
                prompt_state = dict(ctx.state or {})
                prompt_state.setdefault("context_tags", dict(ctx.tags))
                if ctx.plan:
                    prompt_state["plan"] = ctx.plan
                ctx.prompt_pack = self.prompt_builder.build(
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


class MemoryWriteStage(PipelineStage):
    name = "memory_write"

    def run(self, ctx: PipelineContext) -> PipelineContext:
        store_turn = bool(ctx.meta.get("store_turn", True))
        if not store_turn:
            ctx.logs.append("stage=memory_write skipped")
            return ctx

        if ctx.route in {"chat", "command"} and ctx.clean_user_msg:
            turn_tags = dict(ctx.tags)
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
        prompt_builder: PromptBuilder | None = None,
        metadata_extractor: MetadataExtractor | None = None,
        prompt_engine: PromptEngine | None = None,
        memory_manager=None,
        character_engine: CharacterEngine | None = None,
    ):
        self.provider = provider
        self.prompt_builder = prompt_builder or PromptBuilder()
        self.character_engine = character_engine or CharacterEngine()
        self.prompt_engine = prompt_engine or PromptEngine(
            character_engine=self.character_engine,
        )
        self._stages: dict[str, PipelineStage] = {
            "preprocess": PreprocessStage(metadata_extractor=metadata_extractor),
            "plan": PlanStage(),
            "personality": PersonalityStage(
                character_engine=self.character_engine,
            ),
            "memory_retrieve": MemoryRetrieveStage(memory_manager=memory_manager),
            "web_retrieve": WebRetrieveStage(),
            "prompt_build": PromptBuildStage(prompt_builder=self.prompt_builder),
            "prompt_engine": PromptEngineStage(prompt_engine=self.prompt_engine),
            "generate": GenerateStage(
                provider=self.provider,
                prompt_builder=self.prompt_builder,
                character_engine=self.character_engine,
            ),
            "postprocess": PostprocessStage(),
            "tool_router": ToolRouterStage(),
            "verify": VerifyStage(),
            "memory_write": MemoryWriteStage(),
        }
        self._profiles: dict[str, tuple[str, ...]] = {
            PROFILE_FAST: (
                "preprocess",
                "personality",
                "memory_retrieve",
                "prompt_build",
                "prompt_engine",
                "generate",
                "postprocess",
                "verify",
                "memory_write",
            ),
            PROFILE_BALANCED: (
                "preprocess",
                "plan",
                "personality",
                "memory_retrieve",
                "web_retrive",
                "prompt_build",
                "prompt_engine",
                "generate",
                "postprocess",
                "tool_router",
                "verify",
                "memory_write",
            ),
            PROFILE_QUALITY: (
                "preprocess",
                "plan",
                "personality",
                "memory_retrieve",
                "web_retrive",
                "prompt_build",
                "prompt_engine",
                "generate",
                "tool_router",
                "verify",
                "postprocess",
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
        stage_names = self._resolve_stage_names(ctx.profile, ctx.meta, ctx.policies)
        ctx.logs.append(f"profile={ctx.profile}")
        ctx.logs.append(f"stages={','.join(stage_names)}")

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
                    ctx.text = "I got stuck during generation. Please try again."
                    ctx.stop = True
            if ctx.stop:
                break

        return PipelineResult(
            text=str(ctx.text or ""),
            tool_calls=list(ctx.tool_calls or []),
            memory_ops=list(ctx.memory_ops or []),
            ui_actions=list(ctx.ui_actions or []),
            logs=list(ctx.logs or []),
            stats=dict(ctx.stats or {}),
        )

    def _resolve_profile(self, *, meta, state, policies) -> str:
        preferred = _pick(
            _as_dict(meta).get("profile"),
            _as_dict(meta).get("quality_profile"),
            _as_dict(state).get("quality_profile"),
            _as_dict(policies).get("profile"),
            _as_dict(policies).get("quality_profile"),
            PROFILE_BALANCED,
        )
        norm = str(preferred or PROFILE_BALANCED).strip().upper()
        return norm if norm in self._profiles else PROFILE_BALANCED

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
_THINKING_HEADER_RE = re.compile(r"^\s*\[thinking\](?:\s*[РІР‚вЂќ-]\s*)?$", flags=re.IGNORECASE)


def _enforce_response_hygiene(text: str) -> str:
    src = _normalize_text(text)
    if not src:
        return ""
    cleaned = _strip_service_markers(src)
    cleaned = _dedupe_adjacent_blocks(cleaned)
    return _normalize_text(cleaned)


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
    src = re.sub(r"[\"'`Р’В«Р’В»РІР‚С›РІР‚СљРІР‚Сњ]", "", src)
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
        terms_list=_as_list(_pick_value(policy_cfg.get("terms_list"), ["РјРёР»Р°С€РєР°"])),
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
        terms = ["РјРёР»Р°С€РєР°"]
    patterns = [str(x or "").strip().lower() for x in list(disable_patterns or []) if str(x or "").strip()]
    if not patterns:
        patterns = ["РЅРµ РЅР°Р·С‹РІР°Р№", "РїСЂРµРєСЂР°С‚Рё РЅР°Р·С‹РІР°С‚СЊ", "РЅРµ Р·РѕРІРё", "РїСЂРµРєСЂР°С‚Рё Р·РІР°С‚СЊ"]

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


_CHECKIN_RE = re.compile(
    r"(РєР°Рє\s+(?:Сѓ\s+С‚РµР±СЏ\s+)?РґРµР»Р°|РєР°Рє\s+С‚С‹|РєР°Рє\s+СЃР°Рј|С‡С‚Рѕ\s+РЅРѕРІРѕРіРѕ|РєР°Рє\s+РЅР°СЃС‚СЂРѕРµРЅРёРµ|how\s+are\s+you)",
    flags=re.IGNORECASE,
)
_QUESTION_START_RE = re.compile(
    r"^\s*(РєР°Рє|С‡С‚Рѕ|РїРѕС‡РµРјСѓ|Р·Р°С‡РµРј|РєРѕРіРґР°|РіРґРµ|РєС‚Рѕ|С‡РµРј|РєР°РєРѕР№|РєР°РєР°СЏ|РєР°РєРёРµ|СЃРєРѕР»СЊРєРѕ|how|what|why|where|when)\b",
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
        return "РЈ РјРµРЅСЏ РІСЃРµ РЅРѕСЂРјР°Р»СЊРЅРѕ, СЃРїР°СЃРёР±Рѕ. РљР°Рє С‚С‹?"
    return "РџРѕРЅСЏР»Р°. РЇ РЅР° СЃРІСЏР·Рё Рё РіРѕС‚РѕРІР° РїРѕРјРѕС‡СЊ. РЈС‚РѕС‡РЅРё, С‡С‚Рѕ РёРјРµРЅРЅРѕ РЅСѓР¶РЅРѕ."


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
    # Keep practical bounds for chat responses.
    return int(round(180 + (value * 900)))


def _normalize_text(value) -> str:
    text = _repair_mojibake(str(value or ""))
    text = text.replace("\ufeff", "").replace("Р“Р‡Р’В»Р’С—", "")
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
    if "Гђ" not in src and "Г‘" not in src:
        return src
    try:
        repaired = src.encode("latin1", errors="ignore").decode("utf-8", errors="ignore")
    except Exception:
        return src
    return repaired if _contains_cyrillic(repaired) else src


def _contains_cyrillic(text: str) -> bool:
    return bool(re.search(r"[\u0400-\u04FF]", str(text or "")))


def run_response_pipeline(text: str) -> str:
    return str(text or "").strip()



