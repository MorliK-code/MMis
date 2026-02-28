from __future__ import annotations

import json
import re
import time
import unicodedata
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from core.personality_engine import PersonalityEngine, apply_personality
from core.prompt_builder import PromptBuilder, PromptPack
from llm.provider_base import LLMProviderBase, LLMRequest, Message, ToolCall, ToolSpec
from metadata.metadata_extractor import MetadataExtractor
from prompt_engine import PromptEngine


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
        ctx.meta["metadata"] = metadata.to_dict()
        if not clean and ctx.route in {"chat", "command"}:
            ctx.text = "Empty request. Please send text."
            ctx.stop = True
        ctx.logs.append(
            "stage=preprocess "
            f"lang={ctx.tags.get('lang')} intent={ctx.tags.get('intent')} "
            f"mood={ctx.tags.get('mood')}"
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

    def __init__(self, personality_engine: PersonalityEngine | None = None):
        self._engine = personality_engine or PersonalityEngine()

    def run(self, ctx: PipelineContext) -> PipelineContext:
        if ctx.route != "chat":
            ctx.logs.append("stage=personality skipped(route)")
            return ctx

        active_id = str(ctx.state.get("active_personality_id") or "default").strip().lower() or "default"
        locked = bool(ctx.state.get("personality_locked", False))
        try:
            last_switch_ts = float(ctx.state.get("personality_last_switch_ts") or 0.0)
        except Exception:
            last_switch_ts = 0.0
        manual = _pick(
            ctx.meta.get("personality"),
            ctx.meta.get("persona"),
            "",
        )
        user_pref = _pick(
            ctx.meta.get("personality_preference"),
            _as_dict(ctx.state.get("traits")).get("preferred_personality"),
            "",
        )
        recent_context = {
            "mode": str(ctx.state.get("mode") or "chat"),
            "topic": str(ctx.tags.get("topic") or ""),
            "intent": str(ctx.tags.get("intent") or ""),
            "history_len": len(_as_list(ctx.state.get("history"))),
        }
        decision = self._engine.decide(
            intent=str(ctx.tags.get("intent") or "chat"),
            emotion=str(ctx.tags.get("mood") or "neutral"),
            recent_context=recent_context,
            user_pref=user_pref,
            active_personality_id=active_id,
            personality_locked=locked,
            last_switch_ts=last_switch_ts,
            manual_personality=manual,
        )

        target_id = str(decision.target_personality_id or active_id)
        if decision.switched:
            ctx.memory_ops.append(
                {
                    "op": "state_personality",
                    "value": target_id,
                    "locked": bool(decision.lock_after_switch),
                    "blend": dict(decision.blend or {}),
                    "ts": float(decision.ts or time.time()),
                    "reason": decision.reason,
                    "confidence": float(decision.confidence),
                }
            )
            ctx.state["personality_last_switch_ts"] = float(decision.ts or time.time())
            ctx.state["personality_blend"] = dict(decision.blend or {})
            if decision.lock_after_switch:
                ctx.state["personality_locked"] = True
        else:
            blend = _as_dict(ctx.state.get("personality_blend"))
            if blend and bool(blend.get("active")) and str(blend.get("to") or "").strip().lower() == target_id:
                next_blend = self._engine.advance_blend(blend)
                ctx.state["personality_blend"] = next_blend
                if next_blend != blend:
                    ctx.memory_ops.append({"op": "state_personality", "value": target_id, "blend": next_blend, "ts": time.time()})

        profile = self._engine.get_profile(target_id)
        ctx.state["active_personality_id"] = profile.id
        ctx.state["personality_profile"] = profile.to_dict()
        if not str(ctx.state.get("quality_profile") or "").strip() and profile.llm_profile:
            ctx.state["quality_profile"] = profile.llm_profile
        if profile.llm_profile:
            ctx.meta.setdefault("personality_llm_profile", profile.llm_profile)
            ctx.policies.setdefault("personality_llm_profile", profile.llm_profile)

        merged_traits = _as_dict(ctx.traits)
        merged_traits.setdefault("personality", profile.id)
        merged_traits.setdefault("profile", profile.id)
        merged_traits.setdefault("voice_style", profile.voice_style)
        for key, value in dict(profile.traits or {}).items():
            if key not in merged_traits:
                merged_traits[key] = value
        ctx.traits = merged_traits
        ctx.personality = decision.to_dict()
        ctx.logs.append(
            "stage=personality "
            f"active={active_id} target={profile.id} switched={decision.switched} "
            f"reason={decision.reason} conf={decision.confidence:.2f}"
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
        personality_engine: PersonalityEngine | None = None,
    ):
        self.provider = provider
        self.prompt_builder = prompt_builder
        self.personality_engine = personality_engine or PersonalityEngine()

    def run(self, ctx: PipelineContext) -> PipelineContext:
        if ctx.route == "system_event":
            self._handle_system_event(ctx)
            return ctx

        if ctx.route == "command":
            handled = self._handle_internal_command(ctx)
            if handled:
                return ctx

        req = self._build_request(ctx)
        resp = self.provider.generate(req)
        ctx.raw_output = str(resp.text or "")
        ctx.text = ctx.raw_output
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
        if cmd.startswith("/mode "):
            target = cmd.replace("/mode", "", 1).strip()
            if target:
                ctx.text = f"Mode switched to: {target}"
                ctx.memory_ops.append({"op": "state_mode", "value": target})
                ctx.logs.append(f"stage=generate command=mode:{target}")
                return True
        if cmd in {"/persona", "/persona current", "/personality"}:
            current = str(ctx.state.get("active_personality_id") or "default").strip().lower() or "default"
            locked = bool(ctx.state.get("personality_locked", False))
            ctx.text = f"Personality: {current} ({'locked' if locked else 'auto'})"
            ctx.logs.append("stage=generate command=persona:show")
            return True
        if cmd in {"/persona auto", "/personality auto"}:
            current = str(ctx.state.get("active_personality_id") or "default").strip().lower() or "default"
            ctx.text = "Personality auto-switch enabled."
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
                    "ts": time.time(),
                    "reason": "manual_auto",
                }
            )
            ctx.logs.append("stage=generate command=persona:auto")
            return True
        if cmd.startswith("/persona ") or cmd.startswith("/personality "):
            target = cmd.split(" ", 1)[1].strip().lower()
            if not target:
                return False
            known = set(self.personality_engine.list_ids())
            if target not in known:
                ctx.text = f"Unknown personality '{target}'. Available: {', '.join(sorted(known))}"
                ctx.logs.append(f"stage=generate command=persona:unknown:{target}")
                return True
            current = str(ctx.state.get("active_personality_id") or "default").strip().lower() or "default"
            blend = {
                "active": bool(current and current != target),
                "from": current,
                "to": target,
                "step": 1 if current and current != target else 0,
                "steps": 4,
                "old_weight": 0.75 if current and current != target else 0.0,
                "new_weight": 0.25 if current and current != target else 1.0,
            }
            ctx.text = f"Personality switched to: {target}"
            ctx.memory_ops.append(
                {
                    "op": "state_personality",
                    "value": target,
                    "locked": True,
                    "blend": blend,
                    "ts": time.time(),
                    "reason": "manual",
                    "confidence": 1.0,
                }
            )
            ctx.logs.append(f"stage=generate command=persona:{target}")
            return True
        return False

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
        req_metadata = _request_metadata(ctx)
        return LLMRequest(
            model=model,
            messages=messages,
            temperature=_to_float(_pick_value(ctx.meta.get("temperature"), ctx.policies.get("temperature"), None), None),
            top_p=_to_float(_pick_value(ctx.meta.get("top_p"), ctx.policies.get("top_p"), None), None),
            repeat_penalty=_to_float(_pick_value(ctx.meta.get("repeat_penalty"), ctx.policies.get("repeat_penalty"), None), None),
            seed=_to_int(_pick_value(ctx.meta.get("seed"), ctx.policies.get("seed"), None), None),
            max_tokens=_to_int(_pick_value(ctx.meta.get("max_tokens"), ctx.policies.get("max_tokens"), None), None),
            stop=[str(x) for x in _as_list(_pick_value(ctx.meta.get("stop"), ctx.policies.get("stop"), [])) if str(x)],
            json_mode=bool(_pick_value(ctx.meta.get("json_mode"), ctx.policies.get("json_mode"), False)),
            response_format=(dict(response_format) if isinstance(response_format, dict) else None),
            tools=tools,
            metadata=req_metadata,
        )


class PostprocessStage(PipelineStage):
    name = "postprocess"

    def run(self, ctx: PipelineContext) -> PipelineContext:
        text = _normalize_text(ctx.text)
        if not text:
            ctx.text = text
            ctx.logs.append("stage=postprocess empty")
            return ctx

        if not _looks_like_json(text):
            profile = str(ctx.traits.get("profile") or ctx.state.get("profile") or "default")
            text = apply_personality(text, profile=profile)
            text = _normalize_text(text)

            add_emoji = bool(_as_dict(ctx.policies.get("postprocess")).get("add_emoji", False))
            if add_emoji:
                mood = str(ctx.tags.get("mood") or "neutral")
                emoji = {"happy": " :)", "angry": " !!", "neutral": ""}.get(mood, "")
                if emoji and not text.endswith(emoji.strip()):
                    text = f"{text}{emoji}"

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
            personality_id = str(ctx.state.get("active_personality_id") or "").strip().lower()
            if personality_id:
                turn_tags["personality_id"] = personality_id
            ctx.memory_ops.append(
                {
                    "op": "turn_user",
                    "text": ctx.clean_user_msg,
                    "tags": turn_tags,
                    "ts": time.time(),
                }
            )
        if ctx.route in {"chat", "command"} and ctx.text:
            turn_tags = dict(ctx.tags)
            personality_id = str(ctx.state.get("active_personality_id") or "").strip().lower()
            if personality_id:
                turn_tags["personality_id"] = personality_id
            ctx.memory_ops.append(
                {
                    "op": "turn_assistant",
                    "text": str(ctx.text),
                    "tags": turn_tags,
                    "ts": time.time(),
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
                        "ts": time.time(),
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
        personality_engine: PersonalityEngine | None = None,
    ):
        self.provider = provider
        self.prompt_builder = prompt_builder or PromptBuilder()
        self.personality_engine = personality_engine or PersonalityEngine()
        self.prompt_engine = prompt_engine or PromptEngine(personality_engine=self.personality_engine)
        self._stages: dict[str, PipelineStage] = {
            "preprocess": PreprocessStage(metadata_extractor=metadata_extractor),
            "plan": PlanStage(),
            "personality": PersonalityStage(personality_engine=self.personality_engine),
            "memory_retrieve": MemoryRetrieveStage(memory_manager=memory_manager),
            "prompt_build": PromptBuildStage(prompt_builder=self.prompt_builder),
            "prompt_engine": PromptEngineStage(prompt_engine=self.prompt_engine),
            "generate": GenerateStage(
                provider=self.provider,
                prompt_builder=self.prompt_builder,
                personality_engine=self.personality_engine,
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
                "verify",
                "memory_write",
            ),
            PROFILE_BALANCED: (
                "preprocess",
                "plan",
                "personality",
                "memory_retrieve",
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
    personality_id = str(ctx.state.get("active_personality_id") or "").strip().lower()
    if personality_id:
        out["personality_id"] = personality_id
    for key in ("num_ctx", "num_thread", "num_gpu", "num_batch", "keep_alive", "think"):
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


def _to_float(value, default: float | None) -> float | None:
    if value is None:
        return default
    try:
        return float(value)
    except Exception:
        return default


def _to_int(value, default: int | None) -> int | None:
    if value is None:
        return default
    try:
        return int(value)
    except Exception:
        return default


def _normalize_text(value) -> str:
    text = str(value or "")
    text = text.replace("\ufeff", "").replace("ï»¿", "")
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\xa0", " ")
    text = re.sub(r"[\u200B-\u200F\u2060\uFEFF]", "", text)
    text = re.sub(r"[ \t\f\v]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def run_response_pipeline(text: str) -> str:
    return str(text or "").strip()
