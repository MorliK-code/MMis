from __future__ import annotations

import hashlib
import re
import time
from dataclasses import dataclass, field
from threading import RLock
from typing import Any

from config.settings import load_config
from core.character_runtime import CharacterRuntime
from core.response_pipeline import PipelineResult, ResponsePipeline
from llm import build_provider
from llm.provider_base import LLMProviderBase
from memory.memory_manager import MemoryManager
from memory.memory_models import MemoryEvent, MemoryScope, MemoryType
from memory.text_sanitizer import (
    clean_assistant_text_for_memory,
    contains_memory_service_sections,
    sanitize_assistant_memory_text,
)
from metadata.metadata_extractor import MetadataExtractor
from utils.datetime_local import parse_time_to_epoch
from utils.logger import append_human_log, get_logger, log_json


LOGGER = get_logger(__name__)


@dataclass(frozen=True)
class BrainResult:
    text: str
    route: str
    thinking: str = ""
    structured_output: dict[str, Any] = field(default_factory=dict)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    memory_ops: list[dict[str, Any]] = field(default_factory=list)
    ui_actions: list[dict[str, Any]] = field(default_factory=list)
    logs: list[str] = field(default_factory=list)
    status: str = "ok"
    deduped: bool = False
    throttled: bool = False
    error: str = ""
    stats: dict[str, Any] = field(default_factory=dict)


class Brain:
    """Unified orchestrator for user messages and system events."""

    def __init__(
        self,
        *,
        provider: LLMProviderBase | None = None,
        state_manager: CharacterRuntime | None = None,
        memory_manager: MemoryManager | None = None,
        metadata_extractor: MetadataExtractor | None = None,
        response_pipeline: ResponsePipeline | None = None,
        dedup_window_sec: float = 1.6,
        throttle_sec: float = 0.12,
    ):
        if provider is not None:
            self._provider = provider
        else:
            cfg = load_config()
            self._provider = build_provider(cfg.llm_default_provider, default_model=cfg.model_name)
        self.state_manager = state_manager or CharacterRuntime()
        self.memory_manager = memory_manager or MemoryManager()
        self.metadata_extractor = metadata_extractor or MetadataExtractor(cache_size=280)
        self.pipeline = response_pipeline or ResponsePipeline(
            self._provider,
            character_runtime=self.state_manager,
            memory_manager=self.memory_manager,
        )

        self._lock = RLock()
        self._dedup_window_sec = max(0.05, float(dedup_window_sec))
        self._throttle_sec = max(0.0, float(throttle_sec))
        self._recent_events: dict[str, float] = {}
        self._recent_results: dict[str, BrainResult] = {}
        self._last_source_ts: dict[str, float] = {}

    @property
    def provider(self) -> LLMProviderBase:
        return self._provider

    # Compatibility alias for previous integrations.
    @property
    def service(self) -> LLMProviderBase:
        return self._provider

    def handle_message(self, user_msg, meta=None) -> BrainResult:
        now = time.monotonic()
        meta_map = _as_dict(meta)
        route, text, event_name = self._route_input(user_msg, meta_map)
        source = str(meta_map.get("source") or meta_map.get("sender") or route).strip().lower()
        signature = self._signature(route=route, text=text, event_name=event_name, source=source)

        with self._lock:
            if route != "command":
                duplicate_result = self._check_duplicate(signature, now)
                if duplicate_result is not None:
                    return duplicate_result
                if self._is_throttled(source, now):
                    result = BrainResult(
                        text="",
                        route=route,
                        throttled=True,
                        logs=[f"throttled source={source}"],
                        ui_actions=[{"type": "noop", "reason": "throttled"}],
                    )
                    self._remember(signature, now, result)
                    return result

        state_snapshot = self.state_manager.snapshot()
        conversation_id = str(meta_map.get("conversation_id") or "").strip()
        if not conversation_id:
            conversation_id = str(getattr(state_snapshot, "conversation_id", "") or "").strip()
        if not conversation_id:
            conversation_id = str(_as_dict(getattr(state_snapshot, "raw", {})).get("conversation_id") or "").strip()
        conversation_id = conversation_id.lower() or "default"
        studio_active = False
        if route == "chat" and hasattr(self.pipeline, "is_studio_active"):
            try:
                studio_active = bool(self.pipeline.is_studio_active(conversation_id=conversation_id, state=state_snapshot.raw))
            except Exception:
                studio_active = False
        non_persistent_turn = bool(route == "command" or (route == "chat" and studio_active))

        self._sync_active_character_manifest(state_snapshot.raw)
        track_state = bool(meta_map.get("track_state", True))
        state_updates_enabled = bool(track_state and (route == "system_event" or not non_persistent_turn))
        if state_updates_enabled and route in {"chat", "command"} and text:
            self.state_manager.update_on_user_message(
                text,
                {
                    "conversation_id": conversation_id,
                    "mode": meta_map.get("mode") or state_snapshot.mode,
                    "quality_profile": meta_map.get("quality_profile") or state_snapshot.quality_profile,
                    "active_goal": meta_map.get("active_goal") or state_snapshot.active_goal,
                },
            )
            state_snapshot = self.state_manager.snapshot()
        elif state_updates_enabled and route == "system_event":
            self.state_manager.update_on_event(event_name, meta_map)
            state_snapshot = self.state_manager.snapshot()

        state_map = dict(state_snapshot.raw)
        state_map.setdefault("mode", state_snapshot.mode)
        state_map.setdefault("conversation_id", state_snapshot.conversation_id)
        state_map.setdefault("turn_id", state_snapshot.turn_id)
        state_map.setdefault("quality_profile", state_snapshot.quality_profile)
        state_map.setdefault("active_character_id", state_snapshot.active_character_id)
        state_map.setdefault("active_goal", state_snapshot.active_goal)
        state_map.setdefault("dialog_summary", state_snapshot.dialog_summary)
        state_map.setdefault("active_tasks", state_snapshot.active_tasks)
        state_map.setdefault("history", state_snapshot.history)
        state_map.setdefault("context_tags", state_snapshot.context_tags)
        state_map.setdefault("active_mode", state_snapshot.active_mode)
        state_map.setdefault("mode_lock", state_snapshot.mode_lock)
        state_map.setdefault("mode_until", state_snapshot.mode_until)
        state_map.setdefault("last_signals", state_snapshot.last_signals)
        state_map.setdefault("last_actions", state_snapshot.last_actions)
        state_map.setdefault("web_mode", state_snapshot.web_mode)
        state_map.setdefault("thinking_enabled", state_snapshot.thinking_enabled)
        state_map.setdefault("output_format", state_snapshot.output_format)
        character_quality_profile = self._character_quality_profile(state_snapshot=state_snapshot, state_map=state_map)
        if character_quality_profile:
            state_map["quality_profile"] = character_quality_profile
            state_map["personality_llm_profile"] = character_quality_profile

        retrieved_memories = meta_map.get("retrieved_memories", state_snapshot.retrieved_memories)
        traits = meta_map.get("traits", state_snapshot.traits)
        policies = meta_map.get("policies", state_snapshot.policies)
        meta_for_pipeline = dict(meta_map)
        meta_for_pipeline.setdefault("source", source or route)
        meta_for_pipeline.setdefault("memory_manager", self.memory_manager)
        meta_for_pipeline.setdefault("conversation_id", state_snapshot.conversation_id)
        meta_for_pipeline.setdefault("turn_id", state_snapshot.turn_id)
        if character_quality_profile:
            meta_for_pipeline.setdefault("personality_llm_profile", character_quality_profile)
        meta_for_pipeline.setdefault("quality_profile", character_quality_profile or state_snapshot.quality_profile)
        meta_for_pipeline.setdefault("active_mode", state_snapshot.active_mode)
        meta_for_pipeline.setdefault("mode_lock", state_snapshot.mode_lock)
        meta_for_pipeline.setdefault("output_format", state_snapshot.output_format)
        meta_for_pipeline.setdefault("track_state", track_state)
        meta_for_pipeline.setdefault("non_persistent_turn", non_persistent_turn)
        
        cfg = load_config()
        meta_for_pipeline.setdefault("safety_mode", cfg.safety_mode)

        try:
            pipeline_result = self.pipeline.run(
                route=route,
                user_msg=(event_name if route == "system_event" else text),
                state=state_map,
                meta=meta_for_pipeline,
                retrieved_memories=retrieved_memories,
                traits=traits,
                policies=policies,
            )
            result = self._to_brain_result(route=route, pipeline_result=pipeline_result)
            memory_apply_summary = self._apply_memory_ops(result.memory_ops)
            self._update_state_after_success(
                route=route,
                event_name=event_name,
                meta=meta_for_pipeline,
                result=result,
                track_state=state_updates_enabled,
            )
            persisted_summary = self._persist_turns(
                route=route,
                user_text=text,
                result=result,
                meta=meta_for_pipeline,
                non_persistent_turn=non_persistent_turn,
            )
            self._append_turn_summaries(
                result=result,
                route=route,
                user_text=text,
                meta=meta_for_pipeline,
                memory_apply_summary=memory_apply_summary,
                persisted_summary=persisted_summary,
            )
        except Exception as exc:
            result = self._build_error_result(route=route, error=exc)

        with self._lock:
            self._remember(signature, now, result)
        return result

    def _route_input(self, user_msg, meta: dict[str, Any]) -> tuple[str, str, str]:
        event_name = ""
        if isinstance(user_msg, dict):
            event_name = str(
                user_msg.get("event")
                or user_msg.get("event_type")
                or meta.get("event")
                or meta.get("event_type")
                or ""
            ).strip()
            text = str(user_msg.get("text") or user_msg.get("message") or "").strip()
        else:
            text = str(user_msg or "").strip()
            event_name = str(meta.get("event") or meta.get("event_type") or "").strip()

        if event_name:
            return "system_event", text, event_name

        mtype = str(meta.get("type") or meta.get("kind") or "").strip().lower()
        if mtype in {"event", "system_event", "ui_event", "voice_event"}:
            ev = str(meta.get("name") or meta.get("event") or text or "event").strip()
            return "system_event", text, ev

        command_candidate = _normalize_command_candidate(text)
        if command_candidate.startswith("/"):
            return "command", command_candidate, ""
        return "chat", text, ""

    def _to_brain_result(self, *, route: str, pipeline_result: PipelineResult) -> BrainResult:
        return BrainResult(
            text=str(pipeline_result.text or ""),
            route=route,
            thinking=str(pipeline_result.thinking or ""),
            structured_output=dict(pipeline_result.structured_output or {}),
            tool_calls=list(pipeline_result.tool_calls or []),
            memory_ops=list(pipeline_result.memory_ops or []),
            ui_actions=list(pipeline_result.ui_actions or []),
            logs=list(pipeline_result.logs or []),
            stats=dict(pipeline_result.stats or {}),
            status="ok",
        )

    def _apply_memory_ops(self, ops: list[dict[str, Any]]) -> dict[str, Any]:
        summary = self._empty_memory_write_summary()
        for op in list(ops or []):
            if not isinstance(op, dict):
                continue
            key = str(op.get("op") or "").strip().lower()
            if key == "state_mode":
                value = str(op.get("value") or "").strip()
                if value:
                    self.state_manager.set_mode(value)
            elif key == "state_mode_lock":
                self.state_manager.set_mode_lock(bool(op.get("value", False)))
            elif key == "state_output_format":
                value = op.get("value")
                if isinstance(value, dict):
                    show_parameters = value.get("show_parameters")
                    show_summary = value.get("show_summary")
                    self.state_manager.set_output_format(
                        show_parameters=(bool(show_parameters) if isinstance(show_parameters, bool) else None),
                        show_summary=(bool(show_summary) if isinstance(show_summary, bool) else None),
                    )
            elif key == "state_think":
                self.state_manager.patch({"thinking_enabled": bool(op.get("value", False))})
            elif key == "state_web_mode":
                value = str(op.get("value") or "").strip().lower()
                if value in {"auto", "on", "off"}:
                    self.state_manager.patch({"web_mode": value})
            elif key == "state_scoped_settings":
                value = op.get("value")
                if isinstance(value, dict):
                    self.state_manager.patch({"command_scopes": dict(value)})
            elif key == "state_patch":
                value = op.get("value")
                if isinstance(value, dict) and value:
                    self.state_manager.patch(dict(value))
            elif key == "state_personality":
                value = str(op.get("value") or "").strip().lower()
                if value:
                    locked = op.get("locked")
                    blend = op.get("blend") if isinstance(op.get("blend"), dict) else None
                    ts = op.get("ts")
                    ts_val = parse_time_to_epoch(ts, 0.0) if ts is not None else None
                    if ts_val is None or float(ts_val) <= 0:
                        ts_val = None
                    try:
                        self.state_manager.set_active_personality(
                            value,
                            locked=(bool(locked) if isinstance(locked, bool) else None),
                            blend=blend,
                            switch_ts=ts_val,
                        )
                    except Exception:
                        pass
            elif key == "state_character":
                value = str(op.get("value") or "").strip().lower()
                if value:
                    locked = op.get("locked")
                    ts = op.get("ts")
                    ts_val = parse_time_to_epoch(ts, 0.0) if ts is not None else None
                    if ts_val is None or float(ts_val) <= 0:
                        ts_val = None
                    try:
                        self.state_manager.set_active_character(
                            value,
                            locked=(bool(locked) if isinstance(locked, bool) else None),
                            switch_ts=ts_val,
                        )
                    except Exception:
                        pass
                    try:
                        character_engine = getattr(self.pipeline, "character_engine", None)
                        if character_engine is not None and hasattr(character_engine, "set_active_character"):
                            character_engine.set_active_character(value)
                    except Exception:
                        pass
            elif key == "tool_results":
                items = list(op.get("items") or [])
                if items:
                    self.state_manager.set_last_tool_result(items[-1])
                    try:
                        self.memory_manager.set_private_runtime_state(
                            key="last_tool_result",
                            value=items[-1],
                            namespace=str(self.state_manager.get("conversation_id") or "default"),
                        )
                    except Exception:
                        pass
            elif key == "conversation_summary":
                text = str(op.get("text") or "").strip()
                if text:
                    self.state_manager.set_dialog_summary(text)
                    try:
                        ingest_result = self.memory_manager.ingest_event(
                            MemoryEvent(
                                role="system",
                                text=text,
                                namespace=str(self.state_manager.get("conversation_id") or "default"),
                                scope=MemoryScope.SESSION,
                                memory_type=MemoryType.SUMMARY,
                                metadata={
                                    "source": "rolling_summary",
                                    "importance": 0.6,
                                    "confidence": 0.7,
                                    "trace_id": str(op.get("trace_id") or ""),
                                    "request_id": str(op.get("request_id") or ""),
                                    "turn_id": op.get("turn_id"),
                                    "conversation_id": str(
                                        op.get("conversation_id")
                                        or self.state_manager.get("conversation_id")
                                        or "default"
                                    ),
                                },
                            )
                        )
                        self._capture_memory_ingest(
                            summary,
                            ingest_result,
                            bucket="conversation_summary",
                        )
                    except Exception:
                        self._append_summary_warning(summary, "conversation_summary_failed")
            elif key == "web_memory_write":
                if self.memory_manager is None:
                    continue
                default_namespace = str(
                    op.get("namespace")
                    or self.state_manager.get("conversation_id")
                    or "default"
                ).strip() or "default"
                context = self._log_context_from_meta(op, conversation_id=default_namespace)
                items = [x for x in list(op.get("items") or []) if isinstance(x, dict)]
                written = 0
                summary["planned_web_memory_items"] += int(len(items))
                for row in list(items):
                    text = str(row.get("text") or "").strip()
                    if not text:
                        continue
                    scope = _memory_scope_from_name(row.get("scope"), default=MemoryScope.PROJECT)
                    memory_type = _memory_type_from_name(row.get("memory_type"), default=MemoryType.SEMANTIC)
                    namespace = str(row.get("namespace") or default_namespace).strip() or default_namespace
                    metadata = dict(_as_dict(row.get("metadata")) or {})
                    if not str(metadata.get("source") or "").strip():
                        metadata["source"] = "web_v2"
                    write_type = str(row.get("write_type") or "").strip().lower()
                    if write_type:
                        metadata.setdefault("web_v2_write_type", write_type)
                    metadata.setdefault("trace_id", str(context.get("trace_id") or ""))
                    metadata.setdefault("request_id", str(context.get("request_id") or ""))
                    metadata.setdefault("turn_id", context.get("turn_id"))
                    metadata.setdefault("conversation_id", str(context.get("conversation_id") or namespace))
                    confidence = row.get("confidence")
                    importance = row.get("importance")
                    ttl_sec = row.get("ttl_sec")
                    if confidence is not None:
                        try:
                            metadata["confidence"] = max(0.0, min(1.0, float(confidence)))
                        except Exception:
                            pass
                    if importance is not None:
                        try:
                            metadata["importance"] = max(0.0, min(1.0, float(importance)))
                        except Exception:
                            pass
                    if ttl_sec is not None:
                        try:
                            metadata["ttl_sec"] = max(30, int(ttl_sec))
                        except Exception:
                            pass
                    try:
                        self.memory_manager.ingest_event(
                            MemoryEvent(
                                role="system",
                                text=text,
                                namespace=namespace,
                                scope=scope,
                                memory_type=memory_type,
                                metadata=metadata,
                            )
                        )
                        written += 1
                        self._capture_memory_ingest(summary, ingest_result, bucket="web_memory_write")
                    except Exception:
                        self._append_summary_warning(summary, "web_memory_write_failed")
                        continue
                if written > 0:
                    log_json(
                        LOGGER,
                        "web_memory_write_applied",
                        summary=f"written={written} namespace={default_namespace}",
                        context=context,
                        namespace=default_namespace,
                        written=written,
                    )
            elif key == "state_address_terms":
                value = op.get("value")
                if isinstance(value, dict):
                    self.state_manager.set_address_terms(value)
            elif key in {"turn_user", "turn_assistant"}:
                tags = dict(op.get("tags") or {})
                if tags:
                    ctx = dict(self.state_manager.get("context_tags", {}) or {})
                    if tags.get("lang"):
                        ctx["lang"] = str(tags.get("lang"))
                    if tags.get("intent"):
                        ctx["intent"] = str(tags.get("intent"))
                        ctx["last_intent"] = str(tags.get("intent"))
                    if tags.get("mood"):
                        ctx["mood"] = str(tags.get("mood"))
                        ctx["last_mood"] = str(tags.get("mood"))
                    if tags.get("topic"):
                        ctx["topic"] = str(tags.get("topic"))
                    self.state_manager.patch({"context_tags": ctx})
        return summary

    def _sync_active_character_manifest(self, state_map: dict[str, Any]) -> None:
        active = str((state_map or {}).get("active_character_id") or "").strip().lower()
        if not active:
            return
        try:
            character_engine = getattr(self.pipeline, "character_engine", None)
            if character_engine is None:
                return
            manifest = {}
            if hasattr(character_engine, "get_manifest"):
                manifest = dict(character_engine.get_manifest() or {})
            manifest_active = str(manifest.get("active_character_id") or "").strip().lower()
            if manifest_active != active and hasattr(character_engine, "set_active_character"):
                character_engine.set_active_character(active)
        except Exception:
            pass

    def _update_state_after_success(
        self,
        *,
        route: str,
        event_name: str,
        meta: dict[str, Any],
        result: BrainResult,
        track_state: bool,
    ) -> None:
        if not track_state:
            return

        if route in {"chat", "command"} and result.text:
            self.state_manager.update_on_assistant_message(
                result.text,
                {
                    "summary": meta.get("summary") or "",
                    "last_tool_result": (result.tool_calls[0] if result.tool_calls else None),
                },
            )
            return

        if route == "system_event":
            self.state_manager.update_on_event(event_name, meta)

    def _persist_turns(
        self,
        *,
        route: str,
        user_text: str,
        result: BrainResult,
        meta: dict[str, Any],
        non_persistent_turn: bool = False,
    ) -> dict[str, Any]:
        summary = self._empty_memory_write_summary()
        if route not in {"chat", "command"}:
            return summary
        if bool(non_persistent_turn):
            return summary
        if not bool(meta.get("store_turn", True)):
            return summary
        # By default, do not persist slash-commands to memory stores.
        if route == "command" and not bool(meta.get("store_command_turns", False)):
            return summary

        source = str(meta.get("source") or route or "text")
        conversation_id = str(meta.get("conversation_id") or self.state_manager.get("conversation_id") or "")
        turn_raw = meta.get("turn_id")
        if turn_raw is None:
            turn_raw = self.state_manager.get("turn_id")
        try:
            turn_id = int(turn_raw or 0)
        except Exception:
            turn_id = int(self.state_manager.get("turn_id") or 0)
        quality_profile = str(meta.get("quality_profile") or self.state_manager.get("quality_profile") or "BALANCED")
        trace_id = str(meta.get("trace_id") or "")
        request_id = str(meta.get("request_id") or "")
        model = str(result.stats.get("served_model") or meta.get("model") or "")

        state_snapshot = self.state_manager.snapshot()
        state_map = dict(state_snapshot.raw or {})
        history = list(state_snapshot.history or [])
        personality_id = str(
            state_snapshot.active_character_id
            or state_snapshot.active_personality_id
            or state_map.get("active_character_id")
            or state_map.get("active_personality_id")
            or "default"
        )
        persona_snapshot = self._extract_persona_snapshot(
            state_snapshot=state_snapshot,
            state_map=state_map,
            personality_id=personality_id,
        )

        user_payload = str(user_text or "").strip()
        if user_payload:
            user_meta = self._extract_turn_metadata(text=user_payload, state=state_map, last_messages=history)
            user_meta = self._merge_turn_meta(
                base=user_meta,
                op_tags=self._turn_meta_from_ops(result.memory_ops, "turn_user"),
                source=source,
                model=model,
                quality_profile=quality_profile,
                latency_ms=0.0,
                personality_id=personality_id,
            )
            ingest_result = self.memory_manager.ingest_event(
                MemoryEvent(
                    role="user",
                    text=user_payload,
                    namespace=(conversation_id or "default"),
                    scope=MemoryScope.CONVERSATION,
                    memory_type=MemoryType.MESSAGE,
                    metadata={
                        **dict(user_meta or {}),
                        "trace_id": trace_id,
                        "request_id": request_id,
                        "source": source,
                        "model": model,
                        "quality_profile": quality_profile,
                        "turn_id": turn_id,
                        "conversation_id": conversation_id or "default",
                    },
                )
            )
            self._capture_memory_ingest(summary, ingest_result, bucket="user_turn")

        assistant_sanitized = clean_assistant_text_for_memory(result)
        assistant_payload = str(assistant_sanitized.text or "").strip()
        if assistant_sanitized.reason in {"structured_output_text", "response_block_extract", "fallback_strip"}:
            log_json(
                LOGGER,
                "memory_text_sanitized",
                summary=f"reason={assistant_sanitized.reason} changed={int(bool(assistant_sanitized.changed))}",
                context=self._log_context_from_meta(meta, conversation_id=conversation_id),
                reason=assistant_sanitized.reason,
                changed=bool(assistant_sanitized.changed),
                had_service_sections=bool(assistant_sanitized.had_service_sections),
            )
        if contains_memory_service_sections(assistant_payload):
            LOGGER.warning("assistant memory payload still contains service sections; forcing sanitize")
            guard = sanitize_assistant_memory_text(text=assistant_payload)
            assistant_payload = str(guard.text or "").strip()
            log_json(
                LOGGER,
                "memory_text_sanitized",
                summary=(
                    f"reason=guard_{str(guard.reason or 'fallback_strip')} "
                    f"changed={int(bool(guard.changed))}"
                ),
                context=self._log_context_from_meta(meta, conversation_id=conversation_id),
                reason=f"guard_{str(guard.reason or 'fallback_strip')}",
                changed=bool(guard.changed),
                had_service_sections=bool(guard.had_service_sections),
            )
        if assistant_payload:
            try:
                answer_ms = float(result.stats.get("answer_ms") or 0.0)
            except Exception:
                answer_ms = 0.0
            assistant_meta = self._extract_turn_metadata(text=assistant_payload, state=state_map, last_messages=history)
            assistant_meta = self._merge_turn_meta(
                base=assistant_meta,
                op_tags=self._turn_meta_from_ops(result.memory_ops, "turn_assistant"),
                source=source,
                model=model,
                quality_profile=quality_profile,
                latency_ms=answer_ms,
                personality_id=personality_id,
            )
            if persona_snapshot:
                assistant_meta["persona_snapshot"] = dict(persona_snapshot)
            ingest_result = self.memory_manager.ingest_event(
                MemoryEvent(
                    role="assistant",
                    text=assistant_payload,
                    namespace=(conversation_id or "default"),
                    scope=MemoryScope.CONVERSATION,
                    memory_type=MemoryType.MESSAGE,
                    metadata={
                        **dict(assistant_meta or {}),
                        "thinking": str(result.thinking or ""),
                        "trace_id": trace_id,
                        "request_id": request_id,
                        "source": source,
                        "model": model,
                        "quality_profile": quality_profile,
                        "turn_id": turn_id,
                        "conversation_id": conversation_id or "default",
                    },
                )
            )
            self._capture_memory_ingest(summary, ingest_result, bucket="assistant_turn")
        return summary

    def _append_turn_summaries(
        self,
        *,
        result: BrainResult,
        route: str,
        user_text: str,
        meta: dict[str, Any],
        memory_apply_summary: dict[str, Any],
        persisted_summary: dict[str, Any],
    ) -> None:
        turn_summaries = _as_dict(meta.get("turn_log_summaries"))
        planned_write_summary = _as_dict(turn_summaries.get("memory_write_summary"))
        combined = self._merge_memory_write_summaries(memory_apply_summary, persisted_summary)
        memory_hits = int(_as_dict(turn_summaries.get("memory_summary")).get("selected_hits") or 0)
        intent_summary = _as_dict(turn_summaries.get("intent_summary"))
        intent_alignment_summary = _as_dict(turn_summaries.get("intent_alignment_summary"))
        web_summary = _as_dict(turn_summaries.get("web_summary"))
        web_issues = [str(x).strip() for x in list(web_summary.get("issues") or []) if str(x).strip()]
        web_warnings = [str(x).strip() for x in list(web_summary.get("warnings") or []) if str(x).strip()]
        warnings = [str(x).strip() for x in list(meta.get("turn_log_warnings") or []) if str(x).strip()]
        warnings.extend([str(x).strip() for x in list(combined.get("warnings") or []) if str(x).strip()])
        warnings.extend(web_warnings)
        warnings = [x for idx, x in enumerate(warnings) if x and x not in warnings[:idx]]
        issues = [x for idx, x in enumerate(web_issues) if x and x not in web_issues[:idx]]
        context = self._log_context_from_meta(meta, conversation_id=str(meta.get("conversation_id") or ""))

        memory_payload = {
            "route": str(route or ""),
            "queue_only": False,
            "queued_ops": int(planned_write_summary.get("queued_ops") or 0),
            "queued_turn_user": int(planned_write_summary.get("queued_turn_user") or 0),
            "queued_turn_assistant": int(planned_write_summary.get("queued_turn_assistant") or 0),
            "queued_web_memory_items": int(planned_write_summary.get("queued_web_memory_items") or 0),
            "queued_conversation_summaries": int(planned_write_summary.get("queued_conversation_summaries") or 0),
            "attempted_writes": int(combined.get("attempted_writes") or 0),
            "stored_records": int(combined.get("stored_records") or 0),
            "facts_extracted": int(combined.get("facts_extracted") or 0),
            "promotions": int(combined.get("promotions") or 0),
            "user_turns_written": int(combined.get("user_turns_written") or 0),
            "assistant_turns_written": int(combined.get("assistant_turns_written") or 0),
            "conversation_summaries_written": int(combined.get("conversation_summaries_written") or 0),
            "web_memory_items_written": int(combined.get("web_memory_items_written") or 0),
            "warnings": list(warnings),
        }
        memory_summary_text = (
            f"queued={memory_payload['queued_ops']} written={memory_payload['attempted_writes']} "
            f"facts={memory_payload['facts_extracted']} promotions={memory_payload['promotions']}"
        )
        result.logs.append(
            "summary=memory_write_summary "
            f"trace={context.get('trace_id') or '-'} "
            f"request={context.get('request_id') or '-'} "
            f"turn={context.get('turn_id') or '-'} "
            f"conversation={context.get('conversation_id') or '-'} "
            f"{memory_summary_text}"
        )
        log_json(LOGGER, "memory_write_summary", summary=memory_summary_text, context=context, **memory_payload)

        final_payload = {
            "route": str(route or ""),
            "user_text": str(user_text or ""),
            "final_intent": str(
                intent_summary.get("intent")
                or intent_alignment_summary.get("corrected_intent")
                or ""
            ),
            "original_intent": str(
                intent_alignment_summary.get("original_intent")
                or intent_summary.get("original_intent")
                or intent_summary.get("intent")
                or ""
            ),
            "corrected_intent": str(
                intent_alignment_summary.get("corrected_intent")
                or intent_summary.get("corrected_intent")
                or intent_summary.get("intent")
                or ""
            ),
            "intent_alignment_applied": bool(
                intent_alignment_summary.get("applied")
                if intent_alignment_summary.get("applied") is not None
                else intent_summary.get("intent_alignment_applied")
            ),
            "intent_alignment_reason": str(
                intent_alignment_summary.get("reason")
                or intent_summary.get("intent_alignment_reason")
                or ""
            ),
            "web_intent": str(
                intent_alignment_summary.get("web_intent")
                or intent_summary.get("web_intent")
                or web_summary.get("resolved_intent")
                or ""
            ),
            "intent_confidence": float(intent_summary.get("intent_confidence") or 0.0),
            "web_used": bool(web_summary.get("web_used", False)),
            "web_mode": str(web_summary.get("mode") or meta.get("web_mode") or ""),
            "web_result_count": int(web_summary.get("result_count") or 0),
            "web_sources_scanned": int(_as_dict(web_summary.get("sources")).get("scanned") or 0),
            "web_sources_selected": int(_as_dict(web_summary.get("sources")).get("selected") or 0),
            "memory_hits": int(memory_hits),
            "facts_extracted": int(combined.get("facts_extracted") or 0),
            "promotions": int(combined.get("promotions") or 0),
            "issues": list(issues),
            "warnings": list(warnings),
            "status": str(result.status or "ok"),
        }
        final_summary_text = (
            f"intent={final_payload['final_intent'] or '-'} "
            f"web_used={str(bool(final_payload['web_used'])).lower()} "
            f"memory_hits={final_payload['memory_hits']} "
            f"facts={final_payload['facts_extracted']} "
            f"promotions={final_payload['promotions']} "
            f"warnings={len(warnings)}"
        )
        result.logs.append(
            "summary=final_turn_summary "
            f"trace={context.get('trace_id') or '-'} "
            f"request={context.get('request_id') or '-'} "
            f"turn={context.get('turn_id') or '-'} "
            f"conversation={context.get('conversation_id') or '-'} "
            f"{final_summary_text}"
        )
        log_json(LOGGER, "final_turn_summary", summary=final_summary_text, context=context, **final_payload)
        append_human_log(
            "TURN",
            context=context,
            lines=_human_turn_summary_lines(
                route=str(route or ""),
                user_text=str(user_text or ""),
                intent_summary=intent_summary,
                intent_alignment_summary=intent_alignment_summary,
                web_summary=web_summary,
                memory_hits=memory_hits,
                facts_extracted=int(combined.get("facts_extracted") or 0),
                promotions=int(combined.get("promotions") or 0),
                warnings=warnings,
                issues=issues,
                status=str(result.status or "ok"),
            ),
        )

    @staticmethod
    def _empty_memory_write_summary() -> dict[str, Any]:
        return {
            "attempted_writes": 0,
            "stored_records": 0,
            "facts_extracted": 0,
            "promotions": 0,
            "user_turns_written": 0,
            "assistant_turns_written": 0,
            "conversation_summaries_written": 0,
            "web_memory_items_written": 0,
            "planned_web_memory_items": 0,
            "warnings": [],
        }

    @staticmethod
    def _append_summary_warning(summary: dict[str, Any], warning: str) -> None:
        text = str(warning or "").strip()
        if not text:
            return
        warnings = [str(x).strip() for x in list(summary.get("warnings") or []) if str(x).strip()]
        if text not in warnings:
            warnings.append(text)
        summary["warnings"] = warnings

    def _capture_memory_ingest(self, summary: dict[str, Any], ingest_result, *, bucket: str) -> None:
        summary["attempted_writes"] = int(summary.get("attempted_writes") or 0) + 1
        stored_ids = list(getattr(ingest_result, "stored_ids", []) or [])
        promoted_ids = list(getattr(ingest_result, "promoted_ids", []) or [])
        extracted_facts = list(getattr(ingest_result, "extracted_facts", []) or [])
        summary["stored_records"] = int(summary.get("stored_records") or 0) + len(stored_ids)
        summary["facts_extracted"] = int(summary.get("facts_extracted") or 0) + len(extracted_facts)
        summary["promotions"] = int(summary.get("promotions") or 0) + len(promoted_ids)
        key = str(bucket or "").strip().lower()
        if key == "user_turn":
            summary["user_turns_written"] = int(summary.get("user_turns_written") or 0) + 1
        elif key == "assistant_turn":
            summary["assistant_turns_written"] = int(summary.get("assistant_turns_written") or 0) + 1
        elif key == "conversation_summary":
            summary["conversation_summaries_written"] = int(summary.get("conversation_summaries_written") or 0) + 1
        elif key == "web_memory_write":
            summary["web_memory_items_written"] = int(summary.get("web_memory_items_written") or 0) + 1

    @staticmethod
    def _merge_memory_write_summaries(*parts: dict[str, Any]) -> dict[str, Any]:
        out = Brain._empty_memory_write_summary()
        for part in list(parts or []):
            row = dict(part or {})
            for key in (
                "attempted_writes",
                "stored_records",
                "facts_extracted",
                "promotions",
                "user_turns_written",
                "assistant_turns_written",
                "conversation_summaries_written",
                "web_memory_items_written",
                "planned_web_memory_items",
            ):
                out[key] = int(out.get(key) or 0) + int(row.get(key) or 0)
            for warning in list(row.get("warnings") or []):
                Brain._append_summary_warning(out, str(warning or ""))
        return out

    @staticmethod
    def _log_context_from_meta(meta: dict[str, Any], *, conversation_id: str) -> dict[str, Any]:
        row = dict(meta or {})
        return {
            "trace_id": str(row.get("trace_id") or "").strip(),
            "request_id": str(row.get("request_id") or row.get("trace_id") or "").strip(),
            "turn_id": str(row.get("turn_id") or "").strip(),
            "conversation_id": str(row.get("conversation_id") or conversation_id or "").strip(),
        }

    def _extract_turn_metadata(self, *, text: str, state: dict[str, Any], last_messages) -> dict[str, Any]:
        src = str(text or "").strip()
        if not src:
            return {}
        try:
            item = self.metadata_extractor.extract(text=src, state=state, last_messages=last_messages)
            payload = item.to_dict() if hasattr(item, "to_dict") else _as_dict(item)
        except Exception:
            payload = {}
        return self._flatten_turn_metadata(payload)

    @staticmethod
    def _flatten_turn_metadata(payload: dict[str, Any]) -> dict[str, Any]:
        data = _as_dict(payload)
        intent = _as_dict(data.get("intent"))
        emotion = _as_dict(data.get("emotion"))
        meta = _as_dict(data.get("meta"))
        raw_tags = list(data.get("tags") or [])
        tags = [str(x).strip() for x in raw_tags if str(x).strip()]
        topics: list[str] = []
        topic = ""
        for tag in tags:
            if str(tag).startswith("topic_"):
                row = str(tag).replace("topic_", "", 1)
                if row and row not in topics:
                    topics.append(row)
                if not topic:
                    topic = row
        for row in list(meta.get("topics") or []):
            item = str(row or "").strip().replace("topic_", "", 1)
            if not item:
                continue
            if item not in topics:
                topics.append(item)
            if not topic:
                topic = item
        intent_label = str(intent.get("label") or data.get("intent") or "")
        emotion_label = str(emotion.get("label") or data.get("emotion") or "")
        out = {
            "lang": str(data.get("lang") or ""),
            "intent": intent_label,
            "mood": emotion_label,
            "emotion": emotion_label,
            "topic": topic,
            "topics": topics,
            "active_mode": str(data.get("active_mode") or meta.get("active_mode") or ""),
            "tags": tags,
            "has_code": bool(meta.get("has_code", False)),
            "has_link": bool(meta.get("has_link", False)),
            "entities": dict(data.get("entities") or {}),
            "meta": data,
        }
        try:
            out["confidence"] = float(intent.get("conf") or 0.0)
        except Exception:
            out["confidence"] = 0.0
        return out

    def _character_quality_profile(self, *, state_snapshot, state_map: dict[str, Any]) -> str:
        candidate_ids = [
            state_map.get("active_character_id"),
            state_map.get("active_personality_id"),
            getattr(state_snapshot, "active_character_id", ""),
            getattr(state_snapshot, "active_personality_id", ""),
        ]
        cid = next((str(x or "").strip() for x in candidate_ids if str(x or "").strip()), "")
        if not cid:
            return ""
        try:
            meta = self.state_manager.get_meta(cid)
            raw = str(getattr(meta, "llm_profile", "") or "").strip().upper()
        except Exception:
            raw = ""
        if not raw:
            return ""
        if raw == "ECONOM":
            return "FAST"
        if raw in {"FAST", "BALANCED", "QUALITY", "ASYA", "AUTONOMOUS"}:
            return raw
        return ""

    @staticmethod
    def _merge_turn_meta(
        *,
        base: dict[str, Any],
        op_tags: dict[str, Any],
        source: str,
        model: str,
        quality_profile: str,
        latency_ms: float,
        personality_id: str,
    ) -> dict[str, Any]:
        merged = dict(base or {})
        tags = [str(x).strip().lower() for x in list(merged.get("tags") or []) if str(x).strip()]
        extra = dict(op_tags or {})

        for key in ("lang", "intent", "mood", "topic", "active_mode"):
            value = str(extra.get(key) or "").strip()
            if value and not str(merged.get(key) or "").strip():
                merged[key] = value

        for key in ("lang", "intent", "mood", "topic", "active_mode"):
            value = str(merged.get(key) or "").strip().lower()
            if not value:
                continue
            if key == "lang":
                tags.append(f"lang_{value}")
            elif key == "intent":
                tags.append(f"intent_{value}")
            elif key == "mood":
                tags.append(f"emotion_{value}")
            elif key == "topic":
                tags.append(f"topic_{value}")
            elif key == "active_mode":
                tags.append(f"mode_hint_{value}")

        if merged.get("has_code"):
            tags.append("has_code")
        if merged.get("has_link"):
            tags.append("has_link")
        pid = str(personality_id or "").strip().lower()
        if pid:
            merged["personality_id"] = pid
            tags.append(f"personality_{pid}")

        seen: set[str] = set()
        uniq_tags: list[str] = []
        for tag in tags:
            row = str(tag or "").strip().lower()
            if not row or row in seen:
                continue
            seen.add(row)
            uniq_tags.append(row)
        merged["tags"] = uniq_tags
        merged["source"] = str(source or "")
        merged["model"] = str(model or "")
        merged["quality_profile"] = str(quality_profile or "")
        merged["latency_ms"] = max(0.0, float(latency_ms or 0.0))
        return merged

    @staticmethod
    def _extract_persona_snapshot(
        *,
        state_snapshot,
        state_map: dict[str, Any],
        personality_id: str,
    ) -> dict[str, Any]:
        raw = dict(getattr(state_snapshot, "raw", {}) or {})
        chars = dict(raw.get("characters") or {})
        cid = str(
            raw.get("active_character_id")
            or state_map.get("active_character_id")
            or personality_id
            or "default"
        ).strip().lower() or "default"
        entry = dict(chars.get(cid) or {})
        persona = dict(entry.get("persona") or {})
        traits_raw = dict(persona.get("traits") or {})
        context_tags = _as_dict(state_map.get("context_tags"))
        if not persona and not traits_raw:
            return {}
        traits: dict[str, float] = {}
        for key in ("warmth", "sarcasm", "verbosity", "strictness", "teasing", "empathy"):
            if key not in traits_raw:
                continue
            try:
                traits[key] = max(0.0, min(1.0, float(traits_raw.get(key))))
            except Exception:
                continue
        mood = str(
            persona.get("mood")
            or state_map.get("mood")
            or context_tags.get("mood")
            or "neutral"
        ).strip().lower() or "neutral"
        active_mode = str(
            raw.get("active_mode")
            or state_map.get("active_mode")
            or "chatting"
        ).strip().lower() or "chatting"
        if not traits and not mood:
            return {}
        return {
            "character_id": cid,
            "mood": mood,
            "traits": traits,
            "active_mode": active_mode,
        }

    @staticmethod
    def _turn_meta_from_ops(ops: list[dict[str, Any]], op_name: str) -> dict[str, Any]:
        target = str(op_name or "").strip().lower()
        for row in list(ops or []):
            if not isinstance(row, dict):
                continue
            if str(row.get("op") or "").strip().lower() != target:
                continue
            tags = row.get("tags")
            if isinstance(tags, dict):
                return dict(tags)
        return {}

    @staticmethod
    def _build_error_result(*, route: str, error: Exception) -> BrainResult:
        fallback = "Я затупила. Повтори, пожалуйста, еще раз."
        return BrainResult(
            text=fallback,
            route=route,
            status="error",
            error=str(error or ""),
            logs=[f"error={type(error).__name__}", str(error or "")],
            ui_actions=[{"type": "toast", "level": "error", "text": "brain_error"}],
        )

    def _check_duplicate(self, signature: str, now: float) -> BrainResult | None:
        last = float(self._recent_events.get(signature) or 0.0)
        if (now - last) > self._dedup_window_sec:
            return None
        prev = self._recent_results.get(signature)
        if prev is None:
            return BrainResult(text="", route="chat", deduped=True, logs=["dedup(no-cache)"])
        return BrainResult(
            text=prev.text,
            route=prev.route,
            structured_output=dict(prev.structured_output or {}),
            tool_calls=prev.tool_calls,
            memory_ops=prev.memory_ops,
            ui_actions=prev.ui_actions,
            logs=[*prev.logs, "dedup(hit)"],
            status=prev.status,
            deduped=True,
            error=prev.error,
            stats=prev.stats,
        )

    def _is_throttled(self, source: str, now: float) -> bool:
        if self._throttle_sec <= 0:
            self._last_source_ts[source] = now
            return False
        last = float(self._last_source_ts.get(source) or 0.0)
        if (now - last) < self._throttle_sec:
            return True
        self._last_source_ts[source] = now
        return False

    def _remember(self, signature: str, now: float, result: BrainResult) -> None:
        self._recent_events[signature] = now
        self._recent_results[signature] = result
        self._evict_old(now)

    def _evict_old(self, now: float) -> None:
        max_age = max(self._dedup_window_sec * 4.0, 3.0)
        cutoff = now - max_age
        stale = [sig for sig, ts in self._recent_events.items() if float(ts) < cutoff]
        for sig in stale:
            self._recent_events.pop(sig, None)
            self._recent_results.pop(sig, None)

    @staticmethod
    def _signature(*, route: str, text: str, event_name: str, source: str) -> str:
        payload = f"{route}|{source}|{event_name.strip().lower()}|{text.strip()}"
        return hashlib.sha1(payload.encode("utf-8", errors="ignore")).hexdigest()


def _as_dict(value) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if value is None:
        return {}
    try:
        return dict(vars(value))
    except Exception:
        return {}


def _human_turn_summary_lines(
    *,
    route: str,
    user_text: str,
    intent_summary: dict[str, Any],
    intent_alignment_summary: dict[str, Any],
    web_summary: dict[str, Any],
    memory_hits: int,
    facts_extracted: int,
    promotions: int,
    warnings: list[str],
    issues: list[str],
    status: str,
) -> list[str]:
    web_sources = _as_dict(web_summary.get("sources"))
    evidence = _as_dict(web_summary.get("evidence"))
    intent = str(intent_summary.get("intent") or "").strip() or "-"
    intent_conf = float(intent_summary.get("intent_confidence") or 0.0)
    alignment_applied = bool(
        intent_alignment_summary.get("applied")
        if intent_alignment_summary.get("applied") is not None
        else intent_summary.get("intent_alignment_applied")
    )
    original_intent = str(
        intent_alignment_summary.get("original_intent")
        or intent_summary.get("original_intent")
        or ""
    ).strip()
    alignment_reason = str(
        intent_alignment_summary.get("reason")
        or intent_summary.get("intent_alignment_reason")
        or ""
    ).strip()
    web_used = bool(web_summary.get("web_used", False))
    mode = str(web_summary.get("mode") or "").strip() or "NO_SEARCH"
    scanned = int(web_sources.get("scanned") or 0)
    selected = int(web_sources.get("selected") or 0)
    evidence_count = int(evidence.get("count") or 0)
    quality_score = float(evidence.get("quality_score") or 0.0)
    issue_text = ", ".join(str(x).strip() for x in list(issues or []) if str(x).strip()) or "none"
    warning_text = ", ".join(str(x).strip() for x in list(warnings or []) if str(x).strip()) or "none"
    return [
        f"user: {_clip_text(user_text, max_chars=220) or '-'}",
        (
            f"intent: {intent} ({intent_conf:.2f}) route={route or '-'} status={status or 'ok'}"
            + (
                f" aligned_from={original_intent} reason={alignment_reason or '-'}"
                if alignment_applied and original_intent and original_intent != intent
                else ""
            )
        ),
        (
            f"web: used={'yes' if web_used else 'no'} mode={mode} "
            f"sources={scanned}/{selected} evidence={evidence_count} quality={quality_score:.3f}"
        ),
        f"memory: hits={int(memory_hits)} facts={int(facts_extracted)} promotions={int(promotions)}",
        f"issues: {issue_text}",
        f"warnings: {warning_text}",
    ]


def _clip_text(value: Any, *, max_chars: int) -> str:
    text = re.sub(r"\s+", " ", str(value or "").strip())
    limit = max(32, int(max_chars or 0))
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _normalize_command_candidate(value: str) -> str:
    text = str(value or "")
    text = text.lstrip()
    while text and text[0] in {"\ufeff", "\u200b", "\u200c", "\u200d", "\u2060"}:
        text = text[1:].lstrip()
    # Tolerate accidental console prompt prefixes copied into input.
    text = re.sub(r"^\s*(?:you|user|assistant)\s*>\s*", "", text, flags=re.IGNORECASE)
    return text


def _memory_scope_from_name(value: Any, *, default: MemoryScope) -> MemoryScope:
    raw = str(value or "").strip().lower()
    mapping = {
        "global_user": MemoryScope.GLOBAL_USER,
        "conversation": MemoryScope.CONVERSATION,
        "session": MemoryScope.SESSION,
        "project": MemoryScope.PROJECT,
        "character": MemoryScope.CHARACTER,
        "temporary": MemoryScope.TEMPORARY,
        "private_runtime": MemoryScope.PRIVATE_RUNTIME,
    }
    return mapping.get(raw, default)


def _memory_type_from_name(value: Any, *, default: MemoryType) -> MemoryType:
    raw = str(value or "").strip().lower()
    mapping = {
        "message": MemoryType.MESSAGE,
        "summary": MemoryType.SUMMARY,
        "fact": MemoryType.FACT,
        "episode": MemoryType.EPISODE,
        "semantic": MemoryType.SEMANTIC,
        "document": MemoryType.DOCUMENT,
        "document_chunk": MemoryType.DOCUMENT_CHUNK,
        "task_state": MemoryType.TASK_STATE,
        "tool_result": MemoryType.TOOL_RESULT,
        "runtime_state": MemoryType.RUNTIME_STATE,
    }
    return mapping.get(raw, default)
