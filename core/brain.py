from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from threading import RLock
from typing import Any

from config.settings import load_config
from core.response_pipeline import PipelineResult, ResponsePipeline
from core.state_manager import StateManager
from llm import build_provider
from llm.provider_base import LLMProviderBase
from memory.memory_manager import MemoryManager
from metadata.metadata_extractor import MetadataExtractor


@dataclass(frozen=True)
class BrainResult:
    text: str
    route: str
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
        state_manager: StateManager | None = None,
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
        self.state_manager = state_manager or StateManager()
        self.memory_manager = memory_manager or MemoryManager()
        self.metadata_extractor = metadata_extractor or MetadataExtractor(cache_size=280)
        self.pipeline = response_pipeline or ResponsePipeline(self._provider, memory_manager=self.memory_manager)

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
        track_state = bool(meta_map.get("track_state", True))
        if track_state and route in {"chat", "command"} and text:
            self.state_manager.update_on_user_message(
                text,
                {
                    "conversation_id": meta_map.get("conversation_id") or state_snapshot.conversation_id,
                    "mode": meta_map.get("mode") or state_snapshot.mode,
                    "quality_profile": meta_map.get("quality_profile") or state_snapshot.quality_profile,
                    "active_goal": meta_map.get("active_goal") or state_snapshot.active_goal,
                },
            )
            state_snapshot = self.state_manager.snapshot()
        elif track_state and route == "system_event":
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

        retrieved_memories = meta_map.get("retrieved_memories", state_snapshot.retrieved_memories)
        traits = meta_map.get("traits", state_snapshot.traits)
        policies = meta_map.get("policies", state_snapshot.policies)
        meta_for_pipeline = dict(meta_map)
        meta_for_pipeline.setdefault("source", source or route)
        meta_for_pipeline.setdefault("memory_manager", self.memory_manager)
        meta_for_pipeline.setdefault("conversation_id", state_snapshot.conversation_id)
        meta_for_pipeline.setdefault("turn_id", state_snapshot.turn_id)
        meta_for_pipeline.setdefault("quality_profile", state_snapshot.quality_profile)
        meta_for_pipeline.setdefault("track_state", track_state)

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
            self._apply_memory_ops(result.memory_ops)
            self._update_state_after_success(
                route=route,
                event_name=event_name,
                meta=meta_for_pipeline,
                result=result,
                track_state=track_state,
            )
            self._persist_turns(
                route=route,
                user_text=text,
                result=result,
                meta=meta_for_pipeline,
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

        if text.startswith("/"):
            return "command", text, ""
        return "chat", text, ""

    def _to_brain_result(self, *, route: str, pipeline_result: PipelineResult) -> BrainResult:
        return BrainResult(
            text=str(pipeline_result.text or ""),
            route=route,
            tool_calls=list(pipeline_result.tool_calls or []),
            memory_ops=list(pipeline_result.memory_ops or []),
            ui_actions=list(pipeline_result.ui_actions or []),
            logs=list(pipeline_result.logs or []),
            stats=dict(pipeline_result.stats or {}),
            status="ok",
        )

    def _apply_memory_ops(self, ops: list[dict[str, Any]]) -> None:
        for op in list(ops or []):
            if not isinstance(op, dict):
                continue
            key = str(op.get("op") or "").strip().lower()
            if key == "state_mode":
                value = str(op.get("value") or "").strip()
                if value:
                    self.state_manager.set_mode(value)
            elif key == "state_think":
                self.state_manager.patch({"thinking_enabled": bool(op.get("value", False))})
            elif key == "state_personality":
                value = str(op.get("value") or "").strip().lower()
                if value:
                    locked = op.get("locked")
                    blend = op.get("blend") if isinstance(op.get("blend"), dict) else None
                    ts = op.get("ts")
                    try:
                        ts_val = float(ts) if ts is not None else None
                    except Exception:
                        ts_val = None
                    self.state_manager.set_active_personality(
                        value,
                        locked=(bool(locked) if isinstance(locked, bool) else None),
                        blend=blend,
                        switch_ts=ts_val,
                    )
            elif key == "state_character":
                value = str(op.get("value") or "").strip().lower()
                if value:
                    locked = op.get("locked")
                    ts = op.get("ts")
                    try:
                        ts_val = float(ts) if ts is not None else None
                    except Exception:
                        ts_val = None
                    self.state_manager.set_active_character(
                        value,
                        locked=(bool(locked) if isinstance(locked, bool) else None),
                        switch_ts=ts_val,
                    )
            elif key == "tool_results":
                items = list(op.get("items") or [])
                if items:
                    self.state_manager.set_last_tool_result(items[-1])
            elif key == "conversation_summary":
                text = str(op.get("text") or "").strip()
                if text:
                    self.state_manager.set_dialog_summary(text)
                    try:
                        self.memory_manager.long_memory.add_doc(
                            text=text,
                            meta={"source": "rolling_summary"},
                            source="summary",
                            tags=["summary", "long_summary"],
                            importance=0.6,
                            confidence=0.7,
                        )
                    except Exception:
                        pass
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
    ) -> None:
        if route not in {"chat", "command"}:
            return
        if not bool(meta.get("store_turn", True)):
            return

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
            self.memory_manager.ingest_message(
                role="user",
                text=user_payload,
                metadata=user_meta,
                trace_id=trace_id,
                source=source,
                model=model,
                quality_profile=quality_profile,
                conversation_id=conversation_id,
                turn_id=turn_id,
            )

        assistant_payload = str(result.text or "").strip()
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
            self.memory_manager.ingest_message(
                role="assistant",
                text=assistant_payload,
                metadata=assistant_meta,
                trace_id=trace_id,
                source=source,
                model=model,
                quality_profile=quality_profile,
                conversation_id=conversation_id,
                turn_id=turn_id,
            )

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
        topic = ""
        for tag in tags:
            if str(tag).startswith("topic_"):
                topic = str(tag).replace("topic_", "", 1)
                break
        intent_label = str(intent.get("label") or data.get("intent") or "")
        emotion_label = str(emotion.get("label") or data.get("emotion") or "")
        out = {
            "lang": str(data.get("lang") or ""),
            "intent": intent_label,
            "mood": emotion_label,
            "emotion": emotion_label,
            "topic": topic,
            "tags": tags,
            "has_code": bool(meta.get("has_code", False)),
            "has_link": bool(meta.get("has_link", False)),
            "meta": data,
        }
        try:
            out["confidence"] = float(intent.get("conf") or 0.0)
        except Exception:
            out["confidence"] = 0.0
        return out

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

        for key in ("lang", "intent", "mood", "topic"):
            value = str(extra.get(key) or "").strip()
            if value and not str(merged.get(key) or "").strip():
                merged[key] = value

        for key in ("lang", "intent", "mood", "topic"):
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

        if merged.get("has_code"):
            tags.append("has_code")
        if merged.get("has_link"):
            tags.append("contains_link")
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

