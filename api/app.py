from __future__ import annotations

import json
import queue
import re
import threading
import time
from pathlib import Path
from threading import Lock
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse

from api.memory_core_api import register_memory_core_api
from api.schemas import (
    ChatRequest,
    ChatResponse,
    FeedbackRequest,
    HealthResponse,
    JsonModeRequest,
    MemoryInspectorResponse,
    MetadataResponse,
    ModelSetRequest,
    ModelsResponse,
    ThinkingRequest,
    VerboseRequest,
    WebModeRequest,
)
from config.settings import get_profile, load_config
from core.brain import Brain
from core.spec_registry import validate_no_txt_paths
from llm import build_provider
from utils.logger import get_logger, log_json


cfg = load_config(force_reload=True)
validate_no_txt_paths(cfg)
app = FastAPI(title="MMis API", version="2.1.0")
LOGGER = get_logger(__name__)

# Регистрируем Memory Core API
register_memory_core_api(app)


class _Runtime:
    def __init__(self):
        self.settings = load_config(force_reload=True)
        self.lock = Lock()
        self.provider = build_provider(self.settings.llm_default_provider, default_model=self.settings.model_name)
        provider_raw = str(self.settings.llm_default_provider or "ollama").strip().lower()
        self.provider_name = "openai" if provider_raw == "openai" else "ollama"
        self.active_profile = str(self.settings.active_profile or "BALANCED").strip().upper() or "BALANCED"
        quality_raw = str(self.active_profile or "BALANCED").strip().upper()
        if quality_raw == "ECONOM":
            self.quality_profile = "FAST"
        elif quality_raw in {"FAST", "BALANCED", "QUALITY"}:
            self.quality_profile = quality_raw
        else:
            self.quality_profile = "BALANCED"
        self.brain = Brain(provider=self.provider)
        self.model = str(self.settings.model_name or "").strip()
        self.thinking_enabled = bool(self.settings.thinking_enabled)
        self.verbose_enabled = bool(self.brain.state_manager.get("verbose_enabled", False))
        self.web_mode = self.settings.web_mode if bool(self.settings.internet_enabled) else "off"
        self.json_mode_enabled = bool(self.settings.json_mode_enabled)

        self.meta_root = Path(self.settings.memory_dir) / "metadata"
        self.meta_root.mkdir(parents=True, exist_ok=True)
        self._message_seq = self._load_last_message_id()
        self.last_stats: dict[str, Any] = {}
        self.last_thinking: str = ""
        self.last_debug_trace: dict[str, Any] = {}
        self.last_memory_debug_snapshot: dict[str, Any] = {}
        self.last_conversation_id: str = ""

    def next_message_id(self) -> int:
        self._message_seq += 1
        return self._message_seq

    def _load_last_message_id(self) -> int:
        highest = 0
        for jsonl_path in self.meta_root.rglob("metadata_messages.jsonl"):
            try:
                with jsonl_path.open("r", encoding="utf-8-sig") as f:
                    for line in f:
                        raw = str(line or "").strip()
                        if not raw:
                            continue
                        try:
                            row = json.loads(raw)
                        except Exception:
                            continue
                        try:
                            mid = int(row.get("message_id") or 0)
                        except Exception:
                            mid = 0
                        if mid > highest:
                            highest = mid
            except Exception:
                continue
        return highest


_runtime = _Runtime()


def _normalize_profile_name(value: Any, default: str = "BALANCED") -> str:
    token = str(value or "").strip().upper()
    if token in {"FAST", "BALANCED", "QUALITY", "ECONOM", "AUTONOMOUS", "ASYA"}:
        return token
    return str(default or "BALANCED").strip().upper() or "BALANCED"


def _profile_to_quality(profile_name: str) -> str:
    norm = _normalize_profile_name(profile_name, default="BALANCED")
    return "FAST" if norm == "ECONOM" else norm


def _resolve_effective_profiles() -> tuple[str, str]:
    active_profile = _normalize_profile_name(_runtime.active_profile, default="BALANCED")
    try:
        state_mgr = getattr(_runtime.brain, "state_manager", None)
        if state_mgr is not None and hasattr(state_mgr, "get_active_character_id") and hasattr(state_mgr, "get_meta"):
            char_id = str(state_mgr.get_active_character_id() or "").strip().lower()
            if char_id:
                meta = state_mgr.get_meta(char_id)
                llm_profile = ""
                if isinstance(meta, dict):
                    llm_profile = str(meta.get("llm_profile") or "").strip()
                else:
                    llm_profile = str(getattr(meta, "llm_profile", "") or "").strip()
                if llm_profile:
                    active_profile = _normalize_profile_name(llm_profile, default=active_profile)
    except Exception:
        pass
    quality_profile = _profile_to_quality(active_profile)
    return active_profile, quality_profile


def _resolved_profile_payload() -> tuple[Any, dict[str, Any]]:
    active_profile, _quality_profile = _resolve_effective_profiles()
    profile = get_profile(active_profile)
    payload = {
        "generation": {
            "temperature": float(profile.generation.temperature),
            "top_p": float(profile.generation.top_p),
            "repeat_penalty": float(profile.generation.repeat_penalty),
            "max_tokens": (
                int(profile.generation.max_tokens)
                if profile.generation.max_tokens is not None
                else None
            ),
            "stop": [str(x) for x in list(profile.generation.stop or ()) if str(x)],
        },
        "ollama": {
            "num_thread": int(profile.ollama.num_thread),
            "num_ctx": int(profile.ollama.num_ctx),
            "num_gpu": int(profile.ollama.num_gpu),
            "num_batch": int(profile.ollama.num_batch),
            "keep_alive": str(profile.ollama.keep_alive),
        },
        "openai": {
            "model": str(profile.openai.model),
            "reasoning_effort": str(profile.openai.reasoning_effort),
        },
    }
    return profile, payload


def _build_health_response() -> HealthResponse:
    active_profile, quality_profile = _resolve_effective_profiles()
    _profile, profile_payload = _resolved_profile_payload()
    return HealthResponse(
        status="ok",
        model=_runtime.model,
        thinking_enabled=bool(_runtime.thinking_enabled),
        verbose_enabled=bool(_runtime.verbose_enabled),
        json_mode_enabled=bool(_runtime.json_mode_enabled),
        web_mode=str(_runtime.web_mode),
        active_profile=str(active_profile or "BALANCED"),
        quality_profile=str(quality_profile or "BALANCED"),
        profile_parameters=profile_payload,
    )


def _remember_debug_payload(*, meta_map: dict[str, Any] | None) -> None:
    row = dict(meta_map or {})
    _runtime.last_debug_trace = dict(row.get("debug_trace") or {})
    _runtime.last_memory_debug_snapshot = dict(row.get("memory_debug_snapshot") or {})
    _runtime.last_conversation_id = str(row.get("conversation_id") or _runtime.last_conversation_id or "").strip()


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    with _runtime.lock:
        return _build_health_response()


@app.get("/models", response_model=ModelsResponse)
def list_models() -> ModelsResponse:
    with _runtime.lock:
        models = _safe_list_models(_runtime.provider)
        return ModelsResponse(runtime_model=_runtime.model, models=models)


@app.post("/models", response_model=ModelsResponse)
def set_model(req: ModelSetRequest) -> ModelsResponse:
    target = str(req.model or "").strip()
    if not target:
        raise HTTPException(status_code=400, detail="Model name is empty")

    with _runtime.lock:
        ok, error_text, models = _apply_runtime_model(target)
        if not ok:
            raise HTTPException(status_code=404, detail=error_text or f"Model '{target}' is not available")
        return ModelsResponse(runtime_model=_runtime.model, models=models)


@app.post("/thinking", response_model=HealthResponse)
def set_thinking(req: ThinkingRequest) -> HealthResponse:
    with _runtime.lock:
        _runtime.thinking_enabled = bool(req.enabled)
        return _build_health_response()

@app.post("/verbose", response_model=HealthResponse)
def set_verbose(req: VerboseRequest) -> HealthResponse:
    with _runtime.lock:
        _runtime.verbose_enabled = bool(req.enabled)
        try:
            _runtime.brain.state_manager.patch({"verbose_enabled": bool(req.enabled)})
        except Exception:
            pass
        return _build_health_response()

@app.post("/web-mode", response_model=HealthResponse)
def set_web_mode(req: WebModeRequest) -> HealthResponse:
    mode = str(req.mode or "").strip().lower()
    if mode not in {"auto", "on", "off"}:
        raise HTTPException(status_code=400, detail="mode must be one of: auto, on, off")
    
    with _runtime.lock:
        _runtime.web_mode = mode
        return _build_health_response()

@app.post("/json-mode", response_model=HealthResponse)
def set_json_mode(req: JsonModeRequest) -> HealthResponse:
    with _runtime.lock:
        _runtime.json_mode_enabled = bool(req.enabled)
        return _build_health_response()


@app.get("/debug/memory-inspector", response_model=MemoryInspectorResponse)
def memory_inspector_debug(
    conversation_id: str = Query(default="", description="Conversation namespace for store debug snapshot"),
    limit: int = Query(default=80, ge=1, le=500),
    include_store: bool = Query(default=True),
) -> MemoryInspectorResponse:
    with _runtime.lock:
        namespace = str(conversation_id or _runtime.last_conversation_id or "default").strip() or "default"
        debug_trace = dict(_runtime.last_debug_trace or {})
        snapshot = dict(_runtime.last_memory_debug_snapshot or {})
        if snapshot:
            snapshot.setdefault("conversation_id", namespace)
        memory_store_debug = None
        memory_core = getattr(_runtime.brain, "memory_core", None)
        if include_store and memory_core is not None:
            try:
                memory_store_debug = memory_core.debug_snapshot(limit=max(1, int(limit)))
            except Exception:
                memory_store_debug = None
        return MemoryInspectorResponse(
            conversation_id=namespace,
            request_id=str(snapshot.get("request_id") or debug_trace.get("request_id") or ""),
            debug_trace=debug_trace or None,
            memory_debug_snapshot=snapshot or None,
            memory_store_debug=(dict(memory_store_debug or {}) or None),
        )

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    text = str(req.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text is empty")

    with _runtime.lock:
        native = _handle_native_chat_command(text)
        if native is not None:
            pass_text = str(native.get("pass_text") or "").strip()
            if pass_text:
                text = pass_text
            else:
                answer = str(native.get("answer") or "")
                stats = {"served_model": _runtime.model, "native_command": True}
                if bool(req.store_turn) and _should_store_metadata(text=text):
                    _append_metadata_row(model=_runtime.model, role="user", text=text, context=answer)
                    _append_metadata_row(model=_runtime.model, role="assistant", text=answer, context=text)
                return ChatResponse(
                    answer=answer,
                    thinking="",
                    stats=stats,
                    model=_runtime.model,
                    parameters=None,
                    summary=None,
                )

        log_json(
            LOGGER,
            "api_chat_start",
            model=_runtime.model,
            text_chars=len(text),
            store_turn=bool(req.store_turn),
            think=_runtime.thinking_enabled if req.think is None else bool(req.think),
            verbose=_runtime.verbose_enabled if req.verbose is None else bool(req.verbose),
            json_mode=_runtime.json_mode_enabled if req.json_mode is None else bool(req.json_mode),
        )
        meta_map = _build_chat_meta(req=req, source="api")
        result = _runtime.brain.handle_message(
            text,
            meta=meta_map,
        )

        answer_raw = str(result.text or "")
        answer, thinking = _split_visible_and_thinking(answer_raw)
        if not thinking.strip():
            thinking = str(getattr(result, "thinking", "") or "").strip()
        structured = dict(getattr(result, "structured_output", {}) or {})
        
        # Debug trace берётся из result
        debug_trace = dict(getattr(result, "debug_trace", {}) or {})
        memory_debug_snapshot = dict(getattr(result, "memory_debug_snapshot", {}) or {})
        _remember_debug_payload(meta_map={"debug_trace": debug_trace, "memory_debug_snapshot": memory_debug_snapshot})
        parameters = structured.get("parameters") if isinstance(structured.get("parameters"), dict) else None
        summary = structured.get("summary")
        summary_text = str(summary).strip() if summary is not None else None
        if summary_text == "":
            summary_text = None
        stats = dict(result.stats or {})
        stats.setdefault("served_model", _runtime.model)
        _runtime.last_stats = stats
        _runtime.last_thinking = thinking

        if bool(req.store_turn) and _should_store_metadata(text=text, structured_output=structured):
            _append_metadata_row(model=_runtime.model, role="user", text=text, context=answer)
            _append_metadata_row(model=_runtime.model, role="assistant", text=answer, context=text)

        log_json(
            LOGGER,
            "api_chat_done",
            model=_runtime.model,
            answer_chars=len(answer),
            thinking_chars=len(thinking),
            stats_keys=len(list(stats.keys())),
        )
        return ChatResponse(
            answer=answer,
            thinking=thinking,
            stats=stats,
            model=_runtime.model,
            parameters=parameters,
            summary=summary_text,
            debug_trace=debug_trace or None,
            memory_debug_snapshot=memory_debug_snapshot or None,
        )


@app.post("/chat/stream")
def chat_stream(req: ChatRequest):
    text = str(req.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text is empty")

    def generate():
        request_text = text
        
        # Проверяем native команды вне lock
        native = _handle_native_chat_command(request_text)
        if native is not None:
            pass_text = str(native.get("pass_text") or "").strip()
            if pass_text:
                request_text = pass_text
            else:
                answer = str(native.get("answer") or "")
                stats = {"served_model": _runtime.model, "native_command": True}
                payload = {
                    "answer": answer,
                    "thinking": "",
                    "stats": stats,
                    "model": _runtime.model,
                    "parameters": None,
                    "summary": None,
                }
                if bool(req.store_turn) and _should_store_metadata(text=text):
                    with _runtime.lock:
                        _append_metadata_row(model=_runtime.model, role="user", text=text, context=answer)
                        _append_metadata_row(model=_runtime.model, role="assistant", text=answer, context=text)
                for chunk in _split_chunks(answer, chunk_size=48):
                    if chunk:
                        yield _ndjson("chunk", chunk)
                yield _ndjson("final", payload)
                return

        # Запускаем worker вне lock, чтобы yield мог отправлять чанки по ходу
        events: queue.Queue[tuple[str, str]] = queue.Queue(maxsize=0)  # Безграничная очередь для минимальной буферизации
        done = threading.Event()
        state: dict[str, Any] = {"result": None, "error": "", "meta": {}}
        sent_answer = 0
        sent_thinking = 0

        def _on_answer(piece: str) -> None:
            chunk = str(piece or "")
            if chunk:
                events.put(("chunk", chunk), block=False)  # Не блокировать, если очередь полна

        def _on_thinking(piece: str) -> None:
            chunk = str(piece or "")
            if chunk:
                events.put(("thinking", chunk), block=False)  # Не блокировать, если очередь полна

        def _on_debug_event(kind: str, payload: dict[str, Any]) -> None:
            """Эмитить memory-debug событие в stream."""
            try:
                events.put((
                    "memory_debug",
                    json.dumps({"kind": kind, "payload": payload}, ensure_ascii=False)
                ), block=False)
            except Exception:
                pass  # Игнорируем ошибки debug events

        def _worker() -> None:
            try:
                log_json(
                    LOGGER,
                    "api_chat_stream_start",
                    model=_runtime.model,
                    text_chars=len(request_text),
                    store_turn=bool(req.store_turn),
                    think=_runtime.thinking_enabled if req.think is None else bool(req.think),
                    verbose=_runtime.verbose_enabled if req.verbose is None else bool(req.verbose),
                    json_mode=_runtime.json_mode_enabled if req.json_mode is None else bool(req.json_mode),
                )
                meta_map = _build_chat_meta(
                    req=req,
                    source="api",
                    stream_on_answer_chunk=_on_answer,
                    stream_on_thinking_chunk=_on_thinking,
                    stream_on_debug_event=_on_debug_event,
                )
                result = _runtime.brain.handle_message(
                    request_text,
                    meta=meta_map,
                )
                state["result"] = result
                state["meta"] = meta_map
            except Exception as exc:
                state["error"] = str(exc)
            finally:
                done.set()

        thread = threading.Thread(target=_worker, name="mmis-api-stream", daemon=True)
        thread.start()

        while not done.is_set() or not events.empty():
            try:
                kind, payload = events.get(timeout=0.001)  # Минимальный timeout для быстрой отправки
            except queue.Empty:
                continue
            if kind == "chunk":
                sent_answer += len(payload)
            elif kind == "thinking":
                sent_thinking += len(payload)
            yield _ndjson(kind, payload)

        thread.join(timeout=0.5)
        err = str(state.get("error") or "").strip()
        if err:
            yield _ndjson("error", err)
            return

        result = state.get("result")
        if result is None:
            yield _ndjson("error", "empty_stream_result")
            return

        answer_raw = str(result.text or "")
        answer, thinking = _split_visible_and_thinking(answer_raw)
        if not thinking.strip():
            thinking = str(getattr(result, "thinking", "") or "").strip()
        structured = dict(getattr(result, "structured_output", {}) or {})
        
        # Debug trace берётся из result
        debug_trace = dict(getattr(result, "debug_trace", {}) or {})
        memory_debug_snapshot = dict(getattr(result, "memory_debug_snapshot", {}) or {})
        _remember_debug_payload(meta_map={"debug_trace": debug_trace, "memory_debug_snapshot": memory_debug_snapshot})
        parameters = structured.get("parameters") if isinstance(structured.get("parameters"), dict) else None
        summary = structured.get("summary")
        summary_text = str(summary).strip() if summary is not None else None
        if summary_text == "":
            summary_text = None
        stats = dict(result.stats or {})
        stats.setdefault("served_model", _runtime.model)
        stats["streaming_requested"] = True
        stats["streaming_live"] = bool(sent_answer or sent_thinking)
        stats["streamed_answer_chars"] = int(sent_answer)
        stats["streamed_thinking_chars"] = int(sent_thinking)
        payload = {
            "answer": answer,
            "thinking": thinking,
            "stats": stats,
            "model": _runtime.model,
            "parameters": parameters,
            "summary": summary_text,
            "debug_trace": debug_trace or None,
            "memory_debug_snapshot": memory_debug_snapshot or None,
        }
        _runtime.last_thinking = thinking

        if bool(req.store_turn) and _should_store_metadata(text=request_text, structured_output=structured):
            _append_metadata_row(model=_runtime.model, role="user", text=request_text, context=answer)
            _append_metadata_row(model=_runtime.model, role="assistant", text=answer, context=request_text)

        # Safety fallback: если streaming не отдал ни одного chunk, но ответ есть
        if not sent_answer and answer:
            # Эмитим один поздний chunk перед final для совместимости UI
            yield _ndjson("chunk", answer)

        log_json(
            LOGGER,
            "api_chat_stream_done",
            model=_runtime.model,
            answer_chars=len(answer),
            thinking_chars=len(thinking),
            streamed_answer_chars=sent_answer,
            streamed_thinking_chars=sent_thinking,
        )
        yield _ndjson("final", payload)

    return StreamingResponse(generate(), media_type="application/x-ndjson")


@app.post("/feedback")
def feedback(req: FeedbackRequest) -> dict:
    with _runtime.lock:
        score = int(req.feedback)
        penalty = float(req.penalty)
        state_mgr = _runtime.brain.state_manager
        character_id = str(state_mgr.get("active_character_id") or "asya").strip().lower() or "asya"
        if score > 0:
            token = "api_feedback_positive"
        elif score < 0:
            token = "api_feedback_negative"
        else:
            token = "api_feedback_neutral"
        feedback_items = [token]
        try:
            state_mgr.dispatch_action(
                {
                    "type": "FEEDBACK_RECEIVED",
                    "character_id": character_id,
                    "feedback": feedback_items,
                    "score": score,
                    "penalty": penalty,
                    "source": "api_feedback",
                }
            )
        except Exception:
            pass
        try:
            if hasattr(state_mgr, "storage") and hasattr(state_mgr.storage, "append_event"):
                state_mgr.storage.append_event(
                    character_id,
                    {
                        "type": "api_feedback",
                        "character_id": character_id,
                        "feedback": feedback_items,
                        "score": score,
                        "penalty": penalty,
                        "user_text": str(req.user_text or "")[:400],
                        "assistant_text": str(req.assistant_text or "")[:400],
                    },
                )
        except Exception:
            pass
        log_json(
            LOGGER,
            "api_feedback",
            character_id=character_id,
            score=score,
            penalty=penalty,
        )
    return {"status": "ok", "character_id": character_id, "feedback": score}


@app.get("/metadata", response_model=MetadataResponse)
def metadata(limit: int = Query(default=50, ge=1, le=1000)) -> MetadataResponse:
    with _runtime.lock:
        items = _read_recent_metadata(model=_runtime.model, limit=int(limit))
        return MetadataResponse(model=_runtime.model, count=len(items), items=items)


@app.get("/metadata/source")
def metadata_source() -> dict:
    with _runtime.lock:
        return {
            "model": _runtime.model,
            "metadata_file": str(_metadata_model_dir(_runtime.model) / "metadata_messages.jsonl"),
            "metadata_json_file": str(_metadata_model_dir(_runtime.model) / "metadata_messages.json"),
        }


def _safe_list_models(provider_obj) -> list[str]:
    try:
        return list(provider_obj.list_models())
    except Exception:
        return []


def _apply_runtime_model(target: str) -> tuple[bool, str, list[str]]:
    model = str(target or "").strip()
    if not model:
        return False, "Model name is empty", _safe_list_models(_runtime.provider)
    models = _safe_list_models(_runtime.provider)
    if models and model not in models:
        return False, f"Model '{model}' is not available", models
    _runtime.model = model
    try:
        if hasattr(_runtime.provider, "default_model"):
            setattr(_runtime.provider, "default_model", model)
    except Exception:
        pass
    if model and model not in models:
        models = [model, *models]
    return True, "", models


def _build_chat_meta(req: ChatRequest, *, source: str, **extra: Any) -> dict[str, Any]:
    active_profile, quality_profile = _resolve_effective_profiles()
    profile = get_profile(active_profile)
    meta: dict[str, Any] = {
        "model": _runtime.model,
        "think": _runtime.thinking_enabled if req.think is None else bool(req.think),
        "verbose": _runtime.verbose_enabled if req.verbose is None else bool(req.verbose),
        "web_mode": str(_runtime.web_mode),
        "json_mode": _runtime.json_mode_enabled if req.json_mode is None else bool(req.json_mode),
        "store_turn": bool(req.store_turn),
        "source": str(source or "api"),
        "quality_profile": str(quality_profile or "BALANCED"),
        "temperature": float(profile.generation.temperature),
        "top_p": float(profile.generation.top_p),
        "repeat_penalty": float(profile.generation.repeat_penalty),
    }
    if profile.generation.max_tokens is not None:
        meta["max_tokens"] = int(profile.generation.max_tokens)
    if profile.generation.stop:
        meta["stop"] = [str(x) for x in list(profile.generation.stop) if str(x)]
    if str(_runtime.provider_name or "").strip().lower() == "ollama":
        meta.update(
            {
                "num_thread": int(profile.ollama.num_thread),
                "num_ctx": int(profile.ollama.num_ctx),
                "num_gpu": int(profile.ollama.num_gpu),
                "num_batch": int(profile.ollama.num_batch),
                "keep_alive": str(profile.ollama.keep_alive),
            }
        )
    if extra:
        meta.update(dict(extra))
    return meta


def _handle_native_chat_command(text: str) -> dict[str, Any] | None:
    src = str(text or "").strip()
    if not src.startswith("/"):
        return None

    parts = src.split(maxsplit=1)
    cmd = str(parts[0] or "").strip().lower()
    arg = str(parts[1] or "").strip() if len(parts) > 1 else ""

    if cmd in {"/help", "/commands"}:
        return {
            "answer": (
                "Native API commands:\n"
                "/health\n/models\n/model [name]\n/think\n/nothink\n/verbose\n/quiet\n/web [query]\n/no-web\n/json\n/nojson\n/character delete <id>\n/help"
            )
        }
    if cmd == "/health":
        active_profile, quality_profile = _resolve_effective_profiles()
        profile = get_profile(active_profile)
        return {
            "answer": (
                f"status: ok\n"
                f"model: {_runtime.model}\n"
                f"thinking: {'on' if _runtime.thinking_enabled else 'off'}\n"
                f"verbose: {'on' if _runtime.verbose_enabled else 'off'}\n"
                f"web_mode: {_runtime.web_mode}\n"
                f"json_mode: {'on' if _runtime.json_mode_enabled else 'off'}\n"
                f"active_profile: {active_profile}\n"
                f"quality_profile: {quality_profile}\n"
                f"temperature: {float(profile.generation.temperature):.3f}\n"
                f"top_p: {float(profile.generation.top_p):.3f}\n"
                f"repeat_penalty: {float(profile.generation.repeat_penalty):.3f}\n"
                f"max_tokens: {int(profile.generation.max_tokens) if profile.generation.max_tokens is not None else 'none'}"
            )
        }
    if cmd == "/models":
        models = _safe_list_models(_runtime.provider)
        current = str(_runtime.model or "").strip()
        if current and current not in models:
            models = [current, *models]
        if not models:
            return {"answer": "No models available."}
        lines = ["Models:"]
        for name in models:
            mark = "*" if current and name == current else " "
            lines.append(f"{mark} {name}")
        return {"answer": "\n".join(lines)}
    if cmd == "/model":
        if not arg:
            return {"answer": f"Current model: {_runtime.model or '—'}"}
        ok, error_text, _models = _apply_runtime_model(arg)
        if not ok:
            return {"answer": error_text or "Model switch failed."}
        return {"answer": f"Model switched to: {_runtime.model}"}
    if cmd == "/think":
        _runtime.thinking_enabled = True
        return {"answer": "Thinking: on"}
    if cmd == "/nothink":
        _runtime.thinking_enabled = False
        return {"answer": "Thinking: off"}
    if cmd == "/verbose":
        _runtime.verbose_enabled = True
        try:
            _runtime.brain.state_manager.patch({"verbose_enabled": True})
        except Exception:
            pass
        return {"answer": "Verbose stats: on"}
    if cmd == "/quiet":
        _runtime.verbose_enabled = False
        try:
            _runtime.brain.state_manager.patch({"verbose_enabled": False})
        except Exception:
            pass
        return {"answer": "Verbose stats: off"}
    if cmd == "/web":
        _runtime.web_mode = "on"
        if arg:
            return {"pass_text": arg}
        return {"answer": "Web: on"}

    if cmd in {"/no-web", "/noweb", "/no_web"}:
        _runtime.web_mode = "off"
        return {"answer": "Web: off"}
    if cmd == "/json":
        _runtime.json_mode_enabled = True
        return {"answer": "JSON mode: on"}
    if cmd == "/nojson":
        _runtime.json_mode_enabled = False
        return {"answer": "JSON mode: off"}
    return None


def _split_chunks(text: str, chunk_size: int = 64) -> list[str]:
    src = str(text or "")
    if not src:
        return [""]
    return [src[i : i + chunk_size] for i in range(0, len(src), max(1, int(chunk_size)))]


_THINK_RE = re.compile(r"<think>(.*?)</think>", flags=re.IGNORECASE | re.DOTALL)
_THINKING_RE = re.compile(r"<thinking>(.*?)</thinking>", flags=re.IGNORECASE | re.DOTALL)
_REASONING_RE = re.compile(r"<reasoning>(.*?)</reasoning>", flags=re.IGNORECASE | re.DOTALL)


def _split_visible_and_thinking(text: str) -> tuple[str, str]:
    src = str(text or "")
    if not src.strip():
        return "", ""

    thoughts: list[str] = []

    def _collect(match: re.Match) -> str:
        block = str(match.group(1) or "").strip()
        if block:
            thoughts.append(block)
        return ""

    visible = _THINK_RE.sub(_collect, src)
    visible = _THINKING_RE.sub(_collect, visible)
    visible = _REASONING_RE.sub(_collect, visible)
    visible = visible.strip()
    thinking = "\n\n".join([x for x in thoughts if x]).strip()
    return visible, thinking


def _ndjson(event: str, data: Any) -> bytes:
    return (json.dumps({"event": event, "data": data}, ensure_ascii=False) + "\n").encode("utf-8")


def _metadata_model_dir(model: str) -> Path:
    slug = _slug_model(model)
    path = _runtime.meta_root / slug
    path.mkdir(parents=True, exist_ok=True)
    return path


def _should_store_metadata(*, text: str, structured_output: dict[str, Any] | None = None) -> bool:
    msg = str(text or "").strip()
    if not msg:
        return False
    if msg.startswith("/"):
        return False
    payload = dict(structured_output or {})
    if "studio_generator" in payload or "studio" in payload:
        return False
    return True


def _append_metadata_row(*, model: str, role: str, text: str, context: str) -> None:
    role_name = str(role or "assistant").strip().lower()
    msg = str(text or "").strip()
    if not msg:
        return

    meta = _extract_api_metadata(text=msg, context=context)
    meta_flat = _flatten_metadata(meta, role_name)

    row = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
        "model": str(model or ""),
        "message_id": _runtime.next_message_id(),
        "role": role_name,
        "text": msg,
        "meta": dict(meta or {}),
        "meta_flat": meta_flat,
    }

    model_dir = _metadata_model_dir(model)
    jsonl_path = model_dir / "metadata_messages.jsonl"
    json_path = model_dir / "metadata_messages.json"

    with jsonl_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

    current = []
    if json_path.exists():
        try:
            payload = json.loads(json_path.read_text(encoding="utf-8-sig"))
            if isinstance(payload, list):
                current = payload
        except Exception:
            current = []
    current.append(row)
    json_path.write_text(json.dumps(current, ensure_ascii=False, indent=2), encoding="utf-8")


def _read_recent_metadata(*, model: str, limit: int) -> list[dict[str, Any]]:
    path = _metadata_model_dir(model) / "metadata_messages.jsonl"
    if not path.exists():
        return []

    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8-sig") as f:
            for line in f:
                raw = str(line or "").strip()
                if not raw:
                    continue
                try:
                    obj = json.loads(raw)
                except Exception:
                    continue
                if isinstance(obj, dict):
                    rows.append(obj)
    except Exception:
        return []

    return rows[-max(1, int(limit)) :]


def _slug_model(value: str) -> str:
    src = str(value or "unknown").strip()
    if not src:
        src = "unknown"
    return src.replace(":", "_").replace("/", "_").replace("\\", "_").replace(" ", "_")


def _extract_api_metadata(*, text: str, context: str) -> dict[str, Any]:
    payload_text = str(text or "").strip()
    if not payload_text:
        return {}
    try:
        state_snapshot = _runtime.brain.state_manager.snapshot()
        history = list(state_snapshot.history or [])
        if context:
            history = [*history, {"role": "context", "content": str(context)}]
        item = _runtime.brain.metadata_extractor.extract(
            text=payload_text,
            state=state_snapshot.raw,
            last_messages=history,
        )
        if hasattr(item, "to_dict"):
            return dict(item.to_dict() or {})
        if isinstance(item, dict):
            return dict(item)
        return {}
    except Exception:
        return {}


def _flatten_metadata(meta: dict[str, Any], role: str) -> dict[str, Any]:
    item = dict(meta or {})
    intent = item.get("intent")
    emotion = item.get("emotion")
    intent_map = dict(intent or {}) if isinstance(intent, dict) else {}
    emotion_map = dict(emotion or {}) if isinstance(emotion, dict) else {}
    return {
        "role": str(role or ""),
        "lang": str(item.get("lang") or ""),
        "lang_conf": float(item.get("lang_conf") or 0.0),
        "intent": str(intent_map.get("label") or ""),
        "intent_conf": float(intent_map.get("conf") or 0.0),
        "emotion": str(emotion_map.get("label") or ""),
        "emotion_intensity": float(emotion_map.get("intensity") or 0.0),
        "tags": [str(x) for x in list(item.get("tags") or []) if str(x).strip()],
    }
