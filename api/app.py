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

from api.schemas import (
    ChatRequest,
    ChatResponse,
    FeedbackRequest,
    HealthResponse,
    JsonModeRequest,
    MetadataResponse,
    ModelSetRequest,
    ModelsResponse,
    ThinkingRequest,
    WebModeRequest,
)
from config.settings import load_config
from core.brain import Brain
from core.spec_registry import validate_no_txt_paths
from llm import build_provider
from utils.logger import get_logger, log_json


cfg = load_config()
validate_no_txt_paths(cfg)
app = FastAPI(title="MMis API", version="2.1.0")
LOGGER = get_logger(__name__)


class _Runtime:
    def __init__(self):
        self.lock = Lock()
        self.provider = build_provider(cfg.llm_default_provider, default_model=cfg.model_name)
        self.brain = Brain(provider=self.provider)
        self.model = str(cfg.model_name or "").strip()
        self.thinking_enabled = bool(cfg.thinking_enabled)
        self.web_mode = cfg.web_mode if bool(cfg.internet_enabled) else "off"
        self.json_mode_enabled = bool(cfg.json_mode_enabled)

        self.meta_root = Path(cfg.memory_dir) / "metadata"
        self.meta_root.mkdir(parents=True, exist_ok=True)
        self._message_seq = self._load_last_message_id()
        self.last_stats: dict[str, Any] = {}
        self.last_thinking: str = ""

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


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    with _runtime.lock:
        return HealthResponse(
            status="ok",
            model=_runtime.model,
            thinking_enabled=bool(_runtime.thinking_enabled),
            json_mode_enabled=bool(_runtime.json_mode_enabled),
            web_mode=str(_runtime.web_mode),
        )


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
        return HealthResponse(
            status="ok",
            model=_runtime.model,
            thinking_enabled=bool(_runtime.thinking_enabled),
            json_mode_enabled=bool(_runtime.json_mode_enabled),
            web_mode=str(_runtime.web_mode),
        )

@app.post("/web-mode", response_model=HealthResponse)
def set_web_mode(req: WebModeRequest) -> HealthResponse:
    mode = str(req.mode or "").strip().lower()
    if mode not in {"auto", "on", "off"}:
        raise HTTPException(status_code=400, detail="mode must be one of: auto, on, off")
    
    with _runtime.lock:
        _runtime.web_mode = mode
        return HealthResponse(
            status="ok",
            model=_runtime.model,
            thinking_enabled=bool(_runtime.thinking_enabled),
            web_mode=str(_runtime.web_mode),
            json_mode_enabled=bool(_runtime.json_mode_enabled),
            )

@app.post("/json-mode", response_model=HealthResponse)
def set_json_mode(req: JsonModeRequest) -> HealthResponse:
    with _runtime.lock:
        _runtime.json_mode_enabled = bool(req.enabled)
        return HealthResponse(
            status="ok",
            model=_runtime.model,
            thinking_enabled=bool(_runtime.thinking_enabled),
            json_mode_enabled=bool(_runtime.json_mode_enabled),
            web_mode=str(_runtime.web_mode),
        )

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    text = str(req.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text is empty")

    with _runtime.lock:
        native = _handle_native_chat_command(text)
        if native is not None:
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
            json_mode=_runtime.json_mode_enabled if req.json_mode is None else bool(req.json_mode),
        )
        result = _runtime.brain.handle_message(
            text,
            meta={
                "model": _runtime.model,
                "think": _runtime.thinking_enabled if req.think is None else bool(req.think),
                "web_mode":str(_runtime.web_mode),
                "json_mode": _runtime.json_mode_enabled if req.json_mode is None else bool(req.json_mode),
                "store_turn": bool(req.store_turn),
                "source": "api",
            },
        )

        answer_raw = str(result.text or "")
        answer, thinking = _split_visible_and_thinking(answer_raw)
        if not thinking.strip():
            thinking = str(getattr(result, "thinking", "") or "").strip()
        structured = dict(getattr(result, "structured_output", {}) or {})
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
        )


@app.post("/chat/stream")
def chat_stream(req: ChatRequest):
    text = str(req.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text is empty")

    def generate():
        with _runtime.lock:
            native = _handle_native_chat_command(text)
            if native is not None:
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
                    _append_metadata_row(model=_runtime.model, role="user", text=text, context=answer)
                    _append_metadata_row(model=_runtime.model, role="assistant", text=answer, context=text)
                for chunk in _split_chunks(answer, chunk_size=48):
                    if chunk:
                        yield _ndjson("chunk", chunk)
                yield _ndjson("final", payload)
                return

        events: queue.Queue[tuple[str, str]] = queue.Queue()
        done = threading.Event()
        state: dict[str, Any] = {"result": None, "error": ""}
        sent_answer = 0
        sent_thinking = 0

        def _on_answer(piece: str) -> None:
            chunk = str(piece or "")
            if chunk:
                events.put(("chunk", chunk))

        def _on_thinking(piece: str) -> None:
            chunk = str(piece or "")
            if chunk:
                events.put(("thinking", chunk))

        def _worker() -> None:
            try:
                with _runtime.lock:
                    log_json(
                        LOGGER,
                        "api_chat_stream_start",
                        model=_runtime.model,
                        text_chars=len(text),
                        store_turn=bool(req.store_turn),
                        think=_runtime.thinking_enabled if req.think is None else bool(req.think),
                        json_mode=_runtime.json_mode_enabled if req.json_mode is None else bool(req.json_mode),
                    )
                    result = _runtime.brain.handle_message(
                        text,
                        meta={
                            "model": _runtime.model,
                            "think": _runtime.thinking_enabled if req.think is None else bool(req.think),
                            "web_mode":str(_runtime.web_mode),
                            "json_mode": _runtime.json_mode_enabled if req.json_mode is None else bool(req.json_mode),
                            "store_turn": bool(req.store_turn),
                            "source": "api",
                            "stream_on_answer_chunk": _on_answer,
                            "stream_on_thinking_chunk": _on_thinking,
                        },
                    )
                    state["result"] = result
            except Exception as exc:
                state["error"] = str(exc)
            finally:
                done.set()

        thread = threading.Thread(target=_worker, name="mmis-api-stream", daemon=True)
        thread.start()

        while not done.is_set() or not events.empty():
            try:
                kind, payload = events.get(timeout=0.2)
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
        parameters = structured.get("parameters") if isinstance(structured.get("parameters"), dict) else None
        summary = structured.get("summary")
        summary_text = str(summary).strip() if summary is not None else None
        if summary_text == "":
            summary_text = None
        stats = dict(result.stats or {})
        stats.setdefault("served_model", _runtime.model)
        payload = {
            "answer": answer,
            "thinking": thinking,
            "stats": stats,
            "model": _runtime.model,
            "parameters": parameters,
            "summary": summary_text,
        }
        _runtime.last_thinking = thinking

        if bool(req.store_turn) and _should_store_metadata(text=text, structured_output=structured):
            _append_metadata_row(model=_runtime.model, role="user", text=text, context=answer)
            _append_metadata_row(model=_runtime.model, role="assistant", text=answer, context=text)

        if sent_thinking == 0:
            for t_chunk in _split_chunks(thinking, chunk_size=48):
                if t_chunk:
                    yield _ndjson("thinking", t_chunk)
        if sent_answer == 0:
            for chunk in _split_chunks(answer, chunk_size=48):
                if chunk:
                    yield _ndjson("chunk", chunk)

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
                "/health\n/models\n/model [name]\n/think\n/nothink\n/json\n/nojson\n/character delete <id>\n/help"
            )
        }
    if cmd == "/health":
        return {
            "answer": (
                f"status: ok\n"
                f"model: {_runtime.model}\n"
                f"thinking: {'on' if _runtime.thinking_enabled else 'off'}\n"
                f"web_mode: {_runtime.web_mode}\n"
                f"json_mode: {'on' if _runtime.json_mode_enabled else 'off'}"
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
    if cmd == "/web":
        _runtime.web_mode = "on"
        if arg:
            return {"pass_text": arg}
        return {"answer": "Web: on"}

    if cmd in {"/no-web", "/noweb", "/no_web"}:
        _runtime.web_mode = "off"
        return {"answer": "Web: off"}

    if cmd in {"/web-auto", "/web_auto", "/autoweb"}:
        _runtime.web_mode = "auto"
        return {"answer": "Web: auto"}
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
