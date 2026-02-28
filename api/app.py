from __future__ import annotations

import json
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
    MetadataResponse,
    ModelSetRequest,
    ModelsResponse,
    ThinkingRequest,
)
from config.settings import load_config
from core.brain import Brain
from llm import build_provider


cfg = load_config()
app = FastAPI(title="MMis API", version="2.1.0")


class _Runtime:
    def __init__(self):
        self.lock = Lock()
        self.provider = build_provider(cfg.llm_default_provider, default_model=cfg.model_name)
        self.brain = Brain(provider=self.provider)
        self.model = str(cfg.model_name or "").strip()
        self.thinking_enabled = bool(cfg.thinking_enabled)

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
        models = _safe_list_models(_runtime.provider)
        if models and target not in models:
            raise HTTPException(status_code=404, detail=f"Model '{target}' is not available")

        _runtime.model = target
        try:
            if hasattr(_runtime.provider, "default_model"):
                setattr(_runtime.provider, "default_model", target)
        except Exception:
            pass

        return ModelsResponse(runtime_model=_runtime.model, models=models)


@app.post("/thinking", response_model=HealthResponse)
def set_thinking(req: ThinkingRequest) -> HealthResponse:
    with _runtime.lock:
        _runtime.thinking_enabled = bool(req.enabled)
        return HealthResponse(status="ok", model=_runtime.model, thinking_enabled=_runtime.thinking_enabled)


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    text = str(req.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text is empty")

    with _runtime.lock:
        result = _runtime.brain.handle_message(
            text,
            meta={
                "model": _runtime.model,
                "think": _runtime.thinking_enabled if req.think is None else bool(req.think),
                "store_turn": bool(req.store_turn),
                "source": "api",
            },
        )

        answer = str(result.text or "")
        stats = dict(result.stats or {})
        stats.setdefault("served_model", _runtime.model)
        _runtime.last_stats = stats
        _runtime.last_thinking = ""

        if bool(req.store_turn):
            _append_metadata_row(model=_runtime.model, role="user", text=text, context=answer)
            _append_metadata_row(model=_runtime.model, role="assistant", text=answer, context=text)

        return ChatResponse(answer=answer, thinking="", stats=stats, model=_runtime.model)


@app.post("/chat/stream")
def chat_stream(req: ChatRequest):
    text = str(req.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text is empty")

    def generate():
        with _runtime.lock:
            result = _runtime.brain.handle_message(
                text,
                meta={
                    "model": _runtime.model,
                    "think": _runtime.thinking_enabled if req.think is None else bool(req.think),
                    "store_turn": bool(req.store_turn),
                    "source": "api",
                },
            )
            answer = str(result.text or "")
            stats = dict(result.stats or {})
            stats.setdefault("served_model", _runtime.model)
            payload = {"answer": answer, "thinking": "", "stats": stats, "model": _runtime.model}

            if bool(req.store_turn):
                _append_metadata_row(model=_runtime.model, role="user", text=text, context=answer)
                _append_metadata_row(model=_runtime.model, role="assistant", text=answer, context=text)

        for chunk in _split_chunks(answer, chunk_size=48):
            yield _ndjson("chunk", chunk)
        yield _ndjson("final", payload)

    return StreamingResponse(generate(), media_type="application/x-ndjson")


@app.post("/feedback")
def feedback(req: FeedbackRequest) -> dict:
    _ = req
    return {"status": "ok"}


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


def _split_chunks(text: str, chunk_size: int = 64) -> list[str]:
    src = str(text or "")
    if not src:
        return [""]
    return [src[i : i + chunk_size] for i in range(0, len(src), max(1, int(chunk_size)))]


def _ndjson(event: str, data: Any) -> bytes:
    return (json.dumps({"event": event, "data": data}, ensure_ascii=False) + "\n").encode("utf-8")


def _metadata_model_dir(model: str) -> Path:
    slug = _slug_model(model)
    path = _runtime.meta_root / slug
    path.mkdir(parents=True, exist_ok=True)
    return path


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
