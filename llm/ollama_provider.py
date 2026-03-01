from __future__ import annotations

import json
import os
import time
from typing import Any

import ollama

from llm.provider_base import (
    LLMChunk,
    LLMProviderBase,
    LLMRequest,
    LLMResponse,
    Message,
    ModelInfo,
    ProviderHealth,
    Timings,
    ToolCall,
    ToolSpec,
    Usage,
)
from utils.logger import get_logger, log_json


LOGGER = get_logger(__name__)


def _env_str(name: str, default: str) -> str:
    return str(os.getenv(name, default)).strip()


def _env_int(name: str, default: int, minimum: int = 0) -> int:
    raw = os.getenv(name)
    if raw is None:
        return int(default)
    try:
        value = int(str(raw).strip())
    except Exception:
        return int(default)
    return max(minimum, value)


def _env_float(name: str, default: float, minimum: float = 0.0) -> float:
    raw = os.getenv(name)
    if raw is None:
        return float(default)
    try:
        value = float(str(raw).strip())
    except Exception:
        return float(default)
    return float(value if value >= minimum else default)


def _as_dict(obj) -> dict:
    if isinstance(obj, dict):
        return obj
    if obj is None:
        return {}
    dump = getattr(obj, "model_dump", None)
    if callable(dump):
        try:
            value = dump()
            if isinstance(value, dict):
                return value
        except Exception:
            pass
    as_dict = getattr(obj, "dict", None)
    if callable(as_dict):
        try:
            value = as_dict()
            if isinstance(value, dict):
                return value
        except Exception:
            pass
    return {}


def _message_to_dict(msg: Message) -> dict[str, Any]:
    out = {"role": str(msg.role), "content": str(msg.content or "")}
    if msg.name:
        out["name"] = str(msg.name)
    if msg.tool_call_id:
        out["tool_call_id"] = str(msg.tool_call_id)
    return out


def _toolspec_to_ollama(tool: ToolSpec) -> dict[str, Any]:
    schema = dict(tool.input_schema or {})
    return {
        "type": "function",
        "function": {
            "name": str(tool.name),
            "description": str(tool.description or ""),
            "parameters": schema if schema else {"type": "object", "properties": {}},
        },
    }


def _parse_tool_calls_from_message(message: dict[str, Any], text_fallback: str = "") -> list[ToolCall]:
    out: list[ToolCall] = []
    for idx, row in enumerate(list(message.get("tool_calls") or [])):
        if not isinstance(row, dict):
            continue
        fn = dict(row.get("function") or {})
        name = str(fn.get("name") or row.get("name") or "").strip()
        if not name:
            continue
        args_raw = fn.get("arguments")
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
                id=str(row.get("id") or f"tool_{idx+1}"),
                name=name,
                arguments=args,
                raw_arguments=raw_text,
            )
        )
    if out:
        return out
    return _extract_tool_calls_from_text(text_fallback)


def _extract_tool_calls_from_text(text: str) -> list[ToolCall]:
    raw = str(text or "").strip()
    if not raw:
        return []
    try:
        payload = json.loads(raw)
    except Exception:
        return []

    if isinstance(payload, dict) and payload.get("tool"):
        return [
            ToolCall(
                id=str(payload.get("id") or "tool_1"),
                name=str(payload.get("tool") or ""),
                arguments=dict(payload.get("args") or {}),
                raw_arguments=json.dumps(payload.get("args") or {}, ensure_ascii=False),
            )
        ]
    if isinstance(payload, list):
        out: list[ToolCall] = []
        for idx, row in enumerate(payload):
            if not isinstance(row, dict) or not row.get("tool"):
                continue
            out.append(
                ToolCall(
                    id=str(row.get("id") or f"tool_{idx+1}"),
                    name=str(row.get("tool") or ""),
                    arguments=dict(row.get("args") or {}),
                    raw_arguments=json.dumps(row.get("args") or {}, ensure_ascii=False),
                )
            )
        return out
    return []


def _extract_thinking(message: dict[str, Any], payload: dict[str, Any] | None = None) -> str:
    msg = dict(message or {})
    root = dict(payload or {})
    for key in ("thinking", "thought", "reasoning"):
        text = _as_text(msg.get(key))
        if text:
            return text
    for key in ("thinking", "thought", "reasoning"):
        text = _as_text(root.get(key))
        if text:
            return text
    return ""


<<<<<<< HEAD
def _message_log_fields(messages: list[dict[str, Any]]) -> dict[str, Any]:
    rows = list(messages or [])
    roles = [str(m.get("role") or "") for m in rows]
    order = [f"{idx}:{role}" for idx, role in enumerate(roles)]
    system_content = ""
    for row in rows:
        if str(row.get("role") or "").strip().lower() == "system":
            system_content = str(row.get("content") or "")
            break
    return {
        "messages": rows,
        "message_roles": roles,
        "message_order": order,
        "system_message": system_content,
    }


=======
>>>>>>> 51b8456b510b4061cb471a6e7b7574d205e99e04
def _wrap_thinking(text: str, thinking: str, *, trim: bool = True) -> str:
    visible = str(text or "")
    think = str(thinking or "")
    if trim:
        think = think.strip()
    if not think:
        return visible
    low = visible.lower()
    if "<think>" in low or "<thinking>" in low or "<reasoning>" in low:
        return visible
    if not visible.strip():
        return f"<think>{think}</think>"
<<<<<<< HEAD
    return f"<think>{think}</think>\n{visible}"
=======
    return f"{visible}\n<think>{think}</think>"
>>>>>>> 51b8456b510b4061cb471a6e7b7574d205e99e04


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for key in ("text", "content", "value", "reasoning", "thinking"):
            if key in value:
                inner = _as_text(value.get(key))
                if inner:
                    return inner
        try:
            return json.dumps(value, ensure_ascii=False)
        except Exception:
            return ""
    if isinstance(value, list):
        parts = []
        for row in value:
            text = _as_text(row)
            if text:
                parts.append(text)
        return "\n".join(parts).strip()
    return str(value).strip()


def _stitch_thinking_delta(prev_char: str, delta: str) -> tuple[str, str]:
    text = str(delta or "")
    if not text:
        return "", str(prev_char or "")

    next_prev = str(prev_char or "")
    for ch in reversed(text):
        if not ch.isspace():
            next_prev = ch
            break
    return text, next_prev


class OllamaProvider(LLMProviderBase):
    def __init__(
        self,
        *,
        host: str | None = None,
        timeout_sec: float | None = None,
        retries: int | None = None,
        default_model: str | None = None,
        debug_raw: bool | None = None,
    ):
        self.host = str(host or _env_str("OLLAMA_HOST", "http://127.0.0.1:11434")).strip()
        self.timeout_sec = float(timeout_sec if timeout_sec is not None else _env_float("OLLAMA_TIMEOUT_SEC", 120.0, 0.1))
        self.retries = int(retries if retries is not None else _env_int("OLLAMA_RETRIES", 1, 0))
        self.default_model = str(default_model or _env_str("MMIS_MODEL_NAME", "")).strip()
        self.debug_raw = bool(
            _env_int("MMIS_DEBUG_RAW_LLM", 0, 0) if debug_raw is None else debug_raw
        )
        self._client = ollama.Client(host=self.host, timeout=self.timeout_sec)

    def generate(self, req: LLMRequest) -> LLMResponse:
        model = str(req.model or self.default_model or "").strip()
        if not model:
            raise RuntimeError("Ollama model is not configured.")

        log_json(
            LOGGER,
            "llm_generate_start",
            provider="ollama",
            model=model,
            messages=len(list(req.messages or [])),
            tools=len(list(req.tools or [])),
            json_mode=bool(req.json_mode),
            think=bool(req.metadata.get("think", False)),
        )
        payload = self._chat_with_retry(req=req, model=model, stream=False)
        msg = dict(payload.get("message") or {})
        text = str(msg.get("content") or "")
        thinking = _extract_thinking(msg, payload)
        tool_calls = _parse_tool_calls_from_message(msg, text_fallback=text)
        usage = self._extract_usage(payload)
        timings = self._extract_timings(payload)
        log_json(
            LOGGER,
            "llm_generate_done",
            provider="ollama",
            model=str(payload.get("model") or model),
            latency_ms=round(float(timings.latency_ms or 0.0), 2),
            prompt_tokens=int(usage.prompt_tokens or 0),
            completion_tokens=int(usage.completion_tokens or 0),
            total_tokens=int(usage.total_tokens or 0),
            tool_calls=len(tool_calls),
            text_chars=len(text),
            thinking_chars=len(thinking),
        )

        return LLMResponse(
            text=text,
            tool_calls=tool_calls,
            thinking=thinking,
            usage=usage,
            timings=timings,
            model=str(payload.get("model") or model),
            raw=(payload if self.debug_raw else None),
        )

    def stream(self, req: LLMRequest):
        model = str(req.model or self.default_model or "").strip()
        if not model:
            raise RuntimeError("Ollama model is not configured.")

        log_json(
            LOGGER,
            "llm_stream_start",
            provider="ollama",
            model=model,
            messages=len(list(req.messages or [])),
            tools=len(list(req.tools or [])),
            json_mode=bool(req.json_mode),
            think=bool(req.metadata.get("think", False)),
        )
        stream = self._chat_with_retry(req=req, model=model, stream=True)
        chunk_count = 0
        chars = 0
        prev_thinking_char = ""
        for raw_chunk in stream:
            chunk = _as_dict(raw_chunk)
            msg = dict(chunk.get("message") or {})
            text_delta = str(msg.get("content") or "")
            thinking_delta = _extract_thinking(msg, chunk)
            thinking_delta, prev_thinking_char = _stitch_thinking_delta(prev_thinking_char, thinking_delta)
            tool_calls_delta = _parse_tool_calls_from_message(msg, text_fallback=text_delta)
            done = bool(chunk.get("done", False))
            chunk_count += 1
            chars += len(text_delta)
            if done:
                log_json(
                    LOGGER,
                    "llm_stream_done",
                    provider="ollama",
                    model=str(chunk.get("model") or model),
                    chunks=chunk_count,
                    text_chars=chars,
                )
            yield LLMChunk(
                text_delta=text_delta,
                thinking_delta=thinking_delta,
                tool_calls_delta=tool_calls_delta,
                done=done,
                raw=(chunk if self.debug_raw else None),
            )

    def healthcheck(self) -> ProviderHealth:
        try:
            payload = _as_dict(self._client.list())
            rows = list(payload.get("models") or [])
            current = self.default_model
            if not current and rows:
                head = rows[0]
                if isinstance(head, dict):
                    current = str(head.get("name") or head.get("model") or head.get("id") or "").strip()
            return ProviderHealth(ok=True, provider="ollama", detail="ok", model=current)
        except Exception as exc:
            return ProviderHealth(ok=False, provider="ollama", detail=str(exc), model=self.default_model)

    def model_info(self, model: str = "") -> ModelInfo:
        target = str(model or self.default_model or "").strip()
        params = {
            "timeout_sec": self.timeout_sec,
            "retries": self.retries,
        }
        capabilities = {
            "stream": True,
            "tools": True,
            "json_mode": True,
        }
        return ModelInfo(
            provider="ollama",
            model=target,
            endpoint=self.host,
            capabilities=capabilities,
            params=params,
        )

    def list_models(self) -> list[str]:
        try:
            payload = _as_dict(self._client.list())
            rows = list(payload.get("models") or [])
        except Exception:
            return []
        out: list[str] = []
        seen: set[str] = set()
        for row in rows:
            if isinstance(row, dict):
                name = str(row.get("name") or row.get("model") or row.get("id") or "").strip()
            else:
                name = str(getattr(row, "name", "") or getattr(row, "model", "") or getattr(row, "id", "")).strip()
            if not name or name in seen:
                continue
            seen.add(name)
            out.append(name)
        return out

    # Compatibility methods used by existing llm.service.
    def chat_once(self, *, model: str, messages: list[dict], think: bool) -> dict:
        req = LLMRequest(
            model=model,
            messages=[
                Message(
                    role=str(m.get("role") or "user"),  # type: ignore[arg-type]
                    content=str(m.get("content") or ""),
                    name=str(m.get("name") or ""),
                    tool_call_id=str(m.get("tool_call_id") or ""),
                )
                for m in messages
            ],
            metadata={"think": bool(think)},
        )
        payload = self._chat_with_retry(req=req, model=model, stream=False)
        return payload

    def chat_stream(self, *, model: str, messages: list[dict], think: bool):
        req = LLMRequest(
            model=model,
            messages=[
                Message(
                    role=str(m.get("role") or "user"),  # type: ignore[arg-type]
                    content=str(m.get("content") or ""),
                    name=str(m.get("name") or ""),
                    tool_call_id=str(m.get("tool_call_id") or ""),
                )
                for m in messages
            ],
            metadata={"think": bool(think)},
        )
        return self._chat_with_retry(req=req, model=model, stream=True)

    def generate_json(self, *, model: str, prompt: str, max_tokens: int = 360) -> str:
        payload = _as_dict(
            self._client.generate(
                model=model,
                prompt=str(prompt or ""),
                options={"temperature": 0.1, "num_predict": int(max_tokens)},
                keep_alive="5s",
            )
        )
        return str(payload.get("response") or "")

    def _chat_with_retry(self, *, req: LLMRequest, model: str, stream: bool):
        attempts = max(1, int(self.retries) + 1)
        last_exc: Exception | None = None
        for attempt in range(attempts):
            try:
                return self._chat_once(req=req, model=model, stream=stream)
            except Exception as exc:
                last_exc = exc
                LOGGER.warning(
                    "ollama request failed attempt=%s/%s model=%s stream=%s error=%s",
                    attempt + 1,
                    attempts,
                    model,
                    bool(stream),
                    exc,
                )
                if attempt >= attempts - 1:
                    break
                time.sleep(0.25 * (attempt + 1))
        raise RuntimeError(str(last_exc or "Ollama request failed"))

    def _chat_once(self, *, req: LLMRequest, model: str, stream: bool):
        messages = [_message_to_dict(m) for m in list(req.messages or [])]
        options = self._build_options(req)
        tools = [_tools for _tools in [_toolspec_to_ollama(t) for t in list(req.tools or [])] if _tools]

        think = req.metadata.get("think")
        keep_alive = req.metadata.get("keep_alive")
        fmt: str | dict[str, Any] | None = None
        if req.json_mode:
            fmt = "json"
        if isinstance(req.response_format, dict) and req.response_format:
            fmt = dict(req.response_format)

        request_payload = {
            "model": model,
            "messages": messages,
            "stream": bool(stream),
            "think": think,
            "tools": (tools or None),
            "format": fmt,
            "options": options or None,
            "keep_alive": keep_alive,
        }
        log_json(
            LOGGER,
            "llm_request_payload",
            provider="ollama",
            payload=request_payload,
            **_message_log_fields(messages),
        )
        payload = self._client.chat(**request_payload)
        if stream:
            return payload
        return _as_dict(payload)

    @staticmethod
    def _build_options(req: LLMRequest) -> dict[str, Any]:
        options: dict[str, Any] = {}
        if req.temperature is not None:
            options["temperature"] = float(req.temperature)
        if req.top_p is not None:
            options["top_p"] = float(req.top_p)
        if req.repeat_penalty is not None:
            options["repeat_penalty"] = float(req.repeat_penalty)
        if req.seed is not None:
            options["seed"] = int(req.seed)
        if req.max_tokens is not None:
            options["num_predict"] = int(req.max_tokens)
        if req.stop:
            options["stop"] = [str(x) for x in req.stop if str(x)]

        meta = dict(req.metadata or {})
        for key in ("num_ctx", "num_thread", "num_gpu", "num_batch"):
            if key in meta and meta.get(key) is not None:
                try:
                    options[key] = int(meta.get(key))
                except Exception:
                    continue
        return options

    @staticmethod
    def _extract_usage(payload: dict[str, Any]) -> Usage:
        prompt_tokens = int(payload.get("prompt_eval_count") or 0)
        completion_tokens = int(payload.get("eval_count") or 0)
        total = prompt_tokens + completion_tokens
        return Usage(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens, total_tokens=total)

    @staticmethod
    def _extract_timings(payload: dict[str, Any]) -> Timings:
        if payload.get("total_duration") is not None:
            try:
                latency_ms = float(payload.get("total_duration")) / 1_000_000.0
                return Timings(latency_ms=latency_ms)
            except Exception:
                pass
        return Timings(latency_ms=0.0)
