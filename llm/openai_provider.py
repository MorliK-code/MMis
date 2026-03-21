from __future__ import annotations

import json
import time
from typing import Any

from config.settings import load_config
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

try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None


LOGGER = get_logger(__name__)
_cfg = load_config()


def _serialize_tool_arguments(call: ToolCall) -> str:
    raw = str(call.raw_arguments or "").strip()
    if raw:
        return raw
    try:
        return json.dumps(dict(call.arguments or {}), ensure_ascii=False)
    except Exception:
        return "{}"


def _tool_call_to_openai_message_dict(call: ToolCall) -> dict[str, Any]:
    return {
        "id": str(call.id or ""),
        "type": "function",
        "function": {
            "name": str(call.name or ""),
            "arguments": _serialize_tool_arguments(call),
        },
    }


def _tool_calls_from_message_dict(message: dict[str, Any]) -> list[ToolCall]:
    calls: list[ToolCall] = []
    for idx, row in enumerate(list(message.get("tool_calls") or [])):
        if not isinstance(row, dict):
            continue
        fn = dict(row.get("function") or {})
        name = str(fn.get("name") or row.get("name") or "").strip()
        if not name:
            continue
        args_raw = str(fn.get("arguments") or row.get("arguments") or "")
        args: dict[str, Any] = {}
        if args_raw:
            try:
                parsed = json.loads(args_raw)
                if isinstance(parsed, dict):
                    args = parsed
            except Exception:
                args = {}
        calls.append(
            ToolCall(
                id=str(row.get("id") or f"tool_{idx+1}"),
                name=name,
                arguments=args,
                raw_arguments=args_raw,
            )
        )
    return calls


def _message_to_dict(msg: Message) -> dict[str, Any]:
    out = {"role": str(msg.role), "content": str(msg.content or "")}
    if msg.name:
        out["name"] = str(msg.name)
    if msg.tool_call_id and out["role"] == "tool":
        out["tool_call_id"] = str(msg.tool_call_id)
    if msg.tool_calls and out["role"] == "assistant":
        out["tool_calls"] = [_tool_call_to_openai_message_dict(call) for call in list(msg.tool_calls or [])]
    return out


def _toolspec_to_openai(tool: ToolSpec) -> dict[str, Any]:
    schema = dict(tool.input_schema or {})
    return {
        "type": "function",
        "function": {
            "name": str(tool.name),
            "description": str(tool.description or ""),
            "parameters": schema if schema else {"type": "object", "properties": {}},
        },
    }


def _tool_calls_from_openai_message(message) -> list[ToolCall]:
    calls = []
    rows = getattr(message, "tool_calls", None) or []
    for idx, row in enumerate(rows):
        fn = getattr(row, "function", None)
        name = str(getattr(fn, "name", "") or "").strip()
        if not name:
            continue
        args_raw = str(getattr(fn, "arguments", "") or "")
        args = {}
        if args_raw:
            try:
                parsed = json.loads(args_raw)
                if isinstance(parsed, dict):
                    args = parsed
            except Exception:
                args = {}
        calls.append(
            ToolCall(
                id=str(getattr(row, "id", "") or f"tool_{idx+1}"),
                name=name,
                arguments=args,
                raw_arguments=args_raw,
            )
        )
    return calls


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


class OpenAIProvider(LLMProviderBase):
    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout_sec: float | None = None,
        max_retries: int | None = None,
        default_model: str | None = None,
    ):
        self.api_key = str(api_key or _cfg.openai_api_key).strip()
        self.base_url = str(base_url or _cfg.openai_api_url).strip()
        self.timeout_sec = float(timeout_sec if timeout_sec is not None else _cfg.openai_timeout_sec)
        self.max_retries = int(max_retries if max_retries is not None else _cfg.openai_max_retries)
        self.default_model = str(default_model or _cfg.model_name).strip()
        self._client = self._build_client()

    def generate(self, req: LLMRequest) -> LLMResponse:
        if self._client is None:
            raise RuntimeError("OpenAI client is not configured. Install package and set OPENAI_API_KEY.")

        model = str(req.model or self.default_model or "").strip()
        if not model:
            raise RuntimeError("OpenAI model is not configured.")
        verbose = bool(dict(req.metadata or {}).get("verbose", False))

        log_json(
            LOGGER,
            "llm_generate_start",
            provider="openai",
            model=model,
            messages=len(list(req.messages or [])),
            tools=len(list(req.tools or [])),
            json_mode=bool(req.json_mode),
            verbose=verbose,
        )
        kwargs = self._build_completion_kwargs(req=req, model=model, stream=False)
        log_json(
            LOGGER,
            "llm_request_payload",
            provider="openai",
            payload=kwargs,
            **_message_log_fields(list(kwargs.get("messages") or [])),
        )
        t0 = time.perf_counter()
        resp = self._client.chat.completions.create(**kwargs)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        choice = (resp.choices or [None])[0]
        message = getattr(choice, "message", None)
        text = str(getattr(message, "content", "") or "")
        tool_calls = _tool_calls_from_openai_message(message)

        usage_obj = getattr(resp, "usage", None)
        prompt_tokens = int(getattr(usage_obj, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(usage_obj, "completion_tokens", 0) or 0)
        total_tokens = int(getattr(usage_obj, "total_tokens", prompt_tokens + completion_tokens) or 0)

        raw = resp.model_dump() if hasattr(resp, "model_dump") else None
        log_json(
            LOGGER,
            "llm_generate_done",
            provider="openai",
            model=str(getattr(resp, "model", model) or model),
            latency_ms=round(float(latency_ms), 2),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            tool_calls=len(tool_calls),
            text_chars=len(text),
        )
        return LLMResponse(
            text=text,
            tool_calls=tool_calls,
            usage=Usage(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens, total_tokens=total_tokens),
            timings=Timings(latency_ms=latency_ms),
            model=str(getattr(resp, "model", model) or model),
            raw=raw,
        )

    def stream(self, req: LLMRequest):
        if self._client is None:
            raise RuntimeError("OpenAI client is not configured. Install package and set OPENAI_API_KEY.")
        model = str(req.model or self.default_model or "").strip()
        if not model:
            raise RuntimeError("OpenAI model is not configured.")
        verbose = bool(dict(req.metadata or {}).get("verbose", False))

        log_json(
            LOGGER,
            "llm_stream_start",
            provider="openai",
            model=model,
            messages=len(list(req.messages or [])),
            tools=len(list(req.tools or [])),
            json_mode=bool(req.json_mode),
            verbose=verbose,
        )
        kwargs = self._build_completion_kwargs(req=req, model=str(req.model or self.default_model).strip(), stream=True)
        log_json(
            LOGGER,
            "llm_request_payload",
            provider="openai",
            payload=kwargs,
            **_message_log_fields(list(kwargs.get("messages") or [])),
        )
        stream = self._client.chat.completions.create(**kwargs)
        chunk_count = 0
        chars = 0
        for event in stream:
            choice = (getattr(event, "choices", None) or [None])[0]
            delta = getattr(choice, "delta", None)
            text_delta = str(getattr(delta, "content", "") or "")
            tool_calls_delta: list[ToolCall] = []
            rows = getattr(delta, "tool_calls", None) or []
            for row in rows:
                fn = getattr(row, "function", None)
                name = str(getattr(fn, "name", "") or "").strip()
                args_raw = str(getattr(fn, "arguments", "") or "")
                tool_calls_delta.append(
                    ToolCall(
                        id=str(getattr(row, "id", "") or ""),
                        name=name,
                        arguments={},
                        raw_arguments=args_raw,
                    )
                )
            finish_reason = str(getattr(choice, "finish_reason", "") or "")
            chunk_count += 1
            chars += len(text_delta)
            if finish_reason:
                log_json(
                    LOGGER,
                    "llm_stream_done",
                    provider="openai",
                    model=model,
                    chunks=chunk_count,
                    text_chars=chars,
                    finish_reason=finish_reason,
                )
            yield LLMChunk(
                text_delta=text_delta,
                tool_calls_delta=tool_calls_delta,
                usage=Usage(),
                timings=Timings(),
                model=model,
                done=bool(finish_reason),
                raw=(event.model_dump() if hasattr(event, "model_dump") else None),
            )

    def healthcheck(self) -> ProviderHealth:
        if self._client is None:
            return ProviderHealth(ok=False, provider="openai", detail="client not configured", model=self.default_model)
        try:
            _ = self._client.models.list()
            return ProviderHealth(ok=True, provider="openai", detail="ok", model=self.default_model)
        except Exception as exc:
            return ProviderHealth(ok=False, provider="openai", detail=str(exc), model=self.default_model)

    def model_info(self, model: str = "") -> ModelInfo:
        target = str(model or self.default_model or "").strip()
        return ModelInfo(
            provider="openai",
            model=target,
            endpoint=self.base_url,
            capabilities={"stream": True, "tools": True, "json_mode": True},
            params={"timeout_sec": self.timeout_sec, "max_retries": self.max_retries},
        )

    def list_models(self) -> list[str]:
        if self._client is None:
            return []
        try:
            rows = self._client.models.list()
        except Exception:
            return []
        out = []
        for row in list(getattr(rows, "data", []) or []):
            rid = str(getattr(row, "id", "") or "").strip()
            if rid:
                out.append(rid)
        return out

    # Compatibility methods used by existing llm.service wrappers.
    def chat_once(self, *, model: str, messages: list[dict], think: bool) -> dict:
        _ = think
        req = LLMRequest(
            model=model or self.default_model,
            messages=[
                Message(
                    role=str(m.get("role") or "user"),  # type: ignore[arg-type]
                    content=str(m.get("content") or ""),
                    name=str(m.get("name") or ""),
                    tool_call_id=str(m.get("tool_call_id") or ""),
                    tool_calls=_tool_calls_from_message_dict(m),
                )
                for m in messages
            ],
        )
        resp = self.generate(req)
        return {
            "model": resp.model,
            "message": {"content": resp.text, "tool_calls": [_tool_call_to_dict(x) for x in resp.tool_calls]},
            "prompt_eval_count": resp.usage.prompt_tokens,
            "eval_count": resp.usage.completion_tokens,
            "total_duration": int(float(resp.timings.latency_ms) * 1_000_000.0),
            "raw": resp.raw,
        }

    def chat_stream(self, *, model: str, messages: list[dict], think: bool):
        _ = think
        req = LLMRequest(
            model=model or self.default_model,
            messages=[
                Message(
                    role=str(m.get("role") or "user"),  # type: ignore[arg-type]
                    content=str(m.get("content") or ""),
                    name=str(m.get("name") or ""),
                    tool_call_id=str(m.get("tool_call_id") or ""),
                    tool_calls=_tool_calls_from_message_dict(m),
                )
                for m in messages
            ],
        )
        for ch in self.stream(req):
            yield {
                "message": {
                    "content": ch.text_delta,
                    "tool_calls": [_tool_call_to_dict(x) for x in ch.tool_calls_delta],
                },
                "done": ch.done,
                "raw": ch.raw,
            }

    def generate_json(self, *, model: str, prompt: str, max_tokens: int = 360) -> str:
        req = LLMRequest(
            model=model or self.default_model,
            messages=[
                Message(role="system", content="Return valid JSON only."),
                Message(role="user", content=str(prompt or "")),
            ],
            json_mode=True,
            max_tokens=int(max_tokens),
        )
        resp = self.generate(req)
        return str(resp.text or "")

    def _build_client(self):
        if OpenAI is None:
            return None
        if not self.api_key:
            return None
        return OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout_sec,
            max_retries=self.max_retries,
        )

    @staticmethod
    def _build_completion_kwargs(*, req: LLMRequest, model: str, stream: bool) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": [_message_to_dict(m) for m in list(req.messages or [])],
            "stream": bool(stream),
        }
        if req.temperature is not None:
            kwargs["temperature"] = float(req.temperature)
        if req.top_p is not None:
            kwargs["top_p"] = float(req.top_p)
        if req.max_tokens is not None:
            kwargs["max_tokens"] = int(req.max_tokens)
        if req.stop:
            kwargs["stop"] = [str(x) for x in req.stop if str(x)]
        if req.seed is not None:
            kwargs["seed"] = int(req.seed)
        if req.tools:
            kwargs["tools"] = [_toolspec_to_openai(t) for t in list(req.tools or [])]

        if req.json_mode and not req.response_format:
            kwargs["response_format"] = {"type": "json_object"}
        if req.response_format:
            # Keep OpenAI-specific details isolated here.
            kwargs["response_format"] = dict(req.response_format)
        return kwargs


def _tool_call_to_dict(row: ToolCall) -> dict[str, Any]:
    return {
        "id": str(row.id or ""),
        "name": str(row.name or ""),
        "arguments": dict(row.arguments or {}),
        "raw_arguments": str(row.raw_arguments or ""),
    }
