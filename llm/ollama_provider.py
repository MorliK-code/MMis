from __future__ import annotations

import json
import threading
import time
from dataclasses import replace
from typing import Any

import ollama

from config.settings import load_config
from llm.priority_manager import get_priority_manager, LLMPriorityManager
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
_cfg = load_config()

# Глобальный флаг для прерывания Memory LLM
# Устанавливается в True когда основная модель хочет ответить
_memory_llm_interrupt = threading.Event()


def _serialize_tool_arguments(call: ToolCall) -> Any:
    if call.arguments:
        return dict(call.arguments or {})
    raw = str(call.raw_arguments or "").strip()
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except Exception:
        return raw
    return parsed if isinstance(parsed, (dict, list, str, int, float, bool)) or parsed is None else raw


def _tool_call_to_ollama_message_dict(call: ToolCall) -> dict[str, Any]:
    return {
        "id": str(call.id or ""),
        "type": "function",
        "function": {
            "name": str(call.name or ""),
            "arguments": _serialize_tool_arguments(call),
        },
    }


def _tool_calls_from_message_dict(message: dict[str, Any]) -> list[ToolCall]:
    out: list[ToolCall] = []
    for idx, row in enumerate(list(message.get("tool_calls") or [])):
        if not isinstance(row, dict):
            continue
        fn = dict(row.get("function") or {})
        name = str(fn.get("name") or row.get("name") or "").strip()
        if not name:
            continue
        args_raw = fn.get("arguments", row.get("arguments"))
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
    return out


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
    if msg.tool_call_id and out["role"] == "tool":
        out["tool_call_id"] = str(msg.tool_call_id)
    if msg.tool_calls and out["role"] == "assistant":
        out["tool_calls"] = [_tool_call_to_ollama_message_dict(call) for call in list(msg.tool_calls or [])]
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
    return f"<think>{think}</think>\n{visible}"


def _request_with_thinking_disabled(req: LLMRequest) -> LLMRequest:
    metadata = dict(req.metadata or {})
    metadata["think"] = False
    metadata["think_retry_without_reasoning"] = True
    return replace(req, metadata=metadata)


def _should_retry_without_thinking(
    *,
    req: LLMRequest,
    text: str,
    thinking: str,
    tool_calls: list[ToolCall],
) -> bool:
    if not bool(dict(req.metadata or {}).get("think", False)):
        return False
    if str(text or "").strip():
        return False
    if list(tool_calls or []):
        return False
    return bool(str(thinking or "").strip())


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


class OllamaProvider(LLMProviderBase):
    # Reasoning модели требуют больше времени
    REASONING_MODEL_PATTERNS = ["deepseek-r1", "deepseek-reasoner", "o1", "o3", "thinking"]

    def __init__(
        self,
        *,
        host: str | None = None,
        timeout_sec: float | None = None,
        retries: int | None = None,
        default_model: str | None = None,
        debug_raw: bool | None = None,
    ):
        self.host = str(host or _cfg.ollama_base_url).strip()
        self.default_model = str(default_model or _cfg.model_name).strip()
        
        # Автоматически увеличиваем timeout для reasoning моделей
        base_timeout = float(timeout_sec if timeout_sec is not None else _cfg.ollama_timeout_sec)
        if self._is_reasoning_model(self.default_model):
            self.timeout_sec = max(base_timeout, 600.0)  # 10 минут для reasoning
        else:
            self.timeout_sec = base_timeout
            
        self.retries = int(retries if retries is not None else _cfg.ollama_retries)
        self.debug_raw = bool(debug_raw if debug_raw is not None else False)
        self._client = ollama.Client(host=self.host, timeout=self.timeout_sec)
        self._current_model = self.default_model

    @staticmethod
    def _is_reasoning_model(model: str) -> bool:
        """Проверка, является ли модель reasoning моделью."""
        model_lower = str(model or "").lower()
        return any(pattern in model_lower for pattern in OllamaProvider.REASONING_MODEL_PATTERNS)

    def _remember_model(self, model: str) -> None:
        target = str(model or "").strip()
        if target:
            self._current_model = target

    def generate(self, req: LLMRequest) -> LLMResponse:
        model = str(req.model or self.default_model or "").strip()
        if not model:
            raise RuntimeError("Ollama model is not configured.")
        self._remember_model(model)
        verbose = bool(dict(req.metadata or {}).get("verbose", False))
        
        # Определяем приоритет LLM
        is_memory_llm = "memory" in str(req.metadata.get("source", "")).lower()
        priority = LLMPriorityManager.PRIORITY_MEMORY if is_memory_llm else LLMPriorityManager.PRIORITY_MAIN
        
        # Запрашиваем доступ с учётом приоритета
        priority_mgr = get_priority_manager()
        if not priority_mgr.wait_for_turn(priority, timeout=300.0):
            raise TimeoutError(f"LLM {model} timed out waiting for higher priority task")
        
        try:
            log_json(
                LOGGER,
                "llm_generate_start",
                provider="ollama",
                model=model,
                messages=len(list(req.messages or [])),
                tools=len(list(req.tools or [])),
                json_mode=bool(req.json_mode),
                think=bool(req.metadata.get("think", False)),
                verbose=verbose,
                priority="memory" if is_memory_llm else "main",
            )
            payload = self._chat_with_retry(req=req, model=model, stream=False)
            msg = dict(payload.get("message") or {})
            text = str(msg.get("content") or "")
            thinking = _extract_thinking(msg, payload)
            tool_calls = _parse_tool_calls_from_message(msg, text_fallback=text)
            if _should_retry_without_thinking(req=req, text=text, thinking=thinking, tool_calls=tool_calls):
                LOGGER.warning(
                    "ollama generate finished with thinking but no visible answer; retrying without think model=%s",
                    model,
                )
                retry_payload = self._chat_with_retry(
                    req=_request_with_thinking_disabled(req),
                    model=model,
                    stream=False,
                )
                retry_msg = dict(retry_payload.get("message") or {})
                retry_text = str(retry_msg.get("content") or "")
                retry_tool_calls = _parse_tool_calls_from_message(retry_msg, text_fallback=retry_text)
                if str(retry_text or "").strip() or retry_tool_calls:
                    payload = retry_payload
                    msg = retry_msg
                    text = retry_text
                    tool_calls = retry_tool_calls
                else:
                    LOGGER.warning(
                        "ollama retry without think still returned no visible answer model=%s",
                        model,
                    )
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
                raw=(payload if (self.debug_raw or verbose) else None),
            )
        finally:
            priority_mgr.release(priority)

    def stream(self, req: LLMRequest):
        model = str(req.model or self.default_model or "").strip()
        if not model:
            raise RuntimeError("Ollama model is not configured.")
        self._remember_model(model)
        verbose = bool(dict(req.metadata or {}).get("verbose", False))
        is_memory_llm = "memory" in str(req.metadata.get("source", "")).lower()
        priority = LLMPriorityManager.PRIORITY_MEMORY if is_memory_llm else LLMPriorityManager.PRIORITY_MAIN
        priority_mgr = get_priority_manager()
        if not priority_mgr.wait_for_turn(priority, timeout=300.0):
            raise TimeoutError(f"LLM {model} timed out waiting for higher priority task")

        try:
            log_json(
                LOGGER,
                "llm_stream_start",
                provider="ollama",
                model=model,
                messages=len(list(req.messages or [])),
                tools=len(list(req.tools or [])),
                json_mode=bool(req.json_mode),
                think=bool(req.metadata.get("think", False)),
                verbose=verbose,
                priority="memory" if is_memory_llm else "main",
            )
            stream = self._chat_with_retry(req=req, model=model, stream=True)
            chunk_count = 0
            chars = 0
            streamed_thinking_parts: list[str] = []
            streamed_text_parts: list[str] = []
            streamed_tool_calls: list[ToolCall] = []
            try:
                for raw_chunk in stream:
                    chunk = _as_dict(raw_chunk)
                    msg = dict(chunk.get("message") or {})
                    text_delta = str(msg.get("content") or "")
                    thinking_delta = _extract_thinking(msg, chunk)
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
                    usage = self._extract_usage(chunk) if done else Usage()
                    timings = self._extract_timings(chunk) if done else Timings()
                    chunk_obj = LLMChunk(
                        text_delta=text_delta,
                        thinking_delta=thinking_delta,
                        tool_calls_delta=tool_calls_delta,
                        usage=usage,
                        timings=timings,
                        model=str(chunk.get("model") or model),
                        done=done,
                        raw=(chunk if (self.debug_raw or (verbose and done)) else None),
                    )
                    if text_delta:
                        streamed_text_parts.append(text_delta)
                    if thinking_delta:
                        streamed_thinking_parts.append(thinking_delta)
                    if tool_calls_delta:
                        streamed_tool_calls.extend(list(tool_calls_delta))
                    if done and _should_retry_without_thinking(
                        req=req,
                        text="".join(streamed_text_parts),
                        thinking="".join(streamed_thinking_parts),
                        tool_calls=streamed_tool_calls,
                    ):
                        if text_delta or thinking_delta or tool_calls_delta:
                            yield replace(
                                chunk_obj,
                                usage=Usage(),
                                timings=Timings(),
                                done=False,
                                raw=None,
                            )
                        LOGGER.warning(
                            "ollama stream finished with thinking but no visible answer; retrying without think model=%s",
                            model,
                        )
                        retry_stream = self._chat_with_retry(
                            req=_request_with_thinking_disabled(req),
                            model=model,
                            stream=True,
                        )
                        try:
                            for retry_raw_chunk in retry_stream:
                                retry_chunk = _as_dict(retry_raw_chunk)
                                retry_msg = dict(retry_chunk.get("message") or {})
                                retry_text_delta = str(retry_msg.get("content") or "")
                                retry_tool_calls_delta = _parse_tool_calls_from_message(
                                    retry_msg,
                                    text_fallback=retry_text_delta,
                                )
                                retry_done = bool(retry_chunk.get("done", False))
                                chunk_count += 1
                                chars += len(retry_text_delta)
                                if retry_done:
                                    log_json(
                                        LOGGER,
                                        "llm_stream_done",
                                        provider="ollama",
                                        model=str(retry_chunk.get("model") or model),
                                        chunks=chunk_count,
                                        text_chars=chars,
                                    )
                                yield LLMChunk(
                                    text_delta=retry_text_delta,
                                    thinking_delta="",
                                    tool_calls_delta=retry_tool_calls_delta,
                                    usage=(self._extract_usage(retry_chunk) if retry_done else Usage()),
                                    timings=(self._extract_timings(retry_chunk) if retry_done else Timings()),
                                    model=str(retry_chunk.get("model") or model),
                                    done=retry_done,
                                    raw=(retry_chunk if (self.debug_raw or (verbose and retry_done)) else None),
                                )
                        finally:
                            close_retry_stream = getattr(retry_stream, "close", None)
                            if callable(close_retry_stream):
                                try:
                                    close_retry_stream()
                                except Exception:
                                    pass
                        continue
                    yield chunk_obj
            finally:
                close_stream = getattr(stream, "close", None)
                if callable(close_stream):
                    try:
                        close_stream()
                    except Exception:
                        pass
        finally:
            priority_mgr.release(priority)

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
                    tool_calls=_tool_calls_from_message_dict(m),
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
                    tool_calls=_tool_calls_from_message_dict(m),
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
        attempt = 0
        disable_think = False
        while attempt < attempts:
            attempt += 1
            try:
                return self._chat_once(req=req, model=model, stream=stream, disable_think=disable_think)
            except Exception as exc:
                last_exc = exc
                think_requested = bool(dict(req.metadata or {}).get("think"))
                think_unsupported = think_requested and self._is_thinking_unsupported_error(exc)
                if think_unsupported and not disable_think:
                    disable_think = True
                    attempts = max(attempts, attempt + 1)
                    LOGGER.warning(
                        "ollama request failed attempt=%s/%s model=%s stream=%s error=%s; retry without think",
                        attempt,
                        attempts,
                        model,
                        bool(stream),
                        exc,
                    )
                    continue
                LOGGER.warning(
                    "ollama request failed attempt=%s/%s model=%s stream=%s error=%s",
                    attempt,
                    attempts,
                    model,
                    bool(stream),
                    exc,
                )
                if attempt >= attempts:
                    break
                time.sleep(0.25 * attempt)
        raise RuntimeError(str(last_exc or "Ollama request failed"))

    def _chat_once(self, *, req: LLMRequest, model: str, stream: bool, disable_think: bool = False):
        self._remember_model(model)
        messages = [_message_to_dict(m) for m in list(req.messages or [])]
        options = self._build_options(req)
        tools = [_tools for _tools in [_toolspec_to_ollama(t) for t in list(req.tools or [])] if _tools]

        think = req.metadata.get("think")
        if disable_think:
            think = None
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
            "tools": (tools or None),
            "format": fmt,
            "options": options or None,
            "keep_alive": keep_alive,
        }
        if think is not None:
            request_payload["think"] = think
        
        # Проверяем, является ли это Memory LLM
        is_memory_llm = "memory" in str(req.metadata.get("source", "")).lower()
        
        if is_memory_llm:
            # Проверяем флаг прерывания перед началом
            if _memory_llm_interrupt.is_set():
                LOGGER.warning("Memory LLM interrupted before start - main model responding")
                raise InterruptedError("Memory LLM interrupted - main model has priority")
        
        log_json(
            LOGGER,
            "llm_request_payload",
            provider="ollama",
            payload=request_payload,
            **_message_log_fields(messages),
        )
        
        if is_memory_llm and stream:
            # Для streaming проверяем флаг прерывания во время генерации
            return self._interruptible_chat_stream(req, model, request_payload)
        if is_memory_llm:
            return self._interruptible_chat_response(model, request_payload)

        payload = self._client.chat(**request_payload)
        if stream:
            return payload
        return _as_dict(payload)

    def _interruptible_chat_stream(self, req: LLMRequest, model: str, request_payload: dict):
        """Streaming генерация с проверкой прерывания для Memory LLM."""
        try:
            payload = self._client.chat(**request_payload)
            # Проверяем прерывание для каждого чанка
            for chunk in payload:
                if _memory_llm_interrupt.is_set():
                    LOGGER.warning("Memory LLM stream interrupted - main model responding")
                    raise InterruptedError("Memory LLM interrupted - main model has priority")
                yield chunk
        except InterruptedError:
            raise
        except Exception:
            raise

    def _interruptible_chat_response(self, model: str, request_payload: dict) -> dict[str, Any]:
        stream_payload = dict(request_payload)
        stream_payload["stream"] = True
        payload = self._client.chat(**stream_payload)

        content_parts: list[str] = []
        thinking_parts: list[str] = []
        tool_calls: list[Any] = []
        last_chunk: dict[str, Any] = {}

        for chunk in payload:
            if _memory_llm_interrupt.is_set():
                LOGGER.warning("Memory LLM response interrupted - main model responding")
                raise InterruptedError("Memory LLM interrupted - main model has priority")

            data = _as_dict(chunk)
            if not data:
                continue
            last_chunk = data
            message = dict(data.get("message") or {})
            text_delta = str(message.get("content") or "")
            if text_delta:
                content_parts.append(text_delta)

            thinking_delta = _extract_thinking(message, data)
            if thinking_delta:
                thinking_parts.append(thinking_delta)

            chunk_tool_calls = list(message.get("tool_calls") or [])
            if chunk_tool_calls:
                tool_calls = chunk_tool_calls

        payload_dict = dict(last_chunk or {})
        message_dict = dict(payload_dict.get("message") or {})
        message_dict["content"] = "".join(content_parts)
        if thinking_parts:
            message_dict["thinking"] = "".join(thinking_parts)
        if tool_calls:
            message_dict["tool_calls"] = tool_calls
        payload_dict["message"] = message_dict
        payload_dict.setdefault("model", model)
        return payload_dict

    @staticmethod
    def _is_thinking_unsupported_error(exc: Exception) -> bool:
        text = str(exc or "").strip().lower()
        if not text:
            return False
        markers = (
            "does not support thinking",
            "unknown field \"think\"",
            "unknown field 'think'",
            "unsupported parameter: think",
            "unsupported parameter 'think'",
            "unsupported parameter \"think\"",
        )
        return any(marker in text for marker in markers)

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
        
        # Для thinking моделей увеличиваем лимит, чтобы хватило и на thinking, и на ответ
        max_tokens = req.max_tokens
        if max_tokens is not None:
            max_tokens = int(max_tokens)
        if max_tokens is not None and max_tokens <= 0:
            options["num_predict"] = int(max_tokens)
        elif max_tokens is not None:
            think_enabled = bool(dict(req.metadata or {}).get("think", False))
            # Reasoning модели всегда используют thinking
            model = str(req.model or "")
            is_reasoning = any(p in model.lower() for p in ["deepseek-r1", "deepseek-reasoner", "o1", "o3"])
            if think_enabled or is_reasoning:
                # Thinking может занимать до 50% токенов, поэтому увеличиваем лимит
                options["num_predict"] = max(256, int(max_tokens * 2))
            else:
                # Минимум 256 токенов для нормального ответа
                options["num_predict"] = max(256, int(max_tokens))
        else:
            options["num_predict"] = 4096
            
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
        def _ns_to_ms(value: Any) -> float:
            try:
                if value is None:
                    return 0.0
                return max(0.0, float(value) / 1_000_000.0)
            except Exception:
                return 0.0

        total_duration_ms = _ns_to_ms(payload.get("total_duration"))
        load_duration_ms = _ns_to_ms(payload.get("load_duration"))
        prompt_eval_duration_ms = _ns_to_ms(payload.get("prompt_eval_duration"))
        eval_duration_ms = _ns_to_ms(payload.get("eval_duration"))
        return Timings(
            latency_ms=total_duration_ms,
            total_duration_ms=total_duration_ms,
            load_duration_ms=load_duration_ms,
            prompt_eval_duration_ms=prompt_eval_duration_ms,
            eval_duration_ms=eval_duration_ms,
        )

    def _unload_known_model(self, model_name: str | None = None) -> bool:
        target = str(model_name or getattr(self, "_current_model", None) or self.default_model or "").strip()
        if not target:
            return False

        unload_attempts = (
            lambda: self._client.generate(model=target, prompt="", keep_alive=0),
            lambda: self._client.chat(model=target, messages=[], keep_alive=0),
        )
        for attempt in unload_attempts:
            try:
                attempt()
                LOGGER.info(f"Ollama model '{target}' unloaded from VRAM")
                return True
            except Exception as exc:
                LOGGER.debug(f"Ollama unload attempt failed for '{target}': {exc}")
        return False

    def shutdown(self) -> None:
        """
        Полностью освобождает ресурсы LLM provider всеми способами.
        
        Для Ollama:
        1. Выгружаем модель через API
        2. Закрываем все соединения
        3. Очищаем кэш
        4. Сбрасываем приоритет
        """
        LOGGER.info("OllamaProvider.shutdown() called - FULL CLEANUP...")
        self._unload_known_model()
        try:
            client = getattr(self, "_client", None)
            if client:
                if hasattr(client, "close"):
                    client.close()
                    LOGGER.info("OllamaProvider: Provider client closed")
                transport = getattr(client, "transport", None)
                if transport and hasattr(transport, "close"):
                    transport.close()
                    LOGGER.info("OllamaProvider: Provider transport closed")
        except Exception as exc:
            LOGGER.warning(f"OllamaProvider: Provider client close failed: {exc}")
        
        # Способ 1: Выгрузка модели через Ollama API
        try:
            import ollama
            model_name = getattr(self, '_current_model', None)
            if model_name:
                LOGGER.info(f"OllamaProvider: Unloading model '{model_name}' via API...")
                try:
                    # Пытаемся выгрузить модель
                    ollama.generate(model=model_name, prompt="", keep_alive=0)
                    LOGGER.info(f"Ollama model '{model_name}' unloaded via generate()")
                except Exception as e1:
                    LOGGER.debug(f"Unload via generate() failed: {e1}")
                    
                try:
                    # Альтернативный способ через chat
                    ollama.chat(model=model_name, messages=[], keep_alive=0)
                    LOGGER.info(f"Ollama model '{model_name}' unloaded via chat()")
                except Exception as e2:
                    LOGGER.debug(f"Unload via chat() failed: {e2}")
        except Exception as exc:
            LOGGER.warning(f"OllamaProvider: API unload failed: {exc}")
        
        # Способ 2: Закрытие клиента
        try:
            import ollama
            if hasattr(ollama, '_client'):
                client = getattr(ollama, '_client', None)
                if client:
                    if hasattr(client, 'close'):
                        client.close()
                        LOGGER.info("OllamaProvider: Client closed")
                    if hasattr(client, 'transport') and hasattr(client.transport, 'close'):
                        client.transport.close()
                        LOGGER.info("OllamaProvider: Transport closed")
        except Exception as exc:
            LOGGER.warning(f"OllamaProvider: Client close failed: {exc}")
        
        # Способ 3: Принудительное закрытие HTTP сессий
        try:
            import ollama
            if hasattr(ollama, 'client'):
                client = getattr(ollama, 'client', None)
                if client and hasattr(client, 'close'):
                    client.close()
                    LOGGER.info("OllamaProvider: ollama.client closed")
        except Exception as exc:
            LOGGER.warning(f"OllamaProvider: ollama.client close failed: {exc}")
        
        # Способ 4: Сброс приоритета
        try:
            from llm.priority_manager import get_priority_manager
            manager = get_priority_manager()
            if manager:
                pass
        except Exception as exc:
            LOGGER.warning(f"OllamaProvider: Priority release failed: {exc}")
        
        # Способ 5: Очистка внутренних кэшей
        try:
            # Очищаем атрибуты provider
            for attr in ['_client', '_current_model', '_session', '_client_cache']:
                if hasattr(self, attr):
                    setattr(self, attr, None)
            LOGGER.info("OllamaProvider: Internal caches cleared")
        except Exception as exc:
            LOGGER.warning(f"OllamaProvider: Cache clear failed: {exc}")
        
        LOGGER.info("OllamaProvider.shutdown() COMPLETE - VRAM should be freed")

    def pause(self) -> None:
        """
        Приостанавливает LLM provider, освобождая VRAM.
        
        Для Ollama это означает явную выгрузку модели из памяти.
        """
        LOGGER.info("OllamaProvider.pause() called - releasing VRAM...")
        self._unload_known_model()
        LOGGER.info("OllamaProvider.pause() complete - model unloaded, client kept alive")

    def resume(self) -> None:
        """
        Возобновляет работу LLM provider.
        
        Для Ollama это означает готовность к новым запросам.
        """
        LOGGER.info("OllamaProvider.resume() called")
        try:
            self._client = ollama.Client(host=self.host, timeout=self.timeout_sec)
        except Exception as exc:
            LOGGER.warning(f"OllamaProvider.resume() failed to recreate client: {exc}")

    def warmup(self, keep_alive: Any | None = None) -> bool:
        """Прогревает модель в VRAM без полноценной генерации."""
        target = str(getattr(self, "_current_model", None) or self.default_model or "").strip()
        if not target:
            return False

        self._remember_model(target)
        payload: dict[str, Any] = {
            "model": target,
            "prompt": "",
            "options": {"num_predict": 0},
        }
        if keep_alive is not None and str(keep_alive).strip() != "":
            payload["keep_alive"] = keep_alive

        started_at = time.perf_counter()
        try:
            self._client.generate(**payload)
            elapsed_ms = (time.perf_counter() - started_at) * 1000.0
            LOGGER.info(
                "OllamaProvider.warmup() complete for model '%s' in %.1fms",
                target,
                elapsed_ms,
            )
            return True
        except Exception as exc:
            LOGGER.debug(f"OllamaProvider.warmup() failed for '{target}': {exc}")
            return False
    
    def unload_model(self) -> None:
        """
        Принудительно выгружает модель из VRAM.
        
        Для Ollama это означает отправку пустого запроса для выгрузки.
        """
        LOGGER.info("OllamaProvider.unload_model() called - unloading model from VRAM...")
        self._unload_known_model()
        return
        try:
            model_name = getattr(self, '_current_model', None) or getattr(self, 'default_model', None)
            if model_name:
                # Пустой запрос выгружает модель из памяти
                ollama.generate(model=model_name, prompt="")
                LOGGER.info(f"Ollama model '{model_name}' unloaded from VRAM")
        except Exception as exc:
            LOGGER.warning(f"OllamaProvider.unload_model() error: {exc}")
