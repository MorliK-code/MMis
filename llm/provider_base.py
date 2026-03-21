from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal


Role = Literal["system", "user", "assistant", "tool"]


@dataclass(frozen=True)
class Message:
    role: Role
    content: str = ""
    name: str = ""
    tool_call_id: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str = ""
    input_schema: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    raw_arguments: str = ""


@dataclass(frozen=True)
class ToolResult:
    tool_call_id: str
    name: str
    content: str
    is_error: bool = False


@dataclass(frozen=True)
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass(frozen=True)
class Timings:
    latency_ms: float = 0.0
    total_duration_ms: float = 0.0
    load_duration_ms: float = 0.0
    prompt_eval_duration_ms: float = 0.0
    eval_duration_ms: float = 0.0


@dataclass(frozen=True)
class LLMRequest:
    messages: list[Message]
    model: str = ""
    temperature: float | None = None
    top_p: float | None = None
    repeat_penalty: float | None = None
    seed: int | None = None
    max_tokens: int | None = None
    stop: list[str] = field(default_factory=list)
    json_mode: bool = False
    response_format: dict[str, Any] | None = None
    tools: list[ToolSpec] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LLMChunk:
    text_delta: str = ""
    tool_calls_delta: list[ToolCall] = field(default_factory=list)
    thinking_delta: str = ""
    usage: Usage = field(default_factory=Usage)
    timings: Timings = field(default_factory=Timings)
    model: str = ""
    done: bool = False
    raw: Any = None


@dataclass(frozen=True)
class LLMResponse:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    thinking: str = ""
    usage: Usage = field(default_factory=Usage)
    timings: Timings = field(default_factory=Timings)
    model: str = ""
    raw: Any = None


@dataclass(frozen=True)
class ProviderHealth:
    ok: bool
    provider: str
    detail: str = ""
    model: str = ""


@dataclass(frozen=True)
class ModelInfo:
    provider: str
    model: str
    endpoint: str = ""
    capabilities: dict[str, Any] = field(default_factory=dict)
    params: dict[str, Any] = field(default_factory=dict)


class LLMProviderBase(ABC):
    @abstractmethod
    def generate(self, req: LLMRequest) -> LLMResponse:
        raise NotImplementedError

    def stream(self, req: LLMRequest):
        raise NotImplementedError("Streaming is not implemented for this provider")

    @abstractmethod
    def healthcheck(self) -> ProviderHealth:
        raise NotImplementedError

    @abstractmethod
    def model_info(self, model: str = "") -> ModelInfo:
        raise NotImplementedError

    @abstractmethod
    def list_models(self) -> list[str]:
        raise NotImplementedError
