"""LLM package exports with lazy provider imports."""

from __future__ import annotations

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
    ToolResult,
    ToolSpec,
    Usage,
)

from config.settings import load_config


def _import_ollama_provider():
    from llm.ollama_provider import OllamaProvider  # local import for optional dependency

    return OllamaProvider


def _import_openai_provider():
    from llm.openai_provider import OpenAIProvider  # local import for optional dependency

    return OpenAIProvider


def build_provider(name: str | None = None, *, default_model: str | None = None) -> LLMProviderBase:
    if name:
        provider_name = str(name).strip().lower()
    else:
        cfg = load_config()
        provider_name = str(getattr(cfg, "llm_default_provider", "ollama")).strip().lower()

    if provider_name == "openai":
        return _import_openai_provider()(default_model=default_model)
    return _import_ollama_provider()(default_model=default_model)


class _LazyProviderAlias:
    def __init__(self, importer):
        self._importer = importer

    def __call__(self, *args, **kwargs):
        return self._importer()(*args, **kwargs)

    def __getattr__(self, item):
        return getattr(self._importer(), item)


OllamaProvider = _LazyProviderAlias(_import_ollama_provider)
OpenAIProvider = _LazyProviderAlias(_import_openai_provider)


__all__ = [
    "LLMProviderBase",
    "LLMRequest",
    "LLMResponse",
    "LLMChunk",
    "Message",
    "ToolSpec",
    "ToolCall",
    "ToolResult",
    "Usage",
    "Timings",
    "ProviderHealth",
    "ModelInfo",
    "OllamaProvider",
    "OpenAIProvider",
    "build_provider",
]
