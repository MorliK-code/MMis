"""LLM package exports (provider primitives only)."""

from __future__ import annotations

import os

from llm.ollama_provider import OllamaProvider
from llm.openai_provider import OpenAIProvider
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

def build_provider(name: str | None = None, *, default_model: str | None = None) -> LLMProviderBase:
    if name:
        provider_name = str(name).strip().lower()
    else:
        cfg = load_config()
        provider_name = str(getattr(cfg, "llm_default_provider", "ollama")).strip().lower()
        
    if provider_name == "openai":
        return OpenAIProvider(default_model=default_model)
    return OllamaProvider(default_model=default_model)


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
