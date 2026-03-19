from __future__ import annotations

from importlib import import_module

__all__ = [
    "Brain",
    "BrainResult",
    "CharacterRuntime",
    "CharacterMeta",
    "CharacterRuntimeResult",
    "PersonalityProfile",
    "PersonalityDecision",
    "StateSnapshot",
    "PromptBudgets",
    "PromptPack",
    "ResponsePipeline",
    "PipelineContext",
    "PipelineStage",
    "PipelineResult",
    "PROFILE_FAST",
    "PROFILE_BALANCED",
    "PROFILE_QUALITY",
    "PROFILE_AUTONOMOUS",
    "DebugTrace",
]

_LAZY = {
    "Brain": "core.brain",
    "BrainResult": "core.brain",
    "CharacterRuntime": "core.character_runtime",
    "CharacterMeta": "core.character_runtime",
    "CharacterRuntimeResult": "core.character_runtime",
    "PersonalityProfile": "core.character_runtime",
    "PersonalityDecision": "core.character_runtime",
    "StateSnapshot": "core.character_runtime",
    "PromptBudgets": "core.character_runtime",
    "PromptPack": "core.character_runtime",
    "ResponsePipeline": "core.response_pipeline",
    "PipelineContext": "core.response_pipeline",
    "PipelineStage": "core.response_pipeline",
    "PipelineResult": "core.response_pipeline",
    "PROFILE_FAST": "core.response_pipeline",
    "PROFILE_BALANCED": "core.response_pipeline",
    "PROFILE_QUALITY": "core.response_pipeline",
    "PROFILE_AUTONOMOUS": "core.response_pipeline",
    "DebugTrace": "core.debug_trace",
}


def __getattr__(name: str):
    module_path = _LAZY.get(name)
    if not module_path:
        raise AttributeError(name)
    module = import_module(module_path)
    value = getattr(module, name)
    globals()[name] = value
    return value
