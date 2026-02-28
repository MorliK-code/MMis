from __future__ import annotations

from importlib import import_module

__all__ = [
    "Brain",
    "BrainResult",
    "PromptBuilder",
    "PromptPack",
    "ResponsePipeline",
    "PipelineContext",
    "PipelineStage",
    "PipelineResult",
    "PROFILE_FAST",
    "PROFILE_BALANCED",
    "PROFILE_QUALITY",
    "StateManager",
    "StateSnapshot",
]

_LAZY = {
    "Brain": "core.brain",
    "BrainResult": "core.brain",
    "PromptBuilder": "core.prompt_builder",
    "PromptPack": "core.prompt_builder",
    "ResponsePipeline": "core.response_pipeline",
    "PipelineContext": "core.response_pipeline",
    "PipelineStage": "core.response_pipeline",
    "PipelineResult": "core.response_pipeline",
    "PROFILE_FAST": "core.response_pipeline",
    "PROFILE_BALANCED": "core.response_pipeline",
    "PROFILE_QUALITY": "core.response_pipeline",
    "StateManager": "core.state_manager",
    "StateSnapshot": "core.state_manager",
}


def __getattr__(name: str):
    module_path = _LAZY.get(name)
    if not module_path:
        raise AttributeError(name)
    module = import_module(module_path)
    value = getattr(module, name)
    globals()[name] = value
    return value
