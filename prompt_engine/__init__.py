from __future__ import annotations

from importlib import import_module

__all__ = [
    "PromptEngine",
    "PromptEngineResult",
    "PromptLoader",
    "PromptDocument",
    "PromptEntry",
    "PromptRegistry",
    "PromptVersion",
    "PromptVersioning",
    "ContextBlock",
    "TokenBudget",
    "TokenBudgetManager",
]

_LAZY = {
    "PromptEngine": "prompt_engine.prompt_engine",
    "PromptEngineResult": "prompt_engine.prompt_engine",
    "PromptLoader": "prompt_engine.prompt_loader",
    "PromptDocument": "prompt_engine.prompt_loader",
    "PromptEntry": "prompt_engine.prompt_registry",
    "PromptRegistry": "prompt_engine.prompt_registry",
    "PromptVersion": "prompt_engine.prompt_versioning",
    "PromptVersioning": "prompt_engine.prompt_versioning",
    "ContextBlock": "prompt_engine.token_budget_manager",
    "TokenBudget": "prompt_engine.token_budget_manager",
    "TokenBudgetManager": "prompt_engine.token_budget_manager",
}


def __getattr__(name: str):
    module_path = _LAZY.get(name)
    if not module_path:
        raise AttributeError(name)
    module = import_module(module_path)
    value = getattr(module, name)
    globals()[name] = value
    return value
