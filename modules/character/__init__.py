from __future__ import annotations

from modules.character.composer import CharacterComposeResult, CharacterComposer
from modules.character.feedback_detector import detect_feedback
from modules.character.dialog_policies import (
    compute_address_terms_policy,
    compute_dialog_flags,
    compute_dialog_mode,
    deterministic_term_gate_score,
    extract_term_directive,
    is_new_session,
    is_technical,
    is_user_greeting,
)
from modules.character.engine import CharacterEngine, CharacterUpdateResult
from modules.character.evaluator import ResponseConstraintEvaluator, RuleEvaluator
from modules.character.learner import update_persona
from modules.character.signals import CharacterSignals, build_character_signals
from modules.character.storage import CharacterStorage

__all__ = [
    "CharacterEngine",
    "CharacterUpdateResult",
    "CharacterStorage",
    "CharacterRuntime",
    "RuleEvaluator",
    "ResponseConstraintEvaluator",
    "CharacterComposer",
    "CharacterComposeResult",
    "CharacterSignals",
    "build_character_signals",
    "detect_feedback",
    "update_persona",
    "compute_address_terms_policy",
    "is_user_greeting",
    "is_technical",
    "is_new_session",
    "compute_dialog_flags",
    "compute_dialog_mode",
    "extract_term_directive",
    "deterministic_term_gate_score",
]

# Lazy import for CharacterRuntime to avoid circular dependency.
def __getattr__(name: str):
    if name == "CharacterRuntime":
        from core.character_runtime import CharacterRuntime as _CharacterRuntime
        return _CharacterRuntime
    raise AttributeError(name)
