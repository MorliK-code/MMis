from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from memory.claim_models import ClaimCandidate


_VALID_PREDICATES = {"likes", "dislikes", "owns", "uses"}
_STOPWORD_ONLY = {
    "",
    "что",
    "это",
    "ну",
    "ну это",
    "вот",
    "такое",
    "всё",
    "все",
    "ничего",
    "something",
    "anything",
    "everything",
    "this",
    "that",
    "it",
}
_GARBAGE_PUNCT_RE = re.compile(r"^[-_.,:;!? ]+$")
_GARBAGE_TOKENS = {"lol", "kek", "\u043c\u0434\u0430"}
_CLAUSE_EXPLOSION_RE = re.compile(
    r"\b(?:потому\s+что|если|когда|although|because|if|when|while|чтобы)\b",
    re.I,
)
_MAX_OBJECT_LEN = 96
_MAX_TOKENS = 12
_MIN_CONFIDENCE = 0.62


@dataclass(frozen=True)
class ClaimValidationDecision:
    valid: bool
    reason: str = ""
    signals: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": bool(self.valid),
            "reason": str(self.reason or "").strip(),
            "signals": dict(self.signals or {}),
        }


def validate_claim_candidate(candidate: ClaimCandidate) -> ClaimValidationDecision:
    predicate = str(candidate.predicate or "").strip().lower()
    object_surface = str(candidate.object_surface or "").strip()
    normalized_object = str(candidate.normalized_object or "").strip().lower()
    confidence = float(candidate.confidence or 0.0)
    evidence_text = str(candidate.evidence_text or "").strip()
    topic_keys = [str(x).strip().lower() for x in list(candidate.topic_keys or []) if str(x).strip()]

    if predicate not in _VALID_PREDICATES:
        return ClaimValidationDecision(valid=False, reason="invalid_predicate", signals={"predicate": predicate})
    if not object_surface or not normalized_object:
        return ClaimValidationDecision(valid=False, reason="empty_object")
    if normalized_object in _STOPWORD_ONLY or _is_stopword_only(normalized_object):
        return ClaimValidationDecision(valid=False, reason="stopword_only_object")
    tokens = [part for part in normalized_object.split() if part]
    if len(tokens) > _MAX_TOKENS and _CLAUSE_EXPLOSION_RE.search(object_surface):
        return ClaimValidationDecision(
            valid=False,
            reason="clause_explosion",
            signals={"token_count": len(tokens)},
        )
    if len(object_surface) > _MAX_OBJECT_LEN:
        return ClaimValidationDecision(
            valid=False,
            reason="object_too_long",
            signals={"object_len": len(object_surface)},
        )
    if confidence < _MIN_CONFIDENCE:
        return ClaimValidationDecision(
            valid=False,
            reason="low_confidence",
            signals={"confidence": confidence},
        )
    if not evidence_text:
        return ClaimValidationDecision(valid=False, reason="missing_evidence_text")
    if _GARBAGE_PUNCT_RE.search(object_surface) or normalized_object in _GARBAGE_TOKENS:
        return ClaimValidationDecision(valid=False, reason="garbage_object")
    if not topic_keys:
        return ClaimValidationDecision(valid=False, reason="missing_topic_keys")
    return ClaimValidationDecision(
        valid=True,
        reason="claim_candidate_valid",
        signals={
            "predicate": predicate,
            "confidence": confidence,
            "topic_keys_count": len(topic_keys),
        },
    )


def _is_stopword_only(value: str) -> bool:
    tokens = [part for part in str(value or "").split() if part]
    if not tokens:
        return True
    return all(token in _STOPWORD_ONLY for token in tokens)
