from __future__ import annotations

"""Legacy compatibility shim for old claim-extractor imports.

Do not use this module in new code.

Current split:
    claim_candidate_extractor -> raw text -> ClaimCandidate
    claim_promoter -> ClaimCandidate -> ClaimRecord / promotion decisions

This file exists only so older imports keep working during migration.
"""

from memory.claim_models import ClaimCandidate, ClaimPromotionDecision, ClaimRecord
from memory.claim_promoter import (
    decide_claim_promotion as _decide_claim_promotion_impl,
    promote_claim_candidates as _promote_claim_candidates_impl,
)


LEGACY_COMPAT_ONLY = True
LEGACY_SHIM_NOTE = "Do not use this module in new code."


def decide_claim_promotion(candidate: ClaimCandidate, *, repetition_count: int = 0) -> ClaimPromotionDecision:
    """Legacy wrapper. New code should import from `memory.claim_promoter`."""
    return _decide_claim_promotion_impl(candidate, repetition_count=repetition_count)


def promote_claim_candidates(
    candidates: list[ClaimCandidate],
    *,
    event_id: str,
    namespace: str,
    scope: str = "conversation",
) -> list[ClaimRecord]:
    """Legacy wrapper. New code should import from `memory.claim_promoter`."""
    return _promote_claim_candidates_impl(
        candidates,
        event_id=event_id,
        namespace=namespace,
        scope=scope,
    )


__all__ = [
    "LEGACY_COMPAT_ONLY",
    "LEGACY_SHIM_NOTE",
    "decide_claim_promotion",
    "promote_claim_candidates",
]
