from __future__ import annotations

"""Legacy compatibility wrapper for claim promotion.

Primary claim extraction now lives in `memory.claim_candidate_extractor`.
Primary claim promotion now lives in `memory.claim_promoter`.
This module remains as a thin shim so older imports keep working.
"""

from memory.claim_promoter import decide_claim_promotion, promote_claim_candidates

__all__ = ["decide_claim_promotion", "promote_claim_candidates"]
