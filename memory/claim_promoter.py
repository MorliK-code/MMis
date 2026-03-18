from __future__ import annotations

from memory.claim_models import ClaimCandidate, ClaimPromotionDecision, ClaimRecord
from memory.claim_promotion_policy import decide_claim_promotion_level


def decide_claim_promotion(candidate: ClaimCandidate, *, repetition_count: int = 0) -> ClaimPromotionDecision:
    return decide_claim_promotion_level(candidate, repetition_count=repetition_count)


def promote_claim_candidates(
    candidates: list[ClaimCandidate],
    *,
    event_id: str,
    namespace: str,
    scope: str = "conversation",
) -> list[ClaimRecord]:
    out: list[ClaimRecord] = []
    counts = _candidate_repetition_counts(candidates)
    for candidate in list(candidates or []):
        key = _candidate_key(candidate)
        repetition_count = max(0, int(counts.get(key, 0)) - 1)
        decision = decide_claim_promotion(candidate, repetition_count=repetition_count)
        if not decision.allow_write:
            continue

        normalized_object = str(candidate.normalized_object or "").strip().lower()
        predicate = str(candidate.predicate or "").strip().lower()
        promotion_level = str(decision.action or "").strip().lower()
        canonical_key = ".".join(
            part
            for part in (
                str(candidate.subject or "").strip().lower(),
                predicate,
                _slug(normalized_object),
            )
            if part
        )
        salience = min(
            1.0,
            (0.55 * float(candidate.confidence or 0.0))
            + (0.35 * float(candidate.specificity or 0.0))
            + (0.10 * (1.0 if predicate in {"likes", "dislikes"} else 0.7)),
        )
        if promotion_level == "strong_claim":
            salience = min(1.0, salience * 1.12)
        elif promotion_level == "weak_claim":
            salience = min(1.0, salience * 0.82)

        spontaneous = promotion_level == "strong_claim" and predicate in {"likes", "dislikes"}
        recall_mode = "ambient" if promotion_level == "strong_claim" and predicate in {"likes", "dislikes"} else "contextual"
        out.append(
            ClaimRecord(
                subject=str(candidate.subject or "").strip().lower(),
                predicate=predicate,
                obj=normalized_object,
                subject_type="user" if str(candidate.subject or "").strip().lower() == "user" else "",
                object_type=str(candidate.object_type or "").strip().lower(),
                object_surface=str(candidate.object_surface or "").strip(),
                qualifiers=dict(candidate.qualifiers or {}),
                confidence=float(candidate.confidence or 0.0),
                salience=salience,
                evidence_text=str(candidate.evidence_text or "").strip(),
                evidence_span=candidate.evidence_span,
                topic_keys=list(candidate.topic_keys or []),
                trigger_keys=list(candidate.trigger_keys or []),
                recall_mode=recall_mode,
                spontaneous_recall=spontaneous,
                status="active",
                promotion_level=promotion_level,
                promotion_signals=dict(decision.signals or {}),
                canonical_key=canonical_key,
                source_event_id=str(event_id or "").strip(),
                scope=str(scope or "conversation").strip().lower() or "conversation",
                namespace=str(namespace or "default").strip() or "default",
            )
        )
    return _dedupe_records(out)


def _slug(value: str) -> str:
    tokens = [part for part in str(value or "").strip().lower().split() if part]
    return "_".join(tokens)


def _candidate_key(candidate: ClaimCandidate) -> tuple[str, str, str]:
    return (
        str(candidate.subject or "").strip().lower(),
        str(candidate.predicate or "").strip().lower(),
        str(candidate.normalized_object or "").strip().lower(),
    )


def _candidate_repetition_counts(candidates: list[ClaimCandidate]) -> dict[tuple[str, str, str], int]:
    counts: dict[tuple[str, str, str], int] = {}
    for candidate in list(candidates or []):
        key = _candidate_key(candidate)
        counts[key] = int(counts.get(key, 0)) + 1
    return counts


def _dedupe_records(items: list[ClaimRecord]) -> list[ClaimRecord]:
    seen: set[str] = set()
    out: list[ClaimRecord] = []
    for item in list(items or []):
        key = str(item.canonical_key or "").strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out
