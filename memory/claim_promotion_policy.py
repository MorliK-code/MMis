from __future__ import annotations

import re

from memory.claim_models import ClaimCandidate, ClaimPromotionDecision
from memory.claim_validator import validate_claim_candidate


_HIGH_AFFECT_RE = re.compile(r"\b(?:обожаю|люблю|ненавижу|терпеть\s+не\s+могу|i\s+love|i\s+hate)\b", re.I)
_MEDIUM_AFFECT_RE = re.compile(r"\b(?:мне\s+нравится|мне\s+не\s+нравится|i\s+like|i\s+do\s+not\s+like)\b", re.I)
_OWN_RE = re.compile(r"\b(?:у\s+меня(?:\s+есть)?|i\s+have)\b", re.I)
_USE_RE = re.compile(r"\b(?:я\s+использую|использую|пользуюсь|юзаю|i\s+use|i(?:'m| am)\s+using)\b", re.I)
_SELF_RE = re.compile(r"\b(?:я|мне|у\s+меня|мой|моя|моё|мои|i|me|my)\b", re.I)
_HEDGE_RE = re.compile(r"\b(?:кажется|наверное|может|может быть|вроде|как будто|maybe|probably|perhaps|kind of|sort of)\b", re.I)
_PROPER_NAME_RE = re.compile(r"\b[A-ZА-ЯЁІЇЄҐ][A-Za-zА-Яа-яЁёІіЇїЄєҐґ'’ -]{1,40}\b")
_HIGH_SUPPORT_TYPES = {"tool", "person", "appliance", "device", "vehicle", "project", "platform"}


def decide_claim_promotion_level(
    candidate: ClaimCandidate,
    *,
    repetition_count: int = 0,
) -> ClaimPromotionDecision:
    validation = validate_claim_candidate(candidate)
    if not validation.valid:
        return ClaimPromotionDecision(
            action="reject",
            reason=str(validation.reason or "invalid_claim_candidate"),
            allow_write=False,
            signals=dict(validation.signals or {}),
        )

    explicitness = _explicitness_score(candidate)
    self_reference = _self_reference_score(candidate)
    specificity = max(0.0, min(1.0, float(candidate.specificity or 0.0)))
    entity_support = _entity_support_score(candidate)
    repetition = _repetition_score(repetition_count)
    confidence = max(0.0, min(1.0, float(candidate.confidence or 0.0)))
    contradiction_risk = _contradiction_risk_score(candidate)

    score = (
        (0.22 * explicitness)
        + (0.16 * self_reference)
        + (0.18 * specificity)
        + (0.16 * entity_support)
        + (0.14 * repetition)
        + (0.14 * confidence)
        - (0.16 * contradiction_risk)
    )
    score = max(0.0, min(1.0, score))

    if _is_strong_explicit_affect(
        candidate,
        explicitness=explicitness,
        self_reference=self_reference,
        specificity=specificity,
        entity_support=entity_support,
        confidence=confidence,
    ):
        level = "strong_claim"
    elif score >= 0.74:
        level = "strong_claim"
    elif score >= 0.52:
        level = "weak_claim"
    elif score >= 0.42:
        level = "episodic_only"
    else:
        level = "reject"

    return ClaimPromotionDecision(
        action=level,
        reason=f"claim_promotion_{level}",
        allow_write=level in {"weak_claim", "strong_claim"},
        signals={
            "level": level,
            "score": score,
            "explicitness": explicitness,
            "self_reference": self_reference,
            "specificity": specificity,
            "entity_support": entity_support,
            "repetition": repetition,
            "confidence": confidence,
            "contradiction_risk": contradiction_risk,
            "repetition_count": max(0, int(repetition_count)),
        },
    )


def _explicitness_score(candidate: ClaimCandidate) -> float:
    predicate = str(candidate.predicate or "").strip().lower()
    evidence = str(candidate.evidence_text or "").strip()
    if predicate in {"likes", "dislikes"}:
        if _HIGH_AFFECT_RE.search(evidence):
            return 0.96
        if _MEDIUM_AFFECT_RE.search(evidence):
            return 0.72
        return 0.60
    if predicate == "owns":
        return 0.86 if _OWN_RE.search(evidence) else 0.68
    if predicate == "uses":
        return 0.84 if _USE_RE.search(evidence) else 0.66
    return 0.50


def _self_reference_score(candidate: ClaimCandidate) -> float:
    evidence = str(candidate.evidence_text or "").strip()
    subject = str(candidate.subject or "").strip().lower()
    if subject == "user" and _SELF_RE.search(evidence):
        return 1.0
    if subject == "user":
        return 0.72
    return 0.40


def _entity_support_score(candidate: ClaimCandidate) -> float:
    qualifiers = dict(candidate.qualifiers or {})
    matched = list(qualifiers.get("matched_entities") or [])
    object_type = str(candidate.object_type or "").strip().lower()
    alternatives = [str(x).strip() for x in list(candidate.alternatives or []) if str(x).strip()]
    object_surface = str(candidate.object_surface or "").strip()

    score = 0.18
    if matched:
        score = max(score, min(1.0, 0.58 + (0.16 * min(len(matched), 2))))
    if object_type in _HIGH_SUPPORT_TYPES:
        score = max(score, 0.74)
    if _PROPER_NAME_RE.search(object_surface):
        score = max(score, 0.62)
    if len(alternatives) >= 3:
        score = min(1.0, score + 0.08)
    return max(0.0, min(1.0, score))


def _repetition_score(repetition_count: int) -> float:
    count = max(0, int(repetition_count))
    if count <= 0:
        return 0.0
    if count == 1:
        return 0.55
    return 0.85


def _contradiction_risk_score(candidate: ClaimCandidate) -> float:
    predicate = str(candidate.predicate or "").strip().lower()
    evidence = str(candidate.evidence_text or "").strip()
    object_type = str(candidate.object_type or "").strip().lower()
    qualifiers = dict(candidate.qualifiers or {})
    matched = list(qualifiers.get("matched_entities") or [])

    if predicate in {"likes", "dislikes"}:
        risk = 0.34
    elif predicate == "uses":
        risk = 0.24
    elif predicate == "owns":
        risk = 0.18
    else:
        risk = 0.26

    if _HEDGE_RE.search(evidence):
        risk += 0.16
    if matched:
        risk -= 0.10
    if object_type in _HIGH_SUPPORT_TYPES:
        risk -= 0.08
    if _PROPER_NAME_RE.search(str(candidate.object_surface or "")):
        risk -= 0.06
    return max(0.0, min(1.0, risk))


def _is_strong_explicit_affect(
    candidate: ClaimCandidate,
    *,
    explicitness: float,
    self_reference: float,
    specificity: float,
    entity_support: float,
    confidence: float,
) -> bool:
    predicate = str(candidate.predicate or "").strip().lower()
    if predicate not in {"likes", "dislikes"}:
        return False
    if explicitness < 0.94 or self_reference < 0.95 or specificity < 0.90 or confidence < 0.80:
        return False
    object_surface = str(candidate.object_surface or "").strip()
    if entity_support >= 0.55:
        return True
    return bool(_PROPER_NAME_RE.search(object_surface))
