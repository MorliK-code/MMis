from __future__ import annotations

"""Extract pre-promotion claim candidates from raw text.

This module is the claim-ingest layer for open-claim memory.
Its job is only:
    raw text -> bounded ClaimCandidate objects

It does not decide persistence level and does not write claim records.
That promotion step lives in `memory.claim_promoter`.
"""

import re
from typing import Any

from memory.claim_models import ClaimCandidate
from memory.ingest_analyzer import EntityItem, IngestAnchor, NumericFact
from memory.claim_normalizer import normalize_claim_key, normalize_claim_object
from memory.claim_validator import validate_claim_candidate


_GENERIC_OBJECTS = {
    "",
    "это",
    "такое",
    "всё",
    "все",
    "ничего",
    "что-то",
    "something",
    "anything",
    "everything",
    "this",
    "that",
    "it",
}
_PROPER_NAME_RE = re.compile(r"\b[A-ZА-ЯЁІЇЄҐ][A-Za-zА-Яа-яЁёІіЇїЄєҐґ'’ -]{0,40}\b")
_BOUNDARY_RE = re.compile(
    r"\s+(?:а|но|зато|because|but|если|if|когда|when|потому\s+что|so\s+that)\b",
    re.I,
)
_CLAIM_PATTERNS: tuple[dict[str, Any], ...] = (
    {
        "predicate": "dislikes",
        "pattern": re.compile(
            r"(?:терпеть\s+не\s+могу|ненавижу|не\s+люблю|мне\s+не\s+нравится|i\s+do\s+not\s+like|i\s+hate)\s+(?P<object>[^.,;!?\n]{2,120})",
            re.I,
        ),
        "confidence": 0.90,
    },
    {
        "predicate": "likes",
        "pattern": re.compile(
            r"(?:обожаю|люблю|мне\s+нравится|i\s+like|i\s+love)\s+(?P<object>[^.,;!?\n]{2,120})",
            re.I,
        ),
        "confidence": 0.84,
    },
    {
        "predicate": "owns",
        "pattern": re.compile(
            r"(?:у\s+меня(?:\s+есть)?|i\s+have)\s+(?P<object>[^.,;!?\n]{2,120})",
            re.I,
        ),
        "confidence": 0.80,
    },
    {
        "predicate": "uses",
        "pattern": re.compile(
            r"(?:я\s+использую|использую|юзаю|я\s+пользуюсь|пользуюсь|i\s+use|i(?:'m| am)\s+using)\s+(?P<object>[^.,;!?\n]{2,120})",
            re.I,
        ),
        "confidence": 0.80,
    },
)
_ENTITY_TYPE_TO_OBJECT_TYPE = {
    "person_name": "person",
    "tool_name": "tool",
    "project_name": "project",
    "gpu_model": "device",
    "cpu_model": "device",
    "os_name": "platform",
}
_OWNABLE_HEAD_TYPES = {
    "холодильник": "appliance",
    "телефон": "device",
    "ноутбук": "device",
    "машина": "vehicle",
    "fridge": "appliance",
    "phone": "device",
    "laptop": "device",
    "car": "vehicle",
}
_ACTIVITY_HEADS = {
    "смотреть",
    "слушать",
    "читать",
    "рисовать",
    "гулять",
    "watch",
    "listen",
    "read",
    "draw",
    "walk",
}
_ENVIRONMENT_USE_PREFIXES = (
    "python ",
    "питон ",
    "пайтон ",
    "windows",
    "linux",
    "ubuntu",
    "debian",
    "macos",
    "rtx ",
    "gtx ",
    "rx ",
)


def extract_claim_candidates(
    text: str,
    *,
    subject: str = "user",
    entities: list[EntityItem] | None = None,
    numeric_facts: list[NumericFact] | None = None,
    anchors: list[IngestAnchor] | None = None,
    context: dict[str, Any] | None = None,
) -> list[ClaimCandidate]:
    """Extract bounded `ClaimCandidate` items for later promotion."""
    src = str(text or "").strip()
    if not src:
        return []

    out: list[ClaimCandidate] = []
    for spec in _CLAIM_PATTERNS:
        predicate = str(spec.get("predicate") or "").strip().lower()
        pattern = spec.get("pattern")
        if not predicate or not isinstance(pattern, re.Pattern):
            continue
        for match in pattern.finditer(src):
            if predicate == "likes" and _has_negation_prefix(src, match.start()):
                continue
            raw_object = _bounded_object_surface(match.group("object") or "")
            matched_entities = _matched_entities(raw_object, entities)
            normalized = normalize_claim_object(raw_object, matched_entities=matched_entities)
            if normalized is None:
                continue
            object_surface = str(normalized.object_surface or "").strip()
            matched_entities = _matched_entities(object_surface, entities)
            if not _should_keep_candidate(
                object_surface,
                predicate=predicate,
                matched_entities=matched_entities,
                numeric_facts=numeric_facts,
            ):
                continue
            normalized_object = str(normalized.normalized_object or "").strip()
            if not normalized_object:
                continue
            head = _claim_head(object_surface, matched_entities)
            object_type = _infer_object_type(
                object_surface,
                predicate=predicate,
                matched_entities=matched_entities,
            )
            alternatives = _object_alternatives(
                object_surface=object_surface,
                matched_entities=matched_entities,
                normalized_object=normalized_object,
                head=head,
            )
            topic_keys = _topic_keys(
                predicate=predicate,
                object_type=object_type,
                matched_entities=matched_entities,
            )
            trigger_keys = _trigger_keys(
                predicate=predicate,
                head=head,
                normalized_object=normalized_object,
                matched_entities=matched_entities,
            )
            _ = (anchors, context)
            candidate = ClaimCandidate(
                subject=str(subject or "user").strip().lower() or "user",
                predicate=predicate,
                object_surface=object_surface,
                object_type=object_type,
                head=head,
                normalized_object=normalized_object,
                alternatives=alternatives,
                qualifiers={
                    "surface": object_surface,
                    "matched_entities": [
                        {
                            "type": str(item.type or "").strip(),
                            "surface": str(item.surface or "").strip(),
                            "canonical": str(item.canonical or "").strip(),
                        }
                        for item in matched_entities
                    ],
                },
                confidence=float(spec.get("confidence") or 0.0),
                specificity=_specificity_score(normalized_object, alternatives=alternatives),
                evidence_text=src,
                evidence_span=(int(match.start("object")), int(match.start("object")) + len(object_surface)),
                topic_keys=topic_keys,
                trigger_keys=trigger_keys,
            )
            validation = validate_claim_candidate(candidate)
            if not validation.valid:
                continue
            out.append(candidate)
    return _dedupe_candidates(out)


def _bounded_object_surface(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    boundary = _BOUNDARY_RE.search(text)
    if boundary is not None:
        text = text[: boundary.start()]
    return str(text or "").strip()


def _slug(value: str) -> str:
    return "_".join([part for part in normalize_claim_key(value).split() if part])


def _matched_entities(object_surface: str, entities: list[EntityItem] | None) -> list[EntityItem]:
    surface_norm = normalize_claim_key(object_surface)
    if not surface_norm:
        return []
    out: list[EntityItem] = []
    seen: set[tuple[str, str]] = set()
    for item in list(entities or []):
        surface = normalize_claim_key(getattr(item, "surface", ""))
        canonical = normalize_claim_key(getattr(item, "canonical", ""))
        if surface and surface in surface_norm:
            key = (str(item.type or "").strip().lower(), canonical or surface)
            if key not in seen:
                seen.add(key)
                out.append(item)
            continue
        if canonical and canonical in surface_norm:
            key = (str(item.type or "").strip().lower(), canonical)
            if key not in seen:
                seen.add(key)
                out.append(item)
    return out


def _claim_head(object_surface: str, matched_entities: list[EntityItem]) -> str:
    if matched_entities:
        canonical = str(getattr(matched_entities[0], "canonical", "") or "").strip()
        if canonical:
            return canonical
    tokens = [part for part in normalize_claim_key(object_surface).split() if part]
    return tokens[0] if tokens else ""


def _infer_object_type(
    object_surface: str,
    *,
    predicate: str,
    matched_entities: list[EntityItem],
) -> str:
    for item in matched_entities:
        mapped = _ENTITY_TYPE_TO_OBJECT_TYPE.get(str(item.type or "").strip().lower())
        if mapped:
            return mapped
    normalized = normalize_claim_key(object_surface)
    head = normalized.split()[0] if normalized.split() else ""
    if predicate == "owns" and head in _OWNABLE_HEAD_TYPES:
        return str(_OWNABLE_HEAD_TYPES.get(head) or "")
    if head in _ACTIVITY_HEADS:
        return "activity"
    if "запах" in normalized or "smell" in normalized:
        return "sensory_object"
    if _PROPER_NAME_RE.search(str(object_surface or "").strip()):
        return "person"
    return "thing"


def _object_alternatives(
    *,
    object_surface: str,
    matched_entities: list[EntityItem],
    normalized_object: str,
    head: str,
) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []

    def _add(value: str) -> None:
        token = str(value or "").strip()
        if not token:
            return
        key = token.lower()
        if key in seen:
            return
        seen.add(key)
        out.append(token)

    _add(object_surface)
    if normalized_object and normalized_object != object_surface.lower():
        _add(normalized_object)
    if head and head.lower() not in {normalized_object, object_surface.lower()}:
        _add(head)

    for item in matched_entities:
        _add(str(getattr(item, "surface", "") or ""))
        _add(str(getattr(item, "canonical", "") or ""))

    for token in _PROPER_NAME_RE.findall(str(object_surface or "")):
        _add(token)

    tail_tokens = [part for part in normalized_object.split() if part]
    if len(tail_tokens) >= 2:
        _add(" ".join(tail_tokens[-2:]))
    if tail_tokens:
        _add(tail_tokens[-1])
    return out[:6]


def _topic_keys(*, predicate: str, object_type: str, matched_entities: list[EntityItem]) -> list[str]:
    out = [predicate]
    if object_type:
        out.append(object_type)
    for item in matched_entities:
        entity_type = str(item.type or "").strip().lower()
        if entity_type:
            out.append(entity_type)
    if predicate in {"likes", "dislikes"}:
        out.append("preference")
    elif predicate == "owns":
        out.append("ownership")
    elif predicate == "uses":
        out.append("usage")
    return _dedupe_tokens(out)


def _trigger_keys(
    *,
    predicate: str,
    head: str,
    normalized_object: str,
    matched_entities: list[EntityItem],
) -> list[str]:
    out = [predicate]
    if head:
        out.append(normalize_claim_key(head))
    out.extend([part for part in normalized_object.split()[:3] if part])
    for item in matched_entities:
        canonical = _slug(str(item.canonical or ""))
        if canonical:
            out.append(canonical)
    return _dedupe_tokens(out)


def _specificity_score(value: str, *, alternatives: list[str]) -> float:
    tokens = [part for part in normalize_claim_key(value).split() if part]
    if not tokens:
        return 0.0
    unique = len(set(tokens))
    alt_bonus = min(2, len([x for x in list(alternatives or []) if str(x).strip()])) * 0.06
    score = (0.18 * min(unique, 4)) + (0.08 * min(len(value), 40) / 8.0) + alt_bonus
    return max(0.0, min(1.0, score))


def _dedupe_tokens(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for raw in list(values or []):
        token = str(raw or "").strip()
        if not token:
            continue
        key = token.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(token)
    return out


def _should_keep_candidate(
    object_surface: str,
    *,
    predicate: str,
    matched_entities: list[EntityItem],
    numeric_facts: list[NumericFact] | None,
) -> bool:
    normalized = normalize_claim_key(object_surface)
    if not normalized or normalized in _GENERIC_OBJECTS:
        return False
    if len(normalized) < 2:
        return False

    if predicate == "uses" and any(normalized.startswith(prefix) for prefix in _ENVIRONMENT_USE_PREFIXES):
        return False
    if predicate == "uses" and any(str(getattr(item, "kind", "") or "").strip().lower() == "python_version" for item in list(numeric_facts or [])):
        if normalized.startswith(("python ", "питон ", "пайтон ")):
            return False

    if predicate == "owns" and len(normalized.split()) < 2:
        head = normalized.split()[0] if normalized.split() else ""
        if head not in _OWNABLE_HEAD_TYPES and not matched_entities:
            return False

    return True


def _has_negation_prefix(text: str, start: int) -> bool:
    left = str(text or "")[max(0, int(start) - 6) : int(start)].lower()
    return bool(re.search(r"\bне\s*$", left))


def _dedupe_candidates(items: list[ClaimCandidate]) -> list[ClaimCandidate]:
    seen: set[tuple[str, str, str]] = set()
    out: list[ClaimCandidate] = []
    for item in list(items or []):
        key = (
            str(item.subject or "").strip().lower(),
            str(item.predicate or "").strip().lower(),
            str(item.normalized_object or "").strip().lower(),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out
