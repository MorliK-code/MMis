from __future__ import annotations

import re
from dataclasses import dataclass

from memory.ingest_analyzer import EntityItem


_SPACE_RE = re.compile(r"\s+")
_TRIM_EDGE_RE = re.compile(r"^[\s,.:;!?\"'`()\[\]\-]+|[\s,.:;!?\"'`()\[\]\-]+$")
_TRIM_LEADING_RE = re.compile(
    r"^(?:что|это|вот|такой|такая|такие|для\s+меня|ну|ну\s+это|типа|как\s+бы|about|that|this|the|a|an)\s+",
    re.I,
)
_CLAUSE_TAIL_RE = re.compile(
    r"\s*(?:[.!?;]+\s*|,\s*)(?:кстати|вообще|короче|например|наверное|кажется|если\s+что|by\s+the\s+way|anyway|actually)\b.*$",
    re.I,
)
_CLAUSE_EXPLOSION_RE = re.compile(
    r"\b(?:потому\s+что|если|когда|although|because|if|when|while|чтобы)\b",
    re.I,
)
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
    "что то",
    "something",
    "anything",
    "everything",
    "this",
    "that",
    "it",
}
_WINDOWS_RE = re.compile(r"^windows(?:\s+\d+(?:\.\d+)*)?$", re.I)
_CANONICAL_HEAD_MAP = {
    "windows": "Windows",
    "linux": "Linux",
    "ubuntu": "Ubuntu",
    "debian": "Debian",
    "macos": "macOS",
    "vs code": "VS Code",
}


@dataclass(frozen=True)
class ClaimNormalizationResult:
    object_surface: str
    normalized_object: str
    canonical_object: str = ""
    reject_reason: str = ""


def normalize_claim_object(
    value: str,
    *,
    matched_entities: list[EntityItem] | None = None,
) -> ClaimNormalizationResult | None:
    text = _clean_text(value)
    if not text:
        return None
    text = _strip_discourse_tail(text)
    text = _clean_text(text)
    if not text:
        return None

    normalized = _normalize_text(text)
    if not normalized:
        return None
    if normalized in _STOPWORD_ONLY:
        return None
    if _is_stopword_only(normalized):
        return None
    if _is_clause_explosion(text, normalized):
        return None

    canonical = _canonicalize_object(text, normalized=normalized, matched_entities=matched_entities)
    object_surface = canonical or text
    normalized_object = _normalize_text(object_surface)
    if not normalized_object or normalized_object in _STOPWORD_ONLY:
        return None
    return ClaimNormalizationResult(
        object_surface=object_surface,
        normalized_object=normalized_object,
        canonical_object=canonical,
    )


def normalize_claim_key(value: str) -> str:
    return _normalize_text(value)


def _clean_text(value: str) -> str:
    text = _TRIM_EDGE_RE.sub("", str(value or "").strip())
    text = _TRIM_LEADING_RE.sub("", text)
    return _SPACE_RE.sub(" ", text).strip()


def _strip_discourse_tail(value: str) -> str:
    text = str(value or "").strip()
    text = _CLAUSE_TAIL_RE.sub("", text)
    return text.strip()


def _normalize_text(value: str) -> str:
    text = _clean_text(value).lower()
    text = re.sub(r"[^0-9a-zа-яёіїєґ' _-]", " ", text, flags=re.I)
    return _SPACE_RE.sub(" ", text).strip()


def _is_stopword_only(normalized: str) -> bool:
    tokens = [part for part in str(normalized or "").split() if part]
    if not tokens:
        return True
    return all(token in {"что", "это", "ну", "вот", "the", "a", "an", "that", "this", "it"} for token in tokens)


def _is_clause_explosion(surface: str, normalized: str) -> bool:
    token_count = len([part for part in normalized.split() if part])
    if token_count <= 8:
        return False
    return bool(_CLAUSE_EXPLOSION_RE.search(str(surface or "")))


def _canonicalize_object(
    surface: str,
    *,
    normalized: str,
    matched_entities: list[EntityItem] | None,
) -> str:
    for item in list(matched_entities or []):
        canonical = str(getattr(item, "canonical", "") or "").strip()
        if not canonical:
            continue
        canonical_norm = _normalize_text(canonical)
        if canonical_norm and (canonical_norm == normalized or normalized.endswith(canonical_norm)):
            return canonical

    mapped = _CANONICAL_HEAD_MAP.get(normalized)
    if mapped:
        return mapped

    if _WINDOWS_RE.match(normalized):
        parts = normalized.split()
        if not parts:
            return "Windows"
        if len(parts) == 1:
            return "Windows"
        return "Windows " + " ".join(parts[1:])

    head, _, tail = normalized.partition(" ")
    if head in _CANONICAL_HEAD_MAP and tail:
        return f"{_CANONICAL_HEAD_MAP[head]} {tail}"
    return ""
