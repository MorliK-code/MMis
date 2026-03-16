"""Projection helpers for retrieval-friendly memory representations."""

from __future__ import annotations

import re
from typing import Any


_SPACE_RE = re.compile(r"\s+")
_GPU_MODEL_RE = re.compile(r"\b(rtx|gtx|rx)\s*[-_ ]?(\d{3,4})(?:\s*[-_ ]?(ti|super|xt))?\b", re.IGNORECASE)
_GPU_MODEL_SHORT_RE = re.compile(r"\b(\d{4})\s*(ti|super|xt)\b", re.IGNORECASE)
_PYTHON_VERSION_RE = re.compile(r"\bpython\s*(\d+(?:\.\d+)+)\b", re.IGNORECASE)
_MEASUREMENT_RE = re.compile(r"\b(\d+(?:[.,]\d+)?)\s*(tb|gb|gib|mb|mhz|ghz|hz)\b", re.IGNORECASE)
_DATE_RE = re.compile(r"\b(\d{1,2})\s+([a-z\u0400-\u04ff]+)\s+(\d{4})\b", re.IGNORECASE)
_YEAR_RE = re.compile(r"\b(19\d{2}|20\d{2}|21\d{2})\b")

_ENTITY_ALIASES: dict[str, tuple[str, ...]] = {
    "gpu": (
        "gpu",
        "graphics card",
        "graphics adapter",
        "video card",
        "videocard",
        "видеокарта",
        "видюха",
        "видяха",
        "видео карта",
        "графическая карта",
        "nvidia",
        "geforce",
        "radeon",
    ),
    "vram": (
        "vram",
        "video memory",
        "видеопамять",
        "память видеокарты",
    ),
    "python": (
        "python",
        "py",
    ),
}

_MONTHS: dict[str, int] = {
    "january": 1,
    "jan": 1,
    "januar": 1,
    "января": 1,
    "январь": 1,
    "february": 2,
    "feb": 2,
    "февраля": 2,
    "февраль": 2,
    "march": 3,
    "mar": 3,
    "марта": 3,
    "март": 3,
    "april": 4,
    "apr": 4,
    "апреля": 4,
    "апрель": 4,
    "may": 5,
    "мая": 5,
    "май": 5,
    "june": 6,
    "jun": 6,
    "июня": 6,
    "июнь": 6,
    "july": 7,
    "jul": 7,
    "июля": 7,
    "июль": 7,
    "august": 8,
    "aug": 8,
    "августа": 8,
    "август": 8,
    "september": 9,
    "sep": 9,
    "sept": 9,
    "сентября": 9,
    "сентябрь": 9,
    "october": 10,
    "oct": 10,
    "октября": 10,
    "октябрь": 10,
    "november": 11,
    "nov": 11,
    "ноября": 11,
    "ноябрь": 11,
    "december": 12,
    "dec": 12,
    "декабря": 12,
    "декабрь": 12,
}

_LOW_SIGNAL_CANONICAL_TEXTS = {
    "api",
    "ui",
    "cli",
    "chat",
    "system",
}


def merge_projection_keys(*groups: list[str] | tuple[str, ...] | None) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for group in groups:
        for value in list(group or []):
            item = str(value or "").strip().lower()
            if not item or item in seen:
                continue
            seen.add(item)
            out.append(item)
    return out


def _normalize_text(text: str) -> str:
    src = str(text or "").strip().lower()
    if not src:
        return ""
    replacements = {
        "\u201c": '"',
        "\u201d": '"',
        "\u00ab": '"',
        "\u00bb": '"',
        "\u2019": "'",
        "\u2013": " ",
        "\u2014": " ",
        "\u2212": "-",
        "\u00a0": " ",
        "\n": " ",
        "\r": " ",
        "\t": " ",
        "/": " ",
        "\\": " ",
        "|": " ",
        "_": " ",
    }
    for old, new in replacements.items():
        src = src.replace(old, new)
    src = re.sub(r"[^0-9a-z\u0400-\u04ff\.\-\+\:\s]", " ", src)
    return _SPACE_RE.sub(" ", src).strip()


def _key_text(key: str) -> str:
    return _normalize_text(str(key or "").replace(":", " ").replace("_", " ").replace(".", " "))


def _collect_canonical_parts(text: str, metadata: dict[str, Any] | None = None) -> list[str]:
    meta = dict(metadata or {})
    fact = dict(meta.get("fact") or {})
    parts: list[str] = []

    canonical_key = str(meta.get("canonical_key") or fact.get("canonical_key") or "").strip()
    if canonical_key:
        parts.append(canonical_key.replace(".", " ").replace("_", " "))

    for key in ("subject", "predicate", "value", "relation", "key"):
        value = fact.get(key)
        if value is not None:
            parts.append(str(value))

    for key in ("title", "document_source", "runtime_key"):
        value = meta.get(key)
        if value is not None:
            parts.append(str(value))

    if text:
        parts.append(str(text))
    return [part for part in parts if str(part or "").strip()]


def _contains_alias(text: str, markers: tuple[str, ...]) -> bool:
    src = str(text or "")
    if not src:
        return False
    return any(marker in src for marker in markers)


def _normalize_number(value: str) -> str:
    raw = str(value or "").strip().replace(",", ".")
    if not raw:
        return ""
    try:
        number = float(raw)
    except Exception:
        return raw
    if abs(number - round(number)) <= 1e-9:
        return str(int(round(number)))
    return str(number).rstrip("0").rstrip(".")


def _canonical_text(text: str, metadata: dict[str, Any] | None = None) -> str:
    parts = _collect_canonical_parts(text="", metadata=metadata)
    if not parts:
        return ""
    return _normalize_text(" ".join(parts))


def _preserved_canonical_text(value: Any) -> str:
    canonical = str(value or "").strip()
    if not canonical:
        return ""
    if canonical.lower() in _LOW_SIGNAL_CANONICAL_TEXTS:
        return ""
    return canonical


def extract_query_entity_keys(text: str, *, metadata: dict[str, Any] | None = None) -> list[str]:
    src = _normalize_text(" ".join(_collect_canonical_parts(text=text, metadata=metadata)))
    if not src:
        return []

    keys: list[str] = []
    gpu_context = _contains_alias(src, _ENTITY_ALIASES["gpu"])
    if gpu_context:
        keys.append("gpu")
    if _contains_alias(src, _ENTITY_ALIASES["vram"]):
        keys.append("vram")
    if _contains_alias(src, _ENTITY_ALIASES["python"]):
        keys.append("python")

    for match in _GPU_MODEL_RE.finditer(src):
        prefix = str(match.group(1) or "").lower()
        number = str(match.group(2) or "").strip()
        suffix = str(match.group(3) or "").strip().lower()
        keys.append("_".join([part for part in (prefix, number, suffix) if part]))
        keys.append("gpu")

    if gpu_context:
        for match in _GPU_MODEL_SHORT_RE.finditer(src):
            number = str(match.group(1) or "").strip()
            suffix = str(match.group(2) or "").strip().lower()
            prefix = "rtx" if int(number) >= 2000 else "gtx"
            keys.append("_".join([part for part in (prefix, number, suffix) if part]))

    for match in _PYTHON_VERSION_RE.finditer(src):
        version = str(match.group(1) or "").strip()
        if version:
            keys.append("python")
            keys.append(f"python_{version.replace('.', '_')}")

    return merge_projection_keys(keys)


def extract_query_numeric_keys(text: str, *, metadata: dict[str, Any] | None = None) -> list[str]:
    src = _normalize_text(" ".join(_collect_canonical_parts(text=text, metadata=metadata)))
    if not src:
        return []

    keys: list[str] = []
    gpu_context = _contains_alias(src, _ENTITY_ALIASES["gpu"]) or bool(_GPU_MODEL_RE.search(src))
    if _contains_alias(src, _ENTITY_ALIASES["vram"]):
        gpu_context = True

    for match in _MEASUREMENT_RE.finditer(src):
        value = _normalize_number(str(match.group(1) or ""))
        unit = str(match.group(2) or "").lower()
        if not value or not unit:
            continue
        compact = f"{value}{unit}"
        keys.append(f"value:{compact}")
        if gpu_context and unit in {"tb", "gb", "gib", "mb"}:
            keys.append(f"vram:{compact}")

    for match in _PYTHON_VERSION_RE.finditer(src):
        version = str(match.group(1) or "").strip()
        if version:
            keys.append(f"version:{version}")

    for match in _DATE_RE.finditer(src):
        day = int(match.group(1))
        month_raw = str(match.group(2) or "").strip().lower()
        year = int(match.group(3))
        month = _MONTHS.get(month_raw)
        if month is None:
            continue
        keys.append(f"date:{year:04d}-{month:02d}-{day:02d}")
        keys.append(f"year:{year:04d}")

    if not any(item.startswith("year:") for item in keys):
        for match in _YEAR_RE.finditer(src):
            year = str(match.group(1) or "").strip()
            if year:
                keys.append(f"year:{year}")

    return merge_projection_keys(keys)


def build_memory_search_text(
    text: str,
    *,
    metadata: dict[str, Any] | None = None,
    entity_keys: list[str] | None = None,
    numeric_keys: list[str] | None = None,
) -> str:
    normalized_text = _normalize_text(text)
    canonical_text = _canonical_text(text=text, metadata=metadata)
    entity_values = merge_projection_keys(entity_keys, extract_query_entity_keys(text, metadata=metadata))
    numeric_values = merge_projection_keys(numeric_keys, extract_query_numeric_keys(text, metadata=metadata))
    parts = [
        normalized_text,
        canonical_text,
        *[_key_text(key) for key in entity_values],
        *[_key_text(key) for key in numeric_values],
    ]
    search_parts = merge_projection_keys(parts)
    return " ".join(search_parts).strip()


def build_memory_views(text: str, *, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    meta = dict(metadata or {})
    existing = dict(meta.get("memory_views") or {})

    normalized_text = str(existing.get("normalized_text") or "").strip() or _normalize_text(text)
    canonical_text = _preserved_canonical_text(existing.get("canonical_text")) or _canonical_text(text=text, metadata=meta)

    entity_keys = merge_projection_keys(
        list(existing.get("entity_keys") or []),
        extract_query_entity_keys(text, metadata=meta),
    )
    numeric_keys = merge_projection_keys(
        list(existing.get("numeric_keys") or []),
        extract_query_numeric_keys(text, metadata=meta),
    )

    # Build search projection from the raw record text, not from the previous
    # derived search_text, otherwise repeated rebuilds self-amplify tokens.
    search_seed = str(text or "").strip() or normalized_text or canonical_text
    search_text = build_memory_search_text(
        search_seed,
        metadata=meta,
        entity_keys=entity_keys,
        numeric_keys=numeric_keys,
    )

    return {
        "normalized_text": normalized_text,
        "canonical_text": canonical_text,
        "search_text": search_text,
        "entity_keys": entity_keys,
        "numeric_keys": numeric_keys,
    }


def ensure_memory_views(text: str, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    meta = dict(metadata or {})
    meta["memory_views"] = build_memory_views(text, metadata=meta)
    return meta


def record_search_text(text: str, metadata: dict[str, Any] | None = None) -> str:
    views = build_memory_views(text, metadata=metadata)
    value = str(views.get("search_text") or "").strip()
    return value or _normalize_text(text)
