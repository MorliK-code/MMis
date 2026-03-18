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


def _slug_key(value: Any) -> str:
    parts = [part for part in _key_text(value).split() if part]
    return "_".join(parts)


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


def _metadata_entity_keys(metadata: dict[str, Any] | None = None) -> list[str]:
    meta = dict(metadata or {})
    rows = list(meta.get("memory_entities") or [])
    if not rows:
        rows = list(dict(meta.get("memory_analysis") or {}).get("entities") or [])
    out: list[str] = []
    for item in rows:
        row = dict(item or {})
        entity_type = str(row.get("type") or "").strip().lower()
        canonical = str(row.get("canonical") or row.get("surface") or "").strip()
        canonical_key = _slug_key(canonical)
        if entity_type == "gpu_model":
            out.append("gpu")
            if canonical_key:
                out.append(canonical_key)
        elif entity_type == "cpu_model":
            out.append("cpu")
            if canonical_key:
                out.append(canonical_key)
        elif entity_type == "python_version":
            out.append("python")
            version = str(canonical or "").strip().lower().replace("python", "").strip()
            version_key = _slug_key(version)
            if version_key:
                out.append(f"python_{version_key}")
        elif entity_type == "project_name":
            out.append("project")
            if canonical_key:
                out.append(canonical_key)
        elif entity_type == "tool_name":
            out.append("tool")
            if canonical_key:
                out.append(canonical_key)
        elif entity_type == "os_name":
            out.append("os")
            if canonical_key:
                out.append(canonical_key)
        elif entity_type == "person_name":
            out.append("person")
            if canonical_key:
                out.append(canonical_key)
    claim_rows = list(meta.get("claims") or meta.get("claim_candidates") or [])
    for item in claim_rows:
        row = dict(item or {})
        predicate = str(row.get("predicate") or "").strip().lower()
        object_type = str(row.get("object_type") or "").strip().lower()
        normalized_object = str(row.get("obj") or row.get("normalized_object") or row.get("object_surface") or "").strip()
        normalized_key = _slug_key(normalized_object)
        if predicate:
            out.append("claim")
            out.append(predicate)
        if object_type:
            out.append(object_type)
        for token in list(row.get("topic_keys") or []):
            value = _slug_key(token)
            if value:
                out.append(value)
        for token in list(row.get("trigger_keys") or []):
            value = _slug_key(token)
            if value:
                out.append(value)
        if normalized_key:
            out.append(normalized_key)
    fact = dict(meta.get("fact") or {})
    predicate = str(fact.get("predicate") or "").strip().lower()
    value = str(fact.get("value") or "").strip()
    value_key = _slug_key(value)
    if predicate == "environment_gpu_model":
        out.append("gpu")
        if value_key:
            out.append(value_key)
    elif predicate == "environment_cpu_model":
        out.append("cpu")
        if value_key:
            out.append(value_key)
    elif predicate == "environment_runtime_python":
        out.append("python")
        version_key = _slug_key(str(value).replace("python", " ").strip())
        if version_key:
            out.append(f"python_{version_key}")
    elif predicate == "project_name":
        out.append("project")
        if value_key:
            out.append(value_key)
    elif predicate == "environment_tool":
        out.append("tool")
        if value_key:
            out.append(value_key)
    elif predicate == "environment_os":
        out.append("os")
        if value_key:
            out.append(value_key)
    elif predicate == "identity_name":
        out.append("person")
        if value_key:
            out.append(value_key)
    elif predicate == "decision":
        out.append("decision")
        if value_key:
            out.append(value_key)
    elif predicate == "task":
        out.append("task")
        if value_key:
            out.append(value_key)
    elif predicate == "task_goal":
        out.append("task_goal")
        if value_key:
            out.append(value_key)
    elif predicate == "agreed_plan":
        out.append("agreed_plan")
        if value_key:
            out.append(value_key)
    claim = dict(meta.get("claim") or {})
    claim_predicate = str(claim.get("predicate") or "").strip().lower()
    claim_object_type = str(claim.get("object_type") or "").strip().lower()
    claim_object = str(claim.get("obj") or claim.get("object_surface") or "").strip()
    claim_object_key = _slug_key(claim_object)
    if claim_predicate:
        out.append("claim")
        out.append(claim_predicate)
    if claim_object_type:
        out.append(claim_object_type)
    for token in list(claim.get("topic_keys") or []):
        value = _slug_key(token)
        if value:
            out.append(value)
    for token in list(claim.get("trigger_keys") or []):
        value = _slug_key(token)
        if value:
            out.append(value)
    if claim_object_key:
        out.append(claim_object_key)
    return merge_projection_keys(out)


def _metadata_numeric_keys(metadata: dict[str, Any] | None = None) -> list[str]:
    meta = dict(metadata or {})
    rows = list(meta.get("numeric_facts") or [])
    if not rows:
        rows = list(dict(meta.get("memory_analysis") or {}).get("numeric_facts") or [])
    out: list[str] = []
    for item in rows:
        row = dict(item or {})
        kind = str(row.get("kind") or "").strip().lower()
        value = _normalize_number(str(row.get("value") or "").strip())
        unit = str(row.get("unit") or "").strip().lower()
        compact = f"{value}{unit}" if value and unit else value
        if compact and kind not in {"python_version"}:
            out.append(f"value:{compact}")
        if kind == "vram_gb" and compact:
            out.append(f"vram:{compact}")
        elif kind == "ram_gb" and compact:
            out.append(f"ram:{compact}")
        elif kind == "memory_gb" and compact:
            out.append(f"memory:{compact}")
        elif kind == "python_version" and value:
            out.append(f"version:{value}")
        elif kind == "age_years" and value:
            out.append(f"age:{value}")
    fact = dict(meta.get("fact") or {})
    predicate = str(fact.get("predicate") or "").strip().lower()
    value = _normalize_number(str(fact.get("value") or "").strip())
    if predicate in {"environment_gpu_vram_gb", "environment_gpu_vram_size"} and value:
        compact = f"{value}gb"
        out.append(f"value:{compact}")
        out.append(f"vram:{compact}")
    elif predicate in {"environment_ram_gb", "environment_ram_size"} and value:
        compact = f"{value}gb"
        out.append(f"value:{compact}")
        out.append(f"ram:{compact}")
    elif predicate == "environment_memory_gb" and value:
        compact = f"{value}gb"
        out.append(f"value:{compact}")
        out.append(f"memory:{compact}")
    elif predicate == "environment_runtime_python" and value:
        out.append(f"version:{value}")
    elif predicate == "identity_age_years" and value:
        out.append(f"age:{value}")
    return merge_projection_keys(out)


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

    entity_keys = merge_projection_keys(
        list(existing.get("entity_keys") or []),
        _metadata_entity_keys(meta),
        extract_query_entity_keys(text, metadata=meta),
    )
    numeric_keys = merge_projection_keys(
        list(existing.get("numeric_keys") or []),
        _metadata_numeric_keys(meta),
        extract_query_numeric_keys(text, metadata=meta),
    )

    # Only include non-empty keys
    views = {}
    if entity_keys:
        views["entity_keys"] = entity_keys
    if numeric_keys:
        views["numeric_keys"] = numeric_keys
    return views


def ensure_memory_views(text: str, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    meta = dict(metadata or {})
    meta["memory_views"] = build_memory_views(text, metadata=meta)
    return meta


def record_search_text(text: str, metadata: dict[str, Any] | None = None) -> str:
    meta = dict(metadata or {})
    views = build_memory_views(text, metadata=meta)
    entity_keys = list(views.get("entity_keys") or [])
    numeric_keys = list(views.get("numeric_keys") or [])
    search_text = build_memory_search_text(text, metadata=meta, entity_keys=entity_keys, numeric_keys=numeric_keys)
    return search_text or _normalize_text(text)
