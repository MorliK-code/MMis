from __future__ import annotations

import re
from typing import Any

from memory.ingest_analyzer import EntityItem, IngestAnchor, NumericFact


_VRAM_RE = re.compile(r"\b(\d+(?:[.,]\d+)?)\s*(gb|\u0433\u0431)\s*(?:vram|\u0432\u0440\u0430\u043c|\u0432\u0438\u0434\u0435\u043e\u043f\u0430\u043c\w*)\b", re.I)
_RAM_RE = re.compile(r"\b(\d+(?:[.,]\d+)?)\s*(gb|\u0433\u0431)\s*(?:ram|\u043e\u0437\u0443|\u043e\u043f\u0435\u0440\u0430\u0442\u0438\u0432\w*)\b", re.I)
_RAM_REVERSED_RE = re.compile(r"\b(?:ram|\u043e\u0437\u0443|\u043e\u043f\u0435\u0440\u0430\u0442\u0438\u0432\w*)\s*(\d+(?:[.,]\d+)?)(?:\s*(gb|\u0433\u0431))?\b", re.I)
_GB_RE = re.compile(r"\b(\d+(?:[.,]\d+)?)\s*(gb|\u0433\u0431)\b", re.I)
_PYTHON_RE = re.compile(r"\b(?:версия\s+)?(?:python|py|\u043f\u0438\u0442\u043e\u043d\w*|\u043f\u0430\u0439\u0442\u043e\u043d\w*|\u0443\u0434\u0430\u0432\w*)(?:\s+у\s+меня)?\s*(\d+(?:\.\d+){0,2})\b", re.I)
_AGE_RE = re.compile(
    r"\b\u043c\u043d\u0435\s+(?:\u0443\u0436\u0435\s+)?(\d{1,3})(?:\s*(?:\u043b\u0435\u0442|\u0433\u043e\u0434(?:\u0430|\u043e\u0432)?))?\b"
    r"|\bi am\s+(\d{1,3})\s+years?\s+old\b",
    re.I,
)
_AGE_BLOCKERS = ("gb", "\u0433\u0431", "ram", "vram", "python", "rtx", "commit", "commits")


def extract_numeric_facts(
    text: str,
    *,
    entities: list[EntityItem] | None = None,
    anchors: list[IngestAnchor] | None = None,
    context: dict[str, Any] | None = None,
) -> list[NumericFact]:
    src = str(text or "")
    lower = src.lower()
    out: list[NumericFact] = []
    ctx = dict(context or {})
    memory_kind_by_surface = {
        str(key or "").strip(): str(value or "").strip().lower()
        for key, value in dict(ctx.get("memory_kind_by_surface") or {}).items()
        if str(key or "").strip()
    }
    version_kind_by_surface = {
        str(key or "").strip(): str(value or "").strip().lower()
        for key, value in dict(ctx.get("version_kind_by_surface") or {}).items()
        if str(key or "").strip()
    }

    entity_types = {str(item.type or "").strip().lower() for item in list(entities or []) if str(item.type or "").strip()}
    gpu_context = bool({"gpu_model", "vram_size"} & entity_types) or any(
        token in lower for token in ("rtx", "gtx", "rx", "gpu", "\u0432\u0438\u0434\u044e\u0445", "\u0432\u0438\u0434\u0435\u043e\u043a\u0430\u0440\u0442")
    )
    ram_context = "ram_size" in entity_types or any(
        token in lower for token in ("ram", "\u043e\u0437\u0443", "\u043e\u043f\u0435\u0440\u0430\u0442\u0438\u0432")
    )

    for anchor in list(anchors or []):
        kind = str(anchor.kind or "").strip().lower()
        normalized = str(anchor.normalized or "").strip()
        hints = dict(anchor.hints or {})
        if kind == "ram_candidate" and normalized:
            out.append(
                NumericFact(kind="ram_gb", value=_normalize_numeric(normalized), unit="gb", canonical_key="hardware.ram_gb")
            )
            ram_context = True
        elif kind == "memory_size_candidate" and normalized:
            memory_kind = memory_kind_by_surface.get(
                str(anchor.surface or "").strip(),
                str(hints.get("memory_kind") or "").strip().lower(),
            )
            if memory_kind == "vram":
                out.append(
                    NumericFact(kind="vram_gb", value=_normalize_numeric(normalized), unit="gb", canonical_key="hardware.gpu_vram_gb")
                )
                gpu_context = True
            elif memory_kind == "ram":
                out.append(
                    NumericFact(kind="ram_gb", value=_normalize_numeric(normalized), unit="gb", canonical_key="hardware.ram_gb")
                )
                ram_context = True
            else:
                out.append(
                    NumericFact(kind="memory_gb", value=_normalize_numeric(normalized), unit="gb", canonical_key="hardware.memory_gb")
                )
        elif kind == "python_version_candidate" and normalized:
            version_kind = version_kind_by_surface.get(str(anchor.surface or "").strip(), "version")
            if version_kind == "runtime_python":
                out.append(
                    NumericFact(kind="python_version", value=normalized, unit="", canonical_key="environment.python_version")
                )
        elif kind == "person_age_candidate" and normalized:
            out.append(
                NumericFact(kind="age_years", value=int(_normalize_numeric(normalized)), unit="years", canonical_key="identity.age_years")
            )

    for match in _VRAM_RE.finditer(src):
        out.append(
            NumericFact(
                kind="vram_gb",
                value=_normalize_numeric(match.group(1)),
                unit="gb",
                canonical_key="hardware.gpu_vram_gb",
            )
        )

    for match in _RAM_RE.finditer(src):
        out.append(
            NumericFact(
                kind="ram_gb",
                value=_normalize_numeric(match.group(1)),
                unit="gb",
                canonical_key="hardware.ram_gb",
            )
        )

    for match in _RAM_REVERSED_RE.finditer(src):
        out.append(
            NumericFact(
                kind="ram_gb",
                value=_normalize_numeric(match.group(1)),
                unit="gb",
                canonical_key="hardware.ram_gb",
            )
        )

    for match in _GB_RE.finditer(src):
        value = _normalize_numeric(match.group(1))
        if any(item.value == value and item.unit == "gb" for item in out):
            continue
        if gpu_context:
            out.append(NumericFact(kind="vram_gb", value=value, unit="gb", canonical_key="hardware.gpu_vram_gb"))
        elif ram_context:
            out.append(NumericFact(kind="ram_gb", value=value, unit="gb", canonical_key="hardware.ram_gb"))
        else:
            out.append(NumericFact(kind="memory_gb", value=value, unit="gb", canonical_key="hardware.memory_gb"))

    for match in _PYTHON_RE.finditer(src):
        version = str(match.group(1) or "").strip()
        if version:
            out.append(NumericFact(kind="python_version", value=version, unit="", canonical_key="environment.python_version"))

    for match in _AGE_RE.finditer(src):
        raw = match.group(1) or match.group(2)
        if raw:
            if _is_blocked_age_match(src, match):
                continue
            out.append(NumericFact(kind="age_years", value=int(raw), unit="years", canonical_key="identity.age_years"))

    return _dedupe_numeric(out)


def _dedupe_numeric(items: list[NumericFact]) -> list[NumericFact]:
    seen: set[tuple[str, str, str]] = set()
    out: list[NumericFact] = []
    for item in list(items or []):
        key = (
            str(item.kind or "").strip().lower(),
            str(item.value).strip().lower(),
            str(item.unit or "").strip().lower(),
        )
        if not key[0] or key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def _normalize_numeric(value: str | None) -> float | int | str:
    raw = str(value or "").strip().replace(",", ".")
    if not raw:
        return ""
    try:
        number = float(raw)
    except Exception:
        return raw
    if abs(number - round(number)) <= 1e-9:
        return int(round(number))
    return float(number)


def _is_blocked_age_match(text: str, match: re.Match[str]) -> bool:
    value_raw = match.group(1) or match.group(2) or ""
    try:
        age = int(value_raw)
    except Exception:
        return True
    if age <= 0 or age > 120:
        return True
    if match.group(2):
        return False

    tail = str(text[match.end() : match.end() + 24] or "").strip().lower()
    if any(token in tail for token in _AGE_BLOCKERS):
        return True

    if match.group(1):
        next_word = re.match(r"^([a-z\u0430-\u044f\u0451\u0456\u0457\u0454\u0491]+)\b", tail, flags=re.I)
        if next_word is not None:
            token = str(next_word.group(1) or "").strip().lower()
            if token not in {"\u043b\u0435\u0442", "\u0433\u043e\u0434", "\u0433\u043e\u0434\u0430", "\u0433\u043e\u0434\u043e\u0432"}:
                return True

    return False
