from __future__ import annotations

import re

from memory.ingest_analyzer import EntityItem, NumericFact


_VRAM_RE = re.compile(r"\b(\d+(?:[.,]\d+)?)\s*(gb|гб)\s*(?:vram|врам|видеопам)\b", re.I)
_RAM_RE = re.compile(r"\b(\d+(?:[.,]\d+)?)\s*(gb|гб)\s*(?:ram|озу|оператив)\b", re.I)
_GB_RE = re.compile(r"\b(\d+(?:[.,]\d+)?)\s*(gb|гб)\b", re.I)
_PYTHON_RE = re.compile(r"\bpython\s*(\d+(?:\.\d+){1,2})\b", re.I)
_AGE_RE = re.compile(r"\bмне\s+(\d{1,3})(?:\s*(?:лет|год(?:а|ов)?))?\b|\bi am\s+(\d{1,3})\s+years?\s+old\b", re.I)
_AGE_BLOCKERS = ("gb", "гб", "ram", "vram", "python", "rtx", "commit", "commits")


def extract_numeric_facts(text: str, *, entities: list[EntityItem] | None = None) -> list[NumericFact]:
    src = str(text or "")
    lower = src.lower()
    out: list[NumericFact] = []

    entity_types = {str(item.type or "").strip().lower() for item in list(entities or []) if str(item.type or "").strip()}
    gpu_context = bool({"gpu_model", "vram_size"} & entity_types) or any(
        token in lower for token in ("rtx", "gtx", "rx", "gpu", "видюх", "видеокарт", "видеокарта")
    )
    ram_context = "ram_size" in entity_types or any(token in lower for token in ("ram", "озу", "оператив"))

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

    for match in _GB_RE.finditer(src):
        value = _normalize_numeric(match.group(1))
        if any(item.value == value and item.unit == "gb" for item in out):
            continue
        if gpu_context:
            out.append(
                NumericFact(
                    kind="vram_gb",
                    value=value,
                    unit="gb",
                    canonical_key="hardware.gpu_vram_gb",
                )
            )
        elif ram_context:
            out.append(
                NumericFact(
                    kind="ram_gb",
                    value=value,
                    unit="gb",
                    canonical_key="hardware.ram_gb",
                )
            )
        else:
            out.append(
                NumericFact(
                    kind="memory_gb",
                    value=value,
                    unit="gb",
                    canonical_key="hardware.memory_gb",
                )
            )

    for match in _PYTHON_RE.finditer(src):
        version = str(match.group(1) or "").strip()
        if version:
            out.append(
                NumericFact(
                    kind="python_version",
                    value=version,
                    unit="",
                    canonical_key="environment.python_version",
                )
            )

    for match in _AGE_RE.finditer(src):
        raw = match.group(1) or match.group(2)
        if raw:
            if _is_blocked_age_match(src, match):
                continue
            out.append(
                NumericFact(
                    kind="age_years",
                    value=int(raw),
                    unit="years",
                    canonical_key="identity.age_years",
                )
            )

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

    tail = str(text[match.end() : match.end() + 24] or "").strip().lower()
    if any(token in tail for token in _AGE_BLOCKERS):
        return True

    if match.group(1):
        next_word = re.match(r"^([a-zа-яёіїєґ]+)\b", tail, flags=re.I)
        if next_word is not None:
            token = str(next_word.group(1) or "").strip().lower()
            if token not in {"лет", "год", "года", "годов"}:
                return True

    return False
