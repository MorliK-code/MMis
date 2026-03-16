from __future__ import annotations

import re
from typing import Any, Iterable

from memory.ingest_analyzer import EntityItem


_GPU_RE = re.compile(r"\b(rtx|gtx|rx)\s*[-_ ]?(\d{3,4})(?:\s*[-_ ]?(ti|super|xt))?\b", re.I)
_GPU_SHORT_RE = re.compile(r"\b(\d{4})\s*(ti|super|xt)\b", re.I)
_CPU_RE = re.compile(r"\b(i[3579]-\d{4,5}[a-z]{0,2}|ryzen\s+\d\s+\d{4,5}[a-z]{0,2})\b", re.I)
_PYTHON_RE = re.compile(r"\bpython\s*(\d+(?:\.\d+){1,2})\b", re.I)
_LLM_RE = re.compile(
    r"\b(gpt[-\s]?\d(?:\.\d+)?|gpt-4\.1|claude\s*\d(?:\.\d+)?(?:\s+sonnet)?|qwen\d?(?:\.\d+)?|llama\s*\d(?:\.\d+)?|mistral(?:\s+\w+)?)\b",
    re.I,
)
_PROJECT_RE = re.compile(r"\bmmis\b", re.I)
_OS_RE = re.compile(r"\b(windows|linux|ubuntu|debian|macos|mac\s*os)\b", re.I)
_TOOL_RE = re.compile(r"\b(ollama|docker|vscode|vs code|git|chromadb|chroma db|pyside6|pytest|poetry|uv)\b", re.I)
_VRAM_RE = re.compile(r"\b(\d+(?:[.,]\d+)?)\s*(gb|гб)\s*(?:vram|врам|видеопам)\b", re.I)
_RAM_RE = re.compile(r"\b(\d+(?:[.,]\d+)?)\s*(gb|гб)\s*(?:ram|озу|оператив)\b", re.I)
_PERSON_INTRO_RE = re.compile(
    r"(?:меня\s+зовут|мо[её]\s+имя|my\s+name\s+is)\s+([A-ZА-ЯЁ][A-Za-zА-Яа-яЁёІіЇїЄєҐґ' -]{1,40})",
    re.I,
)

_TOOL_CANONICAL = {
    "ollama": "Ollama",
    "docker": "Docker",
    "vscode": "VS Code",
    "vs code": "VS Code",
    "git": "Git",
    "chromadb": "ChromaDB",
    "chroma db": "ChromaDB",
    "pyside6": "PySide6",
    "pytest": "pytest",
    "poetry": "Poetry",
    "uv": "uv",
}
_OS_CANONICAL = {
    "windows": "Windows",
    "linux": "Linux",
    "ubuntu": "Ubuntu",
    "debian": "Debian",
    "macos": "macOS",
    "mac os": "macOS",
}


def resolve_entities(text: str, *, metadata: dict[str, Any] | None = None) -> list[EntityItem]:
    src = str(text or "")
    lower = src.lower()
    out: list[EntityItem] = []
    gpu_context = any(token in lower for token in ("gpu", "видюх", "видях", "видеокарт", "видеокарта", "geforce", "radeon"))
    nvidia_context = any(token in lower for token in ("nvidia", "geforce"))

    for match in _GPU_RE.finditer(src):
        prefix = str(match.group(1) or "").upper()
        number = str(match.group(2) or "")
        suffix = _title_token(match.group(3))
        canonical = " ".join(part for part in (prefix, number, suffix) if part)
        out.append(EntityItem(type="gpu_model", surface=match.group(0), canonical=canonical, confidence=0.96))

    if nvidia_context:
        for match in _GPU_SHORT_RE.finditer(src):
            number = str(match.group(1) or "")
            suffix = _title_token(match.group(2))
            canonical = " ".join(part for part in (number, suffix) if part)
            out.append(EntityItem(type="gpu_model", surface=match.group(0), canonical=canonical, confidence=0.88))

    for match in _CPU_RE.finditer(src):
        canonical = str(match.group(0) or "").upper().replace("RYZEN", "Ryzen")
        out.append(EntityItem(type="cpu_model", surface=match.group(0), canonical=canonical, confidence=0.94))

    for match in _PYTHON_RE.finditer(src):
        version = str(match.group(1) or "").strip()
        out.append(EntityItem(type="python_version", surface=match.group(0), canonical=version, confidence=0.97))

    for match in _LLM_RE.finditer(src):
        surface = str(match.group(0) or "").strip()
        canonical = _canonical_model(surface)
        out.append(EntityItem(type="llm_model", surface=surface, canonical=canonical, confidence=0.91))

    for match in _PROJECT_RE.finditer(src):
        out.append(EntityItem(type="project_name", surface=match.group(0), canonical="MMis", confidence=0.98))

    for match in _OS_RE.finditer(src):
        surface = str(match.group(0) or "").strip()
        canonical = _OS_CANONICAL.get(surface.lower(), surface.title())
        out.append(EntityItem(type="os_name", surface=surface, canonical=canonical, confidence=0.90))

    for match in _TOOL_RE.finditer(src):
        surface = str(match.group(0) or "").strip()
        canonical = _TOOL_CANONICAL.get(surface.lower(), surface)
        out.append(EntityItem(type="tool_name", surface=surface, canonical=canonical, confidence=0.88))

    for match in _VRAM_RE.finditer(src):
        value = _normalize_number(match.group(1))
        unit = "GB"
        out.append(
            EntityItem(
                type="vram_size",
                surface=match.group(0),
                canonical=f"{value} {unit}",
                confidence=0.92,
            )
        )

    for match in _RAM_RE.finditer(src):
        value = _normalize_number(match.group(1))
        unit = "GB"
        out.append(
            EntityItem(
                type="ram_size",
                surface=match.group(0),
                canonical=f"{value} {unit}",
                confidence=0.90,
            )
        )

    for match in _PERSON_INTRO_RE.finditer(src):
        name = str(match.group(1) or "").strip(" ,.!?;:")
        if name:
            out.append(EntityItem(type="person_name", surface=match.group(0), canonical=name, confidence=0.92))

    meta = dict(metadata or {})
    project_name = str(meta.get("project_name") or meta.get("project") or "").strip()
    if project_name:
        out.append(EntityItem(type="project_name", surface=project_name, canonical=project_name, confidence=0.86))

    return _dedupe_entities(out)


def _dedupe_entities(items: Iterable[EntityItem]) -> list[EntityItem]:
    seen: set[tuple[str, str]] = set()
    out: list[EntityItem] = []
    for item in items:
        key = (str(item.type or "").strip().lower(), str(item.canonical or "").strip().lower())
        if not key[0] or not key[1] or key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def _title_token(value: str | None) -> str:
    token = str(value or "").strip().lower()
    if not token:
        return ""
    if token == "ti":
        return "Ti"
    if token == "xt":
        return "XT"
    return token.title()


def _canonical_model(value: str) -> str:
    parts = [part for part in re.split(r"\s+", str(value or "").strip()) if part]
    if not parts:
        return ""
    first = parts[0].upper() if parts[0].lower().startswith("gpt") else parts[0].title()
    rest = [part.upper() if part.isalpha() and len(part) <= 4 else part for part in parts[1:]]
    return " ".join([first, *rest]).strip()


def _normalize_number(value: str | None) -> str:
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
