from __future__ import annotations

import re
from typing import Any, Iterable

from memory.ingest_analyzer import EntityItem, IngestAnchor


_GPU_RE = re.compile(r"\b(rtx|gtx|rx)\s*[-_ ]?(\d{3,4})(?:\s*[-_ ]?(ti|super|xt))?\b", re.I)
_GPU_SHORT_RE = re.compile(r"\b(\d{4})\s*(ti|super|xt)\b", re.I)
_CPU_RE = re.compile(r"\b(i[3579]-\d{4,5}[a-z]{0,2}|ryzen\s+\d\s+\d{4,5}[a-z]{0,2})\b", re.I)
_PYTHON_RE = re.compile(r"\b(?:версия\s+)?(?:python|питон\w*|пайтон\w*|удав\w*)(?:\s+у\s+меня)?\s*(\d+(?:\.\d+){0,2})\b", re.I)
_LLM_RE = re.compile(
    r"\b(gpt[-\s]?\d(?:\.\d+)?|gpt-4\.1|claude\s*\d(?:\.\d+)?(?:\s+sonnet)?|qwen\d?(?:\.\d+)?|llama\s*\d(?:\.\d+)?|mistral(?:\s+\w+)?)\b",
    re.I,
)
_PROJECT_RE = re.compile(r"\bmmis\b", re.I)
_OS_RE = re.compile(r"\b(windows|linux|ubuntu|debian|macos|mac\s*os|винда|винду|виндовс)\b", re.I)
_TOOL_RE = re.compile(r"\b(ollama|docker|vscode|vs code|git|chromadb|chroma db|pyside6|pytest|poetry|uv)\b", re.I)
_VRAM_RE = re.compile(r"\b(\d+(?:[.,]\d+)?)\s*(gb|гб)\s*(?:vram|врам|видеопам\w*)\b", re.I)
_RAM_RE = re.compile(r"\b(\d+(?:[.,]\d+)?)\s*(gb|гб)\s*(?:ram|озу|оператив\w*)\b", re.I)
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
    "винда": "Windows",
    "винду": "Windows",
    "виндовс": "Windows",
    "linux": "Linux",
    "ubuntu": "Ubuntu",
    "debian": "Debian",
    "macos": "macOS",
    "mac os": "macOS",
}


def resolve_entities(
    text: str,
    *,
    metadata: dict[str, Any] | None = None,
    anchors: list[IngestAnchor] | None = None,
    context: dict[str, Any] | None = None,
) -> list[EntityItem]:
    src = str(text or "")
    lower = src.lower()
    out: list[EntityItem] = []
    nvidia_context = any(token in lower for token in ("nvidia", "geforce"))
    ctx = dict(context or {})
    memory_kind_by_surface = {
        str(key or "").strip(): str(value or "").strip().lower()
        for key, value in dict(ctx.get("memory_kind_by_surface") or {}).items()
        if str(key or "").strip()
    }

    for anchor in list(anchors or []):
        kind = str(anchor.kind or "").strip().lower()
        normalized = str(anchor.normalized or "").strip()
        hints = dict(anchor.hints or {})
        if kind == "gpu_model_candidate" and normalized:
            out.append(EntityItem(type="gpu_model", surface=anchor.surface, canonical=normalized, confidence=max(0.88, float(anchor.confidence or 0.0))))
        elif kind == "python_version_candidate" and normalized:
            out.append(EntityItem(type="python_version", surface=anchor.surface, canonical=normalized, confidence=max(0.94, float(anchor.confidence or 0.0))))
        elif kind == "os_candidate" and normalized:
            out.append(EntityItem(type="os_name", surface=anchor.surface, canonical=normalized, confidence=max(0.88, float(anchor.confidence or 0.0))))
        elif kind == "ram_candidate" and normalized:
            out.append(EntityItem(type="ram_size", surface=anchor.surface, canonical=f"{normalized} GB", confidence=max(0.90, float(anchor.confidence or 0.0))))
        elif kind == "memory_size_candidate" and normalized:
            resolved_kind = memory_kind_by_surface.get(str(anchor.surface or "").strip(), str(hints.get("memory_kind") or "").strip().lower())
            if resolved_kind == "vram":
                out.append(EntityItem(type="vram_size", surface=anchor.surface, canonical=f"{normalized} GB", confidence=max(0.90, float(anchor.confidence or 0.0))))
            elif resolved_kind == "ram":
                out.append(EntityItem(type="ram_size", surface=anchor.surface, canonical=f"{normalized} GB", confidence=max(0.88, float(anchor.confidence or 0.0))))

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
        out.append(EntityItem(type="llm_model", surface=surface, canonical=_canonical_model(surface), confidence=0.91))

    for match in _PROJECT_RE.finditer(src):
        out.append(EntityItem(type="project_name", surface=match.group(0), canonical="MMis", confidence=0.98))

    for match in _OS_RE.finditer(src):
        surface = str(match.group(0) or "").strip()
        out.append(EntityItem(type="os_name", surface=surface, canonical=_OS_CANONICAL.get(surface.lower(), surface.title()), confidence=0.90))

    for match in _TOOL_RE.finditer(src):
        surface = str(match.group(0) or "").strip()
        out.append(EntityItem(type="tool_name", surface=surface, canonical=_TOOL_CANONICAL.get(surface.lower(), surface), confidence=0.88))

    for match in _VRAM_RE.finditer(src):
        value = _normalize_number(match.group(1))
        out.append(EntityItem(type="vram_size", surface=match.group(0), canonical=f"{value} GB", confidence=0.92))

    for match in _RAM_RE.finditer(src):
        value = _normalize_number(match.group(1))
        out.append(EntityItem(type="ram_size", surface=match.group(0), canonical=f"{value} GB", confidence=0.90))

    for match in _PERSON_INTRO_RE.finditer(src):
        name = str(match.group(1) or "").strip(" ,.!?;:")
        if name:
            out.append(EntityItem(type="person_name", surface=match.group(0), canonical=name, confidence=0.92))

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
