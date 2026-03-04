from __future__ import annotations

import re
from typing import Any


_DEVICE_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\brtx\s*3050\s*ti\b", re.I), "RTX 3050 Ti"),
    (re.compile(r"\brtx\s*3060\b", re.I), "RTX 3060"),
    (re.compile(r"\brtx\s*3070\b", re.I), "RTX 3070"),
    (re.compile(r"\brtx\s*3080\b", re.I), "RTX 3080"),
    (re.compile(r"\bi5-11400h\b", re.I), "i5-11400H"),
    (re.compile(r"\bi7-12700h\b", re.I), "i7-12700H"),
    (re.compile(r"\biphone\s*\d{1,2}\b", re.I), "iPhone"),
]

_SOFTWARE_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bvs\s*code\b|\bvisual\s+studio\s+code\b", re.I), "VS Code"),
    (re.compile(r"\bdocker(?:\s+compose)?\b", re.I), "Docker"),
    (re.compile(r"\bollama\b", re.I), "Ollama"),
    (re.compile(r"\bchromadb\b|\bchroma\s*db\b|\bchroma\b", re.I), "ChromaDB"),
    (re.compile(r"\bpyside6\b|\bpyside\b", re.I), "PySide6"),
    (re.compile(r"\bpython\b", re.I), "Python"),
    (re.compile(r"\bgit\b", re.I), "Git"),
]

_PROJECT_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bmmis\b", re.I), "MMis"),
]

_OS_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bwindows\b|\bwin11\b|\bwin10\b", re.I), "Windows"),
    (re.compile(r"\blinux\b|\bubuntu\b|\bdebian\b", re.I), "Linux"),
    (re.compile(r"\bmacos\b|\bmac\s*os\b", re.I), "macOS"),
]

_ENTITY_TOPIC_MAP = {
    "Docker": "docker",
    "VS Code": "vscode",
    "Ollama": "ollama",
    "ChromaDB": "chromadb",
    "PySide6": "qt",
    "Git": "git",
    "Python": "python",
    "Windows": "windows",
    "Linux": "linux",
    "MMis": "ui",
}

_TAGGABLE_ENTITY_KINDS = {"device", "software", "project", "os", "places"}


def extract_entities(text: str) -> dict[str, list[str]]:
    src = str(text or "")
    out = {
        "device": _find_entities(src, _DEVICE_PATTERNS),
        "software": _find_entities(src, _SOFTWARE_PATTERNS),
        "project": _find_entities(src, _PROJECT_PATTERNS),
        "os": _find_entities(src, _OS_PATTERNS),
    }
    return {k: v for k, v in out.items() if v}


def infer_topics_from_entities(entities: dict[str, Any] | None) -> list[str]:
    rows = dict(entities or {})
    out: list[str] = []
    seen: set[str] = set()
    for values in rows.values():
        for value in list(values or []):
            name = str(value or "").strip()
            if not name:
                continue
            topic = _ENTITY_TOPIC_MAP.get(name, "")
            if not topic:
                continue
            key = f"topic_{topic}"
            if key in seen:
                continue
            seen.add(key)
            out.append(key)
    return out


def flatten_entity_tags(entities: dict[str, Any] | None) -> list[str]:
    rows = dict(entities or {})
    out: list[str] = []
    seen: set[str] = set()
    for kind, values in rows.items():
        k = str(kind or "").strip().lower()
        if not k or k not in _TAGGABLE_ENTITY_KINDS:
            continue
        for value in list(values or []):
            token = _slug(str(value or ""))
            if not token:
                continue
            tag = f"entity_{k}_{token}"
            if tag in seen:
                continue
            seen.add(tag)
            out.append(tag)
    return out


def _find_entities(text: str, patterns: list[tuple[re.Pattern[str], str]]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for rx, canonical in patterns:
        if not rx.search(text):
            continue
        if canonical in seen:
            continue
        seen.add(canonical)
        out.append(canonical)
    return out


def _slug(value: str) -> str:
    token = re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower()).strip("_")
    return token
