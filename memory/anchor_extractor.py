from __future__ import annotations

import re
from typing import Any

from memory.ingest_analyzer import IngestAnchor


_GPU_MODEL_RE = re.compile(r"\b(?:rtx|gtx|rx)\s*[-_ ]?\d{3,4}(?:\s*[-_ ]?(?:ti|super|xt))?\b", re.I)
_GPU_SHORT_RE = re.compile(r"\b\d{4}\s*(?:ti|super|xt)\b", re.I)
_RAM_RE = re.compile(r"\b(\d+(?:[.,]\d+)?)\s*(gb|\u0433\u0431)\s*(?:ram|\u043e\u0437\u0443|\u043e\u043f\u0435\u0440\u0430\u0442\u0438\u0432\w*)\b", re.I)
_RAM_REVERSED_RE = re.compile(r"\b(?:ram|\u043e\u0437\u0443|\u043e\u043f\u0435\u0440\u0430\u0442\u0438\u0432\w*)\s*(\d+(?:[.,]\d+)?)(?:\s*(gb|\u0433\u0431))?\b", re.I)
_VRAM_RE = re.compile(r"\b(\d+(?:[.,]\d+)?)\s*(gb|\u0433\u0431)\s*(?:vram|\u0432\u0440\u0430\u043c|\u0432\u0438\u0434\u0435\u043e\u043f\u0430\u043c\w*)\b", re.I)
_MEMORY_RE = re.compile(r"\b(\d+(?:[.,]\d+)?)\s*(gb|\u0433\u0431)\s*(?:memory|\u043f\u0430\u043c\u044f\u0442\w*)\b", re.I)
_AGE_RE = re.compile(
    r"\b(\d{1,3})\s*(?:\u0433\u043e\u0434(?:\u0438\u043a|\u0430|\u043e\u0432)?|\u043b\u0435\u0442)\b"
    r"|\bi am\s+(\d{1,3})\s+years?\s+old\b"
    r"|\b\u043c\u043d\u0435\s+(?:\u0443\u0436\u0435\s+)?(\d{1,3})\b",
    re.I,
)
_OS_RE = re.compile(r"\b(windows|linux|ubuntu|debian|macos|mac\s*os|винда|винду|виндовс)\b", re.I)
_PYTHON_RE = re.compile(r"\b(?:версия\s+)?(?:python|py|питон\w*|пайтон\w*|удав\w*)(?:\s+у\s+меня)?\s*(\d+(?:\.\d+){0,2})\b", re.I)
_OS_MENTION_RE = re.compile(r"\b(?:windows|linux|ubuntu|debian|macos|mac\s*os|винда|винду|виндовс|винды|ос|операционк\w*|система)\b", re.I)
_PYTHON_MENTION_RE = re.compile(r"\b(?:python|py|питон\w*|пайтон\w*|удав\w*)\b", re.I)
_CURRENT_OS_CONTEXT_RE = re.compile(
    r"(?:сейчас|теперь|currently|current|now)[^,;.!?\n]{0,32}(windows|linux|ubuntu|debian|macos|mac\s*os|винда|винду|виндовс)",
    re.I,
)
_PAST_OS_CONTEXT_RE = re.compile(
    r"(?:до этого|раньше|ранее|before that|previously|used to)[^,;.!?\n]{0,32}(windows|linux|ubuntu|debian|macos|mac\s*os|винда|винду|виндовс)",
    re.I,
)
_PRICE_RE = re.compile(
    r"\b(\d+(?:[.,]\d+)?)\s*(?:\$|usd|uah|\u0433\u0440\u043d|\u20b4|\u0440\u0443\u0431|eur|\u20ac)\b",
    re.I,
)
_AGE_BLOCKERS = ("gb", "\u0433\u0431", "ram", "vram", "python", "rtx", "commit", "commits", "usd", "eur", "\u0433\u0440\u043d")


def extract_anchors(text: str, *, metadata: dict[str, Any] | None = None) -> list[IngestAnchor]:
    src = str(text or "")
    lower = src.lower()
    _ = dict(metadata or {})
    out: list[IngestAnchor] = []
    current_os_tokens = {
        str(match.group(1) or "").strip().lower()
        for match in _CURRENT_OS_CONTEXT_RE.finditer(src)
        if str(match.group(1) or "").strip()
    }
    past_os_tokens = {
        str(match.group(1) or "").strip().lower()
        for match in _PAST_OS_CONTEXT_RE.finditer(src)
        if str(match.group(1) or "").strip()
    }

    gpu_matches = list(_GPU_MODEL_RE.finditer(src))
    gpu_short_surfaces = [match.group(0) for match in _GPU_SHORT_RE.finditer(src)]
    gpu_context = bool(gpu_matches or gpu_short_surfaces) or any(
        token in lower
        for token in (
            "gpu",
            "\u0432\u0438\u0434\u044f\u0445",
            "\u0432\u0438\u0434\u044e\u0445",
            "\u0432\u0438\u0434\u0435\u043e\u043a\u0430\u0440\u0442",
            "geforce",
            "nvidia",
            "radeon",
        )
    )

    for match in gpu_matches:
        out.append(
            IngestAnchor(
                kind="gpu_model_candidate",
                surface=match.group(0),
                normalized=_normalize_gpu(match.group(0)),
                confidence=0.96,
                hints={"domain": "hardware_gpu"},
            )
        )

    for match in _GPU_SHORT_RE.finditer(src):
        if any(full.start() <= match.start() and match.end() <= full.end() for full in gpu_matches):
            continue
        out.append(
            IngestAnchor(
                kind="gpu_model_candidate",
                surface=match.group(0),
                normalized=_normalize_gpu(match.group(0)),
                confidence=0.88 if gpu_context else 0.78,
                hints={"domain": "hardware_gpu", "short_form": True},
            )
        )

    for match in _RAM_RE.finditer(src):
        out.append(
            IngestAnchor(
                kind="ram_candidate",
                surface=match.group(0),
                normalized=_normalize_size(match.group(1)),
                confidence=0.94,
                hints={"domain": "hardware_ram", "unit": "gb"},
            )
        )

    for match in _RAM_REVERSED_RE.finditer(src):
        value = str(match.group(1) or "").strip()
        if not value:
            continue
        out.append(
            IngestAnchor(
                kind="ram_candidate",
                surface=match.group(0),
                normalized=_normalize_size(value),
                confidence=0.88 if str(match.group(2) or "").strip() else 0.80,
                hints={
                    "domain": "hardware_ram",
                    "unit": "gb",
                    "inferred_unit": not bool(str(match.group(2) or "").strip()),
                },
            )
        )

    for match in _VRAM_RE.finditer(src):
        out.append(
            IngestAnchor(
                kind="memory_size_candidate",
                surface=match.group(0),
                normalized=_normalize_size(match.group(1)),
                confidence=0.93,
                hints={"domain": "hardware_gpu", "memory_kind": "vram", "unit": "gb"},
            )
        )

    for match in _MEMORY_RE.finditer(src):
        domain = "hardware_gpu" if gpu_context else "memory_generic"
        memory_kind = "vram" if gpu_context else "memory"
        out.append(
            IngestAnchor(
                kind="memory_size_candidate",
                surface=match.group(0),
                normalized=_normalize_size(match.group(1)),
                confidence=0.84 if gpu_context else 0.72,
                hints={"domain": domain, "memory_kind": memory_kind, "unit": "gb"},
            )
        )

    for match in _AGE_RE.finditer(src):
        value = match.group(1) or match.group(2) or match.group(3)
        if not value or _is_blocked_age_match(src, match):
            continue
        out.append(
            IngestAnchor(
                kind="person_age_candidate",
                surface=match.group(0),
                normalized=str(int(value)),
                confidence=0.93,
                hints={"domain": "identity"},
            )
        )

    for match in _OS_RE.finditer(src):
        surface = str(match.group(0) or "").strip()
        temporal_state = _os_temporal_state(surface, current_tokens=current_os_tokens, past_tokens=past_os_tokens)
        out.append(
            IngestAnchor(
                kind="os_candidate",
                surface=surface,
                normalized=_normalize_os(surface),
                confidence=0.90,
                hints={
                    "domain": "environment_os",
                    **({"temporal_state": temporal_state} if temporal_state else {}),
                },
            )
        )

    for match in _OS_MENTION_RE.finditer(src):
        surface = str(match.group(0) or "").strip()
        if not surface:
            continue
        temporal_state = _os_temporal_state(surface, current_tokens=current_os_tokens, past_tokens=past_os_tokens)
        out.append(
            IngestAnchor(
                kind="os_mention_candidate",
                surface=surface,
                normalized=_normalize_os_mention(surface),
                confidence=0.72,
                hints={
                    "domain": "environment_os",
                    "mention_only": True,
                    **({"temporal_state": temporal_state} if temporal_state else {}),
                },
            )
        )

    for match in _PYTHON_RE.finditer(src):
        version = str(match.group(1) or "").strip()
        if not version:
            continue
        out.append(
            IngestAnchor(
                kind="python_version_candidate",
                surface=match.group(0),
                normalized=version,
                confidence=0.97,
                hints={"domain": "environment_python"},
            )
        )

    for match in _PYTHON_MENTION_RE.finditer(src):
        surface = str(match.group(0) or "").strip()
        if not surface:
            continue
        out.append(
            IngestAnchor(
                kind="python_present_candidate",
                surface=surface,
                normalized="python",
                confidence=0.70,
                hints={"domain": "environment_python", "mention_only": True},
            )
        )

    for match in _PRICE_RE.finditer(src):
        out.append(
            IngestAnchor(
                kind="price_candidate",
                surface=match.group(0),
                normalized=_normalize_decimal(match.group(1)),
                confidence=0.82,
                hints={"domain": "price"},
            )
        )

    return _dedupe(out)


def _dedupe(items: list[IngestAnchor]) -> list[IngestAnchor]:
    seen: set[tuple[str, str]] = set()
    out: list[IngestAnchor] = []
    for item in list(items or []):
        key = (
            str(item.kind or "").strip().lower(),
            str(item.normalized or item.surface or "").strip().lower(),
            str(dict(item.hints or {}).get("temporal_state") or "").strip().lower(),
        )
        if not key[0] or not key[1] or key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def _normalize_gpu(value: str) -> str:
    raw = re.sub(r"\s+", " ", str(value or "").strip()).replace("-", " ").replace("_", " ")
    parts = [part for part in raw.split(" ") if part]
    normalized: list[str] = []
    for part in parts:
        token = part.lower()
        if token in {"rtx", "gtx", "rx"}:
            normalized.append(token.upper())
        elif token == "ti":
            normalized.append("Ti")
        elif token == "xt":
            normalized.append("XT")
        elif token == "super":
            normalized.append("Super")
        else:
            normalized.append(part)
    return " ".join(normalized).strip()


def _normalize_os(value: str) -> str:
    token = str(value or "").strip().lower()
    mapping = {
        "windows": "Windows",
        "\u0432\u0438\u043d\u0434\u0430": "Windows",
        "\u0432\u0438\u043d\u0434\u0443": "Windows",
        "\u0432\u0438\u043d\u0434\u043e\u0432\u0441": "Windows",
        "linux": "Linux",
        "ubuntu": "Ubuntu",
        "debian": "Debian",
        "macos": "macOS",
        "mac os": "macOS",
    }
    return mapping.get(token, str(value or "").strip())


def _normalize_os_mention(value: str) -> str:
    token = str(value or "").strip().lower()
    mapping = {
        "\u0432\u0438\u043d\u0434\u044b": "Windows",
        "\u043e\u0441": "os",
        "\u0441\u0438\u0441\u0442\u0435\u043c\u0430": "system",
    }
    normalized = _normalize_os(value)
    if normalized != str(value or "").strip():
        return normalized
    return mapping.get(token, str(value or "").strip())


def _os_temporal_state(surface: str, *, current_tokens: set[str], past_tokens: set[str]) -> str:
    token = str(surface or "").strip().lower()
    if token in current_tokens:
        return "current"
    if token in past_tokens:
        return "past"
    return ""


def _normalize_size(value: str | None) -> str:
    raw = _normalize_decimal(value)
    if raw.endswith(".0"):
        raw = raw[:-2]
    return raw


def _normalize_decimal(value: str | None) -> str:
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


def _is_blocked_age_match(text: str, match: re.Match[str]) -> bool:
    raw = match.group(1) or match.group(2) or match.group(3) or ""
    try:
        age = int(raw)
    except Exception:
        return True
    if age <= 0 or age > 120:
        return True
    if match.group(1) or match.group(2):
        return False
    tail = str(text[match.end() : match.end() + 24] or "").strip().lower()
    return any(token in tail for token in _AGE_BLOCKERS)
