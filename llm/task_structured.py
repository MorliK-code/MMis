from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any


class TaskOutputValidationError(ValueError):
    pass


@dataclass(frozen=True)
class TaskOutputSpec:
    kind: str = "text"
    min_chars: int = 1
    max_chars: int = 12000
    required_fields: tuple[str, ...] = ()
    allow_array: bool = False
    allowed_labels: tuple[str, ...] = ()
    label_aliases: dict[str, str] = field(default_factory=dict)
    max_label_length: int = 64


def normalize_output_spec(
    spec: TaskOutputSpec | None = None,
    *,
    kind: str = "text",
    min_chars: int = 1,
    max_chars: int = 12000,
    required_fields: list[str] | tuple[str, ...] | None = None,
    allow_array: bool = False,
    allowed_labels: list[str] | tuple[str, ...] | None = None,
    label_aliases: dict[str, str] | None = None,
    max_label_length: int = 64,
) -> TaskOutputSpec:
    if spec is not None:
        return spec
    return TaskOutputSpec(
        kind=str(kind or "text").strip().lower() or "text",
        min_chars=max(0, int(min_chars)),
        max_chars=max(1, int(max_chars)),
        required_fields=tuple(str(x or "").strip() for x in list(required_fields or []) if str(x or "").strip()),
        allow_array=bool(allow_array),
        allowed_labels=tuple(str(x or "").strip().lower() for x in list(allowed_labels or []) if str(x or "").strip()),
        label_aliases={str(k or "").strip().lower(): str(v or "").strip().lower() for k, v in dict(label_aliases or {}).items() if str(k or "").strip() and str(v or "").strip()},
        max_label_length=max(8, int(max_label_length)),
    )


def _strip_code_fence(text: str) -> str:
    raw = str(text or "").strip()
    if not raw:
        return ""
    match = re.search(r"```(?:json)?\s*(.*?)```", raw, flags=re.IGNORECASE | re.DOTALL)
    if match:
        return str(match.group(1) or "").strip()
    return raw


def _looks_like_garbage(clean: str) -> bool:
    if not clean:
        return True
    if clean in {"...", "…", "n/a", "none", "null", "todo", "tbd", "??"}:
        return True
    if re.fullmatch(r"[`\-\s\.\{\}\[\]:,\"']+", clean):
        return True
    if re.search(r"(.)\1{15,}", clean):
        return True
    alnum_count = sum(1 for ch in clean if ch.isalnum())
    if len(clean) >= 24 and alnum_count / max(1, len(clean)) < 0.18:
        return True
    return False


def validate_text_output(text: str, spec: TaskOutputSpec | None = None) -> str:
    cfg = normalize_output_spec(spec)
    clean = str(text or "").strip()
    if len(clean) < cfg.min_chars:
        raise TaskOutputValidationError("Task model returned empty or too-short output")
    if len(clean) > cfg.max_chars:
        raise TaskOutputValidationError(
            f"Task model output exceeds max length ({len(clean)} > {cfg.max_chars})"
        )
    if _looks_like_garbage(clean):
        raise TaskOutputValidationError("Task model returned clearly low-signal output")
    return clean


def parse_json_task_output(text: str, spec: TaskOutputSpec | None = None) -> Any:
    cfg = normalize_output_spec(spec, kind="json")
    raw = validate_text_output(text, cfg)
    candidates: list[str] = []
    for candidate in (raw, _strip_code_fence(raw)):
        if candidate and candidate not in candidates:
            candidates.append(candidate)
    stripped = candidates[-1] if candidates else raw
    for opener, closer in (("{", "}"), ("[", "]")):
        start = stripped.find(opener)
        end = stripped.rfind(closer)
        if start >= 0 and end > start:
            fragment = stripped[start : end + 1].strip()
            if fragment and fragment not in candidates:
                candidates.append(fragment)
    last_error: Exception | None = None
    parsed: Any = None
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            break
        except Exception as exc:
            last_error = exc
    if parsed is None:
        raise TaskOutputValidationError(f"Task model returned invalid JSON: {last_error}") from last_error
    if isinstance(parsed, list) and not cfg.allow_array:
        raise TaskOutputValidationError("Task model returned JSON array where object was required")
    if cfg.required_fields:
        if not isinstance(parsed, dict):
            raise TaskOutputValidationError("Task model must return JSON object with required fields")
        missing = [
            field_name
            for field_name in cfg.required_fields
            if field_name not in parsed or parsed.get(field_name) in (None, "", [], {})
        ]
        if missing:
            raise TaskOutputValidationError(
                f"Task model JSON output is missing required fields: {', '.join(missing)}"
            )
    return parsed


def parse_short_classification_output(text: str, spec: TaskOutputSpec | None = None) -> str:
    cfg = normalize_output_spec(spec, kind="classification", max_chars=128, max_label_length=64)
    raw = validate_text_output(text, cfg)
    candidate = raw
    if raw.startswith("{") or raw.startswith("```"):
        try:
            payload = parse_json_task_output(raw, normalize_output_spec(cfg, allow_array=False))
            if isinstance(payload, dict):
                for key in ("label", "intent", "category", "class", "result"):
                    value = str(payload.get(key) or "").strip()
                    if value:
                        candidate = value
                        break
        except Exception:
            candidate = raw
    first_line = str(candidate.splitlines()[0] if candidate else "").strip()
    first_line = re.sub(r"^[\-\*\d\.\)\s`]+", "", first_line).strip().lower()
    first_line = re.sub(r"\s+", " ", first_line)
    if len(first_line) > cfg.max_label_length:
        raise TaskOutputValidationError("Classification output is too long")
    if not first_line or _looks_like_garbage(first_line):
        raise TaskOutputValidationError("Classification output is empty or low-signal")
    aliases = dict(cfg.label_aliases or {})
    normalized = aliases.get(first_line, first_line)
    if cfg.allowed_labels and normalized not in set(cfg.allowed_labels):
        raise TaskOutputValidationError(
            f"Classification output '{normalized}' is not in allowed labels"
        )
    return normalized


def validate_task_output(text: str, spec: TaskOutputSpec | None = None) -> tuple[str, Any]:
    cfg = normalize_output_spec(spec)
    if cfg.kind == "json":
        clean = validate_text_output(text, cfg)
        return clean, parse_json_task_output(clean, cfg)
    if cfg.kind == "classification":
        clean = validate_text_output(text, cfg)
        return clean, parse_short_classification_output(clean, cfg)
    clean = validate_text_output(text, cfg)
    return clean, None
