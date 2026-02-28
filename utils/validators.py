from __future__ import annotations

from pathlib import Path
from typing import Any


def non_empty(value: str) -> bool:
    return bool(str(value or "").strip())


def validate_tool_call_schema(obj: Any) -> tuple[bool, str]:
    if not isinstance(obj, dict):
        return False, "tool call must be an object"
    tool = obj.get("tool") or obj.get("name")
    if not non_empty(str(tool or "")):
        return False, "tool/name is required"
    args = obj.get("args", {})
    if args is None:
        args = {}
    if not isinstance(args, dict):
        return False, "args must be an object"
    return True, ""


def validate_profile_update(fact: Any) -> tuple[bool, str]:
    if not isinstance(fact, dict):
        return False, "fact must be an object"

    key = str(fact.get("key") or "").strip()
    if not key:
        return False, "fact.key is required"

    op = str(fact.get("op") or "add").strip().lower()
    if op not in {"add", "update", "remove", "set"}:
        return False, "fact.op must be one of add/update/remove/set"

    conf = fact.get("confidence", 0.5)
    try:
        conf_value = float(conf)
    except Exception:
        return False, "fact.confidence must be a number"
    if conf_value < 0.0 or conf_value > 1.0:
        return False, "fact.confidence must be in range [0..1]"

    if op in {"add", "update", "set"} and "value" not in fact:
        return False, "fact.value is required for add/update/set"

    return True, ""


def safe_path(path: str | Path, root: str | Path) -> Path:
    target = Path(path).expanduser().resolve()
    base = Path(root).expanduser().resolve()
    try:
        target.relative_to(base)
    except Exception as exc:
        raise ValueError(f"Path escapes allowed root: {target}") from exc
    return target


def clamp_params(
    *,
    temperature: float | None = None,
    top_p: float | None = None,
    repeat_penalty: float | None = None,
    max_tokens: int | None = None,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if temperature is not None:
        out["temperature"] = _clamp_float(temperature, 0.0, 2.0)
    if top_p is not None:
        out["top_p"] = _clamp_float(top_p, 0.0, 1.0)
    if repeat_penalty is not None:
        out["repeat_penalty"] = _clamp_float(repeat_penalty, 0.0, 2.0)
    if max_tokens is not None:
        out["max_tokens"] = _clamp_int(max_tokens, 1, 200000)
    return out


def _clamp_float(value: float, low: float, high: float) -> float:
    try:
        item = float(value)
    except Exception:
        item = float(low)
    if item < low:
        return float(low)
    if item > high:
        return float(high)
    return float(item)


def _clamp_int(value: int, low: int, high: int) -> int:
    try:
        item = int(value)
    except Exception:
        item = int(low)
    if item < low:
        return int(low)
    if item > high:
        return int(high)
    return int(item)
