import json
import re
from typing import Any


def extract_json_object(text: str, default: dict[str, Any]) -> dict[str, Any]:
    match = re.search(r"\{[\s\S]*\}", text or "")
    if not match:
        return dict(default)
    try:
        data = json.loads(match.group(0))
    except Exception:
        return dict(default)
    return data if isinstance(data, dict) else dict(default)


def extract_json_array(text: str, default: list[Any] | None = None) -> list[Any]:
    fallback = list(default) if default is not None else []
    match = re.search(r"\[[\s\S]*\]", text or "")
    if not match:
        return fallback
    try:
        data = json.loads(match.group(0))
    except Exception:
        return fallback
    return data if isinstance(data, list) else fallback
