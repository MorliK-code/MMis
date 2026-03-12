from __future__ import annotations

from typing import Any


TRAIT_SOFT_CAPS: dict[str, float] = {
    "professionalism": 0.85,
    "strictness": 0.82,
    "playfulness": 0.72,
    "humor": 0.78,
    "sarcasm": 0.40,
    "teasing": 0.38,
    "emoji_rate": 0.30,
    "warmth": 0.82,
    "empathy": 0.84,
    "softness": 0.78,
    "protectiveness": 0.76,
}

# Backward-compatible alias for older imports/tests.
DEFAULT_TRAIT_SOFT_CAPS = TRAIT_SOFT_CAPS


def normalize_trait_name(value: Any) -> str:
    raw = str(value or "").strip().lower()
    if not raw:
        return ""
    return "".join(ch for ch in raw if ch.isalnum() or ch in {"_", "-"})


def trait_soft_cap(name: Any) -> float | None:
    key = normalize_trait_name(name)
    if not key:
        return None
    value = TRAIT_SOFT_CAPS.get(key)
    if value is None:
        return None
    return float(value)


def _apply_trait_soft_cap(name: Any, value: Any) -> float:
    capped = _clamp(_to_float(value, 0.0), 0.0, 1.0)
    soft_cap = trait_soft_cap(name)
    if soft_cap is None:
        return capped
    return min(capped, float(soft_cap))


def normalize_trait_value(name: Any, value: Any) -> float:
    return _apply_trait_soft_cap(name, value)


def resolve_trait_bounds(name: Any, minimum: Any = 0.0, maximum: Any = 1.0) -> tuple[float, float]:
    lo = _clamp(_to_float(minimum, 0.0), 0.0, 1.0)
    hi = _clamp(_to_float(maximum, 1.0), 0.0, 1.0)
    if hi < lo:
        lo, hi = hi, lo
    soft_cap = trait_soft_cap(name)
    if soft_cap is not None:
        hi = min(hi, float(soft_cap))
    if hi < lo:
        hi = lo
    return float(lo), float(hi)


def clamp_trait_scalar(name: Any, value: Any, *, minimum: Any = 0.0, maximum: Any = 1.0) -> float:
    lo, hi = resolve_trait_bounds(name, minimum=minimum, maximum=maximum)
    raw_value = _to_float(value, lo)
    bounded = _clamp(raw_value, lo, hi)
    soft_cap = trait_soft_cap(name)
    if soft_cap is None:
        return bounded
    return min(bounded, float(soft_cap))


def apply_trait_delta(
    name: Any,
    current: Any,
    delta: Any,
    *,
    minimum: Any = 0.0,
    maximum: Any = 1.0,
    soften: bool = True,
) -> float:
    lo, hi = resolve_trait_bounds(name, minimum=minimum, maximum=maximum)
    cur = clamp_trait_scalar(name, current, minimum=lo, maximum=hi)
    step = _to_float(delta, 0.0)
    if soften and abs(step) > 0.0:
        span = max(1e-6, hi - lo)
        room = (hi - cur) if step > 0.0 else (cur - lo)
        scale = max(0.12, min(1.0, room / span))
        step *= scale
    return clamp_trait_scalar(name, cur + step, minimum=lo, maximum=hi)


def normalize_trait_record(name: Any, row: dict[str, Any] | None) -> dict[str, Any]:
    payload = dict(row or {})
    ttype = str(payload.get("type") or "scalar").strip().lower()
    out = {
        "type": "flag" if ttype == "flag" else "scalar",
        "value": payload.get("value", False if ttype == "flag" else 0.0),
        "min": _to_float(payload.get("min"), 0.0),
        "max": _to_float(payload.get("max"), 1.0),
        "decay_per_day": max(0.0, _to_float(payload.get("decay_per_day"), 0.0)),
        "confidence": _clamp(_to_float(payload.get("confidence"), 0.6), 0.0, 1.0),
        "tags": [str(x).strip().lower() for x in list(payload.get("tags") or []) if str(x).strip()],
        "prompt_file": str(payload.get("prompt_file") or "").strip(),
        "ttl_days": max(0.0, _to_float(payload.get("ttl_days"), 0.0)),
        "disabled": bool(payload.get("disabled", False)),
        "updated_at": str(payload.get("updated_at") or ""),
        "last_used_ts": str(payload.get("last_used_ts") or ""),
        "decay_ts": str(payload.get("decay_ts") or ""),
    }
    lo, hi = resolve_trait_bounds(name, minimum=out["min"], maximum=out["max"])
    out["min"] = lo
    out["max"] = hi
    if out["type"] == "flag":
        out["value"] = bool(out["value"])
    else:
        out["value"] = clamp_trait_scalar(name, out["value"], minimum=lo, maximum=hi)
    return out


def clamp_trait_map(values: dict[str, Any] | None) -> dict[str, float]:
    out: dict[str, float] = {}
    for raw_key, raw_value in dict(values or {}).items():
        key = normalize_trait_name(raw_key)
        if not key:
            continue
        out[key] = clamp_trait_scalar(key, raw_value)
    return out


def _to_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _clamp(value: float, lo: float, hi: float) -> float:
    low = float(min(lo, hi))
    high = float(max(lo, hi))
    return max(low, min(high, float(value)))
