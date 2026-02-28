from __future__ import annotations

from typing import Any


class RuleEvaluator:
    def matches(self, when: dict[str, Any] | None, *, ctx: dict[str, Any], traits: dict[str, Any]) -> bool:
        cond = dict(when or {})
        if not cond:
            return True

        if "intent" in cond and not _match_multi(str(ctx.get("intent") or ""), cond.get("intent")):
            return False
        if "emotion" in cond and not _match_multi(str(ctx.get("emotion") or ""), cond.get("emotion")):
            return False
        if "mode" in cond and not _match_multi(str(ctx.get("mode") or ""), cond.get("mode")):
            return False

        tags = {str(x).strip().lower() for x in list(ctx.get("tags") or []) if str(x).strip()}
        any_tags = {str(x).strip().lower() for x in list(cond.get("tags_any") or []) if str(x).strip()}
        all_tags = {str(x).strip().lower() for x in list(cond.get("tags_all") or []) if str(x).strip()}
        if any_tags and tags.isdisjoint(any_tags):
            return False
        if all_tags and not all_tags.issubset(tags):
            return False

        min_trait = dict(cond.get("min_trait") or {})
        for key, value in min_trait.items():
            if _trait_value(traits, str(key)) < _to_float(value, 0.0):
                return False

        max_trait = dict(cond.get("max_trait") or {})
        for key, value in max_trait.items():
            if _trait_value(traits, str(key)) > _to_float(value, 1.0):
                return False

        return True


def _match_multi(actual: str, expected) -> bool:
    src = str(actual or "").strip().lower()
    if isinstance(expected, (list, tuple, set)):
        values = [str(x).strip().lower() for x in list(expected or []) if str(x).strip()]
        return src in values
    return src == str(expected or "").strip().lower()


def _trait_value(traits: dict[str, Any], key: str) -> float:
    row = dict(traits.get(str(key).strip().lower()) or {})
    value = row.get("value")
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    return _to_float(value, 0.0)


def _to_float(value, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)

