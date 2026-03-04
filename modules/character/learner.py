from __future__ import annotations

from typing import Any

DEFAULT_BASELINE_TRAITS: dict[str, float] = {
    "warmth": 0.58,
    "sarcasm": 0.24,
    "teasing": 0.35,
    "verbosity": 0.46,
    "strictness": 0.56,
    "empathy": 0.58,
    "professionalism": 0.62,
    "directness": 0.55,
    "patience": 0.58,
    "humor": 0.34,
    "emoji_rate": 0.16,
    "energy": 0.55,
    "curiosity": 0.52,
}

# Backward-compatible alias.
BASELINE_TRAITS = dict(DEFAULT_BASELINE_TRAITS)


def update_persona(
    persona: dict[str, Any] | None,
    signals: dict[str, Any] | None,
    *,
    max_delta_per_turn: float = 0.02,
    decay_to_baseline: float = 0.005,
) -> tuple[dict[str, Any], dict[str, Any]]:
    current = dict(persona or {})
    sig = dict(signals or {})

    trait_limit = max(0.001, float(max_delta_per_turn))
    feedback_limit = max(0.05, trait_limit * 2.5)

    traits = _coerce_traits(current.get("traits"))
    locks = dict(current.get("locks") or {})
    locks.setdefault("feminine", True)
    locks.setdefault("informal_you", True)
    bans = [str(x).strip().lower() for x in list(current.get("bans") or []) if str(x).strip()]
    learned = dict(current.get("learned") or {})
    confirmed = [str(x).strip().lower() for x in list(learned.get("preferences_confirmed") or []) if str(x).strip()]
    pending = [str(x).strip().lower() for x in list(learned.get("preferences_pending") or []) if str(x).strip()]
    style_bias = dict(learned.get("style_bias") or {})

    baselines = _coerce_baselines(
        current.get("baseline_traits"),
        learned.get("baseline_traits"),
    )
    if not baselines:
        if traits:
            baselines = {k: float(_clamp01(v)) for k, v in traits.items()}
        else:
            baselines = dict(DEFAULT_BASELINE_TRAITS)

    # Any newly seen trait is anchored to baseline at first appearance.
    for key, value in list(traits.items()):
        baselines.setdefault(str(key), float(_clamp01(value)))

    decay_deltas: dict[str, float] = {}
    for key, baseline in baselines.items():
        trait_name = _normalize_trait_name(key)
        if not trait_name or trait_name not in traits:
            continue
        before = float(traits.get(trait_name, baseline))
        delta = _clamp(float(baseline) - before, -abs(decay_to_baseline), abs(decay_to_baseline))
        after = _clamp01(before + delta)
        if abs(after - before) > 1e-9:
            decay_deltas[trait_name] = float(after - before)
            traits[trait_name] = after

    implicit_raw = _implicit_trait_deltas(sig)
    implicit_deltas: dict[str, float] = {}
    for key, delta in implicit_raw.items():
        trait_name = _normalize_trait_name(key)
        if not trait_name:
            continue
        baseline = float(baselines.get(trait_name, _default_baseline_for_trait(trait_name)))
        if trait_name not in traits:
            traits[trait_name] = baseline
            baselines[trait_name] = baseline
        before = float(traits.get(trait_name, baseline))
        step = _clamp(float(delta), -trait_limit, trait_limit)
        after = _clamp01(before + step)
        if abs(after - before) > 1e-9:
            implicit_deltas[trait_name] = implicit_deltas.get(trait_name, 0.0) + float(after - before)
            traits[trait_name] = after

    feedback_items = [str(x).strip().lower() for x in list(sig.get("user_feedback") or []) if str(x).strip()]
    feedback_deltas: dict[str, float] = {}
    lock_changes: list[str] = []
    ban_changes: list[str] = []

    for item in feedback_items:
        if item == "lock_feminine":
            if not bool(locks.get("feminine", False)):
                lock_changes.append("feminine:true")
            locks["feminine"] = True
            _upsert_unique(confirmed, "lock_feminine")
            continue

        if item.startswith("ban_word:"):
            token = item.split(":", 1)[1].strip().lower()
            if token and token not in bans:
                bans.append(token)
                ban_changes.append(f"add:{token}")
                _upsert_unique(confirmed, f"ban_word:{token}")
            continue

        if item.startswith("unban_word:"):
            token = item.split(":", 1)[1].strip().lower()
            if token and token in bans:
                bans = [x for x in bans if x != token]
                ban_changes.append(f"remove:{token}")
                _upsert_unique(confirmed, f"unban_word:{token}")
            continue

        if item in {
            "less_compliments",
            "more_compliments",
            "no_compliments",
            "less_warmth",
            "more_warmth",
            "no_teasing",
            "more_teasing",
            "less_sarcasm",
            "more_sarcasm",
            "shorter_answers",
            "longer_answers",
            "be_strict",
            "be_softer",
            "more_professional",
            "less_professional",
            "more_direct",
            "less_direct",
            "more_patient",
            "less_patient",
            "more_humor",
            "less_humor",
            "more_emoji",
            "less_emoji",
            "more_energy",
            "less_energy",
            "more_curiosity",
            "less_curiosity",
        }:
            _upsert_unique(confirmed, item)
        else:
            _upsert_unique(pending, item)

        for trait_name, delta in _feedback_trait_deltas(item).items():
            key = _normalize_trait_name(trait_name)
            if not key:
                continue
            baseline = float(baselines.get(key, _default_baseline_for_trait(key)))
            if key not in traits:
                traits[key] = baseline
                baselines[key] = baseline
            before = float(traits.get(key, baseline))
            step = _clamp(float(delta), -feedback_limit, feedback_limit)
            after = _clamp01(before + step)
            if abs(after - before) > 1e-9:
                feedback_deltas[key] = feedback_deltas.get(key, 0.0) + float(after - before)
                traits[key] = after

    for key, baseline in baselines.items():
        if key not in traits:
            continue
        style_bias[key] = float(_clamp01(traits.get(key, baseline)) - float(baseline))

    learned["preferences_confirmed"] = confirmed[-64:]
    learned["preferences_pending"] = pending[-64:]
    learned["style_bias"] = {k: float(v) for k, v in style_bias.items()}
    learned["baseline_traits"] = {k: float(v) for k, v in baselines.items()}

    updated = dict(current)
    updated["traits"] = {k: float(_clamp01(v)) for k, v in traits.items()}
    updated["locks"] = dict(locks)
    updated["bans"] = sorted(set(str(x).strip().lower() for x in bans if str(x).strip()))
    updated["learned"] = learned
    updated["baseline_traits"] = {k: float(v) for k, v in baselines.items()}

    delta_debug = {
        "decay_deltas": decay_deltas,
        "implicit_deltas": implicit_deltas,
        "feedback_deltas": feedback_deltas,
        "feedback_applied": feedback_items,
        "lock_changes": lock_changes,
        "ban_changes": ban_changes,
        "max_delta_per_turn": float(trait_limit),
        "decay_to_baseline": float(decay_to_baseline),
        "baseline_size": len(baselines),
    }
    return updated, delta_debug


def _coerce_traits(value: Any) -> dict[str, float]:
    src = dict(value or {})
    out: dict[str, float] = {}
    for key, raw in src.items():
        name = _normalize_trait_name(key)
        if not name:
            continue
        out[name] = _clamp01(_to_float(raw, _default_baseline_for_trait(name)))
    return out


def _implicit_trait_deltas(signals: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    intent = str(signals.get("intent") or "").strip().lower()
    emotion = str(signals.get("emotion") or "").strip().lower()
    mode = str(signals.get("mode") or "").strip().lower()
    tags = {str(x).strip().lower() for x in list(signals.get("tags") or []) if str(x).strip()}

    def add(trait_name: str, delta: float) -> None:
        key = _normalize_trait_name(trait_name)
        if not key:
            return
        out[key] = out.get(key, 0.0) + float(delta)

    if emotion in {"frustrated", "anxious", "sad"}:
        add("warmth", 0.010)
        add("empathy", 0.012)
        add("patience", 0.010)
        add("sarcasm", -0.010)
        add("energy", -0.006)
    if emotion == "angry":
        add("warmth", 0.006)
        add("sarcasm", -0.015)
        add("strictness", 0.008)
        add("directness", 0.012)
        add("patience", -0.012)
    if emotion == "excited":
        add("teasing", 0.006)
        add("warmth", 0.004)
        add("energy", 0.012)
        add("humor", 0.008)
        add("emoji_rate", 0.010)
        add("curiosity", 0.008)
    if emotion == "happy":
        add("warmth", 0.006)
        add("humor", 0.006)
        add("emoji_rate", 0.008)
    if emotion == "tired":
        add("energy", -0.012)
        add("verbosity", -0.006)
        add("patience", -0.004)

    if {"has_traceback", "has_logs", "has_stacktrace"} & tags:
        add("strictness", 0.010)
        add("verbosity", 0.010)
        add("teasing", -0.010)
        add("professionalism", 0.012)
        add("directness", 0.010)
        add("emoji_rate", -0.012)
        add("humor", -0.008)

    if intent in {"task", "bug_report", "code_review"}:
        add("strictness", 0.010)
        add("verbosity", 0.006)
        add("sarcasm", -0.008)
        add("professionalism", 0.010)
        add("directness", 0.008)
        add("emoji_rate", -0.008)
        add("teasing", -0.006)
    elif intent == "planning":
        add("strictness", 0.008)
        add("verbosity", 0.008)
        add("professionalism", 0.008)
        add("curiosity", 0.006)
    elif intent in {"chat", "clarification"}:
        add("warmth", 0.004)
        add("teasing", 0.003)
        add("humor", 0.004)
        add("emoji_rate", 0.006)

    if mode in {"engineer", "debugger", "planner"}:
        add("professionalism", 0.012)
        add("directness", 0.010)
        add("strictness", 0.006)
        add("patience", 0.004)
        add("teasing", -0.010)
        add("humor", -0.006)
        add("emoji_rate", -0.012)
    elif mode == "helper":
        add("warmth", 0.006)
        add("empathy", 0.010)
        add("patience", 0.010)
        add("directness", 0.002)
    elif mode == "friend_chat":
        add("warmth", 0.006)
        add("humor", 0.006)
        add("emoji_rate", 0.008)
        add("teasing", 0.004)
        add("directness", -0.003)

    return out


def _feedback_trait_deltas(item: str) -> dict[str, float]:
    mapping: dict[str, dict[str, float]] = {
        "less_compliments": {"warmth": -0.04, "humor": -0.03, "emoji_rate": -0.02},
        "more_compliments": {"warmth": 0.04, "humor": 0.03, "emoji_rate": 0.02},
        "no_compliments": {"warmth": -0.05, "humor": -0.04, "emoji_rate": -0.03},
        "less_warmth": {"warmth": -0.05},
        "more_warmth": {"warmth": 0.05},
        "no_teasing": {"teasing": -0.05},
        "more_teasing": {"teasing": 0.05},
        "less_sarcasm": {"sarcasm": -0.05},
        "more_sarcasm": {"sarcasm": 0.05},
        "shorter_answers": {"verbosity": -0.05},
        "longer_answers": {"verbosity": 0.05},
        "be_strict": {"strictness": 0.05},
        "be_softer": {"strictness": -0.05, "warmth": 0.03},
        "more_professional": {"professionalism": 0.05},
        "less_professional": {"professionalism": -0.05},
        "more_direct": {"directness": 0.05},
        "less_direct": {"directness": -0.05},
        "more_patient": {"patience": 0.05},
        "less_patient": {"patience": -0.05},
        "more_humor": {"humor": 0.05},
        "less_humor": {"humor": -0.05},
        "more_emoji": {"emoji_rate": 0.05},
        "less_emoji": {"emoji_rate": -0.05},
        "more_energy": {"energy": 0.05},
        "less_energy": {"energy": -0.05},
        "more_curiosity": {"curiosity": 0.05},
        "less_curiosity": {"curiosity": -0.05},
    }
    return dict(mapping.get(str(item or "").strip().lower(), {}))


def _coerce_baselines(*values: Any) -> dict[str, float]:
    out: dict[str, float] = {}
    for value in values:
        if not isinstance(value, dict):
            continue
        for key, raw in dict(value).items():
            name = _normalize_trait_name(key)
            if not name:
                continue
            out[name] = float(_clamp01(_to_float(raw, _default_baseline_for_trait(name))))
    return out


def _normalize_trait_name(value: Any) -> str:
    raw = str(value or "").strip().lower()
    if not raw:
        return ""
    return "".join(ch for ch in raw if ch.isalnum() or ch in {"_", "-"})


def _default_baseline_for_trait(name: str) -> float:
    key = _normalize_trait_name(name)
    if not key:
        return 0.5
    return float(DEFAULT_BASELINE_TRAITS.get(key, 0.5))


def _upsert_unique(target: list[str], value: str) -> None:
    key = str(value or "").strip().lower()
    if not key:
        return
    if key in target:
        return
    target.append(key)


def _to_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _clamp(value: float, lo: float, hi: float) -> float:
    low = float(min(lo, hi))
    high = float(max(lo, hi))
    return max(low, min(high, float(value)))


def _clamp01(value: float) -> float:
    return _clamp(float(value), 0.0, 1.0)
