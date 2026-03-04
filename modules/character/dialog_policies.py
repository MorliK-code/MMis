
from __future__ import annotations

import hashlib
import importlib.util
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config.paths import DATA_DIR
from config.settings import load_config
from utils.datetime_local import parse_time_to_epoch

_DATA_POLICY_PATH = DATA_DIR / "dialog_police.py"
_POLICY_CACHE: dict[str, Any] = {"mtime": None, "data": None}

_WORD_RE = re.compile(r"\S+")
_NON_WORD_EDGE_RE = re.compile(r"^[^\w]+|[^\w]+$", flags=re.UNICODE)
_TECH_RE = re.compile(
    r"(traceback|exception|error|stack\s*trace|```|[A-Za-z]:\\|/[\w\-.]+(?:/[\w\-.]+)+|"
    r"\bgit\b|\bpip\b|\bpython\b|\bnode\b|\bnpm\b|\bpytest\b|\bhttp\s*[45]\d\d\b)",
    flags=re.IGNORECASE,
)
_HOSTILE_RE = re.compile(
    r"(\bidiot\b|\bstupid\b|\bdumb\b|\bshut\s+up\b|\bwtf\b|\bбред\b|\bтупо\b|\bидиот\b)",
    flags=re.IGNORECASE,
)
_SOFT_RE = re.compile(
    r"(\bplease\b|\bthanks?\b|\bспасибо\b|\bпожалуйста\b|\bзаранее\b|\bбуду\s+благодарен\b)",
    flags=re.IGNORECASE,
)

_NEGATIVE_EMOTIONS = {
    "angry",
    "frustrated",
    "sad",
    "anxious",
    "tired",
}
_POSITIVE_EMOTIONS = {
    "happy",
    "excited",
}
_TECH_INTENTS = {
    "task",
    "bug_report",
    "code_review",
    "planning",
    "question",
}

_SESSION_BAN_MARKER_PREFIX = "session:"


def is_user_greeting(
    text: str,
    *,
    max_words: int | None = None,
    max_chars: int | None = None,
    greetings: list[str] | None = None,
    anti_patterns: list[str] | None = None,
) -> bool:
    cfg = _resolve_config(
        max_words=max_words,
        max_chars=max_chars,
        greetings=greetings,
        anti_patterns=anti_patterns,
        new_session_after_min=None,
        extra={},
    )
    src = _normalize_text(text)
    if not src:
        return False
    if len(src) > int(cfg["max_chars"]):
        return False
    words = _WORD_RE.findall(src)
    if len(words) > int(cfg["max_words"]):
        return False

    low = src.lower()
    for anti in list(cfg["anti_patterns"]):
        item = str(anti or "").strip().lower()
        if item and item in low:
            return False

    return bool(_greeting_start_regex(list(cfg["greetings"])).search(low))


def starts_with_greeting(text: str, *, greetings: list[str] | None = None) -> bool:
    cfg = _resolve_config(
        max_words=None,
        max_chars=None,
        greetings=greetings,
        anti_patterns=None,
        new_session_after_min=None,
        extra={},
    )
    src = _normalize_text(text).lower()
    if not src:
        return False
    return bool(_greeting_start_regex(list(cfg["greetings"])).search(src))


def trim_leading_greeting(text: str, *, greetings: list[str] | None = None) -> str:
    cfg = _resolve_config(
        max_words=None,
        max_chars=None,
        greetings=greetings,
        anti_patterns=None,
        new_session_after_min=None,
        extra={},
    )
    src = _normalize_text(text)
    if not src:
        return ""

    out = src
    start_rx = _greeting_start_regex(list(cfg["greetings"]))
    for _ in range(3):
        match = start_rx.search(out.lower())
        if match is None:
            break
        out = out[match.end() :].lstrip(" ,.!?:;\n\t")
    lines = [x.strip() for x in out.split("\n") if x.strip()]
    if lines:
        first = lines[0].lower()
        if first.startswith("how can i help") or first.startswith("чем помочь") or first.startswith("как дела"):
            lines = lines[1:]
    cleaned = _normalize_text("\n".join(lines))
    return cleaned or src


def is_technical(text: str) -> bool:
    src = str(text or "")
    if not src.strip():
        return False
    return bool(_TECH_RE.search(src))


def is_new_session(now, last_turn_ts, *, threshold_sec: float) -> bool:
    now_ts = _to_epoch(now)
    last_ts = _to_epoch(last_turn_ts)
    if last_ts <= 0:
        return True
    return (now_ts - last_ts) > max(1.0, float(threshold_sec))


def detect_user_tone(text: str) -> str:
    src = _normalize_text(text)
    if not src:
        return "neutral"
    if _HOSTILE_RE.search(src):
        return "hostile"
    if _SOFT_RE.search(src):
        return "friendly"
    return "neutral"


def extract_term_directive(
    text: str,
    terms_list: list[str],
    *,
    disable_patterns: list[str] | None = None,
    enable_patterns: list[str] | None = None,
) -> dict[str, Any]:
    src = _normalize_text(text).lower()
    terms = [str(x or "").strip().lower() for x in list(terms_list or []) if str(x or "").strip()]
    if not src:
        return {"disable_terms": [], "enable_terms": [], "is_disable": False, "is_enable": False}

    disable_patterns_norm = [str(x or "").strip().lower() for x in list(disable_patterns or []) if str(x or "").strip()]
    enable_patterns_norm = [str(x or "").strip().lower() for x in list(enable_patterns or []) if str(x or "").strip()]

    disable_pos = _last_phrase_pos(src, disable_patterns_norm)
    enable_pos = _last_phrase_pos(src, enable_patterns_norm)
    is_disable = disable_pos >= 0
    is_enable = enable_pos >= 0

    if is_disable and is_enable:
        if disable_pos > enable_pos:
            is_enable = False
        elif enable_pos > disable_pos:
            is_disable = False

    mentioned_terms = [term for term in terms if term and term in src]
    disable_terms = mentioned_terms if is_disable and mentioned_terms else (terms if is_disable else [])
    enable_terms = mentioned_terms if is_enable and mentioned_terms else (terms if is_enable else [])
    return {
        "disable_terms": sorted(set(disable_terms)),
        "enable_terms": sorted(set(enable_terms)),
        "is_disable": bool(is_disable),
        "is_enable": bool(is_enable),
    }


def deterministic_term_gate_score(
    *,
    conversation_id: str,
    turn_id: int,
    allowed_term: str,
    user_text: str,
) -> float:
    seed = "|".join(
        [
            str(conversation_id or "").strip().lower(),
            str(int(turn_id) if int(turn_id or 0) > 0 else 0),
            str(allowed_term or "").strip().lower(),
            _normalize_text(user_text).lower(),
        ]
    )
    digest = hashlib.sha256(seed.encode("utf-8", errors="ignore")).hexdigest()
    value = int(digest[:16], 16)
    max_value = float(0xFFFFFFFFFFFFFFFF)
    return float(value / max_value)


def compute_address_terms_policy(
    text: str,
    now,
    state: dict[str, Any] | None = None,
    *,
    metadata: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    state_map = dict(state or {})
    meta_map = dict(metadata or {})
    cfg = _resolve_config(
        max_words=None,
        max_chars=None,
        greetings=None,
        anti_patterns=None,
        new_session_after_min=None,
        extra=(dict(config or {}) if isinstance(config, dict) else {}),
    )
    now_ts = _to_epoch(now if now is not None else datetime.now(timezone.utc).timestamp())

    text_norm = _normalize_text(text)
    conversation_id = _pick_text(
        meta_map.get("conversation_id"),
        state_map.get("conversation_id"),
        _as_dict(state_map.get("cooldowns")).get("session_id"),
    )
    turn_id = _to_int(_pick_text(meta_map.get("turn_id"), state_map.get("turn_id")), 0)
    address_state = _coerce_address_terms_state(state_map.get("address_terms"))

    terms_enabled = bool(cfg["terms_enabled"])
    terms_list = [str(x or "").strip() for x in list(cfg["terms_list"]) if str(x or "").strip()]
    terms_list_low = [x.lower() for x in terms_list]

    directive = extract_term_directive(
        text_norm,
        terms_list_low,
        disable_patterns=list(cfg["terms_disable_patterns"]),
        enable_patterns=list(cfg["terms_enable_patterns"]),
    )

    current_banned = _active_banned_terms_for_session(
        banned_terms=address_state["banned_terms"],
        banned_terms_until=address_state["banned_terms_until"],
        conversation_id=conversation_id,
    )

    ban_updates: dict[str, Any] = {}
    unban_updates: dict[str, Any] = {}

    if directive["is_disable"]:
        target_terms = [x for x in list(directive["disable_terms"]) if x in terms_list_low]
        if target_terms:
            marker = _ban_marker_for_scope(str(cfg["terms_ban_scope"]), conversation_id)
            ban_updates = {
                "add_terms": list(target_terms),
                "banned_terms_until": {term: marker for term in target_terms},
                "last_disable_directive_ts": _iso_from_epoch(now_ts),
            }

    if directive["is_enable"]:
        target_terms = [x for x in list(directive["enable_terms"]) if x in terms_list_low]
        if target_terms:
            unban_updates = {
                "remove_terms": list(target_terms),
                "remove_banned_terms_until": list(target_terms),
                "last_enable_directive_ts": _iso_from_epoch(now_ts),
            }

    effective_banned = set(current_banned)
    for term in list(ban_updates.get("add_terms") or []):
        effective_banned.add(str(term or "").strip().lower())
    for term in list(unban_updates.get("remove_terms") or []):
        effective_banned.discard(str(term or "").strip().lower())

    safety_mode = str(meta_map.get("safety_mode") or state_map.get("safety_mode") or "").strip().lower()
    safety_lock = bool(meta_map.get("safety_lock")) or bool(state_map.get("safety_lock"))
    technical = bool(meta_map.get("is_technical")) or is_technical(text_norm)
    if not technical:
        intent = _pick_text(meta_map.get("intent"), state_map.get("intent")).lower()
        technical = intent in _TECH_INTENTS

    terms_of_endearment_allowed = bool(terms_enabled and not technical and not safety_lock and safety_mode not in {"strict", "locked"})
    allowed_terms = [term for term in terms_list_low if term not in effective_banned]
    allowed_term = allowed_terms[0] if allowed_terms else ""
    term_not_banned = bool(allowed_term)

    cooldown_turns = max(1, int(cfg["terms_cooldown_turns"]))
    cooldown_seconds = max(1, int(cfg["terms_cooldown_seconds"]))
    last_turn_idx = _to_int(address_state.get("term_used_turn_index"), 0)
    last_term_used_at = _to_epoch(address_state.get("last_term_used_at"))

    cooldown_passed_turns = True
    if last_turn_idx > 0 and turn_id > 0:
        cooldown_passed_turns = (turn_id - last_turn_idx) >= cooldown_turns
    elif last_turn_idx > 0 and turn_id <= 0:
        cooldown_passed_turns = False

    cooldown_passed_seconds = True
    if last_term_used_at > 0:
        cooldown_passed_seconds = (now_ts - last_term_used_at) >= cooldown_seconds
    cooldown_passed = bool(cooldown_passed_turns and cooldown_passed_seconds)

    max_per_session = int(cfg["terms_max_per_session"])
    session_limit_ok = max_per_session <= 0 or int(address_state["terms_used_count_session"]) < max_per_session

    insert_probability = _clamp01(float(cfg["terms_insert_probability"]))
    gate_score = 1.0
    gate_passed = False
    if allowed_term and turn_id > 0:
        gate_score = deterministic_term_gate_score(
            conversation_id=conversation_id,
            turn_id=turn_id,
            allowed_term=allowed_term,
            user_text=text_norm,
        )
        gate_passed = gate_score < insert_probability

    use_term_now = bool(
        terms_of_endearment_allowed
        and term_not_banned
        and cooldown_passed
        and session_limit_ok
        and gate_passed
        and not directive["is_disable"]
    )

    recent_disable_directive = bool(meta_map.get("recent_disable_directive", False))
    if recent_disable_directive:
        use_term_now = False

    return {
        "terms_enabled": terms_enabled,
        "terms_list": list(terms_list_low),
        "terms_ban_scope": str(cfg["terms_ban_scope"]),
        "terms_of_endearment_allowed": bool(terms_of_endearment_allowed),
        "technical_context": bool(technical),
        "safety_context": bool(safety_lock or safety_mode in {"strict", "locked"}),
        "term_not_banned": bool(term_not_banned),
        "cooldown_passed_turns": bool(cooldown_passed_turns),
        "cooldown_passed_seconds": bool(cooldown_passed_seconds),
        "cooldown_passed": bool(cooldown_passed),
        "session_limit_ok": bool(session_limit_ok),
        "allowed_terms": list(allowed_terms),
        "allowed_term": str(allowed_term),
        "use_term_now": bool(use_term_now),
        "terms_insert_probability": float(insert_probability),
        "deterministic_gate_score": float(gate_score),
        "deterministic_gate_passed": bool(gate_passed),
        "ban_updates": ban_updates,
        "unban_updates": unban_updates,
        "directive": dict(directive),
        "conversation_id": str(conversation_id),
        "turn_id": int(turn_id),
        "banned_terms_effective": sorted(effective_banned),
        "banned_terms_active": bool(effective_banned),
        "recent_disable_directive": bool(recent_disable_directive),
    }


def compute_dialog_mode(
    text: str,
    now,
    state: dict[str, Any] | None = None,
    *,
    metadata: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    state_map = dict(state or {})
    meta_map = dict(metadata or {})
    cfg = _resolve_config(
        max_words=_dict_get(config, "greeting_max_words"),
        max_chars=_dict_get(config, "greeting_max_chars"),
        greetings=_dict_get(config, "greetings"),
        anti_patterns=_dict_get(config, "greeting_exclusions"),
        new_session_after_min=_dict_get(config, "new_session_after_min"),
        extra=(dict(config or {}) if isinstance(config, dict) else {}),
    )

    now_ts = _to_epoch(now if now is not None else datetime.now(timezone.utc).timestamp())
    local_date = _local_date(now_ts)
    local_region = _local_region(now_ts)

    intent = _pick_text(
        meta_map.get("intent"),
        state_map.get("intent"),
        state_map.get("active_mode"),
        state_map.get("mode"),
    ).lower()
    emotion = _pick_text(meta_map.get("emotion"), meta_map.get("mood"), state_map.get("mood"), state_map.get("emotion")).lower()
    profile = _pick_text(
        meta_map.get("active_personality_profile"),
        meta_map.get("personality"),
        meta_map.get("character"),
        state_map.get("active_character_id"),
        state_map.get("active_personality_id"),
        "default",
    ).lower()

    src = _normalize_text(text)
    message_len = len(src)
    user_words = len(_WORD_RE.findall(src))
    technical = is_technical(src)
    if not technical:
        technical = str(state_map.get("active_mode") or "").strip().lower() in {"engineer", "debugger", "planner"}
    tone = detect_user_tone(src)

    greeted_on_date = str(state_map.get("greeted_on_date") or "").strip()
    greeted_today = bool(greeted_on_date and greeted_on_date == local_date)

    threshold_sec = float(cfg["new_session_after_min"]) * 60.0
    new_session = is_new_session(now_ts, state_map.get("last_turn_ts"), threshold_sec=threshold_sec)
    session_id = str(state_map.get("session_id") or "").strip()
    prev_session_id = str(state_map.get("prev_session_id") or "").strip()
    if session_id and prev_session_id and session_id != prev_session_id:
        new_session = True

    conversation_state = str(state_map.get("conversation_state") or "").strip().lower()
    if not conversation_state:
        conversation_state = "new_session" if new_session else "continuing_smalltalk"

    user_greeting = is_user_greeting(
        src,
        max_words=int(cfg["max_words"]),
        max_chars=int(cfg["max_chars"]),
        greetings=list(cfg["greetings"]),
        anti_patterns=list(cfg["anti_patterns"]),
    )

    warmth = _clamp01(float(cfg["default_warmth_level"]))
    sarcasm = _clamp01(float(cfg["default_sarcasm_level"]))
    strictness = _clamp01(float(cfg["default_strictness_level"]))
    verbosity = _clamp01(float(cfg["default_verbosity_level"]))
    if profile in {"strict", "technical"}:
        warmth -= 0.10
        sarcasm -= 0.10
        strictness += 0.25
        verbosity -= 0.10
    elif profile in {"supportive"}:
        warmth += 0.20
        sarcasm -= 0.10
        strictness -= 0.10
        verbosity += 0.06
    elif profile in {"flirty"}:
        warmth += 0.12
        sarcasm += 0.10
        strictness -= 0.12
        verbosity += 0.05

    if message_len >= 360 or user_words >= 55:
        strictness += 0.12
        verbosity -= 0.08
    elif message_len <= 35 and user_words <= 6:
        verbosity += 0.10

    if intent in _TECH_INTENTS:
        strictness += 0.20
        warmth -= 0.08
        verbosity -= 0.08
    if emotion in _NEGATIVE_EMOTIONS:
        warmth += 0.12
        sarcasm -= 0.16
        strictness += 0.08
        verbosity -= 0.05
    elif emotion in _POSITIVE_EMOTIONS:
        warmth += 0.06
        sarcasm += 0.05
        strictness -= 0.05
    if tone == "hostile":
        strictness += 0.12
        warmth -= 0.10
        sarcasm -= 0.18
    elif tone == "friendly":
        warmth += 0.05
        strictness -= 0.04

    gap_hours = 0.0
    prev_ts = _to_epoch(state_map.get("last_turn_ts"))
    if prev_ts > 0:
        gap_hours = max(0.0, (now_ts - prev_ts) / 3600.0)
    if gap_hours >= 24.0:
        warmth += 0.06
        verbosity += 0.08
        strictness -= 0.05
    elif 0.0 < gap_hours <= 0.15:
        verbosity -= 0.07

    safety_mode = str(meta_map.get("safety_mode") or state_map.get("safety_mode") or "").strip().lower()
    safety_lock = bool(meta_map.get("safety_lock")) or bool(state_map.get("safety_lock"))

    decision_path: list[str] = []
    smalltalk_allowed = True
    greeting_allowed = False

    if safety_lock or safety_mode in {"strict", "locked"}:
        decision_path.append("safety")
        smalltalk_allowed = False
        greeting_allowed = bool(user_greeting and not technical)
        sarcasm = min(sarcasm, 0.02)
        warmth = min(max(warmth, 0.35), 0.55)
        strictness = max(strictness, 0.86)
        verbosity = min(verbosity, 0.35)
    else:
        decision_path.append("hard_constraints")
        if user_greeting and not technical:
            greeting_allowed = True
        elif new_session and not greeted_today and conversation_state != "continuing_smalltalk" and not technical:
            greeting_allowed = True
        if technical:
            decision_path.append("technical")
            smalltalk_allowed = False
            greeting_allowed = False
            sarcasm = min(sarcasm, float(cfg["technical_max_sarcasm"]))
            strictness = max(strictness, 0.82)
            warmth = max(warmth, 0.45)
            verbosity = min(verbosity, 0.40)
        elif emotion in _NEGATIVE_EMOTIONS or tone == "hostile":
            decision_path.append("emotional")
            smalltalk_allowed = False
            sarcasm = min(sarcasm, 0.12)
            warmth = max(warmth, 0.62)
            strictness = max(strictness, 0.62)
            verbosity = min(verbosity, 0.48)
        else:
            decision_path.append("casual")
            smalltalk_allowed = True
            if not greeting_allowed and user_greeting:
                greeting_allowed = True

    should_ask_back = bool(greeting_allowed and smalltalk_allowed and user_greeting and not technical)
    if not smalltalk_allowed:
        should_ask_back = False

    address_terms_policy = compute_address_terms_policy(
        text=src,
        now=now_ts,
        state=state_map,
        metadata={
            **meta_map,
            "intent": intent,
            "is_technical": bool(technical),
        },
        config=cfg,
    )

    dialog_mode = {
        "greeting_allowed": bool(greeting_allowed),
        "smalltalk_allowed": bool(smalltalk_allowed),
        "sarcasm_level": _clamp01(sarcasm),
        "warmth_level": _clamp01(warmth),
        "strictness_level": _clamp01(strictness),
        "verbosity_level": _clamp01(verbosity),
    }
    return {
        "dialog_mode": dialog_mode,
        "allow_greeting": bool(greeting_allowed),
        "smalltalk_allowed": bool(smalltalk_allowed),
        "should_ask_back": bool(should_ask_back),
        "user_greeting": bool(user_greeting),
        "new_session": bool(new_session),
        "greeted_today": bool(greeted_today),
        "conversation_state": conversation_state,
        "local_date": local_date,
        "local_region": local_region,
        "is_technical": bool(technical),
        "intent": intent,
        "emotion": emotion,
        "user_tone": tone,
        "active_profile": profile,
        "decision_path": decision_path,
        "message_len": message_len,
        "address_terms_policy": dict(address_terms_policy),
        "allowed_term": str(address_terms_policy.get("allowed_term") or ""),
        "use_term_now": bool(address_terms_policy.get("use_term_now", False)),
    }


def compute_dialog_flags(
    text: str,
    now,
    state: dict[str, Any] | None = None,
    *,
    config: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    out = compute_dialog_mode(text=text, now=now, state=state, metadata=metadata, config=config)
    mode = dict(out.get("dialog_mode") or {})
    out["allow_greeting"] = bool(mode.get("greeting_allowed", out.get("allow_greeting", False)))
    out["smalltalk_allowed"] = bool(mode.get("smalltalk_allowed", out.get("smalltalk_allowed", True)))
    policy = dict(out.get("address_terms_policy") or {})
    out["address_terms_policy"] = policy
    out["allowed_term"] = str(policy.get("allowed_term") or out.get("allowed_term") or "")
    out["use_term_now"] = bool(policy.get("use_term_now", out.get("use_term_now", False)))
    return out


def local_date_kyiv(now=None) -> str:
    ts = _to_epoch(now if now is not None else datetime.now(timezone.utc).timestamp())
    return _local_date(ts)


def local_region_name(now=None) -> str:
    ts = _to_epoch(now if now is not None else datetime.now(timezone.utc).timestamp())
    return _local_region(ts)


def _resolve_config(
    *,
    max_words: int | None,
    max_chars: int | None,
    greetings: list[str] | None,
    anti_patterns: list[str] | None,
    new_session_after_min: int | None,
    extra: dict[str, Any],
) -> dict[str, Any]:
    settings = load_config()
    defaults = {
        "new_session_after_min": max(1, int(settings.dialog_new_session_after_min)),
        "greeting_max_words": max(1, int(settings.dialog_greeting_max_words)),
        "greeting_max_chars": max(8, int(settings.dialog_greeting_max_chars)),
        "greetings": list(settings.dialog_greetings),
        "greeting_exclusions": list(settings.dialog_greeting_exclusions),
        "default_warmth_level": 0.58,
        "default_sarcasm_level": 0.24,
        "default_strictness_level": 0.56,
        "default_verbosity_level": 0.46,
        "technical_max_sarcasm": 0.08,
        "terms_enabled": True,
        "terms_list": ["милашка"],
        "terms_cooldown_turns": 6,
        "terms_cooldown_seconds": 900,
        "terms_max_per_session": 3,
        "terms_insert_probability": 0.35,
        "terms_ban_scope": "session",
        "terms_disable_patterns": [
            "не называй",
            "прекрати называть",
            "не зови",
            "прекрати звать",
            "stop calling me",
            "don't call me",
            "no pet names",
        ],
        "terms_enable_patterns": [
            "можно снова",
            "можешь снова",
            "можно называть",
            "call me again",
            "you can call me",
        ],
    }
    file_cfg = _load_data_policy_config(defaults)
    merged = dict(defaults)
    merged.update(dict(file_cfg or {}))
    for key, value in dict(extra or {}).items():
        if value is None:
            continue
        merged[key] = value

    greeting_max_words = _cfg_int(max_words if max_words is not None else merged.get("greeting_max_words"), 6, minimum=1)
    greeting_max_chars = _cfg_int(max_chars if max_chars is not None else merged.get("greeting_max_chars"), 35, minimum=8)
    new_session = _cfg_int(new_session_after_min if new_session_after_min is not None else merged.get("new_session_after_min"), 360, minimum=1)
    greetings_val = _cfg_list(greetings if greetings is not None else merged.get("greetings"))
    exclusions_val = _cfg_list(anti_patterns if anti_patterns is not None else merged.get("greeting_exclusions"))

    return {
        "max_words": greeting_max_words,
        "max_chars": greeting_max_chars,
        "greetings": greetings_val,
        "anti_patterns": exclusions_val,
        "new_session_after_min": new_session,
        "default_warmth_level": _clamp01(_cfg_float(merged.get("default_warmth_level"), 0.58)),
        "default_sarcasm_level": _clamp01(_cfg_float(merged.get("default_sarcasm_level"), 0.24)),
        "default_strictness_level": _clamp01(_cfg_float(merged.get("default_strictness_level"), 0.56)),
        "default_verbosity_level": _clamp01(_cfg_float(merged.get("default_verbosity_level"), 0.46)),
        "technical_max_sarcasm": _clamp01(_cfg_float(merged.get("technical_max_sarcasm"), 0.08)),
        "terms_enabled": _cfg_bool(merged.get("terms_enabled"), True),
        "terms_list": _cfg_list(merged.get("terms_list"), fallback=["милашка"]),
        "terms_cooldown_turns": _cfg_int(merged.get("terms_cooldown_turns"), 6, minimum=1),
        "terms_cooldown_seconds": _cfg_int(merged.get("terms_cooldown_seconds"), 900, minimum=1),
        "terms_max_per_session": _cfg_int(merged.get("terms_max_per_session"), 3, minimum=0),
        "terms_insert_probability": _clamp01(_cfg_float(merged.get("terms_insert_probability"), 0.35)),
        "terms_ban_scope": _cfg_ban_scope(merged.get("terms_ban_scope")),
        "terms_disable_patterns": _cfg_list(merged.get("terms_disable_patterns")),
        "terms_enable_patterns": _cfg_list(merged.get("terms_enable_patterns")),
    }


def _local_date(ts: float) -> str:
    return _local_dt(ts).date().isoformat()


def _local_region(ts: float) -> str:
    dt = _local_dt(ts)
    tz = dt.tzinfo
    if tz is None:
        return "local"
    key = getattr(tz, "key", None)
    if isinstance(key, str) and key.strip():
        return key.strip()
    name = dt.tzname()
    if isinstance(name, str) and name.strip():
        return name.strip()
    return str(tz)


def _to_epoch(value) -> float:
    return parse_time_to_epoch(value, 0.0)


def _normalize_text(value) -> str:
    text = _repair_mojibake(str(value or ""))
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t\f\v]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _normalize_tokenized(text: str) -> str:
    src = str(text or "").strip().lower()
    src = _NON_WORD_EDGE_RE.sub("", src)
    src = re.sub(r"\s+", " ", src)
    return src


def _greeting_start_regex(greetings: list[str]) -> re.Pattern:
    items = []
    for row in list(greetings or []):
        norm = _normalize_tokenized(row)
        if norm:
            items.append(re.escape(norm))
    if not items:
        items = [
            re.escape("привет"),
            re.escape("здравствуй"),
            re.escape("hello"),
            re.escape("hi"),
        ]
    items = sorted(set(items), key=len, reverse=True)
    pattern = r"^\s*(?:ну\s+)?(?:" + "|".join(items) + r")(?:$|[\s,!.?:;])"
    return re.compile(pattern, flags=re.IGNORECASE)


def _local_dt(ts: float) -> datetime:
    try:
        return datetime.fromtimestamp(float(ts)).astimezone()
    except Exception:
        return datetime.fromtimestamp(float(ts), timezone.utc)


def _iso_from_epoch(ts: float) -> str:
    try:
        return _local_dt(ts).isoformat(timespec="milliseconds")
    except Exception:
        return datetime.now().astimezone().isoformat(timespec="milliseconds")


def _load_data_policy_config(defaults: dict[str, Any]) -> dict[str, Any]:
    cfg = dict(defaults or {})
    path = Path(_DATA_POLICY_PATH)
    _ensure_data_policy_file(path, cfg)

    try:
        mtime = float(path.stat().st_mtime)
    except Exception:
        return cfg

    if _POLICY_CACHE.get("mtime") == mtime and isinstance(_POLICY_CACHE.get("data"), dict):
        cached_cfg = dict(cfg)
        cached_cfg.update(dict(_POLICY_CACHE.get("data") or {}))
        return cached_cfg

    loaded = _read_data_policy_file(path)
    if isinstance(loaded, dict):
        merged = dict(cfg)
        merged.update(loaded)
        _POLICY_CACHE["mtime"] = mtime
        _POLICY_CACHE["data"] = dict(loaded)
        return merged
    return cfg


def _ensure_data_policy_file(path: Path, defaults: dict[str, Any]) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    content = (
        "# Auto-generated dialog policy config.\n"
        "# You can edit these values and restart the app.\n\n"
        f"NEW_SESSION_AFTER_MIN = {int(defaults.get('new_session_after_min', 360))}\n"
        f"GREETING_MAX_WORDS = {int(defaults.get('greeting_max_words', 6))}\n"
        f"GREETING_MAX_CHARS = {int(defaults.get('greeting_max_chars', 35))}\n"
        f"GREETINGS = {repr(list(defaults.get('greetings') or []))}\n"
        f"GREETING_EXCLUSIONS = {repr(list(defaults.get('greeting_exclusions') or []))}\n"
        f"DEFAULT_WARMTH_LEVEL = {float(defaults.get('default_warmth_level', 0.58))}\n"
        f"DEFAULT_SARCASM_LEVEL = {float(defaults.get('default_sarcasm_level', 0.24))}\n"
        f"DEFAULT_STRICTNESS_LEVEL = {float(defaults.get('default_strictness_level', 0.56))}\n"
        f"DEFAULT_VERBOSITY_LEVEL = {float(defaults.get('default_verbosity_level', 0.46))}\n"
        f"TECHNICAL_MAX_SARCASM = {float(defaults.get('technical_max_sarcasm', 0.08))}\n\n"
        f"TERMS_ENABLED = {bool(defaults.get('terms_enabled', True))}\n"
        f"TERMS_LIST = {repr(list(defaults.get('terms_list') or ['милашка']))}\n"
        f"TERMS_COOLDOWN_TURNS = {int(defaults.get('terms_cooldown_turns', 6))}\n"
        f"TERMS_COOLDOWN_SECONDS = {int(defaults.get('terms_cooldown_seconds', 900))}\n"
        f"TERMS_MAX_PER_SESSION = {int(defaults.get('terms_max_per_session', 3))}\n"
        f"TERMS_INSERT_PROBABILITY = {float(defaults.get('terms_insert_probability', 0.35))}\n"
        f"TERMS_BAN_SCOPE = {repr(str(defaults.get('terms_ban_scope', 'session')))}\n"
        f"TERMS_DISABLE_PATTERNS = {repr(list(defaults.get('terms_disable_patterns') or []))}\n"
        f"TERMS_ENABLE_PATTERNS = {repr(list(defaults.get('terms_enable_patterns') or []))}\n"
    )
    path.write_text(content, encoding="utf-8")


def _read_data_policy_file(path: Path) -> dict[str, Any] | None:
    try:
        spec = importlib.util.spec_from_file_location("mmis_data_dialog_police", str(path))
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except Exception:
        return None

    out: dict[str, Any] = {}
    _read_int_attr(module, out, "NEW_SESSION_AFTER_MIN", "new_session_after_min", minimum=1)
    _read_int_attr(module, out, "GREETING_MAX_WORDS", "greeting_max_words", minimum=1)
    _read_int_attr(module, out, "GREETING_MAX_CHARS", "greeting_max_chars", minimum=8)

    greetings = _read_list_attr(module, "GREETINGS")
    if greetings:
        out["greetings"] = greetings
    excludes = _read_list_attr(module, "GREETING_EXCLUSIONS")
    if excludes:
        out["greeting_exclusions"] = excludes

    _read_float_attr(module, out, "DEFAULT_WARMTH_LEVEL", "default_warmth_level")
    _read_float_attr(module, out, "DEFAULT_SARCASM_LEVEL", "default_sarcasm_level")
    _read_float_attr(module, out, "DEFAULT_STRICTNESS_LEVEL", "default_strictness_level")
    _read_float_attr(module, out, "DEFAULT_VERBOSITY_LEVEL", "default_verbosity_level")
    _read_float_attr(module, out, "TECHNICAL_MAX_SARCASM", "technical_max_sarcasm")

    _read_bool_attr(module, out, "TERMS_ENABLED", "terms_enabled")
    terms = _read_list_attr(module, "TERMS_LIST")
    if terms:
        out["terms_list"] = terms
    _read_int_attr(module, out, "TERMS_COOLDOWN_TURNS", "terms_cooldown_turns", minimum=1)
    _read_int_attr(module, out, "TERMS_COOLDOWN_SECONDS", "terms_cooldown_seconds", minimum=1)
    _read_int_attr(module, out, "TERMS_MAX_PER_SESSION", "terms_max_per_session", minimum=0)
    _read_float_attr(module, out, "TERMS_INSERT_PROBABILITY", "terms_insert_probability")

    try:
        out["terms_ban_scope"] = _cfg_ban_scope(getattr(module, "TERMS_BAN_SCOPE"))
    except Exception:
        pass

    disable_patterns = _read_list_attr(module, "TERMS_DISABLE_PATTERNS")
    if disable_patterns:
        out["terms_disable_patterns"] = disable_patterns
    enable_patterns = _read_list_attr(module, "TERMS_ENABLE_PATTERNS")
    if enable_patterns:
        out["terms_enable_patterns"] = enable_patterns
    return out


def _read_int_attr(module, out: dict[str, Any], attr: str, key: str, *, minimum: int) -> None:
    try:
        out[key] = max(minimum, int(getattr(module, attr)))
    except Exception:
        return


def _read_float_attr(module, out: dict[str, Any], attr: str, key: str) -> None:
    try:
        out[key] = float(getattr(module, attr))
    except Exception:
        return


def _read_bool_attr(module, out: dict[str, Any], attr: str, key: str) -> None:
    try:
        out[key] = bool(getattr(module, attr))
    except Exception:
        return


def _read_list_attr(module, attr: str) -> list[str]:
    try:
        raw = list(getattr(module, attr))
    except Exception:
        return []
    out: list[str] = []
    seen: set[str] = set()
    for row in raw:
        item = str(row or "").strip()
        if not item:
            continue
        low = item.lower()
        if low in seen:
            continue
        seen.add(low)
        out.append(item)
    return out


def _pick_text(*values) -> str:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return ""


def _dict_get(data: dict[str, Any] | None, key: str):
    if not isinstance(data, dict):
        return None
    return data.get(key)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _cfg_int(value, default: int, *, minimum: int) -> int:
    try:
        out = int(value)
    except Exception:
        out = int(default)
    return max(minimum, out)


def _cfg_float(value, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _cfg_bool(value, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return bool(default)
    raw = str(value).strip().lower()
    if raw in {"1", "true", "yes", "on", "y", "t"}:
        return True
    if raw in {"0", "false", "no", "off", "n", "f"}:
        return False
    return bool(default)


def _cfg_list(value, *, fallback: list[str] | None = None) -> list[str]:
    if isinstance(value, (list, tuple, set)):
        raw = [str(x or "").strip() for x in list(value)]
    elif value is None:
        raw = []
    else:
        text = str(value).strip()
        raw = [x.strip() for x in text.split(",")] if text else []
    out: list[str] = []
    seen: set[str] = set()
    for row in raw:
        if not row:
            continue
        low = row.lower()
        if low in seen:
            continue
        seen.add(low)
        out.append(row)
    if not out and fallback:
        out = [str(x or "").strip() for x in list(fallback) if str(x or "").strip()]
    return out


def _cfg_ban_scope(value) -> str:
    scope = str(value or "session").strip().lower()
    if scope not in {"session"}:
        return "session"
    return scope


def _coerce_address_terms_state(value: dict[str, Any] | None) -> dict[str, Any]:
    raw = _as_dict(value)
    banned_terms = [str(x or "").strip().lower() for x in list(raw.get("banned_terms") or []) if str(x or "").strip()]
    unique_banned: list[str] = []
    seen: set[str] = set()
    for term in banned_terms:
        if term in seen:
            continue
        seen.add(term)
        unique_banned.append(term)

    banned_until: dict[str, Any] = {}
    for key, marker in dict(raw.get("banned_terms_until") or {}).items():
        term = str(key or "").strip().lower()
        if not term:
            continue
        banned_until[term] = marker

    return {
        "last_term_used_at": _to_epoch(raw.get("last_term_used_at")),
        "term_used_turn_index": _to_int(raw.get("term_used_turn_index"), 0),
        "terms_used_count_session": max(0, _to_int(raw.get("terms_used_count_session"), 0)),
        "session_id_snapshot": str(raw.get("session_id_snapshot") or "").strip(),
        "banned_terms": unique_banned,
        "banned_terms_until": banned_until,
        "last_disable_directive_ts": _to_epoch(raw.get("last_disable_directive_ts")),
        "last_enable_directive_ts": _to_epoch(raw.get("last_enable_directive_ts")),
    }


def _active_banned_terms_for_session(*, banned_terms: list[str], banned_terms_until: dict[str, Any], conversation_id: str) -> list[str]:
    active: list[str] = []
    for term in list(banned_terms or []):
        name = str(term or "").strip().lower()
        if not name:
            continue
        marker = banned_terms_until.get(name)
        if _is_ban_marker_active(marker=marker, conversation_id=conversation_id):
            active.append(name)
    return sorted(set(active))


def _is_ban_marker_active(*, marker, conversation_id: str) -> bool:
    raw = str(marker or "").strip()
    if not raw:
        return False
    if raw.startswith(_SESSION_BAN_MARKER_PREFIX):
        target = raw[len(_SESSION_BAN_MARKER_PREFIX) :].strip()
        if not target:
            return bool(conversation_id)
        return bool(conversation_id and target == conversation_id)
    return True


def _ban_marker_for_scope(scope: str, conversation_id: str) -> str:
    if str(scope or "").strip().lower() == "session":
        return f"{_SESSION_BAN_MARKER_PREFIX}{str(conversation_id or '').strip()}"
    return f"{_SESSION_BAN_MARKER_PREFIX}{str(conversation_id or '').strip()}"


def _as_dict(value) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if value is None:
        return {}
    try:
        return dict(vars(value))
    except Exception:
        return {}


def _to_int(value, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _last_phrase_pos(text: str, phrases: list[str]) -> int:
    best = -1
    for phrase in list(phrases or []):
        item = str(phrase or "").strip().lower()
        if not item:
            continue
        pos = text.rfind(item)
        if pos > best:
            best = pos
    return best


def _repair_mojibake(text: str) -> str:
    src = str(text or "")
    if not src:
        return ""
    if _contains_cyrillic(src):
        return src
    if "Гђ" not in src and "Г‘" not in src:
        return src
    try:
        repaired = src.encode("latin1", errors="ignore").decode("utf-8", errors="ignore")
    except Exception:
        return src
    return repaired if _contains_cyrillic(repaired) else src


def _contains_cyrillic(text: str) -> bool:
    return bool(re.search(r"[а-яА-ЯёЁ]", str(text or "")))
