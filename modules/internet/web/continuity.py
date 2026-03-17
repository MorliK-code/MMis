from __future__ import annotations

import datetime as dt
import hashlib
import re
from dataclasses import dataclass, field
from typing import Any

from modules.internet.web.query_text import normalize_search_text, strip_service_command_prefix


_WORD_RE = re.compile(r"[A-Za-zА-Яа-яЁёІіЇїЄєҐґ0-9_]+")
_FOLLOWUP_STACK_RE = re.compile(r"\s*;\s*follow-?up:\s*", flags=re.I)
_FOLLOWUP_LEADING_RE = re.compile(r"^(?:\s*(?:а|и|но|ну|ещё|еще|then|and|also)\s+)+", flags=re.I)


@dataclass(frozen=True)
class ContinuityConfig:
    ttl_minutes: int = 20
    max_user_turns: int = 6
    short_followup_max_tokens: int = 9
    followup_markers: tuple[str, ...] = (
        "а ",
        "и ",
        "но ",
        "then ",
        "and ",
        "also ",
        "ещё ",
        "еще ",
        "а на ",
        "а какие",
        "а сколько",
        "на сколько",
        "за сколько",
        "а бесплат",
        "какие лучше",
    )
    explicit_topic_markers: tuple[str, ...] = (
        "weather",
        "forecast",
        "погод",
        "currency",
        "usd",
        "eur",
        "uah",
        "курс",
        "news",
        "новост",
        "version",
        "release",
        "цена",
        "price",
        "rag",
        "embedding",
        "gpu",
        "llm",
    )


@dataclass(frozen=True)
class ContinuationResolution:
    used: bool
    resolved_query: str
    base_query: str = ""
    continuation_ref: str = ""
    context_confidence: float = 0.0
    reason: str = ""


def resolve_continuation(
    *,
    query: str,
    state: dict[str, Any] | None,
    config: ContinuityConfig | None = None,
    now_utc: dt.datetime | None = None,
) -> ContinuationResolution:
    cfg = config or ContinuityConfig()
    source = _sanitize_task_query(query)
    if not source:
        return ContinuationResolution(used=False, resolved_query="")
    if _is_self_memory_query(source):
        return ContinuationResolution(used=False, resolved_query=source, base_query=source, reason="self_memory_query")

    state_map = dict(state or {})
    active_task = _as_dict(state_map.get("web_active_task"))
    if not active_task:
        turns = _as_list(state_map.get("continuity_turns"))
        for row in reversed(turns):
            candidate = _as_dict(row)
            if candidate:
                active_task = candidate
                break
    if not active_task:
        return ContinuationResolution(used=False, resolved_query=source, base_query=source, reason="no_active_task")

    now = now_utc or dt.datetime.now(dt.timezone.utc)
    if _is_expired(active_task=active_task, state=state_map, cfg=cfg, now=now):
        return ContinuationResolution(used=False, resolved_query=source, base_query=source, reason="expired")

    low = source.lower()
    tokens = _WORD_RE.findall(low)
    short = len(tokens) <= max(1, int(cfg.short_followup_max_tokens))
    followup = _is_followup_phrase(low, cfg.followup_markers)
    has_explicit_topic = _has_any(low, cfg.explicit_topic_markers)

    # Do not force continuation when user starts an explicit new topic.
    if has_explicit_topic and not followup:
        return ContinuationResolution(used=False, resolved_query=source, base_query=source, reason="explicit_topic")
    if not followup and not short:
        return ContinuationResolution(used=False, resolved_query=source, base_query=source, reason="independent_query")

    base_query = _task_base_query(active_task=active_task)
    if not base_query:
        return ContinuationResolution(used=False, resolved_query=source, base_query=source, reason="missing_base_query")

    merged = _merge_query(base_query=base_query, followup_query=source)
    confidence = _continuity_confidence(active_task=active_task, followup=followup, short=short, now=now, cfg=cfg)
    return ContinuationResolution(
        used=True,
        resolved_query=merged,
        base_query=base_query,
        continuation_ref=str(active_task.get("task_id") or ""),
        context_confidence=confidence,
        reason="linked_to_active_task",
    )


def build_continuity_patch(
    *,
    state: dict[str, Any] | None,
    query: str,
    resolved_intent: str,
    resolved_concepts: list[str] | None = None,
    resolved_location: str = "",
    active_task: dict[str, Any] | None = None,
    continuation_ref: str = "",
    context_confidence: float = 0.0,
    now_utc: dt.datetime | None = None,
) -> dict[str, Any]:
    state_map = dict(state or {})
    now = now_utc or dt.datetime.now(dt.timezone.utc)
    now_iso = now.isoformat()
    task = _as_dict(active_task)
    cleaned_query = _sanitize_task_query(query)

    if not task:
        payload_key = "|".join(
            [
                str(cleaned_query or "").strip().lower(),
                str(resolved_intent or "").strip().lower(),
                str(resolved_location or "").strip().lower(),
            ]
        )
        task = {
            "task_id": f"task_{hashlib.sha1(payload_key.encode('utf-8', errors='ignore')).hexdigest()[:12]}",
            "query": str(cleaned_query or "").strip(),
            "resolved_intent": str(resolved_intent or "").strip().lower(),
            "resolved_concepts": [str(x or "").strip().lower() for x in list(resolved_concepts or []) if str(x or "").strip()],
            "resolved_location": str(resolved_location or "").strip(),
            "turn_timestamp": now_iso,
        }

    base_query = _task_base_query(active_task=task, fallback_query=cleaned_query)
    latest_query = _sanitize_task_query(task.get("latest_query") or cleaned_query or "")
    if not latest_query:
        latest_query = base_query
    if base_query:
        task["base_query"] = base_query
        task["query"] = base_query
    if latest_query:
        task["latest_query"] = latest_query
    task["turn_timestamp"] = str(task.get("turn_timestamp") or now_iso)
    if continuation_ref:
        task["continuation_ref"] = str(continuation_ref)
    if context_confidence > 0:
        task["context_confidence"] = round(float(context_confidence), 4)

    turns = _as_list(state_map.get("continuity_turns"))
    turns.append(task)
    turns = _truncate_turns(turns, cfg=ContinuityConfig(), now=now)

    return {
        "web_active_task": task,
        "continuity_turns": turns,
    }


def _merge_query(*, base_query: str, followup_query: str) -> str:
    base = _task_base_query(active_task={"query": base_query})
    follow = _normalize_followup_query(followup_query)
    if not base:
        return follow
    if not follow:
        return base
    if follow.lower() in base.lower():
        return base
    return f"{base} {follow}".strip()


def _task_base_query(*, active_task: dict[str, Any], fallback_query: str = "") -> str:
    base = _sanitize_task_query(active_task.get("base_query") or "")
    if base:
        return base
    query = _sanitize_task_query(active_task.get("query") or fallback_query or "")
    if not query:
        return ""
    parts = [str(x or "").strip() for x in _FOLLOWUP_STACK_RE.split(query) if str(x or "").strip()]
    return str(parts[0] if parts else query).strip()


def _normalize_followup_query(value: str) -> str:
    src = _sanitize_task_query(value)
    if not src:
        return ""
    src = _FOLLOWUP_LEADING_RE.sub("", src).strip()
    src = src.strip(" \t\r\n,;:-")
    return src.rstrip(" ?!.")


def _sanitize_task_query(value: Any) -> str:
    raw = strip_service_command_prefix(str(value or ""))
    cleaned = normalize_search_text(raw)
    return str(cleaned or raw or "").strip()


def _is_followup_phrase(text: str, markers: tuple[str, ...]) -> bool:
    low = str(text or "").strip().lower()
    if not low:
        return False
    for marker in markers:
        token = str(marker or "").strip().lower()
        if not token:
            continue
        if low.startswith(token):
            return True
        if f" {token}" in f" {low}":
            return True
    if low.endswith("?") and len(_WORD_RE.findall(low)) <= 8:
        return True
    return False


def _has_any(text: str, tokens: tuple[str, ...]) -> bool:
    low = str(text or "").strip().lower()
    return any(str(token).strip().lower() in low for token in tokens if str(token).strip())


def _is_self_memory_query(text: str) -> bool:
    low = str(text or "").strip().lower()
    if not low:
        return False
    target_markers = (
        "видюх",
        "видях",
        "видеокарт",
        "видеокарта",
        "карточк",
        "gpu",
        "graphics card",
        "video card",
        "python",
        "питон",
        "ос",
        "операционк",
        "operating system",
        "name",
        "имя",
        "age",
        "возраст",
    )
    context_markers = (
        "какая у меня",
        "какой у меня",
        "какое у меня",
        "какие у меня",
        "что у меня за",
        "что у меня с",
        "подскажи мою",
        "подскажи мой",
        "скажи мою",
        "скажи мой",
        "напомни мою",
        "напомни мой",
        "мой ",
        "моя ",
        "мою ",
        "my ",
        "what is my",
        "what's my",
        "remind me my",
        "на чём я",
    )
    if "на чём я" in low and any(token in low for token in ("сижу", "работаю", "кручусь", "run on", "running on")):
        return True
    if not any(token in low for token in target_markers):
        return False
    if any(marker in low for marker in context_markers):
        return True
    return bool(
        re.search(
            r"\b(?:какая|какой|какое|какие|what(?:'s| is)|which)\b.{0,48}\b(?:у меня|my)\b",
            low,
            flags=re.I,
        )
    )


def _is_expired(
    *,
    active_task: dict[str, Any],
    state: dict[str, Any],
    cfg: ContinuityConfig,
    now: dt.datetime,
) -> bool:
    raw_ts = str(active_task.get("turn_timestamp") or "").strip()
    active_turn_id = _to_int(active_task.get("turn_id"), 0)
    current_turn_id = _to_int(state.get("turn_id"), active_turn_id)

    if raw_ts:
        parsed = _parse_iso(raw_ts)
        if parsed is not None:
            delta_minutes = max(0.0, (now - parsed).total_seconds() / 60.0)
            if delta_minutes > max(1, int(cfg.ttl_minutes)):
                return True

    if active_turn_id > 0 and current_turn_id > 0:
        if (current_turn_id - active_turn_id) > max(1, int(cfg.max_user_turns)):
            return True
    return False


def _continuity_confidence(
    *,
    active_task: dict[str, Any],
    followup: bool,
    short: bool,
    now: dt.datetime,
    cfg: ContinuityConfig,
) -> float:
    score = 0.45
    if followup:
        score += 0.25
    if short:
        score += 0.10
    parsed = _parse_iso(str(active_task.get("turn_timestamp") or ""))
    if parsed is not None:
        delta_minutes = max(0.0, (now - parsed).total_seconds() / 60.0)
        freshness = max(0.0, 1.0 - (delta_minutes / max(1.0, float(cfg.ttl_minutes))))
        score += 0.20 * freshness
    return round(max(0.0, min(1.0, score)), 4)


def _truncate_turns(rows: list[dict[str, Any]], *, cfg: ContinuityConfig, now: dt.datetime) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in reversed(rows):
        item = _as_dict(row)
        if not item:
            continue
        parsed = _parse_iso(str(item.get("turn_timestamp") or ""))
        if parsed is not None:
            age_min = max(0.0, (now - parsed).total_seconds() / 60.0)
            if age_min > max(1, int(cfg.ttl_minutes)):
                continue
        out.append(item)
        if len(out) >= max(2, int(cfg.max_user_turns)):
            break
    out.reverse()
    return out


def _parse_iso(value: str) -> dt.datetime | None:
    src = str(value or "").strip()
    if not src:
        return None
    try:
        parsed = dt.datetime.fromisoformat(src.replace("Z", "+00:00"))
    except Exception:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc)


def _to_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return int(default)


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return [dict(x) for x in value if isinstance(x, dict)]
    return []
