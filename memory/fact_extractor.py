from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass
from typing import Any

from utils.logger import get_logger, log_json


MODE_FAST = "FAST"
MODE_BALANCED = "BALANCED"
MODE_QUALITY = "QUALITY"
LOGGER = get_logger(__name__)
_ALLOWED_FACT_KEYS = {"name", "birth_year", "age", "location", "likes", "dislikes", "device"}
_REMOVE_KEY_ALIASES = {
    "name": "name",
    "age": "age",
    "location": "location",
    "birth_year": "birth_year",
    "likes": "likes",
    "dislikes": "dislikes",
    "device": "device",
    "\u0438\u043c\u044f": "name",
    "\u0432\u043e\u0437\u0440\u0430\u0441\u0442": "age",
    "\u0433\u043e\u0440\u043e\u0434": "location",
    "\u043b\u043e\u043a\u0430\u0446\u0438\u044f": "location",
    "\u043c\u0435\u0441\u0442\u043e\u043f\u043e\u043b\u043e\u0436\u0435\u043d\u0438\u0435": "location",
}
_NAME_STOPWORDS = {
    "here",
    "there",
    "not",
    "no",
    "none",
    "nothing",
    "yes",
    "true",
    "false",
    "unknown",
    "test",
    "debug",
    "assistant",
    "bot",
    "ai",
    "\u0437\u0434\u0435\u0441\u044c",
    "\u0442\u0443\u0442",
    "\u043d\u0435",
    "\u043d\u0435\u0442",
    "\u043d\u0438\u043a\u0442\u043e",
    "\u043d\u0438\u0447\u0435\u0433\u043e",
    "\u0438\u043c\u044f",
    "\u0442\u0435\u0441\u0442",
    "\u0430\u0441\u0441\u0438\u0441\u0442\u0435\u043d\u0442",
}
_LOCATION_STOPWORDS = {
    "here",
    "there",
    "home",
    "unknown",
    "none",
    "\u0437\u0434\u0435\u0441\u044c",
    "\u0442\u0443\u0442",
    "\u0434\u043e\u043c\u0430",
    "\u043d\u0438\u0433\u0434\u0435",
}
_DEVICE_CANON = {
    "iphone": "iphone",
    "android": "android",
    "windows": "windows",
    "linux": "linux",
    "mac": "macos",
    "macos": "macos",
    "macbook": "macbook",
}


@dataclass(frozen=True)
class Fact:
    subject: str
    key: str
    value: Any
    op: str = "add"  # add/update/remove
    confidence: float = 0.5
    evidence: str = ""
    source_event_id: str = ""
    valid_from: float | None = None
    valid_to: float | None = None
    replaces_value: Any = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "subject": self.subject,
            "key": self.key,
            "value": self.value,
            "op": self.op,
            "confidence": float(self.confidence),
            "evidence": self.evidence,
            "source_event_id": self.source_event_id,
            "valid_from": self.valid_from,
            "valid_to": self.valid_to,
            "replaces_value": self.replaces_value,
        }


class FactExtractor:
    def extract(self, text: str, metadata: dict | None, speaker: str, mode: str = MODE_BALANCED) -> list[Fact]:
        src = str(text or "").strip()
        if not src:
            return []
        meta = dict(metadata or {})
        subject = _normalize_subject(speaker)
        profile = str(mode or MODE_BALANCED).strip().upper()
        profile = profile if profile in {MODE_FAST, MODE_BALANCED, MODE_QUALITY} else MODE_BALANCED
        event_id = str(meta.get("event_id") or "")
        evidence = src[:300]

        out: list[Fact] = []
        out.extend(_extract_name(src, subject, evidence, event_id))
        out.extend(_extract_likes(src, subject, evidence, event_id))
        out.extend(_extract_location(src, subject, evidence, event_id))
        out.extend(_extract_birth(src, subject, evidence, event_id))
        out.extend(_extract_device(src, subject, evidence, event_id))
        out.extend(_extract_corrections(src, subject, evidence, event_id))
        out.extend(_extract_remove(src, subject, evidence, event_id))

        if profile == MODE_FAST:
            keep = {"name", "birth_year", "likes", "dislikes"}
            out = [x for x in out if x.key in keep]
        elif profile == MODE_QUALITY:
            enhanced: list[Fact] = []
            for item in out:
                conf = min(1.0, float(item.confidence) + (0.05 if _is_explicit_self_statement(src) else 0.0))
                enhanced.append(
                    Fact(
                        subject=item.subject,
                        key=item.key,
                        value=item.value,
                        op=item.op,
                        confidence=conf,
                        evidence=item.evidence,
                        source_event_id=item.source_event_id,
                        valid_from=item.valid_from,
                        valid_to=item.valid_to,
                        replaces_value=item.replaces_value,
                    )
                )
            out = enhanced

        deduped = _dedupe_facts(out)
        validated: list[Fact] = []
        for fact in deduped:
            normalized, reason = _validate_and_normalize_fact(fact)
            if normalized is None:
                log_json(
                    LOGGER,
                    "fact_rejected_validation",
                    key=str(fact.key or ""),
                    op=str(fact.op or ""),
                    subject=str(fact.subject or ""),
                    reason=reason,
                )
                continue
            validated.append(normalized)
        return _dedupe_facts(validated)


def extract(text: str, metadata: dict | None, speaker: str, mode: str = MODE_BALANCED) -> list[Fact]:
    return FactExtractor().extract(text=text, metadata=metadata, speaker=speaker, mode=mode)


# Backward-compat helper.
def extract_facts(text: str) -> dict:
    facts = extract(text=text, metadata={}, speaker="user", mode=MODE_FAST)
    return {"facts": [x.to_dict() for x in facts]}


def _extract_name(text: str, subject: str, evidence: str, event_id: str) -> list[Fact]:
    out: list[Fact] = []
    patterns = [
        r"\b(?:my name is)\s+([A-Za-z][A-Za-z' -]{1,30})\b",
        r"\b(?:\u043c\u0435\u043d\u044f \u0437\u043e\u0432\u0443\u0442)\s+([A-Za-z\u0400-\u04ff][A-Za-z\u0400-\u04ff' -]{1,30})\b",
    ]
    for pattern in patterns:
        m = re.search(pattern, text, re.I)
        if not m:
            continue
        name = str(m.group(1) or "").strip()
        if not name:
            continue
        out.append(
            Fact(
                subject=subject,
                key="name",
                value=name,
                op="add",
                confidence=0.9,
                evidence=evidence,
                source_event_id=event_id,
            )
        )
    return out


def _extract_likes(text: str, subject: str, evidence: str, event_id: str) -> list[Fact]:
    out: list[Fact] = []
    like_patterns = [
        r"\b(?:i like)\s+([^.,;!?]{2,80})",
        r"\b(?:\u043b\u044e\u0431\u043b\u044e)\s+([^.,;!?]{2,80})",
        r"\b(?:\u043c\u043d\u0435 \u043d\u0440\u0430\u0432\u0438\u0442\u0441\u044f)\s+([^.,;!?]{2,80})",
    ]
    dislike_patterns = [
        r"\b(?:i (?:do not|don't) like)\s+([^.,;!?]{2,80})",
        r"\b(?:\u043d\u0435 \u043b\u044e\u0431\u043b\u044e)\s+([^.,;!?]{2,80})",
        r"\b(?:\u043d\u0435\u043d\u0430\u0432\u0438\u0436\u0443)\s+([^.,;!?]{2,80})",
    ]
    for pattern in like_patterns:
        m = re.search(pattern, text, re.I)
        if m:
            out.append(
                Fact(
                    subject=subject,
                    key="likes",
                    value=str(m.group(1) or "").strip(),
                    op="add",
                    confidence=0.74,
                    evidence=evidence,
                    source_event_id=event_id,
                )
            )
    for pattern in dislike_patterns:
        m = re.search(pattern, text, re.I)
        if m:
            out.append(
                Fact(
                    subject=subject,
                    key="dislikes",
                    value=str(m.group(1) or "").strip(),
                    op="add",
                    confidence=0.76,
                    evidence=evidence,
                    source_event_id=event_id,
                )
            )
    return out


def _extract_location(text: str, subject: str, evidence: str, event_id: str) -> list[Fact]:
    out: list[Fact] = []
    patterns = [
        r"\b(?:i(?:'m| am) from)\s+([A-Za-z\u0400-\u04ff][^.,;!?]{1,50})",
        r"\b(?:\u044f \u0438\u0437)\s+([A-Za-z\u0400-\u04ff][^.,;!?]{1,50})",
        r"\b(?:\u0436\u0438\u0432\u0443 \u0432)\s+([A-Za-z\u0400-\u04ff][^.,;!?]{1,50})",
    ]
    for pattern in patterns:
        m = re.search(pattern, text, re.I)
        if not m:
            continue
        value = str(m.group(1) or "").strip()
        if not value:
            continue
        out.append(
            Fact(
                subject=subject,
                key="location",
                value=value,
                op="add",
                confidence=0.72,
                evidence=evidence,
                source_event_id=event_id,
            )
        )
    return out


def _extract_birth(text: str, subject: str, evidence: str, event_id: str) -> list[Fact]:
    out: list[Fact] = []
    m_year = re.search(
        r"\b(?:born|was born|"
        r"\u044f \u0440\u043e\u0434\u0438\u043b\u0441\u044f|"
        r"\u044f \u0440\u043e\u0434\u0438\u043b\u0430\u0441\u044c|"
        r"\u0440\u043e\u0434\u0438\u043b\u0441\u044f|"
        r"\u0440\u043e\u0434\u0438\u043b\u0430\u0441\u044c)"
        r"\s*(?:in|\u0432)?\s*(19\d{2}|20\d{2})\b",
        text,
        re.I,
    )
    if m_year:
        out.append(
            Fact(
                subject=subject,
                key="birth_year",
                value=str(m_year.group(1)),
                op="add",
                confidence=0.84,
                evidence=evidence,
                source_event_id=event_id,
            )
        )
    m_age = re.search(
        r"(?:\bi am\s+(\d{1,3})(?:\s+years?\s+old)?\b)|"
        r"(?:\b(?:\u043c\u043d\u0435)\s+(\d{1,3})(?:\s+\u043b\u0435\u0442)?\b)",
        text,
        re.I,
    )
    if m_age:
        age = next((x for x in m_age.groups() if x), "")
        if age:
            out.append(
                Fact(
                    subject=subject,
                    key="age",
                    value=str(age),
                    op="add",
                    confidence=0.68,
                    evidence=evidence,
                    source_event_id=event_id,
                )
            )
    return out


def _extract_device(text: str, subject: str, evidence: str, event_id: str) -> list[Fact]:
    out: list[Fact] = []
    patterns = [
        r"\b(?:i use|i have)\s+(iphone|android|windows|linux|mac(?:book|os)?)\b",
        r"\b(?:\u0443 \u043c\u0435\u043d\u044f)\s+(iphone|android|windows|linux|mac(?:book|os)?)\b",
    ]
    for pattern in patterns:
        m = re.search(pattern, text, re.I)
        if not m:
            continue
        out.append(
            Fact(
                subject=subject,
                key="device",
                value=str(m.group(1) or "").strip(),
                op="add",
                confidence=0.73,
                evidence=evidence,
                source_event_id=event_id,
            )
        )
    return out


def _extract_corrections(text: str, subject: str, evidence: str, event_id: str) -> list[Fact]:
    out: list[Fact] = []
    patterns = [
        r"\b(?:no[, ]*)?i am not\s+(.{1,30}?)\s*,?\s+but\s+(.{1,30}?)(?:[.!?]|$)",
        r"\b(?:\u043d\u0435\u0442[, ]*)?\u044f\s+\u043d\u0435\s+(.{1,30}?)\s*,?\s+\u0430\s+(.{1,30}?)(?:[.!?]|$)",
    ]
    for pattern in patterns:
        m = re.search(pattern, text, re.I)
        if not m:
            continue
        old = str(m.group(1) or "").strip()
        new = str(m.group(2) or "").strip()
        if not new:
            continue
        key = ""
        if re.fullmatch(r"(19\d{2}|20\d{2})", old) and re.fullmatch(r"(19\d{2}|20\d{2})", new):
            key = "birth_year"
        elif re.fullmatch(r"\d{1,3}", old) and re.fullmatch(r"\d{1,3}", new):
            key = "age"
        elif re.fullmatch(r"[A-Za-z\u0400-\u04ff][A-Za-z\u0400-\u04ff' -]{1,30}", old) and re.fullmatch(
            r"[A-Za-z\u0400-\u04ff][A-Za-z\u0400-\u04ff' -]{1,30}",
            new,
        ):
            key = "name"
        if not key:
            continue
        out.append(
            Fact(
                subject=subject,
                key=key,
                value=new,
                op="update",
                confidence=0.83,
                evidence=evidence,
                source_event_id=event_id,
                replaces_value=old,
            )
        )
    return out


def _extract_remove(text: str, subject: str, evidence: str, event_id: str) -> list[Fact]:
    out: list[Fact] = []
    m = re.search(
        r"\b(?:forget|\u0437\u0430\u0431\u0443\u0434\u044c)\s+(?:that\s+)?(?:\u0447\u0442\u043e\s+)?(?:my\s+)?([A-Za-z\u0400-\u04ff_ ]{2,40})\b",
        text,
        re.I,
    )
    if not m:
        return out
    key_raw = str(m.group(1) or "").strip().lower().replace(" ", "_")
    key = _REMOVE_KEY_ALIASES.get(key_raw, key_raw)
    if key not in _ALLOWED_FACT_KEYS:
        return out
    out.append(
        Fact(
            subject=subject,
            key=key,
            value=None,
            op="remove",
            confidence=0.8,
            evidence=evidence,
            source_event_id=event_id,
        )
    )
    return out


def _validate_and_normalize_fact(fact: Fact) -> tuple[Fact | None, str]:
    key = str(fact.key or "").strip().lower()
    op = str(fact.op or "add").strip().lower()
    if key not in _ALLOWED_FACT_KEYS:
        return None, "unknown_key"
    if op not in {"add", "update", "remove"}:
        return None, "unknown_op"
    if op == "remove":
        return (
            Fact(
                subject=fact.subject,
                key=key,
                value=None,
                op="remove",
                confidence=fact.confidence,
                evidence=fact.evidence,
                source_event_id=fact.source_event_id,
                valid_from=fact.valid_from,
                valid_to=fact.valid_to,
                replaces_value=fact.replaces_value,
            ),
            "",
        )

    if key == "name":
        value = _normalize_name(fact.value)
        if not value:
            return None, "invalid_name"
    elif key == "age":
        value = _normalize_age(fact.value)
        if value is None:
            return None, "invalid_age"
    elif key == "birth_year":
        value = _normalize_birth_year(fact.value)
        if value is None:
            return None, "invalid_birth_year"
    elif key == "location":
        value = _normalize_location(fact.value)
        if not value:
            return None, "invalid_location"
    elif key in {"likes", "dislikes"}:
        value = _normalize_preference(fact.value)
        if not value:
            return None, f"invalid_{key}"
    elif key == "device":
        value = _normalize_device(fact.value)
        if not value:
            return None, "invalid_device"
    else:
        return None, "unsupported_key"

    return (
        Fact(
            subject=fact.subject,
            key=key,
            value=value,
            op=op,
            confidence=fact.confidence,
            evidence=fact.evidence,
            source_event_id=fact.source_event_id,
            valid_from=fact.valid_from,
            valid_to=fact.valid_to,
            replaces_value=fact.replaces_value,
        ),
        "",
    )


def _normalize_name(value: Any) -> str:
    src = re.sub(r"\s+", " ", str(value or "").strip()).strip(" .,;:!?\"'")
    if not src:
        return ""
    low = src.lower()
    if low in _NAME_STOPWORDS:
        return ""
    if re.search(r"\d", src):
        return ""
    if len(src) < 2 or len(src) > 32:
        return ""
    if not re.fullmatch(r"[A-Za-z\u0400-\u04ff][A-Za-z\u0400-\u04ff' -]{1,31}", src):
        return ""
    return src


def _normalize_age(value: Any) -> str | None:
    try:
        age = int(str(value).strip())
    except Exception:
        return None
    if age < 1 or age > 120:
        return None
    return str(age)


def _normalize_birth_year(value: Any) -> str | None:
    try:
        year = int(str(value).strip())
    except Exception:
        return None
    max_year = dt.datetime.now().year
    if year < 1900 or year > max_year:
        return None
    return str(year)


def _normalize_location(value: Any) -> str:
    src = re.sub(r"\s+", " ", str(value or "").strip()).strip(" .,;:!?\"'")
    if not src:
        return ""
    low = src.lower()
    if low in _LOCATION_STOPWORDS:
        return ""
    if len(src) < 2 or len(src) > 64:
        return ""
    if re.fullmatch(r"\d+", src):
        return ""
    return src


def _normalize_preference(value: Any) -> str:
    src = re.sub(r"\s+", " ", str(value or "").strip()).strip(" .,;:!?\"'")
    if len(src) < 2 or len(src) > 80:
        return ""
    if src.lower() in {"nothing", "none", "\u043d\u0438\u0447\u0435\u0433\u043e", "\u043d\u0435\u0442"}:
        return ""
    return src


def _normalize_device(value: Any) -> str:
    src = str(value or "").strip().lower()
    if not src:
        return ""
    canon = _DEVICE_CANON.get(src, "")
    return canon


def _is_explicit_self_statement(text: str) -> bool:
    src = str(text or "").lower()
    return any(
        token in src
        for token in (
            " i ",
            " my ",
            " i am ",
            " i'm ",
            "\u044f ",
            "\u043c\u043d\u0435 ",
            "\u0443 \u043c\u0435\u043d\u044f",
        )
    )


def _normalize_subject(speaker: str) -> str:
    raw = str(speaker or "").strip().lower()
    if raw in {"assistant", "ai", "bot"}:
        return "assistant"
    if raw in {"user", "human"}:
        return "user"
    return "other"


def _dedupe_facts(items: list[Fact]) -> list[Fact]:
    out: list[Fact] = []
    seen: set[tuple[str, str, str, str]] = set()
    for row in items:
        key = (
            str(row.subject),
            str(row.key),
            str(row.op),
            str(row.value),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out
