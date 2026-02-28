from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


MODE_FAST = "FAST"
MODE_BALANCED = "BALANCED"
MODE_QUALITY = "QUALITY"


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
            # In quality mode keep all rules and slightly increase confidence for explicit first-person statements.
            enhanced = []
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

        return _dedupe_facts(out)


def extract(text: str, metadata: dict | None, speaker: str, mode: str = MODE_BALANCED) -> list[Fact]:
    return FactExtractor().extract(text=text, metadata=metadata, speaker=speaker, mode=mode)


# Backward-compat helper.
def extract_facts(text: str) -> dict:
    facts = extract(text=text, metadata={}, speaker="user", mode=MODE_FAST)
    return {"facts": [x.to_dict() for x in facts]}


def _extract_name(text: str, subject: str, evidence: str, event_id: str) -> list[Fact]:
    out = []
    patterns = [
        r"\bменя зовут\s+([А-ЯЁA-Z][а-яёa-zA-Z-]{1,30})\b",
        r"\bmy name is\s+([A-Z][a-zA-Z-]{1,30})\b",
        r"\bя\s+([А-ЯЁA-Z][а-яёa-zA-Z-]{1,30})\b",
    ]
    for p in patterns:
        m = re.search(p, text, re.I)
        if not m:
            continue
        name = str(m.group(1) or "").strip()
        if name:
            out.append(
                Fact(
                    subject=subject,
                    key="name",
                    value=name,
                    op="add",
                    confidence=0.86,
                    evidence=evidence,
                    source_event_id=event_id,
                )
            )
    return out


def _extract_likes(text: str, subject: str, evidence: str, event_id: str) -> list[Fact]:
    out = []
    like_patterns = [
        r"\bлюблю\s+([^.,;!?]+)",
        r"\bмне нравится\s+([^.,;!?]+)",
        r"\bi like\s+([^.,;!?]+)",
    ]
    dislike_patterns = [
        r"\bне люблю\s+([^.,;!?]+)",
        r"\bненавижу\s+([^.,;!?]+)",
        r"\bi (?:do not|don't) like\s+([^.,;!?]+)",
    ]
    for p in like_patterns:
        m = re.search(p, text, re.I)
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
    for p in dislike_patterns:
        m = re.search(p, text, re.I)
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
    out = []
    patterns = [
        r"\bя из\s+([А-ЯЁA-Z][^.,;!?]{1,40})",
        r"\bживу в\s+([А-ЯЁA-Z][^.,;!?]{1,40})",
        r"\bi(?:'m| am) from\s+([A-Z][^.,;!?]{1,40})",
    ]
    for p in patterns:
        m = re.search(p, text, re.I)
        if not m:
            continue
        value = str(m.group(1) or "").strip()
        if value:
            out.append(
                Fact(
                    subject=subject,
                    key="location",
                    value=value,
                    op="add",
                    confidence=0.7,
                    evidence=evidence,
                    source_event_id=event_id,
                )
            )
    return out


def _extract_birth(text: str, subject: str, evidence: str, event_id: str) -> list[Fact]:
    out = []
    m_year = re.search(r"\b(?:родил[ао]с[ья]|born)\s*(?:в|in)?\s*(19\d{2}|20\d{2})\b", text, re.I)
    if m_year:
        out.append(
            Fact(
                subject=subject,
                key="birth_year",
                value=str(m_year.group(1)),
                op="add",
                confidence=0.82,
                evidence=evidence,
                source_event_id=event_id,
            )
        )
    m_age = re.search(r"\bмне\s+(\d{1,2})\b|\bi am\s+(\d{1,2})\b", text, re.I)
    if m_age:
        age = next((x for x in m_age.groups() if x), "")
        if age:
            out.append(
                Fact(
                    subject=subject,
                    key="age",
                    value=str(age),
                    op="add",
                    confidence=0.66,
                    evidence=evidence,
                    source_event_id=event_id,
                )
            )
    return out


def _extract_device(text: str, subject: str, evidence: str, event_id: str) -> list[Fact]:
    out = []
    patterns = [
        r"\bу меня\s+(iphone|android|windows|linux|mac(?:book|os)?)\b",
        r"\bi use\s+(iphone|android|windows|linux|mac(?:book|os)?)\b",
    ]
    for p in patterns:
        m = re.search(p, text, re.I)
        if not m:
            continue
        out.append(
            Fact(
                subject=subject,
                key="device",
                value=str(m.group(1) or "").strip(),
                op="add",
                confidence=0.71,
                evidence=evidence,
                source_event_id=event_id,
            )
        )
    return out


def _extract_corrections(text: str, subject: str, evidence: str, event_id: str) -> list[Fact]:
    out = []
    m = re.search(r"\b(?:нет[, ]*)?я\s+не\s+(.{1,30}?)\s*,?\s+а\s+(.{1,30}?)(?:[.!?]|$)", text, re.I)
    if m:
        old = str(m.group(1) or "").strip()
        new = str(m.group(2) or "").strip()
        key = "statement"
        if re.fullmatch(r"(19\d{2}|20\d{2})", old) and re.fullmatch(r"(19\d{2}|20\d{2})", new):
            key = "birth_year"
        elif re.fullmatch(r"\d{1,2}", old) and re.fullmatch(r"\d{1,2}", new):
            key = "age"
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
    out = []
    m = re.search(r"\b(?:забудь|forget)\s+(?:что\s+)?(?:мой\s+)?([a-zа-я_ ]{2,30})\b", text, re.I)
    if m:
        key_raw = str(m.group(1) or "").strip().lower().replace(" ", "_")
        key = {
            "возраст": "age",
            "имя": "name",
            "город": "location",
            "name": "name",
            "age": "age",
            "location": "location",
        }.get(key_raw, key_raw)
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


def _is_explicit_self_statement(text: str) -> bool:
    src = str(text or "").lower()
    return any(token in src for token in ("я ", "i ", "my ", "мне ", "у меня"))


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

