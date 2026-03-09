from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any

from memory.memory_models import FactRecordV2, MemoryScope, MemoryStatus


MODE_FAST = "FAST"
MODE_BALANCED = "BALANCED"
MODE_QUALITY = "QUALITY"


@dataclass(frozen=True)
class Fact:
    """Legacy fact bridge kept for modules that still import Fact directly."""

    subject: str
    key: str
    value: Any
    op: str = "add"
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
    def extract_v2(
        self,
        *,
        text: str,
        metadata: dict | None,
        speaker: str,
        scope: MemoryScope = MemoryScope.CONVERSATION,
        mode: str = MODE_BALANCED,
    ) -> list[FactRecordV2]:
        src = str(text or "").strip()
        if not src:
            return []
        meta = dict(metadata or {})
        event_id = str(meta.get("event_id") or "")
        subject = self._subject_for_speaker(speaker)
        profile = str(mode or MODE_BALANCED).strip().upper()
        if profile not in {MODE_FAST, MODE_BALANCED, MODE_QUALITY}:
            profile = MODE_BALANCED

        rows: list[FactRecordV2] = []
        rows.extend(self._identity_facts(src, subject=subject, scope=scope, event_id=event_id))
        rows.extend(self._project_facts(src, subject=subject, scope=scope, event_id=event_id))
        rows.extend(self._environment_facts(src, subject=subject, scope=scope, event_id=event_id))
        rows.extend(self._preference_facts(src, subject=subject, scope=scope, event_id=event_id))
        rows.extend(self._task_facts(src, subject=subject, scope=scope, event_id=event_id))
        rows.extend(self._decision_facts(src, subject=subject, scope=scope, event_id=event_id))
        rows.extend(self._issue_facts(src, subject=subject, scope=scope, event_id=event_id))
        rows.extend(self._relationship_facts(src, subject=subject, scope=scope, event_id=event_id))
        rows.extend(self._temporary_facts(src, subject=subject, scope=scope, event_id=event_id))
        rows.extend(self._resolution_facts(src, subject=subject, scope=scope, event_id=event_id))

        if profile == MODE_FAST:
            allowed = {"identity", "preference", "task", "issue", "decision"}
            rows = [x for x in rows if str(x.relation or "") in allowed]
        elif profile == MODE_QUALITY:
            rows = [
                FactRecordV2(
                    subject=row.subject,
                    predicate=row.predicate,
                    value=row.value,
                    scope=row.scope,
                    confidence=min(1.0, float(row.confidence) + 0.05),
                    importance=min(1.0, float(row.importance) + 0.03),
                    evidence=row.evidence,
                    source_event_id=row.source_event_id,
                    valid_from=row.valid_from,
                    valid_to=row.valid_to,
                    status=row.status,
                    canonical_key=row.canonical_key,
                    relation=row.relation,
                )
                for row in rows
            ]

        return self._dedupe_v2(rows)

    def extract(
        self,
        text: str,
        metadata: dict | None,
        speaker: str,
        mode: str = MODE_BALANCED,
    ) -> list[Fact]:
        """Legacy bridge: converts a subset of v2 facts to old Fact shape."""
        v2_rows = self.extract_v2(
            text=text,
            metadata=metadata,
            speaker=speaker,
            scope=MemoryScope.CONVERSATION,
            mode=mode,
        )
        out: list[Fact] = []
        for row in v2_rows:
            key = str(row.predicate or "").strip().lower()
            if not key:
                continue
            out.append(
                Fact(
                    subject=str(row.subject or ""),
                    key=key,
                    value=row.value,
                    op="add",
                    confidence=float(row.confidence),
                    evidence=row.evidence,
                    source_event_id=row.source_event_id,
                    valid_from=row.valid_from,
                    valid_to=row.valid_to,
                )
            )
        return self._dedupe_legacy(out)

    @staticmethod
    def _subject_for_speaker(speaker: str) -> str:
        role = str(speaker or "").strip().lower()
        if role in {"assistant", "ai", "bot"}:
            return "assistant"
        if role in {"user", "human"}:
            return "user"
        return "other"

    @staticmethod
    def _mk(
        *,
        subject: str,
        predicate: str,
        value: Any,
        scope: MemoryScope,
        confidence: float,
        importance: float,
        evidence: str,
        event_id: str,
        relation: str,
        valid_to: float | None = None,
    ) -> FactRecordV2:
        pred = str(predicate or "").strip().lower().replace(" ", "_")
        canonical_key = f"{subject}.{pred}" if subject and pred else pred
        return FactRecordV2(
            subject=subject,
            predicate=pred,
            value=value,
            scope=scope,
            confidence=max(0.0, min(1.0, float(confidence))),
            importance=max(0.0, min(1.0, float(importance))),
            evidence=str(evidence or "")[:280],
            source_event_id=str(event_id or ""),
            valid_from=float(time.time()),
            valid_to=valid_to,
            status=MemoryStatus.ACTIVE,
            canonical_key=canonical_key,
            relation=str(relation or ""),
        )

    def _identity_facts(self, text: str, *, subject: str, scope: MemoryScope, event_id: str) -> list[FactRecordV2]:
        out: list[FactRecordV2] = []
        patterns = [
            r"\b(?:my name is|i am)\s+([A-Za-z\u0400-\u04ff][A-Za-z\u0400-\u04ff' -]{1,40})",
            r"\b(?:меня зовут)\s+([A-Za-z\u0400-\u04ff][A-Za-z\u0400-\u04ff' -]{1,40})",
        ]
        for pattern in patterns:
            m = re.search(pattern, text, re.I)
            if not m:
                continue
            name = str(m.group(1) or "").strip()
            if not name:
                continue
            out.append(
                self._mk(
                    subject=subject,
                    predicate="identity_name",
                    value=name,
                    scope=scope,
                    confidence=0.88,
                    importance=0.82,
                    evidence=text,
                    event_id=event_id,
                    relation="identity",
                )
            )
        return out

    def _project_facts(self, text: str, *, subject: str, scope: MemoryScope, event_id: str) -> list[FactRecordV2]:
        out: list[FactRecordV2] = []
        m = re.search(r"\bproject\s+([A-Za-z0-9_\-]{2,40})", text, re.I)
        if m:
            out.append(
                self._mk(
                    subject=subject,
                    predicate="project_name",
                    value=str(m.group(1) or "").strip(),
                    scope=MemoryScope.PROJECT,
                    confidence=0.74,
                    importance=0.78,
                    evidence=text,
                    event_id=event_id,
                    relation="project",
                )
            )
        return out

    def _environment_facts(self, text: str, *, subject: str, scope: MemoryScope, event_id: str) -> list[FactRecordV2]:
        out: list[FactRecordV2] = []
        env_tokens = {
            "windows": "windows",
            "linux": "linux",
            "mac": "macos",
            "ubuntu": "ubuntu",
            "docker": "docker",
        }
        low = str(text or "").lower()
        for token, value in env_tokens.items():
            if token not in low:
                continue
            out.append(
                self._mk(
                    subject=subject,
                    predicate="environment",
                    value=value,
                    scope=scope,
                    confidence=0.67,
                    importance=0.63,
                    evidence=text,
                    event_id=event_id,
                    relation="environment",
                )
            )
        return out

    def _preference_facts(self, text: str, *, subject: str, scope: MemoryScope, event_id: str) -> list[FactRecordV2]:
        out: list[FactRecordV2] = []
        pref_patterns = [
            r"\b(?:i prefer|prefer)\s+([^.,;!?]{2,120})",
            r"\b(?:i like)\s+([^.,;!?]{2,120})",
            r"\b(?:предпочитаю|люблю)\s+([^.,;!?]{2,120})",
        ]
        for pattern in pref_patterns:
            m = re.search(pattern, text, re.I)
            if not m:
                continue
            value = str(m.group(1) or "").strip()
            if not value:
                continue
            out.append(
                self._mk(
                    subject=subject,
                    predicate="preference",
                    value=value,
                    scope=scope,
                    confidence=0.72,
                    importance=0.66,
                    evidence=text,
                    event_id=event_id,
                    relation="preference",
                )
            )
        return out

    def _task_facts(self, text: str, *, subject: str, scope: MemoryScope, event_id: str) -> list[FactRecordV2]:
        out: list[FactRecordV2] = []
        patterns = [
            r"\b(?:need to|please|todo|task)\s+([^\n]{3,180})",
            r"\b(?:нужно|сделай|задача)\s+([^\n]{3,180})",
        ]
        for pattern in patterns:
            m = re.search(pattern, text, re.I)
            if not m:
                continue
            task = str(m.group(1) or "").strip()
            if not task:
                continue
            out.append(
                self._mk(
                    subject=subject,
                    predicate="task",
                    value=task,
                    scope=scope,
                    confidence=0.69,
                    importance=0.82,
                    evidence=text,
                    event_id=event_id,
                    relation="task",
                )
            )
        return out

    def _decision_facts(self, text: str, *, subject: str, scope: MemoryScope, event_id: str) -> list[FactRecordV2]:
        out: list[FactRecordV2] = []
        patterns = [
            r"\b(?:we decided|decided to|let's use)\s+([^.,;!?]{2,160})",
            r"\b(?:решили|давай использовать)\s+([^.,;!?]{2,160})",
        ]
        for pattern in patterns:
            m = re.search(pattern, text, re.I)
            if not m:
                continue
            decision = str(m.group(1) or "").strip()
            if not decision:
                continue
            out.append(
                self._mk(
                    subject=subject,
                    predicate="decision",
                    value=decision,
                    scope=scope,
                    confidence=0.78,
                    importance=0.84,
                    evidence=text,
                    event_id=event_id,
                    relation="decision",
                )
            )
        return out

    def _issue_facts(self, text: str, *, subject: str, scope: MemoryScope, event_id: str) -> list[FactRecordV2]:
        out: list[FactRecordV2] = []
        low = str(text or "").lower()
        markers = ["error", "failed", "exception", "traceback", "bug", "ошибка", "не работает"]
        if any(token in low for token in markers):
            out.append(
                self._mk(
                    subject=subject,
                    predicate="issue",
                    value=text[:200],
                    scope=scope,
                    confidence=0.76,
                    importance=0.81,
                    evidence=text,
                    event_id=event_id,
                    relation="issue",
                )
            )
        return out

    def _relationship_facts(self, text: str, *, subject: str, scope: MemoryScope, event_id: str) -> list[FactRecordV2]:
        out: list[FactRecordV2] = []
        patterns = [
            r"\bmy\s+(team|manager|colleague|client)\b",
            r"\b(команда|менеджер|коллега|клиент)\b",
        ]
        for pattern in patterns:
            m = re.search(pattern, text, re.I)
            if not m:
                continue
            value = str(m.group(1) or "").strip().lower()
            if not value:
                continue
            out.append(
                self._mk(
                    subject=subject,
                    predicate="relationship",
                    value=value,
                    scope=scope,
                    confidence=0.64,
                    importance=0.57,
                    evidence=text,
                    event_id=event_id,
                    relation="relationship",
                )
            )
        return out

    def _temporary_facts(self, text: str, *, subject: str, scope: MemoryScope, event_id: str) -> list[FactRecordV2]:
        low = str(text or "").lower()
        if not any(token in low for token in ("for now", "temporarily", "временно", "пока")):
            return []
        return [
            self._mk(
                subject=subject,
                predicate="temporary_fact",
                value=text[:160],
                scope=MemoryScope.TEMPORARY,
                confidence=0.62,
                importance=0.44,
                evidence=text,
                event_id=event_id,
                relation="temporary",
                valid_to=float(time.time() + (4 * 3600)),
            )
        ]

    def _resolution_facts(self, text: str, *, subject: str, scope: MemoryScope, event_id: str) -> list[FactRecordV2]:
        low = str(text or "").lower()
        out: list[FactRecordV2] = []
        if any(token in low for token in ("resolved", "fixed", "починил", "решено")):
            out.append(
                self._mk(
                    subject=subject,
                    predicate="issue_status",
                    value="resolved",
                    scope=scope,
                    confidence=0.73,
                    importance=0.70,
                    evidence=text,
                    event_id=event_id,
                    relation="resolved",
                )
            )
        elif any(token in low for token in ("still", "unresolved", "не решено", "все еще")):
            out.append(
                self._mk(
                    subject=subject,
                    predicate="issue_status",
                    value="unresolved",
                    scope=scope,
                    confidence=0.71,
                    importance=0.76,
                    evidence=text,
                    event_id=event_id,
                    relation="unresolved",
                )
            )
        return out

    @staticmethod
    def _dedupe_v2(rows: list[FactRecordV2]) -> list[FactRecordV2]:
        out: list[FactRecordV2] = []
        seen: set[tuple[str, str, str]] = set()
        for row in list(rows or []):
            key = (str(row.subject), str(row.predicate), str(row.value))
            if key in seen:
                continue
            seen.add(key)
            out.append(row)
        return out

    @staticmethod
    def _dedupe_legacy(rows: list[Fact]) -> list[Fact]:
        out: list[Fact] = []
        seen: set[tuple[str, str, str]] = set()
        for row in list(rows or []):
            key = (str(row.subject), str(row.key), str(row.value))
            if key in seen:
                continue
            seen.add(key)
            out.append(row)
        return out


def extract(text: str, metadata: dict | None, speaker: str, mode: str = MODE_BALANCED) -> list[Fact]:
    return FactExtractor().extract(text=text, metadata=metadata, speaker=speaker, mode=mode)


def extract_facts(text: str) -> dict[str, Any]:
    rows = FactExtractor().extract_v2(text=text, metadata={}, speaker="user", scope=MemoryScope.CONVERSATION)
    return {"facts": [x.to_dict() for x in rows]}

