from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class ClaimCandidate:
    subject: str
    predicate: str
    object_surface: str
    object_type: str = ""
    head: str = ""
    normalized_object: str = ""
    alternatives: list[str] = field(default_factory=list)
    qualifiers: dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    specificity: float = 0.0
    evidence_text: str = ""
    evidence_span: tuple[int, int] | None = None
    topic_keys: list[str] = field(default_factory=list)
    trigger_keys: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ClaimRecord:
    subject: str
    predicate: str
    obj: str
    subject_type: str = ""
    object_type: str = ""
    object_surface: str = ""
    qualifiers: dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    salience: float = 0.0
    evidence_text: str = ""
    evidence_span: tuple[int, int] | None = None
    topic_keys: list[str] = field(default_factory=list)
    trigger_keys: list[str] = field(default_factory=list)
    recall_mode: str = "contextual"
    spontaneous_recall: bool = False
    status: str = "active"
    promotion_level: str = ""
    promotion_signals: dict[str, Any] = field(default_factory=dict)
    canonical_key: str = ""
    source_event_id: str = ""
    scope: str = "conversation"
    namespace: str = "default"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ClaimPromotionDecision:
    action: str = "allow"
    reason: str = ""
    allow_write: bool = True
    allow_supersede: bool = False
    signals: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": str(self.action or "").strip().lower() or "allow",
            "reason": str(self.reason or "").strip(),
            "allow_write": bool(self.allow_write),
            "allow_supersede": bool(self.allow_supersede),
            "signals": dict(self.signals or {}),
        }
