from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class Segment:
    text: str
    kind: str
    confidence: float
    meta: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"Segment(kind={self.kind!r}, confidence={self.confidence:.2f}, "
            f"text={self.text!r})"
        )


@dataclass(slots=True)
class Intent:
    name: str
    confidence: float
    source: str = "rule"


@dataclass(slots=True)
class Concept:
    name: str
    confidence: float
    source: str = "rule"


@dataclass(slots=True)
class Fact:
    key: str
    value: str
    confidence: float
    source: str = "rule"
    stable: bool = True
    evidence_count: int = 1

    def __repr__(self) -> str:
        return (
            "Fact("
            f"key={self.key!r}, value={self.value!r}, confidence={self.confidence:.2f}, "
            f"stable={self.stable}, evidence_count={self.evidence_count}"
            ")"
        )


@dataclass(slots=True)
class Emotion:
    name: str
    confidence: float
    source: str = "rule"
    scope: str = "current"


@dataclass(slots=True)
class Location:
    name: str
    kind: str
    confidence: float
    source: str = "message"


@dataclass(slots=True)
class SearchTask:
    kind: str
    query: str
    priority: int = 1
    location: str = ""
    origin_task_id: str = ""
    requires_freshness: bool = False
    geo_hint: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        location_part = f", location={self.location!r}" if self.location else ""
        origin_part = f", origin_task_id={self.origin_task_id!r}" if self.origin_task_id else ""
        return (
            f"SearchTask(kind={self.kind!r}, priority={self.priority}, "
            f"query={self.query!r}{location_part}{origin_part}, "
            f"requires_freshness={self.requires_freshness})"
        )


@dataclass(slots=True)
class UnderstandingResult:
    raw_text: str
    normalized_text: str
    continuation_ref: str = ""
    context_confidence: float = 0.0
    segments: list[Segment] = field(default_factory=list)
    intents: list[Intent] = field(default_factory=list)
    concepts: list[Concept] = field(default_factory=list)
    facts: list[Fact] = field(default_factory=list)
    emotions: list[Emotion] = field(default_factory=list)
    locations: list[Location] = field(default_factory=list)
    aliases_used: list[dict[str, Any]] = field(default_factory=list)
    search_tasks: list[SearchTask] = field(default_factory=list)
    memory_candidates: list[Fact] = field(default_factory=list)

    def __repr__(self) -> str:
        return (
            "UnderstandingResult("
            f"raw_text={self.raw_text!r}, normalized_text={self.normalized_text!r}, "
            f"continuation_ref={self.continuation_ref!r}, context_confidence={self.context_confidence:.2f}, "
            f"segments={len(self.segments)}, intents={len(self.intents)}, "
            f"concepts={len(self.concepts)}, facts={len(self.facts)}, "
            f"emotions={len(self.emotions)}, locations={len(self.locations)}, "
            f"search_tasks={len(self.search_tasks)}, memory_candidates={len(self.memory_candidates)}"
            ")"
        )
