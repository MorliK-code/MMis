from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class DialogEpisode:
    id: str
    topic: str
    turn_ids: list[str]
    summary_short: str
    summary_reasoning: str
    decisions: list[str]
    open_questions: list[str]
    participants: list[str]
    salience: float
    topic_keys: list[str]
    entity_keys: list[str]
    created_at: float = field(default_factory=lambda: float(time.time()))
    updated_at: float = field(default_factory=lambda: float(time.time()))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "DialogEpisode":
        return cls(
            id=str(row.get("id") or ""),
            topic=str(row.get("topic") or ""),
            turn_ids=[str(x).strip() for x in list(row.get("turn_ids") or []) if str(x).strip()],
            summary_short=str(row.get("summary_short") or ""),
            summary_reasoning=str(row.get("summary_reasoning") or ""),
            decisions=[str(x).strip() for x in list(row.get("decisions") or []) if str(x).strip()],
            open_questions=[str(x).strip() for x in list(row.get("open_questions") or []) if str(x).strip()],
            participants=[str(x).strip() for x in list(row.get("participants") or []) if str(x).strip()],
            salience=float(row.get("salience") or 0.0),
            topic_keys=[str(x).strip().lower() for x in list(row.get("topic_keys") or []) if str(x).strip()],
            entity_keys=[str(x).strip().lower() for x in list(row.get("entity_keys") or []) if str(x).strip()],
            created_at=float(row.get("created_at") or time.time()),
            updated_at=float(row.get("updated_at") or time.time()),
        )
