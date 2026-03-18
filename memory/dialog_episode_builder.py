from __future__ import annotations

import re
import time
import uuid
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from memory.dialog_episode_models import DialogEpisode
from memory.retrieval_projection import build_memory_views
from modules.nlu.normalizer import normalize_text


_TOPIC_TAG_RE = re.compile(r"^topic_(.+)$", re.I)
_EXPLICIT_END_RE = re.compile(
    r"(?:\b(?:договорились|решили|закрыли|закончили|ладно[,]?\s*дальше|ок(?:ей)?[,]?\s*дальше|идем\s+дальше|перейдем\s+дальше|done|resolved|let'?s\s+move\s+on|moving\s+on)\b)",
    re.I,
)
_DECISION_RE = re.compile(
    r"(?:\b(?:решили|договорились|будем|сделаем|следующим\s+шагом|next\s+step|we\s+will|let'?s)\b)",
    re.I,
)
_QUESTIONISH_RE = re.compile(
    r"(?:\?|(?:\b(?:непонятно|не\s+ясно|что\s+дальше|осталось\s+понять|open\s+question|todo)\b))",
    re.I,
)
_WORD_RE = re.compile(r"[a-zа-яёіїєґ0-9_]{3,}", re.I)
_STOPWORDS = {
    "about",
    "after",
    "also",
    "because",
    "been",
    "from",
    "that",
    "this",
    "with",
    "если",
    "еще",
    "ещё",
    "как",
    "надо",
    "очень",
    "про",
    "сделать",
    "сделаем",
    "сейчас",
    "так",
    "там",
    "только",
    "это",
    "что",
}


@dataclass(frozen=True)
class DialogTurn:
    turn_id: str
    role: str
    text: str
    ts: float = field(default_factory=lambda: float(time.time()))
    topic: str = ""
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DialogEpisodeBoundary:
    should_build: bool
    reason: str = ""
    split_index: int = 0


@dataclass
class DialogEpisodeBuilder:
    max_window_turns: int = 8
    time_gap_sec: float = 20.0 * 60.0
    min_turns_for_episode: int = 2

    def evaluate_boundary(self, turns: list[DialogTurn]) -> DialogEpisodeBoundary:
        rows = [row for row in list(turns or []) if str(row.text or "").strip()]
        if len(rows) < max(1, int(self.min_turns_for_episode)):
            return DialogEpisodeBoundary(should_build=False)

        for idx in range(1, len(rows)):
            prev = rows[idx - 1]
            cur = rows[idx]
            if float(cur.ts or 0.0) - float(prev.ts or 0.0) >= float(self.time_gap_sec):
                if idx >= int(self.min_turns_for_episode):
                    return DialogEpisodeBoundary(should_build=True, reason="time_gap", split_index=idx)

        for idx in range(1, len(rows)):
            prev_topic = self._turn_topic(rows[idx - 1])
            cur_topic = self._turn_topic(rows[idx])
            if prev_topic and cur_topic and prev_topic != cur_topic and idx >= int(self.min_turns_for_episode):
                return DialogEpisodeBoundary(should_build=True, reason="topic_shift", split_index=idx)

        if any(_EXPLICIT_END_RE.search(str(turn.text or "")) for turn in rows[-2:]):
            return DialogEpisodeBoundary(should_build=True, reason="explicit_completion", split_index=len(rows))

        if len(rows) >= int(self.max_window_turns):
            return DialogEpisodeBoundary(
                should_build=True,
                reason="window_limit",
                split_index=min(len(rows), int(self.max_window_turns)),
            )

        return DialogEpisodeBoundary(should_build=False)

    def build_if_needed(
        self,
        turns: list[DialogTurn],
        *,
        now_ts: float | None = None,
        episode_id: str | None = None,
    ) -> tuple[DialogEpisodeBoundary, DialogEpisode | None]:
        boundary = self.evaluate_boundary(turns)
        if not boundary.should_build:
            return boundary, None
        return boundary, self.build_episode(
            self.boundary_turns(turns, boundary),
            episode_id=episode_id,
            now_ts=now_ts,
        )

    def boundary_turns(self, turns: list[DialogTurn], boundary: DialogEpisodeBoundary) -> list[DialogTurn]:
        rows = [row for row in list(turns or []) if str(row.text or "").strip()]
        if not rows:
            return []
        split_index = int(boundary.split_index or 0)
        if split_index <= 0 or split_index >= len(rows):
            return rows
        return rows[:split_index]

    def build_episode(
        self,
        turns: list[DialogTurn],
        *,
        episode_id: str | None = None,
        now_ts: float | None = None,
    ) -> DialogEpisode | None:
        rows = [row for row in list(turns or []) if str(row.text or "").strip()]
        if len(rows) < max(1, int(self.min_turns_for_episode)):
            return None

        topic = self._episode_topic(rows)
        decisions = self._extract_decisions(rows)
        open_questions = self._extract_open_questions(rows)
        participants = self._participants(rows)
        topic_keys = self._topic_keys(rows, topic=topic)
        entity_keys = self._entity_keys(rows)
        summary_short = self._summary_short(topic=topic, decisions=decisions, open_questions=open_questions)
        summary_reasoning = self._summary_reasoning(topic=topic, decisions=decisions, open_questions=open_questions)
        ts = float(now_ts or rows[-1].ts or time.time())

        return DialogEpisode(
            id=str(episode_id or f"episode:{uuid.uuid4().hex[:18]}"),
            topic=topic,
            turn_ids=[str(row.turn_id or "").strip() for row in rows if str(row.turn_id or "").strip()],
            summary_short=summary_short,
            summary_reasoning=summary_reasoning,
            decisions=decisions,
            open_questions=open_questions,
            participants=participants,
            salience=self._salience(rows, decisions=decisions, open_questions=open_questions),
            topic_keys=topic_keys,
            entity_keys=entity_keys,
            created_at=ts,
            updated_at=ts,
        )

    def _turn_topic(self, turn: DialogTurn) -> str:
        explicit = str(turn.topic or "").strip().lower()
        if explicit:
            return explicit

        for tag in list(turn.tags or []):
            token = str(tag or "").strip().lower()
            match = _TOPIC_TAG_RE.match(token)
            if match:
                return str(match.group(1) or "").strip().lower()

        meta = dict(turn.metadata or {})
        explicit_meta = str(meta.get("topic") or "").strip().lower()
        if explicit_meta:
            return explicit_meta

        for item in list(meta.get("topics") or []):
            token = str(item or "").strip().lower().replace("topic_", "", 1)
            if token:
                return token

        return self._fallback_topic(str(turn.text or ""))

    def _episode_topic(self, turns: list[DialogTurn]) -> str:
        topics = [self._turn_topic(turn) for turn in list(turns or [])]
        topics = [token for token in topics if token]
        if topics:
            return Counter(topics).most_common(1)[0][0]

        merged = " ".join(str(turn.text or "").strip() for turn in list(turns or []))
        fallback = self._fallback_topic(merged)
        return fallback or "dialog_context"

    def _fallback_topic(self, text: str) -> str:
        tokens = [
            token
            for token in _WORD_RE.findall(normalize_text(str(text or "")))
            if token and token not in _STOPWORDS
        ]
        if not tokens:
            return ""
        return Counter(tokens).most_common(1)[0][0]

    def _extract_decisions(self, turns: list[DialogTurn]) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()
        for turn in list(turns or []):
            text = str(turn.text or "").strip()
            if not text or not _DECISION_RE.search(text):
                continue
            token = text[:220].strip()
            key = normalize_text(token)
            if not key or key in seen:
                continue
            seen.add(key)
            out.append(token)
        return out

    def _extract_open_questions(self, turns: list[DialogTurn]) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()
        for turn in list(turns or []):
            text = str(turn.text or "").strip()
            if not text or not _QUESTIONISH_RE.search(text):
                continue
            token = text[:220].strip()
            key = normalize_text(token)
            if not key or key in seen:
                continue
            seen.add(key)
            out.append(token)
        return out

    def _participants(self, turns: list[DialogTurn]) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for turn in list(turns or []):
            token = str(turn.role or "").strip().lower()
            if not token or token in seen:
                continue
            seen.add(token)
            out.append(token)
        return out

    def _topic_keys(self, turns: list[DialogTurn], *, topic: str) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []

        def _add(value: str) -> None:
            token = str(value or "").strip().lower()
            if not token or token in seen:
                return
            seen.add(token)
            out.append(token)

        _add(topic)
        for turn in list(turns or []):
            _add(self._turn_topic(turn))
            for tag in list(turn.tags or []):
                raw = str(tag or "").strip().lower()
                match = _TOPIC_TAG_RE.match(raw)
                if match:
                    _add(str(match.group(1) or "").strip().lower())
        return out

    def _entity_keys(self, turns: list[DialogTurn]) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []

        def _add(value: str) -> None:
            token = str(value or "").strip().lower()
            if not token or token in seen:
                return
            seen.add(token)
            out.append(token)

        for turn in list(turns or []):
            meta = dict(turn.metadata or {})
            for token in list(dict(meta.get("memory_views") or {}).get("entity_keys") or []):
                _add(str(token or ""))
            for token in list(meta.get("entity_keys") or []):
                _add(str(token or ""))
            for row in list(meta.get("memory_entities") or []):
                if isinstance(row, dict):
                    _add(str(row.get("canonical") or row.get("surface") or ""))

        if out:
            return out

        merged_text = "\n".join(str(turn.text or "").strip() for turn in turns if str(turn.text or "").strip())
        views = build_memory_views(merged_text)
        for token in list(views.get("entity_keys") or []):
            _add(str(token or ""))
        return out

    def _summary_short(self, *, topic: str, decisions: list[str], open_questions: list[str]) -> str:
        if decisions and open_questions:
            return f"Discussed {topic}; made {len(decisions)} decision(s) and left {len(open_questions)} open question(s)."
        if decisions:
            return f"Discussed {topic}; made {len(decisions)} decision(s)."
        if open_questions:
            return f"Discussed {topic}; left {len(open_questions)} open question(s)."
        return f"Discussed {topic}."

    def _summary_reasoning(self, *, topic: str, decisions: list[str], open_questions: list[str]) -> str:
        parts: list[str] = [f"The conversation focused on {topic}."]
        if decisions:
            parts.append("Decisions: " + "; ".join(list(decisions or [])[:2]) + ".")
        if open_questions:
            parts.append("Open questions: " + "; ".join(list(open_questions or [])[:2]) + ".")
        if not decisions and not open_questions:
            parts.append("Turns clarified context and next steps.")
        return " ".join(part.strip() for part in parts if part.strip())

    def _salience(self, turns: list[DialogTurn], *, decisions: list[str], open_questions: list[str]) -> float:
        turn_score = min(0.36, 0.08 * len(list(turns or [])))
        decision_score = min(0.28, 0.10 * len(list(decisions or [])))
        question_score = min(0.20, 0.06 * len(list(open_questions or [])))
        role_bonus = 0.08 if len(self._participants(turns)) > 1 else 0.0
        return max(0.0, min(1.0, 0.24 + turn_score + decision_score + question_score + role_bonus))
