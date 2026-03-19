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
    r"(?:"
    r"\b(?:решили|договорились|будем|сделаем|следующим\s+шагом|agreed|next\s+step|we\s+will|let'?s)\b"
    r"|"
    r"\b(?:сначала|сперва|first)\b.{0,120}\b(?:потом|затем|then)\b"
    r")",
    re.I,
)
_QUESTIONISH_RE = re.compile(
    r"(?:\?|(?:\b(?:непонятно|не\s+ясно|что\s+дальше|осталось\s+понять|open\s+question|todo)\b))",
    re.I,
)
_WORD_RE = re.compile(r"[a-zа-яёіїєґ0-9_]{3,}", re.I)
_SUMMARY_SPLIT_RE = re.compile(r"[\n\r]+|(?<=[.!?])\s+")
_SUMMARY_DISCOURSE_PREFIX_RE = re.compile(
    r"^(?:ну|ладно|ок(?:ей)?|хорошо|так|слушай|смотри|короче|в общем|итак|значит)\b[\s,:;-]*",
    re.I,
)
_SUMMARY_NOISE_RE = re.compile(
    r"^(?:привет|здравствуй(?:те)?|добрый\s+(?:день|вечер|утро)|hello|hi|ага|угу|ок(?:ей)?|ясно|понятно|спасибо|thanks?)$",
    re.I,
)
_BAD_TOPIC_TOKENS = {
    "ага",
    "да",
    "нет",
    "ок",
    "окей",
    "ладно",
    "понял",
    "поняла",
    "привет",
    "hello",
    "hi",
    "no",
    "okay",
    "yes",
}
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

        decisions = self._extract_decisions(rows)
        open_questions = self._extract_open_questions(rows)
        topic = self._episode_topic(
            rows,
            decisions=decisions,
            open_questions=open_questions,
        )
        participants = self._participants(rows)
        topic_keys = self._topic_keys(rows, topic=topic)
        entity_keys = self._entity_keys(rows)
        summary_short = self._summary_short(
            turns=rows,
            topic=topic,
            decisions=decisions,
            open_questions=open_questions,
        )
        summary_reasoning = self._summary_reasoning(
            turns=rows,
            topic=topic,
            decisions=decisions,
            open_questions=open_questions,
        )
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
        explicit = self._topic_candidate(turn.topic)
        if explicit:
            return explicit

        for tag in list(turn.tags or []):
            token = str(tag or "").strip().lower()
            match = _TOPIC_TAG_RE.match(token)
            if match:
                candidate = self._topic_candidate(match.group(1))
                if candidate:
                    return candidate

        meta = dict(turn.metadata or {})
        explicit_meta = self._topic_candidate(meta.get("topic"))
        if explicit_meta:
            return explicit_meta

        for item in list(meta.get("topics") or []):
            candidate = self._topic_candidate(str(item or "").strip().replace("topic_", "", 1))
            if candidate:
                return candidate

        return self._fallback_topic(str(turn.text or ""))

    def _episode_topic(
        self,
        turns: list[DialogTurn],
        *,
        decisions: list[str] | None = None,
        open_questions: list[str] | None = None,
    ) -> str:
        topics = [self._turn_topic(turn) for turn in list(turns or [])]
        topics = [token for token in topics if token]
        if topics:
            return Counter(topics).most_common(1)[0][0]

        signal_topics = self._topic_signal_candidates(turns)
        if signal_topics:
            return signal_topics[0]

        dialog_signal_topic = self._fallback_topic(" ".join([*(decisions or []), *(open_questions or [])]).strip())
        if dialog_signal_topic:
            return dialog_signal_topic

        merged = " ".join(str(turn.text or "").strip() for turn in list(turns or []))
        fallback = self._fallback_topic(merged)
        return fallback or "general_dialog"

    def _topic_signal_candidates(self, turns: list[DialogTurn]) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []

        def _add(value: Any) -> None:
            token = self._topic_candidate(value)
            if not token or token in seen:
                return
            seen.add(token)
            out.append(token)

        for turn in list(turns or []):
            meta = dict(turn.metadata or {})
            views = dict(meta.get("memory_views") or {})
            for token in list(views.get("topic_keys") or []):
                _add(token)
            for token in list(meta.get("topic_keys") or []):
                _add(token)
            for token in list(views.get("entity_keys") or []):
                _add(token)
            for token in list(meta.get("entity_keys") or []):
                _add(token)
            for row in list(meta.get("memory_entities") or []):
                if isinstance(row, dict):
                    _add(row.get("canonical") or row.get("surface") or "")
        return out

    def _topic_candidate(self, value: Any) -> str:
        token = self._normalize_topic_token(value)
        if not token:
            return ""
        compact = token.replace("_", " ")
        words = [item for item in _WORD_RE.findall(compact) if item]
        if len(words) == 1 and words[0] in _STOPWORDS:
            return ""
        if self._is_bad_topic(token):
            return ""
        return token

    def _normalize_topic_token(self, value: Any) -> str:
        token = normalize_text(str(value or "")).strip().lower()
        token = re.sub(r"\s+", " ", token)
        return token

    def _is_bad_topic(self, value: Any) -> bool:
        token = self._normalize_topic_token(value)
        if not token:
            return False
        compact = token.replace("_", " ")
        if compact in _BAD_TOPIC_TOKENS:
            return True
        words = [item for item in _WORD_RE.findall(compact) if item]
        return len(words) == 1 and words[0] in _BAD_TOPIC_TOKENS

    def _fallback_topic(self, text: str) -> str:
        tokens = [
            token
            for token in _WORD_RE.findall(normalize_text(str(text or "")))
            if token and token not in _STOPWORDS and token not in _BAD_TOPIC_TOKENS
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

    def _episode_text(self, rows: list[DialogTurn]) -> str:
        return self._join_episode_fragments(
            self._episode_fragments(rows),
            maximum=520,
        )

    def _episode_user_text(self, rows: list[DialogTurn]) -> str:
        return self._join_episode_fragments(
            self._episode_fragments(
                rows,
                include_assistant=False,
            ),
            maximum=420,
        )

    def _episode_compact_text(self, rows: list[DialogTurn]) -> str:
        summary_rows = self._summary_source_turns(rows)
        source_rows = summary_rows or list(rows or [])
        compact_fragments = self._episode_fragments(
            source_rows,
            assistant_must_be_important=True,
        )
        if not compact_fragments:
            compact_fragments = self._episode_fragments(source_rows)
        return self._join_episode_fragments(compact_fragments, maximum=220)

    def _summary_source_turns(self, rows: list[DialogTurn], *, maximum: int = 4) -> list[DialogTurn]:
        meaningful: list[DialogTurn] = []
        for turn in list(rows or []):
            if self._is_noise_turn(turn):
                continue
            role = str(turn.role or "").strip().lower()
            if role == "assistant" and not self._is_important_assistant_turn(turn):
                continue
            meaningful.append(turn)
            if len(meaningful) >= maximum:
                break
        if meaningful:
            return meaningful
        return [
            turn
            for turn in list(rows or [])
            if str(turn.text or "").strip()
        ][:maximum]

    def _episode_fragments(
        self,
        rows: list[DialogTurn],
        *,
        include_user: bool = True,
        include_assistant: bool = True,
        important_assistant_only: bool = False,
        assistant_must_be_important: bool = False,
    ) -> list[str]:
        fragments: list[str] = []
        for turn in list(rows or []):
            role = str(turn.role or "").strip().lower()
            if role == "user" and not include_user:
                continue
            if role == "assistant":
                if not include_assistant:
                    continue
                if (important_assistant_only or assistant_must_be_important) and not self._is_important_assistant_turn(turn):
                    continue
            if role != "assistant" and important_assistant_only:
                continue
            for raw_fragment in self._split_summary_fragments(str(turn.text or "").strip()):
                fragment = self._clean_summary_fragment(raw_fragment)
                if not self._is_meaningful_summary_fragment(fragment):
                    continue
                fragments.append(fragment)
        return self._dedupe_fragments(fragments)

    def _dedupe_fragments(self, fragments: list[str]) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for value in list(fragments or []):
            item = self._clean_summary_fragment(value)
            if not self._is_meaningful_summary_fragment(item):
                continue
            key = normalize_text(item)
            if not key or key in seen:
                continue
            seen.add(key)
            out.append(item)
        return out

    def _join_episode_fragments(self, fragments: list[str], *, maximum: int) -> str:
        selected: list[str] = []
        for value in list(fragments or []):
            candidate = ". ".join([*selected, value]).strip()
            if selected and len(candidate) > maximum:
                break
            selected.append(value)
        return self._trim_summary_text(". ".join(selected).strip(), maximum=maximum)

    def _is_noise_turn(self, turn: DialogTurn) -> bool:
        text = self._clean_summary_fragment(str(turn.text or "").strip())
        if not text:
            return True
        normalized = normalize_text(text)
        if not normalized:
            return True
        if _SUMMARY_NOISE_RE.match(normalized):
            return True
        compact = re.sub(r"[^a-zа-яёіїєґ0-9_]+", " ", normalized, flags=re.I).strip()
        if compact and all(token in _BAD_TOPIC_TOKENS for token in compact.split()):
            return True
        return not any(
            self._is_meaningful_summary_fragment(self._clean_summary_fragment(raw_fragment))
            for raw_fragment in self._split_summary_fragments(text)
        )

    def _is_important_assistant_turn(self, turn: DialogTurn) -> bool:
        text = str(turn.text or "").strip()
        if not text:
            return False
        if _DECISION_RE.search(text) or _QUESTIONISH_RE.search(text):
            return True
        fragment = self._clean_summary_fragment(text)
        normalized = normalize_text(fragment)
        words = [token for token in _WORD_RE.findall(normalized) if token]
        if len(words) >= 6:
            return True
        return len(fragment) >= 44 and len(words) >= 4

    def _summary_short(
        self,
        *,
        turns: list[DialogTurn],
        topic: str,
        decisions: list[str],
        open_questions: list[str],
    ) -> str:
        base = self._summary_seed_text(
            turns,
            decisions=decisions,
            open_questions=open_questions,
        )
        if not base:
            base = f"Discussed {topic}." if topic else "Discussed the conversation."
        return base.strip()

    def _summary_seed_text(
        self,
        turns: list[DialogTurn],
        *,
        decisions: list[str] | None = None,
        open_questions: list[str] | None = None,
    ) -> str:
        summary_rows = self._summary_source_turns(turns)
        source_rows = summary_rows or list(turns or [])
        compact_source = self._episode_compact_text(source_rows)
        user_source = self._episode_user_text(source_rows)
        full_source = self._episode_text(source_rows)
        source_text = compact_source or user_source or full_source
        fragments = self._dedupe_fragments(
            [
                self._clean_summary_fragment(raw_fragment)
                for raw_fragment in self._split_summary_fragments(source_text)
            ]
        )

        if fragments:
            selected: list[str] = []
            seen: set[str] = set()

            def _try_add(value: str) -> bool:
                item = self._clean_summary_fragment(value)
                if not self._is_meaningful_summary_fragment(item):
                    return False
                key = normalize_text(item)
                if not key or key in seen:
                    return False
                candidate = ". ".join([*selected, item]).strip()
                if selected and len(candidate) > 145:
                    return False
                selected.append(item)
                seen.add(key)
                return True

            _try_add(fragments[0])
            priority_fragments = [
                self._clean_summary_fragment(str((decisions or [""])[0] or "").strip()),
                self._clean_summary_fragment(str((open_questions or [""])[0] or "").strip()),
            ]
            for item in list(fragments[1:4]):
                priority_fragments.append(item)
            for item in priority_fragments:
                if not item:
                    continue
                _try_add(item)
                if len(". ".join(selected).strip()) >= 110:
                    break
            if selected:
                return self._trim_summary_text(". ".join(selected).strip())

        fallback = self._clean_summary_fragment(full_source)
        return self._trim_summary_text(fallback)

    def _split_summary_fragments(self, text: str) -> list[str]:
        return [
            str(chunk or "").strip()
            for chunk in _SUMMARY_SPLIT_RE.split(str(text or ""))
            if str(chunk or "").strip()
        ]

    def _clean_summary_fragment(self, text: str) -> str:
        value = str(text or "").strip()
        value = _SUMMARY_DISCOURSE_PREFIX_RE.sub("", value).strip()
        value = re.sub(r"\s+", " ", value).strip(" \t\r\n-,:;.!?")
        return value

    def _is_meaningful_summary_fragment(self, text: str) -> bool:
        value = str(text or "").strip()
        if not value:
            return False
        normalized = normalize_text(value)
        if not normalized:
            return False
        if _SUMMARY_NOISE_RE.match(normalized):
            return False
        words = [token for token in _WORD_RE.findall(normalized) if token]
        if len(words) >= 4:
            return True
        if len(value) >= 28 and len(words) >= 3:
            return True
        if _DECISION_RE.search(value) or _QUESTIONISH_RE.search(value):
            return True
        return False

    def _trim_summary_text(self, text: str, *, minimum: int = 100, maximum: int = 150) -> str:
        value = re.sub(r"\s+", " ", str(text or "")).strip(" \t\r\n")
        if not value:
            return ""
        if len(value) > maximum:
            trimmed = value[:maximum].rstrip(" ,;:-")
            if len(trimmed) > minimum:
                parts = trimmed.rsplit(" ", 1)
                if len(parts) == 2 and len(parts[0]) >= minimum:
                    trimmed = parts[0]
            value = trimmed.rstrip(" ,;:-")
        if value and value[-1] not in ".!?":
            value += "."
        return value

    def _trim_reasoning_text(self, text: str, *, minimum: int = 140, maximum: int = 280) -> str:
        value = re.sub(r"\s+", " ", str(text or "")).strip(" \t\r\n")
        if not value:
            return ""
        if len(value) > maximum:
            trimmed = value[:maximum].rstrip(" ,;:-")
            if len(trimmed) > minimum:
                parts = trimmed.rsplit(" ", 1)
                if len(parts) == 2 and len(parts[0]) >= minimum:
                    trimmed = parts[0]
            value = trimmed.rstrip(" ,;:-")
        if value and value[-1] not in ".!?":
            value += "."
        return value

    def _summary_reasoning(
        self,
        *,
        turns: list[DialogTurn],
        topic: str,
        decisions: list[str],
        open_questions: list[str],
    ) -> str:
        discussed = self._summary_seed_text(turns).rstrip(".")
        parts: list[str] = []
        if discussed:
            parts.append(f"Discussed: {discussed}.")
        elif topic:
            parts.append(f"Discussed: {topic}.")
        if decisions:
            decision_text = "; ".join(
                self._clean_summary_fragment(item)
                for item in list(decisions or [])[:2]
                if self._clean_summary_fragment(item)
            )
            if decision_text:
                parts.append(f"Decided: {decision_text}.")
        if open_questions:
            open_text = "; ".join(
                self._clean_summary_fragment(item)
                for item in list(open_questions or [])[:2]
                if self._clean_summary_fragment(item)
            )
            if open_text:
                parts.append(f"Open questions: {open_text}.")
        if not decisions and not open_questions:
            parts.append("Context clarified next steps.")
        return self._trim_reasoning_text(" ".join(part.strip() for part in parts if part.strip()))

    def _salience(self, turns: list[DialogTurn], *, decisions: list[str], open_questions: list[str]) -> float:
        turn_score = min(0.36, 0.08 * len(list(turns or [])))
        decision_score = min(0.28, 0.10 * len(list(decisions or [])))
        question_score = min(0.20, 0.06 * len(list(open_questions or [])))
        role_bonus = 0.08 if len(self._participants(turns)) > 1 else 0.0
        return max(0.0, min(1.0, 0.24 + turn_score + decision_score + question_score + role_bonus))
