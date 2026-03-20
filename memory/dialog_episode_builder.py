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
    r"\b(?:решили|договорились|будем|сделаем|следующим\s+шагом|agreed|decided|next\s+step|we\s+will|let'?s)\b"
    r"|"
    r"\b(?:сначала|сперва|first)\b.{0,120}\b(?:потом|затем|then)\b"
    r")",
    re.I,
)
_QUESTIONISH_RE = re.compile(
    r"(?:\?|(?:\b(?:непонятно|не\s+ясно|что\s+дальше|осталось\s+понять|open\s+question|todo)\b))",
    re.I,
)
_REASON_SPLIT_RE = re.compile(
    r"\b(?:because|since|so\s+that|потому\s+что|так\s+как|чтобы)\b",
    re.I,
)
_REASON_HINT_RE = re.compile(
    r"(?:"
    r"\bpollut(?:e|es|ed|ing)\s+retrieval\b|"
    r"\bnois(?:e|y)\b|"
    r"\bself[- ]loops?\b|"
    r"\bclean\s+retrieval\b|"
    r"\bstale\s+evidence\s+bleed\b|"
    r"\bkeep\s+one\s+active\b|"
    r"\bactive\s+canonical\b|"
    r"\bcanonical\b|"
    r"\bisolat(?:e|ed|ion)\b|"
    r"\bavoid\b|"
    r"\bprevent\b|"
    r"\bleak(?:s|ed|ing)?\b|"
    r"\bbleed\b|"
    r"\bdrift\b|"
    r"\bclean\b|"
    r"\bactive\s+fact\b|"
    r"\bactive\s+value\b|"
    r"\bcanonical\s+value\b|"
    r"шум\w*|"
    r"изолир\w*|"
    r"канонич\w*|"
    r"избеж\w*|"
    r"загрязн\w*|"
    r"самоцикл\w*|"
    r"ретрив\w*"
    r")",
    re.I,
)
_REASON_PREFIX_RE = re.compile(
    r"^(?:because|since|so\s+that|right|exactly|yes|потому\s+что|так\s+как|чтобы|да|именно\s+поэтому)[\s,:-]*",
    re.I,
)
_WORD_RE = re.compile(r"[a-zа-яёіїєґ0-9_]{3,}", re.I)
_SUMMARY_SPLIT_RE = re.compile(r"[\n\r]+|(?<=[.!?])\s+")
_SUMMARY_DISCOURSE_PREFIX_RE = re.compile(
    r"^(?:ну|ладно|ок(?:ей)?|хорошо|так|слушай|смотри|короче|в общем|итак|значит)\b[\s,:;-]*",
    re.I,
)
_SUMMARY_NOISE_RE = re.compile(
    r"^(?:привет|здравствуй(?:те)?|добрый\s+(?:день|вечер|утро)|hello|hi|ага|угу|ок(?:ей)?|да|ясно|понятно|понял|поняла|спасибо|thanks?)$",
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
    "into",
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

        semantic_rows = self._semantic_source_turns(rows)
        summary_rows = semantic_rows or rows
        semantic_entity_keys = self._entity_keys(semantic_rows) if semantic_rows else []
        decisions = self._extract_decisions(rows)
        open_questions = self._extract_open_questions(rows)
        topic = self._episode_topic(
            semantic_rows,
            decisions=decisions,
            open_questions=open_questions,
        )
        participants = self._participants(rows)
        entity_keys = self._entity_keys(rows)
        summary_seed = self._summary_seed_text(summary_rows, maximum=180)
        summary_short = self._summary_short(
            turns=summary_rows,
            decisions=decisions,
            open_questions=open_questions,
        )
        summary_reasoning = self._summary_reasoning(
            turns=rows,
            decisions=decisions,
            open_questions=open_questions,
        )
        focus_keys = self._focus_keys(
            semantic_rows,
            entity_keys=semantic_entity_keys,
            summary_seed=summary_seed,
            decisions=decisions,
            open_questions=open_questions,
        )
        topic_keys = self._topic_keys(
            semantic_rows,
            topic=topic,
            summary_short=summary_short,
            summary_seed=summary_seed,
            decisions=decisions,
            open_questions=open_questions,
            focus_keys=focus_keys,
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
            focus_keys=focus_keys,
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

    def _topic_keys(
        self,
        turns: list[DialogTurn],
        *,
        topic: str,
        summary_short: str = "",
        summary_seed: str = "",
        decisions: list[str] | None = None,
        open_questions: list[str] | None = None,
        focus_keys: list[str] | None = None,
    ) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []

        def _add(value: str) -> None:
            token = str(value or "").strip().lower()
            if not token or token in seen:
                return
            seen.add(token)
            out.append(token)

        _add(topic)
        for token in list(focus_keys or []):
            _add(str(token or "").strip().lower())
        for turn in list(turns or []):
            _add(self._turn_topic(turn))
            for tag in list(turn.tags or []):
                raw = str(tag or "").strip().lower()
                match = _TOPIC_TAG_RE.match(raw)
                if match:
                    _add(str(match.group(1) or "").strip().lower())
        for token in self._semantic_topic_tokens(
            [
                str(summary_seed or "").strip(),
                str(summary_short or "").strip(),
                *[str(x).strip() for x in list(decisions or []) if str(x).strip()],
                *[str(x).strip() for x in list(open_questions or []) if str(x).strip()],
            ],
            limit=10,
        ):
            _add(token)
        return out

    def _semantic_topic_tokens(
        self,
        texts: list[str],
        *,
        limit: int = 8,
        extra_stopwords: set[str] | None = None,
    ) -> list[str]:
        extra_stopwords = {
            "build",
            "context",
            "decide",
            "decided",
            "doing",
            "discussion",
            "discuss",
            "discussed",
            "episode",
            "finish",
            "keep",
            "make",
            "need",
            "next",
            "open",
            "plan",
            "question",
            "questions",
            "return",
            "separate",
            "should",
            "split",
            "step",
            "steps",
            "store",
            "stabilize",
            "then",
            "they",
            "using",
            "want",
            "were",
            "what",
            "when",
            "which",
            "will",
            "with",
            "would",
            "your",
        }.union(set(extra_stopwords or set()))
        seen: set[str] = set()
        out: list[str] = []
        for text in list(texts or []):
            normalized = normalize_text(str(text or "")).lower()
            if not normalized:
                continue
            for token in _WORD_RE.findall(normalized):
                item = str(token or "").strip().lower()
                if (
                    not item
                    or len(item) < 4
                    or item.isdigit()
                    or item in seen
                    or item in _STOPWORDS
                    or item in extra_stopwords
                    or item in _BAD_TOPIC_TOKENS
                ):
                    continue
                seen.add(item)
                out.append(item)
                if len(out) >= max(1, int(limit)):
                    return out
        return out

    def _focus_keys(
        self,
        turns: list[DialogTurn],
        *,
        entity_keys: list[str] | None = None,
        summary_seed: str = "",
        decisions: list[str] | None = None,
        open_questions: list[str] | None = None,
    ) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []

        def _add(value: str) -> None:
            token = str(value or "").strip().lower()
            if not token or token in seen:
                return
            seen.add(token)
            out.append(token)

        def _add_many(values: list[str]) -> None:
            for value in list(values or []):
                _add(str(value or "").strip().lower())

        for turn in list(turns or []):
            meta = dict(turn.metadata or {})
            for row in list(meta.get("memory_entities") or []):
                if isinstance(row, dict):
                    _add_many(self._focus_keys_from_entity_row(row))
            for row in list(meta.get("stable_facts") or []):
                if isinstance(row, dict):
                    _add_many(self._focus_keys_from_fact_row(row))
            fact_row = meta.get("fact")
            if isinstance(fact_row, dict):
                _add_many(self._focus_keys_from_fact_row(fact_row))

        _add_many(
            self._semantic_topic_tokens(
                [
                    str(summary_seed or "").strip(),
                    *[str(x).strip() for x in list(decisions or []) if str(x).strip()],
                    *[str(x).strip() for x in list(open_questions or []) if str(x).strip()],
                ],
                limit=16,
            )
        )

        for turn in list(turns or []):
            meta = dict(turn.metadata or {})
            claim_rows = [
                *[row for row in list(meta.get("claims") or []) if isinstance(row, dict)],
                *[row for row in list(meta.get("claim_candidates") or []) if isinstance(row, dict)],
            ]
            claim_row = meta.get("claim")
            if isinstance(claim_row, dict):
                claim_rows.append(claim_row)
            for row in claim_rows:
                _add_many(self._focus_keys_from_claim_row(row))

        for token in list(entity_keys or []):
            _add_many(self._semantic_topic_tokens([str(token or "").replace("_", " ")], limit=3))
        return out[:16]

    def _focus_keys_from_entity_row(self, row: dict[str, Any]) -> list[str]:
        entity_type = normalize_text(str(row.get("type") or "")).strip().lower()
        base_keys = {
            "cpu_model": ["hardware"],
            "gpu_model": ["hardware"],
            "llm_model": ["environment"],
            "os_name": ["environment"],
            "person_name": ["identity"],
            "project_name": ["project"],
            "python_version": ["python", "environment"],
            "ram_size": ["hardware"],
            "tool_name": ["environment"],
            "vram_size": ["hardware"],
        }.get(entity_type, [])
        surface = str(row.get("canonical") or row.get("surface") or "").strip()
        return [
            *base_keys,
            *self._semantic_topic_tokens([surface], limit=3),
        ]

    def _focus_keys_from_fact_row(self, row: dict[str, Any]) -> list[str]:
        relation = str(row.get("relation") or "").strip().lower()
        predicate = str(row.get("predicate") or "").strip()
        value = str(row.get("value") or "").strip()
        return [
            *([relation] if relation else []),
            *self._semantic_topic_tokens(
                [predicate.replace("_", " ").replace(".", " ")],
                limit=4,
                extra_stopwords={"model", "name", "runtime", "size", "value", "version"},
            ),
            *self._semantic_topic_tokens([value], limit=2),
        ]

    def _focus_keys_from_claim_row(self, row: dict[str, Any]) -> list[str]:
        predicate = str(row.get("predicate") or "").strip()
        object_type = str(row.get("object_type") or "").strip().lower()
        object_value = str(
            row.get("normalized_object")
            or row.get("obj")
            or row.get("object_surface")
            or ""
        ).strip()
        return [
            *self._semantic_topic_tokens([predicate.replace("_", " ").replace(".", " ")], limit=3),
            *[str(x).strip().lower() for x in list(row.get("topic_keys") or []) if str(x).strip()],
            *([object_type] if object_type else []),
            *self._semantic_topic_tokens([object_value], limit=2),
        ]

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

    def _is_noise_determiner_turn(self, turn: DialogTurn) -> bool:
        text = self._clean_summary_fragment(str(turn.text or "").strip())
        if not text:
            return True
        normalized = normalize_text(text)
        if not normalized:
            return True
        if _SUMMARY_NOISE_RE.match(normalized):
            return True
        tokens = [
            token
            for token in re.sub(r"[^a-zа-яёіїєґ0-9_]+", " ", normalized, flags=re.I).split()
            if token
        ]
        if tokens and len(tokens) <= 3 and all(token in _BAD_TOPIC_TOKENS for token in tokens):
            return True
        return False

    def _semantic_source_turns(self, rows: list[DialogTurn]) -> list[DialogTurn]:
        return [
            turn
            for turn in list(rows or [])
            if str(turn.text or "").strip() and not self._is_noise_determiner_turn(turn)
        ]

    def _episode_text_turns(self, rows: list[DialogTurn]) -> list[DialogTurn]:
        out: list[DialogTurn] = []
        for turn in self._semantic_source_turns(rows):
            role = str(turn.role or "").strip().lower()
            if role == "assistant" and not self._is_important_assistant_turn(turn):
                continue
            if role in {"user", "assistant"}:
                out.append(turn)
        return out

    def episode_text(self, rows: list[DialogTurn], *, maximum: int = 520) -> str:
        source_rows = self._episode_text_turns(rows)
        if not source_rows:
            return ""
        return self._join_episode_fragments(
            self._episode_fragments(source_rows),
            maximum=maximum,
        )

    def _episode_text(self, rows: list[DialogTurn]) -> str:
        return self.episode_text(rows)

    def _episode_user_text(self, rows: list[DialogTurn]) -> str:
        source_rows = self._semantic_source_turns(rows) or list(rows or [])
        return self._join_episode_fragments(
            self._episode_fragments(
                source_rows,
                include_assistant=False,
            ),
            maximum=420,
        )

    def _episode_compact_text(self, rows: list[DialogTurn]) -> str:
        summary_rows = self._summary_source_turns(rows)
        source_rows = summary_rows or self._semantic_source_turns(rows) or list(rows or [])
        compact_fragments = self._episode_fragments(
            source_rows,
            assistant_must_be_important=True,
        )
        if not compact_fragments:
            compact_fragments = self._episode_fragments(source_rows)
        return self._join_episode_fragments(compact_fragments, maximum=220)

    def _summary_source_turns(self, rows: list[DialogTurn], *, maximum: int = 4) -> list[DialogTurn]:
        source_rows = self._semantic_source_turns(rows)
        meaningful: list[DialogTurn] = []
        for turn in source_rows:
            role = str(turn.role or "").strip().lower()
            if role == "assistant" and not self._is_important_assistant_turn(turn):
                continue
            meaningful.append(turn)
            if len(meaningful) >= maximum:
                break
        if meaningful:
            return meaningful
        return source_rows[:maximum]

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
        decisions: list[str],
        open_questions: list[str],
    ) -> str:
        lang = self._episode_language(turns)
        structured = self._structured_summary_short(
            turns,
            decisions=decisions,
            open_questions=open_questions,
            lang=lang,
        )
        if structured:
            return structured
        open_sentence = self._open_question_summary_sentence(open_questions=open_questions, lang=lang)
        base = self._summary_seed_text(
            turns,
            maximum=90 if open_sentence and not decisions else 120,
        )
        if base:
            prefix = "Обсудили: " if lang == "ru" else "Discussed: "
            context_sentence = self._trim_summary_text(
                prefix + base,
                maximum=90 if open_sentence and not decisions else 120,
            )
            if open_sentence and not decisions:
                return self._trim_summary_text(f"{context_sentence} {open_sentence}")
            return context_sentence
        if open_sentence and not decisions:
            return self._trim_summary_text(open_sentence)
        return "Обсудили рабочий эпизод." if lang == "ru" else "Discussed a work episode."

    def _structured_summary_short(
        self,
        turns: list[DialogTurn],
        *,
        decisions: list[str],
        open_questions: list[str],
        lang: str,
    ) -> str:
        catalog = self._summary_anchor_catalog(turns)
        summary_seed = self._summary_seed_text(turns, maximum=180)
        summary_text = normalize_text(
            " ".join(
                [
                    str(summary_seed or "").strip(),
                    *[str(x).strip() for x in list(decisions or []) if str(x).strip()],
                    *[str(x).strip() for x in list(open_questions or []) if str(x).strip()],
                ]
            )
        ).lower()
        signal_tokens = {
            str(x).strip().lower()
            for x in self._semantic_topic_tokens([summary_text], limit=18)
            if str(x).strip()
        }

        anchor_sentence = self._structured_anchor_sentence(
            catalog=catalog,
            signal_tokens=signal_tokens,
            lang=lang,
        )
        decision_sentence = self._decision_summary_sentence(decisions=decisions, lang=lang)
        open_sentence = self._open_question_summary_sentence(open_questions=open_questions, lang=lang)
        if decision_sentence and anchor_sentence:
            merged = f"{decision_sentence} {anchor_sentence}".strip()
            if len(merged) <= 150:
                return self._trim_summary_text(merged)
            return self._trim_summary_text(decision_sentence)
        if decision_sentence:
            return self._trim_summary_text(decision_sentence)
        if open_sentence:
            context_sentence = self._summary_context_sentence(
                anchor_sentence=anchor_sentence,
                summary_seed=summary_seed,
                lang=lang,
            )
            if context_sentence:
                return self._trim_summary_text(f"{context_sentence} {open_sentence}")
            return self._trim_summary_text(open_sentence)
        if anchor_sentence:
            return self._trim_summary_text(anchor_sentence)
        return ""

    def _summary_context_sentence(self, *, anchor_sentence: str, summary_seed: str, lang: str) -> str:
        if anchor_sentence:
            return self._trim_summary_text(anchor_sentence, maximum=90)
        seed = str(summary_seed or "").rstrip(".!?")
        if not seed:
            return ""
        prefix = "Обсудили: " if lang == "ru" else "Discussed: "
        return self._trim_summary_text(prefix + seed, maximum=90)

    def _structured_anchor_sentence(
        self,
        *,
        catalog: dict[str, list[str]],
        signal_tokens: set[str],
        lang: str,
    ) -> str:
        memory_items = self._memory_anchor_items(catalog=catalog, signal_tokens=signal_tokens)
        if len(memory_items) >= 2:
            subject = "Обсудили архитектуру памяти" if lang == "ru" else "Discussed memory architecture"
            return f"{subject}: {self._join_anchor_items(memory_items, lang=lang)}."

        environment_items = list(catalog.get("environment_labels") or [])
        if environment_items:
            subject = "Обсудили окружение пользователя" if lang == "ru" else "Discussed the user's environment"
            return f"{subject}: {self._join_anchor_items(environment_items[:3], lang=lang)}."

        preference_items = list(catalog.get("claim_objects") or [])
        preference_topics = {str(x).strip().lower() for x in list(catalog.get("claim_topics") or []) if str(x).strip()}
        if preference_items and preference_topics.intersection({"preference", "relationship"}):
            if lang == "ru":
                if {"preference", "relationship"}.issubset(preference_topics):
                    subject = "Обсудили предпочтения и отношения"
                elif "relationship" in preference_topics:
                    subject = "Обсудили отношения"
                else:
                    subject = "Обсудили предпочтения пользователя"
            else:
                if {"preference", "relationship"}.issubset(preference_topics):
                    subject = "Discussed preferences and relationships"
                elif "relationship" in preference_topics:
                    subject = "Discussed relationships"
                else:
                    subject = "Discussed user preferences"
            return f"{subject}: {self._join_anchor_items(preference_items[:3], lang=lang)}."

        generic_items = self._generic_anchor_items(catalog=catalog, signal_tokens=signal_tokens)
        if generic_items:
            subject = "Обсудили ключевые детали" if lang == "ru" else "Discussed the key details"
            return f"{subject}: {self._join_anchor_items(generic_items[:3], lang=lang)}."
        return ""

    def _summary_anchor_catalog(self, turns: list[DialogTurn]) -> dict[str, list[str]]:
        entity_labels: list[str] = []
        environment_labels: list[str] = []
        claim_objects: list[str] = []
        claim_topics: list[str] = []
        fact_values: list[str] = []
        fact_relations: list[str] = []
        topic_labels: list[str] = []

        def _push(target: list[str], value: str) -> None:
            item = self._summary_anchor_label(value)
            if not item or item in target:
                return
            target.append(item)

        for turn in list(turns or []):
            meta = dict(turn.metadata or {})
            topic_value = self._turn_topic(turn)
            if topic_value and topic_value not in topic_labels:
                topic_labels.append(topic_value)
            for row in list(meta.get("memory_entities") or []):
                if not isinstance(row, dict):
                    continue
                label = str(row.get("canonical") or row.get("surface") or "").strip()
                entity_type = normalize_text(str(row.get("type") or "")).strip().lower()
                _push(entity_labels, label)
                if entity_type in {"python_version", "os_name", "tool_name", "llm_model", "gpu_model", "cpu_model", "ram_size", "vram_size"}:
                    _push(environment_labels, label)

            fact_rows = [row for row in list(meta.get("stable_facts") or []) if isinstance(row, dict)]
            fact_row = meta.get("fact")
            if isinstance(fact_row, dict):
                fact_rows.append(fact_row)
            for row in fact_rows:
                relation = str(row.get("relation") or "").strip().lower()
                if relation:
                    _push(fact_relations, relation)
                value = str(row.get("value") or "").strip()
                if value:
                    _push(fact_values, value)
                    if relation == "environment":
                        _push(environment_labels, value)

            claim_rows = [row for row in list(meta.get("claims") or []) if isinstance(row, dict)]
            claim_row = meta.get("claim")
            if isinstance(claim_row, dict):
                claim_rows.append(claim_row)
            for row in claim_rows:
                obj = str(row.get("object_surface") or row.get("obj") or row.get("normalized_object") or "").strip()
                if obj:
                    _push(claim_objects, obj)
                for token in list(row.get("topic_keys") or []):
                    value = str(token or "").strip().lower()
                    if value and value not in claim_topics:
                        claim_topics.append(value)

        return {
            "entity_labels": entity_labels,
            "environment_labels": environment_labels,
            "claim_objects": claim_objects,
            "claim_topics": claim_topics,
            "fact_values": fact_values,
            "fact_relations": fact_relations,
            "topic_labels": topic_labels,
        }

    def _memory_anchor_items(self, *, catalog: dict[str, list[str]], signal_tokens: set[str]) -> list[str]:
        normalized_text = " ".join(sorted(signal_tokens))
        topic_labels = {str(x).strip().lower() for x in list(catalog.get("topic_labels") or []) if str(x).strip()}
        items: list[str] = []

        def _has_token(*markers: str) -> bool:
            for marker in markers:
                marker_value = str(marker or "").strip().lower()
                if not marker_value:
                    continue
                if marker_value in signal_tokens or marker_value in normalized_text:
                    return True
                if any(str(token).startswith(marker_value) for token in signal_tokens):
                    return True
            return False

        def _has_memory_context() -> bool:
            if "memory" in topic_labels:
                return True
            if "memory" in signal_tokens:
                return True
            return any(token.startswith("памят") for token in signal_tokens)

        if _has_token("facts", "fact", "факт", "факты"):
            items.append("facts")
        if _has_token("claims", "claim", "клейм"):
            items.append("claims")
        if (_has_token("dialog") and _has_token("episodes", "episode")) or (_has_token("диалог") and _has_token("эпизод")):
            items.append("dialog episodes")
        elif _has_token("episodes", "episode", "эпизод", "эпизоды"):
            items.append("episodes")
        if _has_token("document") and (_has_token("memory") or _has_memory_context()):
            items.append("document memory")
        if not _has_memory_context() and "memory" not in {x.lower() for x in list(catalog.get("fact_relations") or [])}:
            return []
        out: list[str] = []
        for item in items:
            if item not in out:
                out.append(item)
            if len(out) >= 3:
                break
        return out

    def _generic_anchor_items(self, *, catalog: dict[str, list[str]], signal_tokens: set[str]) -> list[str]:
        items: list[str] = []
        for value in [
            *list(catalog.get("entity_labels") or []),
            *list(catalog.get("claim_objects") or []),
            *list(catalog.get("fact_values") or []),
        ]:
            label = self._summary_anchor_label(value)
            if not label or label in items:
                continue
            items.append(label)
            if len(items) >= 3:
                break
        if items:
            return items
        return [
            str(x).strip()
            for x in list(signal_tokens or [])
            if str(x).strip()
        ][:3]

    def _summary_anchor_label(self, value: str) -> str:
        item = self._clean_summary_fragment(str(value or "").replace("_", " ").replace(".", " "))
        if not item:
            return ""
        words = [token for token in re.split(r"\s+", item) if token]
        return " ".join(words[:4]).strip()

    def _decision_summary_sentence(self, *, decisions: list[str], lang: str) -> str:
        if not decisions:
            return ""
        ranked = self._ordered_decisions(decisions)
        if not ranked:
            return ""
        value = self._decision_summary_fragment(str(ranked[0] or "").strip())
        if not value:
            return ""
        if lang == "ru":
            return f"Решили: {value}."
        return f"Decision: {value}."

    def _decision_summary_fragment(self, text: str) -> str:
        value = self._clean_summary_fragment(text)
        if not value:
            return ""
        value = re.sub(
            r"^(?:we\s+decided(?:\s+to)?|decided(?:\s+to)?|agreed(?:\s+to)?|we\s+agreed(?:\s+to)?|решили|договорились|следующим\s+шагом|будем|сделаем|we\s+will)\b[\s,:-]*",
            "",
            value,
            flags=re.I,
        ).strip()
        return value or self._clean_summary_fragment(text)

    def _ordered_decisions(self, decisions: list[str]) -> list[str]:
        scored: list[tuple[int, int, str]] = []
        for idx, item in enumerate(list(decisions or [])):
            value = str(item or "").strip()
            if not value:
                continue
            explicit = re.search(
                r"^(?:we\s+decided|decided|we\s+agreed|agreed|решили|договорились|we\s+will|будем|сделаем)\b",
                value,
                flags=re.I,
            )
            next_step = re.search(r"\b(?:next\s+step|следующим\s+шагом)\b", value, flags=re.I)
            proposal = re.search(r"^(?:let'?s|давай)\b", value, flags=re.I)
            sequence = re.search(r"\b(?:first|сначала)\b.{0,120}\b(?:then|потом|затем)\b", value, flags=re.I)
            if explicit:
                priority = 0
            elif next_step:
                priority = 1
            elif proposal:
                priority = 2
            elif sequence:
                priority = 3
            else:
                priority = 4
            scored.append((priority, idx, value))
        scored.sort(key=lambda item: (item[0], item[1]))
        return [value for _, _, value in scored]

    def _reasoning_decision_text(self, decisions: list[str]) -> str:
        fragments: list[str] = []
        seen: set[str] = set()
        for item in self._ordered_decisions(decisions)[:2]:
            fragment = self._decision_without_reason(str(item or ""))
            if not fragment:
                continue
            key = normalize_text(fragment)
            if not key or key in seen:
                continue
            seen.add(key)
            fragments.append(fragment)
        return "; ".join(fragments)

    def _reasoning_open_text(self, open_questions: list[str]) -> str:
        fragments: list[str] = []
        seen: set[str] = set()
        for item in list(open_questions or [])[:2]:
            fragment = self._clean_summary_fragment(str(item or ""))
            if not fragment:
                continue
            key = normalize_text(fragment)
            if not key or key in seen:
                continue
            seen.add(key)
            fragments.append(fragment)
        return "; ".join(fragments)

    def _open_question_summary_sentence(self, *, open_questions: list[str], lang: str) -> str:
        open_text = self._reasoning_open_text(open_questions)
        if not open_text:
            return ""
        if lang == "ru":
            return f"Открытым осталось: {open_text}."
        return f"Still open: {open_text}."

    def _summary_reason_fragment(self, *, turns: list[DialogTurn], decisions: list[str]) -> str:
        seen: set[str] = set()
        candidates: list[str] = []
        for text in [*list(decisions or []), *(str(turn.text or "") for turn in list(turns or []))]:
            candidate = self._reason_candidate_from_text(text)
            if not candidate:
                continue
            key = normalize_text(candidate)
            if not key or key in seen:
                continue
            seen.add(key)
            candidates.append(candidate)
        return candidates[0] if candidates else ""

    def _reason_candidate_from_text(self, text: str) -> str:
        value = self._clean_summary_fragment(text)
        if not value or _QUESTIONISH_RE.search(value):
            return ""
        explicit = self._explicit_reason_fragment(value)
        if explicit:
            return explicit
        if _DECISION_RE.search(value):
            return ""
        normalized = normalize_text(value)
        if not normalized or not _REASON_HINT_RE.search(normalized):
            return ""
        return self._clean_reason_fragment(value)

    def _explicit_reason_fragment(self, text: str) -> str:
        value = self._clean_summary_fragment(text)
        if not value:
            return ""
        match = _REASON_SPLIT_RE.search(value)
        if not match:
            return ""
        return self._clean_reason_fragment(value[match.end() :])

    def _decision_without_reason(self, text: str) -> str:
        value = self._decision_summary_fragment(text)
        if not value:
            return ""
        match = _REASON_SPLIT_RE.search(value)
        if not match:
            return value
        trimmed = value[: match.start()].rstrip(" ,;:-")
        return trimmed or value

    def _clean_reason_fragment(self, text: str) -> str:
        value = self._clean_summary_fragment(text)
        if not value:
            return ""
        value = _REASON_PREFIX_RE.sub("", value).strip()
        value = re.sub(r"^(?:that)\s+", "", value, flags=re.I)
        return value.strip(" \t\r\n,;:-")

    def _join_anchor_items(self, items: list[str], *, lang: str) -> str:
        values = [self._summary_anchor_label(x) for x in list(items or []) if self._summary_anchor_label(x)]
        if not values:
            return ""
        if len(values) == 1:
            return values[0]
        if len(values) == 2:
            joiner = " и " if lang == "ru" else " and "
            return f"{values[0]}{joiner}{values[1]}"
        joiner = " и " if lang == "ru" else " and "
        return f"{', '.join(values[:-1])}{joiner}{values[-1]}"

    def _episode_language(self, turns: list[DialogTurn]) -> str:
        cyrillic = 0
        latin = 0
        for turn in list(turns or []):
            text = str(turn.text or "")
            cyrillic += len(re.findall(r"[а-яёіїєґ]", text, flags=re.I))
            latin += len(re.findall(r"[a-z]", text, flags=re.I))
        return "ru" if cyrillic >= latin else "en"

    def _summary_seed_text(
        self,
        turns: list[DialogTurn],
        *,
        maximum: int = 150,
    ) -> str:
        fragments = self._summary_seed_fragments(turns)
        if fragments:
            return self._join_episode_fragments(fragments[:6], maximum=maximum)

        fallback = self._fallback_summary_text(turns, maximum=maximum)
        if fallback:
            return fallback
        return ""

    def _summary_seed_fragments(self, turns: list[DialogTurn]) -> list[str]:
        summary_rows = self._summary_source_turns(turns)
        row_groups: list[list[DialogTurn]] = []
        if summary_rows:
            row_groups.append(summary_rows)
        all_rows = list(turns or [])
        if all_rows and all_rows != summary_rows:
            row_groups.append(all_rows)

        fragments: list[str] = []
        for rows in row_groups:
            fragments.extend(
                self._episode_fragments(
                    rows,
                    assistant_must_be_important=True,
                )
            )
            fragments.extend(
                self._episode_fragments(
                    rows,
                    include_assistant=False,
                )
            )
            fragments.extend(self._episode_fragments(rows))
        return self._dedupe_fragments(fragments)

    def _fallback_summary_text(self, turns: list[DialogTurn], *, maximum: int) -> str:
        fragments: list[str] = []
        for turn in self._semantic_source_turns(turns):
            text = str(turn.text or "").strip()
            if not text:
                continue
            for raw_fragment in self._split_summary_fragments(text):
                fragment = self._clean_summary_fragment(raw_fragment)
                if not fragment:
                    continue
                normalized = normalize_text(fragment)
                if not normalized or _SUMMARY_NOISE_RE.match(normalized):
                    continue
                fragments.append(fragment)
        cleaned = self._dedupe_fragments(fragments)
        if not cleaned:
            return ""
        return self._join_episode_fragments(cleaned[:4], maximum=maximum)

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
        decisions: list[str],
        open_questions: list[str],
    ) -> str:
        lang = self._episode_language(turns)
        decision_text = self._reasoning_decision_text(decisions)
        reason_text = self._summary_reason_fragment(turns=turns, decisions=decisions)
        open_text = self._reasoning_open_text(open_questions)
        summary_seed = self._summary_seed_text(turns, maximum=190).rstrip(".!?")
        parts: list[str] = []
        if decision_text and reason_text:
            if lang == "ru":
                parts.append(f"Решили {decision_text}, потому что {reason_text}.")
            else:
                parts.append(f"We decided: {decision_text} because {reason_text}.")
        elif decision_text:
            if lang == "ru":
                parts.append(f"Решили {decision_text}.")
            else:
                parts.append(f"We decided: {decision_text}.")
        elif summary_seed:
            if lang == "ru":
                parts.append(f"Зафиксировали направление: {summary_seed}.")
            else:
                parts.append(f"Current direction: {summary_seed}.")
        if open_text:
            if lang == "ru":
                parts.append(f"Открытым осталось: {open_text}.")
            else:
                parts.append(f"Still open: {open_text}.")
        if not parts:
            if lang == "ru":
                parts.append("Зафиксировали эпизод диалога.")
            else:
                parts.append("Captured the dialog episode.")
        return self._trim_reasoning_text(" ".join(part.strip() for part in parts if part.strip()))

    def _salience(self, turns: list[DialogTurn], *, decisions: list[str], open_questions: list[str]) -> float:
        turn_score = min(0.36, 0.08 * len(list(turns or [])))
        decision_score = min(0.28, 0.10 * len(list(decisions or [])))
        question_score = min(0.20, 0.06 * len(list(open_questions or [])))
        role_bonus = 0.08 if len(self._participants(turns)) > 1 else 0.0
        return max(0.0, min(1.0, 0.24 + turn_score + decision_score + question_score + role_bonus))
