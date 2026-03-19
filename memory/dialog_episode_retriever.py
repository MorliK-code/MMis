from __future__ import annotations

import re
from dataclasses import dataclass

from memory.dialog_episode_models import DialogEpisode
from memory.memory_models import MemoryRecord, MemoryScope, MemoryStatus, MemoryType, RetrievalQuery
from memory.vector_store import VectorStore
from modules.nlu.normalizer import normalize_text


_DISCUSSION_QUERY_RE = re.compile(
    r"(?:что\s+мы\s+(?:обсуждали|говорили)|о\s+чем\s+мы\s+(?:обсуждали|говорили)|what\s+did\s+we\s+(?:discuss|talk about))",
    re.I,
)
_WHY_QUERY_RE = re.compile(r"(?:почему|зачем|why)\b", re.I)
_DECISION_QUERY_RE = re.compile(
    r"(?:что\s+решили|к\s+чему\s+пришли|what\s+did\s+we\s+decide|what\s+was\s+decided|decisions?)",
    re.I,
)
_OPEN_QUERY_RE = re.compile(
    r"(?:что\s+осталось|что\s+открыто|open\s+question|unresolved|что\s+не\s+закрыли)",
    re.I,
)
_PLAN_QUERY_RE = re.compile(
    r"(?:какой\s+(?:был|у\s+нас)\s+план|что\s+по\s+плану|next\s+steps?|plan\b)",
    re.I,
)
_QUERY_STOPWORDS = {
    "a",
    "about",
    "and",
    "did",
    "do",
    "for",
    "is",
    "of",
    "on",
    "the",
    "to",
    "we",
    "what",
    "why",
    "в",
    "к",
    "мы",
    "на",
    "о",
    "об",
    "по",
    "про",
    "что",
    "это",
}


@dataclass(frozen=True)
class DialogEpisodeQueryHints:
    query_text: str
    topic_keys: set[str]
    trigger_keys: set[str]
    reasoning_preferred: bool = False
    decisions_preferred: bool = False
    open_questions_preferred: bool = False
    discussion_recall_preferred: bool = False
    plan_preferred: bool = False


@dataclass(frozen=True)
class DialogSupportingTurn:
    turn_id: str
    role: str
    text: str
    ts: float = 0.0

    def to_dict(self) -> dict[str, object]:
        return {
            "turn_id": str(self.turn_id or ""),
            "role": str(self.role or ""),
            "text": str(self.text or ""),
            "ts": float(self.ts or 0.0),
        }


@dataclass(frozen=True)
class DialogEpisodeHit:
    record: MemoryRecord
    episode: DialogEpisode
    score: float
    summary_short: str
    summary_reasoning: str
    decisions: list[str]
    supporting_turns: list[DialogSupportingTurn]
    source: str = "episode_channel"

    def to_dict(self) -> dict[str, object]:
        return {
            "record_id": str(self.record.id or ""),
            "episode": self.episode.to_dict(),
            "score": float(self.score),
            "summary_short": str(self.summary_short or ""),
            "summary_reasoning": str(self.summary_reasoning or ""),
            "decisions": [str(x) for x in list(self.decisions or []) if str(x).strip()],
            "supporting_turns": [row.to_dict() for row in list(self.supporting_turns or [])],
            "source": str(self.source or "episode_channel"),
        }


@dataclass
class DialogEpisodeRetriever:
    store: VectorStore
    supporting_turn_limit: int = 4

    def retrieve(
        self,
        *,
        query: RetrievalQuery,
        query_text: str | None = None,
        namespace: str | None = None,
        scopes: list[MemoryScope] | None = None,
        top_k: int | None = None,
        filters: dict[str, object] | None = None,
    ) -> list[DialogEpisodeHit]:
        query_text_value = str(query_text or query.search_text or query.query_text or "").strip()
        namespace_value = str(namespace or query.namespace or "default")
        scope_values = list(scopes or query.scopes or [])
        top_k_value = max(1, int(top_k or query.top_k or 4))
        episode_filters = {
            **dict(filters or {}),
            "memory_type": [MemoryType.EPISODE.value],
        }

        semantic_hits = self.store.semantic_search(
            query_text=query_text_value,
            top_k=max(1, int(top_k_value * 2)),
            namespace=namespace_value,
            scopes=scope_values,
            include_stale=bool(query.include_stale),
            metadata_filters=episode_filters,
        )
        lexical_hits = self.store.lexical_search(
            query_text=query_text_value,
            top_k=max(1, int(top_k_value * 2)),
            namespace=namespace_value,
            scopes=scope_values,
            include_stale=bool(query.include_stale),
            metadata_filters=episode_filters,
        )

        hints = self.build_query_hints(query)
        anchor_hits = self._anchor_hits(
            namespace=namespace_value,
            scopes=scope_values,
            include_stale=bool(query.include_stale),
            hints=hints,
            top_k=max(1, int(top_k_value * 3)),
        )
        rows = self._merge_hits(
            semantic_hits=semantic_hits,
            lexical_hits=lexical_hits,
            anchor_hits=anchor_hits,
        )
        out: list[DialogEpisodeHit] = []
        for row in rows:
            record = row.get("record")
            if not isinstance(record, MemoryRecord):
                continue
            episode = self._episode_from_record(record)
            if episode is None:
                continue
            score = (
                max(float(row.get("semantic_score") or 0.0), float(row.get("lexical_score") or 0.0))
                + float(row.get("anchor_score") or 0.0)
                + self.score_bonus(
                    query=query,
                    record=record,
                    episode=episode,
                    source=str(row.get("source") or "episode_channel"),
                    hints=hints,
                )
                + min(0.08, max(0.0, float(episode.salience or 0.0)) * 0.08)
            )
            supporting_turns = self._supporting_turns(
                record=record,
                episode=episode,
                namespace=namespace_value,
                hints=hints,
                limit=max(2, int(self.supporting_turn_limit)),
            )
            out.append(
                DialogEpisodeHit(
                    record=record,
                    episode=episode,
                    score=score,
                    summary_short=str(episode.summary_short or ""),
                    summary_reasoning=str(episode.summary_reasoning or ""),
                    decisions=[str(x) for x in list(episode.decisions or []) if str(x).strip()],
                    supporting_turns=supporting_turns,
                    source=str(row.get("source") or "episode_channel"),
                )
            )

        out.sort(key=lambda row: float(row.score), reverse=True)
        return out[:top_k_value]

    def score_bonus(
        self,
        *,
        query: RetrievalQuery,
        record: MemoryRecord,
        episode: DialogEpisode,
        source: str,
        hints: DialogEpisodeQueryHints | None = None,
    ) -> float:
        _ = record
        hints = hints or self.build_query_hints(query)
        bonus = 0.08
        if str(source or "") in {"episode_channel", "episode_anchor_channel"}:
            bonus += 0.05
        if hints.discussion_recall_preferred:
            bonus += 0.04
        if hints.reasoning_preferred and str(episode.summary_reasoning or "").strip():
            bonus += 0.10
        if hints.decisions_preferred and list(episode.decisions or []):
            bonus += 0.12
        if hints.open_questions_preferred and list(episode.open_questions or []):
            bonus += 0.10
        if hints.plan_preferred and (list(episode.decisions or []) or list(episode.open_questions or [])):
            bonus += 0.12

        topic_overlap = len({str(x).strip().lower() for x in list(episode.topic_keys or []) if str(x).strip()}.intersection(hints.topic_keys))
        entity_overlap = len({str(x).strip().lower() for x in list(episode.entity_keys or []) if str(x).strip()}.intersection(hints.trigger_keys))
        text_overlap = self._fuzzy_overlap(self._episode_tokens(episode), hints.trigger_keys)

        if topic_overlap:
            bonus += min(0.14, 0.04 * topic_overlap)
        if entity_overlap:
            bonus += min(0.10, 0.03 * entity_overlap)
        if text_overlap:
            bonus += min(0.12, 0.03 * text_overlap)
        return min(0.48, max(0.0, bonus))

    def build_query_hints(self, query: RetrievalQuery) -> DialogEpisodeQueryHints:
        query_text = normalize_text(
            " ".join(
                part
                for part in (
                    str(query.query_text or "").strip(),
                    str(query.search_text or "").strip(),
                )
                if part
            )
        ).lower()
        topic_keys: set[str] = set()
        trigger_keys: set[str] = set()

        for raw in list(query.entity_keys or []):
            token = str(raw or "").strip().lower()
            if token:
                topic_keys.add(token)
                trigger_keys.add(token)

        for raw in re.split(r"\s+", query_text):
            token = str(raw or "").strip().lower()
            if not token or token in _QUERY_STOPWORDS or len(token) < 3:
                continue
            trigger_keys.add(token)
            topic_keys.add(token)

        return DialogEpisodeQueryHints(
            query_text=query_text,
            topic_keys=topic_keys,
            trigger_keys=trigger_keys,
            reasoning_preferred=bool(_WHY_QUERY_RE.search(query_text)),
            decisions_preferred=bool(_DECISION_QUERY_RE.search(query_text) or _PLAN_QUERY_RE.search(query_text)),
            open_questions_preferred=bool(_OPEN_QUERY_RE.search(query_text)),
            discussion_recall_preferred=bool(_DISCUSSION_QUERY_RE.search(query_text)),
            plan_preferred=bool(_PLAN_QUERY_RE.search(query_text)),
        )

    def _anchor_hits(
        self,
        *,
        namespace: str,
        scopes: list[MemoryScope],
        include_stale: bool,
        hints: DialogEpisodeQueryHints,
        top_k: int,
    ) -> list[tuple[MemoryRecord, float]]:
        if not hasattr(self.store, "iter_records"):
            return []

        allowed_scopes = {str(x.value) for x in list(scopes or []) if hasattr(x, "value")}
        out: list[tuple[MemoryRecord, float]] = []
        for record in list(self.store.iter_records(namespace=namespace) or []):
            if not isinstance(record, MemoryRecord):
                continue
            if record.memory_type != MemoryType.EPISODE:
                continue
            if record.scope == MemoryScope.PRIVATE_RUNTIME:
                continue
            if allowed_scopes and str(record.scope.value) not in allowed_scopes:
                continue
            if not include_stale and record.status in {MemoryStatus.STALE, MemoryStatus.ARCHIVED, MemoryStatus.DELETED}:
                continue
            episode = self._episode_from_record(record)
            if episode is None:
                continue
            anchor_score = self._anchor_match_score(episode=episode, hints=hints)
            if anchor_score <= 0.0:
                continue
            out.append((record, anchor_score))

        out.sort(key=lambda item: float(item[1]), reverse=True)
        return out[: max(1, int(top_k))]

    def _anchor_match_score(
        self,
        *,
        episode: DialogEpisode,
        hints: DialogEpisodeQueryHints,
    ) -> float:
        topic_tokens = {
            str(episode.topic or "").strip().lower(),
            *[str(x).strip().lower() for x in list(episode.topic_keys or []) if str(x).strip()],
        }
        entity_tokens = {str(x).strip().lower() for x in list(episode.entity_keys or []) if str(x).strip()}
        decision_tokens = self._text_tokens(list(episode.decisions or []))
        open_question_tokens = self._text_tokens(list(episode.open_questions or []))
        reasoning_tokens = self._text_tokens([episode.summary_reasoning])

        topic_overlap = self._fuzzy_overlap(topic_tokens, hints.topic_keys)
        entity_overlap = self._fuzzy_overlap(entity_tokens, hints.trigger_keys)
        decision_overlap = self._fuzzy_overlap(decision_tokens, hints.trigger_keys)
        open_overlap = self._fuzzy_overlap(open_question_tokens, hints.trigger_keys)
        reasoning_overlap = self._fuzzy_overlap(reasoning_tokens, hints.trigger_keys)

        score = 0.0
        if topic_overlap:
            score += min(0.16, 0.08 * topic_overlap)
        if entity_overlap:
            score += min(0.16, 0.08 * entity_overlap)
        if decision_overlap:
            score += min(0.18, 0.09 * decision_overlap)
        if open_overlap:
            score += min(0.16, 0.08 * open_overlap)
        if reasoning_overlap:
            score += min(0.12, 0.06 * reasoning_overlap)

        if hints.discussion_recall_preferred and (topic_overlap or entity_overlap or decision_overlap or open_overlap):
            score += 0.08
        if hints.reasoning_preferred and (reasoning_overlap or decision_overlap or open_overlap):
            score += 0.08
        if hints.decisions_preferred and list(episode.decisions or []):
            score += 0.10
        if hints.open_questions_preferred and list(episode.open_questions or []):
            score += 0.08
        if hints.plan_preferred and (list(episode.decisions or []) or list(episode.open_questions or [])):
            score += 0.10
        return min(0.42, max(0.0, score))

    def _supporting_turns(
        self,
        *,
        record: MemoryRecord,
        episode: DialogEpisode,
        namespace: str,
        hints: DialogEpisodeQueryHints,
        limit: int,
    ) -> list[DialogSupportingTurn]:
        turns = self._explicit_supporting_turns(record)
        if not turns:
            turns = self._resolve_supporting_turns_from_store(record=record, episode=episode, namespace=namespace)
        if not turns:
            return []
        return self._select_supporting_turns(turns=turns, hints=hints, limit=limit)

    def _explicit_supporting_turns(self, record: MemoryRecord) -> list[DialogSupportingTurn]:
        meta = dict(record.metadata or {})
        raw_rows = (
            list(meta.get("supporting_turns") or [])
            or list(meta.get("episode_supporting_turns") or [])
            or list(dict(meta.get("dialog_episode") or {}).get("supporting_turns") or [])
        )
        out: list[DialogSupportingTurn] = []
        for idx, row in enumerate(raw_rows):
            if isinstance(row, dict):
                text = str(row.get("text") or "").strip()
                if not text:
                    continue
                out.append(
                    DialogSupportingTurn(
                        turn_id=str(row.get("turn_id") or row.get("id") or f"{record.id}:support:{idx}"),
                        role=str(row.get("role") or "").strip().lower(),
                        text=text,
                        ts=float(row.get("ts") or 0.0),
                    )
                )
            else:
                text = str(row or "").strip()
                if not text:
                    continue
                out.append(
                    DialogSupportingTurn(
                        turn_id=f"{record.id}:support:{idx}",
                        role="",
                        text=text,
                        ts=0.0,
                    )
                )
        return out

    def _resolve_supporting_turns_from_store(
        self,
        *,
        record: MemoryRecord,
        episode: DialogEpisode,
        namespace: str,
    ) -> list[DialogSupportingTurn]:
        if not hasattr(self.store, "iter_records"):
            return []

        target_ids = [str(x).strip() for x in list(episode.turn_ids or []) if str(x).strip()]
        if not target_ids:
            return []

        target_set = set(target_ids)
        target_order = {value: index for index, value in enumerate(target_ids)}
        out: list[tuple[int, float, DialogSupportingTurn]] = []

        for row in list(self.store.iter_records(namespace=namespace) or []):
            if not isinstance(row, MemoryRecord):
                continue
            if row.id == record.id or row.memory_type not in {MemoryType.MESSAGE, MemoryType.SUMMARY, MemoryType.SEMANTIC}:
                continue
            meta = dict(row.metadata or {})
            aliases = {
                str(row.id or "").strip(),
                str(row.source_event_id or "").strip(),
                str(meta.get("turn_id") or "").strip(),
                str(meta.get("event_id") or "").strip(),
            }
            aliases = {value for value in aliases if value}
            matched = [value for value in aliases if value in target_set]
            if not matched:
                continue
            matched_id = sorted(matched, key=lambda item: target_order.get(item, 10_000))[0]
            out.append(
                (
                    target_order.get(matched_id, 10_000),
                    float(row.updated_at or row.created_at or 0.0),
                    DialogSupportingTurn(
                        turn_id=matched_id,
                        role=self._record_role(row),
                        text=str(row.text or "").strip(),
                        ts=float(row.updated_at or row.created_at or 0.0),
                    ),
                )
            )

        out.sort(key=lambda item: (item[0], item[1]))
        return [item[2] for item in out if str(item[2].text or "").strip()]

    def _select_supporting_turns(
        self,
        *,
        turns: list[DialogSupportingTurn],
        hints: DialogEpisodeQueryHints,
        limit: int,
    ) -> list[DialogSupportingTurn]:
        rows = [row for row in list(turns or []) if str(row.text or "").strip()]
        if len(rows) <= limit:
            return rows

        ranked: list[tuple[float, int, DialogSupportingTurn]] = []
        for idx, row in enumerate(rows):
            text_tokens = {part for part in re.split(r"\s+", normalize_text(str(row.text or "")).lower()) if part}
            overlap = len(text_tokens.intersection(hints.trigger_keys))
            score = float(overlap)
            if hints.reasoning_preferred and row.role == "assistant":
                score += 0.4
            if hints.decisions_preferred and _DECISION_QUERY_RE.search(str(row.text or "").lower()):
                score += 0.6
            if hints.open_questions_preferred and ("?" in str(row.text or "") or _OPEN_QUERY_RE.search(str(row.text or "").lower())):
                score += 0.6
            ranked.append((score, idx, row))

        ranked.sort(key=lambda item: (item[0], -item[1]), reverse=True)
        selected = sorted(ranked[:limit], key=lambda item: item[1])
        return [item[2] for item in selected]

    def _episode_from_record(self, record: MemoryRecord) -> DialogEpisode | None:
        meta = dict(record.metadata or {})
        payload = dict(meta.get("dialog_episode") or meta.get("episode") or {})
        if payload:
            try:
                return DialogEpisode.from_dict(payload)
            except Exception:
                pass

        topic = str(meta.get("topic") or "").strip()
        summary_short = str(meta.get("summary_short") or record.text or "").strip()
        summary_reasoning = str(meta.get("summary_reasoning") or "").strip()
        turn_ids = [str(x).strip() for x in list(meta.get("turn_ids") or meta.get("episode_turn_ids") or []) if str(x).strip()]
        decisions = [str(x).strip() for x in list(meta.get("decisions") or []) if str(x).strip()]
        open_questions = [str(x).strip() for x in list(meta.get("open_questions") or []) if str(x).strip()]
        participants = [str(x).strip() for x in list(meta.get("participants") or []) if str(x).strip()]
        topic_keys = [str(x).strip().lower() for x in list(meta.get("topic_keys") or []) if str(x).strip()]
        entity_keys = [str(x).strip().lower() for x in list(meta.get("entity_keys") or []) if str(x).strip()]
        if not any((topic, summary_short, summary_reasoning, turn_ids, decisions, open_questions)):
            return None

        return DialogEpisode(
            id=str(record.id or ""),
            topic=topic or "dialog_context",
            turn_ids=turn_ids,
            summary_short=summary_short or f"Discussed {topic or 'dialog_context'}.",
            summary_reasoning=summary_reasoning or "Turns captured a discussion episode.",
            decisions=decisions,
            open_questions=open_questions,
            participants=participants,
            salience=float(meta.get("salience") or record.importance or 0.5),
            topic_keys=topic_keys,
            entity_keys=entity_keys,
            created_at=float(record.created_at or 0.0),
            updated_at=float(record.updated_at or record.created_at or 0.0),
        )

    @staticmethod
    def _record_role(record: MemoryRecord) -> str:
        meta = dict(record.metadata or {})
        role = str(meta.get("role") or "").strip().lower()
        if role:
            return role
        source_kind = str(meta.get("source_kind") or "").strip().lower()
        if source_kind == "assistant_reply":
            return "assistant"
        if source_kind == "tool_result":
            return "tool"
        if source_kind == "system_decision":
            return "system"
        if source_kind == "user":
            return "user"
        return ""

    @staticmethod
    def _episode_tokens(episode: DialogEpisode) -> set[str]:
        tokens: set[str] = set()
        for text in [episode.topic, episode.summary_short, episode.summary_reasoning, *list(episode.decisions or []), *list(episode.open_questions or [])]:
            for part in re.split(r"\s+", normalize_text(str(text or "")).lower()):
                token = str(part or "").strip()
                if token and token not in _QUERY_STOPWORDS and len(token) >= 3:
                    tokens.add(token)
        for token in list(episode.topic_keys or []) + list(episode.entity_keys or []):
            value = str(token or "").strip().lower()
            if value:
                tokens.add(value)
        return tokens

    @staticmethod
    def _text_tokens(texts: list[str]) -> set[str]:
        tokens: set[str] = set()
        for text in list(texts or []):
            for part in re.split(r"\s+", normalize_text(str(text or "")).lower()):
                token = str(part or "").strip()
                if token and token not in _QUERY_STOPWORDS and len(token) >= 3:
                    tokens.add(token)
        return tokens

    @staticmethod
    def _merge_hits(
        *,
        semantic_hits: list[tuple[MemoryRecord, float]],
        lexical_hits: list[tuple[MemoryRecord, float]],
        anchor_hits: list[tuple[MemoryRecord, float]] | None = None,
    ) -> list[dict[str, object]]:
        merged: dict[str, dict[str, object]] = {}
        for record, score in semantic_hits:
            merged[record.id] = {
                "record": record,
                "semantic_score": float(score),
                "lexical_score": 0.0,
                "anchor_score": 0.0,
                "source": "episode_channel",
            }
        for record, score in lexical_hits:
            row = merged.setdefault(
                record.id,
                {
                    "record": record,
                    "semantic_score": 0.0,
                    "lexical_score": 0.0,
                    "anchor_score": 0.0,
                    "source": "episode_channel",
                },
            )
            row["lexical_score"] = max(float(row.get("lexical_score") or 0.0), float(score))
        for record, score in list(anchor_hits or []):
            row = merged.setdefault(
                record.id,
                {
                    "record": record,
                    "semantic_score": 0.0,
                    "lexical_score": 0.0,
                    "anchor_score": 0.0,
                    "source": "episode_anchor_channel",
                },
            )
            row["anchor_score"] = max(float(row.get("anchor_score") or 0.0), float(score))
            if not float(row.get("semantic_score") or 0.0) and not float(row.get("lexical_score") or 0.0):
                row["source"] = "episode_anchor_channel"
        out = list(merged.values())
        out.sort(
            key=lambda row: max(
                float(row.get("semantic_score") or 0.0),
                float(row.get("lexical_score") or 0.0),
                float(row.get("anchor_score") or 0.0),
            ),
            reverse=True,
        )
        return out

    @staticmethod
    def _fuzzy_overlap(left: set[str], right: set[str]) -> int:
        count = 0
        for a in set(left or set()):
            for b in set(right or set()):
                if DialogEpisodeRetriever._token_match(a, b):
                    count += 1
                    break
        return count

    @staticmethod
    def _token_match(a: str, b: str) -> bool:
        left = str(a or "").strip().lower()
        right = str(b or "").strip().lower()
        if not left or not right:
            return False
        if left == right:
            return True
        if len(left) >= 4 and len(right) >= 4:
            if left.startswith(right[:4]) or right.startswith(left[:4]):
                return True
        if len(left) >= 3 and len(right) >= 3:
            if left.startswith(right[:3]) or right.startswith(left[:3]):
                return True
        return False
