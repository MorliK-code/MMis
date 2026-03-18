from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

from memory.memory_models import MemoryLevel, MemoryRecord, MemoryScope, MemoryType, RetrievalQuery
from memory.vector_store import VectorStore
from modules.nlu.normalizer import normalize_text


_SELF_SUBJECT_RE = re.compile(r"\b(?:я|мне|меня|мой|моя|моё|мои|у\s+меня|i|me|my)\b", re.I)
_LIKES_QUERY_RE = re.compile(r"(?:нравит|люблю|обожаю|like|love|favorite|любим)", re.I)
_DISLIKES_QUERY_RE = re.compile(r"(?:не\s+люблю|ненавиж|терпеть\s+не\s+могу|hate|dislike)", re.I)
_OWNS_QUERY_RE = re.compile(r"(?:у\s+меня|мой|моя|моё|что\s+у\s+меня\s+есть|i\s+have|my)", re.I)
_USES_QUERY_RE = re.compile(r"(?:использ|пользуюсь|юзаю|use|using|пользую)", re.I)
_SPONTANEOUS_CLAIM_QUERY_RE = re.compile(
    r"(?:что\s+мне\s+нравится|что\s+я\s+люблю|что\s+я\s+не\s+люблю|what\s+do\s+i\s+like|what\s+do\s+i\s+hate)",
    re.I,
)
_QUERY_STOPWORDS = {
    "что",
    "мне",
    "я",
    "мой",
    "моя",
    "мою",
    "мои",
    "у",
    "меня",
    "как",
    "про",
    "там",
    "есть",
    "do",
    "i",
    "my",
    "me",
    "what",
    "about",
    "the",
    "a",
    "an",
}


class ClaimLexicalScoreHook(Protocol):
    def __call__(self, query: RetrievalQuery, record: MemoryRecord, lexical_score: float) -> float:
        ...


@dataclass(frozen=True)
class ClaimQueryHints:
    query_text: str
    subjects: set[str]
    predicates: set[str]
    topic_keys: set[str]
    trigger_keys: set[str]
    spontaneous_recall_preferred: bool = False


@dataclass
class ClaimRetriever:
    store: VectorStore
    lexical_score_hook: ClaimLexicalScoreHook | None = None

    def retrieve(
        self,
        *,
        query: RetrievalQuery,
        query_text: str,
        namespace: str,
        scopes: list[MemoryScope],
        top_k: int,
        filters: dict[str, object],
    ) -> list[dict[str, object]]:
        hints = self.build_query_hints(query)
        claim_filters = self.build_filters(query=query, filters=filters, hints=hints)
        semantic_hits = self.store.semantic_search(
            query_text=query_text,
            top_k=max(1, int(top_k * 2)),
            namespace=namespace,
            scopes=scopes,
            include_stale=bool(query.include_stale),
            metadata_filters=claim_filters,
        )
        lexical_hits = self.store.lexical_search(
            query_text=query_text,
            top_k=max(1, int(top_k * 2)),
            namespace=namespace,
            scopes=scopes,
            include_stale=bool(query.include_stale),
            metadata_filters=claim_filters,
        )
        rows = self._merge_hits(query=query, semantic_hits=semantic_hits, lexical_hits=lexical_hits)
        rows = [
            row
            for row in list(rows or [])
            if getattr(row.get("record"), "level", None) != MemoryLevel.L4_DOCUMENT
            and not bool(dict(getattr(row.get("record"), "metadata", {}) or {}).get("document_id"))
            and not bool(dict(getattr(row.get("record"), "metadata", {}) or {}).get("doc_id"))
        ]
        for row in rows:
            row["source"] = "claim_channel"
        return rows

    def build_filters(
        self,
        *,
        query: RetrievalQuery,
        filters: dict[str, object],
        hints: ClaimQueryHints | None = None,
    ) -> dict[str, object]:
        query_hints = hints or self.build_query_hints(query)
        out: dict[str, object] = {
            **dict(filters or {}),
            "memory_type": [MemoryType.CLAIM.value],
        }
        if len(query_hints.subjects) == 1:
            out["metadata.claim.subject"] = next(iter(query_hints.subjects))
        return out

    def score_bonus(
        self,
        *,
        query: RetrievalQuery,
        record: MemoryRecord,
        source: str,
    ) -> float:
        claim = dict(dict(record.metadata or {}).get("claim") or {})
        if not claim:
            return 0.0
        hints = self.build_query_hints(query)
        subject = str(claim.get("subject") or "").strip().lower()
        predicate = str(claim.get("predicate") or "").strip().lower()
        obj = str(claim.get("obj") or claim.get("object_surface") or "").strip().lower()
        topic_keys = {
            str(x).strip().lower()
            for x in list(claim.get("topic_keys") or [])
            if str(x).strip()
        }
        trigger_keys = {
            str(x).strip().lower()
            for x in list(claim.get("trigger_keys") or [])
            if str(x).strip()
        }
        recall_mode = str(claim.get("recall_mode") or "").strip().lower()
        spontaneous_recall = bool(claim.get("spontaneous_recall"))

        bonus = 0.08
        if str(source or "") == "claim_channel":
            bonus += 0.04
        if hints.subjects and subject in hints.subjects:
            bonus += 0.06
        if hints.predicates and predicate in hints.predicates:
            bonus += 0.10

        topic_overlap = len(topic_keys.intersection(hints.topic_keys))
        trigger_overlap = len(trigger_keys.intersection(hints.trigger_keys))
        fuzzy_trigger_overlap = self._fuzzy_overlap(trigger_keys, hints.trigger_keys)
        if topic_overlap:
            bonus += min(0.09, 0.03 * topic_overlap)
        if trigger_overlap:
            bonus += min(0.10, 0.02 * trigger_overlap)
        elif fuzzy_trigger_overlap:
            bonus += min(0.10, 0.04 * fuzzy_trigger_overlap)

        if obj and (obj in hints.query_text or self._text_matches_query_tokens(obj, hints.trigger_keys)):
            bonus += 0.06
        if recall_mode == "ambient" and hints.spontaneous_recall_preferred:
            bonus += 0.08
        elif recall_mode == "contextual":
            bonus += 0.01
        if spontaneous_recall and hints.spontaneous_recall_preferred:
            bonus += 0.06
        return min(0.44, max(0.0, bonus))

    def build_query_hints(self, query: RetrievalQuery) -> ClaimQueryHints:
        query_text = normalize_text(str(query.query_text or query.search_text or "")).lower()
        subjects: set[str] = set()
        predicates: set[str] = set()
        topic_keys: set[str] = set()
        trigger_keys: set[str] = set()

        if _SELF_SUBJECT_RE.search(query_text):
            subjects.add("user")
        if _LIKES_QUERY_RE.search(query_text):
            predicates.add("likes")
            topic_keys.add("preference")
        if _DISLIKES_QUERY_RE.search(query_text):
            predicates.add("dislikes")
            topic_keys.add("preference")
        if _OWNS_QUERY_RE.search(query_text):
            predicates.add("owns")
            topic_keys.add("ownership")
        if _USES_QUERY_RE.search(query_text):
            predicates.add("uses")
            topic_keys.add("usage")

        for raw in list(query.entity_keys or []):
            token = str(raw or "").strip().lower()
            if not token:
                continue
            topic_keys.add(token)
            trigger_keys.add(token)

        for raw in re.split(r"\s+", query_text):
            token = str(raw or "").strip().lower()
            if not token or token in _QUERY_STOPWORDS or len(token) < 3:
                continue
            trigger_keys.add(token)

        return ClaimQueryHints(
            query_text=query_text,
            subjects=subjects,
            predicates=predicates,
            topic_keys=topic_keys,
            trigger_keys=trigger_keys,
            spontaneous_recall_preferred=bool(_SPONTANEOUS_CLAIM_QUERY_RE.search(query_text)),
        )

    def _merge_hits(
        self,
        *,
        query: RetrievalQuery,
        semantic_hits: list[tuple[MemoryRecord, float]],
        lexical_hits: list[tuple[MemoryRecord, float]],
    ) -> list[dict[str, object]]:
        merged: dict[str, dict[str, object]] = {}
        for record, score in semantic_hits:
            merged[record.id] = {
                "record": record,
                "semantic_score": float(score),
                "lexical_score": 0.0,
            }
        for record, score in lexical_hits:
            row = merged.setdefault(
                record.id,
                {
                    "record": record,
                    "semantic_score": 0.0,
                    "lexical_score": 0.0,
                },
            )
            lexical_score = float(score)
            if callable(self.lexical_score_hook):
                lexical_score = float(self.lexical_score_hook(query, record, lexical_score))
            row["lexical_score"] = max(float(row.get("lexical_score") or 0.0), lexical_score)

        out = list(merged.values())
        out.sort(
            key=lambda row: max(float(row.get("semantic_score") or 0.0), float(row.get("lexical_score") or 0.0)),
            reverse=True,
        )
        return out

    @staticmethod
    def _fuzzy_overlap(left: set[str], right: set[str]) -> int:
        count = 0
        for a in set(left or set()):
            for b in set(right or set()):
                if ClaimRetriever._token_match(a, b):
                    count += 1
                    break
        return count

    @staticmethod
    def _text_matches_query_tokens(text: str, query_tokens: set[str]) -> bool:
        tokens = {part for part in re.split(r"\s+", str(text or "").strip().lower()) if part}
        return ClaimRetriever._fuzzy_overlap(tokens, set(query_tokens or set())) > 0

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
