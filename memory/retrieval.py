"""Retrieval orchestration for Memory V2 (semantic + lexical fusion)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Protocol

from memory.claim_retrieval import ClaimRetriever
from memory.memory_models import (
    MemoryLevel,
    MemoryScope,
    MemoryStatus,
    MemoryRecord,
    MemoryType,
    RetrievalCandidate,
    RetrievalQuery,
    RetrievalResult,
    ScoreBreakdown,
)
from memory.memory_scoring import ScoreWeights, build_score_breakdown
from memory.recall_policy import classify_query_recall_profile, recall_policy_adjustment
from memory.vector_store import VectorStore
from modules.nlu.normalizer import normalize_text


_SELF_QUERY_RE = re.compile(
    r"(?:какая|какой|какое|напомни|подскажи|скажи|what(?:'s| is)|remind me)[^?.!\n]{0,56}(?:у меня|мой|мою|моя|моё|my)\b",
    re.I,
)
_GPU_QUERY_RE = re.compile(r"\b(?:gpu|rtx|gtx|rx|видеокарт|видюх|карточк|video card|graphics card)\b", re.I)
_PYTHON_QUERY_RE = re.compile(r"\b(?:python|питон|пайтон)\b", re.I)
_OS_QUERY_RE = re.compile(r"\b(?:os|windows|linux|ubuntu|debian|macos|операционк|ос|винда|винды|система)\b", re.I)
_NAME_QUERY_RE = re.compile(r"\b(?:name|имя|зовут)\b", re.I)
_AGE_QUERY_RE = re.compile(r"\b(?:age|возраст|лет|год)\b", re.I)
_ASSISTANT_MEMORY_MISS_RE = re.compile(
    r"(?:не\s+помню|не\s+вижу\s+в\s+памяти|не\s+вижу\s+в\s+истории|don't\s+remember|can't\s+remember|cannot\s+remember)",
    re.I,
)
_ASSISTANT_SELF_CHECK_RE = re.compile(
    r"(?:проверь\s+сам|можешь\s+проверить|можно\s+проверить|посмотри\s+сам|как\s+посмотреть|you\s+can\s+check|check\s+it\s+yourself|to\s+check)",
    re.I,
)
_ASSISTANT_HELP_COMMAND_RE = re.compile(
    r"(?:\bpython\s+--version\b|\bwinver\b|\bwmic\b|\bdxdiag\b|\blspci\b|\blshw\b|\bdevice manager\b|диспетчер устройств)",
    re.I,
)

_MESSAGE_CHANNEL_TYPES = [
    MemoryType.MESSAGE.value,
    MemoryType.SUMMARY.value,
    MemoryType.EPISODE.value,
    MemoryType.SEMANTIC.value,
    MemoryType.TASK_STATE.value,
    MemoryType.TOOL_RESULT.value,
    MemoryType.RUNTIME_STATE.value,
]


class LexicalScoreHook(Protocol):
    def __call__(self, query: RetrievalQuery, record: MemoryRecord, lexical_score: float) -> float:
        ...


@dataclass
class HybridRetriever:
    store: VectorStore
    stale_after_days: int = 30
    weights: ScoreWeights = field(default_factory=ScoreWeights)
    lexical_score_hook: LexicalScoreHook | None = None
    min_candidate_score: float = 0.28
    min_memory_admit_score: float = 0.38
    claim_retriever: ClaimRetriever = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.claim_retriever = ClaimRetriever(store=self.store, lexical_score_hook=self.lexical_score_hook)

    def retrieve(self, query: RetrievalQuery) -> RetrievalResult:
        namespace = str(query.namespace or "default")
        scopes = list(query.scopes or [])
        top_k = max(1, int(query.top_k))
        filters = dict(query.metadata_filters or {})
        query_text = str(query.search_text or query.query_text or "").strip()

        fact_hits = self._retrieve_fact_hits(
            query=query,
            query_text=query_text,
            namespace=namespace,
            scopes=scopes,
            top_k=top_k,
            filters=filters,
        )
        claim_hits = self._retrieve_claim_hits(
            query=query,
            query_text=query_text,
            namespace=namespace,
            scopes=scopes,
            top_k=top_k,
            filters=filters,
        )
        message_hits = self._retrieve_message_hits(
            query=query,
            query_text=query_text,
            namespace=namespace,
            scopes=scopes,
            top_k=top_k,
            filters=filters,
        )
        rows = self._fuse_memory_hits(query=query, fact_hits=fact_hits, claim_hits=claim_hits, message_hits=message_hits)

        candidates: list[RetrievalCandidate] = []
        for row in rows:
            record = row.get("record")
            if record is None:
                continue
            if record.scope == MemoryScope.PRIVATE_RUNTIME and not bool(query.include_private_runtime):
                continue
            if record.status in {MemoryStatus.ARCHIVED, MemoryStatus.DELETED}:
                continue
            breakdown = build_score_breakdown(
                query=query,
                record=record,
                semantic_similarity=float(row.get("semantic_score") or 0.0),
                lexical_score=float(row.get("lexical_score") or 0.0),
                stale_after_days=int(self.stale_after_days),
                weights=self.weights,
            )
            boosted = self._apply_channel_boosts(
                query=query,
                record=record,
                breakdown=breakdown,
                source=str(row.get("source") or "hybrid"),
            )
            candidates.append(
                RetrievalCandidate(
                    record=record,
                    score_breakdown=boosted,
                    source=str(row.get("source") or "hybrid"),
                )
            )

        candidates.sort(key=lambda x: float(x.final_score), reverse=True)
        candidates = self._collapse_candidates(candidates)
        candidates = self._apply_thresholds(candidates)
        return RetrievalResult(
            query=query,
            candidates=candidates[: top_k],
            reindex_required=bool(self.store.reindex_required),
        )

    def _retrieve_fact_hits(
        self,
        *,
        query: RetrievalQuery,
        query_text: str,
        namespace: str,
        scopes: list[MemoryScope],
        top_k: int,
        filters: dict[str, object],
    ) -> list[dict[str, object]]:
        fact_filters = {
            **dict(filters or {}),
            "memory_type": [MemoryType.FACT.value],
        }
        semantic_hits = self.store.semantic_search(
            query_text=query_text,
            top_k=max(1, int(top_k * 3)),
            namespace=namespace,
            scopes=scopes,
            include_stale=bool(query.include_stale),
            metadata_filters=fact_filters,
        )
        lexical_hits = self.store.lexical_search(
            query_text=query_text,
            top_k=max(1, int(top_k * 3)),
            namespace=namespace,
            scopes=scopes,
            include_stale=bool(query.include_stale),
            metadata_filters=fact_filters,
        )
        rows = self._merge_hits(query=query, semantic_hits=semantic_hits, lexical_hits=lexical_hits)
        for row in rows:
            row["source"] = "fact_channel"
        return rows

    def _retrieve_message_hits(
        self,
        *,
        query: RetrievalQuery,
        query_text: str,
        namespace: str,
        scopes: list[MemoryScope],
        top_k: int,
        filters: dict[str, object],
    ) -> list[dict[str, object]]:
        message_filters = {
            **dict(filters or {}),
            "memory_type": list(_MESSAGE_CHANNEL_TYPES),
        }
        semantic_hits = self.store.semantic_search(
            query_text=query_text,
            top_k=max(1, int(top_k * 3)),
            namespace=namespace,
            scopes=scopes,
            include_stale=bool(query.include_stale),
            metadata_filters=message_filters,
        )
        lexical_hits = self.store.lexical_search(
            query_text=query_text,
            top_k=max(1, int(top_k * 3)),
            namespace=namespace,
            scopes=scopes,
            include_stale=bool(query.include_stale),
            metadata_filters=message_filters,
        )
        rows = self._merge_hits(query=query, semantic_hits=semantic_hits, lexical_hits=lexical_hits)
        rows = [
            row
            for row in list(rows or [])
            if getattr(row.get("record"), "level", None) != MemoryLevel.L4_DOCUMENT
        ]
        for row in rows:
            row["source"] = "message_channel"
        return rows

    def _retrieve_claim_hits(
        self,
        *,
        query: RetrievalQuery,
        query_text: str,
        namespace: str,
        scopes: list[MemoryScope],
        top_k: int,
        filters: dict[str, object],
    ) -> list[dict[str, object]]:
        return self.claim_retriever.retrieve(
            query=query,
            query_text=query_text,
            namespace=namespace,
            scopes=scopes,
            top_k=top_k,
            filters=filters,
        )

    def _fuse_memory_hits(
        self,
        *,
        query: RetrievalQuery,
        fact_hits: list[dict[str, object]],
        claim_hits: list[dict[str, object]],
        message_hits: list[dict[str, object]],
    ) -> list[dict[str, object]]:
        merged: dict[str, dict[str, object]] = {}
        for row in list(fact_hits or []) + list(claim_hits or []) + list(message_hits or []):
            record = row.get("record")
            if not isinstance(record, MemoryRecord):
                continue
            current = merged.setdefault(
                record.id,
                {
                    "record": record,
                    "semantic_score": 0.0,
                    "lexical_score": 0.0,
                    "source": str(row.get("source") or "hybrid"),
                },
            )
            current["semantic_score"] = max(float(current.get("semantic_score") or 0.0), float(row.get("semantic_score") or 0.0))
            current["lexical_score"] = max(float(current.get("lexical_score") or 0.0), float(row.get("lexical_score") or 0.0))
            if str(current.get("source") or "") != "fact_channel" and str(row.get("source") or "") == "fact_channel":
                current["source"] = "fact_channel"
            elif str(current.get("source") or "") == "message_channel" and str(row.get("source") or "") == "claim_channel":
                current["source"] = "claim_channel"
        out = list(merged.values())
        out.sort(
            key=lambda row: (
                1.0 if getattr(row.get("record"), "memory_type", None) == MemoryType.FACT else 0.0,
                1.0 if getattr(row.get("record"), "memory_type", None) == MemoryType.CLAIM else 0.0,
                max(float(row.get("semantic_score") or 0.0), float(row.get("lexical_score") or 0.0)),
            ),
            reverse=True,
        )
        return out

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

    def _apply_channel_boosts(
        self,
        *,
        query: RetrievalQuery,
        record: MemoryRecord,
        breakdown: ScoreBreakdown,
        source: str,
    ) -> ScoreBreakdown:
        final = float(breakdown.final_score)
        fact_bonus = 0.0
        claim_bonus = 0.0
        profile = classify_query_recall_profile(str(query.query_text or query.search_text or ""))
        if record.memory_type == MemoryType.FACT:
            fact_bonus += 0.12
            if str(source or "") == "fact_channel":
                fact_bonus += 0.06
            expected = self._expected_self_fact_predicates(query)
            record_predicate = self._record_fact_predicate(record)
            record_subject = self._record_fact_subject(record)
            if record.status == MemoryStatus.ACTIVE:
                fact_bonus += 0.06
            if record_subject == "user":
                fact_bonus += 0.08
            if record_predicate and record_predicate in expected:
                fact_bonus += 0.40
            elif self._is_self_like_query(query) and record_predicate and record_subject == "user":
                fact_bonus += 0.16
        if record.memory_type == MemoryType.CLAIM and record.level != MemoryLevel.L4_DOCUMENT:
            claim_bonus += self.claim_retriever.score_bonus(
                query=query,
                record=record,
                source=source,
            )
        recall_policy_bonus = recall_policy_adjustment(profile=profile, record=record)
        assistant_penalty = self._assistant_reply_penalty(query=query, record=record)
        boosted_final = max(0.0, min(1.0, final + fact_bonus + claim_bonus + recall_policy_bonus - assistant_penalty))
        return ScoreBreakdown(
            semantic_similarity=breakdown.semantic_similarity,
            lexical_score=breakdown.lexical_score,
            recency_score=breakdown.recency_score,
            importance_score=breakdown.importance_score,
            confidence_score=breakdown.confidence_score,
            entity_overlap_score=breakdown.entity_overlap_score,
            numeric_overlap_score=breakdown.numeric_overlap_score,
            exact_match_boost=breakdown.exact_match_boost,
            scope_match_score=breakdown.scope_match_score,
            final_score=boosted_final,
        )

    def _assistant_reply_penalty(self, *, query: RetrievalQuery, record: MemoryRecord) -> float:
        meta = dict(record.metadata or {})
        source_kind = str(meta.get("source_kind") or "").strip().lower()
        if source_kind != "assistant_reply":
            return 0.0
        penalty = 0.06
        if record.level == MemoryLevel.L2_EPISODIC:
            penalty += 0.04
        # Use assistant_write_reason directly from metadata (no longer stored in policy)
        write_reason = str(meta.get("assistant_write_reason") or "").strip().lower()
        reply_kind = str(meta.get("assistant_reply_kind") or "").strip().lower()
        if write_reason == "assistant_memory_miss_help_temporary_only" or reply_kind == "memory_miss_help":
            penalty += 0.24

        low_text = normalize_text(str(record.text or "")).lower()
        if _ASSISTANT_MEMORY_MISS_RE.search(low_text):
            penalty += 0.14
        if _ASSISTANT_SELF_CHECK_RE.search(low_text):
            penalty += 0.10
        if _ASSISTANT_HELP_COMMAND_RE.search(str(record.text or "")):
            penalty += 0.12
        if self._is_self_like_query(query):
            penalty += 0.10
        return max(0.0, min(0.60, penalty))

    @staticmethod
    def _record_fact_predicate(record: MemoryRecord) -> str:
        meta = dict(record.metadata or {})
        fact = dict(meta.get("fact") or {})
        return str(fact.get("predicate") or "").strip().lower()

    @staticmethod
    def _record_fact_subject(record: MemoryRecord) -> str:
        meta = dict(record.metadata or {})
        fact = dict(meta.get("fact") or {})
        return str(fact.get("subject") or "").strip().lower()

    @staticmethod
    def _is_self_like_query(query: RetrievalQuery) -> bool:
        text = str(query.query_text or query.search_text or "").strip().lower()
        if not text:
            return False
        if _SELF_QUERY_RE.search(text):
            return True
        return any(token in text for token in ("у меня", "мой ", "мою ", "моя ", "моё ", "my "))

    @classmethod
    def _expected_self_fact_predicates(cls, query: RetrievalQuery) -> set[str]:
        text = str(query.query_text or query.search_text or "").strip().lower()
        if not text:
            return set()
        out: set[str] = set()
        if _GPU_QUERY_RE.search(text):
            out.add("environment_gpu_model")
        if _PYTHON_QUERY_RE.search(text):
            out.add("environment_runtime_python")
        if _OS_QUERY_RE.search(text):
            out.add("environment_os")
        if _NAME_QUERY_RE.search(text):
            out.add("identity_name")
        if _AGE_QUERY_RE.search(text):
            out.add("identity_age_years")
        return out

    @staticmethod
    def _collapse_key(record: MemoryRecord) -> str:
        meta = dict(record.metadata or {})
        canonical_key = str(meta.get("canonical_key") or "").strip().lower()
        if canonical_key:
            return f"canonical:{canonical_key}"
        if record.parent_id and record.chunk_index is not None:
            return f"chunk:{record.parent_id}:{int(record.chunk_index)}"
        if record.source_event_id and record.memory_type.value not in {"document", "document_chunk"}:
            return f"event:{record.source_event_id}"
        return f"id:{record.id}"

    def _collapse_candidates(self, candidates: list[RetrievalCandidate]) -> list[RetrievalCandidate]:
        out: list[RetrievalCandidate] = []
        seen: set[str] = set()
        for candidate in list(candidates or []):
            key = self._collapse_key(candidate.record)
            if key in seen:
                continue
            seen.add(key)
            out.append(candidate)
        return out

    def _apply_thresholds(self, candidates: list[RetrievalCandidate]) -> list[RetrievalCandidate]:
        admitted = [x for x in list(candidates or []) if float(x.final_score) >= float(self.min_memory_admit_score)]
        if admitted:
            return admitted
        return [x for x in list(candidates or []) if float(x.final_score) >= float(self.min_candidate_score)]
