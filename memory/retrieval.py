"""Retrieval orchestration for Memory V2 (semantic + lexical fusion)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from memory.memory_models import (
    MemoryScope,
    MemoryStatus,
    MemoryRecord,
    RetrievalCandidate,
    RetrievalQuery,
    RetrievalResult,
)
from memory.memory_scoring import ScoreWeights, build_score_breakdown
from memory.vector_store import VectorStore


class LexicalScoreHook(Protocol):
    def __call__(self, query: RetrievalQuery, record: MemoryRecord, lexical_score: float) -> float:
        ...


@dataclass
class HybridRetriever:
    store: VectorStore
    stale_after_days: int = 30
    weights: ScoreWeights = field(default_factory=ScoreWeights)
    lexical_score_hook: LexicalScoreHook | None = None

    def retrieve(self, query: RetrievalQuery) -> RetrievalResult:
        namespace = str(query.namespace or "default")
        scopes = list(query.scopes or [])
        top_k = max(1, int(query.top_k))
        filters = dict(query.metadata_filters or {})

        semantic_hits = self.store.semantic_search(
            query_text=query.query_text,
            top_k=max(1, int(top_k * 3)),
            namespace=namespace,
            scopes=scopes,
            include_stale=bool(query.include_stale),
            metadata_filters=filters,
        )
        lexical_hits = self.store.lexical_search(
            query_text=query.query_text,
            top_k=max(1, int(top_k * 3)),
            namespace=namespace,
            scopes=scopes,
            include_stale=bool(query.include_stale),
            metadata_filters=filters,
        )
        rows = self._merge_hits(query=query, semantic_hits=semantic_hits, lexical_hits=lexical_hits)

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
            candidates.append(RetrievalCandidate(record=record, score_breakdown=breakdown, source="hybrid"))

        candidates.sort(key=lambda x: float(x.final_score), reverse=True)
        return RetrievalResult(
            query=query,
            candidates=candidates[: top_k],
            reindex_required=bool(self.store.reindex_required),
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
