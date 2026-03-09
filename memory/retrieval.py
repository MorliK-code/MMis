from __future__ import annotations

from dataclasses import dataclass

from memory.memory_models import (
    MemoryScope,
    MemoryStatus,
    RetrievalCandidate,
    RetrievalQuery,
    RetrievalResult,
)
from memory.memory_scoring import ScoreWeights, build_score_breakdown
from memory.vector_store import VectorStore


@dataclass
class HybridRetriever:
    store: VectorStore
    stale_after_days: int = 30
    weights: ScoreWeights = ScoreWeights()

    def retrieve(self, query: RetrievalQuery) -> RetrievalResult:
        rows = self.store.search(
            query_text=query.query_text,
            top_k=max(1, int(query.top_k)),
            namespace=str(query.namespace or "default"),
            scopes=list(query.scopes or []),
            include_stale=bool(query.include_stale),
            metadata_filters=dict(query.metadata_filters or {}),
        )

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
            candidates=candidates[: max(1, int(query.top_k))],
            reindex_required=bool(self.store.reindex_required),
        )
