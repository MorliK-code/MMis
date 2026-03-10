"""Debugging views and traces for Memory V2 state."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from memory.memory_models import DebugRequest, MemoryRecord, MemoryScope, MemoryStatus, RetrievalCandidate
from memory.vector_store import VectorStore


@dataclass
class BasicMemoryDebugger:
    store: VectorStore
    _last_retrieval_trace: dict[str, Any] = field(default_factory=dict)

    def snapshot(self, request: DebugRequest) -> dict[str, Any]:
        rows = self.store.iter_records(namespace=str(request.namespace or "default"))
        filtered: list[MemoryRecord] = []
        for row in rows:
            if request.scope is not None and row.scope != request.scope:
                continue
            if request.status is not None and row.status != request.status:
                continue
            filtered.append(row)
        filtered.sort(key=lambda x: float(x.updated_at), reverse=True)
        out = filtered[: max(1, int(request.limit))]
        status_counts: dict[str, int] = {}
        for row in filtered:
            key = str(row.status.value)
            status_counts[key] = int(status_counts.get(key, 0)) + 1
        stale_rows = [x.to_dict() for x in filtered if x.status == MemoryStatus.STALE][:20]
        superseded_rows = [x.to_dict() for x in filtered if x.status == MemoryStatus.SUPERSEDED][:20]
        return {
            "count": len(out),
            "items": [x.to_dict() for x in out],
            "reindex_required": bool(self.store.reindex_required),
            "last_retrieval_trace": dict(self._last_retrieval_trace or {}),
            "status_counts": status_counts,
            "inspection": {
                "stale": stale_rows,
                "superseded": superseded_rows,
            },
        }

    def record_retrieval_trace(
        self,
        *,
        query: str,
        selected: list[RetrievalCandidate],
        dropped: list[dict[str, Any]],
        score_breakdowns: list[dict[str, Any]] | None = None,
        truncation_log: list[dict[str, Any]] | None = None,
        context_blocks: dict[str, str] | None = None,
    ) -> None:
        selected_rows: list[dict[str, Any]] = []
        for row in list(selected or []):
            item = row.to_dict()
            breakdown = dict(item.get("score_breakdown") or {})
            item["why_selected"] = _why_selected_from_breakdown(breakdown)
            selected_rows.append(item)
        block_tokens = {}
        for key, value in dict(context_blocks or {}).items():
            text = str(value or "").strip()
            if text:
                block_tokens[str(key)] = max(1, len(text) // 4)
        self._last_retrieval_trace = {
            "query": str(query or ""),
            "selected": selected_rows,
            "dropped": [dict(x) for x in list(dropped or [])],
            "score_breakdowns": [dict(x) for x in list(score_breakdowns or [])],
            "truncation_log": [dict(x) for x in list(truncation_log or [])],
            "context_block_tokens_estimate": block_tokens,
        }

    def conflicts(self, *, namespace: str = "default", limit: int = 100) -> dict[str, Any]:
        rows = self.store.iter_records(namespace=namespace)
        superseded = [x.to_dict() for x in rows if x.status == MemoryStatus.SUPERSEDED][: max(1, int(limit))]
        stale = [x.to_dict() for x in rows if x.status == MemoryStatus.STALE][: max(1, int(limit))]
        archived = [x.to_dict() for x in rows if x.status == MemoryStatus.ARCHIVED][: max(1, int(limit))]
        return {
            "superseded": superseded,
            "stale": stale,
            "archived": archived,
        }


def is_private_runtime_scope(scope: MemoryScope) -> bool:
    return scope == MemoryScope.PRIVATE_RUNTIME


def _why_selected_from_breakdown(breakdown: dict[str, Any]) -> list[str]:
    weights = {
        "semantic_similarity": float(breakdown.get("semantic_similarity") or 0.0),
        "lexical_score": float(breakdown.get("lexical_score") or 0.0),
        "recency_score": float(breakdown.get("recency_score") or 0.0),
        "importance_score": float(breakdown.get("importance_score") or 0.0),
        "confidence_score": float(breakdown.get("confidence_score") or 0.0),
        "entity_overlap_score": float(breakdown.get("entity_overlap_score") or 0.0),
        "exact_match_boost": float(breakdown.get("exact_match_boost") or 0.0),
        "scope_match_score": float(breakdown.get("scope_match_score") or 0.0),
    }
    ranked = sorted(weights.items(), key=lambda x: float(x[1]), reverse=True)
    out = [str(name) for name, value in ranked if float(value) > 0.0][:3]
    return out or ["fallback_score"]
