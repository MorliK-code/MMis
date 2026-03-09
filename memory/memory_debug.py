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
        return {
            "count": len(out),
            "items": [x.to_dict() for x in out],
            "reindex_required": bool(self.store.reindex_required),
            "last_retrieval_trace": dict(self._last_retrieval_trace or {}),
        }

    def record_retrieval_trace(
        self,
        *,
        query: str,
        selected: list[RetrievalCandidate],
        dropped: list[dict[str, Any]],
    ) -> None:
        self._last_retrieval_trace = {
            "query": str(query or ""),
            "selected": [x.to_dict() for x in list(selected or [])],
            "dropped": [dict(x) for x in list(dropped or [])],
        }

    def conflicts(self, *, namespace: str = "default", limit: int = 100) -> dict[str, Any]:
        rows = self.store.iter_records(namespace=namespace)
        superseded = [x.to_dict() for x in rows if x.status == MemoryStatus.SUPERSEDED][: max(1, int(limit))]
        stale = [x.to_dict() for x in rows if x.status == MemoryStatus.STALE][: max(1, int(limit))]
        return {
            "superseded": superseded,
            "stale": stale,
        }


def is_private_runtime_scope(scope: MemoryScope) -> bool:
    return scope == MemoryScope.PRIVATE_RUNTIME
