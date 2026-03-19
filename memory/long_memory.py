"""Long-term memory facade for Memory V2.

Phase 3 keeps this module as an adapter around the new V2 store/document
pipeline so older "long memory" references have a single clear integration
point instead of parallel storage paths.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from memory.document_memory import DocumentMemory
from memory.memory_models import DocumentIngestRequest, DocumentIngestResult, MemoryScope, MemoryType
from memory.vector_store import VectorStore


LONG_MEMORY_ROLE = "thin_document_facade"


@dataclass
class LongMemoryV2:
    """Thin facade over document-memory services for legacy manager integration.

    New document logic should live in:
        DocumentMemory -> low-level chunk/document storage
        DocumentIngestPipeline -> high-level analysis/summaries/claims pipeline
    """

    store: VectorStore
    document_memory: DocumentMemory

    def ingest_document(self, request: DocumentIngestRequest) -> DocumentIngestResult:
        return self.document_memory.ingest_document(request)

    def retrieve(
        self,
        *,
        query_text: str,
        namespace: str,
        top_k: int = 8,
        scopes: list[MemoryScope] | None = None,
        include_documents: bool = True,
        include_chunks: bool = True,
        include_facts: bool = True,
    ) -> list[dict[str, Any]]:
        types: list[str] = []
        if include_documents:
            types.append(MemoryType.DOCUMENT.value)
        if include_chunks:
            types.append(MemoryType.DOCUMENT_CHUNK.value)
        if include_facts:
            types.append(MemoryType.FACT.value)
        filters = {"memory_type": types} if types else {}
        return self.store.search(
            query_text=str(query_text or ""),
            top_k=max(1, int(top_k)),
            namespace=str(namespace or "default"),
            scopes=list(scopes or []),
            include_stale=False,
            metadata_filters=filters,
        )

    def document_tree(self, *, document_id: str, namespace: str) -> dict[str, Any]:
        return self.document_memory.document_tree(document_id=document_id, namespace=namespace)
