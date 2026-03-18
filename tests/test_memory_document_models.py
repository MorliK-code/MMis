from __future__ import annotations

from memory.document_models import DocumentClaim, DocumentChunk, DocumentRecord, DocumentSummary
from memory.memory_models import MemoryScope, MemoryStatus


def test_document_summary_roundtrip() -> None:
    summary = DocumentSummary(
        id="docsum:1",
        document_id="doc:1",
        text="Memory architecture summary.",
        summary_kind="overview",
        source_chunk_ids=["chunk:1", "chunk:2"],
        topic_keys=["memory", "architecture"],
        entity_keys=["memory_manager"],
        confidence=0.88,
        salience=0.77,
        metadata={"lang": "en"},
        created_at=100.0,
        updated_at=120.0,
        status=MemoryStatus.ACTIVE,
    )

    restored = DocumentSummary.from_dict(summary.to_dict())

    assert restored == summary


def test_document_claim_roundtrip() -> None:
    claim = DocumentClaim(
        id="docclaim:1",
        document_id="doc:1",
        chunk_id="chunk:1",
        subject="memory",
        predicate="uses",
        obj="hybrid retrieval",
        object_type="architecture_pattern",
        object_surface="hybrid retrieval",
        qualifiers={"stage": "retrieval"},
        confidence=0.86,
        salience=0.72,
        evidence_text="Retrieval combines facts, claims, messages, and docs.",
        topic_keys=["memory", "retrieval"],
        trigger_keys=["hybrid", "retrieval"],
        metadata={"source": "doc"},
        created_at=100.0,
        updated_at=120.0,
        status=MemoryStatus.ACTIVE,
    )

    restored = DocumentClaim.from_dict(claim.to_dict())

    assert restored == claim


def test_document_record_and_chunk_from_dict() -> None:
    record = DocumentRecord.from_dict(
        {
            "id": "doc:alpha",
            "source": "notes.md",
            "text": "Long document body",
            "summary": "Short document summary",
            "scope": MemoryScope.PROJECT.value,
            "namespace": "default",
            "metadata": {"title": "Alpha"},
            "status": MemoryStatus.ACTIVE.value,
        }
    )
    chunk = DocumentChunk.from_dict(
        {
            "id": "chunk:alpha:0",
            "document_id": "doc:alpha",
            "chunk_index": 0,
            "text": "First chunk",
            "scope": MemoryScope.PROJECT.value,
            "namespace": "default",
            "metadata": {"title": "Alpha"},
            "status": MemoryStatus.ACTIVE.value,
        }
    )

    assert record.id == "doc:alpha"
    assert record.scope == MemoryScope.PROJECT
    assert chunk.document_id == "doc:alpha"
    assert chunk.chunk_index == 0
