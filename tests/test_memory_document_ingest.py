from __future__ import annotations

from pathlib import Path

from memory.document_ingest import DocumentIngestPipeline
from memory.memory_models import MemoryScope, MemoryType


class _StoreSpy:
    def __init__(self) -> None:
        self.records: dict[str, object] = {}
        self.batch_calls: list[list[str]] = []

    def batch_upsert(self, records) -> None:
        ids: list[str] = []
        for record in list(records or []):
            self.records[str(record.id)] = record
            ids.append(str(record.id))
        self.batch_calls.append(ids)


def test_document_ingest_load_file_parses_text(tmp_path: Path) -> None:
    path = tmp_path / "notes.md"
    path.write_text("\ufeff\r\n# Memory Notes\r\n\r\nWe use Python 3.11.\r\n", encoding="utf-8")

    row = DocumentIngestPipeline.load_file(path)

    assert row["title"] == "notes"
    assert row["metadata"]["extension"] == ".md"
    assert row["text"] == "# Memory Notes\n\nWe use Python 3.11."


def test_document_ingest_pipeline_builds_artifacts_and_stores_records(tmp_path: Path) -> None:
    path = tmp_path / "memory_notes.md"
    path.write_text(
        "\n".join(
            [
                "# Memory Architecture",
                "",
                "We use Python 3.11 and Windows 11 in the project.",
                "",
                "## Tools",
                "I use VS Code and Docker.",
                "I like dark themes.",
            ]
        ),
        encoding="utf-8",
    )

    store = _StoreSpy()
    pipeline = DocumentIngestPipeline(store=store)

    result = pipeline.ingest_file(str(path), namespace="default", scope=MemoryScope.PROJECT)

    assert result.document.id.startswith("doc:")
    assert len(result.chunks) >= 1
    assert len(result.chunk_analyses) == len(result.chunks)
    assert len(result.section_summaries) == len(result.chunks)
    assert result.document_outline is not None
    assert any(row.predicate == "uses" for row in result.document_claims)

    record_types = {getattr(row, "memory_type", None) for row in store.records.values()}
    assert MemoryType.DOCUMENT in record_types
    assert MemoryType.DOCUMENT_CHUNK in record_types
    assert MemoryType.SUMMARY in record_types
    assert MemoryType.CLAIM in record_types

    chunk_records = [row for row in store.records.values() if getattr(row, "memory_type", None) == MemoryType.DOCUMENT_CHUNK]
    assert chunk_records
    chunk_metadata = dict(chunk_records[0].metadata or {})
    assert "memory_analysis" not in chunk_metadata
    assert "memory_views" in chunk_metadata
    assert "search_text" not in dict(chunk_metadata.get("memory_views") or {})
    assert "normalized_text" not in dict(chunk_metadata.get("memory_views") or {})
    assert "canonical_text" not in dict(chunk_metadata.get("memory_views") or {})
    assert "claim_candidates" not in chunk_metadata
    assert "memory_tags" in chunk_metadata
    assert any(dict(getattr(row, "metadata", {}) or {}).get("claims") for row in chunk_records)

    outline_records = [
        row
        for row in store.records.values()
        if getattr(row, "memory_type", None) == MemoryType.SUMMARY
        and str(dict(row.metadata or {}).get("summary_kind") or "") == "outline"
    ]
    assert outline_records
    assert "Document outline" in str(outline_records[0].text or "")

    assert result.stored_ids
    assert result.document.id in result.stored_ids
