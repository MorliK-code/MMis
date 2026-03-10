from __future__ import annotations

from _output_utils import enable_unittest_json_output

enable_unittest_json_output()

import shutil
import tempfile
import time
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from config.settings import load_config
from memory.fact_extractor import FactExtractor
from memory.memory_lifecycle import MemoryLifecycleManager
from memory.memory_manager import MemoryManager
from memory.memory_models import MemoryEvent, MemoryLevel, MemoryRecord, MemoryScope, MemoryStatus, MemoryType
from memory.memory_scoring import build_salience_score


class MemoryPhase3LongTermTests(unittest.TestCase):
    def test_document_event_ingest_creates_document_and_chunks(self) -> None:
        cfg = replace(
            load_config(force_reload=True),
            memory_backend="chroma",
            memory_chunk_size=140,
            memory_chunk_overlap=24,
        )
        tmpdir = tempfile.mkdtemp(prefix="mmis_phase3_doc_ingest_")
        try:
            with patch("memory.memory_manager.load_config", return_value=cfg):
                manager = MemoryManager(root_dir=Path(tmpdir))
            try:
                text = (
                    "# Build Notes\n"
                    "Project MMis release checklist.\n"
                    "1) Validate migration steps.\n"
                    "2) Run tests.\n"
                    "3) Publish artifact.\n"
                    "This section is intentionally long to force chunking overlap behavior."
                )
                out = manager.ingest_event(
                    MemoryEvent(
                        role="user",
                        text=text,
                        namespace="n1",
                        scope=MemoryScope.PROJECT,
                        memory_type=MemoryType.DOCUMENT,
                        metadata={"source": "docs/build_notes.md", "title": "Build Notes", "language": "markdown"},
                    )
                )
                self.assertGreaterEqual(len(out.stored_ids), 2)
                rows = manager._store.iter_records(namespace="n1")
                docs = [x for x in rows if x.memory_type == MemoryType.DOCUMENT]
                chunks = [x for x in rows if x.memory_type == MemoryType.DOCUMENT_CHUNK]
                self.assertTrue(docs)
                self.assertTrue(chunks)
                doc = docs[0]
                self.assertEqual(str(doc.metadata.get("title") or ""), "Build Notes")
                self.assertEqual(str(doc.metadata.get("source") or ""), "docs/build_notes.md")
                for chunk in chunks:
                    self.assertEqual(str(chunk.parent_id or ""), doc.id)
                    self.assertEqual(str(chunk.metadata.get("doc_id") or ""), doc.id)
                    self.assertIn("chunk_id", dict(chunk.metadata or {}))
            finally:
                manager.close()
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_fact_extractor_v2_covers_multiple_relations(self) -> None:
        extractor = FactExtractor()
        text = (
            "My name is Alex. We decided to use chromadb for project mmis. "
            "I prefer concise answers. Need to fix docker issue for now. "
            "This error is unresolved."
        )
        rows = extractor.extract_v2(
            text=text,
            metadata={"event_id": "evt:1", "namespace": "n1"},
            speaker="user",
            scope=MemoryScope.CONVERSATION,
        )
        relations = {str(x.relation or "") for x in rows}
        self.assertIn("identity", relations)
        self.assertIn("decision", relations)
        self.assertIn("preference", relations)
        self.assertIn("task", relations)
        self.assertIn("issue", relations)
        self.assertIn("unresolved", relations)
        self.assertIn("temporary", relations)
        one = rows[0]
        self.assertTrue(str(one.canonical_key or "").strip())
        self.assertTrue(str(one.source_event_id or "").strip())

    def test_lifecycle_decay_and_conflict_resolution(self) -> None:
        manager = MemoryLifecycleManager(stale_after_days=1, archive_after_days=2, parallel_margin=0.09)
        now_ts = float(time.time())
        old = MemoryRecord(
            id="fact:old",
            text="user.preference=short",
            memory_type=MemoryType.FACT,
            level=MemoryLevel.L3_SEMANTIC,
            scope=MemoryScope.CONVERSATION,
            namespace="n1",
            metadata={"canonical_key": "user.preference"},
            importance=0.70,
            confidence=0.75,
            created_at=now_ts - (3 * 86400),
            updated_at=now_ts - (3 * 86400),
            status=MemoryStatus.ACTIVE,
        )
        decayed = manager.apply_decay([old], now_ts=now_ts)[0]
        self.assertEqual(decayed.status, MemoryStatus.ARCHIVED)
        self.assertGreater(decayed.version, old.version)

        close_new = MemoryRecord(
            id="fact:new_close",
            text="user.preference=brief",
            memory_type=MemoryType.FACT,
            level=MemoryLevel.L3_SEMANTIC,
            scope=MemoryScope.CONVERSATION,
            namespace="n1",
            metadata={"canonical_key": "user.preference"},
            importance=0.69,
            confidence=0.74,
            created_at=now_ts,
            updated_at=old.updated_at + 120.0,
            status=MemoryStatus.ACTIVE,
        )
        parallel = manager.resolve_conflict(old=old, new=close_new)
        self.assertEqual(str(parallel.action), "parallel")

        strong_new = MemoryRecord(
            id="fact:new_strong",
            text="user.preference=concise",
            memory_type=MemoryType.FACT,
            level=MemoryLevel.L3_SEMANTIC,
            scope=MemoryScope.CONVERSATION,
            namespace="n1",
            metadata={"canonical_key": "user.preference"},
            importance=0.95,
            confidence=0.95,
            created_at=now_ts,
            updated_at=now_ts,
            status=MemoryStatus.ACTIVE,
        )
        winner = manager.resolve_conflict(old=old, new=strong_new)
        self.assertIn(str(winner.action), {"supersede", "archive"})

    def test_salience_score_uses_explicit_signal_and_relevance(self) -> None:
        score_plain = build_salience_score(
            text="Just a random note",
            metadata={},
            recent_texts=["Just a random note", "Another unrelated line"],
        )
        score_save = build_salience_score(
            text="Please remember this project decision forever",
            metadata={"explicit_save": True, "project_id": "mmis", "is_task": True},
            recent_texts=["old memory one", "old memory two"],
        )
        self.assertGreater(float(score_save), float(score_plain))


if __name__ == "__main__":
    unittest.main()
