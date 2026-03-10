from __future__ import annotations

from _output_utils import enable_unittest_json_output

enable_unittest_json_output()

import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from config.settings import load_config
from memory.auto_migration import MIGRATION_STATE_FILE, run_auto_migration
from memory.fact_extractor import FactExtractor
from memory.memory_manager import MemoryManager
from memory.memory_models import MemoryEvent, MemoryLevel, MemoryRecord, MemoryScope, MemoryType, RetrievalQuery
from memory.vector_store import embed_text


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in rows) + "\n", encoding="utf-8")


class MemoryQualityPlanTests(unittest.TestCase):
    def test_embedding_is_deterministic_across_processes(self) -> None:
        text = "Deterministic embedding text 42 hello"
        local = embed_text(text, dim=64)
        child_code = (
            "import json\n"
            "from memory.vector_store import embed_text\n"
            f"print(json.dumps(embed_text({text!r}, dim=64)))\n"
        )
        proc = subprocess.run(
            [sys.executable, "-c", child_code],
            capture_output=True,
            text=True,
            check=True,
        )
        remote = json.loads(proc.stdout.strip())
        self.assertEqual(local, remote)

    def test_fact_extractor_name_pattern_is_reasonable(self) -> None:
        extractor = FactExtractor()
        noisy = extractor.extract(text="hello there", metadata={}, speaker="user")
        clean = extractor.extract(text="my name is Alex", metadata={}, speaker="user")
        noisy_names = [x for x in noisy if str(x.key) == "identity_name"]
        clean_names = [x for x in clean if str(x.key) == "identity_name"]
        self.assertFalse(noisy_names)
        self.assertTrue(clean_names)

    def test_auto_migration_archives_v1_and_writes_v2_marker(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mmis_auto_migrate_") as tmpdir:
            root = Path(tmpdir)
            self._seed_migration_fixture(root)

            first = run_auto_migration(
                memory_dir=root,
                target_schema_version=2,
                auto_on_start=True,
                facts_scope="user_only",
            )
            self.assertTrue(bool(first.get("ran")))
            self.assertTrue(bool(first.get("success")))

            archive_dir = Path(str(first.get("archive_dir") or ""))
            self.assertTrue(archive_dir.exists())
            self.assertTrue((archive_dir / "events.jsonl").exists())
            self.assertTrue((archive_dir / "short_memory.json").exists())

            marker_path = root / "memory_v2" / "schema_marker.json"
            self.assertTrue(marker_path.exists())
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
            self.assertEqual(int(marker.get("schema_version") or 0), 2)

            state = json.loads((root / MIGRATION_STATE_FILE).read_text(encoding="utf-8"))
            self.assertEqual(int(state.get("schema_version") or 0), 2)
            self.assertEqual(str(state.get("mode") or ""), "clean_break_v2_only")

            second = run_auto_migration(
                memory_dir=root,
                target_schema_version=2,
                auto_on_start=True,
                facts_scope="user_only",
            )
            self.assertFalse(bool(second.get("ran")))
            self.assertEqual(str(second.get("reason") or ""), "up_to_date")

    def test_reindex_required_flow_when_embedding_fingerprint_changes(self) -> None:
        tmpdir = tempfile.mkdtemp(prefix="mmis_reindex_flow_")
        root = Path(tmpdir)
        try:
            manager = MemoryManager(root_dir=root)
            try:
                manager.ingest_event(
                    MemoryEvent(
                        role="user",
                        text="Memory V2 supports hybrid retrieval",
                        namespace="n1",
                        scope=MemoryScope.CONVERSATION,
                        memory_type=MemoryType.MESSAGE,
                    )
                )
            finally:
                manager.close()

            meta_path = root / "memory_v2" / "index_meta.json"
            payload = json.loads(meta_path.read_text(encoding="utf-8"))
            payload["embedding_fingerprint"] = "forced-mismatch"
            meta_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

            manager2 = MemoryManager(root_dir=root)
            try:
                first = manager2.retrieve(
                    RetrievalQuery(
                        query_text="hybrid retrieval",
                        namespace="n1",
                        scopes=[MemoryScope.CONVERSATION],
                    )
                )
                self.assertTrue(first.reindex_required)
                manager2.reindex_embeddings(namespace="n1", incremental=True)
                second = manager2.retrieve(
                    RetrievalQuery(
                        query_text="hybrid retrieval",
                        namespace="n1",
                        scopes=[MemoryScope.CONVERSATION],
                    )
                )
                self.assertFalse(second.reindex_required)
            finally:
                manager2.close()
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_memory_manager_rejects_local_backend_fail_fast(self) -> None:
        cfg = replace(load_config(force_reload=True), memory_backend="local")
        with tempfile.TemporaryDirectory(prefix="mmis_mm_local_backend_") as tmpdir:
            with patch("memory.memory_manager.load_config", return_value=cfg):
                with self.assertRaises(ValueError) as ctx:
                    MemoryManager(root_dir=Path(tmpdir))
        self.assertIn("Use 'chroma' or 'chromadb'", str(ctx.exception))

    def test_memory_manager_fails_when_chromadb_missing(self) -> None:
        cfg = replace(load_config(force_reload=True), memory_backend="chroma")
        real_import = __import__

        def _import_fail_chroma(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "chromadb":
                raise ModuleNotFoundError("No module named 'chromadb'")
            return real_import(name, globals, locals, fromlist, level)

        with tempfile.TemporaryDirectory(prefix="mmis_mm_no_chromadb_") as tmpdir:
            with patch("memory.memory_manager.load_config", return_value=cfg):
                with patch("builtins.__import__", side_effect=_import_fail_chroma):
                    with self.assertRaises(RuntimeError) as ctx:
                        MemoryManager(root_dir=Path(tmpdir))
        self.assertIn("chromadb is required", str(ctx.exception).lower())

    def test_no_startup_backfill_from_records_but_manual_reindex_populates_vectors(self) -> None:
        cfg = replace(load_config(force_reload=True), memory_backend="chroma")
        tmpdir = tempfile.mkdtemp(prefix="mmis_no_backfill_")
        root = Path(tmpdir)
        try:
            memory_v2 = root / "memory_v2"
            memory_v2.mkdir(parents=True, exist_ok=True)
            legacy_record = MemoryRecord(
                id="message:legacy_bootstrap_1",
                text="Legacy bootstrap memory text",
                memory_type=MemoryType.MESSAGE,
                level=MemoryLevel.L0_WORKING,
                scope=MemoryScope.CONVERSATION,
                namespace="n1",
                metadata={},
                embedding=None,
            )
            _write_json(memory_v2 / "records.json", {"records": [legacy_record.to_dict()]})

            with patch("memory.memory_manager.load_config", return_value=cfg):
                manager = MemoryManager(root_dir=root)
            try:
                before = manager._store.search(
                    query_text="Legacy bootstrap memory text",
                    top_k=4,
                    namespace="n1",
                    scopes=[MemoryScope.CONVERSATION],
                    include_stale=False,
                    metadata_filters={},
                )
                self.assertTrue(before)
                before_row = next((x for x in before if str(x.get("record").id) == legacy_record.id), None)
                self.assertIsNotNone(before_row)
                self.assertGreater(float(before_row.get("lexical_score") or 0.0), 0.0)
                self.assertEqual(float(before_row.get("semantic_score") or 0.0), 0.0)

                manager.reindex_embeddings(namespace="n1", incremental=False)
                after = manager._store.search(
                    query_text="Legacy bootstrap memory text",
                    top_k=4,
                    namespace="n1",
                    scopes=[MemoryScope.CONVERSATION],
                    include_stale=False,
                    metadata_filters={},
                )
                after_row = next((x for x in after if str(x.get("record").id) == legacy_record.id), None)
                self.assertIsNotNone(after_row)
                self.assertGreater(float(after_row.get("semantic_score") or 0.0), 0.0)
            finally:
                manager.close()
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_records_json_does_not_store_embeddings(self) -> None:
        cfg = replace(load_config(force_reload=True), memory_backend="chroma")
        tmpdir = tempfile.mkdtemp(prefix="mmis_records_no_embeddings_")
        root = Path(tmpdir)
        try:
            with patch("memory.memory_manager.load_config", return_value=cfg):
                manager = MemoryManager(root_dir=root)
            try:
                manager.ingest_event(
                    MemoryEvent(
                        role="user",
                        text="Store this memory without embedding in records json",
                        namespace="n1",
                        scope=MemoryScope.CONVERSATION,
                        memory_type=MemoryType.MESSAGE,
                    )
                )
            finally:
                manager.close()

            payload = json.loads((root / "memory_v2" / "records.json").read_text(encoding="utf-8"))
            rows = [dict(x) for x in list(payload.get("records") or []) if isinstance(x, dict)]
            self.assertTrue(rows)
            row = rows[0]
            self.assertTrue("embedding" not in row or row.get("embedding") is None)
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def _seed_migration_fixture(self, root: Path) -> None:
        _write_jsonl(
            root / "events.jsonl",
            [
                {
                    "event_id": "e1",
                    "ts": "2026-03-05T10:00:00+02:00",
                    "type": "user_message",
                    "payload": {"role": "user", "text": "I was born in 2003", "metadata": {"lang": "en"}},
                    "tags": [],
                }
            ],
        )
        _write_json(root / "short_memory.json", {"items": [], "rolling_summary": "", "rolling_summary_meta": {}})
        _write_json(root / "long_memory_docs.json", {"docs": []})
        _write_json(root / "vector_store.json", {"records": []})
        _write_json(root / "user_profile_store.json", {"profiles": {}, "versions": {}, "pending_facts": {}, "confirmed_facts": {}})
        _write_json(
            root / "assistant_profile_store.json",
            {"profiles": {"default": {"name": {"value": "Luna"}}}, "versions": {}, "pending_facts": {}, "confirmed_facts": {}},
        )
        _write_json(root / MIGRATION_STATE_FILE, {"schema_version": 1, "status": "legacy"})


if __name__ == "__main__":
    unittest.main()

