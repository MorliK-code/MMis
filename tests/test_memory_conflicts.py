from __future__ import annotations

from _output_utils import enable_unittest_json_output

enable_unittest_json_output()

import tempfile
import unittest
from pathlib import Path

from memory.memory_manager import MemoryManager
from memory.memory_models import DebugRequest, MemoryEvent, MemoryScope, MemoryType


def _fact_rows(snapshot: dict, *, canonical_key: str) -> list[dict]:
    rows: list[dict] = []
    for item in list(snapshot.get("items") or []):
        row = dict(item or {})
        if str(row.get("memory_type") or "") != "fact":
            continue
        meta = dict(row.get("metadata") or {})
        if str(meta.get("canonical_key") or "") != canonical_key:
            continue
        rows.append(row)
    return rows


class MemoryConflictTests(unittest.TestCase):
    def test_conflicting_fact_marks_previous_as_superseded(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = MemoryManager(root_dir=Path(tmpdir))
            try:
                manager.ingest_event(
                    MemoryEvent(
                        role="user",
                        text="I prefer Vim",
                        namespace="default",
                        scope=MemoryScope.CONVERSATION,
                        memory_type=MemoryType.MESSAGE,
                    )
                )
                manager.ingest_event(
                    MemoryEvent(
                        role="user",
                        text="I prefer Emacs",
                        namespace="default",
                        scope=MemoryScope.CONVERSATION,
                        memory_type=MemoryType.MESSAGE,
                    )
                )

                snapshot = manager.debug_snapshot(DebugRequest(namespace="default", limit=200))
                rows = _fact_rows(snapshot, canonical_key="user.preference")
                self.assertTrue(rows)

                active = [x for x in rows if str(x.get("status") or "") == "active"]
                superseded = [x for x in rows if str(x.get("status") or "") == "superseded"]
                self.assertTrue(active)
                self.assertTrue(superseded)
                self.assertTrue(any("emacs" in str(x.get("text") or "").lower() for x in active))
                self.assertTrue(any("vim" in str(x.get("text") or "").lower() for x in superseded))
            finally:
                manager.close()

    def test_repeating_same_fact_does_not_create_superseded_row(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = MemoryManager(root_dir=Path(tmpdir))
            try:
                for _ in range(2):
                    manager.ingest_event(
                        MemoryEvent(
                            role="user",
                            text="I prefer Python",
                            namespace="default",
                            scope=MemoryScope.CONVERSATION,
                            memory_type=MemoryType.MESSAGE,
                        )
                    )
                snapshot = manager.debug_snapshot(DebugRequest(namespace="default", limit=200))
                rows = _fact_rows(snapshot, canonical_key="user.preference")
                self.assertTrue(rows)
                superseded = [x for x in rows if str(x.get("status") or "") == "superseded"]
                self.assertFalse(superseded)
            finally:
                manager.close()

    def test_temporary_fact_is_stored_in_temporary_scope_with_ttl(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = MemoryManager(root_dir=Path(tmpdir))
            try:
                manager.ingest_event(
                    MemoryEvent(
                        role="user",
                        text="For now, temporarily use local cache.",
                        namespace="default",
                        scope=MemoryScope.CONVERSATION,
                        memory_type=MemoryType.MESSAGE,
                    )
                )
                snapshot = manager.debug_snapshot(DebugRequest(namespace="default", limit=200))
                rows = _fact_rows(snapshot, canonical_key="user.temporary_fact")
                self.assertTrue(rows)
                self.assertTrue(all(str(x.get("scope") or "") == "temporary" for x in rows))
                self.assertTrue(all(x.get("expires_at") is not None for x in rows))
            finally:
                manager.close()


if __name__ == "__main__":
    unittest.main()

