from __future__ import annotations

from _output_utils import enable_unittest_json_output
enable_unittest_json_output()

import tempfile
import unittest
from pathlib import Path

from memory.event_store import EventStore
from memory.fact_extractor import FactExtractor
from memory.long_memory import LongMemory
from memory.memory_manager import MemoryManager
from memory.profile_store import AssistantProfileStore, UserProfileStore
from memory.short_memory import ShortMemory
from memory.vector_store import VectorStore


class MemoryEntityTests(unittest.TestCase):
    def test_ingest_extracts_entities_and_entity_tags(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manager = MemoryManager(
                short_memory=ShortMemory(path=root / "short.json", autosave=False),
                long_memory=LongMemory(path=root / "long.json"),
                vector_store=VectorStore(path=root / "vectors.json"),
                fact_extractor=FactExtractor(),
                user_profile_store=UserProfileStore(path=root / "user_profile.json"),
                assistant_profile_store=AssistantProfileStore(path=root / "assistant_profile.json"),
                event_store=EventStore(path=root / "events.jsonl"),
            )

            manager.ingest_message(
                "user",
                "Docker compose fails in VS Code on Windows path C:\\Users\\dev and RTX 3060",
                metadata={"lang": "en"},
            )

            tail = manager.short_memory.tail(1)
            self.assertEqual(len(tail), 1)
            row = dict(tail[0] or {})
            tags = {str(x) for x in list(row.get("tags") or [])}
            self.assertIn("topic_docker", tags)
            self.assertIn("topic_vscode", tags)
            self.assertIn("topic_windows", tags)
            self.assertIn("entity_software_docker", tags)

            meta = dict(row.get("meta") or {})
            entities = dict(meta.get("entities") or {})
            self.assertIn("software", entities)
            self.assertIn("Docker", list(entities.get("software") or []))

            hits = manager.vector_store.query_text("docker vscode windows", k=3)
            self.assertTrue(hits)
            self.assertIsInstance(dict(hits[0][2] or {}).get("entities"), dict)


if __name__ == "__main__":
    unittest.main()
