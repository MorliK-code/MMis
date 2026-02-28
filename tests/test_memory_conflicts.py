from __future__ import annotations

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


class MemoryConflictTests(unittest.TestCase):
    def test_birth_year_update_overwrites_current_and_keeps_history(self) -> None:
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

            manager.ingest_message("user", "I was born in 2003", {"lang": "en"})
            manager.ingest_message("user", "I was born in 2004", {"lang": "en"})

            current = manager.user_profile_store.get("birth_year", profile_id="default") or {}
            history = manager.user_profile_store.history("birth_year", profile_id="default")

            self.assertEqual(str(current.get("value")), "2004")
            self.assertGreaterEqual(len(history), 2)


if __name__ == "__main__":
    unittest.main()
