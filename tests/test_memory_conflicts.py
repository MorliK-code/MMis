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


class MemoryConflictTests(unittest.TestCase):
    def test_fact_pending_then_confirmed_by_repeat(self) -> None:
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

            manager.ingest_message("user", "I was born in 2003", metadata={"lang": "en"})
            pending = manager.user_profile_store.get_pending_facts(profile_id="default")
            self.assertIn("birth_year", pending)

            manager.ingest_message("user", "I was born in 2003", metadata={"lang": "en"})
            current = manager.user_profile_store.get("birth_year", profile_id="default") or {}
            confirmed = manager.user_profile_store.get_confirmed_facts(profile_id="default")

            self.assertEqual(str(current.get("value")), "2003")
            self.assertFalse(bool(current.get("needs_confirmation", False)))
            self.assertEqual(str(dict(confirmed.get("birth_year") or {}).get("value")), "2003")

    def test_pending_fact_can_be_confirmed_by_yes_message_with_context(self) -> None:
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

            manager.ingest_message("user", "I was born in 2003", metadata={"lang": "en"})
            manager.ingest_message("assistant", "You were born in 2003, right?", metadata={"lang": "en"})
            manager.ingest_message("user", "yes", metadata={"lang": "en"})

            current = manager.user_profile_store.get("birth_year", profile_id="default") or {}
            history = manager.user_profile_store.history("birth_year", profile_id="default")

            self.assertEqual(str(current.get("value")), "2003")
            self.assertTrue(any(str(row.get("status") or "") == "confirmed_by_user" for row in history))

    def test_yes_without_context_does_not_confirm_pending_fact(self) -> None:
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

            manager.ingest_message("user", "I was born in 2003", metadata={"lang": "en"})
            manager.ingest_message("user", "yes", metadata={"lang": "en"})
            confirmed = manager.user_profile_store.get_confirmed_facts(profile_id="default")
            self.assertNotIn("birth_year", confirmed)

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

            manager.ingest_message("user", "I was born in 2003", metadata={"lang": "en"})
            manager.ingest_message("user", "I was born in 2003", metadata={"lang": "en"})
            manager.ingest_message("user", "I was born in 2004", metadata={"lang": "en"})
            manager.ingest_message("user", "I was born in 2004", metadata={"lang": "en"})

            current = manager.user_profile_store.get("birth_year", profile_id="default") or {}
            history = manager.user_profile_store.history("birth_year", profile_id="default")

            self.assertEqual(str(current.get("value")), "2004")
            self.assertGreaterEqual(len(history), 2)


if __name__ == "__main__":
    unittest.main()

