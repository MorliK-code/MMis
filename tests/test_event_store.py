from __future__ import annotations

from _output_utils import enable_unittest_json_output
enable_unittest_json_output()

import tempfile
import unittest
from pathlib import Path

from memory.event_store import EventStore


class EventStoreTests(unittest.TestCase):
    def test_append_get_range_persist(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "events.jsonl"
            store = EventStore(path=path)
            e1 = store.append({"type": "user_message", "payload": {"text": "hello"}})
            _ = store.append({"type": "assistant_message", "payload": {"text": "hi"}})

            self.assertEqual(store.get(e1["event_id"])["payload"]["text"], "hello")
            self.assertEqual(len(store.last(2)), 2)
            self.assertGreaterEqual(len(store.range(limit=2)), 1)

            store2 = EventStore(path=path)
            self.assertGreaterEqual(len(store2.last(10)), 2)


if __name__ == "__main__":
    unittest.main()

