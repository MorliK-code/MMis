from __future__ import annotations

import unittest

from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.memory_inspector_router import create_memory_inspector_router


class MemoryInspectorRouterTests(unittest.TestCase):
    def test_inspector_page_is_served_without_cache_and_with_clarified_labels(self) -> None:
        app = FastAPI()
        app.include_router(create_memory_inspector_router(memory_core=object()))

        client = TestClient(app)
        response = client.get("/memory-core")

        self.assertEqual(response.status_code, 200)
        self.assertIn("no-store", response.headers.get("cache-control", ""))
        self.assertIn("Финально завершено", response.text)
        self.assertIn("Попыток в сессии", response.text)


if __name__ == "__main__":
    unittest.main()
