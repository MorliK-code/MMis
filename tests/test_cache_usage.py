from __future__ import annotations

from _output_utils import enable_unittest_json_output
enable_unittest_json_output()

import tempfile
import unittest
from pathlib import Path

from modules.internet.search import SearchClient, SearchResult
from utils.cache import DiskTTLCache


class CacheUsageTests(unittest.TestCase):
    def test_disk_ttl_cache_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            c1 = DiskTTLCache(namespace="unit", root=root, default_ttl_s=300, enabled=True)
            c1.set("k1", {"x": 1, "y": "ok"})

            c2 = DiskTTLCache(namespace="unit", root=root, default_ttl_s=300, enabled=True)
            row = c2.get("k1")
            self.assertIsInstance(row, dict)
            self.assertEqual(row.get("x"), 1)
            self.assertEqual(row.get("y"), "ok")

    def test_search_client_uses_disk_cache_between_instances(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            q = "mmis cache check"
            expected = SearchResult(
                title="cache title",
                snippet="cache snippet",
                url="https://example.com/a",
                source="example.com",
                published_date="",
                score=0.0,
                raw={"provider": "unit"},
            )

            c1 = SearchClient(
                endpoint="http://searxng:8080/search?format=json",
                strict_endpoint=True,
                cache_ttl_s=600,
                cache_dir=root,
                use_disk_cache=True,
            )
            c1._search_endpoint = lambda _q: [expected]  # type: ignore[method-assign]
            first = c1.search(q, k=1)
            self.assertEqual(len(first), 1)
            self.assertEqual(first[0].url, expected.url)

            c2 = SearchClient(
                endpoint="http://searxng:8080/search?format=json",
                strict_endpoint=True,
                cache_ttl_s=600,
                cache_dir=root,
                use_disk_cache=True,
            )

            def _should_not_call(*_args, **_kwargs):
                raise AssertionError("network call should not happen on cache hit")

            c2._search_endpoint = _should_not_call  # type: ignore[method-assign]
            second = c2.search(q, k=1)
            self.assertEqual(len(second), 1)
            self.assertEqual(second[0].url, expected.url)


if __name__ == "__main__":
    unittest.main()

