from __future__ import annotations

from _output_utils import enable_unittest_json_output

enable_unittest_json_output()

import json
import tempfile
import unittest
from pathlib import Path
from urllib.error import HTTPError
from unittest.mock import patch

from modules.internet.search import SearchClient, SearchResult, _rank_results


class SearchSearxngStrictTests(unittest.TestCase):
    def test_strict_mode_never_uses_duckduckgo_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            client = SearchClient(
                endpoint="http://searxng:8080/search?format=json",
                strict_endpoint=True,
                cache_dir=Path(tmpdir),
            )
            client._search_endpoint = lambda _q: []  # type: ignore[method-assign]

            def _should_not_call(_q: str):
                raise AssertionError("duckduckgo fallback should not run")

            client._search_duckduckgo = _should_not_call  # type: ignore[method-assign]
            out = client.search("strict endpoint test", k=3)
            self.assertEqual(out, [])

    def test_parses_searxng_results_payload(self) -> None:
        payload = {
            "results": [
                {
                    "title": "Rate update",
                    "url": "https://example.com/rates",
                    "content": "USD to UAH update",
                    "publishedDate": "2026-03-04T10:00:00Z",
                }
            ]
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            client = SearchClient(
                endpoint="http://searxng:8080/search?format=json",
                strict_endpoint=True,
                cache_dir=Path(tmpdir),
            )
            with patch("modules.internet.search._http_get", return_value=json.dumps(payload, ensure_ascii=False)):
                out = client.search("usd uah", k=1)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].url, "https://example.com/rates")
        self.assertEqual(out[0].source, "example.com")
        self.assertEqual(out[0].published_date, "2026-03-04")
        self.assertEqual(str((out[0].raw or {}).get("provider") or ""), "searxng")

    def test_volatile_bypasses_cache(self) -> None:
        payload_a = {
            "results": [
                {
                    "title": "A",
                    "url": "https://example.com/a",
                    "content": "first",
                    "publishedDate": "2026-03-04T09:00:00Z",
                }
            ]
        }
        payload_b = {
            "results": [
                {
                    "title": "B",
                    "url": "https://example.com/b",
                    "content": "second",
                    "publishedDate": "2026-03-04T09:01:00Z",
                }
            ]
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            client = SearchClient(
                endpoint="http://searxng:8080/search?format=json",
                strict_endpoint=True,
                cache_dir=Path(tmpdir),
            )
            with patch("modules.internet.search._http_get", side_effect=[json.dumps(payload_a), json.dumps(payload_b)]) as mocked_get:
                first = client.search("usd uah", k=1, query_intent="fx_rate")
                second = client.search("usd uah", k=1, volatile=True, query_intent="fx_rate")
        self.assertEqual(len(first), 1)
        self.assertEqual(len(second), 1)
        self.assertEqual(first[0].url, "https://example.com/a")
        self.assertEqual(second[0].url, "https://example.com/b")
        self.assertEqual(mocked_get.call_count, 2)

    def test_fx_market_first_ranking_prefers_market_source(self) -> None:
        items = [
            SearchResult(
                title="USD UAH rate",
                snippet="рынок курс usd uah",
                url="https://bank.gov.ua/rate",
                source="bank.gov.ua",
                published_date="2026-03-04",
            ),
            SearchResult(
                title="USD UAH rate",
                snippet="рынок курс usd uah",
                url="https://minfin.com.ua/currency",
                source="minfin.com.ua",
                published_date="2026-03-04",
            ),
        ]
        ranked = _rank_results(items, query="курс доллара usd uah", recency_days=1, query_intent="fx_rate")
        self.assertEqual(ranked[0].source, "minfin.com.ua")

    def test_json_403_falls_back_to_searx_html_results(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            client = SearchClient(
                endpoint="http://127.0.0.1:8080/search?format=json",
                strict_endpoint=True,
                cache_dir=Path(tmpdir),
            )
            html_payload = (
                '<article class="result result-default">'
                '<a href="https://minfin.com.ua/currency/usd/">u</a>'
                "<h3>USD/UAH today</h3>"
                '<p class="content">Курс доллара сегодня 42.10</p>'
                "</article>"
            )

            with patch("modules.internet.search._http_get", side_effect=[HTTPError("x", 403, "forbidden", None, None), html_payload]):
                out = client.search("usd uah", k=1, query_intent="fx_rate")
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].source, "minfin.com.ua")
        self.assertEqual(str((out[0].raw or {}).get("format") or ""), "html_fallback")


if __name__ == "__main__":
    unittest.main()
