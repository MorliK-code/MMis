from __future__ import annotations

from _output_utils import enable_unittest_json_output

enable_unittest_json_output()

import unittest
import tempfile
import json
from pathlib import Path

from modules.internet.content_cleaner import clean_web_content
from modules.internet.scraper import WebScraper


class _StubScraper(WebScraper):
    def __init__(self, html_text: str, *, cache_dir: str | Path | None = None, use_disk_cache: bool = False):
        super().__init__(respect_robots=False, retries=0, use_disk_cache=use_disk_cache, cache_dir=cache_dir)
        self._html_text = html_text

    def _fetch_with_meta(self, url: str) -> tuple[str, int, str, str]:
        return self._html_text, 200, "text/html", url


class WebContentCleanerTests(unittest.TestCase):
    def test_cleaner_removes_cookie_ads_and_comments_noise(self) -> None:
        html = """
        <html>
          <head><title>Demo Article</title></head>
          <body>
            <div id="cookie-banner">We use cookies. Accept all cookies</div>
            <div class="ad-slot promo">Sponsored block</div>
            <article>
              <h1>Main topic</h1>
              <p>Useful paragraph with real content for the model.</p>
            </article>
            <section class="comments">Leave a comment</section>
          </body>
        </html>
        """
        cleaned = clean_web_content(html, max_chars=2000, min_chars=40)
        text_low = cleaned.text.lower()
        self.assertIn("useful paragraph", text_low)
        self.assertNotIn("accept all cookies", text_low)
        self.assertNotIn("leave a comment", text_low)
        self.assertGreaterEqual(cleaned.removed_blocks, 2)
        self.assertEqual(cleaned.title, "Demo Article")

    def test_scrape_includes_cleaning_metadata(self) -> None:
        html = """
        <html>
          <head><title>Metadata Demo</title></head>
          <body>
            <div class="cookie">cookie consent</div>
            <article><p>Important clean body text.</p></article>
          </body>
        </html>
        """
        scraper = _StubScraper(html)
        out = scraper.scrape("https://example.com/page")
        self.assertTrue(out.text)
        self.assertIn("clean_method", out.metadata)
        self.assertIn("removed_blocks", out.metadata)
        self.assertIn("raw_len", out.metadata)
        self.assertIn("clean_len", out.metadata)
        self.assertGreaterEqual(int(out.metadata.get("raw_len") or 0), int(out.metadata.get("clean_len") or 0))

    def test_scraper_cache_writes_v2_dual_payload(self) -> None:
        html = "<html><head><title>T</title></head><body><article><p>Body text</p></article></body></html>"
        with tempfile.TemporaryDirectory(prefix="mmis_scraper_cache_") as tmpdir:
            scraper = _StubScraper(html, cache_dir=tmpdir, use_disk_cache=True)
            _ = scraper.scrape("https://example.com/test")
            files = list((Path(tmpdir) / "internet_scraper_fetch").glob("*.json"))
            self.assertTrue(files)
            payload = json.loads(files[0].read_text(encoding="utf-8-sig"))
            value = dict(payload.get("value") or {})
            self.assertEqual(str(value.get("schema") or ""), "internet_scraper_fetch_v2")
            raw = dict(value.get("raw") or {})
            cleaned = dict(value.get("cleaned") or {})
            self.assertIn("html_truncated", raw)
            self.assertIn("full_len", raw)
            self.assertIn("truncated", raw)
            self.assertIn("sha1", raw)
            self.assertTrue(str(cleaned.get("text") or "").strip())
            self.assertIn("clean_method", cleaned)

    def test_scraper_cache_upgrades_v1_to_v2_on_scrape(self) -> None:
        html = "<html><head><title>Legacy</title></head><body><article><p>Legacy body</p></article></body></html>"
        url = "https://example.com/legacy"
        with tempfile.TemporaryDirectory(prefix="mmis_scraper_cache_") as tmpdir:
            scraper = _StubScraper(html, cache_dir=tmpdir, use_disk_cache=True)
            cache_key = f"{url}|{scraper.max_bytes}|{int(scraper.respect_robots)}"
            scraper._disk_cache.set(
                cache_key,
                {
                    "html": html,
                    "status": 200,
                    "content_type": "text/html",
                    "final_url": url,
                },
                ttl_s=600,
            )
            out = scraper.scrape(url)
            self.assertIn("Legacy", out.title)
            upgraded = dict(scraper._disk_cache.get(cache_key) or {})
            self.assertEqual(str(upgraded.get("schema") or ""), "internet_scraper_fetch_v2")
            self.assertTrue(str(dict(upgraded.get("cleaned") or {}).get("text") or "").strip())


if __name__ == "__main__":
    unittest.main()
