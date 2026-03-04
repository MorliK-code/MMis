from __future__ import annotations

from _output_utils import enable_unittest_json_output

enable_unittest_json_output()

import unittest

from modules.internet.content_cleaner import clean_web_content
from modules.internet.scraper import WebScraper


class _StubScraper(WebScraper):
    def __init__(self, html_text: str):
        super().__init__(respect_robots=False, retries=0, use_disk_cache=False)
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


if __name__ == "__main__":
    unittest.main()
