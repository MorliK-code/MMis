from __future__ import annotations

import html
import re
import time
import urllib.robotparser
from dataclasses import asdict, dataclass, field
from typing import Any
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen


_USER_AGENT = "MMisBot/1.0 (+https://local.mmis)"


@dataclass(frozen=True)
class LinkRef:
    text: str
    url: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ScrapeResult:
    url: str
    final_url: str
    status_code: int
    content_type: str
    title: str
    headings: list[str] = field(default_factory=list)
    text: str = ""
    links: list[LinkRef] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "url": self.url,
            "final_url": self.final_url,
            "status_code": self.status_code,
            "content_type": self.content_type,
            "title": self.title,
            "headings": list(self.headings),
            "text": self.text,
            "links": [x.to_dict() for x in self.links],
            "metadata": dict(self.metadata or {}),
        }


class WebScraper:
    def __init__(
        self,
        *,
        timeout_s: int = 12,
        retries: int = 1,
        max_bytes: int = 1_500_000,
        respect_robots: bool = True,
    ):
        self.timeout_s = max(3, int(timeout_s))
        self.retries = max(0, int(retries))
        self.max_bytes = max(50_000, int(max_bytes))
        self.respect_robots = bool(respect_robots)

    def fetch(self, url: str) -> str:
        html_text, _, _, _ = self._fetch_with_meta(url)
        return html_text

    def extract_readable(self, html_text: str) -> tuple[str, str, list[str]]:
        src = str(html_text or "")
        title = _first_group(r"<title[^>]*>(.*?)</title>", src)
        headings = _all_groups(r"<(h1|h2|h3)[^>]*>(.*?)</\1>", src, use_group=2)

        cleaned = re.sub(r"<!--.*?-->", " ", src, flags=re.S)
        cleaned = re.sub(r"<script[^>]*>.*?</script>", " ", cleaned, flags=re.S | re.I)
        cleaned = re.sub(r"<style[^>]*>.*?</style>", " ", cleaned, flags=re.S | re.I)
        cleaned = re.sub(r"<noscript[^>]*>.*?</noscript>", " ", cleaned, flags=re.S | re.I)
        cleaned = re.sub(r"<[^>]+>", " ", cleaned)
        cleaned = html.unescape(cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()

        return cleaned, title, headings

    def extract_links(self, html_text: str, base_url: str | None = None) -> list[LinkRef]:
        src = str(html_text or "")
        out: list[LinkRef] = []
        seen: set[str] = set()

        for href, text in re.findall(r"<a[^>]*href=[\"']([^\"']+)[\"'][^>]*>(.*?)</a>", src, flags=re.I | re.S):
            url = html.unescape(href.strip())
            if not url:
                continue
            if base_url:
                url = urljoin(base_url, url)
            if not url.startswith(("http://", "https://")):
                continue
            key = url.lower().strip()
            if key in seen:
                continue
            seen.add(key)
            out.append(LinkRef(text=_clean_text(text), url=url))
        return out

    def scrape(self, url: str) -> ScrapeResult:
        started = time.perf_counter()
        html_text, status_code, content_type, final_url = self._fetch_with_meta(url)
        text, title, headings = self.extract_readable(html_text)
        links = self.extract_links(html_text, base_url=final_url)

        return ScrapeResult(
            url=url,
            final_url=final_url,
            status_code=status_code,
            content_type=content_type,
            title=title,
            headings=headings,
            text=text,
            links=links,
            metadata={
                "elapsed_ms": (time.perf_counter() - started) * 1000.0,
                "text_len": len(text),
                "links_count": len(links),
            },
        )

    def _fetch_with_meta(self, url: str) -> tuple[str, int, str, str]:
        target = str(url or "").strip()
        if not target:
            raise ValueError("url is empty")
        parsed = urlparse(target)
        if parsed.scheme not in {"http", "https"}:
            raise ValueError("only http/https URLs are allowed")

        if self.respect_robots and not self._is_allowed_by_robots(target):
            raise PermissionError(f"robots.txt disallows URL: {target}")

        req = Request(target, headers={"User-Agent": _USER_AGENT})

        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                with urlopen(req, timeout=self.timeout_s) as resp:
                    payload = resp.read(self.max_bytes + 1)
                    if len(payload) > self.max_bytes:
                        raise ValueError(f"response exceeds max_bytes={self.max_bytes}")

                    final_url = str(getattr(resp, "url", target) or target)
                    status = int(getattr(resp, "status", 200) or 200)
                    content_type = str(resp.headers.get("Content-Type", ""))
                    html_text = payload.decode("utf-8", errors="ignore")
                    return html_text, status, content_type, final_url
            except Exception as exc:
                last_error = exc
                if attempt < self.retries:
                    time.sleep(0.25 * (attempt + 1))
                    continue
                break

        raise RuntimeError(f"fetch failed: {last_error}")

    def _is_allowed_by_robots(self, url: str) -> bool:
        parsed = urlparse(url)
        robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
        parser = urllib.robotparser.RobotFileParser()
        parser.set_url(robots_url)
        try:
            parser.read()
            return bool(parser.can_fetch(_USER_AGENT, url))
        except Exception:
            # If robots fetch fails, fail-open to avoid hard blocking due to connectivity issues.
            return True


def fetch(url: str) -> str:
    return _DEFAULT_SCRAPER.fetch(url)


def extract_readable(html_text: str) -> tuple[str, str, list[str]]:
    return _DEFAULT_SCRAPER.extract_readable(html_text)


def extract_links(html_text: str) -> list[LinkRef]:
    return _DEFAULT_SCRAPER.extract_links(html_text)


def scrape(url: str) -> dict:
    return _DEFAULT_SCRAPER.scrape(url).to_dict()


def _first_group(pattern: str, text: str) -> str:
    m = re.search(pattern, text, flags=re.I | re.S)
    if not m:
        return ""
    return _clean_text(m.group(1))


def _all_groups(pattern: str, text: str, *, use_group: int) -> list[str]:
    out: list[str] = []
    for m in re.finditer(pattern, text, flags=re.I | re.S):
        out.append(_clean_text(m.group(use_group)))
    return [x for x in out if x]


def _clean_text(value: str) -> str:
    txt = re.sub(r"<[^>]+>", " ", str(value or ""))
    txt = html.unescape(txt)
    txt = re.sub(r"\s+", " ", txt)
    return txt.strip()


_DEFAULT_SCRAPER = WebScraper()
