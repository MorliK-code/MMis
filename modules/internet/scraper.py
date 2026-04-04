from __future__ import annotations

import datetime as dt
import hashlib
import html
import re
import time
import urllib.robotparser
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen

from modules.internet.content_cleaner import clean_web_content
from utils.cache import DiskTTLCache
from utils.logger import get_logger, log_json


_USER_AGENT = "MMisBot/1.0 (+https://local.mmis)"
LOGGER = get_logger(__name__)
_CACHE_SCHEMA_V2 = "internet_scraper_fetch_v2"
_DEFAULT_RAW_HTML_TRUNCATE_LIMIT = 300_000


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
        cache_ttl_s: int = 1800,
        cache_dir: str | Path | None = None,
        use_disk_cache: bool = True,
        clean_max_chars: int = 4000,
        clean_min_chars: int = 200,
        clean_language_hint: str = "",
        raw_html_truncate_limit: int = _DEFAULT_RAW_HTML_TRUNCATE_LIMIT,
    ):
        self.timeout_s = max(3, int(timeout_s))
        self.retries = max(0, int(retries))
        self.max_bytes = max(50_000, int(max_bytes))
        self.respect_robots = bool(respect_robots)
        self.cache_ttl_s = max(60, int(cache_ttl_s))
        self.clean_max_chars = max(256, int(clean_max_chars))
        self.clean_min_chars = max(40, int(clean_min_chars))
        self.clean_language_hint = str(clean_language_hint or "").strip()
        self.raw_html_truncate_limit = max(32_768, int(raw_html_truncate_limit))
        self._disk_cache = DiskTTLCache(
            namespace="internet_scraper_fetch",
            root=cache_dir,
            default_ttl_s=self.cache_ttl_s,
            max_memory_entries=512,
            enabled=bool(use_disk_cache),
        )
        try:
            self._disk_cache.purge_expired(max_files=800)
        except Exception as exc:
            LOGGER.debug("scraper cache purge skipped: %s", exc)

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
        target = str(url or "").strip()
        cache_key = self._cache_key(target)

        cached = self._disk_cache.get(cache_key)
        if isinstance(cached, dict):
            cached_result = self._read_cached_cleaned(cached, request_url=target)
            if cached_result is not None:
                self._log_scrape_event("scraper_fetch_ok", target=target, result=cached_result, cache_hit=True)
                return cached_result

        html_text, status_code, content_type, final_url = self._fetch_with_meta(url)
        _, fallback_title, headings = self.extract_readable(html_text)
        clean = clean_web_content(
            html_text,
            base_url=final_url,
            max_chars=self.clean_max_chars,
            min_chars=self.clean_min_chars,
            language_hint=self.clean_language_hint,
        )
        text = str(clean.text or "")
        title = str(clean.title or fallback_title or "").strip()
        links = self.extract_links(html_text, base_url=final_url)
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        result = ScrapeResult(
            url=url,
            final_url=final_url,
            status_code=status_code,
            content_type=content_type,
            title=title,
            headings=headings,
            text=text,
            links=links,
            metadata={
                "elapsed_ms": elapsed_ms,
                "text_len": len(text),
                "links_count": len(links),
                "clean_method": str(clean.method or ""),
                "removed_blocks": int(clean.removed_blocks),
                "raw_len": int(clean.raw_len),
                "clean_len": int(clean.clean_len),
            },
        )
        cleaned_payload = {
            "title": title,
            "text": text,
            "headings": list(headings or []),
            "clean_method": str(clean.method or ""),
            "removed_blocks": int(clean.removed_blocks),
            "raw_len": int(clean.raw_len),
            "clean_len": int(clean.clean_len),
            "links": [x.to_dict() for x in list(links or [])],
            "elapsed_ms": float(elapsed_ms),
        }
        try:
            self._disk_cache.set(
                cache_key,
                self._build_cache_payload_v2(
                    request_url=target,
                    final_url=final_url,
                    status=status_code,
                    content_type=content_type,
                    html_text=html_text,
                    cleaned=cleaned_payload,
                ),
                ttl_s=self.cache_ttl_s,
            )
        except Exception as exc:
            LOGGER.debug("scraper disk cache set(cleaned) failed: %s", exc)
        self._log_scrape_event("scraper_fetch_ok", target=target, result=result, cache_hit=False)
        return result

    def _fetch_with_meta(self, url: str) -> tuple[str, int, str, str]:
        target = str(url or "").strip()
        if not target:
            raise ValueError("url is empty")
        parsed = urlparse(target)
        if parsed.scheme not in {"http", "https"}:
            raise ValueError("only http/https URLs are allowed")

        cache_key = self._cache_key(target)
        cached = self._disk_cache.get(cache_key)
        normalized_cached = self._read_cached_entry(cached, request_url=target) if isinstance(cached, dict) else None
        if normalized_cached is not None:
            html_text = str(normalized_cached.get("html") or "")
            status = int(normalized_cached.get("status") or 200)
            content_type = str(normalized_cached.get("content_type") or "")
            final_url = str(normalized_cached.get("final_url") or target)
            if html_text:
                LOGGER.debug("scraper cache hit disk url=%s schema=%s", target, normalized_cached.get("schema"))
                return html_text, status, content_type, final_url

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
                    try:
                        self._disk_cache.set(
                            cache_key,
                            self._build_cache_payload_v2(
                                request_url=target,
                                final_url=final_url,
                                status=status,
                                content_type=content_type,
                                html_text=html_text,
                            ),
                            ttl_s=self.cache_ttl_s,
                        )
                    except Exception as exc:
                        LOGGER.debug("scraper disk cache set failed: %s", exc)
                    return html_text, status, content_type, final_url
            except Exception as exc:
                last_error = exc
                if attempt < self.retries:
                    time.sleep(0.25 * (attempt + 1))
                    continue
                break

        raise RuntimeError(f"fetch failed: {last_error}")

    def _cache_key(self, target: str) -> str:
        return f"{target}|{self.max_bytes}|{int(self.respect_robots)}"

    @staticmethod
    def _utc_now_iso() -> str:
        return dt.datetime.now(dt.timezone.utc).isoformat()

    def _truncate_raw_html(self, html_text: str) -> tuple[str, bool]:
        src = str(html_text or "")
        if len(src) <= self.raw_html_truncate_limit:
            return src, False
        return src[: self.raw_html_truncate_limit], True

    @staticmethod
    def _sha1_text(value: str) -> str:
        src = str(value or "")
        if not src:
            return ""
        return hashlib.sha1(src.encode("utf-8", errors="ignore")).hexdigest()

    def _build_cache_payload_v2(
        self,
        *,
        request_url: str,
        final_url: str,
        status: int,
        content_type: str,
        html_text: str,
        cleaned: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        raw_html = str(html_text or "")
        html_truncated, truncated = self._truncate_raw_html(raw_html)
        return {
            "schema": _CACHE_SCHEMA_V2,
            "request_url": str(request_url or ""),
            "final_url": str(final_url or request_url or ""),
            "fetched_at": self._utc_now_iso(),
            "http": {
                "status": int(status or 200),
                "content_type": str(content_type or ""),
            },
            "raw": {
                "html_truncated": html_truncated,
                "full_len": int(len(raw_html)),
                "truncated": bool(truncated),
                "sha1": self._sha1_text(raw_html),
            },
            "cleaned": dict(cleaned or {}),
        }

    def _read_cached_entry(self, cached: dict[str, Any], *, request_url: str) -> dict[str, Any] | None:
        row = dict(cached or {})
        schema = str(row.get("schema") or "").strip().lower()
        if schema == _CACHE_SCHEMA_V2:
            http = dict(row.get("http") or {})
            raw = dict(row.get("raw") or {})
            html_text = str(raw.get("html_truncated") or "")
            if not html_text:
                return None
            return {
                "schema": _CACHE_SCHEMA_V2,
                "html": html_text,
                "status": int(http.get("status") or 200),
                "content_type": str(http.get("content_type") or ""),
                "final_url": str(row.get("final_url") or row.get("request_url") or request_url),
            }

        # Backward compatibility with legacy v1 cache payload.
        html_text = str(row.get("html") or "")
        if not html_text:
            return None
        return {
            "schema": "internet_scraper_fetch_v1",
            "html": html_text,
            "status": int(row.get("status") or 200),
            "content_type": str(row.get("content_type") or ""),
            "final_url": str(row.get("final_url") or request_url),
        }

    def _read_cached_cleaned(self, cached: dict[str, Any], *, request_url: str) -> ScrapeResult | None:
        row = dict(cached or {})
        schema = str(row.get("schema") or "").strip().lower()
        if schema != _CACHE_SCHEMA_V2:
            return None
        cleaned = dict(row.get("cleaned") or {})
        text = str(cleaned.get("text") or "")
        if not text:
            return None
        http = dict(row.get("http") or {})
        links_data = list(cleaned.get("links") or [])
        links: list[LinkRef] = []
        for item in links_data:
            if not isinstance(item, dict):
                continue
            link_url = str(item.get("url") or "").strip()
            if not link_url:
                continue
            links.append(LinkRef(text=str(item.get("text") or ""), url=link_url))
        headings = [str(x) for x in list(cleaned.get("headings") or []) if str(x).strip()]
        final_url = str(row.get("final_url") or row.get("request_url") or request_url)
        raw = dict(row.get("raw") or {})
        metadata = {
            "elapsed_ms": float(cleaned.get("elapsed_ms") or 0.0),
            "text_len": len(text),
            "links_count": len(links),
            "clean_method": str(cleaned.get("clean_method") or ""),
            "removed_blocks": int(cleaned.get("removed_blocks") or 0),
            "raw_len": int(cleaned.get("raw_len") or raw.get("full_len") or len(str(raw.get("html_truncated") or ""))),
            "clean_len": int(cleaned.get("clean_len") or len(text)),
            "cache_hit": "v2_cleaned",
        }
        return ScrapeResult(
            url=request_url,
            final_url=final_url,
            status_code=int(http.get("status") or 200),
            content_type=str(http.get("content_type") or ""),
            title=str(cleaned.get("title") or ""),
            headings=headings,
            text=text,
            links=links,
            metadata=metadata,
        )

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

    def _log_scrape_event(self, event: str, *, target: str, result: ScrapeResult, cache_hit: bool) -> None:
        text = str(getattr(result, "text", "") or "")
        preview = text[:500]
        payload = {
            "request_url": str(target or ""),
            "final_url": str(getattr(result, "final_url", "") or ""),
            "status": int(getattr(result, "status_code", 0) or 0),
            "content_type": str(getattr(result, "content_type", "") or ""),
            "cache_hit": bool(cache_hit),
            "title": str(getattr(result, "title", "") or ""),
            "clean_method": str((getattr(result, "metadata", {}) or {}).get("clean_method") or ""),
            "removed_blocks": int((getattr(result, "metadata", {}) or {}).get("removed_blocks") or 0),
            "raw_len": int((getattr(result, "metadata", {}) or {}).get("raw_len") or 0),
            "clean_len": int((getattr(result, "metadata", {}) or {}).get("clean_len") or len(text)),
            "text_500": preview,
            "text_sha1": self._sha1_text(text),
            "text_len": len(text),
        }
        log_json(LOGGER, event, **payload)


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
