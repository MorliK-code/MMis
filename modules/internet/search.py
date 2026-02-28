from __future__ import annotations

import datetime as dt
import html
import json
import re
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, quote_plus, unquote, urlparse
from urllib.request import Request, urlopen

from utils.cache import DiskTTLCache
from utils.logger import get_logger


_USER_AGENT = "MMisBot/1.0 (+https://local.mmis)"
LOGGER = get_logger(__name__)


@dataclass(frozen=True)
class SearchResult:
    title: str
    snippet: str
    url: str
    source: str
    published_date: str = ""
    score: float = 0.0
    raw: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["raw"] = dict(self.raw or {})
        return out


class SearchClient:
    def __init__(
        self,
        *,
        cache_ttl_s: int = 900,
        endpoint: str | None = None,
        cache_dir: str | Path | None = None,
        use_disk_cache: bool = True,
    ):
        self.cache_ttl_s = max(60, int(cache_ttl_s))
        self.endpoint = str(endpoint or "").strip()
        self._cache: dict[str, tuple[float, list[SearchResult]]] = {}
        self._lock = threading.RLock()
        self._disk_cache = DiskTTLCache(
            namespace="internet_search",
            root=cache_dir,
            default_ttl_s=self.cache_ttl_s,
            max_memory_entries=1024,
            enabled=bool(use_disk_cache),
        )
        try:
            self._disk_cache.purge_expired(max_files=800)
        except Exception as exc:
            LOGGER.debug("search cache purge skipped: %s", exc)

    def search(
        self,
        query: str,
        recency_days: int | None = None,
        domain_filter: list[str] | None = None,
        k: int = 5,
    ) -> list[SearchResult]:
        text = str(query or "").strip()
        if not text:
            return []

        limit = max(1, int(k))
        domains = [x.strip().lower() for x in list(domain_filter or []) if str(x).strip()]
        cache_key = f"{text}|{recency_days}|{','.join(sorted(domains))}|{limit}"

        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached[:limit]

        results: list[SearchResult] = []
        if self.endpoint:
            results = self._search_endpoint(text)
        if not results:
            results = self._search_duckduckgo(text)

        if domains:
            results = [x for x in results if _domain_match(x.source, domains)]

        ranked = _rank_results(results, query=text, recency_days=recency_days)
        final = ranked[:limit]
        self._cache_set(cache_key, final)
        return final

    def _search_endpoint(self, query: str) -> list[SearchResult]:
        url = self.endpoint
        sep = "&" if "?" in url else "?"
        req_url = f"{url}{sep}q={quote_plus(query)}"
        try:
            payload = _http_get(req_url)
            raw = json.loads(payload)
        except Exception:
            return []

        rows = raw.get("results") if isinstance(raw, dict) else raw
        if not isinstance(rows, list):
            return []

        out: list[SearchResult] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            link = str(row.get("url") or row.get("link") or "").strip()
            if not link:
                continue
            title = str(row.get("title") or "").strip()
            snippet = str(row.get("snippet") or row.get("description") or "").strip()
            published_date = str(row.get("published_date") or row.get("date") or "").strip()
            source = urlparse(link).netloc.lower()
            out.append(
                SearchResult(
                    title=title,
                    snippet=snippet,
                    url=link,
                    source=source,
                    published_date=published_date,
                    score=0.0,
                    raw=row,
                )
            )
        return out

    def _search_duckduckgo(self, query: str) -> list[SearchResult]:
        url = f"https://duckduckgo.com/html/?q={quote_plus(query)}"
        try:
            html_page = _http_get(url)
        except Exception:
            return []

        links = re.findall(r'<a[^>]*class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>', html_page, flags=re.I | re.S)
        snippets = re.findall(r'<a[^>]*class="result__snippet"[^>]*>(.*?)</a>', html_page, flags=re.I | re.S)

        out: list[SearchResult] = []
        for idx, (href, title_html) in enumerate(links):
            title = _clean_html(title_html)
            url_clean = _unwrap_ddg_redirect(href)
            if not url_clean:
                continue
            source = urlparse(url_clean).netloc.lower()
            snippet = _clean_html(snippets[idx]) if idx < len(snippets) else ""
            out.append(
                SearchResult(
                    title=title,
                    snippet=snippet,
                    url=url_clean,
                    source=source,
                    published_date=_extract_date(snippet),
                    score=0.0,
                    raw={"provider": "duckduckgo"},
                )
            )
        return out

    def _cache_get(self, key: str) -> list[SearchResult] | None:
        now = time.time()
        with self._lock:
            row = self._cache.get(key)
            if row is None:
                data = None
            else:
                expires, data = row
                if expires < now:
                    self._cache.pop(key, None)
                    data = None
                else:
                    LOGGER.debug("search cache hit memory")
                    return list(data)

        disk_hit = self._disk_cache.get(key)
        if not isinstance(disk_hit, list):
            return None
        out: list[SearchResult] = []
        for row in disk_hit:
            if isinstance(row, dict):
                item = _search_result_from_dict(row)
                if item is not None:
                    out.append(item)
        if not out:
            return None
        with self._lock:
            self._cache[key] = (time.time() + self.cache_ttl_s, list(out))
        LOGGER.debug("search cache hit disk")
        return out

    def _cache_set(self, key: str, data: list[SearchResult]) -> None:
        with self._lock:
            self._cache[key] = (time.time() + self.cache_ttl_s, list(data))
        try:
            self._disk_cache.set(key, [x.to_dict() for x in list(data or [])], ttl_s=self.cache_ttl_s)
        except Exception as exc:
            LOGGER.debug("search disk cache set failed: %s", exc)


def search(query: str, recency_days: int | None = None, domain_filter: list[str] | None = None, k: int = 5) -> list[SearchResult]:
    return _DEFAULT_SEARCH.search(query=query, recency_days=recency_days, domain_filter=domain_filter, k=k)


def search_web(query: str) -> list[dict]:
    return [x.to_dict() for x in _DEFAULT_SEARCH.search(query=query, k=5)]


def _http_get(url: str, timeout: int = 12) -> str:
    req = Request(url, headers={"User-Agent": _USER_AGENT})
    with urlopen(req, timeout=max(3, int(timeout))) as resp:
        payload = resp.read(1_500_000)
    return payload.decode("utf-8", errors="ignore")


def _unwrap_ddg_redirect(url: str) -> str:
    src = html.unescape(str(url or "").strip())
    if not src:
        return ""
    parsed = urlparse(src)
    if parsed.path == "/l/":
        q = parse_qs(parsed.query)
        target = q.get("uddg", [""])[0]
        return unquote(target)
    return src


def _clean_html(value: str) -> str:
    text = re.sub(r"<[^>]+>", " ", value or "")
    text = html.unescape(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _extract_date(text: str) -> str:
    src = str(text or "")
    m = re.search(r"\b(20\d{2})[-/.](\d{1,2})[-/.](\d{1,2})\b", src)
    if not m:
        return ""
    y, mo, d = m.groups()
    try:
        dt.date(int(y), int(mo), int(d))
        return f"{int(y):04d}-{int(mo):02d}-{int(d):02d}"
    except Exception:
        return ""


def _domain_match(host: str, domains: list[str]) -> bool:
    target = str(host or "").lower()
    if not target:
        return False
    return any(target == d or target.endswith(f".{d}") for d in domains)


def _search_result_from_dict(row: dict[str, Any]) -> SearchResult | None:
    if not isinstance(row, dict):
        return None
    url = str(row.get("url") or "").strip()
    if not url:
        return None
    return SearchResult(
        title=str(row.get("title") or ""),
        snippet=str(row.get("snippet") or ""),
        url=url,
        source=str(row.get("source") or ""),
        published_date=str(row.get("published_date") or ""),
        score=float(row.get("score") or 0.0),
        raw=dict(row.get("raw") or {}),
    )


def _rank_results(items: list[SearchResult], query: str, recency_days: int | None) -> list[SearchResult]:
    tokens = {x.lower() for x in re.findall(r"[A-Za-zА-Яа-яЁё0-9_]+", str(query or "")) if x}
    trusted_domains = {
        "docs.python.org": 0.15,
        "github.com": 0.12,
        "stackoverflow.com": 0.08,
        "openai.com": 0.12,
        "wikipedia.org": 0.1,
    }

    now = dt.datetime.utcnow().date()
    ranked: list[SearchResult] = []

    for item in items:
        blob = f"{item.title} {item.snippet}".lower()
        overlap = 0.0
        if tokens:
            hits = sum(1 for t in tokens if t in blob)
            overlap = hits / max(1, len(tokens))

        trust = 0.0
        for domain, score in trusted_domains.items():
            if item.source == domain or item.source.endswith(f".{domain}"):
                trust = max(trust, score)

        freshness = 0.0
        if recency_days is not None and recency_days > 0 and item.published_date:
            try:
                day = dt.datetime.strptime(item.published_date, "%Y-%m-%d").date()
                age = (now - day).days
                freshness = 1.0 if age <= recency_days else max(0.0, 1.0 - (age - recency_days) / max(1, recency_days * 3))
            except Exception:
                freshness = 0.0

        score = (0.72 * overlap) + trust + (0.18 * freshness)
        ranked.append(
            SearchResult(
                title=item.title,
                snippet=item.snippet,
                url=item.url,
                source=item.source,
                published_date=item.published_date,
                score=score,
                raw=item.raw,
            )
        )

    ranked.sort(key=lambda x: x.score, reverse=True)
    return ranked


_DEFAULT_SEARCH = SearchClient()
