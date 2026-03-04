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
from urllib.parse import parse_qs, quote_plus, unquote, urlencode, urlparse, urlunparse
from urllib.request import Request, urlopen

from utils.cache import DiskTTLCache
from utils.logger import get_logger


_USER_AGENT = "MMisBot/1.0 (+https://local.mmis)"
LOGGER = get_logger(__name__)
_QUERY_INTENTS = {"generic", "fx_rate", "weather", "news_release"}

_TRUST_GENERIC = {
    "docs.python.org": 0.15,
    "github.com": 0.12,
    "stackoverflow.com": 0.08,
    "openai.com": 0.12,
    "wikipedia.org": 0.10,
}

_TRUST_FX_MARKET_FIRST = {
    "minfin.com.ua": 0.34,
    "finance.ua": 0.31,
    "kurs.com.ua": 0.30,
    "obmenka.ua": 0.28,
    "privatbank.ua": 0.22,
    "monobank.ua": 0.22,
    "bank.gov.ua": 0.18,
}

_TRUST_WEATHER = {
    "sinoptik.ua": 0.30,
    "meteo.ua": 0.27,
    "open-meteo.com": 0.32,
    "weather.com": 0.27,
    "accuweather.com": 0.27,
    "gismeteo.ua": 0.24,
}

_TRUST_NEWS_RELEASE = {
    "openai.com": 0.24,
    "github.com": 0.20,
    "docs.python.org": 0.18,
    "wikipedia.org": 0.10,
}


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
        provider: str | None = None,
        strict_endpoint: bool = True,
        timeout_s: float = 12.0,
        cache_dir: str | Path | None = None,
        use_disk_cache: bool = True,
    ):
        self.cache_ttl_s = max(60, int(cache_ttl_s))
        self.endpoint = str(endpoint or "").strip()
        self.provider = str(provider or "searxng").strip().lower() or "searxng"
        self.strict_endpoint = bool(strict_endpoint)
        self.timeout_s = max(3.0, float(timeout_s))
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
        volatile: bool = False,
        query_intent: str = "",
    ) -> list[SearchResult]:
        text = str(query or "").strip()
        if not text:
            return []

        limit = max(1, int(k))
        domains = [x.strip().lower() for x in list(domain_filter or []) if str(x).strip()]
        intent = _normalize_query_intent(query_intent)
        cache_key = f"{text}|{recency_days}|{','.join(sorted(domains))}|{limit}|{intent}"

        if not volatile:
            cached = self._cache_get(cache_key)
            if cached is not None:
                LOGGER.debug("search cache hit provider=%s intent=%s", self.provider, intent)
                return cached[:limit]
        else:
            LOGGER.debug("search cache bypass provider=%s intent=%s mode=volatile", self.provider, intent)

        results: list[SearchResult] = []
        if self.endpoint:
            results = self._search_endpoint(text)
        if not results and not self.strict_endpoint:
            results = self._search_duckduckgo(text)
        if not results and self.strict_endpoint:
            LOGGER.debug("search strict endpoint returned no results provider=%s endpoint=%s", self.provider, self.endpoint)

        if domains:
            results = [x for x in results if _domain_match(x.source, domains)]

        ranked = _rank_results(results, query=text, recency_days=recency_days, query_intent=intent)
        final = ranked[:limit]
        if final and not volatile:
            self._cache_set(cache_key, final)
        LOGGER.debug(
            "search done provider=%s intent=%s results=%s returned=%s cache_mode=%s",
            self.provider,
            intent,
            len(results),
            len(final),
            ("bypass" if volatile else "default"),
        )
        return final

    def _search_endpoint(self, query: str) -> list[SearchResult]:
        if not self.endpoint:
            return []
        req_url = _build_search_url(self.endpoint, query=query)
        try:
            payload = _http_get(req_url, timeout=self.timeout_s)
            raw = json.loads(payload)
        except Exception as exc:
            # Some SearxNG setups can forbid JSON format (403), while HTML search remains available.
            LOGGER.debug("search endpoint json request failed provider=%s err=%s", self.provider, type(exc).__name__)
            try:
                html_url = _build_search_url(_drop_format_param(self.endpoint), query=query)
                html_page = _http_get(html_url, timeout=self.timeout_s)
                parsed = _parse_searx_html_results(html_page, provider=self.provider)
                if parsed:
                    LOGGER.debug("search endpoint html fallback used provider=%s results=%s", self.provider, len(parsed))
                return parsed
            except Exception as exc2:
                LOGGER.debug("search endpoint html fallback failed provider=%s err=%s", self.provider, type(exc2).__name__)
                return []

        rows = raw.get("results")
        if rows is None:
            rows = raw.get("items")
        if rows is None and isinstance(raw, list):
            rows = raw

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
            snippet = str(row.get("content") or row.get("snippet") or row.get("description") or "").strip()
            published_raw = str(
                row.get("published_date")
                or row.get("publishedDate")
                or row.get("date")
                or row.get("published")
                or ""
            ).strip()
            published_date = _extract_date(published_raw) or published_raw
            source = urlparse(link).netloc.lower()
            out.append(
                SearchResult(
                    title=title,
                    snippet=snippet,
                    url=link,
                    source=source,
                    published_date=published_date,
                    score=0.0,
                    raw={"provider": self.provider, **dict(row or {})},
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


def search(
    query: str,
    recency_days: int | None = None,
    domain_filter: list[str] | None = None,
    k: int = 5,
    volatile: bool = False,
    query_intent: str = "",
) -> list[SearchResult]:
    return _DEFAULT_SEARCH.search(
        query=query,
        recency_days=recency_days,
        domain_filter=domain_filter,
        k=k,
        volatile=volatile,
        query_intent=query_intent,
    )


def search_web(query: str) -> list[dict]:
    return [x.to_dict() for x in _DEFAULT_SEARCH.search(query=query, k=5)]


def _http_get(url: str, timeout: float = 12.0) -> str:
    req = Request(url, headers={"User-Agent": _USER_AGENT})
    with urlopen(req, timeout=max(3.0, float(timeout))) as resp:
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


def _drop_format_param(endpoint: str) -> str:
    src = str(endpoint or "").strip()
    if not src:
        return src
    parsed = urlparse(src)
    q = parse_qs(parsed.query, keep_blank_values=True)
    q.pop("format", None)
    new_query = urlencode(q, doseq=True)
    return urlunparse((parsed.scheme, parsed.netloc, parsed.path, parsed.params, new_query, parsed.fragment))


def _build_search_url(endpoint: str, *, query: str) -> str:
    base = str(endpoint or "").strip()
    if not base:
        return ""
    sep = "&" if "?" in base else "?"
    return f"{base}{sep}q={quote_plus(str(query or '').strip())}"


def _parse_searx_html_results(html_page: str, *, provider: str) -> list[SearchResult]:
    src = str(html_page or "")
    if not src:
        return []
    # Extract article result blocks from SearxNG HTML.
    blocks = re.findall(r"<article[^>]*class=\"[^\"]*\bresult\b[^\"]*\"[^>]*>(.*?)</article>", src, flags=re.I | re.S)
    out: list[SearchResult] = []
    for block in blocks:
        href_m = re.search(r"<a[^>]*href=\"([^\"]+)\"[^>]*>", block, flags=re.I | re.S)
        if not href_m:
            continue
        link = html.unescape(str(href_m.group(1) or "").strip())
        if not link:
            continue
        title_m = re.search(r"<h3[^>]*>(.*?)</h3>", block, flags=re.I | re.S)
        title = _clean_html(title_m.group(1) if title_m else "")
        snippet_m = re.search(r"<p[^>]*class=\"[^\"]*\bcontent\b[^\"]*\"[^>]*>(.*?)</p>", block, flags=re.I | re.S)
        snippet = _clean_html(snippet_m.group(1) if snippet_m else "")
        source = urlparse(link).netloc.lower()
        out.append(
            SearchResult(
                title=title,
                snippet=snippet,
                url=link,
                source=source,
                published_date=_extract_date(snippet),
                score=0.0,
                raw={"provider": provider, "format": "html_fallback"},
            )
        )
    return out


def _extract_date(text: str) -> str:
    src = str(text or "")
    m = re.search(r"(20\d{2})[-/.](\d{1,2})[-/.](\d{1,2})(?=[^0-9]|$)", src)
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


def _rank_results(
    items: list[SearchResult],
    query: str,
    recency_days: int | None,
    query_intent: str = "",
) -> list[SearchResult]:
    tokens = {x.lower() for x in re.findall(r"[A-Za-zА-Яа-яЁё0-9_]+", str(query or "")) if x}
    intent = _normalize_query_intent(query_intent)
    if intent == "fx_rate":
        trusted_domains = dict(_TRUST_FX_MARKET_FIRST)
        overlap_w, freshness_w = 0.56, 0.30
    elif intent == "weather":
        trusted_domains = dict(_TRUST_WEATHER)
        overlap_w, freshness_w = 0.56, 0.30
    elif intent == "news_release":
        trusted_domains = dict(_TRUST_NEWS_RELEASE)
        overlap_w, freshness_w = 0.62, 0.24
    else:
        trusted_domains = dict(_TRUST_GENERIC)
        overlap_w, freshness_w = 0.72, 0.18

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

        score = (overlap_w * overlap) + trust + (freshness_w * freshness)
        LOGGER.debug(
            "search rank intent=%s domain=%s overlap=%.3f freshness=%.3f trust=%.3f score=%.3f",
            intent,
            str(item.source or ""),
            overlap,
            freshness,
            trust,
            score,
        )
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


_DEFAULT_SEARCH = SearchClient(provider="searxng", strict_endpoint=True)


def _normalize_query_intent(value: str | None) -> str:
    src = str(value or "").strip().lower()
    return src if src in _QUERY_INTENTS else "generic"
