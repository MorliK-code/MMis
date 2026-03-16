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

from modules.internet.web.search_policy import (
    classify_engine_state,
    compute_search_confidence,
    normalize_search_locale,
    region_domain_adjustment,
)
from utils.cache import DiskTTLCache
from utils.logger import get_logger, log_json


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

_SOURCE_PRIORITY_GENERIC = {
    "wikipedia.org": 0.12,
    "allrecipes.com": 0.14,
    "seriouseats.com": 0.12,
    "docs.python.org": 0.10,
    "developer.mozilla.org": 0.10,
    "github.com": 0.08,
}

_SOURCE_PRIORITY_NEWS = {
    "openai.com": 0.10,
    "github.com": 0.08,
    "docs.python.org": 0.08,
    "reuters.com": 0.10,
    "apnews.com": 0.09,
}

_SOURCE_PRIORITY_PENALTY_GENERIC = {
    "pinterest.com": -0.12,
    "quora.com": -0.08,
}


@dataclass(frozen=True)
class SearchResult:
    title: str
    snippet: str
    url: str
    source: str
    published_date: str = ""
    score: float = 0.0
    score_breakdown: dict[str, float] | None = None
    raw: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["score_breakdown"] = dict(self.score_breakdown or {})
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
        query_category: str = "",
        query_locale: str = "",
        region_bias: str = "",
        engine_manual_states: dict[str, str] | None = None,
        suspended_engines: list[str] | None = None,
    ) -> list[SearchResult]:
        rows, _ = self.search_with_meta(
            query=query,
            recency_days=recency_days,
            domain_filter=domain_filter,
            k=k,
            volatile=volatile,
            query_intent=query_intent,
            query_category=query_category,
            query_locale=query_locale,
            region_bias=region_bias,
            engine_manual_states=engine_manual_states,
            suspended_engines=suspended_engines,
        )
        return rows

    def search_with_meta(
        self,
        query: str,
        recency_days: int | None = None,
        domain_filter: list[str] | None = None,
        k: int = 5,
        volatile: bool = False,
        query_intent: str = "",
        query_category: str = "",
        query_locale: str = "",
        region_bias: str = "",
        engine_manual_states: dict[str, str] | None = None,
        suspended_engines: list[str] | None = None,
    ) -> tuple[list[SearchResult], dict[str, Any]]:
        text = str(query or "").strip()
        if not text:
            return ([], {"provider": self.provider, "effective_success_reason": "empty_query", "effective_success": False})

        limit = max(1, int(k))
        domains = [x.strip().lower() for x in list(domain_filter or []) if str(x).strip()]
        intent = _normalize_query_intent(query_intent)
        category = str(query_category or "").strip().lower()
        locale = normalize_search_locale(query_locale, default="ru-RU")
        region = str(region_bias or "").strip().lower()
        manual_engine_states = {str(k or "").strip().lower(): str(v or "").strip().lower() for k, v in dict(engine_manual_states or {}).items() if str(k or "").strip()}
        suspended = [str(x or "").strip().lower() for x in list(suspended_engines or []) if str(x or "").strip()]
        cache_key = f"{text}|{recency_days}|{','.join(sorted(domains))}|{limit}|{intent}|{category}|{locale}|{region}"

        if not volatile:
            cached = self._cache_get(cache_key)
            if cached is not None:
                cached_debug = _search_debug_from_results(cached)
                LOGGER.debug("search cache hit provider=%s intent=%s", self.provider, intent)
                log_json(
                    LOGGER,
                    "search_cache_policy",
                    provider=self.provider,
                    query=text,
                    query_intent=intent,
                    cache_mode="default",
                    cache_hit=True,
                )
                return (cached[:limit], cached_debug)
        else:
            LOGGER.debug("search cache bypass provider=%s intent=%s mode=volatile", self.provider, intent)
            log_json(
                LOGGER,
                "search_cache_policy",
                provider=self.provider,
                query=text,
                query_intent=intent,
                cache_mode="read_bypass_write",
                cache_hit=False,
            )

        results: list[SearchResult] = []
        provider_debug: dict[str, Any] = {}
        if self.endpoint:
            results, provider_debug = self._search_endpoint_with_meta(
                text,
                query_locale=locale,
                engine_manual_states=manual_engine_states,
                suspended_engines=suspended,
            )
        if not results and not self.strict_endpoint:
            results = self._search_duckduckgo(text)
        if not results and self.strict_endpoint:
            LOGGER.debug("search strict endpoint returned no results provider=%s endpoint=%s", self.provider, self.endpoint)

        if domains:
            results = [x for x in results if _domain_match(x.source, domains)]

        ranked = _rank_results(
            results,
            query=text,
            recency_days=recency_days,
            query_intent=intent,
            query_category=category,
            region_bias=region,
        )
        final = ranked[:limit]
        usable_results_count = int(len(results))
        effective_search_confidence = compute_search_confidence(
            raw_results_count=int(_coerce_int(provider_debug.get("raw_result_count"), len(results)) or 0),
            usable_results_count=usable_results_count,
            top_scores=[float(item.score or 0.0) for item in list(final or [])[:3]],
            engine_health=dict(provider_debug.get("engine_health") or {}),
        )
        effective_success = bool(usable_results_count > 0)
        provider_debug = {
            **dict(provider_debug or {}),
            "usable_results_count": int(usable_results_count),
            "raw_results_count": int(_coerce_int(provider_debug.get("raw_result_count"), len(results)) or 0),
            "reported_number_of_results": _coerce_int(provider_debug.get("reported_result_count"), None),
            "effective_success": bool(effective_success),
            "effective_search_confidence": round(float(effective_search_confidence), 6),
            "query_locale": str(locale or ""),
            "region_bias": str(region or ""),
            "top_domains": _top_domains(final, limit=5),
        }
        final = [
            SearchResult(
                title=item.title,
                snippet=item.snippet,
                url=item.url,
                source=item.source,
                published_date=item.published_date,
                score=item.score,
                score_breakdown=dict(item.score_breakdown or {}),
                raw={**dict(item.raw or {}), **dict(provider_debug or {})},
            )
            for item in list(final or [])
        ]
        if final:
            self._cache_set(cache_key, final)
            cache_mode = "read_bypass_write" if volatile else "default"
        else:
            cache_mode = "read_bypass_no_write" if volatile else "default"
        LOGGER.debug(
            "search done provider=%s intent=%s results=%s returned=%s cache_mode=%s",
            self.provider,
            intent,
            len(results),
            len(final),
            cache_mode,
        )
        log_json(
            LOGGER,
            "search_done",
            provider=self.provider,
            query=text,
            query_intent=intent,
            cache_mode=cache_mode,
            result_count=len(results),
            returned_count=len(final),
            raw_result_count=int(_coerce_int(provider_debug.get("raw_result_count"), len(results)) or 0),
            reported_result_count=_coerce_int(provider_debug.get("reported_result_count"), None),
            usable_results_count=int(usable_results_count),
            engine_success_count=int(_coerce_int(provider_debug.get("engine_success_count"), 0) or 0),
            engine_failure_count=int(_coerce_int(provider_debug.get("engine_failure_count"), 0) or 0),
            effective_success=bool(effective_success),
            effective_success_reason=str(
                provider_debug.get("effective_success_reason")
                or ("results_present" if usable_results_count > 0 else "empty_results")
            ),
            effective_search_confidence=float(provider_debug.get("effective_search_confidence") or 0.0),
            query_locale=str(provider_debug.get("query_locale") or ""),
            region_bias=str(provider_debug.get("region_bias") or ""),
            engine_health=dict(provider_debug.get("engine_health") or {}),
            top_results=[
                {
                    "rank": idx + 1,
                    "title": str(item.title or ""),
                    "url": str(item.url or ""),
                    "domain": str(item.source or ""),
                    "published_date": str(item.published_date or ""),
                    "score_total": float(item.score or 0.0),
                    "score_breakdown": dict(item.score_breakdown or {}),
                }
                for idx, item in enumerate(final[:5])
            ],
        )
        return final, provider_debug

    def _search_endpoint(self, query: str) -> list[SearchResult]:
        rows, _ = self._search_endpoint_with_meta(query)
        return rows

    def _search_endpoint_with_meta(
        self,
        query: str,
        *,
        query_locale: str = "",
        engine_manual_states: dict[str, str] | None = None,
        suspended_engines: list[str] | None = None,
    ) -> tuple[list[SearchResult], dict[str, Any]]:
        if not self.endpoint:
            return ([], {"provider": self.provider, "effective_success_reason": "no_endpoint"})
        req_url = _build_search_url(self.endpoint, query=query, query_locale=query_locale)
        try:
            payload = _http_get(req_url, timeout=self.timeout_s)
            raw = json.loads(payload)
        except Exception as exc:
            # Some SearxNG setups can forbid JSON format (403), while HTML search remains available.
            LOGGER.debug("search endpoint json request failed provider=%s err=%s", self.provider, type(exc).__name__)
            try:
                html_url = _build_search_url(_drop_format_param(self.endpoint), query=query, query_locale=query_locale)
                html_page = _http_get(html_url, timeout=self.timeout_s)
                parsed = _parse_searx_html_results(html_page, provider=self.provider)
                if parsed:
                    LOGGER.debug("search endpoint html fallback used provider=%s results=%s", self.provider, len(parsed))
                return (
                    parsed,
                    {
                        "provider": self.provider,
                        "raw_result_count": int(len(parsed)),
                        "reported_result_count": None,
                        "effective_success_reason": "html_results_present" if parsed else "html_empty_results",
                    },
                )
            except Exception as exc2:
                LOGGER.debug("search endpoint html fallback failed provider=%s err=%s", self.provider, type(exc2).__name__)
                return ([], {"provider": self.provider, "effective_success_reason": "endpoint_request_failed"})

        rows = raw.get("results")
        if rows is None:
            rows = raw.get("items")
        if rows is None and isinstance(raw, list):
            rows = raw

        if not isinstance(rows, list):
            return (
                [],
                {
                    "provider": self.provider,
                    "reported_result_count": _coerce_int((raw if isinstance(raw, dict) else {}).get("number_of_results"), None),
                    "raw_result_count": 0,
                    "effective_success_reason": "missing_results_list",
                },
            )

        raw_result_count = int(len(rows))
        reported_result_count = _coerce_int((raw if isinstance(raw, dict) else {}).get("number_of_results"), None)
        if raw_result_count > 0 and reported_result_count == 0:
            effective_success_reason = "results_present_number_of_results_zero"
        elif raw_result_count > 0:
            effective_success_reason = "results_present"
        elif reported_result_count and reported_result_count > 0:
            effective_success_reason = "reported_results_without_rows"
        else:
            effective_success_reason = "empty_results"
        engine_debug = _extract_engine_summary(
            raw=raw,
            rows=rows,
            manual_states=engine_manual_states,
            suspended_engines=suspended_engines,
        )
        provider_debug = {
            "provider": self.provider,
            "reported_result_count": reported_result_count,
            "reported_number_of_results": reported_result_count,
            "raw_result_count": raw_result_count,
            "raw_results_count": raw_result_count,
            "effective_success_reason": effective_success_reason,
            **dict(engine_debug or {}),
        }

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
                    raw={
                        **dict(provider_debug or {}),
                        **dict(row or {}),
                    },
                )
            )
        return (out, provider_debug)

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


def _build_search_url(endpoint: str, *, query: str, query_locale: str = "") -> str:
    base = str(endpoint or "").strip()
    if not base:
        return ""
    parsed = urlparse(base)
    q = parse_qs(parsed.query, keep_blank_values=True)
    q["q"] = [str(query or "").strip()]
    locale = normalize_search_locale(query_locale, default="ru-RU")
    if locale:
        q["language"] = [locale]
    new_query = urlencode(q, doseq=True, quote_via=quote_plus)
    return urlunparse((parsed.scheme, parsed.netloc, parsed.path, parsed.params, new_query, parsed.fragment))


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
        score_breakdown=dict(row.get("score_breakdown") or {}),
        raw=dict(row.get("raw") or {}),
    )


def _rank_results(
    items: list[SearchResult],
    query: str,
    recency_days: int | None,
    query_intent: str = "",
    query_category: str = "",
    region_bias: str = "",
) -> list[SearchResult]:
    tokens = {x.lower() for x in re.findall(r"[A-Za-zА-Яа-яЁё0-9_]+", str(query or "")) if x}
    intent = _normalize_query_intent(query_intent)
    category = str(query_category or "").strip().lower()
    source_priority_map: dict[str, float] = {}
    penalty_map: dict[str, float] = {}
    if intent == "fx_rate":
        trusted_domains = dict(_TRUST_FX_MARKET_FIRST)
        overlap_w, freshness_w = 0.56, 0.30
    elif intent == "weather":
        trusted_domains = dict(_TRUST_WEATHER)
        overlap_w, freshness_w = 0.56, 0.30
    elif intent == "news_release":
        trusted_domains = dict(_TRUST_NEWS_RELEASE)
        overlap_w, freshness_w = 0.62, 0.24
        source_priority_map = dict(_SOURCE_PRIORITY_NEWS)
        penalty_map: dict[str, float] = {}
    else:
        trusted_domains = dict(_TRUST_GENERIC)
        overlap_w, freshness_w = 0.72, 0.18
        source_priority_map = dict(_SOURCE_PRIORITY_GENERIC)
        penalty_map = dict(_SOURCE_PRIORITY_PENALTY_GENERIC)

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

        source_priority = 0.0
        for domain, score in source_priority_map.items():
            if item.source == domain or item.source.endswith(f".{domain}"):
                source_priority = max(source_priority, float(score))
        for domain, penalty in penalty_map.items():
            if item.source == domain or item.source.endswith(f".{domain}"):
                source_priority = min(source_priority, float(penalty))

        freshness = 0.0
        if recency_days is not None and recency_days > 0 and item.published_date:
            try:
                day = dt.datetime.strptime(item.published_date, "%Y-%m-%d").date()
                age = (now - day).days
                freshness = 1.0 if age <= recency_days else max(0.0, 1.0 - (age - recency_days) / max(1, recency_days * 3))
            except Exception:
                freshness = 0.0

        region_bonus = region_domain_adjustment(
            domain=str(item.source or ""),
            region_bias=region_bias,
            query_intent=intent,
            query_category=category,
            query_text=query,
        )
        score = (overlap_w * overlap) + trust + (freshness_w * freshness) + source_priority + region_bonus
        lexical_component = overlap_w * overlap
        freshness_component = freshness_w * freshness
        breakdown = {
            "lexical": round(float(lexical_component), 6),
            "trust": round(float(trust), 6),
            "freshness": round(float(freshness_component), 6),
            "source_priority": round(float(source_priority), 6),
            "region_bias": round(float(region_bonus), 6),
        }
        LOGGER.debug(
            "search rank intent=%s domain=%s overlap=%.3f freshness=%.3f trust=%.3f source_priority=%.3f region_bias=%.3f score=%.3f",
            intent,
            str(item.source or ""),
            overlap,
            freshness,
            trust,
            source_priority,
            region_bonus,
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
                score_breakdown=breakdown,
                raw=item.raw,
            )
        )

    ranked.sort(key=lambda x: x.score, reverse=True)
    return ranked


def _extract_engine_summary(
    *,
    raw: Any,
    rows: list[Any],
    manual_states: dict[str, str] | None = None,
    suspended_engines: list[str] | None = None,
) -> dict[str, Any]:
    payload = dict(raw or {}) if isinstance(raw, dict) else {}
    success: set[str] = set()
    failures: dict[str, str] = {}
    manual = {str(k or "").strip().lower(): str(v or "").strip().lower() for k, v in dict(manual_states or {}).items() if str(k or "").strip()}
    suspended = {str(x or "").strip().lower() for x in list(suspended_engines or []) if str(x or "").strip()}

    for row in list(rows or []):
        if not isinstance(row, dict):
            continue
        engines_value = row.get("engines")
        if isinstance(engines_value, str):
            candidate = _normalize_engine_name(engines_value)
            if candidate:
                success.add(candidate)
        elif isinstance(engines_value, list):
            for item in engines_value:
                candidate = _normalize_engine_name(item)
                if candidate:
                    success.add(candidate)
        candidate = _normalize_engine_name(row.get("engine"))
        if candidate:
            success.add(candidate)

    unresponsive = payload.get("unresponsive_engines")
    if isinstance(unresponsive, list):
        for item in unresponsive:
            engine, reason = _normalize_engine_failure(item)
            if engine:
                failures[engine] = reason or "unresponsive"
    elif isinstance(unresponsive, dict):
        for key, value in unresponsive.items():
            engine = _normalize_engine_name(key)
            reason = str(value or "").strip()
            if engine:
                failures[engine] = reason or "unresponsive"

    for key in ("engine_failures", "errors", "warnings"):
        value = payload.get(key)
        if isinstance(value, dict):
            for engine_key, reason_value in value.items():
                engine = _normalize_engine_name(engine_key)
                if engine:
                    failures[engine] = str(reason_value or "").strip()
        elif isinstance(value, list):
            for item in value:
                engine, reason = _normalize_engine_failure(item)
                if engine:
                    failures[engine] = reason or failures.get(engine, "")

    health: dict[str, str] = {}
    all_engines = sorted(set(success) | set(failures) | set(manual) | set(suspended))
    for engine in all_engines:
        manual_state = manual.get(engine) or ("suspended" if engine in suspended else "")
        reason = failures.get(engine, "")
        state = classify_engine_state(engine=engine, reason=reason, manual_state=manual_state)
        if engine in success and not reason and not manual_state:
            state = "healthy"
        health[engine] = state

    healthy = sum(1 for state in health.values() if state == "healthy")
    failed = {engine: reason for engine, reason in failures.items() if engine not in success or health.get(engine) != "healthy"}
    return {
        "engines_succeeded": sorted(success),
        "engines_failed": [{"engine": engine, "reason": str(reason or "").strip(), "state": health.get(engine, "degraded")} for engine, reason in sorted(failed.items())],
        "engine_health": dict(health),
        "engine_success_count": int(healthy if health else len(success)),
        "engine_failure_count": int(sum(1 for state in health.values() if state in {"degraded", "blocked", "suspended"})),
    }


def _normalize_engine_failure(value: Any) -> tuple[str, str]:
    if isinstance(value, dict):
        engine = _normalize_engine_name(value.get("engine") or value.get("name"))
        reason = str(value.get("reason") or value.get("error") or value.get("message") or "").strip()
        return engine, reason
    text = str(value or "").strip()
    if not text:
        return "", ""
    parts = re.split(r"[:\-]\s*", text, maxsplit=1)
    engine = _normalize_engine_name(parts[0])
    reason = parts[1].strip() if len(parts) > 1 else text
    return engine, reason


def _normalize_engine_name(value: Any) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return ""
    return re.sub(r"[^a-z0-9_+\-\.]+", "", text)


def _search_debug_from_results(rows: list[SearchResult]) -> dict[str, Any]:
    if not rows:
        return {
            "provider": "cache",
            "raw_result_count": 0,
            "raw_results_count": 0,
            "usable_results_count": 0,
            "reported_result_count": None,
            "reported_number_of_results": None,
            "engine_success_count": 0,
            "engine_failure_count": 0,
            "effective_success": False,
            "effective_success_reason": "empty_results",
            "effective_search_confidence": 0.0,
            "engine_health": {},
        }
    raw = dict(getattr(rows[0], "raw", {}) or {})
    return {
        "provider": str(raw.get("provider") or "cache"),
        "raw_result_count": int(_coerce_int(raw.get("raw_result_count"), len(list(rows or []))) or 0),
        "raw_results_count": int(_coerce_int(raw.get("raw_results_count"), raw.get("raw_result_count")) or len(list(rows or []))),
        "usable_results_count": int(_coerce_int(raw.get("usable_results_count"), len(list(rows or []))) or len(list(rows or []))),
        "reported_result_count": _coerce_int(raw.get("reported_result_count"), None),
        "reported_number_of_results": _coerce_int(raw.get("reported_number_of_results"), raw.get("reported_result_count")),
        "engine_success_count": int(_coerce_int(raw.get("engine_success_count"), 0) or 0),
        "engine_failure_count": int(_coerce_int(raw.get("engine_failure_count"), 0) or 0),
        "engine_health": dict(raw.get("engine_health") or {}),
        "effective_success": bool(raw.get("effective_success", bool(rows))),
        "effective_success_reason": str(raw.get("effective_success_reason") or ("results_present" if rows else "empty_results")),
        "effective_search_confidence": float(raw.get("effective_search_confidence") or 0.0),
        "query_locale": str(raw.get("query_locale") or ""),
        "region_bias": str(raw.get("region_bias") or ""),
        "top_domains": _top_domains(rows, limit=5),
    }


def _top_domains(rows: list[SearchResult], *, limit: int) -> list[str]:
    out: list[str] = []
    for item in list(rows or []):
        domain = str(getattr(item, "source", "") or "").strip().lower()
        if not domain or domain in out:
            continue
        out.append(domain)
        if len(out) >= max(1, int(limit)):
            break
    return out


def _coerce_int(value: Any, default: int | None = 0) -> int | None:
    try:
        if value is None or str(value).strip() == "":
            return default
        return int(value)
    except Exception:
        return default


_DEFAULT_SEARCH = SearchClient(provider="searxng", strict_endpoint=True)


def _normalize_query_intent(value: str | None) -> str:
    src = str(value or "").strip().lower()
    return src if src in _QUERY_INTENTS else "generic"
