from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

from config.settings import load_config
from modules.internet.search import SearchClient, SearchResult
from modules.internet.scraper import WebScraper
from utils.logger import get_logger, log_json

_TIME_SENSITIVE_RE = re.compile(
    r"\b(today|now|latest|breaking|news|price|rate|currency|exchange|weather|forecast|release|version|"
    r"сегодня|сейчас|последн(ие|яя|ий)|новост|цена|курс|актуал|погод|прогноз|релиз|версии|"
    r"сьогодні|зараз|останні|новин|ціна|курс|актуаль|погод|прогноз|реліз|версії)\b",
    flags=re.I,
)
_FX_RE = re.compile(
    r"\b(курс|доллар|долар|евро|євро|гривн|грн|usd|eur|uah|forex|fx|exchange\s+rate|обмен|обмін)\b",
    flags=re.I,
)
_WEATHER_RE = re.compile(
    r"\b(weather|forecast|temperature|temp|rain|snow|wind|humidity|погод|прогноз|температур|дожд|дощ|снег|вітер|влаж|волог)\b",
    flags=re.I,
)
_NEWS_RE = re.compile(
    r"\b(latest|breaking|release|version|changelog|новост|релиз|обновлен|верс|анонс|announce)\b",
    flags=re.I,
)

_FX_DOMAIN_FILTER = [
    "minfin.com.ua",
    "finance.ua",
    "kurs.com.ua",
    "obmenka.ua",
    "privatbank.ua",
    "monobank.ua",
    "bank.gov.ua",
]
_WEATHER_DOMAIN_FILTER = [
    "open-meteo.com",
    "sinoptik.ua",
    "meteo.ua",
    "weather.com",
    "accuweather.com",
    "gismeteo.ua",
]

LOGGER = get_logger(__name__)


def _needs_web(text: str, tags: dict[str, Any], meta: dict[str, Any], *, query_intent: str) -> bool:
    if meta.get("no_web") is True:
        return False

    # Explicit web request from user/client.
    if meta.get("use_web") is True:
        return True

    # High-priority live-data intents are always web-backed in auto mode.
    if query_intent in {"fx_rate", "weather", "news_release"}:
        return True

    intent = str(tags.get("intent", "")).lower()
    if intent in {"question", "implementation", "action_request", "chat"} and _TIME_SENSITIVE_RE.search(text or ""):
        return True

    return False


def _strip_web_prefix(text: str) -> str:
    s = str(text or "").strip()
    low = s.lower()
    for p in ("/web", "/no-web"):
        if low == p or low.startswith(p + " "):
            return s[len(p) :].strip()
    return s


def _domain(url: str) -> str:
    try:
        return urlparse(url).netloc.lower().strip()
    except Exception:
        return ""


def _clip(s: str, n: int) -> str:
    s = str(s or "").strip()
    if len(s) <= n:
        return s
    return s[: n - 1].rstrip() + "..."


def _top_domains(results: list[SearchResult], *, limit: int = 3) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for row in list(results or []):
        domain = _domain(str(getattr(row, "url", "") or "")) or str(getattr(row, "source", "") or "").strip().lower()
        if not domain or domain in seen:
            continue
        seen.add(domain)
        out.append(domain)
        if len(out) >= max(1, int(limit)):
            break
    return out


def _classify_web_query_intent(query: str) -> str:
    text = str(query or "").strip()
    if not text:
        return "generic"
    lower = text.lower()
    fx_markers = (
        "курс",
        "доллар",
        "долар",
        "евро",
        "євро",
        "грив",
        "usd",
        "eur",
        "uah",
        "forex",
        "fx",
        "exchange rate",
        "обмен",
        "обмін",
    )
    weather_markers = (
        "weather",
        "forecast",
        "temperature",
        "temp",
        "rain",
        "snow",
        "wind",
        "humidity",
        "погод",
        "прогноз",
        "температур",
        "дожд",
        "дощ",
        "снег",
        "вітер",
        "влаж",
        "волог",
    )
    news_markers = (
        "latest",
        "breaking",
        "release",
        "version",
        "changelog",
        "новост",
        "релиз",
        "обновлен",
        "верс",
        "анонс",
        "announce",
    )
    if any(token in lower for token in fx_markers) or _FX_RE.search(text):
        return "fx_rate"
    if any(token in lower for token in weather_markers) or _WEATHER_RE.search(text):
        return "weather"
    if any(token in lower for token in news_markers) or _NEWS_RE.search(text):
        return "news_release"
    return "generic"


def _resolve_recency_days(*, query: str, query_intent: str) -> int | None:
    if query_intent in {"fx_rate", "weather"}:
        return 1
    if query_intent == "news_release":
        return 3
    if _TIME_SENSITIVE_RE.search(query or ""):
        return 7
    return None


def _domain_filter_for_intent(query_intent: str) -> list[str] | None:
    if query_intent == "fx_rate":
        return list(_FX_DOMAIN_FILTER)
    if query_intent == "weather":
        return list(_WEATHER_DOMAIN_FILTER)
    return None


def _fallback_queries_for_intent(query: str, *, query_intent: str) -> list[str]:
    src = str(query or "").strip()
    low = src.lower()
    if query_intent == "fx_rate":
        out = [
            f"{src} usd uah",
            "usd uah exchange rate ukraine today",
            "курс доллара usd uah украина сегодня",
        ]
    elif query_intent == "weather":
        out = [
            f"{src} weather forecast",
            "weather kyiv today",
            "погода киев сегодня",
        ]
    else:
        out = []
    # Remove duplicates and already-included query fragments.
    seen: set[str] = set()
    cleaned: list[str] = []
    for row in out:
        q = str(row or "").strip()
        if not q:
            continue
        if q.lower() == low:
            continue
        key = q.lower()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(q)
    return cleaned


def _set_web_flags(
    ctx,
    *,
    query_intent: str,
    fresh_required: bool,
    fresh_missing: bool,
    web_used: bool | None = None,
) -> None:
    tags = dict(getattr(ctx, "tags", {}) or {})
    tags["web_query_intent"] = str(query_intent or "generic")
    tags["web_fresh_required"] = "true" if bool(fresh_required) else "false"
    tags["web_fresh_missing"] = "true" if bool(fresh_missing) else "false"
    if web_used is not None:
        tags["web_used"] = "true" if bool(web_used) else "false"
    ctx.tags = tags

    if isinstance(getattr(ctx, "meta", None), dict):
        ctx.meta["web_query_intent"] = str(query_intent or "generic")
        ctx.meta["web_fresh_required"] = bool(fresh_required)
        ctx.meta["web_fresh_missing"] = bool(fresh_missing)
        ctx.meta["web_guardrail_local_reply"] = bool(fresh_missing and str(query_intent or "") in {"fx_rate", "weather"})
        if web_used is not None:
            ctx.meta["web_used"] = bool(web_used)


@dataclass
class WebRagConfig:
    k_search: int = 5
    k_fetch: int = 2
    recency_days: int = 30
    max_text_chars: int = 900


class WebRetrieveStage:
    name = "web_retrieve"

    def __init__(
        self,
        *,
        search_client: SearchClient | None = None,
        scraper: WebScraper | None = None,
        cfg: WebRagConfig | None = None,
    ):
        app = load_config()
        self._app = app
        self._cfg = cfg or WebRagConfig()
        self._search = search_client or SearchClient(
            endpoint=app.search_api_url,
            provider=app.search_provider,
            strict_endpoint=bool(app.search_strict_endpoint),
            timeout_s=float(app.search_timeout_sec),
            cache_dir=app.cache_dir,
            cache_ttl_s=900,
        )
        self._scraper = scraper or WebScraper(
            cache_dir=app.cache_dir,
            cache_ttl_s=1800,
            timeout_s=int(app.web_fetch_timeout_sec),
            retries=int(app.web_fetch_retries),
            clean_max_chars=int(app.web_clean_max_chars),
            clean_min_chars=int(app.web_clean_min_chars),
            clean_language_hint=str(app.web_clean_language_hint or ""),
        )

    def run(self, ctx):
        trace = str((ctx.meta or {}).get("web_trace_id") or "").strip() or "-"
        mode = str((ctx.meta or {}).get("web_mode") or (ctx.state or {}).get("web_mode") or "auto").lower()
        if mode not in {"on", "off", "auto"}:
            mode = "auto"
        if isinstance(getattr(ctx, "meta", None), dict):
            ctx.meta["web_mode"] = mode

        text = str(ctx.clean_user_msg or ctx.user_msg or "").strip()
        preview = _clip(text.replace("\n", " "), 120)
        ctx.logs.append(
            f"stage=web_retrieve start trace={trace} mode={mode} text_len={len(text)} text_preview={preview}"
        )
        log_json(
            LOGGER,
            "web_retrieve_start",
            trace=trace,
            mode=mode,
            text_len=len(text),
            text_preview=preview,
        )

        if not text:
            _set_web_flags(
                ctx,
                query_intent="generic",
                fresh_required=False,
                fresh_missing=False,
                web_used=False,
            )
            ctx.logs.append(f"stage=web_retrieve skipped(empty_text) trace={trace}")
            log_json(LOGGER, "web_retrieve_skip", trace=trace, reason="empty_text")
            return ctx

        query = _strip_web_prefix(text)
        query_intent = _classify_web_query_intent(query)
        recency = _resolve_recency_days(query=query, query_intent=query_intent)
        fresh_required = bool(query_intent in {"fx_rate", "weather"} or recency is not None)
        _set_web_flags(
            ctx,
            query_intent=query_intent,
            fresh_required=fresh_required,
            fresh_missing=False,
            web_used=False,
        )

        if not query:
            ctx.logs.append(f"stage=web_retrieve skipped(empty_query) trace={trace}")
            log_json(LOGGER, "web_retrieve_skip", trace=trace, reason="empty_query")
            return ctx

        if mode == "off":
            _set_web_flags(
                ctx,
                query_intent=query_intent,
                fresh_required=fresh_required,
                fresh_missing=fresh_required,
                web_used=False,
            )
            ctx.logs.append(f"stage=web_retrieve skipped(mode=off) trace={trace}")
            log_json(LOGGER, "web_retrieve_skip", trace=trace, reason="mode_off")
            return ctx

        if not bool(self._app.internet_enabled):
            _set_web_flags(
                ctx,
                query_intent=query_intent,
                fresh_required=fresh_required,
                fresh_missing=fresh_required,
                web_used=False,
            )
            ctx.logs.append(f"stage=web_retrieve skipped(internet_disabled) trace={trace}")
            log_json(LOGGER, "web_retrieve_skip", trace=trace, reason="internet_disabled")
            return ctx

        force = (mode == "on") or (query != text)
        needs_web = _needs_web(query, ctx.tags or {}, ctx.meta or {}, query_intent=query_intent)
        domain_filter = _domain_filter_for_intent(query_intent)
        volatile = bool(force or query_intent in {"fx_rate", "weather"} or recency is not None)
        if isinstance(getattr(ctx, "meta", None), dict):
            ctx.meta["web_query"] = str(query)
            ctx.meta["web_force"] = bool(force)

        query_preview = _clip(query.replace("\n", " "), 120)
        ctx.logs.append(
            "stage=web_retrieve decision "
            f"trace={trace} force={int(bool(force))} needs_web={int(bool(needs_web))} "
            f"intent={query_intent} fresh_required={int(fresh_required)} "
            f"query_len={len(query)} query_preview={query_preview}"
        )
        log_json(
            LOGGER,
            "web_retrieve_decision",
            trace=trace,
            force=bool(force),
            needs_web=bool(needs_web),
            query_intent=query_intent,
            fresh_required=bool(fresh_required),
            query_len=len(query),
            query_preview=query_preview,
        )

        if not force and not needs_web:
            ctx.logs.append(f"stage=web_retrieve skipped(auto=no) trace={trace}")
            log_json(LOGGER, "web_retrieve_skip", trace=trace, reason="auto_no")
            return ctx

        if query != text:
            ctx.clean_user_msg = query

        domain_count = len(domain_filter or [])
        ctx.logs.append(
            "stage=web_retrieve search_start "
            f"trace={trace} intent={query_intent} recency_days={recency if recency is not None else 0} "
            f"k={int(self._cfg.k_search)} domains={domain_count} volatile={int(volatile)}"
        )
        log_json(
            LOGGER,
            "web_retrieve_search_start",
            trace=trace,
            query_intent=query_intent,
            recency_days=(int(recency) if recency is not None else 0),
            domain_filter=list(domain_filter or []),
            volatile=bool(volatile),
            k=int(self._cfg.k_search),
        )

        try:
            results: list[SearchResult] = self._search.search(
                query,
                recency_days=recency,
                domain_filter=domain_filter,
                k=self._cfg.k_search,
                volatile=volatile,
                query_intent=query_intent,
            )
        except Exception as exc:
            _set_web_flags(
                ctx,
                query_intent=query_intent,
                fresh_required=fresh_required,
                fresh_missing=fresh_required,
                web_used=False,
            )
            ctx.logs.append(f"stage=web_retrieve search_fail trace={trace} err={type(exc).__name__}")
            log_json(
                LOGGER,
                "web_retrieve_search_fail",
                trace=trace,
                query_intent=query_intent,
                fresh_required=bool(fresh_required),
                error=type(exc).__name__,
            )
            return ctx

        if not results and query_intent in {"fx_rate", "weather"}:
            for alt_query in _fallback_queries_for_intent(query, query_intent=query_intent):
                ctx.logs.append(
                    "stage=web_retrieve search_retry "
                    f"trace={trace} intent={query_intent} alt_query={_clip(alt_query, 120)}"
                )
                log_json(
                    LOGGER,
                    "web_retrieve_search_retry",
                    trace=trace,
                    query_intent=query_intent,
                    alt_query=alt_query,
                )
                try:
                    results = self._search.search(
                        alt_query,
                        recency_days=recency,
                        domain_filter=domain_filter,
                        k=self._cfg.k_search,
                        volatile=True,
                        query_intent=query_intent,
                    )
                except Exception:
                    results = []
                if results:
                    if isinstance(getattr(ctx, "meta", None), dict):
                        ctx.meta["web_query_effective"] = alt_query
                    break

        domains = _top_domains(results, limit=3)
        if isinstance(getattr(ctx, "meta", None), dict):
            ctx.meta["web_result_count"] = int(len(results))
            ctx.meta["web_result_domains"] = list(domains)
        ctx.logs.append(
            "stage=web_retrieve search_done "
            f"trace={trace} results={len(results)} domains={','.join(domains) if domains else '-'}"
        )
        log_json(
            LOGGER,
            "web_retrieve_search_done",
            trace=trace,
            results=len(results),
            domains=list(domains),
        )

        if not results:
            _set_web_flags(
                ctx,
                query_intent=query_intent,
                fresh_required=fresh_required,
                fresh_missing=fresh_required,
                web_used=False,
            )
            ctx.logs.append(f"stage=web_retrieve results=0 trace={trace}")
            log_json(
                LOGGER,
                "web_retrieve_done",
                trace=trace,
                query_intent=query_intent,
                fetched=0,
                fresh_missing=bool(fresh_required),
                web_used=False,
            )
            return ctx

        fetched = 0
        fetched_at = dt.datetime.now(dt.timezone.utc).isoformat()
        for idx, r in enumerate(results):
            if fetched >= self._cfg.k_fetch:
                break

            url = str(r.url or "").strip()
            if not url:
                continue

            domain = _domain(url)
            ctx.logs.append(f"stage=web_retrieve fetch_start trace={trace} idx={idx + 1} domain={domain or '-'}")
            try:
                page = self._scraper.scrape(url)
            except Exception as exc:
                ctx.logs.append(
                    f"stage=web_retrieve scrape_fail trace={trace} idx={idx + 1} domain={domain} err={type(exc).__name__}"
                )
                log_json(
                    LOGGER,
                    "web_retrieve_fetch_fail",
                    trace=trace,
                    idx=(idx + 1),
                    domain=domain,
                    error=type(exc).__name__,
                )
                # Fallback: keep search snippet as web evidence when page scrape is blocked.
                snippet = _clip(str(r.snippet or ""), 320)
                title = str(r.title or "").strip()
                published_date = str(r.published_date or "").strip()
                mem_text = (
                    f"[WEB_SEARCH] {title}\n"
                    f"source_url: {url}\n"
                    f"source_domain: {domain or r.source}\n"
                    f"source_published: {published_date or '-'}\n"
                    f"fetched_at: {fetched_at}\n"
                    f"snippet: {snippet}\n"
                    "text: (page scrape blocked, snippet-only fallback)"
                ).strip()
                ctx.retrieved_memories.append(
                    {
                        "text": mem_text,
                        "confidence": 0.66,
                        "priority": 0.72,
                        "source": domain or str(r.source or "web"),
                        "topic": f"web:{query_intent}",
                        "relevant": True,
                        "source_url": url,
                        "source_domain": domain or str(r.source or ""),
                        "clean_method": "search_snippet_fallback",
                        "published_date": published_date,
                        "fetched_at": fetched_at,
                    }
                )
                fetched += 1
                ctx.logs.append(
                    "stage=web_retrieve fetch_snippet_fallback "
                    f"trace={trace} idx={idx + 1} domain={domain or '-'} snippet_len={len(snippet)}"
                )
                log_json(
                    LOGGER,
                    "web_retrieve_fetch_snippet_fallback",
                    trace=trace,
                    idx=(idx + 1),
                    domain=domain,
                    snippet_len=len(snippet),
                )
                continue

            title = str(getattr(page, "title", "") or r.title or "").strip()
            body = _clip(str(getattr(page, "text", "") or ""), self._cfg.max_text_chars)
            page_meta = dict(getattr(page, "metadata", {}) or {})
            clean_method = str(page_meta.get("clean_method") or "").strip()
            published_date = str(r.published_date or page_meta.get("published_date") or "").strip()

            mem_text = (
                f"[WEB] {title}\n"
                f"source_url: {url}\n"
                f"source_domain: {domain or r.source}\n"
                f"source_published: {published_date or '-'}\n"
                f"fetched_at: {fetched_at}\n"
                f"snippet: {_clip(r.snippet, 220)}\n"
                f"text: {body}"
            ).strip()

            confidence = 0.84 if volatile else 0.74
            priority = 0.88 if volatile else 0.78
            ctx.retrieved_memories.append(
                {
                    "text": mem_text,
                    "confidence": confidence,
                    "priority": priority,
                    "source": domain or "web",
                    "topic": f"web:{query_intent}",
                    "relevant": True,
                    "source_url": url,
                    "source_domain": domain or str(r.source or ""),
                    "clean_method": clean_method,
                    "published_date": published_date,
                    "fetched_at": fetched_at,
                }
            )
            fetched += 1
            ctx.logs.append(
                "stage=web_retrieve fetch_ok "
                f"trace={trace} idx={idx + 1} domain={domain or '-'} title_len={len(title)} text_len={len(body)} clean={clean_method or '-'}"
            )
            log_json(
                LOGGER,
                "web_retrieve_fetch_ok",
                trace=trace,
                idx=(idx + 1),
                domain=domain,
                title_len=len(title),
                text_len=len(body),
                clean_method=clean_method,
            )

        fresh_missing = bool(fresh_required and fetched <= 0)
        _set_web_flags(
            ctx,
            query_intent=query_intent,
            fresh_required=fresh_required,
            fresh_missing=fresh_missing,
            web_used=(fetched > 0),
        )
        if isinstance(getattr(ctx, "meta", None), dict):
            ctx.meta["web_fetched"] = int(fetched)

        ctx.logs.append(
            "stage=web_retrieve done "
            f"trace={trace} query_len={len(query)} fetched={fetched} web_used={ctx.tags.get('web_used')} "
            f"fresh_missing={ctx.tags.get('web_fresh_missing')}"
        )
        log_json(
            LOGGER,
            "web_retrieve_done",
            trace=trace,
            query_intent=query_intent,
            query_len=len(query),
            fetched=fetched,
            fresh_missing=bool(fresh_missing),
            web_used=bool(fetched > 0),
        )
        return ctx
