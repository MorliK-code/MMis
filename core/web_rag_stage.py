from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

from config.settings import load_config
from modules.internet.search import SearchClient, SearchResult
from modules.internet.scraper import WebScraper
from utils.logger import get_logger, log_json

_TIME_SENSITIVE_RE = re.compile(
    r"\b(сегодня|сейчас|в\s+202\d|последн(ие|яя|ий)|новост|цена|курс|актуал|релиз|версии)\b",
    flags=re.I,
)
LOGGER = get_logger(__name__)


def _needs_web(text: str, tags: dict[str, Any], meta: dict[str, Any]) -> bool:
    if meta.get("no_web") is True:
        return False

    # Explicit web request from user/client.
    if meta.get("use_web") is True:
        return True

    intent = str(tags.get("intent", "")).lower()
    if intent in {"question", "implementation", "action_request"} and _TIME_SENSITIVE_RE.search(text or ""):
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
        self._search = search_client or SearchClient(endpoint=app.search_api_url, cache_dir=app.cache_dir, cache_ttl_s=900)
        self._scraper = scraper or WebScraper(cache_dir=app.cache_dir, cache_ttl_s=1800, retries=1)

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

        if mode == "off":
            ctx.tags["web_used"] = "false"
            ctx.logs.append(f"stage=web_retrieve skipped(mode=off) trace={trace}")
            log_json(LOGGER, "web_retrieve_skip", trace=trace, reason="mode_off")
            return ctx

        if not bool(self._app.internet_enabled):
            ctx.tags["web_used"] = "false"
            ctx.logs.append(f"stage=web_retrieve skipped(internet_disabled) trace={trace}")
            log_json(LOGGER, "web_retrieve_skip", trace=trace, reason="internet_disabled")
            return ctx

        if not text:
            ctx.tags["web_used"] = "false"
            ctx.logs.append(f"stage=web_retrieve skipped(empty_text) trace={trace}")
            log_json(LOGGER, "web_retrieve_skip", trace=trace, reason="empty_text")
            return ctx

        query = _strip_web_prefix(text)
        if not query:
            ctx.tags["web_used"] = "false"
            ctx.logs.append(f"stage=web_retrieve skipped(empty_query) trace={trace}")
            log_json(LOGGER, "web_retrieve_skip", trace=trace, reason="empty_query")
            return ctx

        force = (mode == "on") or (query != text)
        needs_web = _needs_web(query, ctx.tags or {}, ctx.meta or {})
        if isinstance(getattr(ctx, "meta", None), dict):
            ctx.meta["web_query"] = str(query)
            ctx.meta["web_force"] = bool(force)
        query_preview = _clip(query.replace("\n", " "), 120)
        ctx.logs.append(
            "stage=web_retrieve decision "
            f"trace={trace} force={int(bool(force))} needs_web={int(bool(needs_web))} "
            f"query_len={len(query)} query_preview={query_preview}"
        )
        log_json(
            LOGGER,
            "web_retrieve_decision",
            trace=trace,
            force=bool(force),
            needs_web=bool(needs_web),
            query_len=len(query),
            query_preview=query_preview,
        )

        if not force and not needs_web:
            ctx.tags["web_used"] = "false"
            ctx.logs.append(f"stage=web_retrieve skipped(auto=no) trace={trace}")
            log_json(LOGGER, "web_retrieve_skip", trace=trace, reason="auto_no")
            return ctx

        if query != text:
            ctx.clean_user_msg = query

        recency = self._cfg.recency_days if _TIME_SENSITIVE_RE.search(query) else None
        ctx.logs.append(
            "stage=web_retrieve search_start "
            f"trace={trace} recency_days={recency if recency is not None else 0} k={int(self._cfg.k_search)}"
        )
        log_json(
            LOGGER,
            "web_retrieve_search_start",
            trace=trace,
            recency_days=(int(recency) if recency is not None else 0),
            k=int(self._cfg.k_search),
        )
        results: list[SearchResult] = self._search.search(query, recency_days=recency, k=self._cfg.k_search)
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
            ctx.tags["web_used"] = "false"
            ctx.logs.append(f"stage=web_retrieve results=0 trace={trace}")
            log_json(LOGGER, "web_retrieve_done", trace=trace, fetched=0, web_used=False)
            return ctx

        fetched = 0
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
                continue

            title = str(page.title or r.title or "").strip()
            body = _clip(page.text, self._cfg.max_text_chars)

            mem_text = (
                f"[WEB] {title}\n"
                f"source: {url}\n"
                f"snippet: {_clip(r.snippet, 220)}\n"
                f"text: {body}"
            ).strip()

            ctx.retrieved_memories.append(
                {
                    "text": mem_text,
                    "confidence": 0.74,
                    "priority": 0.78,
                    "source": domain or "web",
                    "topic": "web",
                    "relevant": True,
                }
            )
            fetched += 1
            ctx.logs.append(
                "stage=web_retrieve fetch_ok "
                f"trace={trace} idx={idx + 1} domain={domain or '-'} title_len={len(title)} text_len={len(body)}"
            )
            log_json(
                LOGGER,
                "web_retrieve_fetch_ok",
                trace=trace,
                idx=(idx + 1),
                domain=domain,
                title_len=len(title),
                text_len=len(body),
            )

        ctx.tags["web_used"] = "true" if fetched > 0 else "false"
        if isinstance(getattr(ctx, "meta", None), dict):
            ctx.meta["web_fetched"] = int(fetched)
        ctx.logs.append(
            "stage=web_retrieve done "
            f"trace={trace} query_len={len(query)} fetched={fetched} web_used={ctx.tags.get('web_used')}"
        )
        log_json(
            LOGGER,
            "web_retrieve_done",
            trace=trace,
            query_len=len(query),
            fetched=fetched,
            web_used=bool(fetched > 0),
        )
        return ctx
