from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

from config.settings import load_config
from modules.internet.search import SearchClient, SearchResult
from modules.internet.scraper import WebScraper

_TIME_SENSITIVE_RE = re.compile(
    r"\b(сегодня|сейчас|в\s+202\d|последн(ие|яя|ий)|новост|цена|курс|актуал|релиз|верси)\b",
    flags=re.I,
)


def _needs_web(text: str, tags: dict[str, Any], meta: dict[str, Any]) -> bool:
    if meta.get("no_web") is True:
        return False

    # если пользователь/клиент явно включил web
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
    return s[: n - 1].rstrip() + "…"


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
        mode = str((ctx.meta or {}).get("web_mode") or "auto").lower()
        if mode == "off":
            ctx.logs.append("stage=web_retrieve skipped(mode=off)")
            return ctx

        if not bool(self._app.internet_enabled):
            ctx.logs.append("stage=web_retrieve skipped(internet_disabled)")
            return ctx

        text = str(ctx.clean_user_msg or ctx.user_msg or "").strip()
        if not text:
            ctx.logs.append("stage=web_retrieve skipped(empty_text)")
            return ctx

        query = _strip_web_prefix(text)
        if not query:
            ctx.logs.append("stage=web_retrieve skipped(empty_query)")
            return ctx

        # /web ... -> force
        force = (mode == "on") or (query != text)

        if not force and not _needs_web(query, ctx.tags or {}, ctx.meta or {}):
            ctx.logs.append("stage=web_retrieve skipped(auto=no)")
            return ctx

        # убираем "/web" из сообщения перед LLM
        if query != text:
            ctx.clean_user_msg = query

        recency = self._cfg.recency_days if _TIME_SENSITIVE_RE.search(query) else None
        results: list[SearchResult] = self._search.search(query, recency_days=recency, k=self._cfg.k_search)

        if not results:
            ctx.tags["web_used"] = "false"
            ctx.logs.append("stage=web_retrieve results=0")
            return ctx

        fetched = 0
        for r in results:
            if fetched >= self._cfg.k_fetch:
                break

            url = str(r.url or "").strip()
            if not url:
                continue

            try:
                page = self._scraper.scrape(url)
            except Exception as exc:
                ctx.logs.append(f"stage=web_retrieve scrape_fail domain={_domain(url)} err={type(exc).__name__}")
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
                    "source": _domain(url) or "web",
                    "topic": "web",
                    "relevant": True,
                }
            )
            fetched += 1

        ctx.tags["web_used"] = "true" if fetched > 0 else "false"
        ctx.logs.append(f"stage=web_retrieve query_len={len(query)} fetched={fetched}")
        return ctx