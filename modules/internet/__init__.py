from __future__ import annotations

from modules.internet.content_cleaner import CleanResult, clean_web_content
from modules.internet.scraper import LinkRef, ScrapeResult, WebScraper, extract_links, extract_readable, fetch, scrape
from modules.internet.search import SearchClient, SearchResult, search, search_web
from modules.internet.search_planner import SearchPlanner
from modules.internet.web import WebStageConfig, WebStageV2

__all__ = [
    "CleanResult",
    "clean_web_content",
    "SearchResult",
    "SearchClient",
    "search",
    "search_web",
    "SearchPlanner",
    "LinkRef",
    "ScrapeResult",
    "WebScraper",
    "fetch",
    "extract_readable",
    "extract_links",
    "scrape",
    "WebStageV2",
    "WebStageConfig",
]
