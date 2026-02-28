from __future__ import annotations

from modules.internet.scraper import LinkRef, ScrapeResult, WebScraper, extract_links, extract_readable, fetch, scrape
from modules.internet.search import SearchClient, SearchResult, search, search_web

__all__ = [
    "SearchResult",
    "SearchClient",
    "search",
    "search_web",
    "LinkRef",
    "ScrapeResult",
    "WebScraper",
    "fetch",
    "extract_readable",
    "extract_links",
    "scrape",
]
