from __future__ import annotations

import datetime as dt
import time
from dataclasses import dataclass
from typing import Any

from modules.internet.search import SearchClient, SearchResult
from modules.internet.scraper import WebScraper
from modules.internet.web.web_models import FreshnessAssessment, QueryClassification, WebPolicyDecision, WebQueryPlan, WebSearchMode


@dataclass(frozen=True)
class SearchExecutionResult:
    results: list[SearchResult]
    fetched_pages: dict[str, dict[str, Any]]
    queries_used: list[str]
    domain_filter: list[str]
    recency_days: int | None
    scout_queries_used: list[str] = None  # type: ignore[assignment]
    focused_queries_used: list[str] = None  # type: ignore[assignment]
    retries_used: int = 0
    cooldown_applied: bool = False
    query_runs: list[dict[str, Any]] = None  # type: ignore[assignment]
    fetch_summary: dict[str, Any] = None  # type: ignore[assignment]
    warnings: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        object.__setattr__(self, "scout_queries_used", list(self.scout_queries_used or []))
        object.__setattr__(self, "focused_queries_used", list(self.focused_queries_used or []))
        object.__setattr__(self, "query_runs", [dict(x or {}) for x in list(self.query_runs or []) if isinstance(x, dict)])
        object.__setattr__(self, "fetch_summary", dict(self.fetch_summary or {}))
        object.__setattr__(self, "warnings", [str(x or "").strip() for x in list(self.warnings or []) if str(x or "").strip()])


@dataclass(frozen=True)
class _ModePolicy:
    max_queries: int
    max_sources: int
    max_fetches: int
    scout_cap: int
    focused_cap: int
    retries: int


class SearchManager:
    def __init__(
        self,
        *,
        search_client: SearchClient,
        scraper: WebScraper,
        cooldown_seconds: int = 45,
        retry_policy: dict[str, Any] | None = None,
    ):
        self._search = search_client
        self._scraper = scraper
        self._cooldown_seconds = max(0, int(cooldown_seconds))
        self._retry_policy = dict(retry_policy or {})
        self._topic_last_ts: dict[str, float] = {}

    def execute(
        self,
        *,
        plan: WebQueryPlan,
        decision: WebPolicyDecision,
        classification: QueryClassification,
        freshness: FreshnessAssessment,
        preferred_domains: list[str] | None = None,
        geo_hint: str = "",
    ) -> SearchExecutionResult:
        if decision.mode == WebSearchMode.NO_SEARCH:
            return SearchExecutionResult(
                results=[],
                fetched_pages={},
                queries_used=[],
                domain_filter=[],
                recency_days=None,
                scout_queries_used=[],
                focused_queries_used=[],
                retries_used=0,
                cooldown_applied=False,
                query_runs=[],
                fetch_summary={},
                warnings=[],
            )

        policy = self._policy_for_mode(mode=decision.mode, decision=decision)
        candidate_limit = _candidate_limit(policy.max_sources)
        intent = _compat_query_intent(query=plan.all_queries()[0] if plan.all_queries() else "", classification=classification)
        recency_days = _recency_days(intent=intent, freshness=freshness)
        domain_filter = _domain_filter_for_intent(intent=intent, preferred_domains=preferred_domains, geo_hint=geo_hint)
        topic_key = _topic_key(plan=plan, classification=classification)

        cooldown_applied = False
        now = time.time()
        previous = self._topic_last_ts.get(topic_key)
        if (
            self._cooldown_seconds > 0
            and previous is not None
            and now - float(previous) < float(self._cooldown_seconds)
        ):
            policy = _ModePolicy(
                max_queries=max(1, policy.max_queries - 1),
                max_sources=max(1, policy.max_sources - 1),
                max_fetches=max(1, policy.max_fetches - 1),
                scout_cap=max(1, min(policy.scout_cap, policy.max_queries)),
                focused_cap=max(0, policy.focused_cap),
                retries=policy.retries,
            )
            cooldown_applied = True

        collected: list[SearchResult] = []
        seen_urls: set[str] = set()
        queries_used: list[str] = []
        scout_used: list[str] = []
        focused_used: list[str] = []
        retries_used = 0
        search_errors = 0
        query_runs: list[dict[str, Any]] = []

        scout_candidates = list(plan.scout_queries or plan.all_queries())
        focused_candidates = list(plan.focused_queries or [])
        fallback_candidates = list(plan.fallback_queries or [])

        scout_limit = min(policy.scout_cap, policy.max_queries)
        used, retries_delta, errors_delta, query_stats = self._run_query_batch(
            candidates=scout_candidates,
            query_limit=scout_limit,
            source_limit=candidate_limit,
            collected=collected,
            seen_urls=seen_urls,
            recency_days=recency_days,
            domain_filter=domain_filter,
            volatile=bool(freshness.needs_refresh),
            query_intent=intent,
            retries=policy.retries,
            stage_label="scout",
        )
        scout_used.extend(used)
        queries_used.extend(used)
        retries_used += retries_delta
        search_errors += errors_delta
        query_runs.extend(query_stats)

        remaining_queries = max(0, policy.max_queries - len(queries_used))
        should_focus = bool(
            remaining_queries > 0
            and (
                decision.mode in {WebSearchMode.SOFT_SEARCH, WebSearchMode.TARGETED_SEARCH, WebSearchMode.DEEP_SEARCH}
                or freshness.needs_refresh
                or len(collected) < max(1, min(candidate_limit, policy.max_sources + 2))
            )
        )
        if should_focus:
            focused_limit = min(policy.focused_cap + len(fallback_candidates), remaining_queries)
            focused_pool = list(focused_candidates)
            if len(collected) < max(1, candidate_limit):
                focused_pool.extend(fallback_candidates)
            used, retries_delta, errors_delta, query_stats = self._run_query_batch(
                candidates=focused_pool,
                query_limit=focused_limit,
                source_limit=candidate_limit,
                collected=collected,
                seen_urls=seen_urls,
                recency_days=recency_days,
                domain_filter=domain_filter,
                volatile=True if decision.mode in {WebSearchMode.TARGETED_SEARCH, WebSearchMode.DEEP_SEARCH} else bool(freshness.needs_refresh),
                query_intent=intent,
                retries=policy.retries,
                stage_label="focused",
            )
            focused_used.extend(used)
            queries_used.extend(used)
            retries_used += retries_delta
            search_errors += errors_delta
            query_runs.extend(query_stats)

        if search_errors > 0 and not collected:
            raise RuntimeError("search execution failed after retries")

        fetched_pages, fetch_summary = self._fetch_pages(
            rows=_pick_fetch_rows(collected, max_fetches=policy.max_fetches),
            max_fetches=policy.max_fetches,
        )
        self._topic_last_ts[topic_key] = now

        warnings: list[str] = []
        if cooldown_applied:
            warnings.append("cooldown_applied")
        if search_errors > 0:
            warnings.append(f"query_errors:{search_errors}")
        if int(fetch_summary.get("snippet_fallback_count") or 0) > 0:
            warnings.append(f"fetch_fallbacks:{int(fetch_summary.get('snippet_fallback_count') or 0)}")
        if int(fetch_summary.get("failed_fetches") or 0) > 0:
            warnings.append(f"fetch_failures:{int(fetch_summary.get('failed_fetches') or 0)}")

        return SearchExecutionResult(
            results=collected[: max(1, candidate_limit)],
            fetched_pages=fetched_pages,
            queries_used=queries_used[: policy.max_queries],
            domain_filter=list(domain_filter),
            recency_days=recency_days,
            scout_queries_used=scout_used,
            focused_queries_used=focused_used,
            retries_used=int(retries_used),
            cooldown_applied=bool(cooldown_applied),
            query_runs=query_runs,
            fetch_summary=fetch_summary,
            warnings=warnings,
        )

    def _run_query_batch(
        self,
        *,
        candidates: list[str],
        query_limit: int,
        source_limit: int,
        collected: list[SearchResult],
        seen_urls: set[str],
        recency_days: int | None,
        domain_filter: list[str],
        volatile: bool,
        query_intent: str,
        retries: int,
        stage_label: str,
    ) -> tuple[list[str], int, int, list[dict[str, Any]]]:
        used: list[str] = []
        retries_used = 0
        errors_count = 0
        query_runs: list[dict[str, Any]] = []
        for raw_query in list(candidates or []):
            if len(used) >= max(0, int(query_limit)):
                break
            if len(collected) >= max(1, int(source_limit)):
                break
            query_text = str(raw_query or "").strip()
            if not query_text:
                continue
            dynamic_filter = _merge_domain_filters(
                base=domain_filter,
                site_domain=_extract_site_domain(query_text),
            )
            rows, attempts, failed = self._search_with_retry(
                query=query_text,
                recency_days=recency_days,
                domain_filter=dynamic_filter,
                k=_per_query_k(source_limit),
                volatile=volatile,
                query_intent=query_intent,
                retries=retries,
            )
            retries_used += max(0, attempts - 1)
            if failed:
                errors_count += 1
            used.append(query_text)
            query_runs.append(
                {
                    "query": query_text,
                    "stage": str(stage_label or "").strip().lower() or "scout",
                    "attempts": int(attempts),
                    "failed": bool(failed),
                    "result_count": int(len(list(rows or []))),
                    "domain_filter": list(dynamic_filter),
                    "recency_days": recency_days,
                    "volatile": bool(volatile),
                    "site_filter": str(_extract_site_domain(query_text) or ""),
                    "top_domains": _top_domains(rows, limit=4),
                }
            )
            for item in list(rows or []):
                if len(collected) >= max(1, int(source_limit)):
                    break
                url = str(item.url or "").strip().lower()
                if not url or url in seen_urls:
                    continue
                seen_urls.add(url)
                collected.append(item)
        return (used, retries_used, errors_count, query_runs)

    def _search_with_retry(
        self,
        *,
        query: str,
        recency_days: int | None,
        domain_filter: list[str],
        k: int,
        volatile: bool,
        query_intent: str,
        retries: int,
    ) -> tuple[list[SearchResult], int, bool]:
        attempts = max(1, int(retries) + 1)
        backoff_ms = max(50, int(self._retry_policy.get("backoff_ms") or 250))
        last_error: Exception | None = None
        for idx in range(attempts):
            try:
                rows = self._search.search(
                    query,
                    recency_days=recency_days,
                    domain_filter=domain_filter or None,
                    k=max(1, int(k)),
                    volatile=bool(volatile),
                    query_intent=query_intent,
                )
                return (list(rows or []), idx + 1, False)
            except Exception as exc:
                last_error = exc
                if idx + 1 >= attempts:
                    break
                time.sleep(float(backoff_ms) / 1000.0 * float(idx + 1))
        if last_error is not None:
            return ([], attempts, True)
        return ([], 1, False)

    def _fetch_pages(self, *, rows: list[SearchResult], max_fetches: int) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
        fetched_at = dt.datetime.now(dt.timezone.utc).isoformat()
        fetched_pages: dict[str, dict[str, Any]] = {}
        attempted = 0
        fetched = 0
        snippet_fallback_count = 0
        failed_fetches = 0
        domains: list[str] = []
        for item in list(rows or [])[: max(1, int(max_fetches))]:
            attempted += 1
            url = str(item.url or "").strip()
            if not url:
                continue
            domain = str(item.source or "").strip().lower()
            if domain and domain not in domains:
                domains.append(domain)
            try:
                page = self._scraper.scrape(url)
            except Exception:
                snippet_fallback_count += 1
                failed_fetches += 1
                fetched_pages[url] = {
                    "url": url,
                    "domain": domain,
                    "status": 0,
                    "final_url": url,
                    "title": str(item.title or "").strip(),
                    "text": "",
                    "snippet": str(item.snippet or "").strip(),
                    "published_date": str(item.published_date or "").strip(),
                    "fetched_at": fetched_at,
                    "clean_method": "search_snippet_fallback",
                    "removed_blocks": 0,
                    "raw_len": 0,
                    "clean_len": 0,
                }
                continue
            meta = dict(getattr(page, "metadata", {}) or {})
            status_code = int(getattr(page, "status_code", 200) or 200)
            clean_method = str(meta.get("clean_method") or "").strip()
            if status_code > 0:
                fetched += 1
            if clean_method == "search_snippet_fallback":
                snippet_fallback_count += 1
            if status_code <= 0:
                failed_fetches += 1
            fetched_pages[url] = {
                "url": url,
                "domain": domain,
                "status": status_code,
                "final_url": str(getattr(page, "final_url", "") or url),
                "title": str(getattr(page, "title", "") or item.title or "").strip(),
                "text": str(getattr(page, "text", "") or "").strip(),
                "snippet": str(item.snippet or "").strip(),
                "published_date": str(item.published_date or "").strip(),
                "fetched_at": fetched_at,
                "clean_method": clean_method,
                "removed_blocks": int(meta.get("removed_blocks") or 0),
                "raw_len": int(meta.get("raw_len") or 0),
                "clean_len": int(meta.get("clean_len") or 0),
            }
        return fetched_pages, {
            "attempted": int(attempted),
            "fetched": int(fetched),
            "snippet_fallback_count": int(snippet_fallback_count),
            "failed_fetches": int(failed_fetches),
            "domains": list(domains),
        }

    def _policy_for_mode(self, *, mode: WebSearchMode, decision: WebPolicyDecision) -> _ModePolicy:
        budget = decision.budget
        max_queries = max(0, int(getattr(budget, "max_queries", 0)))
        max_sources = max(0, int(getattr(budget, "max_sources", 0)))
        max_fetches = max(0, int(budget.effective_max_fetches() if hasattr(budget, "effective_max_fetches") else getattr(budget, "max_pages", 0)))

        mode_defaults = {
            WebSearchMode.VERIFY_ONLY: _ModePolicy(max_queries=2, max_sources=2, max_fetches=1, scout_cap=1, focused_cap=1, retries=0),
            WebSearchMode.SOFT_SEARCH: _ModePolicy(max_queries=3, max_sources=3, max_fetches=2, scout_cap=1, focused_cap=2, retries=1),
            WebSearchMode.TARGETED_SEARCH: _ModePolicy(max_queries=4, max_sources=5, max_fetches=3, scout_cap=2, focused_cap=2, retries=1),
            WebSearchMode.DEEP_SEARCH: _ModePolicy(max_queries=6, max_sources=8, max_fetches=5, scout_cap=2, focused_cap=3, retries=2),
        }
        base = mode_defaults.get(mode, _ModePolicy(max_queries=2, max_sources=2, max_fetches=1, scout_cap=1, focused_cap=1, retries=0))

        retries_override = self._retry_for_mode(mode)
        return _ModePolicy(
            max_queries=max(1, min(base.max_queries, max_queries) if max_queries > 0 else base.max_queries),
            max_sources=max(1, min(base.max_sources, max_sources) if max_sources > 0 else base.max_sources),
            max_fetches=max(1, min(base.max_fetches, max_fetches) if max_fetches > 0 else base.max_fetches),
            scout_cap=max(1, min(base.scout_cap, max(1, max_queries or base.max_queries))),
            focused_cap=max(0, base.focused_cap),
            retries=max(0, int(retries_override)),
        )

    def _retry_for_mode(self, mode: WebSearchMode) -> int:
        mode_key = str(mode.value or "").strip().lower()
        fallback = int(self._retry_policy.get("default_attempts") or 1) - 1
        if mode_key in self._retry_policy:
            try:
                return max(0, int(self._retry_policy.get(mode_key)) - 1)
            except Exception:
                return max(0, fallback)
        try:
            if mode == WebSearchMode.DEEP_SEARCH:
                return max(0, int(self._retry_policy.get("deep_search") or 2) - 1)
            if mode == WebSearchMode.TARGETED_SEARCH:
                return max(0, int(self._retry_policy.get("targeted_search") or 2) - 1)
            if mode == WebSearchMode.SOFT_SEARCH:
                return max(0, int(self._retry_policy.get("soft_search") or 1) - 1)
            if mode == WebSearchMode.VERIFY_ONLY:
                return max(0, int(self._retry_policy.get("verify_only") or 1) - 1)
        except Exception:
            return max(0, fallback)
        return max(0, fallback)


def _topic_key(*, plan: WebQueryPlan, classification: QueryClassification) -> str:
    base_query = str((plan.scout_queries or plan.all_queries() or [""])[0] or "").strip().lower()
    category = str(classification.primary_category or "").strip().lower()
    return f"{category}|{base_query[:120]}"


def _candidate_limit(max_sources: int) -> int:
    base = max(1, int(max_sources or 1))
    return min(12, max(base + 2, base * 2))


def _per_query_k(source_limit: int) -> int:
    base = max(1, int(source_limit or 1))
    return min(16, base + 2)


def _pick_fetch_rows(rows: list[SearchResult], *, max_fetches: int) -> list[SearchResult]:
    limit = max(1, int(max_fetches or 1))
    pool = list(rows or [])
    if len(pool) <= limit:
        return pool

    selected: list[SearchResult] = []
    selected_urls: set[str] = set()
    seen_domains: set[str] = set()

    # Pass 1: prioritize domain diversity.
    for item in pool:
        if len(selected) >= limit:
            break
        url = str(getattr(item, "url", "") or "").strip().lower()
        if not url or url in selected_urls:
            continue
        domain = str(getattr(item, "source", "") or "").strip().lower()
        if domain and domain in seen_domains:
            continue
        selected.append(item)
        selected_urls.add(url)
        if domain:
            seen_domains.add(domain)

    # Pass 2: fill remaining slots by score order.
    for item in pool:
        if len(selected) >= limit:
            break
        url = str(getattr(item, "url", "") or "").strip().lower()
        if not url or url in selected_urls:
            continue
        selected.append(item)
        selected_urls.add(url)

    return selected[:limit]


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


def _merge_domain_filters(*, base: list[str], site_domain: str) -> list[str]:
    out: list[str] = []
    for row in list(base or []):
        item = str(row or "").strip().lower()
        if item and item not in out:
            out.append(item)
    extra = str(site_domain or "").strip().lower()
    if extra and extra not in out:
        out.append(extra)
    return out


def _extract_site_domain(query: str) -> str:
    src = str(query or "").strip().lower()
    if "site:" not in src:
        return ""
    for token in src.split():
        if not token.startswith("site:"):
            continue
        value = token.replace("site:", "", 1).strip().strip("/")
        if value.startswith("www."):
            value = value[4:]
        return value
    return ""


def _recency_days(*, intent: str, freshness: FreshnessAssessment) -> int | None:
    if intent in {"fx_rate", "weather"}:
        return 1
    if intent == "news_release":
        return 3
    if freshness.needs_refresh:
        return 7
    return None


def _compat_query_intent(*, query: str, classification: QueryClassification) -> str:
    text = str(query or "").strip().lower()
    if any(token in text for token in ("курс", "usd", "eur", "uah", "exchange rate", "forex")):
        return "fx_rate"
    if any(token in text for token in ("weather", "forecast", "погод", "температур")):
        return "weather"
    if any(token in text for token in ("news", "release", "changelog", "новост", "релиз", "верс")):
        return "news_release"
    if classification.query_type == "external_factual" and classification.primary_category in {"version", "news"}:
        return "news_release"
    return "generic"


def _domain_filter_for_intent(*, intent: str, preferred_domains: list[str] | None = None, geo_hint: str = "") -> list[str]:
    # Hard domain filtering is disabled to keep a healthy mix:
    # trusted sources + open web sources. Location relevance is handled
    # via geo-biased query planning and ranking bonuses.
    _ = (intent, preferred_domains, geo_hint)
    return []
