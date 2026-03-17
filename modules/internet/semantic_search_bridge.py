from __future__ import annotations

"""Legacy bridge between semantic NLU outputs and the pre-web-v2 search stack.

The active internet pipeline lives under ``modules.internet.web.*``.
This adapter is retained only for backward compatibility.
"""

import re
from dataclasses import replace
from typing import Any
from urllib.parse import urlparse

from modules.internet.consensus_engine import ConsensusEngine
from modules.internet.source_ranker import SourceRanker
from modules.nlu.types import SearchTask, UnderstandingResult


class SemanticSearchBridge:
    """
    Legacy adapter layer between semantic NLU outputs and older internet search code.

    Responsibilities:
    - convert semantic SearchTask objects into search-layer compatible parameters
    - apply source quality ranking/diversity post-processing
    - prepare normalized claim payloads for consensus hooks
    """

    _NUMBER_RE = re.compile(r"-?\d+(?:[.,]\d+)?")

    _QUERY_INTENT_BY_TASK_KIND: dict[str, str] = {
        "weather": "weather",
        "currency_rate": "fx_rate",
        "news": "news_release",
        "web_search": "generic",
        "search": "generic",
        "generic": "generic",
    }

    _DEFAULT_RECENCY_DAYS: dict[str, int | None] = {
        "weather": 2,
        "currency_rate": 1,
        "news": 3,
        "web_search": 14,
        "search": 14,
        "generic": 14,
    }

    _DEFAULT_RESULT_LIMIT: dict[str, int] = {
        "weather": 6,
        "currency_rate": 6,
        "news": 8,
        "web_search": 8,
        "search": 8,
        "generic": 8,
    }

    _CLAIM_PREDICATE_BY_TASK_KIND: dict[str, str] = {
        "weather": "weather_snapshot",
        "currency_rate": "currency_rate",
        "news": "news_headline",
        "web_search": "web_fact",
        "search": "web_fact",
        "generic": "web_fact",
    }

    def __init__(
        self,
        *,
        source_ranker: SourceRanker | None = None,
        consensus_engine: ConsensusEngine | None = None,
    ) -> None:
        self._source_ranker = source_ranker or SourceRanker()
        self._consensus_engine = consensus_engine

    def tasks_from_understanding(self, understanding: UnderstandingResult) -> list[SearchTask]:
        tasks = list(getattr(understanding, "search_tasks", []) or [])
        if not tasks:
            return []

        normalized = self._normalize_tasks(tasks)
        enriched: list[SearchTask] = []
        for task in normalized:
            kwargs = self._search_kwargs_for_task(task)
            meta = dict(task.meta or {})
            meta["query_intent"] = kwargs["query_intent"]
            meta["search_kwargs"] = kwargs
            enriched.append(replace(task, meta=meta))

        return enriched

    def rank_results(self, results: list[Any], task: SearchTask) -> list[Any]:
        if not results:
            return []

        query_intent = self._task_query_intent(task)
        limit = self._task_limit(task)

        ranked = self._source_ranker.select_diverse_results(
            results=list(results),
            query_intent=query_intent,
            limit=limit,
        )
        return ranked

    def prepare_claim_inputs(self, results: list[Any], task: SearchTask) -> list[dict[str, Any]]:
        if not results:
            return []

        task_kind = self._normalize_task_kind(task.kind)
        predicate = self._CLAIM_PREDICATE_BY_TASK_KIND.get(task_kind, "web_fact")
        subject = self._task_subject(task)

        claims: list[dict[str, Any]] = []
        for row in results:
            source_domain = self._extract_domain(row)
            if not source_domain:
                continue

            claim_value = self._extract_claim_value(row=row, task_kind=task_kind)
            if not claim_value:
                continue

            claims.append(
                {
                    "subject": subject,
                    "predicate": predicate,
                    "value": claim_value,
                    "source_domain": source_domain,
                    "confidence": self._extract_result_confidence(row),
                    "published_at": self._extract_text_field(row, "published_date"),
                    "url": self._extract_text_field(row, "url"),
                    "title": self._extract_text_field(row, "title"),
                    "snippet": self._extract_text_field(row, "snippet"),
                }
            )

        # Hook point for future automated consensus integration.
        # Example:
        # if self._consensus_engine is not None:
        #     consensus = self._consensus_engine.build_consensus([EvidenceClaim(**x) for x in claims])
        #     ...
        _ = self._consensus_engine
        return claims

    def _normalize_tasks(self, tasks: list[SearchTask]) -> list[SearchTask]:
        ordered = sorted(tasks, key=lambda t: (int(t.priority), str(t.kind), str(t.query)))
        dedup: dict[tuple[str, str, str], SearchTask] = {}

        for task in ordered:
            kind = self._normalize_task_kind(task.kind)
            query = self._normalize_text(task.query)
            location = self._normalize_text(task.location)
            if not query:
                continue

            key = (kind, query, location)
            existing = dedup.get(key)
            if existing is None or int(task.priority) < int(existing.priority):
                dedup[key] = replace(
                    task,
                    kind=kind,
                    query=query,
                    location=location,
                    meta=dict(task.meta or {}),
                )

        return list(dedup.values())

    def _search_kwargs_for_task(self, task: SearchTask) -> dict[str, Any]:
        task_kind = self._normalize_task_kind(task.kind)
        meta = dict(task.meta or {})

        query_intent = self._task_query_intent(task)
        recency_days = meta.get("recency_days")
        if recency_days is None:
            recency_days = self._DEFAULT_RECENCY_DAYS.get(task_kind, 14)

        limit = meta.get("k")
        if limit is None:
            limit = meta.get("limit")
        if limit is None:
            limit = self._task_limit(task)

        domain_filter = meta.get("domain_filter")
        if isinstance(domain_filter, list):
            domain_filter = [str(x).strip().lower() for x in domain_filter if str(x).strip()]
        else:
            domain_filter = None

        volatile = bool(meta.get("volatile", False))

        return {
            "query": str(task.query or "").strip(),
            "recency_days": recency_days,
            "domain_filter": domain_filter,
            "k": int(limit),
            "volatile": volatile,
            "query_intent": query_intent,
        }

    def _task_query_intent(self, task: SearchTask) -> str:
        meta_intent = self._extract_meta_query_intent(task)
        if meta_intent:
            return meta_intent
        task_kind = self._normalize_task_kind(task.kind)
        return self._QUERY_INTENT_BY_TASK_KIND.get(task_kind, "generic")

    def _extract_meta_query_intent(self, task: SearchTask) -> str:
        meta = dict(task.meta or {})
        raw = str(meta.get("query_intent") or "").strip().lower()
        if raw in {"weather", "fx_rate", "news_release", "generic"}:
            return raw
        return ""

    def _task_limit(self, task: SearchTask) -> int:
        meta = dict(task.meta or {})
        raw = meta.get("max_results", meta.get("result_limit"))
        if raw is None:
            return self._DEFAULT_RESULT_LIMIT.get(self._normalize_task_kind(task.kind), 8)
        try:
            return max(1, min(20, int(raw)))
        except Exception:
            return self._DEFAULT_RESULT_LIMIT.get(self._normalize_task_kind(task.kind), 8)

    def _task_subject(self, task: SearchTask) -> str:
        meta = dict(task.meta or {})
        if str(meta.get("subject") or "").strip():
            return str(meta.get("subject")).strip()
        if str(task.location or "").strip():
            return str(task.location).strip()
        if str(meta.get("location") or "").strip():
            return str(meta.get("location")).strip()
        return self._normalize_task_kind(task.kind)

    def _extract_claim_value(self, *, row: Any, task_kind: str) -> str:
        title = self._extract_text_field(row, "title")
        snippet = self._extract_text_field(row, "snippet")
        blob = " ".join(x for x in (title, snippet) if x).strip()
        if not blob:
            return ""

        if task_kind in {"weather", "currency_rate"}:
            numeric = self._extract_first_number(blob)
            if numeric:
                return numeric

        if task_kind == "news":
            return title or snippet[:180]

        return title or snippet[:180]

    def _extract_result_confidence(self, row: Any) -> float:
        score_raw = self._extract_numeric_field(row, "score")
        if score_raw <= 0.0:
            return 0.55
        # Convert open-ended ranking score into [0, 1] interval.
        confidence = score_raw / (1.0 + score_raw)
        if confidence < 0.0:
            return 0.0
        if confidence > 1.0:
            return 1.0
        return round(confidence, 4)

    def _extract_domain(self, row: Any) -> str:
        source = self._extract_text_field(row, "source")
        if source:
            return self._normalize_domain(source)
        url = self._extract_text_field(row, "url")
        if not url:
            return ""
        return self._normalize_domain(urlparse(url).netloc)

    @staticmethod
    def _normalize_domain(value: str) -> str:
        host = str(value or "").strip().lower()
        if host.startswith("www."):
            host = host[4:]
        return host

    @staticmethod
    def _normalize_task_kind(value: str) -> str:
        token = str(value or "").strip().lower()
        return token or "generic"

    @staticmethod
    def _normalize_text(value: str) -> str:
        return str(value or "").strip()

    @staticmethod
    def _extract_text_field(row: Any, field: str) -> str:
        if isinstance(row, dict):
            return str(row.get(field) or "").strip()
        return str(getattr(row, field, "") or "").strip()

    @staticmethod
    def _extract_numeric_field(row: Any, field: str) -> float:
        raw: Any
        if isinstance(row, dict):
            raw = row.get(field)
        else:
            raw = getattr(row, field, 0.0)
        try:
            return float(raw or 0.0)
        except Exception:
            return 0.0

    def _extract_first_number(self, text: str) -> str:
        match = self._NUMBER_RE.search(str(text or ""))
        if not match:
            return ""
        return match.group(0).replace(",", ".")
