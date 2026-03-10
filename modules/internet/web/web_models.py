from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class WebSearchMode(str, Enum):
    NO_SEARCH = "NO_SEARCH"
    VERIFY_ONLY = "VERIFY_ONLY"
    SOFT_SEARCH = "SOFT_SEARCH"
    TARGETED_SEARCH = "TARGETED_SEARCH"
    DEEP_SEARCH = "DEEP_SEARCH"


@dataclass(frozen=True)
class SearchBudget:
    max_queries: int = 0
    max_sources: int = 0
    max_pages: int = 0

    def to_dict(self) -> dict[str, int]:
        return {
            "max_queries": int(self.max_queries),
            "max_sources": int(self.max_sources),
            "max_pages": int(self.max_pages),
        }


@dataclass(frozen=True)
class QueryClassification:
    query_type: str
    primary_category: str
    is_temporal: bool
    is_local_project_question: bool
    is_external_fact_question: bool
    is_ambiguous: bool
    requires_freshness: bool
    stakes_level: str
    expected_search_need: str
    explicit_search_intent: bool
    temporal_markers: list[str] = field(default_factory=list)
    category_hits: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_type": str(self.query_type),
            "primary_category": str(self.primary_category),
            "is_temporal": bool(self.is_temporal),
            "is_local_project_question": bool(self.is_local_project_question),
            "is_external_fact_question": bool(self.is_external_fact_question),
            "is_ambiguous": bool(self.is_ambiguous),
            "requires_freshness": bool(self.requires_freshness),
            "stakes_level": str(self.stakes_level),
            "expected_search_need": str(self.expected_search_need),
            "explicit_search_intent": bool(self.explicit_search_intent),
            "temporal_markers": list(self.temporal_markers),
            "category_hits": list(self.category_hits),
        }


@dataclass(frozen=True)
class ConfidenceAssessment:
    score: float
    level: str
    reasons: list[str] = field(default_factory=list)
    breakdown: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": float(self.score),
            "level": str(self.level),
            "reasons": list(self.reasons),
            "breakdown": {str(k): float(v) for k, v in dict(self.breakdown or {}).items()},
        }


@dataclass(frozen=True)
class FreshnessAssessment:
    temporal_risk: float
    risk_level: str
    needs_refresh: bool
    stale_web_fact_detected: bool
    category: str
    reasons: list[str] = field(default_factory=list)
    breakdown: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "temporal_risk": float(self.temporal_risk),
            "risk_level": str(self.risk_level),
            "needs_refresh": bool(self.needs_refresh),
            "stale_web_fact_detected": bool(self.stale_web_fact_detected),
            "category": str(self.category),
            "reasons": list(self.reasons),
            "breakdown": {str(k): float(v) for k, v in dict(self.breakdown or {}).items()},
        }


@dataclass(frozen=True)
class WebPolicyDecision:
    mode: WebSearchMode
    should_search: bool
    web_need_score: float
    reason: str
    budget: SearchBudget
    decision_breakdown: dict[str, float] = field(default_factory=dict)
    local_scope_cap_applied: bool = False
    category_penalty_applied: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": str(self.mode.value),
            "should_search": bool(self.should_search),
            "web_need_score": float(self.web_need_score),
            "reason": str(self.reason),
            "budget": self.budget.to_dict(),
            "decision_breakdown": {str(k): float(v) for k, v in dict(self.decision_breakdown or {}).items()},
            "local_scope_cap_applied": bool(self.local_scope_cap_applied),
            "category_penalty_applied": float(self.category_penalty_applied),
        }


@dataclass(frozen=True)
class WebQueryPlan:
    mode: WebSearchMode
    scout_queries: list[str] = field(default_factory=list)
    focused_queries: list[str] = field(default_factory=list)
    fallback_queries: list[str] = field(default_factory=list)
    preferred_domains: list[str] = field(default_factory=list)

    def all_queries(self) -> list[str]:
        out: list[str] = []
        for row in list(self.scout_queries) + list(self.focused_queries) + list(self.fallback_queries):
            query = str(row or "").strip()
            if query and query not in out:
                out.append(query)
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": str(self.mode.value),
            "scout_queries": list(self.scout_queries),
            "focused_queries": list(self.focused_queries),
            "fallback_queries": list(self.fallback_queries),
            "preferred_domains": list(self.preferred_domains),
        }


@dataclass(frozen=True)
class WebEvidenceItem:
    title: str
    url: str
    domain: str
    snippet: str = ""
    text: str = ""
    published_date: str = ""
    fetched_at: str = ""
    trust_score: float = 0.0
    trust_tier: str = "unknown"
    source_type: str = "search"
    clean_method: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": str(self.title),
            "url": str(self.url),
            "domain": str(self.domain),
            "snippet": str(self.snippet),
            "text": str(self.text),
            "published_date": str(self.published_date),
            "fetched_at": str(self.fetched_at),
            "trust_score": float(self.trust_score),
            "trust_tier": str(self.trust_tier),
            "source_type": str(self.source_type),
            "clean_method": str(self.clean_method),
        }


@dataclass(frozen=True)
class WebEvidencePack:
    items: list[WebEvidenceItem] = field(default_factory=list)
    compact_citations: list[str] = field(default_factory=list)
    conflicting_sources: bool = False
    summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "items": [x.to_dict() for x in list(self.items)],
            "compact_citations": list(self.compact_citations),
            "conflicting_sources": bool(self.conflicting_sources),
            "summary": str(self.summary),
        }
