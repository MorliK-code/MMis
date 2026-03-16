from __future__ import annotations

from modules.internet.web.web_models import (
    ConfidenceAssessment,
    FreshnessAssessment,
    QueryClassification,
    SearchBudget,
    WebEvidence,
    WebEvidencePack,
    WebPolicyDecision,
    WebQueryPlan,
    WebSearchRequest,
    WebSearchResult,
    WebSearchMode,
)

__all__ = [
    "WebStageV2",
    "WebStageConfig",
    "WebSearchMode",
    "SearchBudget",
    "WebSearchRequest",
    "WebSearchResult",
    "QueryClassification",
    "ConfidenceAssessment",
    "FreshnessAssessment",
    "WebPolicyDecision",
    "WebQueryPlan",
    "WebEvidence",
    "WebEvidencePack",
]


def __getattr__(name: str):
    if name in {"WebStageV2", "WebStageConfig"}:
        from modules.internet.web.stage import WebStageConfig, WebStageV2

        return {"WebStageV2": WebStageV2, "WebStageConfig": WebStageConfig}[name]
    raise AttributeError(name)
