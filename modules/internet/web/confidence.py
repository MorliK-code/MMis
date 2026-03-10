from __future__ import annotations

from typing import Any

from modules.internet.web.web_models import ConfidenceAssessment, QueryClassification


def assess_confidence(
    *,
    query: str,
    classification: QueryClassification,
    retrieved_memories: list[Any] | None = None,
    memory_context: dict[str, Any] | None = None,
) -> ConfidenceAssessment:
    rows = list(retrieved_memories or [])
    selected = list(_as_dict(memory_context).get("selected") or [])
    pool = rows if rows else selected

    local_hits = 0
    web_hits = 0
    doc_hits = 0
    for row in pool:
        item = _as_dict(row)
        source = str(item.get("source") or "").strip().lower()
        topic = str(item.get("topic") or "").strip().lower()
        text = str(item.get("text") or item.get("content") or "").strip().lower()
        is_web = bool(
            topic.startswith("web:")
            or source in {"web", "search", "internet"}
            or text.startswith("[web]")
            or text.startswith("[web_search]")
        )
        if is_web:
            web_hits += 1
            continue
        local_hits += 1
        if source in {"document", "doc", "project_doc", "knowledge_base"}:
            doc_hits += 1

    breakdown = {
        "base": 0.62,
        "local_hits_bonus": min(0.20, 0.06 * float(local_hits)),
        "doc_hits_bonus": min(0.10, 0.05 * float(doc_hits)),
        "local_scope_bonus": 0.12 if classification.is_local_project_question else 0.0,
        "external_penalty": -0.14 if classification.is_external_fact_question else 0.0,
        "temporal_penalty": -0.18 if classification.requires_freshness else 0.0,
        "ambiguity_penalty": -0.22 if classification.is_ambiguous else 0.0,
        "missing_evidence_penalty": -0.10 if local_hits <= 0 else 0.0,
    }

    score = 0.0
    for value in breakdown.values():
        score += float(value)
    score = max(0.0, min(1.0, score))

    reasons: list[str] = []
    if local_hits > 0:
        reasons.append(f"local_hits:{local_hits}")
    if doc_hits > 0:
        reasons.append(f"doc_hits:{doc_hits}")
    if web_hits > 0:
        reasons.append(f"web_hits:{web_hits}")
    if classification.requires_freshness:
        reasons.append("freshness_required")
    if classification.is_ambiguous:
        reasons.append("ambiguous_query")
    if classification.is_local_project_question:
        reasons.append("local_project_scope")

    if score >= 0.75:
        level = "high"
    elif score >= 0.50:
        level = "medium"
    else:
        level = "low"

    return ConfidenceAssessment(
        score=score,
        level=level,
        reasons=reasons,
        breakdown=breakdown,
    )


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}

