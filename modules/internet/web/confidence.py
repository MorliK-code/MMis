from __future__ import annotations

import datetime as dt
import re
from typing import Any

from modules.internet.web.web_models import ConfidenceAssessment, QueryClassification


_WORD_RE = re.compile(r"[A-Za-zА-Яа-яЁёЇїІіЄєҐґ0-9_]+")
_STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "что",
    "это",
    "как",
    "или",
    "and",
    "this",
    "that",
}


def assess_confidence(
    *,
    query: str,
    classification: QueryClassification,
    retrieved_memories: list[Any] | None = None,
    memory_context: dict[str, Any] | None = None,
    web_cache_items: list[Any] | None = None,
    conflict_count: int | None = None,
    query_entities: list[str] | None = None,
) -> ConfidenceAssessment:
    rows = list(retrieved_memories or [])
    selected = list(_as_dict(memory_context).get("selected") or [])
    pool = rows if rows else selected

    local_hits = 0
    web_hits = 0
    doc_hits = 0
    inferred_conflicts = 0
    entity_tokens: set[str] = set()
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
        memory_type = str(item.get("memory_type") or item.get("type") or "").strip().lower()
        if source in {"document", "doc", "project_doc", "knowledge_base"} or memory_type in {"document", "document_chunk"}:
            doc_hits += 1
        status = str(item.get("status") or "").strip().lower()
        metadata = _as_dict(item.get("metadata"))
        if status in {"superseded", "stale"}:
            inferred_conflicts += 1
        if _as_dict(metadata.get("conflict_resolution")):
            inferred_conflicts += 1
        entity_tokens.update(_extract_item_entities(item))

    conflicts = max(0, int(conflict_count if conflict_count is not None else inferred_conflicts))
    cache_items = list(web_cache_items or [])
    if not cache_items:
        cache_items = [x for x in pool if _is_web_item(_as_dict(x))]
    has_fresh_web_cache = _has_fresh_web_cache(cache_items)

    entities_from_query = set(_extract_query_entities(query, explicit=query_entities))
    overlap_score = _entity_overlap_score(query_entities=entities_from_query, item_entities=entity_tokens)
    specificity = _query_specificity(query)

    breakdown = {
        "base": 0.38,
        "memory_hits_bonus": min(0.20, 0.04 * float(local_hits)),
        "local_docs_bonus": min(0.14, 0.05 * float(doc_hits)),
        "entity_overlap_bonus": min(0.16, 0.16 * float(overlap_score)),
        "fresh_web_cache_bonus": 0.12 if has_fresh_web_cache else 0.0,
        "query_specificity_bonus": min(0.12, 0.12 * float(specificity)),
        "conflicts_penalty": -min(0.24, 0.08 * float(conflicts)),
        "external_penalty": -0.10 if classification.is_external_fact_question and not has_fresh_web_cache else 0.0,
        "temporal_penalty": -0.12 if classification.requires_freshness and not has_fresh_web_cache else 0.0,
        "ambiguity_penalty": -0.18 if classification.is_ambiguous else 0.0,
        "missing_memory_penalty": -0.08 if local_hits <= 0 else 0.0,
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
    if overlap_score > 0.0:
        reasons.append(f"entity_overlap:{overlap_score:.2f}")
    if specificity > 0.0:
        reasons.append(f"query_specificity:{specificity:.2f}")
    if web_hits > 0:
        reasons.append(f"web_hits:{web_hits}")
    if has_fresh_web_cache:
        reasons.append("fresh_web_cache")
    if conflicts > 0:
        reasons.append(f"conflicts:{conflicts}")
    if classification.requires_freshness:
        reasons.append("freshness_required")
    if classification.is_ambiguous:
        reasons.append("ambiguous_query")
    if classification.is_local_project_question:
        reasons.append("local_project_scope")

    if score >= 0.74:
        level = "high"
    elif score >= 0.48:
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


def _is_web_item(item: dict[str, Any]) -> bool:
    source = str(item.get("source") or "").strip().lower()
    topic = str(item.get("topic") or "").strip().lower()
    text = str(item.get("text") or item.get("content") or "").strip().lower()
    return bool(
        topic.startswith("web:")
        or source in {"web", "search", "internet"}
        or text.startswith("[web]")
        or text.startswith("[web_search]")
    )


def _extract_query_entities(query: str, *, explicit: list[str] | None = None) -> list[str]:
    out: list[str] = []
    if explicit:
        for row in list(explicit or []):
            token = str(row or "").strip().lower()
            if token and token not in out:
                out.append(token)
    for token in _WORD_RE.findall(str(query or "").lower()):
        item = str(token or "").strip().lower()
        if len(item) < 3 or item in _STOPWORDS:
            continue
        if item not in out:
            out.append(item)
    return out[:16]


def _extract_item_entities(item: dict[str, Any]) -> set[str]:
    found: set[str] = set()
    metadata = _as_dict(item.get("metadata"))
    entities = metadata.get("entities")
    if isinstance(entities, dict):
        for key, value in entities.items():
            token = str(key or "").strip().lower()
            if token and len(token) >= 3:
                found.add(token)
            if isinstance(value, list):
                for row in value:
                    t = str(row or "").strip().lower()
                    if t and len(t) >= 3:
                        found.add(t)
            elif value is not None:
                t = str(value).strip().lower()
                if t and len(t) >= 3:
                    found.add(t)
    fact = _as_dict(metadata.get("fact"))
    for key in ("subject", "predicate", "value"):
        token = str(fact.get(key) or "").strip().lower()
        if token and len(token) >= 3:
            found.add(token)
    text = str(item.get("text") or item.get("content") or "").strip().lower()
    for token in _WORD_RE.findall(text):
        t = str(token or "").strip().lower()
        if len(t) >= 3 and t not in _STOPWORDS:
            found.add(t)
        if len(found) >= 80:
            break
    return found


def _entity_overlap_score(*, query_entities: set[str], item_entities: set[str]) -> float:
    if not query_entities or not item_entities:
        return 0.0
    inter = query_entities.intersection(item_entities)
    if not inter:
        return 0.0
    return max(0.0, min(1.0, float(len(inter)) / float(max(1, len(query_entities)))))


def _query_specificity(query: str) -> float:
    tokens = [str(x or "").strip().lower() for x in _WORD_RE.findall(str(query or ""))]
    tokens = [x for x in tokens if x and x not in _STOPWORDS]
    if not tokens:
        return 0.0
    unique = len(set(tokens))
    length_score = min(1.0, float(len(tokens)) / 10.0)
    uniqueness_score = float(unique) / float(max(1, len(tokens)))
    mixed_score = 1.0 if any(any(ch.isdigit() for ch in t) for t in tokens) else 0.0
    out = (0.55 * length_score) + (0.35 * uniqueness_score) + (0.10 * mixed_score)
    return max(0.0, min(1.0, out))


def _has_fresh_web_cache(items: list[Any], *, max_age_hours: int = 24) -> bool:
    if not items:
        return False
    now = dt.datetime.now(dt.timezone.utc)
    for row in list(items or []):
        item = _as_dict(row) if not isinstance(row, dict) else dict(row)
        if not _is_web_item(item):
            continue
        stamp = str(item.get("fetched_at") or item.get("published_date") or "").strip()
        if not stamp:
            continue
        parsed = _parse_dt(stamp)
        if parsed is None:
            continue
        age_h = max(0.0, (now - parsed).total_seconds() / 3600.0)
        if age_h <= float(max_age_hours):
            return True
    return False


def _parse_dt(value: str) -> dt.datetime | None:
    src = str(value or "").strip()
    if not src:
        return None
    try:
        parsed = dt.datetime.fromisoformat(src.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=dt.timezone.utc)
        return parsed.astimezone(dt.timezone.utc)
    except Exception:
        return None
