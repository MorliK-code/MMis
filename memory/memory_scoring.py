from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass

from memory.memory_models import MemoryRecord, MemoryScope, RetrievalQuery, ScoreBreakdown


_TOKEN_RE = re.compile(r"[A-Za-z\u0400-\u04ff0-9_]+")


@dataclass(frozen=True)
class ScoreWeights:
    semantic_similarity: float = 0.34
    lexical_score: float = 0.25
    recency_score: float = 0.10
    importance_score: float = 0.09
    confidence_score: float = 0.08
    entity_overlap_score: float = 0.07
    exact_match_boost: float = 0.04
    scope_match_score: float = 0.03


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _tokens(text: str) -> set[str]:
    return {x.lower() for x in _TOKEN_RE.findall(str(text or "")) if x}


def entity_overlap_score(query: str, text: str) -> float:
    q = _tokens(query)
    t = _tokens(text)
    if not q or not t:
        return 0.0
    inter = len(q.intersection(t))
    denom = max(1, len(q))
    return _clamp01(inter / denom)


def recency_score(*, updated_at: float, now_ts: float, stale_after_days: int = 30) -> float:
    age_sec = max(0.0, float(now_ts) - float(updated_at or 0.0))
    half_life = max(3600.0, float(stale_after_days) * 86400.0)
    return _clamp01(math.exp(-math.log(2.0) * age_sec / half_life))


def exact_match_boost(query: str, text: str) -> float:
    q = str(query or "").strip().lower()
    if not q:
        return 0.0
    t = str(text or "").strip().lower()
    if not t:
        return 0.0
    if q == t:
        return 1.0
    if q in t:
        return 0.65
    return 0.0


def scope_match_score(record_scope: MemoryScope, query: RetrievalQuery) -> float:
    if not query.scopes:
        return 0.6
    return 1.0 if record_scope in set(query.scopes) else 0.0


def build_score_breakdown(
    *,
    query: RetrievalQuery,
    record: MemoryRecord,
    semantic_similarity: float,
    lexical_score: float,
    now_ts: float | None = None,
    stale_after_days: int = 30,
    weights: ScoreWeights | None = None,
) -> ScoreBreakdown:
    w = weights or ScoreWeights()
    now = float(now_ts or time.time())

    sem = _clamp01(semantic_similarity)
    lex = _clamp01(lexical_score)
    rec = recency_score(updated_at=record.updated_at, now_ts=now, stale_after_days=stale_after_days)
    imp = _clamp01(record.importance)
    conf = _clamp01(record.confidence)
    ent = entity_overlap_score(query.query_text, record.text)
    exact = exact_match_boost(query.query_text, record.text)
    scope = scope_match_score(record.scope, query)

    final = (
        (w.semantic_similarity * sem)
        + (w.lexical_score * lex)
        + (w.recency_score * rec)
        + (w.importance_score * imp)
        + (w.confidence_score * conf)
        + (w.entity_overlap_score * ent)
        + (w.exact_match_boost * exact)
        + (w.scope_match_score * scope)
    )

    return ScoreBreakdown(
        semantic_similarity=sem,
        lexical_score=lex,
        recency_score=rec,
        importance_score=imp,
        confidence_score=conf,
        entity_overlap_score=ent,
        exact_match_boost=exact,
        scope_match_score=scope,
        final_score=_clamp01(final),
    )
