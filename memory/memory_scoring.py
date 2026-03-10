"""Scoring utilities for Memory V2 retrieval candidates."""

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


@dataclass(frozen=True)
class SalienceWeights:
    novelty: float = 0.22
    permanence: float = 0.20
    repetition: float = 0.14
    project_relevance: float = 0.16
    task_relevance: float = 0.16
    explicit_save_signal: float = 0.12


@dataclass(frozen=True)
class SalienceBreakdown:
    novelty: float = 0.0
    permanence: float = 0.0
    repetition: float = 0.0
    project_relevance: float = 0.0
    task_relevance: float = 0.0
    explicit_save_signal: float = 0.0
    final_score: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return {
            "novelty": float(self.novelty),
            "permanence": float(self.permanence),
            "repetition": float(self.repetition),
            "project_relevance": float(self.project_relevance),
            "task_relevance": float(self.task_relevance),
            "explicit_save_signal": float(self.explicit_save_signal),
            "final_score": float(self.final_score),
        }


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _tokens(text: str) -> set[str]:
    return {x.lower() for x in _TOKEN_RE.findall(str(text or "")) if x}


def novelty_score(text: str, recent_texts: list[str]) -> float:
    src = _tokens(text)
    if not src:
        return 0.0
    rows = [x for x in list(recent_texts or []) if str(x or "").strip()]
    if not rows:
        return 1.0
    best_overlap = 0.0
    for row in rows:
        tgt = _tokens(row)
        if not tgt:
            continue
        inter = len(src.intersection(tgt))
        union = max(1, len(src.union(tgt)))
        best_overlap = max(best_overlap, float(inter) / float(union))
    return _clamp01(1.0 - best_overlap)


def repetition_score(text: str, recent_texts: list[str]) -> float:
    src = str(text or "").strip().lower()
    if not src:
        return 0.0
    rows = [str(x or "").strip().lower() for x in list(recent_texts or []) if str(x or "").strip()]
    if not rows:
        return 0.0
    exact = sum(1 for row in rows if row == src)
    soft = sum(1 for row in rows if src in row or row in src)
    ratio = (float(exact) * 1.0 + float(soft) * 0.35) / float(max(1, len(rows)))
    return _clamp01(ratio)


def permanence_score(text: str, metadata: dict[str, object] | None = None) -> float:
    src = str(text or "").lower()
    meta = dict(metadata or {})
    if bool(meta.get("temporary")) or bool(meta.get("ephemeral")):
        return 0.0
    if str(meta.get("ttl_sec") or "").strip():
        return 0.12
    stable_markers = (
        "always",
        "default",
        "policy",
        "decision",
        "architecture",
        "strategy",
        "предпочитаю",
        "всегда",
        "по умолчанию",
    )
    if any(token in src for token in stable_markers):
        return 0.86
    return 0.46


def project_relevance_score(text: str, metadata: dict[str, object] | None = None) -> float:
    src = str(text or "").lower()
    meta = dict(metadata or {})
    if str(meta.get("project_id") or "").strip() or str(meta.get("project") or "").strip():
        return 1.0
    markers = ("project", "release", "roadmap", "repo", "архитект", "релиз", "проект")
    return 0.84 if any(token in src for token in markers) else 0.24


def task_relevance_score(text: str, metadata: dict[str, object] | None = None) -> float:
    src = str(text or "").lower()
    meta = dict(metadata or {})
    if str(meta.get("task_id") or "").strip() or bool(meta.get("is_task")):
        return 1.0
    markers = ("todo", "task", "issue", "bug", "need to", "fix", "нужно", "задача", "исправ")
    return 0.82 if any(token in src for token in markers) else 0.22


def explicit_save_signal_score(text: str, metadata: dict[str, object] | None = None) -> float:
    src = str(text or "").lower()
    meta = dict(metadata or {})
    if bool(meta.get("pin")) or bool(meta.get("remember")) or bool(meta.get("explicit_save")):
        return 1.0
    markers = ("remember", "save this", "important", "запомни", "важно", "сохрани")
    return 0.92 if any(token in src for token in markers) else 0.0


def build_salience_breakdown(
    *,
    text: str,
    metadata: dict[str, object] | None = None,
    recent_texts: list[str] | None = None,
    weights: SalienceWeights | None = None,
) -> SalienceBreakdown:
    rows = list(recent_texts or [])
    w = weights or SalienceWeights()
    nov = novelty_score(text, rows)
    perm = permanence_score(text, metadata)
    rep = repetition_score(text, rows)
    proj = project_relevance_score(text, metadata)
    task = task_relevance_score(text, metadata)
    save = explicit_save_signal_score(text, metadata)
    final = (
        (w.novelty * nov)
        + (w.permanence * perm)
        + (w.repetition * rep)
        + (w.project_relevance * proj)
        + (w.task_relevance * task)
        + (w.explicit_save_signal * save)
    )
    return SalienceBreakdown(
        novelty=nov,
        permanence=perm,
        repetition=rep,
        project_relevance=proj,
        task_relevance=task,
        explicit_save_signal=save,
        final_score=_clamp01(final),
    )


def build_salience_score(
    *,
    text: str,
    metadata: dict[str, object] | None = None,
    recent_texts: list[str] | None = None,
    weights: SalienceWeights | None = None,
) -> float:
    return float(
        build_salience_breakdown(
            text=text,
            metadata=metadata,
            recent_texts=recent_texts,
            weights=weights,
        ).final_score
    )


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
