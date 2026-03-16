"""Scoring utilities for Memory V2 retrieval candidates."""

from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass

from memory.memory_models import MemoryRecord, MemoryScope, RetrievalQuery, ScoreBreakdown
from memory.retrieval_projection import build_memory_views, record_search_text


_TOKEN_RE = re.compile(r"[A-Za-z\u0400-\u04ff0-9_]+")


@dataclass(frozen=True)
class ScoreWeights:
    semantic_similarity: float = 0.34
    lexical_score: float = 0.25
    recency_score: float = 0.10
    importance_score: float = 0.09
    confidence_score: float = 0.08
    entity_overlap_score: float = 0.07
    numeric_overlap_score: float = 0.07
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


@dataclass(frozen=True)
class MessageSignalBreakdown:
    project_relevance: float = 0.0
    task_intent: float = 0.0
    architecture_signal: float = 0.0
    preference_signal: float = 0.0
    stable_fact_signal: float = 0.0
    decision_signal: float = 0.0
    issue_signal: float = 0.0
    technical_relevance: float = 0.0
    repeated_theme: float = 0.0
    smalltalk: float = 0.0
    meaningful_signal: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return {
            "project_relevance": float(self.project_relevance),
            "task_intent": float(self.task_intent),
            "architecture_signal": float(self.architecture_signal),
            "preference_signal": float(self.preference_signal),
            "stable_fact_signal": float(self.stable_fact_signal),
            "decision_signal": float(self.decision_signal),
            "issue_signal": float(self.issue_signal),
            "technical_relevance": float(self.technical_relevance),
            "repeated_theme": float(self.repeated_theme),
            "smalltalk": float(self.smalltalk),
            "meaningful_signal": float(self.meaningful_signal),
        }


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _tokens(text: str) -> set[str]:
    return {x.lower() for x in _TOKEN_RE.findall(str(text or "")) if x}


def _contains_any(text: str, markers: tuple[str, ...]) -> bool:
    src = str(text or "").lower()
    if not src:
        return False
    return any(token in src for token in markers)


def _marker_score(text: str, markers: tuple[str, ...], *, max_hits: int = 3) -> float:
    src = str(text or "").lower()
    if not src:
        return 0.0
    hits = sum(1 for token in markers if token and token in src)
    if hits <= 0:
        return 0.0
    capped = min(max(1, int(max_hits)), hits)
    return _clamp01(0.35 + (0.20 * capped))


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
        "i use ",
        "my project",
        "we decided",
        "i prefer",
        "i like",
        "\u043f\u0440\u0435\u0434\u043f\u043e\u0447\u0438\u0442\u0430\u044e",
        "\u0432\u0441\u0435\u0433\u0434\u0430",
        "\u043f\u043e \u0443\u043c\u043e\u043b\u0447\u0430\u043d\u0438\u044e",
        "\u044f \u0438\u0441\u043f\u043e\u043b\u044c\u0437\u0443\u044e",
        "\u043c\u043e\u0439 \u043f\u0440\u043e\u0435\u043a\u0442",
        "\u043c\u044b \u0440\u0435\u0448\u0438\u043b\u0438",
        "\u043c\u043d\u0435 \u043d\u0440\u0430\u0432\u0438\u0442\u0441\u044f",
    )
    stable = _marker_score(src, stable_markers)
    if stable > 0.0:
        return _clamp01(0.52 + (0.34 * stable))
    return 0.46


def project_relevance_score(text: str, metadata: dict[str, object] | None = None) -> float:
    src = str(text or "").lower()
    meta = dict(metadata or {})
    if str(meta.get("project_id") or "").strip() or str(meta.get("project") or "").strip():
        return 1.0
    markers = (
        "project",
        "release",
        "roadmap",
        "repo",
        "repository",
        "milestone",
        "module",
        "architecture",
        "memory v2",
        "mmis",
        "\u043f\u0440\u043e\u0435\u043a\u0442",
        "\u0440\u0435\u043b\u0438\u0437",
        "\u0440\u0435\u043f\u043e",
        "\u0430\u0440\u0445\u0438\u0442\u0435\u043a\u0442",
        "\u043c\u043e\u0434\u0443\u043b\u044c",
    )
    score = _marker_score(src, markers)
    return _clamp01(0.24 + (0.64 * score)) if score > 0.0 else 0.24


def task_relevance_score(text: str, metadata: dict[str, object] | None = None) -> float:
    src = str(text or "").lower()
    meta = dict(metadata or {})
    if str(meta.get("task_id") or "").strip() or bool(meta.get("is_task")):
        return 1.0
    markers = (
        "todo",
        "task",
        "issue",
        "bug",
        "need to",
        "fix",
        "implement",
        "update",
        "ship",
        "must ",
        "should ",
        "i want to",
        "\u043d\u0430\u0434\u043e",
        "\u043d\u0443\u0436\u043d\u043e",
        "\u0437\u0430\u0434\u0430\u0447\u0430",
        "\u0441\u0434\u0435\u043b\u0430\u0442\u044c",
        "\u043f\u043e\u0447\u0438\u043d",
        "\u0438\u0441\u043f\u0440\u0430\u0432",
        "\u0434\u043e\u0440\u0430\u0431\u043e\u0442",
        "\u043f\u043e\u0434\u0434\u0435\u0440\u0436",
        "\u043e\u0431\u043d\u043e\u0432",
        "\u044f \u0445\u043e\u0447\u0443",
    )
    score = _marker_score(src, markers)
    return _clamp01(0.22 + (0.66 * score)) if score > 0.0 else 0.22


def explicit_save_signal_score(text: str, metadata: dict[str, object] | None = None) -> float:
    src = str(text or "").lower()
    meta = dict(metadata or {})
    if bool(meta.get("pin")) or bool(meta.get("remember")) or bool(meta.get("explicit_save")):
        return 1.0
    markers = (
        "remember",
        "save this",
        "important",
        "keep this",
        "\u0437\u0430\u043f\u043e\u043c\u043d\u0438",
        "\u0432\u0430\u0436\u043d\u043e",
        "\u0441\u043e\u0445\u0440\u0430\u043d\u0438",
    )
    score = _marker_score(src, markers, max_hits=2)
    return _clamp01(0.48 + (0.44 * score)) if score > 0.0 else 0.0


def build_message_signal_breakdown(
    *,
    text: str,
    metadata: dict[str, object] | None = None,
    recent_texts: list[str] | None = None,
) -> MessageSignalBreakdown:
    src = str(text or "").strip().lower()
    meta = dict(metadata or {})
    recent = [str(x or "").strip().lower() for x in list(recent_texts or []) if str(x or "").strip()]

    project_markers = (
        "project",
        "release",
        "roadmap",
        "milestone",
        "repo",
        "repository",
        "module",
        "for the project",
        "in the project",
        "memory v2",
        "mmis",
        "\u043f\u0440\u043e\u0435\u043a\u0442",
        "\u0432 \u043f\u0440\u043e\u0435\u043a\u0442\u0435",
        "\u0434\u043b\u044f \u043f\u0440\u043e\u0435\u043a\u0442\u0430",
        "\u043c\u043e\u0439 \u043f\u0440\u043e\u0435\u043a\u0442",
        "\u0440\u0435\u043b\u0438\u0437",
        "\u0440\u0435\u043f\u043e",
        "\u043c\u043e\u0434\u0443\u043b\u044c",
    )
    task_markers = (
        "todo",
        "task",
        "need to",
        "please",
        "implement",
        "fix",
        "update",
        "ship",
        "must ",
        "should ",
        "i want to",
        "need ",
        "\u043d\u0430\u0434\u043e",
        "\u043d\u0443\u0436\u043d\u043e",
        "\u044f \u0445\u043e\u0447\u0443",
        "\u0441\u0434\u0435\u043b\u0430\u0442\u044c",
        "\u043f\u043e\u0447\u0438\u043d",
        "\u0438\u0441\u043f\u0440\u0430\u0432",
        "\u0434\u043e\u0440\u0430\u0431\u043e\u0442",
        "\u043e\u0431\u043d\u043e\u0432",
        "\u043f\u043e\u0434\u0434\u0435\u0440\u0436",
        "\u0437\u0430\u0434\u0430\u0447\u0430",
    )
    architecture_markers = (
        "architecture",
        "design",
        "refactor",
        "pipeline",
        "service",
        "api",
        "schema",
        "backend",
        "frontend",
        "query planner",
        "lifecycle",
        "\u0430\u0440\u0445\u0438\u0442\u0435\u043a\u0442",
        "\u0434\u0438\u0437\u0430\u0439\u043d",
        "\u0440\u0435\u0444\u0430\u043a\u0442",
        "\u043f\u0430\u0439\u043f\u043b\u0430\u0439\u043d",
        "\u0441\u0435\u0440\u0432\u0438\u0441",
        "\u0441\u0445\u0435\u043c",
        "\u0431\u044d\u043a\u0435\u043d\u0434",
        "\u0444\u0440\u043e\u043d\u0442\u0435\u043d\u0434",
        "\u043f\u043b\u0430\u043d\u043d\u0435\u0440",
        "\u043b\u043e\u0433\u0438\u043a",
    )
    preference_markers = (
        "i prefer",
        "prefer ",
        "i like",
        "i love",
        "my preference",
        "better for me",
        "\u043f\u0440\u0435\u0434\u043f\u043e\u0447\u0438\u0442\u0430\u044e",
        "\u043b\u044e\u0431\u043b\u044e",
        "\u043c\u043d\u0435 \u043d\u0440\u0430\u0432\u0438\u0442\u0441\u044f",
        "\u043c\u043d\u0435 \u0443\u0434\u043e\u0431\u043d\u0435\u0435",
        "\u043c\u043d\u0435 \u0432\u0430\u0436\u043d\u0435\u0435",
    )
    stable_fact_markers = (
        "my name is",
        "i am ",
        "i use ",
        "we use ",
        "i have ",
        "my project",
        "in the project",
        "for now",
        "always",
        "default",
        "\u043c\u0435\u043d\u044f \u0437\u043e\u0432\u0443\u0442",
        "\u044f \u0438\u0441\u043f\u043e\u043b\u044c\u0437\u0443\u044e",
        "\u0443 \u043c\u0435\u043d\u044f",
        "\u043c\u043e\u0439 \u043f\u0440\u043e\u0435\u043a\u0442",
        "\u0432 \u043f\u0440\u043e\u0435\u043a\u0442\u0435",
        "\u0440\u0430\u0431\u043e\u0442\u0430\u044e \u043d\u0430",
        "\u043c\u043d\u0435 \u043d\u0440\u0430\u0432\u0438\u0442\u0441\u044f",
        "\u043f\u043e \u0443\u043c\u043e\u043b\u0447\u0430\u043d\u0438\u044e",
    )
    decision_markers = (
        "we decided",
        "decision",
        "let's use",
        "lets use",
        "choose",
        "go with",
        "we will use",
        "\u043c\u044b \u0440\u0435\u0448\u0438\u043b\u0438",
        "\u0440\u0435\u0448\u0438\u043b\u0438",
        "\u0440\u0435\u0448\u0435\u043d\u043e",
        "\u043e\u0441\u0442\u0430\u0432\u043b\u044f\u0435\u043c",
        "\u0432\u044b\u0431\u0440\u0430\u043b\u0438",
        "\u0431\u0443\u0434\u0435\u043c",
        "\u043d\u0435 \u0431\u0443\u0434\u0435\u043c",
    )
    issue_markers = (
        "error",
        "failed",
        "exception",
        "traceback",
        "bug",
        "problem",
        "incident",
        "broken",
        "does not work",
        "doesn't work",
        "\u043e\u0448\u0438\u0431\u043a",
        "\u043f\u0440\u043e\u0431\u043b\u0435\u043c",
        "\u043d\u0435 \u0440\u0430\u0431\u043e\u0442",
        "\u0441\u043b\u043e\u043c\u0430",
        "\u043b\u043e\u043c\u0430",
        "\u043f\u0430\u0434\u0430\u0435\u0442",
        "\u0441\u044b\u043f\u0435",
        "\u043d\u0435 \u043b\u043e\u0432\u0438\u0442",
        "\u0437\u0430\u0441\u0442\u0440\u0435\u0432",
    )
    technical_markers = (
        "python",
        "typescript",
        "javascript",
        "sql",
        "docker",
        "kubernetes",
        "chromadb",
        "api",
        "backend",
        "frontend",
        "stack trace",
        "powershell",
        "windows",
        "linux",
        "query planner",
        "memory v2",
    )
    smalltalk_markers = (
        "hi",
        "hello",
        "how are you",
        "good morning",
        "good evening",
        "thanks",
        "thank you",
        "nice to meet",
        "lol",
        "\u043f\u0440\u0438\u0432\u0435\u0442",
        "\u043a\u0430\u043a \u0434\u0435\u043b\u0430",
        "\u0434\u043e\u0431\u0440\u043e\u0435 \u0443\u0442\u0440\u043e",
        "\u0434\u043e\u0431\u0440\u044b\u0439 \u0432\u0435\u0447\u0435\u0440",
        "\u0441\u043f\u0430\u0441\u0438\u0431\u043e",
        "\u0445\u0430\u0445\u0430",
    )

    project = 1.0 if str(meta.get("project_id") or "").strip() else _marker_score(src, project_markers)
    task = 1.0 if (bool(meta.get("is_task")) or str(meta.get("task_id") or "").strip()) else _marker_score(src, task_markers)
    architecture = _marker_score(src, architecture_markers)
    preference = _marker_score(src, preference_markers)
    stable_fact = _marker_score(src, stable_fact_markers)
    decision = _marker_score(src, decision_markers)
    issue = _marker_score(src, issue_markers)
    technical = _marker_score(src, technical_markers)
    smalltalk = _marker_score(src, smalltalk_markers, max_hits=2)

    repeated = 0.0
    src_terms = [x for x in list(_tokens(src)) if len(x) >= 4][:6]
    if src_terms and recent:
        hits = 0
        for row in recent[-8:]:
            if any(term in row for term in src_terms):
                hits += 1
        repeated = _clamp01(float(hits) / 3.0)

    meaningful_context = max(project, task, architecture, preference, stable_fact, decision, issue, technical)
    if smalltalk > 0.0 and meaningful_context >= 0.55:
        smalltalk = _clamp01(smalltalk * 0.35)

    meaningful = _clamp01(
        (0.22 * project)
        + (0.20 * task)
        + (0.16 * architecture)
        + (0.12 * preference)
        + (0.14 * stable_fact)
        + (0.18 * decision)
        + (0.18 * issue)
        + (0.10 * technical)
        + (0.10 * repeated)
        - (0.26 * smalltalk)
    )

    return MessageSignalBreakdown(
        project_relevance=project,
        task_intent=task,
        architecture_signal=architecture,
        preference_signal=preference,
        stable_fact_signal=stable_fact,
        decision_signal=decision,
        issue_signal=issue,
        technical_relevance=technical,
        repeated_theme=repeated,
        smalltalk=smalltalk,
        meaningful_signal=meaningful,
    )


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


def metadata_entity_overlap(query_keys: list[str], record: MemoryRecord) -> float:
    q = {str(x).strip().lower() for x in list(query_keys or []) if str(x).strip()}
    if not q:
        return 0.0
    views = build_memory_views(record.text, metadata=record.metadata)
    record_keys = {str(x).strip().lower() for x in list(views.get("entity_keys") or []) if str(x).strip()}
    if not record_keys:
        return 0.0
    inter = len(q.intersection(record_keys))
    return _clamp01(inter / max(1, len(q)))


def metadata_numeric_overlap(query_keys: list[str], record: MemoryRecord) -> float:
    q = {str(x).strip().lower() for x in list(query_keys or []) if str(x).strip()}
    if not q:
        return 0.0
    views = build_memory_views(record.text, metadata=record.metadata)
    record_keys = {str(x).strip().lower() for x in list(views.get("numeric_keys") or []) if str(x).strip()}
    if not record_keys:
        return 0.0
    inter = len(q.intersection(record_keys))
    return _clamp01(inter / max(1, len(q)))


def recency_score(*, updated_at: float, now_ts: float, stale_after_days: int = 30) -> float:
    age_sec = max(0.0, float(now_ts) - float(updated_at or 0.0))
    half_life = max(3600.0, float(stale_after_days) * 86400.0)
    return _clamp01(math.exp(-math.log(2.0) * age_sec / half_life))


def _text_exact_match_score(query: str, text: str) -> float:
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


def exact_match_boost(query: RetrievalQuery, record: MemoryRecord, *, entity_overlap: float, numeric_overlap: float) -> float:
    query_text = str(query.search_text or query.query_text or "")
    text_score = _text_exact_match_score(query_text, record_search_text(record.text, metadata=record.metadata))
    structured_score = 1.0 if entity_overlap >= 0.99 or numeric_overlap >= 0.99 else 0.0
    return max(text_score, structured_score)


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
    ent = metadata_entity_overlap(query.entity_keys, record)
    num = metadata_numeric_overlap(query.numeric_keys, record)
    exact = exact_match_boost(query, record, entity_overlap=ent, numeric_overlap=num)
    scope = scope_match_score(record.scope, query)

    final = (
        (w.semantic_similarity * sem)
        + (w.lexical_score * lex)
        + (w.recency_score * rec)
        + (w.importance_score * imp)
        + (w.confidence_score * conf)
        + (w.entity_overlap_score * ent)
        + (w.numeric_overlap_score * num)
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
        numeric_overlap_score=num,
        exact_match_boost=exact,
        scope_match_score=scope,
        final_score=_clamp01(final),
    )
