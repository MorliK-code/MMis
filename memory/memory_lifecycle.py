"""Lifecycle decisions for Memory V2 records."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from memory.memory_models import (
    ConflictDecision,
    LifecycleDecision,
    MemoryLevel,
    MemoryRecord,
    MemoryStatus,
    MemoryType,
)


@dataclass
class MemoryLifecycleManager:
    stale_after_days: int = 30
    archive_after_days: int = 90
    promote_message_importance_threshold: float = 0.55
    promote_message_confidence_threshold: float = 0.50
    promote_project_signal_boost: float = 0.12
    promote_fact_signal_boost: float = 0.16
    promote_decision_signal_boost: float = 0.12
    promote_smalltalk_penalty: float = 0.20
    parallel_margin: float = 0.08

    def decide(self, record: MemoryRecord, *, now_ts: float | None = None) -> LifecycleDecision:
        now = float(now_ts or time.time())
        if record.expires_at is not None and float(record.expires_at) <= now:
            return LifecycleDecision(
                mark_status=MemoryStatus.DELETED,
                archive=False,
                reason="ttl_expired",
                route="expired",
                next_version=int(record.version) + 1,
                chain_parent_id=record.parent_id or record.id,
            )

        age_sec = max(0.0, now - float(record.updated_at))
        stale_sec = max(1, int(self.stale_after_days)) * 86400.0
        archive_sec = max(1, int(self.archive_after_days)) * 86400.0
        if age_sec >= archive_sec and record.status not in {MemoryStatus.DELETED, MemoryStatus.ARCHIVED}:
            return LifecycleDecision(
                mark_status=MemoryStatus.ARCHIVED,
                archive=True,
                reason="age_archive_threshold",
                route="archive",
                next_version=int(record.version) + 1,
                chain_parent_id=record.parent_id or record.id,
            )
        if age_sec >= stale_sec and record.status == MemoryStatus.ACTIVE:
            return LifecycleDecision(
                mark_status=MemoryStatus.STALE,
                reason="age_stale_threshold",
                route="stale",
                next_version=int(record.version) + 1,
                chain_parent_id=record.parent_id or record.id,
            )

        if record.memory_type in {MemoryType.MESSAGE, MemoryType.TOOL_RESULT, MemoryType.TASK_STATE}:
            return self._decide_message_like(record)

        if record.memory_type == MemoryType.FACT:
            return LifecycleDecision(promote_to=MemoryLevel.L3_SEMANTIC, reason="fact_to_semantic", route="semantic")

        if record.memory_type in {MemoryType.DOCUMENT, MemoryType.DOCUMENT_CHUNK}:
            return LifecycleDecision(promote_to=MemoryLevel.L4_DOCUMENT, reason="document_level", route="document")

        return LifecycleDecision(reason="no_change", route="working")

    def _decide_message_like(self, record: MemoryRecord) -> LifecycleDecision:
        meta = dict(record.metadata or {})
        importance = self._clamp01(record.importance)
        confidence = self._clamp01(record.confidence)

        project_signal = self._signal(
            meta.get("promotion_project_signal"),
            fallback=self._contains_any(record.text, ("project", "release", "repo", "roadmap", "milestone")),
        )
        task_signal = self._signal(
            meta.get("promotion_task_signal"),
            fallback=self._contains_any(record.text, ("todo", "task", "need to", "implement", "fix", "ship")),
        )
        decision_signal = self._signal(
            meta.get("promotion_decision_signal"),
            fallback=self._contains_any(record.text, ("decision", "decided", "let's use", "lets use", "go with")),
        )
        preference_signal = self._signal(
            meta.get("promotion_preference_signal"),
            fallback=self._contains_any(record.text, ("i prefer", "prefer ", "my preference", "i like")),
        )
        issue_signal = self._signal(
            meta.get("promotion_issue_signal"),
            fallback=self._contains_any(record.text, ("error", "failed", "exception", "traceback", "bug", "problem")),
        )
        technical_signal = self._signal(
            meta.get("promotion_technical_signal"),
            fallback=self._contains_any(record.text, ("python", "sql", "api", "backend", "frontend", "docker")),
        )
        repeated_signal = self._signal(meta.get("promotion_repeated_topic_signal"), fallback=False)
        smalltalk_signal = self._signal(
            meta.get("promotion_smalltalk_signal"),
            fallback=self._contains_any(
                record.text,
                ("hi", "hello", "how are you", "good morning", "good evening", "thanks", "thank you", "lol"),
            ),
        )

        facts_count = self._to_int(meta.get("extracted_facts_count"), 0)
        fact_signal = self._signal(meta.get("promotion_fact_signal"), fallback=(facts_count > 0))

        if record.memory_type in {MemoryType.TASK_STATE, MemoryType.TOOL_RESULT}:
            task_signal = max(task_signal, 0.85)
            technical_signal = max(technical_signal, 0.75)

        base_score = (0.62 * importance) + (0.22 * confidence)
        composite = (
            base_score
            + (self._clamp01(self.promote_project_signal_boost) * project_signal)
            + (0.09 * task_signal)
            + (self._clamp01(self.promote_decision_signal_boost) * decision_signal)
            + (0.08 * preference_signal)
            + (0.08 * issue_signal)
            + (0.06 * technical_signal)
            + (0.05 * repeated_signal)
            + (self._clamp01(self.promote_fact_signal_boost) * fact_signal)
            - (self._clamp01(self.promote_smalltalk_penalty) * smalltalk_signal)
        )
        composite = self._clamp01(composite)

        importance_threshold = self._clamp01(self.promote_message_importance_threshold)
        confidence_threshold = self._clamp01(self.promote_message_confidence_threshold)
        composite_threshold = self._clamp01(max(0.38, float(importance_threshold) - 0.08))
        confidence_soft_gate = self._clamp01(max(0.30, float(confidence_threshold) - 0.10))
        strong_signal = max(
            project_signal,
            task_signal,
            decision_signal,
            preference_signal,
            issue_signal,
            technical_signal,
            fact_signal,
            repeated_signal,
        )
        is_smalltalk_only = smalltalk_signal >= 0.75 and strong_signal < 0.45 and facts_count <= 0

        debug_payload = {
            "importance": float(importance),
            "confidence": float(confidence),
            "importance_threshold": float(importance_threshold),
            "confidence_threshold": float(confidence_threshold),
            "composite_score": float(composite),
            "composite_threshold": float(composite_threshold),
            "confidence_soft_gate": float(confidence_soft_gate),
            "signals": {
                "project": float(project_signal),
                "task": float(task_signal),
                "decision": float(decision_signal),
                "preference": float(preference_signal),
                "issue": float(issue_signal),
                "technical": float(technical_signal),
                "repeated": float(repeated_signal),
                "fact": float(fact_signal),
                "smalltalk": float(smalltalk_signal),
                "facts_count": int(facts_count),
            },
        }

        if is_smalltalk_only:
            return LifecycleDecision(
                reason="smalltalk_keep_working",
                route="working",
                decision_debug=debug_payload,
            )

        hard_threshold = importance >= importance_threshold and confidence >= confidence_threshold
        composite_threshold_pass = (
            composite >= composite_threshold
            and confidence >= confidence_soft_gate
            and strong_signal >= 0.25
        )
        if hard_threshold or composite_threshold_pass:
            reason = "message_promoted_hard_threshold" if hard_threshold else "message_promoted_composite"
            return LifecycleDecision(
                promote_to=MemoryLevel.L2_EPISODIC,
                reason=reason,
                route="episodic",
                chain_parent_id=record.id,
                decision_debug=debug_payload,
            )

        return LifecycleDecision(
            reason="message_keep_working",
            route="working",
            decision_debug=debug_payload,
        )

    @staticmethod
    def _clamp01(value: float) -> float:
        return max(0.0, min(1.0, float(value)))

    @staticmethod
    def _to_int(value: Any, default: int) -> int:
        try:
            return int(value)
        except Exception:
            return int(default)

    def _signal(self, value: Any, *, fallback: bool) -> float:
        if value is None:
            return 1.0 if fallback else 0.0
        try:
            parsed = float(value)
        except Exception:
            return 1.0 if fallback else 0.0
        return self._clamp01(parsed)

    @staticmethod
    def _contains_any(text: str, markers: tuple[str, ...]) -> bool:
        src = str(text or "").lower()
        if not src:
            return False
        return any(token in src for token in markers)

    def apply_decay(self, records: list[MemoryRecord], *, now_ts: float | None = None) -> list[MemoryRecord]:
        now = float(now_ts or time.time())
        stale_sec = max(1, int(self.stale_after_days)) * 86400.0
        archive_sec = max(1, int(self.archive_after_days)) * 86400.0

        out: list[MemoryRecord] = []
        for record in list(records or []):
            age = max(0.0, now - float(record.updated_at))
            status = record.status
            reason = ""
            if age >= archive_sec:
                status = MemoryStatus.ARCHIVED
                reason = "age_archive_threshold"
            elif age >= stale_sec and status == MemoryStatus.ACTIVE:
                status = MemoryStatus.STALE
                reason = "age_stale_threshold"
            next_version = int(record.version)
            meta = dict(record.metadata or {})
            if status != record.status:
                next_version += 1
                meta = {
                    **meta,
                    "previous_status": str(record.status.value),
                    "lifecycle_reason": reason or "decay",
                    "version_chain_parent": record.parent_id or record.id,
                }
            out.append(
                MemoryRecord(
                    id=record.id,
                    text=record.text,
                    memory_type=record.memory_type,
                    level=record.level,
                    scope=record.scope,
                    namespace=record.namespace,
                    metadata=meta,
                    embedding=list(record.embedding or []) if isinstance(record.embedding, list) else None,
                    importance=record.importance,
                    confidence=record.confidence,
                    created_at=record.created_at,
                    updated_at=(now if status != record.status else record.updated_at),
                    expires_at=record.expires_at,
                    status=status,
                    version=next_version,
                    parent_id=(record.parent_id or record.id),
                    chunk_index=record.chunk_index,
                    source_event_id=record.source_event_id,
                    embedding_model=record.embedding_model,
                    embedding_fingerprint=record.embedding_fingerprint,
                    embedding_version=record.embedding_version,
                )
            )
        return out

    def resolve_conflict(self, *, old: MemoryRecord, new: MemoryRecord) -> ConflictDecision:
        now = max(float(time.time()), float(old.updated_at), float(new.updated_at))
        old_recency = 1.0 / (1.0 + max(0.0, now - float(old.updated_at)) / 86400.0)
        new_recency = 1.0 / (1.0 + max(0.0, now - float(new.updated_at)) / 86400.0)
        old_score = float(old.confidence) * 0.45 + float(old.importance) * 0.35 + float(old_recency) * 0.20
        new_score = float(new.confidence) * 0.45 + float(new.importance) * 0.35 + float(new_recency) * 0.20
        delta = float(new_score - old_score)

        old_key = str(dict(old.metadata or {}).get("canonical_key") or "").strip().lower()
        new_key = str(dict(new.metadata or {}).get("canonical_key") or "").strip().lower()
        if old_key and new_key and old_key != new_key:
            return ConflictDecision(
                keep_record_id=new.id,
                superseded_record_id=None,
                reason="different_canonical_keys_parallel_facts",
                status=MemoryStatus.ACTIVE,
                action="parallel",
                parallel_with_record_id=old.id,
                score_delta=delta,
            )

        if abs(delta) <= float(self.parallel_margin):
            return ConflictDecision(
                keep_record_id=new.id,
                superseded_record_id=None,
                reason="scores_close_parallel_facts",
                status=MemoryStatus.ACTIVE,
                action="parallel",
                parallel_with_record_id=old.id,
                score_delta=delta,
            )

        if delta >= 0.0:
            loser = old
            winner_id = new.id
        else:
            loser = new
            winner_id = old.id

        loser_age_days = max(0.0, now - float(loser.updated_at)) / 86400.0
        if loser_age_days >= float(self.archive_after_days) and float(loser.confidence) < 0.45:
            return ConflictDecision(
                keep_record_id=winner_id,
                superseded_record_id=None,
                reason="loser_archived_old_low_confidence",
                status=MemoryStatus.ARCHIVED,
                action="archive",
                archive_record_id=loser.id,
                score_delta=delta,
            )

        return ConflictDecision(
            keep_record_id=winner_id,
            superseded_record_id=loser.id,
            reason="winner_by_recency_confidence_importance",
            status=MemoryStatus.SUPERSEDED,
            action="supersede",
            score_delta=delta,
        )
