"""Lifecycle decisions for Memory V2 records."""

from __future__ import annotations

import time
from dataclasses import dataclass

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
    promote_message_importance_threshold: float = 0.72
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
            if float(record.importance) >= float(self.promote_message_importance_threshold) and float(record.confidence) >= 0.50:
                return LifecycleDecision(
                    promote_to=MemoryLevel.L2_EPISODIC,
                    reason="high_importance_recent",
                    route="episodic",
                    chain_parent_id=record.id,
                )
            return LifecycleDecision(reason="keep_working", route="working")

        if record.memory_type == MemoryType.FACT:
            return LifecycleDecision(promote_to=MemoryLevel.L3_SEMANTIC, reason="fact_to_semantic", route="semantic")

        if record.memory_type in {MemoryType.DOCUMENT, MemoryType.DOCUMENT_CHUNK}:
            return LifecycleDecision(promote_to=MemoryLevel.L4_DOCUMENT, reason="document_level", route="document")

        return LifecycleDecision(reason="no_change", route="working")

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
