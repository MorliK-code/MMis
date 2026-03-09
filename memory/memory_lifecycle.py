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

    def decide(self, record: MemoryRecord, *, now_ts: float | None = None) -> LifecycleDecision:
        now = float(now_ts or time.time())
        if record.expires_at is not None and float(record.expires_at) <= now:
            return LifecycleDecision(mark_status=MemoryStatus.DELETED, archive=False, reason="ttl_expired")

        if record.memory_type in {MemoryType.MESSAGE, MemoryType.TOOL_RESULT, MemoryType.TASK_STATE}:
            if float(record.importance) >= 0.72:
                return LifecycleDecision(promote_to=MemoryLevel.L2_EPISODIC, reason="high_importance_recent")
            return LifecycleDecision(reason="keep_working")

        if record.memory_type == MemoryType.FACT:
            return LifecycleDecision(promote_to=MemoryLevel.L3_SEMANTIC, reason="fact_to_semantic")

        if record.memory_type in {MemoryType.DOCUMENT, MemoryType.DOCUMENT_CHUNK}:
            return LifecycleDecision(promote_to=MemoryLevel.L4_DOCUMENT, reason="document_level")

        return LifecycleDecision(reason="no_change")

    def apply_decay(self, records: list[MemoryRecord], *, now_ts: float | None = None) -> list[MemoryRecord]:
        now = float(now_ts or time.time())
        stale_sec = max(1, int(self.stale_after_days)) * 86400.0
        archive_sec = max(1, int(self.archive_after_days)) * 86400.0

        out: list[MemoryRecord] = []
        for record in list(records or []):
            age = max(0.0, now - float(record.updated_at))
            status = record.status
            if age >= archive_sec:
                status = MemoryStatus.ARCHIVED
            elif age >= stale_sec and status == MemoryStatus.ACTIVE:
                status = MemoryStatus.STALE
            out.append(
                MemoryRecord(
                    id=record.id,
                    text=record.text,
                    memory_type=record.memory_type,
                    level=record.level,
                    scope=record.scope,
                    namespace=record.namespace,
                    metadata=dict(record.metadata or {}),
                    embedding=list(record.embedding or []) if isinstance(record.embedding, list) else None,
                    importance=record.importance,
                    confidence=record.confidence,
                    created_at=record.created_at,
                    updated_at=record.updated_at,
                    expires_at=record.expires_at,
                    status=status,
                    version=record.version,
                    parent_id=record.parent_id,
                    chunk_index=record.chunk_index,
                    source_event_id=record.source_event_id,
                    embedding_model=record.embedding_model,
                    embedding_fingerprint=record.embedding_fingerprint,
                    embedding_version=record.embedding_version,
                )
            )
        return out

    def resolve_conflict(self, *, old: MemoryRecord, new: MemoryRecord) -> ConflictDecision:
        old_score = float(old.confidence) * 0.45 + float(old.importance) * 0.35 + float(old.updated_at) * 0.20
        new_score = float(new.confidence) * 0.45 + float(new.importance) * 0.35 + float(new.updated_at) * 0.20
        if new_score >= old_score:
            return ConflictDecision(
                keep_record_id=new.id,
                superseded_record_id=old.id,
                reason="new_record_wins_recency_confidence_importance",
                status=MemoryStatus.SUPERSEDED,
            )
        return ConflictDecision(
            keep_record_id=old.id,
            superseded_record_id=new.id,
            reason="old_record_kept_higher_score",
            status=MemoryStatus.SUPERSEDED,
        )
