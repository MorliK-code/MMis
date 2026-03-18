from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from memory.memory_models import MemoryRecord, MemorySourceKind, MemoryStatus, MemoryType
from memory.memory_policy import MemoryPolicy
from memory.retrieval_projection import build_memory_views
from memory.storage_profile import DEFAULT_STORAGE_PROFILE, normalize_storage_profile, sanitize_storage_metadata


@dataclass(frozen=True)
class BackfillRecordChange:
    record_id: str
    namespace: str
    changed_fields: list[str] = field(default_factory=list)
    assistant_noise_archived: bool = False


@dataclass(frozen=True)
class BackfillSummary:
    dry_run: bool = True
    storage_profile: str = DEFAULT_STORAGE_PROFILE
    namespace: str = "*"
    scanned: int = 0
    rewritten: int = 0
    assistant_noise_archived: int = 0
    namespace_stats: dict[str, dict[str, int]] = field(default_factory=dict)
    changes: list[BackfillRecordChange] = field(default_factory=list)
    reindex_result: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dry_run": bool(self.dry_run),
            "storage_profile": str(self.storage_profile or DEFAULT_STORAGE_PROFILE),
            "namespace": str(self.namespace or "*"),
            "scanned": int(self.scanned),
            "rewritten": int(self.rewritten),
            "assistant_noise_archived": int(self.assistant_noise_archived),
            "namespace_stats": dict(self.namespace_stats or {}),
            "changes": [
                {
                    "record_id": str(item.record_id or ""),
                    "namespace": str(item.namespace or ""),
                    "changed_fields": [str(x) for x in list(item.changed_fields or []) if str(x).strip()],
                    "assistant_noise_archived": bool(item.assistant_noise_archived),
                }
                for item in list(self.changes or [])
            ],
            "reindex_result": dict(self.reindex_result or {}),
        }


def run_final_backfill(
    *,
    manager: Any,
    namespace: str | None = None,
    storage_profile: str = DEFAULT_STORAGE_PROFILE,
    dry_run: bool = True,
    archive_assistant_noise: bool = True,
    reindex: bool = True,
) -> BackfillSummary:
    profile = normalize_storage_profile(storage_profile)
    target_namespace = str(namespace or "").strip()
    rows = list(manager._store.iter_records(namespace=(target_namespace or None)))
    policy = MemoryPolicy()

    scanned = 0
    rewritten = 0
    archived_noise = 0
    namespace_stats: dict[str, dict[str, int]] = {}
    changes: list[BackfillRecordChange] = []

    for row in list(rows or []):
        scanned += 1
        ns = str(row.namespace or "").strip() or "(empty)"
        stats = namespace_stats.setdefault(
            ns,
            {
                "scanned": 0,
                "rewritten": 0,
                "assistant_noise_archived": 0,
            },
        )
        stats["scanned"] += 1

        updated, changed_fields, assistant_noise_archived = backfill_record(
            row,
            policy=policy,
            storage_profile=profile,
            archive_assistant_noise=archive_assistant_noise,
        )
        if not changed_fields:
            continue

        rewritten += 1
        stats["rewritten"] += 1
        if assistant_noise_archived:
            archived_noise += 1
            stats["assistant_noise_archived"] += 1
        changes.append(
            BackfillRecordChange(
                record_id=str(updated.id or ""),
                namespace=ns,
                changed_fields=list(changed_fields or []),
                assistant_noise_archived=bool(assistant_noise_archived),
            )
        )
        if not dry_run:
            manager._store.upsert(updated)

    reindex_result: dict[str, Any] = {}
    if not dry_run and reindex:
        reindex_result = dict(
            manager.reindex_embeddings(
                namespace=(target_namespace or None),
                incremental=False,
            )
            or {}
        )

    return BackfillSummary(
        dry_run=bool(dry_run),
        storage_profile=profile,
        namespace=(target_namespace or "*"),
        scanned=scanned,
        rewritten=rewritten,
        assistant_noise_archived=archived_noise,
        namespace_stats=namespace_stats,
        changes=changes,
        reindex_result=reindex_result,
    )


def backfill_record(
    record: MemoryRecord,
    *,
    policy: MemoryPolicy,
    storage_profile: str,
    archive_assistant_noise: bool = True,
) -> tuple[MemoryRecord, list[str], bool]:
    changed_fields: list[str] = []
    updated = record

    rebuilt_metadata = rebuild_record_metadata(
        record,
        policy=policy,
        storage_profile=storage_profile,
    )
    if dict(rebuilt_metadata or {}) != dict(record.metadata or {}):
        changed_fields.append("metadata")
        updated = _replace_record(updated, metadata=rebuilt_metadata)

    assistant_noise_archived = False
    if archive_assistant_noise and is_assistant_memory_noise_record(updated, policy=policy):
        if updated.status not in {MemoryStatus.ARCHIVED, MemoryStatus.DELETED}:
            changed_fields.append("status")
            changed_fields.append("assistant_noise_archive")
            assistant_noise_archived = True
            next_meta = dict(updated.metadata or {})
            next_meta["previous_status"] = str(updated.status.value)
            next_meta["lifecycle_reason"] = "backfill_assistant_noise_archive"
            next_meta["assistant_noise_archived"] = True
            updated = _replace_record(
                updated,
                metadata=sanitize_storage_metadata(metadata=next_meta, storage_profile=storage_profile),
                status=MemoryStatus.ARCHIVED,
                updated_at=float(time.time()),
                version=int(updated.version) + 1,
                parent_id=(updated.parent_id or updated.id),
            )

    return updated, changed_fields, assistant_noise_archived


def rebuild_record_metadata(
    record: MemoryRecord,
    *,
    policy: MemoryPolicy,
    storage_profile: str,
) -> dict[str, Any]:
    meta = dict(record.metadata or {})
    rebuilt_views = build_memory_views(str(record.text or ""), metadata=meta)
    entity_keys = list(rebuilt_views.get("entity_keys") or [])
    numeric_keys = list(rebuilt_views.get("numeric_keys") or [])
    if entity_keys or numeric_keys:
        compact_views: dict[str, Any] = {}
        if entity_keys:
            compact_views["entity_keys"] = entity_keys
        if numeric_keys:
            compact_views["numeric_keys"] = numeric_keys
        meta["memory_views"] = compact_views
    else:
        meta.pop("memory_views", None)

    source_kind = infer_source_kind(record)
    return policy.sanitize_metadata_for_storage(
        metadata=meta,
        source_kind=source_kind,
        thinking="",
        storage_profile=storage_profile,
        compact_for_storage=True,
    )


def infer_source_kind(record: MemoryRecord) -> MemorySourceKind:
    meta = dict(record.metadata or {})
    raw = str(meta.get("source_kind") or "").strip().lower()
    if raw:
        try:
            return MemorySourceKind(raw)
        except Exception:
            pass

    source_role = str(meta.get("source_role") or "").strip().lower()
    if source_role == "assistant":
        return MemorySourceKind.ASSISTANT_REPLY
    if record.memory_type == MemoryType.TOOL_RESULT:
        return MemorySourceKind.TOOL_RESULT
    return MemorySourceKind.USER


def is_assistant_memory_noise_record(record: MemoryRecord, *, policy: MemoryPolicy) -> bool:
    if record.memory_type not in {MemoryType.MESSAGE, MemoryType.SUMMARY}:
        return False
    if infer_source_kind(record) != MemorySourceKind.ASSISTANT_REPLY:
        return False

    meta = dict(record.metadata or {})
    if str(meta.get("assistant_reply_kind") or "").strip().lower() == "memory_miss_help":
        return True

    decision = policy.decide_assistant_message_write(
        text=str(record.text or ""),
        metadata=meta,
        memory_type=record.memory_type,
        requested_scope=record.scope,
    )
    return str(decision.reason or "").strip().lower() == "assistant_memory_miss_help_temporary_only"


def _replace_record(record: MemoryRecord, **changes: Any) -> MemoryRecord:
    data = record.to_dict()
    data.update(changes)
    for key in ("memory_type", "level", "scope", "status"):
        value = data.get(key)
        if hasattr(value, "value"):
            data[key] = str(value.value)
    return MemoryRecord.from_dict(data)
