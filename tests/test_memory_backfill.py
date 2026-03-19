from __future__ import annotations

import time

from memory.backfill import run_final_backfill
from memory.memory_models import MemoryLevel, MemoryRecord, MemoryScope, MemoryStatus, MemoryType


class _BackfillStore:
    def __init__(self, rows: list[MemoryRecord]) -> None:
        self._rows = {str(row.id): row for row in list(rows or [])}

    def iter_records(self, namespace: str | None = None) -> list[MemoryRecord]:
        rows = list(self._rows.values())
        if namespace is None:
            return rows
        return [row for row in rows if str(row.namespace or "") == str(namespace or "")]

    def upsert(self, record: MemoryRecord) -> None:
        self._rows[str(record.id)] = record


class _BackfillManager:
    def __init__(self, rows: list[MemoryRecord]) -> None:
        self._store = _BackfillStore(rows)
        self.reindex_calls: list[tuple[str | None, bool]] = []

    def reindex_embeddings(self, *, namespace: str | None = None, incremental: bool = False) -> dict[str, object]:
        self.reindex_calls.append((namespace, incremental))
        return {
            "records_indexed": len(list(self._store.iter_records(namespace=namespace))),
            "namespace": namespace or "*",
            "incremental": bool(incremental),
            "reindex_required": False,
        }


def _record(
    record_id: str,
    text: str,
    *,
    namespace: str = "default",
    memory_type: MemoryType = MemoryType.MESSAGE,
    level: MemoryLevel = MemoryLevel.L2_EPISODIC,
    metadata: dict | None = None,
    status: MemoryStatus = MemoryStatus.ACTIVE,
) -> MemoryRecord:
    now = time.time()
    return MemoryRecord(
        id=record_id,
        text=text,
        memory_type=memory_type,
        level=level,
        scope=MemoryScope.CONVERSATION,
        namespace=namespace,
        metadata=dict(metadata or {}),
        importance=0.5,
        confidence=0.5,
        created_at=now,
        updated_at=now,
        status=status,
    )


def test_final_backfill_rewrites_old_message_metadata_to_compact_profile() -> None:
    manager = _BackfillManager(
        [
            _record(
                "msg:claim-old",
                "I use VS Code.",
                metadata={
                    "source_kind": "user",
                    "memory_analysis": {"entities": [{"type": "tool_name", "canonical": "VS Code"}]},
                    "claim_candidates": [
                        {
                            "subject": "user",
                            "predicate": "uses",
                            "object_surface": "VS Code",
                            "normalized_object": "vs code",
                            "object_type": "tool",
                            "confidence": 0.88,
                            "topic_keys": ["uses", "tool"],
                            "trigger_keys": ["uses", "vs", "code"],
                        }
                    ],
                    "memory_views": {
                        "search_text": "use vs code",
                        "entity_keys": ["old"],
                    },
                },
            )
        ]
    )

    summary = run_final_backfill(
        manager=manager,
        dry_run=False,
        storage_profile="compact",
        archive_assistant_noise=True,
        reindex=False,
    )

    assert summary.rewritten >= 1
    row = manager._store.iter_records(namespace="default")[0]
    metadata = dict(row.metadata or {})
    assert "memory_analysis" not in metadata
    assert "claim_candidates" not in metadata
    assert "search_text" not in metadata
    views = dict(metadata.get("memory_views") or {})
    assert "search_text" not in views
    assert views.get("entity_keys")


def test_final_backfill_archives_old_assistant_memory_miss_noise() -> None:
    manager = _BackfillManager(
        [
            _record(
                "msg:assistant-noise",
                "Не помню твою видеокарту. Проверь сам через winver.",
                metadata={
                    "source_kind": "assistant_reply",
                },
            )
        ]
    )

    summary = run_final_backfill(
        manager=manager,
        dry_run=False,
        storage_profile="compact",
        archive_assistant_noise=True,
        reindex=False,
    )

    assert summary.assistant_noise_archived == 1
    row = manager._store.iter_records(namespace="default")[0]
    assert row.status == MemoryStatus.ARCHIVED
    metadata = dict(row.metadata or {})
    assert metadata.get("assistant_noise_archived") is True
    assert metadata.get("lifecycle_reason") == "backfill_assistant_noise_archive"


def test_final_backfill_reports_namespace_stats_and_runs_reindex() -> None:
    manager = _BackfillManager(
        [
            _record(
                "msg:one",
                "I use VS Code.",
                namespace="alpha",
                metadata={
                    "source_kind": "user",
                    "claim_candidates": [
                        {
                            "subject": "user",
                            "predicate": "uses",
                            "object_surface": "VS Code",
                            "normalized_object": "vs code",
                            "object_type": "tool",
                            "confidence": 0.88,
                            "topic_keys": ["uses", "tool"],
                            "trigger_keys": ["uses", "vs", "code"],
                        }
                    ],
                },
            ),
            _record(
                "msg:two",
                "What did we discuss?",
                namespace="beta",
                metadata={"source_kind": "user"},
            ),
        ]
    )

    summary = run_final_backfill(
        manager=manager,
        dry_run=False,
        storage_profile="compact",
        archive_assistant_noise=True,
        reindex=True,
    )

    assert summary.namespace_stats["alpha"]["scanned"] == 1
    assert summary.namespace_stats["beta"]["scanned"] == 1
    assert manager.reindex_calls == [(None, False)]
    assert dict(summary.reindex_result or {}).get("namespace") == "*"


def test_final_backfill_rebuilds_legacy_episode_summary_from_turns() -> None:
    manager = _BackfillManager(
        [
            _record(
                "turn:1",
                "привет",
                namespace="dialog",
                metadata={"source_kind": "user"},
            ),
            _record(
                "turn:2",
                "ок",
                namespace="dialog",
                metadata={"source_kind": "assistant_reply"},
            ),
            _record(
                "turn:3",
                "Давай разделим память на facts, claims и dialog episodes.",
                namespace="dialog",
                metadata={"source_kind": "user", "topic": "memory"},
            ),
            _record(
                "turn:4",
                "И document memory оставим отдельным контуром.",
                namespace="dialog",
                metadata={"source_kind": "assistant_reply", "topic": "memory"},
            ),
            _record(
                "episode:legacy",
                "Discussed привет.",
                namespace="dialog",
                memory_type=MemoryType.EPISODE,
                metadata={
                    "dialog_episode": {
                        "id": "episode:legacy",
                        "topic": "привет",
                        "turn_ids": ["turn:1", "turn:2", "turn:3", "turn:4"],
                        "summary_short": "Discussed привет.",
                        "summary_reasoning": "Discussed привет.",
                        "decisions": [],
                        "open_questions": [],
                        "participants": ["user", "assistant"],
                        "salience": 0.4,
                        "topic_keys": ["привет"],
                        "entity_keys": [],
                        "created_at": 100.0,
                        "updated_at": 120.0,
                    },
                    "topic": "привет",
                    "summary_short": "Discussed привет.",
                    "summary_reasoning": "Discussed привет.",
                    "turn_ids": ["turn:1", "turn:2", "turn:3", "turn:4"],
                },
            ),
        ]
    )

    summary = run_final_backfill(
        manager=manager,
        namespace="dialog",
        dry_run=False,
        storage_profile="compact",
        archive_assistant_noise=False,
        reindex=False,
    )

    assert summary.rewritten >= 1
    row = next(record for record in manager._store.iter_records(namespace="dialog") if record.memory_type == MemoryType.EPISODE)
    metadata = dict(row.metadata or {})
    episode = dict(metadata.get("dialog_episode") or {})
    assert str(row.text or "").lower() != "discussed привет."
    assert "привет" not in str(metadata.get("summary_short") or "").lower()
    assert "facts, claims и dialog episodes" in str(metadata.get("summary_short") or "").lower()
    assert str(metadata.get("topic") or "") == "memory"
    assert str(episode.get("topic") or "") == "memory"
