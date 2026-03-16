from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from memory.memory_manager import MemoryManager
from memory.memory_models import MemoryRecord


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Rename a memory namespace in Memory V2 storage.")
    parser.add_argument("--from", dest="source_namespace", required=True, help="Old namespace")
    parser.add_argument("--to", dest="target_namespace", required=True, help="New namespace")
    parser.add_argument("--dry-run", action="store_true", help="Show what would change without writing")
    parser.add_argument(
        "--rewrite-events",
        action="store_true",
        help="Also rewrite events_v2.jsonl payload namespace/conversation_id fields when they match the old namespace",
    )
    return parser


def _clone_with_namespace(record: MemoryRecord, *, namespace: str) -> MemoryRecord:
    return MemoryRecord(
        id=record.id,
        text=record.text,
        memory_type=record.memory_type,
        level=record.level,
        scope=record.scope,
        namespace=str(namespace or "default"),
        metadata=dict(record.metadata or {}),
        embedding=list(record.embedding or []) if isinstance(record.embedding, list) else None,
        importance=record.importance,
        confidence=record.confidence,
        created_at=record.created_at,
        updated_at=record.updated_at,
        expires_at=record.expires_at,
        status=record.status,
        version=record.version,
        parent_id=record.parent_id,
        chunk_index=record.chunk_index,
        source_event_id=record.source_event_id,
        embedding_model=record.embedding_model,
        embedding_fingerprint=record.embedding_fingerprint,
        embedding_version=record.embedding_version,
    )


def _backup_file(path: Path) -> Path:
    backup = path.with_suffix(path.suffix + ".bak")
    shutil.copy2(path, backup)
    return backup


def _rewrite_events_file(path: Path, *, source_namespace: str, target_namespace: str) -> int:
    if not path.exists():
        return 0
    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except Exception:
        return 0

    changed = 0
    out_lines: list[str] = []
    for raw in lines:
        line = str(raw or "").strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except Exception:
            out_lines.append(raw)
            continue
        if not isinstance(row, dict):
            out_lines.append(raw)
            continue
        payload = dict(row.get("payload") or {})
        row_changed = False
        for key in ("namespace", "conversation_id"):
            if str(payload.get(key) or "") == source_namespace:
                payload[key] = target_namespace
                row_changed = True
        if row_changed:
            row["payload"] = payload
            changed += 1
        out_lines.append(json.dumps(row, ensure_ascii=False))

    if changed > 0:
        _backup_file(path)
        path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    return changed


def main() -> int:
    args = _build_parser().parse_args()
    source_namespace = str(args.source_namespace or "").strip()
    target_namespace = str(args.target_namespace or "").strip()
    if not source_namespace or not target_namespace:
        print("Both --from and --to are required.")
        return 2
    if source_namespace == target_namespace:
        print("Source and target namespaces are the same; nothing to do.")
        return 0

    manager = MemoryManager()
    try:
        source_rows = list(manager._store.iter_records(namespace=source_namespace))
        working_matches = [row for row in list(manager._working_records or []) if str(row.namespace or "") == source_namespace]
        private_runtime_matches = {
            key: value
            for key, value in dict(manager._private_runtime or {}).items()
            if str(dict(value or {}).get("namespace") or "") == source_namespace
        }

        print(f"records_to_move: {len(source_rows)}")
        print(f"working_records_to_move: {len(working_matches)}")
        print(f"private_runtime_entries_to_move: {len(private_runtime_matches)}")

        if bool(args.dry_run):
            print("dry_run: no changes written")
            return 0

        for record in source_rows:
            manager._store.upsert(_clone_with_namespace(record, namespace=target_namespace))

        updated_working: list[MemoryRecord] = []
        for row in list(manager._working_records or []):
            if str(row.namespace or "") == source_namespace:
                updated_working.append(_clone_with_namespace(row, namespace=target_namespace))
            else:
                updated_working.append(row)
        manager._working_records = updated_working

        updated_private_runtime: dict[str, dict] = {}
        for key, value in dict(manager._private_runtime or {}).items():
            payload = dict(value or {})
            if str(payload.get("namespace") or "") == source_namespace:
                payload["namespace"] = target_namespace
            updated_private_runtime[str(key)] = payload
        manager._private_runtime = updated_private_runtime

        manager._save_state()

        changed_events = 0
        if bool(args.rewrite_events):
            changed_events = _rewrite_events_file(
                Path(manager._event_store.path),
                source_namespace=source_namespace,
                target_namespace=target_namespace,
            )

        print(f"moved_records: {len(source_rows)}")
        print(f"updated_working_records: {len(working_matches)}")
        print(f"updated_private_runtime_entries: {len(private_runtime_matches)}")
        if bool(args.rewrite_events):
            print(f"rewritten_events: {changed_events}")
        print("done")
        return 0
    finally:
        manager.close()


if __name__ == "__main__":
    raise SystemExit(main())
