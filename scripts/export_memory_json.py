from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import load_config
from memory.memory_manager import MemoryManager
from memory.memory_models import MemoryRecord
from memory.retrieval_projection import build_memory_views


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export Memory V2 records into a JSON file.")
    parser.add_argument(
        "--namespace",
        default="all",
        help="Memory namespace to export. Use 'all' to export every namespace. Default: all",
    )
    parser.add_argument(
        "--type",
        dest="memory_type",
        default="",
        help="Filter by memory type: message, fact, claim, summary, document, document_chunk, tool_result, task_state",
    )
    parser.add_argument(
        "--status",
        default="",
        help="Filter by status: active, stale, superseded, archived, deleted",
    )
    parser.add_argument("--contains", default="", help="Only export records whose text contains this substring")
    parser.add_argument("--limit", type=int, default=0, help="Maximum number of records to export. 0 = all")
    parser.add_argument("--show-embedding", action="store_true", help="Include embedding vectors in the export")
    parser.add_argument(
        "--output",
        default="data/exports/memory_export.json",
        help="Output JSON path. Default: data/exports/memory_export.json",
    )
    parser.add_argument("--list-namespaces", action="store_true", help="Print available namespaces and exit")
    return parser


def _normalize_enum_token(value: str) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def _match_memory_type(record: MemoryRecord, expected: str) -> bool:
    token = _normalize_enum_token(expected)
    if not token:
        return True
    return _normalize_enum_token(record.memory_type.value) == token


def _match_status(record: MemoryRecord, expected: str) -> bool:
    token = _normalize_enum_token(expected)
    if not token:
        return True
    return _normalize_enum_token(record.status.value) == token


def _memory_views(record: MemoryRecord) -> dict[str, Any]:
    return dict(dict(record.metadata or {}).get("memory_views") or {})


def _recomputed_memory_views(record: MemoryRecord) -> dict[str, Any]:
    meta = dict(record.metadata or {})
    recompute_meta = {**meta, "memory_views": {}}
    return dict(build_memory_views(str(record.text or ""), metadata=recompute_meta) or {})


def _views_debug(record: MemoryRecord) -> dict[str, Any]:
    stored = _memory_views(record)
    recomputed = _recomputed_memory_views(record)
    stored_has_keys = bool(list(stored.get("entity_keys") or []) or list(stored.get("numeric_keys") or []))
    recomputed_has_keys = bool(list(recomputed.get("entity_keys") or []) or list(recomputed.get("numeric_keys") or []))
    if stored_has_keys:
        status = "stored_has_keys"
    elif recomputed_has_keys:
        status = "recomputed_has_keys_only"
    else:
        status = "no_structured_keys"
    return {
        "status": status,
        "stored_has_keys": stored_has_keys,
        "recomputed_has_keys": recomputed_has_keys,
        "stored": stored,
        "recomputed": recomputed,
    }


def _effective_memory_views(record: MemoryRecord) -> dict[str, Any]:
    debug = _views_debug(record)
    if str(debug.get("status") or "") == "recomputed_has_keys_only":
        return dict(debug.get("recomputed") or {})
    return _memory_views(record)


def _record_payload(record: MemoryRecord, *, include_embedding: bool) -> dict[str, Any]:
    meta = dict(record.metadata or {})
    views_debug = _views_debug(record)
    views = _effective_memory_views(record)
    payload: dict[str, Any] = {
        "id": str(record.id),
        "memory_type": str(record.memory_type.value),
        "scope": str(record.scope.value),
        "level": str(record.level.value),
        "status": str(record.status.value),
        "namespace": str(record.namespace or ""),
        "importance": float(record.importance),
        "confidence": float(record.confidence),
        "created_at": float(record.created_at),
        "updated_at": float(record.updated_at),
        "expires_at": (float(record.expires_at) if record.expires_at is not None else None),
        "source_event_id": str(record.source_event_id or ""),
        "text": str(record.text or ""),
        "search_text": str(views.get("search_text") or ""),
        "entity_keys": list(views.get("entity_keys") or []),
        "numeric_keys": list(views.get("numeric_keys") or []),
        "tags": list(meta.get("tags") or []) if isinstance(meta.get("tags"), list) else meta.get("tags"),
        "memory_tags": (
            list(meta.get("memory_tags") or [])
            if isinstance(meta.get("memory_tags"), list)
            else meta.get("memory_tags")
        ),
        "memory_views_debug": views_debug,
        "metadata": meta,
    }
    if include_embedding:
        payload["embedding"] = list(record.embedding or [])
    return payload


def _iter_filtered_records(
    manager: MemoryManager,
    *,
    namespace: str,
    memory_type: str,
    status: str,
    contains: str,
) -> list[MemoryRecord]:
    if _normalize_enum_token(namespace) in {"all", "*"}:
        rows = list(manager._store.iter_records())
    else:
        rows = list(manager._store.iter_records(namespace=namespace))
    rows.sort(key=lambda row: (float(row.updated_at), float(row.created_at)), reverse=True)
    needle = str(contains or "").strip().lower()
    out: list[MemoryRecord] = []
    for record in rows:
        if not _match_memory_type(record, memory_type):
            continue
        if not _match_status(record, status):
            continue
        if needle and needle not in str(record.text or "").lower():
            continue
        out.append(record)
    return out


def _namespace_counts(manager: MemoryManager) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in list(manager._store.iter_records()):
        key = str(record.namespace or "").strip() or "(empty)"
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: (-int(item[1]), str(item[0]))))


def _export_payload(
    *,
    manager: MemoryManager,
    records: list[MemoryRecord],
    include_embedding: bool,
    namespace: str,
    memory_type: str,
    status: str,
    contains: str,
) -> dict[str, Any]:
    cfg = load_config()
    namespaces = sorted({str(row.namespace or "") for row in records if str(row.namespace or "").strip()})
    return {
        "exported_at": float(time.time()),
        "filters": {
            "namespace": str(namespace or "all"),
            "memory_type": str(memory_type or ""),
            "status": str(status or ""),
            "contains": str(contains or ""),
            "include_embedding": bool(include_embedding),
        },
        "store": {
            "memory_backend": str(getattr(cfg, "memory_backend", "") or ""),
            "memory_dir": str(getattr(cfg, "memory_dir", "") or ""),
            "db_path": str(getattr(cfg, "db_path", "") or ""),
            "exported_namespaces": namespaces,
            "exported_count": len(records),
            "available_namespaces": _namespace_counts(manager),
        },
        "records": [_record_payload(row, include_embedding=include_embedding) for row in records],
    }


def main() -> int:
    args = _build_parser().parse_args()
    manager = MemoryManager()
    try:
        if bool(args.list_namespaces):
            print(json.dumps(_namespace_counts(manager), ensure_ascii=False, indent=2))
            return 0

        rows = _iter_filtered_records(
            manager,
            namespace=str(args.namespace or "all"),
            memory_type=str(args.memory_type or ""),
            status=str(args.status or ""),
            contains=str(args.contains or ""),
        )
        limit = max(0, int(args.limit or 0))
        if limit > 0:
            rows = rows[:limit]

        payload = _export_payload(
            manager=manager,
            records=rows,
            include_embedding=bool(args.show_embedding),
            namespace=str(args.namespace or "all"),
            memory_type=str(args.memory_type or ""),
            status=str(args.status or ""),
            contains=str(args.contains or ""),
        )

        out_path = Path(str(args.output or "data/exports/memory_export.json")).expanduser()
        if not out_path.is_absolute():
            out_path = (ROOT / out_path).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(str(out_path))
        return 0
    finally:
        manager.close()


if __name__ == "__main__":
    raise SystemExit(main())
