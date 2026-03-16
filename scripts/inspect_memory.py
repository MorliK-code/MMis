from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from memory.memory_manager import MemoryManager
from memory.memory_models import MemoryRecord, MemoryStatus, MemoryType


def _configure_stdout() -> None:
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if callable(reconfigure):
        try:
            reconfigure(encoding="utf-8")
        except Exception:
            pass


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect Memory V2 records in a readable form.")
    parser.add_argument(
        "--namespace",
        default="default",
        help="Memory namespace to inspect (default: default). Use 'all' to scan all namespaces.",
    )
    parser.add_argument("--limit", type=int, default=10, help="Maximum number of records to print")
    parser.add_argument(
        "--type",
        dest="memory_type",
        default="",
        help="Filter by memory type: message, fact, summary, document, document_chunk, tool_result, task_state",
    )
    parser.add_argument(
        "--status",
        default="",
        help="Filter by status: active, stale, superseded, archived, deleted",
    )
    parser.add_argument("--contains", default="", help="Only show records whose text contains this substring")
    parser.add_argument("--full-metadata", action="store_true", help="Print full metadata JSON")
    parser.add_argument("--show-embedding", action="store_true", help="Print embedding length and first values")
    parser.add_argument("--json", action="store_true", help="Print records as JSON instead of human-readable blocks")
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


def _text_preview(value: Any, limit: int = 220) -> str:
    text = str(value or "").replace("\r", " ").replace("\n", " ").strip()
    if len(text) <= max(32, int(limit)):
        return text
    return text[: max(32, int(limit)) - 3].rstrip() + "..."


def _memory_views(record: MemoryRecord) -> dict[str, Any]:
    return dict(dict(record.metadata or {}).get("memory_views") or {})


def _record_payload(record: MemoryRecord, *, include_full_metadata: bool, include_embedding: bool) -> dict[str, Any]:
    meta = dict(record.metadata or {})
    views = _memory_views(record)
    payload: dict[str, Any] = {
        "id": str(record.id),
        "memory_type": str(record.memory_type.value),
        "scope": str(record.scope.value),
        "level": str(record.level.value),
        "status": str(record.status.value),
        "namespace": str(record.namespace),
        "importance": float(record.importance),
        "confidence": float(record.confidence),
        "created_at": float(record.created_at),
        "updated_at": float(record.updated_at),
        "source_event_id": str(record.source_event_id or ""),
        "text": str(record.text or ""),
        "search_text": str(views.get("search_text") or ""),
        "entity_keys": list(views.get("entity_keys") or []),
        "numeric_keys": list(views.get("numeric_keys") or []),
        "tags": list(meta.get("tags") or []) if isinstance(meta.get("tags"), list) else meta.get("tags"),
    }
    if include_embedding:
        embedding = list(record.embedding or [])
        payload["embedding_dim"] = len(embedding)
        payload["embedding_preview"] = embedding[:8]
    if include_full_metadata:
        payload["metadata"] = meta
    else:
        payload["metadata_preview"] = {
            "analysis_version": meta.get("analysis_version"),
            "canonical_text": meta.get("canonical_text"),
            "emotion_profile": meta.get("emotion_profile"),
            "memory_entities": meta.get("memory_entities"),
            "numeric_facts": meta.get("numeric_facts"),
            "stable_facts": meta.get("stable_facts"),
        }
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


def _print_human(records: list[MemoryRecord], *, include_full_metadata: bool, include_embedding: bool) -> None:
    if not records:
        print("No records matched.")
        return
    for index, record in enumerate(records, start=1):
        payload = _record_payload(
            record,
            include_full_metadata=include_full_metadata,
            include_embedding=include_embedding,
        )
        print("=" * 100)
        print(f"[{index}] {payload['id']}")
        print(
            f"type={payload['memory_type']} scope={payload['scope']} level={payload['level']} "
            f"status={payload['status']} namespace={payload['namespace']}"
        )
        print(
            f"importance={payload['importance']:.3f} confidence={payload['confidence']:.3f} "
            f"updated_at={payload['updated_at']:.3f}"
        )
        if payload["source_event_id"]:
            print(f"source_event_id={payload['source_event_id']}")
        print(f"text: {_text_preview(payload['text'])}")
        if payload["search_text"]:
            print(f"search_text: {_text_preview(payload['search_text'])}")
        if payload["entity_keys"]:
            print("entity_keys:", json.dumps(payload["entity_keys"], ensure_ascii=False))
        if payload["numeric_keys"]:
            print("numeric_keys:", json.dumps(payload["numeric_keys"], ensure_ascii=False))
        if payload["tags"]:
            print("tags:", json.dumps(payload["tags"], ensure_ascii=False))
        if include_embedding:
            print(
                "embedding:",
                json.dumps(
                    {
                        "dim": payload.get("embedding_dim", 0),
                        "preview": payload.get("embedding_preview", []),
                    },
                    ensure_ascii=False,
                ),
            )
        if include_full_metadata:
            print("metadata:")
            print(json.dumps(payload["metadata"], ensure_ascii=False, indent=2))
        else:
            print("metadata_preview:")
            print(json.dumps(payload["metadata_preview"], ensure_ascii=False, indent=2))


def main() -> int:
    _configure_stdout()
    args = _build_parser().parse_args()
    manager = MemoryManager()
    try:
        if bool(args.list_namespaces):
            counts = _namespace_counts(manager)
            if not counts:
                print("No namespaces found.")
                return 0
            print("Namespaces:")
            for name, count in counts.items():
                print(f"- {name}: {count}")
            return 0
        rows = _iter_filtered_records(
            manager,
            namespace=str(args.namespace or "default"),
            memory_type=str(args.memory_type or ""),
            status=str(args.status or ""),
            contains=str(args.contains or ""),
        )[: max(1, int(args.limit or 10))]
        if bool(args.json):
            payload = [
                _record_payload(
                    row,
                    include_full_metadata=bool(args.full_metadata),
                    include_embedding=bool(args.show_embedding),
                )
                for row in rows
            ]
            print(json.dumps(payload, ensure_ascii=False, indent=2))
            return 0
        _print_human(
            rows,
            include_full_metadata=bool(args.full_metadata),
            include_embedding=bool(args.show_embedding),
        )
        return 0
    finally:
        manager.close()


if __name__ == "__main__":
    raise SystemExit(main())
