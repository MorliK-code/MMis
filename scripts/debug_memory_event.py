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
from memory.memory_models import MemoryRecord, MemoryType


def _configure_stdout() -> None:
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if callable(reconfigure):
        try:
            reconfigure(encoding="utf-8")
        except Exception:
            pass


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect one memory event and its related fact records.")
    parser.add_argument("--event-id", default="", help="Exact source_event_id to inspect")
    parser.add_argument("--contains", default="", help="Fallback: find message events whose text contains this substring")
    parser.add_argument("--namespace", default="all", help="Namespace to inspect. Use 'all' for every namespace.")
    parser.add_argument("--json", action="store_true", help="Print JSON")
    return parser


def _iter_rows(manager: MemoryManager, namespace: str) -> list[MemoryRecord]:
    token = str(namespace or "").strip().lower()
    if token in {"", "all", "*"}:
        return list(manager._store.iter_records())
    return list(manager._store.iter_records(namespace=namespace))


def _preview_metadata(record: MemoryRecord) -> dict[str, Any]:
    meta = dict(record.metadata or {})
    return {
        "analysis_version": meta.get("analysis_version"),
        "memory_views": dict(meta.get("memory_views") or {}),
        "memory_entities": list(meta.get("memory_entities") or []),
        "numeric_facts": list(meta.get("numeric_facts") or []),
        "stable_facts": list(meta.get("stable_facts") or []),
        "tags": list(meta.get("tags") or []),
        "memory_tags": list(meta.get("memory_tags") or []),
        "source_kind": meta.get("source_kind"),
        "assistant_write_reason": meta.get("assistant_write_reason"),
        "write_policy": meta.get("write_policy"),
    }


def _record_payload(record: MemoryRecord) -> dict[str, Any]:
    payload = record.to_dict()
    payload["metadata_preview"] = _preview_metadata(record)
    return payload


def _resolve_event_ids(manager: MemoryManager, *, namespace: str, event_id: str, contains: str) -> list[str]:
    if str(event_id or "").strip():
        return [str(event_id).strip()]
    needle = str(contains or "").strip().lower()
    if not needle:
        return []
    out: list[str] = []
    for row in _iter_rows(manager, namespace):
        if row.memory_type != MemoryType.MESSAGE:
            continue
        if needle not in str(row.text or "").lower():
            continue
        token = str(row.source_event_id or "").strip()
        if token and token not in out:
            out.append(token)
    return out


def _event_payload(manager: MemoryManager, *, namespace: str, event_id: str) -> dict[str, Any]:
    rows = [row for row in _iter_rows(manager, namespace) if str(row.source_event_id or "") == str(event_id or "")]
    rows.sort(key=lambda row: (float(row.created_at), str(row.id)))
    message_rows = [row for row in rows if row.memory_type == MemoryType.MESSAGE]
    fact_rows = [row for row in rows if row.memory_type == MemoryType.FACT]
    return {
        "event_id": str(event_id or ""),
        "message_records": [_record_payload(row) for row in message_rows],
        "fact_records": [_record_payload(row) for row in fact_rows],
    }


def main() -> int:
    _configure_stdout()
    args = _build_parser().parse_args()
    manager = MemoryManager()
    try:
        event_ids = _resolve_event_ids(
            manager,
            namespace=str(args.namespace or "all"),
            event_id=str(args.event_id or ""),
            contains=str(args.contains or ""),
        )
        if not event_ids:
            print("No matching event ids found.", file=sys.stderr)
            return 1
        payload = {
            "namespace": str(args.namespace or "all"),
            "events": [
                _event_payload(
                    manager,
                    namespace=str(args.namespace or "all"),
                    event_id=event_token,
                )
                for event_token in event_ids
            ],
        }
        if bool(args.json):
            print(json.dumps(payload, ensure_ascii=False, indent=2))
            return 0
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0
    finally:
        manager.close()


if __name__ == "__main__":
    raise SystemExit(main())
