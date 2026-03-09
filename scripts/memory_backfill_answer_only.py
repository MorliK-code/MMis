from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config.settings import load_config
from memory.memory_manager import MemoryManager
from memory.memory_models import MemoryEvent, MemoryScope, MemoryType
from memory.text_sanitizer import sanitize_assistant_memory_text
from memory.vector_store import embed_text
from utils.datetime_local import parse_time_to_epoch


TARGET_FILES = (
    "events.jsonl",
    "short_memory.json",
    "long_memory_docs.json",
    "vector_store.json",
    "user_profile_store.json",
    "assistant_profile_store.json",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backfill memory text to keep assistant `text` answer-only across all memory stores.",
    )
    parser.add_argument("--memory-dir", type=str, default="", help="Path to memory storage directory.")
    parser.add_argument("--backup-dir", type=str, default="", help="Base directory for timestamped backups.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned changes without writing files.")
    parser.add_argument("--rebuild-facts", action="store_true", help="Rebuild profile facts after cleanup.")
    parser.add_argument("--yes", action="store_true", help="Confirm non-dry-run write operations.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = load_config()
    memory_dir = (
        Path(str(args.memory_dir or "")).expanduser().resolve()
        if str(args.memory_dir or "").strip()
        else Path(cfg.memory_dir).expanduser().resolve()
    )
    if not memory_dir.exists() or not memory_dir.is_dir():
        print(f"Memory dir not found: {memory_dir}")
        return 1
    if not args.dry_run and not args.yes:
        print("Refusing to modify files without --yes. Use --dry-run for preview.")
        return 2

    events_path = memory_dir / "events.jsonl"
    short_path = memory_dir / "short_memory.json"
    long_path = memory_dir / "long_memory_docs.json"
    vector_path = memory_dir / "vector_store.json"
    user_profile_path = memory_dir / "user_profile_store.json"
    assistant_profile_path = memory_dir / "assistant_profile_store.json"

    events_rows = _read_jsonl(events_path)
    short_payload = _read_json(short_path, {"items": [], "rolling_summary": "", "rolling_summary_meta": {}})
    long_payload = _read_json(long_path, {"docs": []})
    vector_payload = _read_json(vector_path, {"records": []})

    cleanup_stats: dict[str, int] = {
        "events_cleaned": 0,
        "short_cleaned": 0,
        "long_cleaned": 0,
        "vector_cleaned": 0,
        "vector_reembedded": 0,
    }

    events_rows, changed = _cleanup_events(events_rows)
    cleanup_stats["events_cleaned"] = changed
    short_payload, changed = _cleanup_short(short_payload)
    cleanup_stats["short_cleaned"] = changed
    long_payload, changed = _cleanup_long(long_payload)
    cleanup_stats["long_cleaned"] = changed
    vector_payload, changed, reembedded = _cleanup_vector(vector_payload)
    cleanup_stats["vector_cleaned"] = changed
    cleanup_stats["vector_reembedded"] = reembedded

    print(f"Memory dir: {memory_dir}")
    print("Cleanup summary:")
    print(f"  events_cleaned={cleanup_stats['events_cleaned']}")
    print(f"  short_cleaned={cleanup_stats['short_cleaned']}")
    print(f"  long_cleaned={cleanup_stats['long_cleaned']}")
    print(f"  vector_cleaned={cleanup_stats['vector_cleaned']}")
    print(f"  vector_reembedded={cleanup_stats['vector_reembedded']}")

    if args.dry_run:
        if args.rebuild_facts:
            print("Dry-run: --rebuild-facts requested (rebuild simulation skipped, no files changed).")
        return 0

    backup_root = (
        Path(str(args.backup_dir)).expanduser().resolve()
        if str(args.backup_dir or "").strip()
        else (memory_dir / "backups")
    )
    backup_dir = backup_root / f"answer_only_{time.strftime('%Y%m%d_%H%M%S')}"
    backup_count = _backup_memory_files(memory_dir=memory_dir, backup_dir=backup_dir)

    _write_jsonl_atomic(events_path, events_rows)
    _write_json_atomic(short_path, short_payload)
    _write_json_atomic(long_path, long_payload)
    _write_json_atomic(vector_path, vector_payload)

    print(f"Backup created: {backup_dir} (files={backup_count})")

    if args.rebuild_facts:
        rebuild_stats = _rebuild_facts(
            memory_dir=memory_dir,
            events_rows=events_rows,
            user_profile_path=user_profile_path,
            assistant_profile_path=assistant_profile_path,
            long_path=long_path,
            vector_path=vector_path,
        )
        print("Facts rebuild summary:")
        print(f"  long_fact_removed={rebuild_stats['long_fact_removed']}")
        print(f"  vector_fact_removed={rebuild_stats['vector_fact_removed']}")
        print(f"  facts_extracted={rebuild_stats['facts_extracted']}")
        print(f"  user_profile_keys={rebuild_stats['user_profile_keys']}")
        print(f"  assistant_profile_keys={rebuild_stats['assistant_profile_keys']}")

    print("Backfill completed.")
    return 0


def _cleanup_events(rows: list[Any]) -> tuple[list[Any], int]:
    changed = 0
    out: list[Any] = []
    for raw in list(rows or []):
        if not isinstance(raw, dict):
            out.append(raw)
            continue
        row = dict(raw)
        payload = _coerce_dict(row.get("payload"))
        role = _detect_message_role(row=row, payload=payload)
        if role == "assistant":
            original = str(payload.get("text") or "").strip()
            cleaned = sanitize_assistant_memory_text(text=original).text
            if cleaned != original:
                payload["text"] = cleaned
                row["payload"] = payload
                changed += 1
        out.append(row)
    return out, changed


def _cleanup_short(payload: dict[str, Any]) -> tuple[dict[str, Any], int]:
    body = _coerce_dict(payload)
    items = list(body.get("items") or [])
    changed = 0
    out_items: list[dict[str, Any]] = []
    for raw in items:
        if not isinstance(raw, dict):
            continue
        row = dict(raw)
        role = str(row.get("role") or "").strip().lower()
        if role == "assistant":
            original = str(row.get("text") or "").strip()
            cleaned = sanitize_assistant_memory_text(text=original).text
            if cleaned != original:
                row["text"] = cleaned
                changed += 1
        out_items.append(row)
    body["items"] = out_items
    return body, changed


def _cleanup_long(payload: dict[str, Any]) -> tuple[dict[str, Any], int]:
    body = _coerce_dict(payload)
    docs = list(body.get("docs") or [])
    changed = 0
    out_docs: list[dict[str, Any]] = []
    for raw in docs:
        if not isinstance(raw, dict):
            continue
        row = dict(raw)
        meta = _coerce_dict(row.get("meta"))
        source = str(row.get("source") or "").strip().lower()
        role = str(meta.get("role") or "").strip().lower()
        if source == "chat" or role == "assistant":
            original = str(row.get("text") or "").strip()
            cleaned = sanitize_assistant_memory_text(text=original).text
            if cleaned != original:
                row["text"] = cleaned
                changed += 1
        out_docs.append(row)
    body["docs"] = out_docs
    return body, changed


def _cleanup_vector(payload: dict[str, Any]) -> tuple[dict[str, Any], int, int]:
    body = _coerce_dict(payload)
    records = [dict(x) for x in list(body.get("records") or []) if isinstance(x, dict)]
    dim = _detect_vector_dim(records)
    changed = 0
    reembedded = 0
    out: list[dict[str, Any]] = []
    for row in records:
        meta = _coerce_dict(row.get("metadata"))
        if _should_clean_vector_record(row=row, meta=meta):
            original = str(row.get("text") or "").strip()
            cleaned = sanitize_assistant_memory_text(text=original).text
            if cleaned != original:
                row["text"] = cleaned
                row["embedding"] = embed_text(cleaned, dim=dim)
                changed += 1
                reembedded += 1
        out.append(row)
    body["records"] = out
    return body, changed, reembedded


def _should_clean_vector_record(*, row: dict[str, Any], meta: dict[str, Any]) -> bool:
    role = str(meta.get("role") or "").strip().lower()
    kind = str(meta.get("type") or "").strip().lower()
    rec_id = str(row.get("id") or "").strip().lower()
    source = str(meta.get("source") or "").strip().lower()
    if role == "assistant" and kind in {"message", "summary"}:
        return True
    if rec_id.startswith("msg:") and role == "assistant":
        return True
    if rec_id.startswith("doc:") and (role == "assistant" or kind == "summary" or source == "chat"):
        return True
    return False


def _detect_vector_dim(records: list[dict[str, Any]]) -> int:
    for row in list(records or []):
        emb = row.get("embedding")
        if isinstance(emb, list) and emb:
            return max(32, int(len(emb)))
    return 128


def _detect_message_role(*, row: dict[str, Any], payload: dict[str, Any]) -> str:
    role = str(payload.get("role") or "").strip().lower()
    if role in {"assistant", "user"}:
        return role
    event_type = str(row.get("type") or "").strip().lower()
    if event_type.endswith("_message"):
        prefix = event_type.replace("_message", "", 1)
        if prefix in {"assistant", "user"}:
            return prefix
    return ""


def _backup_memory_files(*, memory_dir: Path, backup_dir: Path) -> int:
    backup_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for name in TARGET_FILES:
        src = memory_dir / name
        if not src.exists():
            continue
        shutil.copy2(src, backup_dir / name)
        copied += 1
    return copied


def _rebuild_facts(
    *,
    memory_dir: Path,
    events_rows: list[Any],
    user_profile_path: Path,
    assistant_profile_path: Path,
    long_path: Path,
    vector_path: Path,
) -> dict[str, int]:
    long_payload = _read_json(long_path, {"docs": []})
    vector_payload = _read_json(vector_path, {"records": []})

    long_docs = [dict(x) for x in list(long_payload.get("docs") or []) if isinstance(x, dict)]
    vector_records = [dict(x) for x in list(vector_payload.get("records") or []) if isinstance(x, dict)]

    long_kept: list[dict[str, Any]] = []
    removed_long = 0
    for row in long_docs:
        if _is_fact_doc(row):
            removed_long += 1
            continue
        long_kept.append(row)

    vector_kept: list[dict[str, Any]] = []
    removed_vector = 0
    for row in vector_records:
        if _is_fact_record(row):
            removed_vector += 1
            continue
        vector_kept.append(row)

    _write_json_atomic(long_path, {"docs": long_kept})
    _write_json_atomic(vector_path, {"records": vector_kept})
    _write_json_atomic(user_profile_path, _empty_profile_payload())
    _write_json_atomic(assistant_profile_path, _empty_profile_payload())

    manager = MemoryManager(root_dir=memory_dir)

    facts_extracted = 0
    for idx, row in enumerate(_iter_message_events(events_rows)):
        payload = _coerce_dict(row.get("payload"))
        text = str(payload.get("text") or "").strip()
        if not text:
            continue
        role = _detect_message_role(row=row, payload=payload)
        if role not in {"assistant", "user"}:
            continue
        meta = _coerce_dict(payload.get("metadata"))
        ids = _coerce_dict(payload.get("ids"))
        event_id = str(row.get("event_id") or f"rebuild:{idx}")
        ingest = manager.ingest_event(
            MemoryEvent(
                role=role,
                text=text,
                namespace=str(meta.get("conversation_id") or "default"),
                scope=MemoryScope.CONVERSATION,
                memory_type=MemoryType.MESSAGE,
                metadata={**meta, **ids, "event_id": event_id},
            )
        )
        facts_extracted += len(list(ingest.extracted_facts or []))

    manager.close()

    user_keys = _profile_key_count(user_profile_path)
    assistant_keys = _profile_key_count(assistant_profile_path)
    return {
        "long_fact_removed": removed_long,
        "vector_fact_removed": removed_vector,
        "facts_extracted": facts_extracted,
        "user_profile_keys": user_keys,
        "assistant_profile_keys": assistant_keys,
    }


def _iter_message_events(rows: list[Any]) -> list[dict[str, Any]]:
    collected: list[tuple[float, int, dict[str, Any]]] = []
    for idx, raw in enumerate(list(rows or [])):
        if not isinstance(raw, dict):
            continue
        row = dict(raw)
        payload = _coerce_dict(row.get("payload"))
        role = _detect_message_role(row=row, payload=payload)
        if role not in {"assistant", "user"}:
            continue
        ts = parse_time_to_epoch(row.get("ts"), 0.0)
        collected.append((float(ts), idx, row))
    collected.sort(key=lambda x: (x[0], x[1]))
    return [item[2] for item in collected]


def _is_fact_doc(row: dict[str, Any]) -> bool:
    source = str(row.get("source") or "").strip().lower()
    if source == "fact":
        return True
    tags = {str(x).strip().lower() for x in list(row.get("tags") or []) if str(x).strip()}
    if "fact" in tags:
        return True
    meta = _coerce_dict(row.get("meta"))
    return isinstance(meta.get("fact"), dict)


def _is_fact_record(row: dict[str, Any]) -> bool:
    rec_id = str(row.get("id") or "").strip().lower()
    if rec_id.startswith("fact:"):
        return True
    meta = _coerce_dict(row.get("metadata"))
    return str(meta.get("type") or "").strip().lower() == "fact" or str(meta.get("source") or "").strip().lower() == "fact"


def _profile_key_count(path: Path) -> int:
    payload = _read_json(path, _empty_profile_payload())
    profiles = _coerce_dict(payload.get("profiles"))
    total = 0
    for row in profiles.values():
        if isinstance(row, dict):
            total += len(dict(row))
    return total


def _empty_profile_payload() -> dict[str, Any]:
    return {
        "profiles": {},
        "versions": {},
        "pending_facts": {},
        "confirmed_facts": {},
    }


def _read_json(path: Path, default: dict[str, Any]) -> dict[str, Any]:
    if not path.exists():
        return dict(default)
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return dict(default)
    if isinstance(payload, dict):
        return payload
    return dict(default)


def _read_jsonl(path: Path) -> list[Any]:
    if not path.exists():
        return []
    out: list[Any] = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        raw = str(line or "").strip()
        if not raw:
            continue
        try:
            parsed = json.loads(raw)
        except Exception:
            out.append(raw)
            continue
        out.append(parsed)
    return out


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    _write_text_atomic(path, text + "\n")


def _write_jsonl_atomic(path: Path, rows: list[Any]) -> None:
    lines: list[str] = []
    for row in list(rows or []):
        if isinstance(row, dict):
            lines.append(json.dumps(row, ensure_ascii=False))
        else:
            lines.append(str(row))
    text = "\n".join(lines).rstrip() + ("\n" if lines else "")
    _write_text_atomic(path, text)


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp-{time.time_ns()}")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _coerce_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if value is None:
        return {}
    try:
        return dict(value)
    except Exception:
        return {}


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
