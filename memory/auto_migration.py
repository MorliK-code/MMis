from __future__ import annotations

import json
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

from memory.event_store import EventStore
from memory.fact_extractor import FactExtractor, MODE_BALANCED
from memory.long_memory import LongMemory
from memory.memory_manager import MemoryManager
from memory.profile_store import AssistantProfileStore, UserProfileStore
from memory.short_memory import ShortMemory
from memory.text_sanitizer import sanitize_assistant_memory_text
from memory.vector_store import EMBED_VERSION, VectorStore, embed_text
from utils.datetime_local import now_local_iso, parse_time_to_epoch
from utils.logger import get_logger, log_json


LOGGER = get_logger(__name__)
MIGRATION_STATE_FILE = "memory_migration_state.json"
TARGET_FILES = (
    "events.jsonl",
    "short_memory.json",
    "long_memory_docs.json",
    "vector_store.json",
    "user_profile_store.json",
    "assistant_profile_store.json",
)


def run_auto_migration(
    *,
    memory_dir: str | Path,
    target_schema_version: int,
    auto_on_start: bool = True,
    facts_scope: str = "user_only",
) -> dict[str, Any]:
    root = Path(memory_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    state_path = root / MIGRATION_STATE_FILE
    state = _read_json(state_path, {})
    current_version = int(state.get("schema_version") or 0)
    target_version = max(1, int(target_schema_version))
    mode = str(facts_scope or "").strip().lower() or "user_only"

    if not bool(auto_on_start):
        return {
            "ran": False,
            "success": True,
            "reason": "disabled",
            "schema_version": current_version,
            "target_schema_version": target_version,
        }
    if current_version >= target_version:
        return {
            "ran": False,
            "success": True,
            "reason": "up_to_date",
            "schema_version": current_version,
            "target_schema_version": target_version,
        }

    stamp = time.strftime("%Y%m%d_%H%M%S")
    backup_dir = root / "backups" / f"auto_migrate_{stamp}"
    backup_count = _backup_files(root=root, backup_dir=backup_dir, names=TARGET_FILES + (MIGRATION_STATE_FILE,))
    log_json(
        LOGGER,
        "migration_start",
        memory_dir=str(root),
        from_schema=current_version,
        to_schema=target_version,
        backup_dir=str(backup_dir),
        backup_files=backup_count,
    )

    try:
        events_rows = _read_jsonl(root / "events.jsonl")
        short_payload = _read_json(root / "short_memory.json", {"items": [], "rolling_summary": "", "rolling_summary_meta": {}})
        long_payload = _read_json(root / "long_memory_docs.json", {"docs": []})
        vector_payload = _read_json(root / "vector_store.json", {"records": []})

        cleaned_events, events_cleaned = _cleanup_events(events_rows)
        cleaned_short, short_cleaned = _cleanup_short(short_payload)
        cleaned_long, long_cleaned = _cleanup_long(long_payload)
        cleaned_vector, vector_cleaned, vector_reembedded = _cleanup_vector(vector_payload)

        non_fact_docs, long_fact_removed = _drop_fact_docs(list(cleaned_long.get("docs") or []))
        non_fact_records, vector_fact_removed = _drop_fact_records(list(cleaned_vector.get("records") or []))
        vector_dim = _detect_vector_dim(non_fact_records)

        rebuilt = _rebuild_fact_artifacts_from_events(
            events_rows=cleaned_events,
            vector_dim=vector_dim,
            facts_scope=mode,
        )
        fact_docs = list(rebuilt.get("fact_docs") or [])
        fact_records = list(rebuilt.get("fact_records") or [])
        user_profile_payload = dict(rebuilt.get("user_profile_payload") or _empty_profile_payload())
        assistant_profile_payload = dict(rebuilt.get("assistant_profile_payload") or _empty_profile_payload())
        if mode == "user_only":
            assistant_profile_payload = _empty_profile_payload()

        long_final = {"docs": list(non_fact_docs) + list(fact_docs)}
        vector_final = {
            "embed_version": EMBED_VERSION,
            "records": list(non_fact_records) + list(fact_records),
        }
        for row in list(vector_final.get("records") or []):
            if not isinstance(row, dict):
                continue
            metadata = dict(row.get("metadata") or {})
            metadata["embed_version"] = EMBED_VERSION
            row["metadata"] = metadata

        _write_jsonl_atomic(root / "events.jsonl", cleaned_events)
        _write_json_atomic(root / "short_memory.json", cleaned_short)
        _write_json_atomic(root / "long_memory_docs.json", long_final)
        _write_json_atomic(root / "vector_store.json", vector_final)
        _write_json_atomic(root / "user_profile_store.json", _normalize_profile_payload(user_profile_payload))
        _write_json_atomic(root / "assistant_profile_store.json", _normalize_profile_payload(assistant_profile_payload))

        stats = {
            "events_cleaned": events_cleaned,
            "short_cleaned": short_cleaned,
            "long_cleaned": long_cleaned,
            "vector_cleaned": vector_cleaned,
            "vector_reembedded": vector_reembedded,
            "long_fact_removed": long_fact_removed,
            "vector_fact_removed": vector_fact_removed,
            "facts_extracted": int(rebuilt.get("facts_extracted") or 0),
            "fact_docs_rebuilt": len(fact_docs),
            "fact_records_rebuilt": len(fact_records),
        }
        state_payload = {
            "schema_version": target_version,
            "status": "success",
            "updated_at": now_local_iso(),
            "backup_dir": str(backup_dir),
            "stats": stats,
        }
        _write_json_atomic(state_path, state_payload)
        log_json(
            LOGGER,
            "migration_success",
            memory_dir=str(root),
            from_schema=current_version,
            to_schema=target_version,
            backup_dir=str(backup_dir),
            stats=stats,
        )
        return {
            "ran": True,
            "success": True,
            "schema_version": target_version,
            "backup_dir": str(backup_dir),
            "stats": stats,
        }
    except Exception as exc:
        _restore_from_backup(root=root, backup_dir=backup_dir, names=TARGET_FILES)
        fail_state = {
            "schema_version": current_version,
            "status": "failed",
            "updated_at": now_local_iso(),
            "backup_dir": str(backup_dir),
            "error": f"{type(exc).__name__}: {exc}",
        }
        _write_json_atomic(state_path, fail_state)
        log_json(
            LOGGER,
            "migration_fail",
            memory_dir=str(root),
            from_schema=current_version,
            to_schema=target_version,
            backup_dir=str(backup_dir),
            error=f"{type(exc).__name__}: {exc}",
        )
        return {
            "ran": True,
            "success": False,
            "schema_version": current_version,
            "backup_dir": str(backup_dir),
            "error": f"{type(exc).__name__}: {exc}",
        }


def _rebuild_fact_artifacts_from_events(
    *,
    events_rows: list[Any],
    vector_dim: int,
    facts_scope: str,
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="mmis_fact_migrate_") as tmpdir:
        tmp = Path(tmpdir)
        manager = MemoryManager(
            short_memory=ShortMemory(path=tmp / "short_memory.json", autosave=False),
            long_memory=LongMemory(path=tmp / "long_memory_docs.json"),
            vector_store=VectorStore(path=tmp / "vector_store.json", dim=max(32, int(vector_dim))),
            fact_extractor=FactExtractor(),
            user_profile_store=UserProfileStore(path=tmp / "user_profile_store.json"),
            assistant_profile_store=AssistantProfileStore(path=tmp / "assistant_profile_store.json"),
            event_store=EventStore(path=tmp / "events.jsonl"),
            facts_scope=str(facts_scope or "user_only"),
            include_pending_facts_in_retrieval=False,
        )
        facts_extracted = 0
        for idx, row in enumerate(_iter_message_events(events_rows)):
            payload = _coerce_dict(row.get("payload"))
            role = _detect_message_role(row=row, payload=payload)
            if role not in {"assistant", "user"}:
                continue
            if facts_scope == "user_only" and role != "user":
                continue
            text = str(payload.get("text") or "").strip()
            if not text:
                continue
            meta = _coerce_dict(payload.get("metadata"))
            ids = _coerce_dict(payload.get("ids"))
            profile = str(ids.get("quality_profile") or meta.get("quality_profile") or MODE_BALANCED).upper()
            profile_id = str(ids.get("profile_id") or meta.get("user_id") or "default")
            event_id = str(row.get("event_id") or f"migrate:{idx}")
            facts = manager.fact_extractor.extract(
                text=text,
                metadata={"event_id": event_id, **meta},
                speaker=role,
                mode=profile,
            )
            if not facts:
                continue
            manager.write_facts(facts, profile_id=profile_id)
            facts_extracted += len(facts)

        long_payload = _read_json(tmp / "long_memory_docs.json", {"docs": []})
        vector_payload = _read_json(tmp / "vector_store.json", {"records": []})
        user_payload = _read_json(tmp / "user_profile_store.json", _empty_profile_payload())
        assistant_payload = _read_json(tmp / "assistant_profile_store.json", _empty_profile_payload())
        fact_docs = [dict(x) for x in list(long_payload.get("docs") or []) if _is_fact_doc(dict(x or {}))]
        fact_records = [dict(x) for x in list(vector_payload.get("records") or []) if _is_fact_record(dict(x or {}))]
        return {
            "fact_docs": fact_docs,
            "fact_records": fact_records,
            "user_profile_payload": _normalize_profile_payload(user_payload),
            "assistant_profile_payload": _normalize_profile_payload(assistant_payload),
            "facts_extracted": facts_extracted,
        }


def _cleanup_events(rows: list[Any]) -> tuple[list[Any], int]:
    changed = 0
    out: list[Any] = []
    for raw in list(rows or []):
        if not isinstance(raw, dict):
            out.append(raw)
            continue
        row = dict(raw)
        payload = _coerce_dict(row.get("payload"))
        if _detect_message_role(row=row, payload=payload) == "assistant":
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
    changed = 0
    out_items: list[dict[str, Any]] = []
    for raw in list(body.get("items") or []):
        if not isinstance(raw, dict):
            continue
        row = dict(raw)
        if str(row.get("role") or "").strip().lower() == "assistant":
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
    changed = 0
    out_docs: list[dict[str, Any]] = []
    for raw in list(body.get("docs") or []):
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
        original = str(row.get("text") or "").strip()
        cleaned = original
        if _should_clean_vector_record(row=row, meta=meta):
            cleaned = sanitize_assistant_memory_text(text=original).text
            if cleaned != original:
                changed += 1
        row["text"] = cleaned
        row["embedding"] = embed_text(cleaned, dim=dim)
        row["metadata"] = {**meta, "embed_version": EMBED_VERSION}
        reembedded += 1
        out.append(row)
    body["embed_version"] = EMBED_VERSION
    body["records"] = out
    return body, changed, reembedded


def _drop_fact_docs(rows: list[Any]) -> tuple[list[dict[str, Any]], int]:
    kept: list[dict[str, Any]] = []
    removed = 0
    for raw in list(rows or []):
        row = dict(raw or {})
        if _is_fact_doc(row):
            removed += 1
            continue
        kept.append(row)
    return kept, removed


def _drop_fact_records(rows: list[Any]) -> tuple[list[dict[str, Any]], int]:
    kept: list[dict[str, Any]] = []
    removed = 0
    for raw in list(rows or []):
        row = dict(raw or {})
        if _is_fact_record(row):
            removed += 1
            continue
        kept.append(row)
    return kept, removed


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


def _normalize_profile_payload(payload: dict[str, Any]) -> dict[str, Any]:
    row = _coerce_dict(payload)
    return {
        "profiles": _coerce_dict(row.get("profiles")),
        "versions": _coerce_dict(row.get("versions")),
        "pending_facts": _coerce_dict(row.get("pending_facts")),
        "confirmed_facts": _coerce_dict(row.get("confirmed_facts")),
        "confirmation_context": _coerce_dict(row.get("confirmation_context")),
    }


def _empty_profile_payload() -> dict[str, Any]:
    return {
        "profiles": {},
        "versions": {},
        "pending_facts": {},
        "confirmed_facts": {},
        "confirmation_context": {},
    }


def _backup_files(*, root: Path, backup_dir: Path, names: tuple[str, ...]) -> int:
    backup_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for name in names:
        src = root / str(name)
        if not src.exists():
            continue
        shutil.copy2(src, backup_dir / str(name))
        copied += 1
    return copied


def _restore_from_backup(*, root: Path, backup_dir: Path, names: tuple[str, ...]) -> None:
    for name in names:
        src = backup_dir / str(name)
        if not src.exists():
            continue
        dst = root / str(name)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def _read_json(path: Path, default: dict[str, Any]) -> dict[str, Any]:
    if not path.exists():
        return dict(default)
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return dict(default)
    return payload if isinstance(payload, dict) else dict(default)


def _read_jsonl(path: Path) -> list[Any]:
    if not path.exists():
        return []
    out: list[Any] = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        raw = str(line or "").strip()
        if not raw:
            continue
        try:
            out.append(json.loads(raw))
        except Exception:
            out.append(raw)
    return out


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    _write_text_atomic(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def _write_jsonl_atomic(path: Path, rows: list[Any]) -> None:
    lines: list[str] = []
    for row in list(rows or []):
        lines.append(json.dumps(row, ensure_ascii=False) if isinstance(row, dict) else str(row))
    _write_text_atomic(path, "\n".join(lines).rstrip() + ("\n" if lines else ""))


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
