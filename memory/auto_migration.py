from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any

from utils.datetime_local import now_local_iso
from utils.logger import get_logger, log_json


LOGGER = get_logger(__name__)
MIGRATION_STATE_FILE = "memory_v2_migration_state.json"
LEGACY_FILES = (
    "events.jsonl",
    "short_memory.json",
    "long_memory_docs.json",
    "vector_store.json",
    "user_profile_store.json",
    "assistant_profile_store.json",
)
LEGACY_DIRS = (
    "brain_state_store",
    "metadata",
    "profiles",
    "summaries",
    "chroma_db",
)


def run_auto_migration(
    *,
    memory_dir: str | Path,
    target_schema_version: int,
    auto_on_start: bool = True,
    facts_scope: str = "user_only",
) -> dict[str, Any]:
    _ = facts_scope
    root = Path(memory_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)

    state_path = root / MIGRATION_STATE_FILE
    state = _read_json(state_path, {})
    current_schema = int(state.get("schema_version") or 0)
    target_schema = max(1, int(target_schema_version or 1))

    if not bool(auto_on_start):
        return {
            "ran": False,
            "success": True,
            "reason": "disabled",
            "schema_version": current_schema,
            "target_schema_version": target_schema,
        }

    legacy_existing_files = [name for name in LEGACY_FILES if (root / name).exists()]
    legacy_existing_dirs = [name for name in LEGACY_DIRS if (root / name).exists() and (root / name).is_dir()]
    legacy_existing = list(legacy_existing_files) + list(legacy_existing_dirs)
    if current_schema >= target_schema and not legacy_existing:
        return {
            "ran": False,
            "success": True,
            "reason": "up_to_date",
            "schema_version": current_schema,
            "target_schema_version": target_schema,
        }

    archive_dir = root / "backups" / f"v1_archive_{time.strftime('%Y%m%d_%H%M%S')}"
    archived: list[str] = []
    if legacy_existing:
        archive_dir.mkdir(parents=True, exist_ok=True)
        for name in legacy_existing:
            src = root / name
            dst = archive_dir / name
            try:
                shutil.move(str(src), str(dst))
                archived.append(name)
            except Exception:
                try:
                    if src.is_dir():
                        shutil.copytree(src, dst, dirs_exist_ok=True)
                        shutil.rmtree(src, ignore_errors=True)
                    else:
                        shutil.copy2(src, dst)
                        src.unlink(missing_ok=True)
                    archived.append(name)
                except Exception:
                    continue

    memory_v2_root = root / "memory_v2"
    memory_v2_root.mkdir(parents=True, exist_ok=True)
    marker = {
        "schema_version": target_schema,
        "created_at": now_local_iso(),
        "mode": "clean_break_v2_only",
        "archived_legacy_files": archived,
        "archive_dir": (str(archive_dir) if archived else ""),
    }
    (memory_v2_root / "schema_marker.json").write_text(
        json.dumps(marker, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    state_payload = {
        "schema_version": target_schema,
        "status": "success",
        "updated_at": now_local_iso(),
        "mode": "clean_break_v2_only",
        "archive_dir": (str(archive_dir) if archived else ""),
        "archived_legacy_files": archived,
    }
    state_path.write_text(json.dumps(state_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    log_json(
        LOGGER,
        "memory_v2_migration",
        memory_dir=str(root),
        schema_version=target_schema,
        archived_files=len(archived),
        archive_dir=(str(archive_dir) if archived else ""),
    )

    return {
        "ran": True,
        "success": True,
        "schema_version": target_schema,
        "archive_dir": (str(archive_dir) if archived else ""),
        "archived_files": archived,
    }


def _read_json(path: Path, default: dict[str, Any]) -> dict[str, Any]:
    if not path.exists():
        return dict(default)
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig") or "{}")
    except Exception:
        return dict(default)
    return payload if isinstance(payload, dict) else dict(default)
