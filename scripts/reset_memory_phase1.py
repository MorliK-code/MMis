"""
! For startup cleanup need add command --yes

* py scripts/reset_memory_phase1.py --yes
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config.settings import load_config


TARGET_FILES = {
    "short_memory.json",
    "long_memory_docs.json",
    "vector_store.json",
    "events.jsonl",
    "user_profile_store.json",
    "assistant_profile_store.json",
    "brain_state.json",
}

TARGET_DIRS = {
    "brain_state_store",
    "metadata",
}


def _remove_path(path: Path) -> tuple[str, str]:
    if not path.exists():
        return ("skipped", str(path))
    try:
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
    except Exception as exc:
        return ("error", f"{path} ({exc})")
    return ("removed", str(path))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Hard reset memory storage for MMis Phase 1 migration.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="confirm destructive deletion",
    )
    args = parser.parse_args(argv)

    if not args.yes:
        print("Refusing to run without --yes.")
        print("Example: python scripts/reset_memory_phase1.py --yes")
        return 2

    cfg = load_config()
    memory_dir = Path(cfg.memory_dir).expanduser().resolve()
    if not memory_dir.exists() or not memory_dir.is_dir():
        print(f"Memory dir does not exist: {memory_dir}")
        return 1

    results: list[tuple[str, str]] = []
    for name in sorted(TARGET_FILES):
        results.append(_remove_path(memory_dir / name))
    for name in sorted(TARGET_DIRS):
        results.append(_remove_path(memory_dir / name))

    print(f"Memory reset target: {memory_dir}")
    removed = 0
    skipped = 0
    errors = 0
    for status, message in results:
        print(f"[{status}] {message}")
        if status == "removed":
            removed += 1
        elif status == "skipped":
            skipped += 1
        else:
            errors += 1

    print(f"Summary: removed={removed} skipped={skipped} errors={errors}")
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
