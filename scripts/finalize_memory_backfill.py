from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config.settings import load_config
from memory.backfill import run_final_backfill
from memory.memory_manager import MemoryManager


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Final memory backfill pass: compact migration, assistant-noise cleanup, and reindex.",
    )
    parser.add_argument("--memory-dir", type=str, default="", help="Override memory root directory")
    parser.add_argument("--namespace", type=str, default="", help="Only process this namespace")
    parser.add_argument("--storage-profile", type=str, default="compact", help="compact or debug")
    parser.add_argument("--dry-run", action="store_true", help="Preview changes without writing")
    parser.add_argument("--skip-noise-cleanup", action="store_true", help="Do not archive old assistant memory-miss noise")
    parser.add_argument("--skip-reindex", action="store_true", help="Skip final full reindex after rewrite")
    parser.add_argument("--json", action="store_true", help="Print summary as JSON")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = load_config()
    base = (
        Path(str(args.memory_dir or "")).expanduser().resolve()
        if str(args.memory_dir or "").strip()
        else Path(cfg.memory_dir).resolve()
    )

    manager = MemoryManager(root_dir=base)
    try:
        summary = run_final_backfill(
            manager=manager,
            namespace=(str(args.namespace).strip() or None),
            storage_profile=str(args.storage_profile or "compact"),
            dry_run=bool(args.dry_run),
            archive_assistant_noise=not bool(args.skip_noise_cleanup),
            reindex=not bool(args.skip_reindex),
        )
    finally:
        manager.close()

    payload = summary.to_dict()
    if bool(args.json):
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    print("Final memory backfill completed")
    print(f"- dry_run: {payload['dry_run']}")
    print(f"- storage_profile: {payload['storage_profile']}")
    print(f"- namespace: {payload['namespace']}")
    print(f"- scanned: {payload['scanned']}")
    print(f"- rewritten: {payload['rewritten']}")
    print(f"- assistant_noise_archived: {payload['assistant_noise_archived']}")
    if dict(payload.get("reindex_result") or {}):
        print("- reindex:")
        for key, value in dict(payload.get("reindex_result") or {}).items():
            print(f"  - {key}: {value}")
    if dict(payload.get("namespace_stats") or {}):
        print("- namespaces:")
        for ns, stats in sorted(dict(payload.get("namespace_stats") or {}).items()):
            print(
                f"  - {ns}: scanned={int(dict(stats or {}).get('scanned') or 0)} "
                f"rewritten={int(dict(stats or {}).get('rewritten') or 0)} "
                f"assistant_noise_archived={int(dict(stats or {}).get('assistant_noise_archived') or 0)}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
