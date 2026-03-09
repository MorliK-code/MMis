from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config.settings import load_config
from memory.memory_manager import MemoryManager


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reindex Memory V2 embeddings.")
    parser.add_argument("--memory-dir", type=str, default="", help="Override memory root directory")
    parser.add_argument("--namespace", type=str, default="", help="Reindex only this namespace")
    parser.add_argument("--incremental", action="store_true", help="Incremental reindex instead of full rebuild")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = load_config()
    base = Path(str(args.memory_dir or "")).expanduser().resolve() if str(args.memory_dir or "").strip() else Path(cfg.memory_dir).resolve()

    manager = MemoryManager(root_dir=base)
    try:
        result = manager.reindex_embeddings(
            namespace=(str(args.namespace).strip() or None),
            incremental=bool(args.incremental),
        )
    finally:
        manager.close()

    print("Reindex completed")
    for key, value in dict(result or {}).items():
        print(f"- {key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
