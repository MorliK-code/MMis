from __future__ import annotations

import subprocess
import sys


BLOCKED_SUFFIXES = (".pyc", ".pyo", ".pyd")


def _staged_paths() -> list[str]:
    cmd = ["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR"]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        sys.stderr.write("Failed to list staged files for pre-commit check.\n")
        if proc.stderr:
            sys.stderr.write(proc.stderr)
        return []
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def _is_blocked(path: str) -> bool:
    normalized = path.replace("\\", "/").lower()
    parts = normalized.split("/")
    if "__pycache__" in parts:
        return True
    return normalized.endswith(BLOCKED_SUFFIXES)


def main() -> int:
    staged = _staged_paths()
    blocked = [path for path in staged if _is_blocked(path)]
    if not blocked:
        return 0

    sys.stderr.write("Commit blocked: staged cache/compiled Python artifacts detected.\n")
    sys.stderr.write("Remove these files from the index:\n")
    for path in blocked:
        sys.stderr.write(f"  - {path}\n")
    sys.stderr.write("\nSuggested fix:\n")
    sys.stderr.write("  git rm --cached -r **/__pycache__\n")
    sys.stderr.write("  git rm --cached '*.pyc' '*.pyo' '*.pyd'\n")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

