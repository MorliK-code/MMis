from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import zipfile


TEXT_EXTS = {
    ".py",
    ".txt",
    ".md",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".ini",
    ".cfg",
    ".env",
    ".bat",
    ".ps1",
    ".sh",
    ".sql",
    ".csv",
    ".ts",
    ".tsx",
    ".js",
    ".jsx",
    ".html",
    ".css",
    ".xml",
}

SKIP_DIRS = {
    ".git",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".idea",
    ".vscode",
    "node_modules",
    "dist",
    "build",
    "out",
}

MOJI_MARKERS = (
    "С„",
    "С‚",
    "СЏ",
    "СЊ",
    "С€",
    "С‰",
    "С‡",
    "СЋ",
    "СЌ",
    "Р°",
    "Рѕ",
    "Рµ",
    "Рё",
    "Рє",
    "Р»",
    "РЅ",
    "Рї",
    "СЂ",
    "СЃ",
    "Сѓ",
)


@dataclass
class PlannedChange:
    path: Path
    rel: Path
    reason: str
    original_bytes: bytes
    new_bytes: bytes


def should_process(path: Path) -> bool:
    if path.is_dir():
        return False
    if should_skip_path(path):
        return False
    ext = path.suffix.lower()
    if ext in TEXT_EXTS:
        return True
    if path.name.lower() in {"dockerfile", "makefile"}:
        return True
    return False


def is_virtualenv_root(path: Path) -> bool:
    return (path / "pyvenv.cfg").exists() or (
        (path / "Scripts").is_dir() and (path / "Lib" / "site-packages").is_dir()
    )


def should_skip_path(path: Path) -> bool:
    if any(part in SKIP_DIRS for part in path.parts):
        return True
    return any(is_virtualenv_root(parent) for parent in path.parents)


def find_project_root(start: Path) -> Path:
    cur = start.resolve()
    for p in [cur, *cur.parents]:
        if (p / ".git").exists():
            return p
        if (p / "pyproject.toml").exists():
            return p
        if (p / "requirements.txt").exists():
            return p
    return start.resolve().parents[2]


def try_read_text(path: Path) -> tuple[str, str] | None:
    for enc in ("utf-8-sig", "utf-8", "cp1251", "cp866", "latin1"):
        try:
            return path.read_text(encoding=enc), enc
        except UnicodeDecodeError:
            continue
        except Exception:
            return None
    return None


def is_mojibake_line(line: str) -> bool:
    hits = sum(line.count(marker) for marker in MOJI_MARKERS)
    return hits >= 2


def fix_mojibake_text(text: str) -> tuple[str, int]:
    lines = text.splitlines(keepends=True)
    out: list[str] = []
    changed = 0

    for line in lines:
        if is_mojibake_line(line):
            try:
                fixed = line.encode("cp1251").decode("utf-8")
                out.append(fixed)
                changed += 1
                continue
            except UnicodeEncodeError:
                fixed = line.encode("cp1251", errors="ignore").decode("utf-8", errors="replace")
                out.append(fixed)
                changed += 1
                continue
            except UnicodeDecodeError:
                out.append(line)
                continue

        out.append(line)

    return "".join(out), changed


def write_zip(zip_path: Path, items: list[tuple[Path, bytes]]) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for rel, blob in items:
            archive.writestr(rel.as_posix(), blob)


def write_manifest(path: Path, planned: list[PlannedChange]) -> None:
    lines = [f"files={len(planned)}"]
    lines.extend(f"{item.rel.as_posix()} | {item.reason}" for item in planned)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plan_changes(project_root: Path) -> list[PlannedChange]:
    planned: list[PlannedChange] = []

    for path in project_root.rglob("*"):
        if not should_process(path):
            continue

        read = try_read_text(path)
        if not read:
            continue
        text, enc = read

        new_text, changed_lines = fix_mojibake_text(text)
        if changed_lines <= 0:
            continue

        new_bytes = new_text.encode("utf-8")
        orig_bytes = path.read_bytes()
        if new_bytes == orig_bytes:
            continue

        rel = path.resolve().relative_to(project_root.resolve())
        planned.append(
            PlannedChange(
                path=path,
                rel=rel,
                reason=f"mojibake_lines:{changed_lines} (read:{enc})",
                original_bytes=orig_bytes,
                new_bytes=new_bytes,
            )
        )

    return planned


def apply_changes(planned: list[PlannedChange]) -> None:
    for item in planned:
        item.path.write_bytes(item.new_bytes)


def _print_plan(planned: list[PlannedChange], limit: int = 200) -> None:
    for i, change in enumerate(planned[:limit], 1):
        print(f" {i:03d}. {change.rel} ({change.reason})")
    if len(planned) > limit:
        print(f" ... and {len(planned) - limit} more files")


def _print_samples(planned: list[PlannedChange], limit: int = 15) -> None:
    print("\n--- samples (first 15 files) ---")
    for change in planned[:limit]:
        try:
            text = change.original_bytes.decode("utf-8", errors="ignore")
        except Exception:
            continue
        for line in text.splitlines():
            if is_mojibake_line(line):
                print(f"{change.rel}: {line[:160]}")
                break


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Fix mojibake (UTF-8 text opened as CP1251). "
            "Default mode applies fixes immediately and creates backups."
        )
    )
    parser.add_argument("--root", default=None, help="Project root (auto-detected if omitted)")
    parser.add_argument("--dry-run", action="store_true", help="Only scan and create will_change.zip")
    parser.add_argument("--show-samples", action="store_true", help="Show sample mojibake lines")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    project_root = Path(args.root).resolve() if args.root else find_project_root(script_dir)

    planned = plan_changes(project_root)
    if not planned:
        print(f"[OK] No changes required. Project root: {project_root}")
        return 0

    backups_dir = script_dir / "backups"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_dir = backups_dir / ts

    will_zip = batch_dir / "will_change.zip"
    write_zip(will_zip, [(c.rel, c.original_bytes) for c in planned])
    write_manifest(batch_dir / "manifest.txt", planned)

    apply_mode = not bool(args.dry_run)

    print(f"Project root: {project_root}")
    print(f"Files to change: {len(planned)}")
    print(f"Mode: {'APPLY (default)' if apply_mode else 'DRY-RUN'}")
    print(f"[ZIP] Backup before changes: {will_zip}")
    print(f"[META] Plan manifest: {batch_dir / 'manifest.txt'}")
    _print_plan(planned)

    if args.show_samples:
        _print_samples(planned)

    if not apply_mode:
        print("[DRY] No files changed.")
        return 0

    apply_changes(planned)

    after_zip = batch_dir / "after.zip"
    write_zip(after_zip, [(c.rel, c.new_bytes) for c in planned])

    print("[OK] Changes applied.")
    print(f"[ZIP] Snapshot after changes: {after_zip}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
