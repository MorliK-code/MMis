from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TARGET_NAME = "studio_generator.py"
MARKERS = ("ф", "т", "я", "ь", "ш", "щ", "ч", "ю", "э", "а", "о", "е", "и")


def is_mojibake_line(text: str) -> bool:
    hits = sum(text.count(marker) for marker in MARKERS)
    return hits >= 2


def find_target(root: Path, target_name: str) -> Path:
    matches = list(root.rglob(target_name))
    if not matches:
        raise FileNotFoundError(f"Target file not found: {target_name} under {root}")
    if len(matches) > 1:
        variants = "\n".join(f" - {p}" for p in matches)
        raise RuntimeError(f"Multiple files found for '{target_name}':\n{variants}")
    return matches[0]


def fix_content(text: str) -> tuple[str, int]:
    lines = text.splitlines(keepends=True)
    out: list[str] = []
    changed = 0
    for line in lines:
        if not is_mojibake_line(line):
            out.append(line)
            continue
        try:
            fixed_line = line.encode("cp1251").decode("utf-8")
        except UnicodeEncodeError:
            fixed_line = line.encode("cp1251", errors="ignore").decode("utf-8", errors="replace")
        except UnicodeDecodeError:
            fixed_line = line
        if fixed_line != line:
            changed += 1
        out.append(fixed_line)
    return "".join(out), changed


def write_zip(zip_path: Path, rel: Path, payload: bytes) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(rel.as_posix(), payload)


def main() -> int:
    parser = argparse.ArgumentParser(description="Fix one mojibake-affected file with backups")
    parser.add_argument("--root", default=str(ROOT), help="Project root")
    parser.add_argument("--target", default=DEFAULT_TARGET_NAME, help="Target filename to fix")
    parser.add_argument("--dry-run", action="store_true", help="Do not modify file, only create backup + preview")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    target_path = find_target(root, str(args.target))
    rel = target_path.resolve().relative_to(root)

    original_bytes = target_path.read_bytes()
    text = target_path.read_text(encoding="utf-8-sig", errors="strict")
    new_text, changed_lines = fix_content(text)
    new_bytes = new_text.encode("utf-8")

    batch_dir = Path(__file__).resolve().parent / "backups" / datetime.now().strftime("%Y%m%d_%H%M%S")
    before_zip = batch_dir / "will_change.zip"
    write_zip(before_zip, rel, original_bytes)

    print(f"Target: {target_path}")
    print(f"Changed lines: {changed_lines}")
    print(f"[ZIP] Backup before: {before_zip}")

    if original_bytes == new_bytes:
        print("[OK] Nothing to change.")
        return 0

    if args.dry_run:
        print("[DRY] File was not modified.")
        return 0

    target_path.write_bytes(new_bytes)
    after_zip = batch_dir / "after.zip"
    write_zip(after_zip, rel, new_bytes)
    print("[OK] Changes applied.")
    print(f"[ZIP] Backup after: {after_zip}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
