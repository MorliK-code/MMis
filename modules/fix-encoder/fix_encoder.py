from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import zipfile

# Что считаем "текстовыми" файлами проекта
TEXT_EXTS = {
    ".py", ".txt", ".md", ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg",
    ".env", ".bat", ".ps1", ".sh", ".sql", ".csv", ".ts", ".tsx", ".js", ".jsx",
    ".html", ".css", ".xml"
}

# Какие папки пропускаем (можешь дополнять)
SKIP_DIRS = {
    ".git", ".venv", "venv", "__pycache__", ".mypy_cache", ".pytest_cache",
    ".ruff_cache", ".idea", ".vscode", "node_modules", "dist", "build", "out",
}

# Сигнатуры "крякозябр" твоего типа: "С„Р°Р·Р°", "С‚РµРє..." и т.п.
# (UTF-8 русский, открыли как CP1251 и сохранили в UTF-8)
MOJI_MARKERS = (
    "ф", "т", "я", "ь", "ш", "щ", "ч", "ю", "э",
    "а", "о", "е", "и", "к", "л", "н", "п", "р", "с", "у",
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
    if any(part in SKIP_DIRS for part in path.parts):
        return False
    ext = path.suffix.lower()
    if ext in TEXT_EXTS:
        return True
    if path.name.lower() in {"dockerfile", "makefile"}:
        return True
    return False


def find_project_root(start: Path) -> Path:
    cur = start.resolve()
    for p in [cur, *cur.parents]:
        if (p / ".git").exists():
            return p
        if (p / "pyproject.toml").exists():
            return p
        if (p / "requirements.txt").exists():
            return p
    # fallback: modules/fix-encoder -> project root
    return start.resolve().parents[2]


def try_read_text(path: Path) -> tuple[str, str] | None:
    """
    Читаем текст максимально безопасно:
    - сначала utf-8-sig (срежет BOM)
    - потом utf-8
    - потом cp1251/cp866/latin1 (на всякий)
    """
    for enc in ("utf-8-sig", "utf-8", "cp1251", "cp866", "latin1"):
        try:
            return path.read_text(encoding=enc), enc
        except UnicodeDecodeError:
            continue
        except Exception:
            return None
    return None


def is_mojibake_line(line: str) -> bool:
    # Мягкий, но рабочий критерий: в строке есть 2+ маркера
    hits = sum(line.count(m) for m in MOJI_MARKERS)
    return hits >= 2


def fix_mojibake_text(text: str) -> tuple[str, int]:
    """
    Исправляем ПОСТРОЧНО:
    - Только строки, похожие на кракозябры
    - Основной метод: encode cp1251 -> decode utf-8
    - Если попадается мусорный символ (как \x98) — игнорируем его в ЭТОЙ строке
    Возвращает (new_text, changed_lines_count)
    """
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
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for rel, blob in items:
            z.writestr(rel.as_posix(), blob)


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

        # сохраняем как utf-8 (без BOM)
        new_bytes = new_text.encode("utf-8")
        orig_bytes = path.read_bytes()
        if new_bytes == orig_bytes:
            continue

        rel = path.resolve().relative_to(project_root.resolve())
        planned.append(PlannedChange(
            path=path,
            rel=rel,
            reason=f"mojibake_lines:{changed_lines} (read:{enc})",
            original_bytes=orig_bytes,
            new_bytes=new_bytes
        ))

    return planned


def apply_changes(planned: list[PlannedChange]) -> None:
    for c in planned:
        c.path.write_bytes(c.new_bytes)


def main() -> int:
    ap = argparse.ArgumentParser(description="Fix mojibake (cp1251-opened utf8) across project with backups zip.")
    ap.add_argument("--root", default=None, help="Корень проекта (если не задан — авто)")
    ap.add_argument("--dry-run", action="store_true", help="Только анализ + will_change.zip")
    ap.add_argument("--show-samples", action="store_true", help="Показать 1 пример строки для первых 15 файлов")
    args = ap.parse_args()

    script_dir = Path(__file__).resolve().parent
    backups_dir = script_dir / "backups"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    project_root = Path(args.root).resolve() if args.root else find_project_root(script_dir)

    planned = plan_changes(project_root)
    if not planned:
        print(f"[OK] Изменений не требуется. Корень проекта: {project_root}")
        return 0

    batch_dir = backups_dir / ts
    will_zip = batch_dir / "will_change.zip"
    write_zip(will_zip, [(c.rel, c.original_bytes) for c in planned])

    print(f"Корень проекта: {project_root}")
    print(f"Файлов будет изменено: {len(planned)}")
    print(f"[ZIP] Оригиналы (ТОЛЬКО изменяемые): {will_zip}")

    for i, c in enumerate(planned[:200], 1):
        print(f" {i:03d}. {c.rel} ({c.reason})")
    if len(planned) > 200:
        print(f" ... и ещё {len(planned) - 200} файлов")

    if args.show_samples:
        print("\n--- samples (первые 15) ---")
        for c in planned[:15]:
            try:
                t = c.original_bytes.decode("utf-8", errors="ignore")
                # покажем первую строку, где есть маркеры
                for ln in t.splitlines():
                    if is_mojibake_line(ln):
                        print(f"{c.rel}: {ln[:160]}")
                        break
            except Exception:
                pass

    if args.dry_run:
        print("[DRY] Ничего не изменено (только анализ + will_change.zip).")
        return 0

    apply_changes(planned)

    after_zip = batch_dir / "after.zip"
    write_zip(after_zip, [(c.rel, c.new_bytes) for c in planned])

    print("[OK] Изменения применены.")
    print(f"[ZIP] Исправленные версии этих файлов: {after_zip}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())