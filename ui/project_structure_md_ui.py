from __future__ import annotations

import tkinter as tk
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, ttk


DEFAULT_IGNORE_DIRS = {
    ".git",
    ".idea",
    ".vscode",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "venv",
    ".venv",
    "node_modules",
    "build",
    "dist",
}


@dataclass
class FileRow:
    rel_path: str
    size_bytes: int
    lines: int | None
    modified_local: str


def now_local_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def safe_stem(name: str) -> str:
    raw = str(name or "").strip().lower()
    out = "".join(ch if ch.isalnum() or ch in {"_", "-", "."} else "_" for ch in raw)
    return out.strip("_") or "root"


def count_text_lines(path: Path) -> int | None:
    try:
        text = path.read_text(encoding="utf-8-sig")
        return len(text.splitlines())
    except Exception:
        return None


def collect_grouped_rows(root: Path, ignore_dirs: set[str]) -> dict[str, list[FileRow]]:
    grouped: dict[str, list[FileRow]] = {"root": []}
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        parts = [p.lower() for p in path.relative_to(root).parts]
        if any(p in ignore_dirs for p in parts):
            continue

        rel = path.relative_to(root).as_posix()
        top = "root" if "/" not in rel else rel.split("/", 1)[0]
        stat = path.stat()
        modified = datetime.fromtimestamp(stat.st_mtime).astimezone().isoformat(timespec="seconds")
        row = FileRow(
            rel_path=rel,
            size_bytes=int(stat.st_size),
            lines=count_text_lines(path),
            modified_local=modified,
        )
        grouped.setdefault(top, []).append(row)

    for key in list(grouped.keys()):
        grouped[key] = sorted(grouped[key], key=lambda x: x.rel_path)
    return grouped


def build_markdown(folder_name: str, rows: list[FileRow], root: Path) -> str:
    total_size = sum(r.size_bytes for r in rows)
    header = [
        f"# {folder_name}.md",
        "",
        f"- project_root: `{root}`",
        f"- generated_at: `{now_local_iso()}`",
        f"- files_count: `{len(rows)}`",
        f"- total_size_bytes: `{total_size}`",
        "",
        "| file | size_bytes | lines | modified_local |",
        "|---|---:|---:|---|",
    ]
    body: list[str] = []
    for row in rows:
        lines = "" if row.lines is None else str(row.lines)
        body.append(f"| `{row.rel_path}` | {row.size_bytes} | {lines} | `{row.modified_local}` |")
    return "\n".join(header + body).strip() + "\n"


def write_folder_markdowns(root: Path, out_dir: Path, ignore_dirs: set[str]) -> list[Path]:
    grouped = collect_grouped_rows(root, ignore_dirs)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for folder, rows in sorted(grouped.items(), key=lambda x: x[0].lower()):
        out_name = f"{safe_stem(folder)}.md"
        out_path = out_dir / out_name
        out_path.write_text(build_markdown(folder, rows, root), encoding="utf-8")
        written.append(out_path)
    return written


class StructureUI(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Project Structure -> folder_name.md")
        self.geometry("920x620")

        self.root_var = tk.StringVar(value=str(Path(__file__).resolve().parents[1]))
        self.out_var = tk.StringVar(value=str(Path(__file__).resolve().parents[1] / "structure_md"))
        self.ignore_var = tk.StringVar(value=",".join(sorted(DEFAULT_IGNORE_DIRS)))
        self.interval_var = tk.StringVar(value="300")
        self.auto_var = tk.BooleanVar(value=True)
        self._after_id: str | None = None

        self._build_ui()
        self._refresh_tree()
        self._schedule_next()

    def _build_ui(self) -> None:
        pad = {"padx": 8, "pady": 6}
        frm = ttk.Frame(self)
        frm.pack(fill="both", expand=True)

        ttk.Label(frm, text="Project root").grid(row=0, column=0, sticky="w", **pad)
        ttk.Entry(frm, textvariable=self.root_var).grid(row=0, column=1, sticky="ew", **pad)
        ttk.Button(frm, text="Browse", command=self._pick_root).grid(row=0, column=2, **pad)

        ttk.Label(frm, text="Output folder").grid(row=1, column=0, sticky="w", **pad)
        ttk.Entry(frm, textvariable=self.out_var).grid(row=1, column=1, sticky="ew", **pad)
        ttk.Button(frm, text="Browse", command=self._pick_out).grid(row=1, column=2, **pad)

        ttk.Label(frm, text="Ignore dirs (comma)").grid(row=2, column=0, sticky="w", **pad)
        ttk.Entry(frm, textvariable=self.ignore_var).grid(row=2, column=1, sticky="ew", **pad)

        ttk.Label(frm, text="Auto-update interval, sec").grid(row=3, column=0, sticky="w", **pad)
        ttk.Entry(frm, textvariable=self.interval_var, width=12).grid(row=3, column=1, sticky="w", **pad)
        ttk.Checkbutton(frm, text="Enable auto-update", variable=self.auto_var).grid(row=3, column=1, sticky="e", **pad)

        btns = ttk.Frame(frm)
        btns.grid(row=4, column=0, columnspan=3, sticky="w", **pad)
        ttk.Button(btns, text="Update Now", command=self._run_update).pack(side="left", padx=4)
        ttk.Button(btns, text="Refresh Tree", command=self._refresh_tree).pack(side="left", padx=4)
        ttk.Button(btns, text="Open Output Folder", command=self._open_output_folder).pack(side="left", padx=4)

        pane = ttk.Panedwindow(frm, orient="horizontal")
        pane.grid(row=5, column=0, columnspan=3, sticky="nsew", **pad)

        left = ttk.Frame(pane)
        right = ttk.Frame(pane)
        pane.add(left, weight=3)
        pane.add(right, weight=2)

        self.tree = ttk.Treeview(
            left,
            columns=("kind", "size", "lines", "modified"),
            show="tree headings",
            height=24,
        )
        self.tree.heading("#0", text="Path")
        self.tree.heading("kind", text="Type")
        self.tree.heading("size", text="Size")
        self.tree.heading("lines", text="Lines")
        self.tree.heading("modified", text="Modified")
        self.tree.column("#0", width=360, stretch=True)
        self.tree.column("kind", width=60, anchor="center")
        self.tree.column("size", width=90, anchor="e")
        self.tree.column("lines", width=70, anchor="e")
        self.tree.column("modified", width=180, anchor="center")
        y_tree = ttk.Scrollbar(left, orient="vertical", command=self.tree.yview)
        x_tree = ttk.Scrollbar(left, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=y_tree.set, xscrollcommand=x_tree.set)
        self.tree.grid(row=0, column=0, sticky="nsew")
        y_tree.grid(row=0, column=1, sticky="ns")
        x_tree.grid(row=1, column=0, sticky="ew")
        left.rowconfigure(0, weight=1)
        left.columnconfigure(0, weight=1)

        self.log = tk.Text(right, wrap="word", height=24)
        y_log = ttk.Scrollbar(right, orient="vertical", command=self.log.yview)
        self.log.configure(yscrollcommand=y_log.set)
        self.log.grid(row=0, column=0, sticky="nsew")
        y_log.grid(row=0, column=1, sticky="ns")
        right.rowconfigure(0, weight=1)
        right.columnconfigure(0, weight=1)

        frm.columnconfigure(1, weight=1)
        frm.rowconfigure(5, weight=1)

    def _pick_root(self) -> None:
        selected = filedialog.askdirectory(initialdir=self.root_var.get() or ".")
        if selected:
            self.root_var.set(selected)
            self._refresh_tree()

    def _pick_out(self) -> None:
        selected = filedialog.askdirectory(initialdir=self.out_var.get() or ".")
        if selected:
            self.out_var.set(selected)

    def _open_output_folder(self) -> None:
        path = Path(self.out_var.get().strip() or ".").expanduser()
        path.mkdir(parents=True, exist_ok=True)
        messagebox.showinfo("Output", f"Output folder:\n{path}")

    def _append_log(self, text: str) -> None:
        self.log.insert("end", f"[{now_local_iso()}] {text}\n")
        self.log.see("end")

    def _parse_ignore_dirs(self) -> set[str]:
        raw = str(self.ignore_var.get() or "")
        rows = [x.strip().lower() for x in raw.split(",") if x.strip()]
        return set(rows)

    def _run_update(self) -> None:
        root = Path(self.root_var.get().strip() or ".").expanduser()
        out_dir = Path(self.out_var.get().strip() or "./structure_md").expanduser()
        if not root.exists() or not root.is_dir():
            messagebox.showerror("Error", f"Project root does not exist:\n{root}")
            return

        ignore_dirs = self._parse_ignore_dirs()
        try:
            written = write_folder_markdowns(root.resolve(), out_dir.resolve(), ignore_dirs)
            self._refresh_tree()
            self._append_log(f"Updated {len(written)} files in {out_dir.resolve()}")
        except Exception as exc:
            self._append_log(f"ERROR: {exc}")

    def _refresh_tree(self) -> None:
        self.tree.delete(*self.tree.get_children(""))
        root = Path(self.root_var.get().strip() or ".").expanduser()
        if not root.exists() or not root.is_dir():
            return
        root = root.resolve()
        ignore_dirs = self._parse_ignore_dirs()
        root_text = root.name or str(root)
        root_node = self.tree.insert("", "end", text=root_text, values=("dir", "", "", ""), open=True)
        self._insert_dir_nodes(root_node, root, ignore_dirs)

    def _insert_dir_nodes(self, parent_node: str, current_dir: Path, ignore_dirs: set[str]) -> None:
        try:
            entries = sorted(
                list(current_dir.iterdir()),
                key=lambda p: (p.is_file(), p.name.lower()),
            )
        except Exception:
            return

        for entry in entries:
            if entry.is_symlink():
                continue
            name_l = entry.name.strip().lower()
            if entry.is_dir():
                if name_l in ignore_dirs:
                    continue
                node = self.tree.insert(parent_node, "end", text=entry.name, values=("dir", "", "", ""))
                self._insert_dir_nodes(node, entry, ignore_dirs)
                continue
            if not entry.is_file():
                continue

            try:
                stat = entry.stat()
            except Exception:
                continue
            modified = datetime.fromtimestamp(stat.st_mtime).astimezone().isoformat(timespec="seconds")
            line_count = count_text_lines(entry)
            line_text = "" if line_count is None else str(line_count)
            self.tree.insert(
                parent_node,
                "end",
                text=entry.name,
                values=("file", str(int(stat.st_size)), line_text, modified),
            )

    def _schedule_next(self) -> None:
        if self._after_id is not None:
            self.after_cancel(self._after_id)
            self._after_id = None

        interval = 300
        try:
            interval = max(10, int(self.interval_var.get().strip() or "300"))
        except Exception:
            interval = 300

        if self.auto_var.get():
            self._after_id = self.after(interval * 1000, self._on_auto_tick)

    def _on_auto_tick(self) -> None:
        if self.auto_var.get():
            self._run_update()
        self._schedule_next()


def main() -> None:
    app = StructureUI()
    app.mainloop()


if __name__ == "__main__":
    main()
