from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PromptDocument:
    rel_path: str
    source_path: str
    text: str
    id: str
    version: str
    role: str
    tags: list[str] = field(default_factory=list)
    min_ctx: int = 0
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class _CacheEntry:
    mtime_ns: int
    doc: PromptDocument


class PromptLoader:
    """Legacy prompt loader. TXT prompt files are blocked in runtime (JSON specs only)."""

    def __init__(self, root: str | Path | None = None):
        base = Path(root).expanduser() if root is not None else (Path(__file__).resolve().parent.parent / "prompts")
        self.root = base.resolve()
        self._cache: dict[str, _CacheEntry] = {}

    def load(self, rel_path: str, *, use_cache: bool = True) -> str:
        return self.load_document(rel_path, use_cache=use_cache).text

    def load_document(self, rel_path: str, *, use_cache: bool = True, hot_reload: bool = True) -> PromptDocument:
        key = _normalize_rel_path(rel_path)
        if key.lower().endswith(".txt"):
            raise ValueError(f"TXT prompt files are disabled in runtime: {key}")
        path = self._resolve_path(key)
        mtime_ns = int(path.stat().st_mtime_ns)

        if use_cache and key in self._cache:
            cached = self._cache[key]
            if not hot_reload or int(cached.mtime_ns) == mtime_ns:
                return cached.doc

        raw = path.read_text(encoding="utf-8-sig")
        meta, body = _parse_frontmatter(raw)
        doc = PromptDocument(
            rel_path=key,
            source_path=str(path),
            text=str(body or "").strip(),
            id=str(meta.get("id") or path.stem),
            version=str(meta.get("version") or "0.0.0"),
            role=str(meta.get("role") or "system"),
            tags=_coerce_tags(meta.get("tags")),
            min_ctx=_coerce_int(meta.get("min_ctx"), default=0, minimum=0),
            meta=dict(meta),
        )
        if use_cache:
            self._cache[key] = _CacheEntry(mtime_ns=mtime_ns, doc=doc)
        return doc

    def invalidate(self, rel_path: str | None = None) -> None:
        if rel_path is None:
            self._cache.clear()
            return
        key = _normalize_rel_path(rel_path)
        self._cache.pop(key, None)

    def exists(self, rel_path: str) -> bool:
        try:
            path = self._resolve_path(_normalize_rel_path(rel_path))
        except Exception:
            return False
        return path.exists() and path.is_file()

    def _resolve_path(self, key: str) -> Path:
        path = (self.root / key).resolve()
        try:
            path.relative_to(self.root)
        except Exception as exc:
            raise ValueError(f"Prompt path escapes root: {path}") from exc
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f"Prompt file not found: {path}")
        return path


def _normalize_rel_path(rel_path: str) -> str:
    key = str(rel_path or "").replace("\\", "/").strip().strip("/")
    if not key:
        raise ValueError("rel_path is empty")
    return key


def _parse_frontmatter(raw: str) -> tuple[dict[str, Any], str]:
    src = str(raw or "")
    lines = src.splitlines()
    if len(lines) < 3:
        return {}, src
    if lines[0].strip() != "---":
        return {}, src

    idx = 1
    meta_lines: list[str] = []
    while idx < len(lines):
        line = lines[idx]
        if line.strip() == "---":
            idx += 1
            break
        meta_lines.append(line)
        idx += 1
    else:
        return {}, src

    body = "\n".join(lines[idx:]).strip()
    meta = _parse_meta_lines(meta_lines)
    return meta, body


def _parse_meta_lines(lines: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for raw in list(lines or []):
        line = str(raw or "").strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        name = str(key or "").strip().lower()
        if not name:
            continue
        out[name] = _parse_meta_value(value.strip())
    return out


def _parse_meta_value(value: str):
    src = str(value or "").strip()
    if not src:
        return ""
    if src.startswith("[") and src.endswith("]"):
        inner = src[1:-1].strip()
        if not inner:
            return []
        items = []
        for part in inner.split(","):
            item = part.strip().strip('"').strip("'")
            if item:
                items.append(item)
        return items
    low = src.lower()
    if low in {"true", "false"}:
        return low == "true"
    try:
        if "." in src:
            return float(src)
        return int(src)
    except Exception:
        return src.strip('"').strip("'")


def _coerce_tags(value) -> list[str]:
    out: list[str] = []
    if isinstance(value, list):
        rows = value
    elif value is None:
        rows = []
    else:
        rows = [value]
    seen: set[str] = set()
    for row in rows:
        tag = str(row or "").strip().lower()
        if not tag or tag in seen:
            continue
        seen.add(tag)
        out.append(tag)
    return out


def _coerce_int(value, *, default: int, minimum: int) -> int:
    try:
        out = int(value)
    except Exception:
        out = int(default)
    return max(int(minimum), out)
