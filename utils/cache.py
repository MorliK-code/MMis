from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from collections import OrderedDict
from pathlib import Path
from threading import RLock
from typing import Any

from config.paths import CACHE_DIR
from config.settings import load_config
from utils.logger import get_logger


LOGGER = get_logger(__name__)


class DiskTTLCache:
    """Small disk-backed TTL cache with in-process hot cache."""

    def __init__(
        self,
        *,
        namespace: str,
        root: str | Path | None = None,
        default_ttl_s: int = 900,
        max_memory_entries: int = 512,
        enabled: bool = True,
    ):
        self.namespace = _safe_name(namespace)
        self.default_ttl_s = max(1, int(default_ttl_s))
        self.max_memory_entries = max(8, int(max_memory_entries))
        self.enabled = bool(enabled)
        self._lock = RLock()
        self._mem: OrderedDict[str, tuple[float, Any]] = OrderedDict()
        self.root = _resolve_cache_root(root=root) / self.namespace
        self.root.mkdir(parents=True, exist_ok=True)

    def get(self, key: str) -> Any | None:
        if not self.enabled:
            return None
        cache_key = str(key or "").strip()
        if not cache_key:
            return None
        now = time.time()

        with self._lock:
            mem_hit = self._mem.get(cache_key)
            if mem_hit is not None:
                expires_at, value = mem_hit
                if expires_at > now:
                    self._mem.move_to_end(cache_key)
                    return value
                self._mem.pop(cache_key, None)

        path = self._entry_path(cache_key)
        if not path.exists():
            return None

        try:
            payload = json.loads(path.read_text(encoding="utf-8-sig"))
        except Exception:
            _safe_unlink(path)
            return None
        if not isinstance(payload, dict):
            _safe_unlink(path)
            return None

        expires_at = float(payload.get("expires_at") or 0.0)
        value = payload.get("value")
        if expires_at <= now:
            _safe_unlink(path)
            return None

        with self._lock:
            self._mem[cache_key] = (expires_at, value)
            self._mem.move_to_end(cache_key)
            self._trim_memory()
        return value

    def set(self, key: str, value: Any, *, ttl_s: int | None = None) -> None:
        if not self.enabled:
            return
        cache_key = str(key or "").strip()
        if not cache_key:
            return

        now = time.time()
        ttl = max(1, int(ttl_s if ttl_s is not None else self.default_ttl_s))
        expires_at = now + ttl
        payload = {
            "key": cache_key,
            "created_at": now,
            "expires_at": expires_at,
            "value": value,
        }

        with self._lock:
            self._mem[cache_key] = (expires_at, value)
            self._mem.move_to_end(cache_key)
            self._trim_memory()

        path = self._entry_path(cache_key)
        tmp = None
        try:
            fd, tmp_raw = tempfile.mkstemp(prefix="cache_", suffix=".tmp", dir=str(self.root))
            os.close(fd)
            tmp = Path(tmp_raw)
            tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            tmp.replace(path)
        except Exception as exc:
            LOGGER.debug("disk cache write failed ns=%s key=%s err=%s", self.namespace, cache_key, exc)
            if tmp is not None:
                _safe_unlink(tmp)

    def delete(self, key: str) -> None:
        cache_key = str(key or "").strip()
        if not cache_key:
            return
        with self._lock:
            self._mem.pop(cache_key, None)
        _safe_unlink(self._entry_path(cache_key))

    def purge_expired(self, *, max_files: int = 5000) -> int:
        if not self.enabled:
            return 0
        now = time.time()
        removed = 0
        scanned = 0
        for path in self.root.glob("*.json"):
            scanned += 1
            if scanned > max(1, int(max_files)):
                break
            try:
                payload = json.loads(path.read_text(encoding="utf-8-sig"))
            except Exception:
                _safe_unlink(path)
                removed += 1
                continue
            expires_at = 0.0
            if isinstance(payload, dict):
                expires_at = float(payload.get("expires_at") or 0.0)
            if expires_at <= now:
                _safe_unlink(path)
                removed += 1
        return removed

    def _entry_path(self, key: str) -> Path:
        digest = hashlib.sha1(str(key).encode("utf-8", errors="ignore")).hexdigest()
        return self.root / f"{digest}.json"

    def _trim_memory(self) -> None:
        while len(self._mem) > self.max_memory_entries:
            self._mem.popitem(last=False)


def _safe_name(value: str) -> str:
    raw = str(value or "").strip().lower()
    out = "".join(ch for ch in raw if ch.isalnum() or ch in {"_", "-", "."})
    return out or "default"


def _resolve_cache_root(*, root: str | Path | None) -> Path:
    if root is not None:
        return Path(root).expanduser().resolve()
    try:
        cfg = load_config()
        return Path(cfg.cache_dir).expanduser().resolve()
    except Exception:
        return CACHE_DIR.expanduser().resolve()


def _safe_unlink(path: Path) -> None:
    try:
        if path.exists():
            path.unlink()
    except Exception:
        pass

