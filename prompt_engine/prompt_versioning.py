from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

from prompt_engine.prompt_loader import PromptLoader
from prompt_engine.prompt_registry import PromptRegistry


@dataclass(frozen=True)
class PromptVersion:
    key: str
    rel_path: str
    prompt_id: str
    version: str
    sha1: str
    ts: float


class PromptVersioning:
    """Tracks prompt file hashes and stores snapshots to data/cache."""

    def __init__(
        self,
        registry: PromptRegistry | None = None,
        loader: PromptLoader | None = None,
        store_path: str | Path | None = None,
    ):
        self.loader = loader or PromptLoader()
        self.registry = registry or PromptRegistry(loader=self.loader)
        default_store = Path(__file__).resolve().parent.parent / "data" / "cache" / "prompt_versions.json"
        self.store_path = Path(store_path).expanduser() if store_path is not None else default_store
        self.store_path.parent.mkdir(parents=True, exist_ok=True)

    def snapshot(self) -> list[PromptVersion]:
        out: list[PromptVersion] = []
        now = time.time()
        for key in self.registry.keys():
            entry = self.registry.get_entry(key)
            prompt = self.registry.get_prompt(key, use_cache=False, hot_reload=True)
            text = str(prompt.text or "")
            digest = hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()
            out.append(
                PromptVersion(
                    key=entry.key,
                    rel_path=entry.rel_path,
                    prompt_id=str(prompt.id or ""),
                    version=str(prompt.version or "0.0.0"),
                    sha1=digest,
                    ts=now,
                )
            )
        return out

    def save_snapshot(self) -> list[PromptVersion]:
        rows = self.snapshot()
        payload = [
            {
                "key": x.key,
                "rel_path": x.rel_path,
                "prompt_id": x.prompt_id,
                "version": x.version,
                "sha1": x.sha1,
                "ts": x.ts,
            }
            for x in rows
        ]
        self.store_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return rows

    def load_snapshot(self) -> dict[str, PromptVersion]:
        if not self.store_path.exists():
            return {}
        try:
            payload = json.loads(self.store_path.read_text(encoding="utf-8-sig"))
        except Exception:
            return {}
        out: dict[str, PromptVersion] = {}
        if not isinstance(payload, list):
            return out
        for row in payload:
            if not isinstance(row, dict):
                continue
            key = str(row.get("key") or "").strip()
            rel = str(row.get("rel_path") or "").strip()
            prompt_id = str(row.get("prompt_id") or "").strip()
            version = str(row.get("version") or "0.0.0").strip() or "0.0.0"
            sha1 = str(row.get("sha1") or "").strip()
            ts = float(row.get("ts") or 0.0)
            if not key or not rel or not sha1:
                continue
            out[key] = PromptVersion(
                key=key,
                rel_path=rel,
                prompt_id=prompt_id,
                version=version,
                sha1=sha1,
                ts=ts,
            )
        return out

    def diff(self) -> dict[str, list[str]]:
        old = self.load_snapshot()
        new = {x.key: x for x in self.snapshot()}
        added = sorted([k for k in new if k not in old])
        removed = sorted([k for k in old if k not in new])
        changed = sorted([k for k in new if k in old and new[k].sha1 != old[k].sha1])
        return {"added": added, "removed": removed, "changed": changed}
