from __future__ import annotations

import json
import time
from pathlib import Path
from threading import RLock
from typing import Any

from config.settings import load_config
from memory.fact_extractor import Fact
from utils.datetime_local import now_local_iso, now_local_ts, parse_time_to_epoch


SENSITIVE_KEYS = {"age", "birth_year", "birthday", "location"}


class _BaseProfileStore:
    def __init__(self, *, profile_kind: str, path: str | Path | None = None):
        cfg = load_config()
        default_path = cfg.memory_dir / f"{profile_kind}_profile_store.json"
        self.path = Path(path).expanduser() if path is not None else default_path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.profile_kind = str(profile_kind)
        self._lock = RLock()
        self._data: dict[str, Any] = {"profiles": {}, "versions": {}}
        self.load()

    def get(self, key: str, profile_id: str = "default"):
        pid = _profile_id(profile_id)
        name = _key(key)
        with self._lock:
            return _deepcopy(self._data.get("profiles", {}).get(pid, {}).get(name))

    def set(
        self,
        key: str,
        value,
        confidence: float = 0.7,
        source_event_id: str = "",
        *,
        profile_id: str = "default",
        source: str = "fact",
        op: str = "set",
    ) -> dict[str, Any]:
        pid = _profile_id(profile_id)
        name = _key(key)
        conf = _clamp01(confidence)
        now = time.time()

        with self._lock:
            profiles = self._data.setdefault("profiles", {})
            versions = self._data.setdefault("versions", {})
            profile_row = profiles.setdefault(pid, {})
            profile_versions = versions.setdefault(pid, {})
            entry = dict(profile_row.get(name) or {})
            old_value = entry.get("value") if entry else None
            old_conf = float(entry.get("confidence") or 0.0)
            old_ts = parse_time_to_epoch(entry.get("updated_at"), 0.0)

            resolution = self._resolve_conflict(
                key=name,
                old_value=old_value,
                old_conf=old_conf,
                old_ts=old_ts,
                new_value=value,
                new_conf=conf,
                new_ts=now,
                source=source,
            )
            status = resolution.get("status", "applied")
            applied = bool(resolution.get("applied", True))

            if applied:
                entry = {
                    "value": value,
                    "confidence": conf,
                    "source_event_id": str(source_event_id or ""),
                    "source": str(source or ""),
                    "updated_at": now_local_iso(),
                    "needs_confirmation": bool(resolution.get("needs_confirmation", False)),
                    "pending": list(entry.get("pending") or []),
                }
                profile_row[name] = entry
            else:
                pending = list(entry.get("pending") or [])
                pending.append(
                    {
                        "value": value,
                        "confidence": conf,
                        "source_event_id": str(source_event_id or ""),
                        "source": str(source or ""),
                        "ts": now_local_ts(),
                    }
                )
                entry["pending"] = pending[-6:]
                entry["needs_confirmation"] = True
                profile_row[name] = entry

            history_row = profile_versions.setdefault(name, [])
            history_row.append(
                {
                    "ts": now_local_ts(),
                    "op": op,
                    "old": old_value,
                    "new": value,
                    "confidence": conf,
                    "source_event_id": str(source_event_id or ""),
                    "source": str(source or ""),
                    "status": status,
                }
            )
            self.save()
            result = dict(profile_row[name])
            result["status"] = status
            return result

    def update_fact(self, fact: Fact, profile_id: str = "default") -> dict[str, Any]:
        pid = _profile_id(profile_id)
        if not isinstance(fact, Fact):
            raise TypeError("update_fact expects Fact")
        key = _key(fact.key)
        op = str(fact.op or "add").strip().lower()

        if op == "remove":
            return self.set(
                key=key,
                value=None,
                confidence=fact.confidence,
                source_event_id=fact.source_event_id,
                profile_id=pid,
                source="fact_remove",
                op="remove",
            )
        if op in {"update", "add", "set"}:
            return self.set(
                key=key,
                value=fact.value,
                confidence=fact.confidence,
                source_event_id=fact.source_event_id,
                profile_id=pid,
                source=f"fact_{op}",
                op=op,
            )
        return self.set(
            key=key,
            value=fact.value,
            confidence=fact.confidence,
            source_event_id=fact.source_event_id,
            profile_id=pid,
            source="fact_unknown",
            op=op,
        )

    def history(self, key: str, profile_id: str = "default") -> list[dict[str, Any]]:
        pid = _profile_id(profile_id)
        name = _key(key)
        with self._lock:
            rows = list(self._data.get("versions", {}).get(pid, {}).get(name, []))
            return [_deepcopy(x) for x in rows]

    def load(self) -> None:
        if not self.path.exists():
            return
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8-sig"))
        except Exception:
            return
        if not isinstance(payload, dict):
            return
        with self._lock:
            self._data = {
                "profiles": dict(payload.get("profiles") or {}),
                "versions": dict(payload.get("versions") or {}),
            }

    def save(self) -> None:
        with self._lock:
            payload = _deepcopy(self._data)
        self.path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def _resolve_conflict(
        *,
        key: str,
        old_value,
        old_conf: float,
        old_ts: float,
        new_value,
        new_conf: float,
        new_ts: float,
        source: str,
    ) -> dict[str, Any]:
        if old_value is None:
            return {"applied": True, "status": "new", "needs_confirmation": _needs_confirmation(key, new_conf)}
        if new_value == old_value:
            return {"applied": True, "status": "same", "needs_confirmation": _needs_confirmation(key, max(old_conf, new_conf))}

        source_bonus = 0.12 if source in {"fact_update", "fact_add"} else 0.04
        recency_bonus = 0.08 if new_ts >= old_ts else 0.0
        new_score = float(new_conf) + source_bonus + recency_bonus
        old_score = float(old_conf) + (0.04 if old_ts > 0 else 0.0)

        if new_score >= old_score + 0.08:
            return {"applied": True, "status": "updated", "needs_confirmation": _needs_confirmation(key, new_conf)}
        return {"applied": False, "status": "conflict", "needs_confirmation": True}


class UserProfileStore(_BaseProfileStore):
    def __init__(self, path: str | Path | None = None):
        super().__init__(profile_kind="user", path=path)


class AssistantProfileStore(_BaseProfileStore):
    def __init__(self, path: str | Path | None = None):
        super().__init__(profile_kind="assistant", path=path)


class ProfileStore:
    """Backward-compatible generic profile store facade."""

    def __init__(self):
        self._user = UserProfileStore()

    def get(self, user_id: str) -> dict:
        pid = _profile_id(user_id)
        with self._user._lock:
            return _deepcopy(self._user._data.get("profiles", {}).get(pid, {}))

    def set(self, user_id: str, profile: dict) -> None:
        pid = _profile_id(user_id)
        for key, value in dict(profile or {}).items():
            self._user.set(
                key=key,
                value=value,
                confidence=0.6,
                source_event_id="",
                profile_id=pid,
                source="profile_set",
                op="set",
            )


def _needs_confirmation(key: str, confidence: float) -> bool:
    return (_key(key) in SENSITIVE_KEYS) and float(confidence) < 0.75


def _profile_id(value: str) -> str:
    raw = str(value or "default").strip()
    return raw if raw else "default"


def _key(value: str) -> str:
    raw = str(value or "").strip().lower().replace(" ", "_")
    return raw


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _deepcopy(value):
    try:
        return json.loads(json.dumps(value, ensure_ascii=False))
    except Exception:
        return value
