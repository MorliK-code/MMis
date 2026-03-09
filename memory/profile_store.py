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
        self._data: dict[str, Any] = {
            "profiles": {},
            "versions": {},
            "pending_facts": {},
            "confirmed_facts": {},
            "confirmation_context": {},
        }
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
        value = None if op == "remove" else fact.value
        confidence = _clamp01(fact.confidence)
        evidence = str(fact.evidence or "").strip()
        event_id = str(fact.source_event_id or "")
        now_iso = now_local_ts()

        with self._lock:
            pending_root = self._data.setdefault("pending_facts", {})
            confirmed_root = self._data.setdefault("confirmed_facts", {})
            profiles = self._data.setdefault("profiles", {})
            versions = self._data.setdefault("versions", {})

            pending_profile = pending_root.setdefault(pid, {})
            confirmed_profile = confirmed_root.setdefault(pid, {})
            profile_row = profiles.setdefault(pid, {})
            profile_versions = versions.setdefault(pid, {})

            pending_items = list(pending_profile.get(key) or [])
            merged = self._merge_pending_item(
                pending_items,
                value=value,
                op=op,
                confidence=confidence,
                evidence=evidence,
                event_id=event_id,
            )
            pending_profile[key] = merged
            candidate = self._pick_pending_candidate(merged)
            should_confirm = bool(candidate and int(candidate.get("count") or 0) >= 2)
            if _is_confirmation_evidence(evidence):
                should_confirm = True

            status = "pending"
            applied = False
            current_entry = dict(profile_row.get(key) or {})
            previous_value = current_entry.get("value")

            if should_confirm and candidate:
                old_confirmed = dict(confirmed_profile.get(key) or {})
                old_value = old_confirmed.get("value")
                old_conf = float(old_confirmed.get("confidence") or 0.0)
                old_ts = parse_time_to_epoch(old_confirmed.get("confirmed_at"), 0.0)
                new_value = candidate.get("value")
                new_conf = _clamp01(candidate.get("confidence"))
                resolution = self._resolve_conflict(
                    key=key,
                    old_value=old_value,
                    old_conf=old_conf,
                    old_ts=old_ts,
                    new_value=new_value,
                    new_conf=new_conf,
                    new_ts=time.time(),
                    source=f"fact_{op}",
                )
                if bool(resolution.get("applied", True)):
                    status = "confirmed"
                    applied = True
                    pending_profile.pop(key, None)
                    confirmed_profile[key] = {
                        "key": key,
                        "value": new_value,
                        "op": op,
                        "confidence": new_conf,
                        "count": int(candidate.get("count") or 1),
                        "evidence": list(candidate.get("evidence") or []),
                        "source_event_ids": list(candidate.get("source_event_ids") or []),
                        "confirmed_at": now_iso,
                        "status": str(resolution.get("status") or "updated"),
                    }
                    current_entry = {
                        "value": new_value,
                        "confidence": new_conf,
                        "source_event_id": event_id,
                        "source": f"fact_{op}",
                        "updated_at": now_local_iso(),
                        "needs_confirmation": False,
                        "pending": [],
                    }
                    profile_row[key] = current_entry
                else:
                    status = "conflict_pending"
                    applied = False
                    current_entry = dict(profile_row.get(key) or {})
                    if current_entry:
                        current_entry["needs_confirmation"] = True
                        profile_row[key] = current_entry
            else:
                status = "pending"
                applied = False
                if current_entry:
                    current_entry["needs_confirmation"] = True
                    profile_row[key] = current_entry

            history_row = profile_versions.setdefault(key, [])
            history_row.append(
                {
                    "ts": now_iso,
                    "op": op,
                    "old": previous_value,
                    "new": value,
                    "confidence": confidence,
                    "source_event_id": event_id,
                    "source": f"fact_{op}",
                    "status": status,
                }
            )
            self.save()

            result = dict(profile_row.get(key) or {})
            if not result and candidate:
                result = {
                    "value": candidate.get("value"),
                    "confidence": candidate.get("confidence"),
                    "source_event_id": event_id,
                    "source": f"fact_{op}",
                    "updated_at": now_local_iso(),
                    "needs_confirmation": True,
                }
            result["status"] = status
            result["applied"] = bool(applied)
            result["needs_confirmation"] = not bool(applied)
            result["pending_count"] = int(candidate.get("count") or 1) if candidate else 0
            return result

    def get_pending_facts(self, profile_id: str = "default") -> dict[str, Any]:
        pid = _profile_id(profile_id)
        with self._lock:
            row = dict(self._data.get("pending_facts", {}).get(pid, {}))
            return _deepcopy(row)

    def get_confirmed_facts(self, profile_id: str = "default") -> dict[str, Any]:
        pid = _profile_id(profile_id)
        with self._lock:
            row = dict(self._data.get("confirmed_facts", {}).get(pid, {}))
            return _deepcopy(row)

    def set_confirmation_context(self, profile_id: str = "default", context: dict[str, Any] | None = None) -> None:
        pid = _profile_id(profile_id)
        row = dict(context or {})
        with self._lock:
            root = self._data.setdefault("confirmation_context", {})
            if row:
                root[pid] = row
            else:
                root.pop(pid, None)
            self.save()

    def get_confirmation_context(self, profile_id: str = "default") -> dict[str, Any]:
        pid = _profile_id(profile_id)
        with self._lock:
            root = dict(self._data.get("confirmation_context", {}))
            return _deepcopy(dict(root.get(pid) or {}))

    def clear_confirmation_context(self, profile_id: str = "default") -> None:
        self.set_confirmation_context(profile_id=profile_id, context={})

    def confirm_pending(
        self,
        profile_id: str = "default",
        *,
        limit: int = 4,
        keys: list[str] | None = None,
        expected_values: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        pid = _profile_id(profile_id)
        max_items = max(1, int(limit))
        key_filter = {_key(x) for x in list(keys or []) if _key(x)}
        expected_map: dict[str, Any] = {}
        for raw_key, raw_value in dict(expected_values or {}).items():
            key_norm = _key(raw_key)
            if not key_norm:
                continue
            expected_map[key_norm] = raw_value
        with self._lock:
            pending_root = self._data.setdefault("pending_facts", {})
            confirmed_root = self._data.setdefault("confirmed_facts", {})
            profiles = self._data.setdefault("profiles", {})
            versions = self._data.setdefault("versions", {})

            pending_profile = dict(pending_root.get(pid) or {})
            confirmed_profile = confirmed_root.setdefault(pid, {})
            profile_row = profiles.setdefault(pid, {})
            profile_versions = versions.setdefault(pid, {})

            candidates: list[tuple[str, dict[str, Any]]] = []
            for key, items in pending_profile.items():
                if key_filter and str(key) not in key_filter:
                    continue
                item_rows = [dict(x) for x in list(items or []) if isinstance(x, dict)]
                expected_value = expected_map.get(str(key)) if expected_map else None
                if expected_map and str(key) in expected_map:
                    matched = [dict(x) for x in item_rows if _value_equals(x.get("value"), expected_value)]
                    picked = self._pick_pending_candidate(matched)
                else:
                    picked = self._pick_pending_candidate(item_rows)
                if not picked:
                    continue
                candidates.append((str(key), dict(picked)))
            candidates.sort(
                key=lambda x: parse_time_to_epoch(dict(x[1]).get("updated_at"), 0.0),
                reverse=True,
            )
            candidates = candidates[:max_items]
            out: list[dict[str, Any]] = []
            for key, row in candidates:
                value = row.get("value")
                op = str(row.get("op") or "add").strip().lower()
                confidence = _clamp01(row.get("confidence"))
                event_ids = list(row.get("source_event_ids") or [])
                last_event_id = str(event_ids[-1] if event_ids else "")
                old_confirmed = dict(confirmed_profile.get(key) or {})
                resolution = self._resolve_conflict(
                    key=key,
                    old_value=old_confirmed.get("value"),
                    old_conf=float(old_confirmed.get("confidence") or 0.0),
                    old_ts=parse_time_to_epoch(old_confirmed.get("confirmed_at"), 0.0),
                    new_value=value,
                    new_conf=confidence,
                    new_ts=time.time(),
                    source="fact_confirm_user",
                )
                if not bool(resolution.get("applied", True)):
                    continue
                now_iso = now_local_ts()
                confirmed_profile[key] = {
                    "key": key,
                    "value": value,
                    "op": op,
                    "confidence": confidence,
                    "count": int(row.get("count") or 1),
                    "evidence": list(row.get("evidence") or []),
                    "source_event_ids": event_ids,
                    "confirmed_at": now_iso,
                    "status": str(resolution.get("status") or "updated"),
                }
                profile_row[key] = {
                    "value": value,
                    "confidence": confidence,
                    "source_event_id": last_event_id,
                    "source": "fact_confirm_user",
                    "updated_at": now_local_iso(),
                    "needs_confirmation": False,
                    "pending": [],
                }
                profile_versions.setdefault(key, []).append(
                    {
                        "ts": now_iso,
                        "op": op,
                        "old": old_confirmed.get("value"),
                        "new": value,
                        "confidence": confidence,
                        "source_event_id": last_event_id,
                        "source": "fact_confirm_user",
                        "status": "confirmed_by_user",
                    }
                )
                pending_profile.pop(key, None)
                out.append(dict(confirmed_profile[key]))

            pending_root[pid] = pending_profile
            self.save()
            return out

    def history(self, key: str, profile_id: str = "default") -> list[dict[str, Any]]:
        pid = _profile_id(profile_id)
        name = _key(key)
        with self._lock:
            rows = list(self._data.get("versions", {}).get(pid, {}).get(name, []))
            return [_deepcopy(x) for x in rows]

    @staticmethod
    def _merge_pending_item(
        items: list[dict[str, Any]],
        *,
        value: Any,
        op: str,
        confidence: float,
        evidence: str,
        event_id: str,
    ) -> list[dict[str, Any]]:
        rows = [dict(x) for x in list(items or []) if isinstance(x, dict)]
        now_iso = now_local_ts()
        op_norm = str(op or "add").strip().lower() or "add"
        matched = False
        for row in rows:
            if str(row.get("op") or "").strip().lower() != op_norm:
                continue
            if not _value_equals(row.get("value"), value):
                continue
            row["count"] = int(row.get("count") or 1) + 1
            row["confidence"] = _clamp01(max(float(row.get("confidence") or 0.0), float(confidence)))
            evidence_list = [str(x) for x in list(row.get("evidence") or []) if str(x).strip()]
            if evidence:
                evidence_list.append(evidence[:300])
            row["evidence"] = evidence_list[-8:]
            source_ids = [str(x) for x in list(row.get("source_event_ids") or []) if str(x).strip()]
            if event_id:
                source_ids.append(event_id)
            row["source_event_ids"] = source_ids[-8:]
            row["updated_at"] = now_iso
            matched = True
            break
        if not matched:
            rows.append(
                {
                    "value": value,
                    "op": op_norm,
                    "confidence": _clamp01(confidence),
                    "count": 1,
                    "evidence": [evidence[:300]] if evidence else [],
                    "source_event_ids": [event_id] if event_id else [],
                    "first_seen_at": now_iso,
                    "updated_at": now_iso,
                }
            )
        rows = rows[-8:]
        return rows

    @staticmethod
    def _pick_pending_candidate(items: list[dict[str, Any]]) -> dict[str, Any] | None:
        rows = [dict(x) for x in list(items or []) if isinstance(x, dict)]
        if not rows:
            return None
        rows.sort(
            key=lambda x: (
                int(x.get("count") or 0),
                float(x.get("confidence") or 0.0),
                parse_time_to_epoch(x.get("updated_at"), 0.0),
            ),
            reverse=True,
        )
        return rows[0]

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
                "pending_facts": dict(payload.get("pending_facts") or {}),
                "confirmed_facts": dict(payload.get("confirmed_facts") or {}),
                "confirmation_context": dict(payload.get("confirmation_context") or {}),
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


def _value_equals(a: Any, b: Any) -> bool:
    if a is None and b is None:
        return True
    if isinstance(a, (dict, list, tuple)) or isinstance(b, (dict, list, tuple)):
        try:
            return json.dumps(a, ensure_ascii=False, sort_keys=True) == json.dumps(b, ensure_ascii=False, sort_keys=True)
        except Exception:
            return str(a) == str(b)
    return str(a) == str(b)


def _is_confirmation_evidence(text: str) -> bool:
    src = str(text or "").strip().lower()
    if not src:
        return False
    if "?" in src:
        return False
    compact = " ".join(src.split())
    if len(compact) > 80:
        return False
    markers = {
        "yes",
        "correct",
        "exactly",
        "that's right",
        "right",
        "affirmative",
        "true",
        "da",
        "verno",
        "tochno",
        "\u0434\u0430",
        "\u0432\u0435\u0440\u043d\u043e",
        "\u0442\u043e\u0447\u043d\u043e",
    }
    return compact in markers or any(f" {m} " in f" {compact} " for m in markers)
