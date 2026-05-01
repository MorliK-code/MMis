from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ui.client_config_store import CLIENT_DATA_DIR

AUTH_STATE_PATH = Path(CLIENT_DATA_DIR) / "auth.json"

DEFAULT_AUTH_STATE: dict[str, Any] = {
    "token": "",
    "account_id": "",
    "login": "",
    "display_name": "",
    "sessions": {},
}


def _session_key(data: dict[str, Any]) -> str:
    return str(data.get("account_id") or data.get("login") or "").strip()


def _session_payload(data: dict[str, Any]) -> dict[str, Any]:
    return {
        "token": str(data.get("token") or ""),
        "account_id": str(data.get("account_id") or ""),
        "login": str(data.get("login") or ""),
        "display_name": str(data.get("display_name") or ""),
    }


def load_auth_state() -> dict[str, Any]:
    if not AUTH_STATE_PATH.exists():
        return dict(DEFAULT_AUTH_STATE)
    try:
        data = json.loads(AUTH_STATE_PATH.read_text(encoding="utf-8-sig"))
        if not isinstance(data, dict):
            return dict(DEFAULT_AUTH_STATE)
        out = dict(DEFAULT_AUTH_STATE)
        out.update(data)
        if not isinstance(out.get("sessions"), dict):
            out["sessions"] = {}
        return out
    except Exception:
        return dict(DEFAULT_AUTH_STATE)


def save_auth_state(data: dict[str, Any]) -> dict[str, Any]:
    current = load_auth_state()
    sessions = dict(current.get("sessions") or {})
    current_session = _session_payload(current)
    current_key = _session_key(current_session)
    if current_key and current_session.get("token"):
        sessions[current_key] = current_session
    out = dict(DEFAULT_AUTH_STATE)
    if isinstance(data, dict):
        out.update(data)
    out["sessions"] = sessions
    session = _session_payload(out)
    key = _session_key(session)
    if key and session.get("token"):
        out["sessions"][key] = session
    AUTH_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    AUTH_STATE_PATH.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def clear_auth_state(*, forget_current: bool = False) -> None:
    current = load_auth_state()
    sessions = dict(current.get("sessions") or {})
    if forget_current:
        key = _session_key(current)
        if key:
            sessions.pop(key, None)
    out = dict(DEFAULT_AUTH_STATE)
    out["sessions"] = sessions
    AUTH_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    AUTH_STATE_PATH.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")


def list_auth_sessions() -> list[dict[str, Any]]:
    state = load_auth_state()
    sessions = state.get("sessions") or {}
    if not isinstance(sessions, dict):
        return []
    out: list[dict[str, Any]] = []
    for item in sessions.values():
        if not isinstance(item, dict):
            continue
        session = _session_payload(item)
        if _session_key(session) and session.get("token"):
            out.append(session)
    out.sort(key=lambda row: str(row.get("display_name") or row.get("login") or row.get("account_id") or "").lower())
    return out


def activate_auth_session(account_id: str) -> dict[str, Any]:
    state = load_auth_state()
    sessions = state.get("sessions") or {}
    if not isinstance(sessions, dict):
        raise ValueError("session_not_found")
    key = str(account_id or "").strip()
    session = sessions.get(key)
    if not isinstance(session, dict):
        raise ValueError("session_not_found")
    return save_auth_state(session)


def get_auth_token() -> str:
    return str(load_auth_state().get("token") or "").strip()


def get_auth_display_name() -> str:
    state = load_auth_state()
    return str(state.get("display_name") or state.get("login") or "").strip()
