"""Chat session persistence and migration helpers for the desktop UI."""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path

HistoryRow = tuple[str, str, str | None, int | None, str | None]
SINGLE_VISIBLE_CHAT_ID = "visible-main-chat"
SINGLE_VISIBLE_CHAT_TITLE = "Чат"


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def history_to_serializable(history: list[HistoryRow]) -> list[dict]:
    out: list[dict] = []
    for role, text, stat_line, feedback, thinking in history:
        out.append(
            {
                "role": str(role),
                "text": str(text or ""),
                "stat_line": None if stat_line is None else str(stat_line),
                "feedback": None if feedback is None else int(feedback),
                "thinking": None if not thinking else str(thinking),
            }
        )
    return out


def history_from_serializable(rows: list[dict] | None) -> list[HistoryRow]:
    out: list[HistoryRow] = []
    if not isinstance(rows, list):
        return out
    for row in rows:
        if not isinstance(row, dict):
            continue
        role = str(row.get("role") or "")
        if role not in {"system", "user", "ai"}:
            continue
        text = str(row.get("text") or "")
        stat_line_raw = row.get("stat_line")
        stat_line = None if stat_line_raw is None else str(stat_line_raw)
        feedback_raw = row.get("feedback")
        feedback = None
        if feedback_raw is not None:
            try:
                feedback = int(feedback_raw)
            except Exception:
                feedback = None
        thinking_raw = row.get("thinking")
        thinking = None if thinking_raw is None else str(thinking_raw)
        if role == "system" and text.strip().startswith("MMis UI запущен"):
            continue
        out.append((role, text, stat_line, feedback, thinking))
    return out


def make_new_chat_payload(existing_count: int, title: str | None = None, incognito: bool = False) -> dict:
    ts = now_iso()
    next_num = int(existing_count) + 1
    return {
        "id": f"chat-{int(datetime.now().timestamp() * 1000)}-{next_num}",
        "title": title or f"Чат {next_num}",
        "incognito": bool(incognito),
        "created_at": ts,
        "updated_at": ts,
        "history": [],
    }


def collapse_to_single_visible_chat(chats: list[dict] | None, active_chat_id: str | None = None) -> dict:
    rows = [dict(chat or {}) for chat in list(chats or []) if isinstance(chat, dict) and not bool(chat.get("incognito", False))]
    chosen: dict | None = None
    target_id = str(active_chat_id or "").strip()
    if target_id:
        chosen = next((row for row in rows if str(row.get("id") or "").strip() == target_id), None)
    if chosen is None and rows:
        chosen = max(rows, key=lambda row: str(row.get("updated_at") or row.get("created_at") or ""))
    if chosen is None:
        ts = now_iso()
        return {
            "id": SINGLE_VISIBLE_CHAT_ID,
            "title": SINGLE_VISIBLE_CHAT_TITLE,
            "incognito": False,
            "created_at": ts,
            "updated_at": ts,
            "history": [],
        }
    return {
        "id": SINGLE_VISIBLE_CHAT_ID,
        "title": SINGLE_VISIBLE_CHAT_TITLE,
        "incognito": False,
        "created_at": str(chosen.get("created_at") or now_iso()),
        "updated_at": str(chosen.get("updated_at") or now_iso()),
        "history": list(chosen.get("history") or []),
    }


def session_dir(sessions_dir: Path, chat_id: str) -> Path:
    return sessions_dir / str(chat_id)


def session_payload_path(sessions_dir: Path, chat_id: str) -> Path:
    return session_dir(sessions_dir, chat_id) / "chat.json"


def load_legacy_sessions(legacy_sessions_path: Path) -> tuple[list[dict], str | None]:
    loaded: list[dict] = []
    active_id: str | None = None
    if not legacy_sessions_path.exists():
        return loaded, active_id
    try:
        payload = json.loads(legacy_sessions_path.read_text(encoding="utf-8-sig"))
        rows = payload.get("chats", []) if isinstance(payload, dict) else []
        active_id_raw = payload.get("active_chat_id") if isinstance(payload, dict) else None
        active_id = str(active_id_raw) if active_id_raw else None
        if isinstance(rows, list):
            for row in rows:
                if not isinstance(row, dict):
                    continue
                chat_id = str(row.get("id") or "").strip()
                if not chat_id:
                    continue
                loaded.append(
                    {
                        "id": chat_id,
                        "title": str(row.get("title") or "Чат"),
                        "incognito": bool(row.get("incognito", False)),
                        "created_at": str(row.get("created_at") or now_iso()),
                        "updated_at": str(row.get("updated_at") or now_iso()),
                        "history": history_from_serializable(row.get("history")),
                    }
                )
        loaded = [c for c in loaded if not bool(c.get("incognito", False))]
        if active_id and not any(str(c.get("id")) == active_id for c in loaded):
            active_id = None
    except Exception:
        loaded = []
        active_id = None
    return loaded, active_id


def load_sessions(sessions_dir: Path, sessions_index_path: Path, legacy_sessions_path: Path) -> tuple[list[dict], str | None]:
    loaded: list[dict] = []
    active_id: str | None = None
    if sessions_index_path.exists():
        try:
            payload = json.loads(sessions_index_path.read_text(encoding="utf-8-sig"))
            rows = payload.get("chats", []) if isinstance(payload, dict) else []
            active_id_raw = payload.get("active_chat_id") if isinstance(payload, dict) else None
            active_id = str(active_id_raw) if active_id_raw else None
            if isinstance(rows, list):
                for row in rows:
                    if not isinstance(row, dict):
                        continue
                    chat_id = str(row.get("id") or "").strip()
                    if not chat_id:
                        continue
                    payload_path = session_payload_path(sessions_dir, chat_id)
                    if not payload_path.exists():
                        continue
                    chat_payload = json.loads(payload_path.read_text(encoding="utf-8-sig"))
                    loaded.append(
                        {
                            "id": chat_id,
                            "title": str(chat_payload.get("title") or row.get("title") or "Чат"),
                            "incognito": False,
                            "created_at": str(chat_payload.get("created_at") or row.get("created_at") or now_iso()),
                            "updated_at": str(chat_payload.get("updated_at") or row.get("updated_at") or now_iso()),
                            "history": history_from_serializable(chat_payload.get("history")),
                        }
                    )
        except Exception:
            loaded = []

    if not loaded:
        loaded, active_id = load_legacy_sessions(legacy_sessions_path)

    return loaded, active_id


def save_sessions(sessions_dir: Path, sessions_index_path: Path, chats: list[dict], active_chat_id: str | None) -> None:
    sessions_dir.mkdir(parents=True, exist_ok=True)
    persisted_ids: set[str] = set()
    persisted_meta: list[dict] = []

    for chat in chats:
        if bool(chat.get("incognito", False)):
            continue
        chat_id = str(chat.get("id") or "").strip()
        if not chat_id:
            continue
        persisted_ids.add(chat_id)
        payload = {
            "id": chat_id,
            "title": str(chat.get("title") or "Чат"),
            "incognito": False,
            "created_at": str(chat.get("created_at") or now_iso()),
            "updated_at": str(chat.get("updated_at") or now_iso()),
            "history": history_to_serializable(chat.get("history") or []),
        }
        chat_path = session_payload_path(sessions_dir, chat_id)
        chat_path.parent.mkdir(parents=True, exist_ok=True)
        chat_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        persisted_meta.append(
            {
                "id": payload["id"],
                "title": payload["title"],
                "created_at": payload["created_at"],
                "updated_at": payload["updated_at"],
            }
        )

    for child in sessions_dir.iterdir():
        if not child.is_dir():
            continue
        if child.name in persisted_ids:
            continue
        shutil.rmtree(child, ignore_errors=True)

    index_payload = {"version": 2, "active_chat_id": active_chat_id, "chats": persisted_meta}
    sessions_index_path.write_text(json.dumps(index_payload, ensure_ascii=False, indent=2), encoding="utf-8")
