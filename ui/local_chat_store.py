from __future__ import annotations

import json
import sqlite3
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Chat:
    chat_id: str
    title: str
    created_at: float
    updated_at: float
    persona_id: str
    archived: bool = False


@dataclass(frozen=True)
class ChatMessage:
    message_id: str
    chat_id: str
    role: str
    text: str
    metadata: dict[str, Any]
    created_at: float


class ChatStore:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path).expanduser().resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def create_chat(self, title: str, persona_id: str = "default", chat_id: str | None = None) -> Chat:
        now = time.time()
        chat = Chat(str(chat_id or f"chat_{uuid.uuid4().hex}"), str(title or "New chat"), now, now, str(persona_id or "default"))
        with self._connect() as db:
            db.execute(
                "INSERT INTO chats (chat_id, title, created_at, updated_at, persona_id, archived) VALUES (?, ?, ?, ?, ?, 0)",
                (chat.chat_id, chat.title, chat.created_at, chat.updated_at, chat.persona_id),
            )
        return chat

    def upsert_chat(self, chat_id: str, title: str, persona_id: str = "default", *, updated_at: float | None = None) -> Chat:
        existing = self.get_chat(chat_id)
        now = float(updated_at or time.time())
        if existing is None:
            chat = Chat(str(chat_id), str(title or "New chat"), now, now, str(persona_id or "default"))
            with self._connect() as db:
                db.execute(
                    "INSERT INTO chats (chat_id, title, created_at, updated_at, persona_id, archived) VALUES (?, ?, ?, ?, ?, 0)",
                    (chat.chat_id, chat.title, chat.created_at, chat.updated_at, chat.persona_id),
                )
            return chat
        with self._connect() as db:
            db.execute(
                "UPDATE chats SET title = ?, persona_id = ?, updated_at = ? WHERE chat_id = ?",
                (str(title or existing.title), str(persona_id or existing.persona_id), now, str(chat_id)),
            )
        return self.get_chat(chat_id) or existing

    def list_chats(self, *, include_archived: bool = False) -> list[Chat]:
        sql = "SELECT chat_id, title, created_at, updated_at, persona_id, archived FROM chats"
        if not include_archived:
            sql += " WHERE archived = 0"
        sql += " ORDER BY updated_at DESC"
        with self._connect() as db:
            return [self._row_to_chat(row) for row in db.execute(sql).fetchall()]

    def get_chat(self, chat_id: str) -> Chat | None:
        with self._connect() as db:
            row = db.execute(
                "SELECT chat_id, title, created_at, updated_at, persona_id, archived FROM chats WHERE chat_id = ?",
                (str(chat_id),),
            ).fetchone()
        return self._row_to_chat(row) if row is not None else None

    def add_message(self, chat_id: str, role: str, text: str, metadata: dict[str, Any] | None = None, message_id: str | None = None) -> ChatMessage:
        created_at = time.time()
        message = ChatMessage(str(message_id or f"msg_{uuid.uuid4().hex}"), str(chat_id), str(role), str(text), dict(metadata or {}), created_at)
        with self._connect() as db:
            db.execute(
                "INSERT INTO messages (message_id, chat_id, role, text, metadata_json, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (message.message_id, message.chat_id, message.role, message.text, json.dumps(message.metadata, ensure_ascii=False, sort_keys=True), message.created_at),
            )
            db.execute("UPDATE chats SET updated_at = ? WHERE chat_id = ?", (created_at, message.chat_id))
        return message

    def list_messages(self, chat_id: str, *, limit: int = 10000) -> list[ChatMessage]:
        with self._connect() as db:
            rows = db.execute(
                "SELECT message_id, chat_id, role, text, metadata_json, created_at FROM messages WHERE chat_id = ? ORDER BY created_at ASC LIMIT ?",
                (str(chat_id), max(1, int(limit))),
            ).fetchall()
        return [self._row_to_message(row) for row in rows]

    def replace_messages(self, chat_id: str, messages: list[dict[str, Any]]) -> None:
        with self._connect() as db:
            db.execute("DELETE FROM messages WHERE chat_id = ?", (str(chat_id),))
            for index, row in enumerate(list(messages or [])):
                if not isinstance(row, dict):
                    continue
                db.execute(
                    "INSERT INTO messages (message_id, chat_id, role, text, metadata_json, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        str(row.get("message_id") or f"{chat_id}:{index}"),
                        str(chat_id),
                        str(row.get("role") or ""),
                        str(row.get("text") or ""),
                        json.dumps(dict(row.get("metadata") or {}), ensure_ascii=False, sort_keys=True),
                        float(row.get("created_at") or time.time() + (index / 1000.0)),
                    ),
                )

    def archive_chat(self, chat_id: str, archived: bool = True) -> None:
        with self._connect() as db:
            db.execute("UPDATE chats SET archived = ?, updated_at = ? WHERE chat_id = ?", (1 if archived else 0, time.time(), str(chat_id)))

    def archive_chats_except(self, keep_ids: set[str]) -> None:
        keep = {str(item) for item in set(keep_ids or set()) if str(item).strip()}
        with self._connect() as db:
            if not keep:
                db.execute("UPDATE chats SET archived = 1, updated_at = ?", (time.time(),))
                return
            placeholders = ",".join("?" for _ in keep)
            db.execute(f"UPDATE chats SET archived = 1, updated_at = ? WHERE chat_id NOT IN ({placeholders})", (time.time(), *sorted(keep)))

    def _init_db(self) -> None:
        with self._connect() as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS chats (chat_id TEXT PRIMARY KEY, title TEXT NOT NULL, created_at REAL NOT NULL, updated_at REAL NOT NULL, persona_id TEXT NOT NULL DEFAULT 'default', archived INTEGER NOT NULL DEFAULT 0)"
            )
            db.execute(
                "CREATE TABLE IF NOT EXISTS messages (message_id TEXT PRIMARY KEY, chat_id TEXT NOT NULL, role TEXT NOT NULL, text TEXT NOT NULL, metadata_json TEXT NOT NULL, created_at REAL NOT NULL)"
            )
            db.execute("CREATE INDEX IF NOT EXISTS idx_messages_chat_created ON messages(chat_id, created_at)")

    @contextmanager
    def _connect(self):
        db = sqlite3.connect(str(self.db_path))
        db.row_factory = sqlite3.Row
        try:
            yield db
            db.commit()
        finally:
            db.close()

    @staticmethod
    def _row_to_chat(row: sqlite3.Row) -> Chat:
        return Chat(str(row["chat_id"]), str(row["title"]), float(row["created_at"]), float(row["updated_at"]), str(row["persona_id"] or "default"), bool(row["archived"]))

    @staticmethod
    def _row_to_message(row: sqlite3.Row) -> ChatMessage:
        try:
            metadata = dict(json.loads(str(row["metadata_json"] or "{}")) or {})
        except Exception:
            metadata = {}
        return ChatMessage(str(row["message_id"]), str(row["chat_id"]), str(row["role"]), str(row["text"]), metadata, float(row["created_at"]))
