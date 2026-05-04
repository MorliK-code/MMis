from __future__ import annotations

import hashlib
import secrets
import sqlite3
import time
from pathlib import Path
from typing import Any

from ui.client_config_store import CLIENT_DATA_DIR


AUTH_DB_PATH = Path(CLIENT_DATA_DIR) / "accounts" / "auth.db"


def _now() -> float:
    return time.time()


def _norm_login(value: str) -> str:
    return str(value or "").strip().lower()


def _make_account_id() -> str:
    return "acc_" + secrets.token_hex(12)


class AuthStore:
    def __init__(self, db_path: str | Path | None = None):
        self.db_path = Path(db_path or AUTH_DB_PATH)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._init_db()

    def _execute(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        cur = self._conn.cursor()
        cur.execute(sql, params)
        return cur

    def _init_db(self) -> None:
        self._execute(
            """
            CREATE TABLE IF NOT EXISTS accounts (
                account_id TEXT PRIMARY KEY,
                login TEXT NOT NULL UNIQUE,
                display_name TEXT NOT NULL,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                disabled INTEGER NOT NULL DEFAULT 0,
                role TEXT NOT NULL DEFAULT 'user',
                api_access_key_hash TEXT NOT NULL DEFAULT ''
            )
            """
        )
        self._execute(
            """
            CREATE TABLE IF NOT EXISTS account_api_access_keys (
                key_hash TEXT PRIMARY KEY,
                account_id TEXT NOT NULL,
                label TEXT NOT NULL DEFAULT '',
                raw_key TEXT NOT NULL DEFAULT '',
                created_at REAL NOT NULL,
                revoked_at REAL,
                FOREIGN KEY(account_id) REFERENCES accounts(account_id) ON DELETE CASCADE
            )
            """
        )
        self._execute("CREATE INDEX IF NOT EXISTS idx_accounts_login ON accounts(login)")
        self._execute("CREATE INDEX IF NOT EXISTS idx_account_api_access_keys_account ON account_api_access_keys(account_id)")

    def _ensure_account(self, login: str) -> sqlite3.Row:
        clean = _norm_login(login)
        row = self._execute("SELECT * FROM accounts WHERE login=? AND disabled=0", (clean,)).fetchone()
        if row is not None:
            return row
        if not clean:
            raise ValueError("account_not_found")
        ts = _now()
        self._execute(
            """
            INSERT INTO accounts(account_id, login, display_name, created_at, updated_at, role)
            VALUES(?, ?, ?, ?, ?, ?)
            """,
            (_make_account_id(), clean, clean, ts, ts, "admin" if clean == "admin" else "user"),
        )
        row = self._execute("SELECT * FROM accounts WHERE login=? AND disabled=0", (clean,)).fetchone()
        if row is None:
            raise ValueError("account_not_found")
        return row

    def list_accounts(self) -> list[dict[str, Any]]:
        rows = self._execute("SELECT account_id, login, display_name, role, disabled FROM accounts ORDER BY lower(login)").fetchall()
        return [
            {
                "account_id": str(row["account_id"]),
                "login": str(row["login"]),
                "display_name": str(row["display_name"]),
                "role": str(row["role"] or "user"),
                "disabled": bool(row["disabled"]),
            }
            for row in rows
        ]

    def set_api_access_key_hash(self, login: str, key_hash: str, *, label: str = "default", raw_key: str = "") -> dict[str, Any]:
        row = self._ensure_account(login)
        clean_hash = str(key_hash or "").strip().lower()
        clean_label = str(label or "default").strip() or "default"
        self._execute("UPDATE accounts SET api_access_key_hash=?, updated_at=? WHERE account_id=?", (clean_hash, _now(), str(row["account_id"])))
        if clean_hash:
            self._execute(
                """
                INSERT INTO account_api_access_keys(key_hash, account_id, label, raw_key, created_at, revoked_at)
                VALUES(?, ?, ?, ?, ?, NULL)
                ON CONFLICT(key_hash) DO UPDATE SET
                    account_id=excluded.account_id,
                    label=excluded.label,
                    raw_key=CASE WHEN excluded.raw_key != '' THEN excluded.raw_key ELSE account_api_access_keys.raw_key END,
                    revoked_at=NULL
                """,
                (clean_hash, str(row["account_id"]), clean_label, str(raw_key or "").strip(), _now()),
            )
        return {"account_id": str(row["account_id"]), "login": str(row["login"]), "display_name": str(row["display_name"]), "role": str(row["role"] or "user")}

    def list_api_access_keys(self, login: str = "") -> list[dict[str, Any]]:
        clean = _norm_login(login)
        sql = """
            SELECT a.login, a.display_name, k.key_hash, k.label, k.raw_key, k.created_at, k.revoked_at
            FROM account_api_access_keys k
            JOIN accounts a ON a.account_id = k.account_id
        """
        params: tuple[Any, ...] = ()
        if clean:
            sql += " WHERE a.login=?"
            params = (clean,)
        sql += " ORDER BY a.login, k.created_at DESC"
        rows = self._execute(sql, params).fetchall()
        return [
            {
                "login": str(row["login"]),
                "display_name": str(row["display_name"]),
                "key_hash": str(row["key_hash"]),
                "label": str(row["label"] or ""),
                "key": str(row["raw_key"] or ""),
                "created_at": float(row["created_at"] or 0.0),
                "revoked": row["revoked_at"] is not None,
            }
            for row in rows
        ]

    def set_api_access_key_enabled(self, key_hash: str, enabled: bool) -> bool:
        clean = str(key_hash or "").strip().lower()
        cur = self._execute("UPDATE account_api_access_keys SET revoked_at=? WHERE key_hash=?", (None if enabled else _now(), clean))
        return bool(cur.rowcount)

    def delete_api_access_key(self, key_hash: str) -> bool:
        clean = str(key_hash or "").strip().lower()
        cur = self._execute("DELETE FROM account_api_access_keys WHERE key_hash=?", (clean,))
        self._execute("UPDATE accounts SET api_access_key_hash='', updated_at=? WHERE api_access_key_hash=?", (_now(), clean))
        return bool(cur.rowcount)

    def delete_api_access_keys_for_account(self, login: str) -> int:
        row = self._ensure_account(login)
        cur = self._execute("DELETE FROM account_api_access_keys WHERE account_id=?", (str(row["account_id"]),))
        self._execute("UPDATE accounts SET api_access_key_hash='', updated_at=? WHERE account_id=?", (_now(), str(row["account_id"])))
        try:
            return int(cur.rowcount or 0)
        except Exception:
            return 0


def hash_api_access_key(value: str) -> str:
    return hashlib.sha256(str(value or "").encode("utf-8")).hexdigest()
