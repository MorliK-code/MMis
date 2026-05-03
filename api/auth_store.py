from __future__ import annotations

import hashlib
import hmac
import secrets
import sqlite3
import time
from pathlib import Path
from typing import Any

from config.settings import DATA_DIR
from core.account_manager import AccountManager


AUTH_DB_PATH = Path(DATA_DIR) / "accounts" / "auth.db"
LEGACY_ACCOUNTS_DB_PATH = Path(DATA_DIR) / "accounts" / "accounts.db"
PBKDF2_ITERATIONS = 260_000


def _now() -> float:
    return time.time()


def _norm_login(value: str) -> str:
    return str(value or "").strip().lower()


def _hash_token(token: str) -> str:
    return hashlib.sha256(str(token or "").encode("utf-8")).hexdigest()


def _make_password_hash(password: str, salt_hex: str) -> str:
    raw = hashlib.pbkdf2_hmac(
        "sha256",
        str(password or "").encode("utf-8"),
        bytes.fromhex(salt_hex),
        PBKDF2_ITERATIONS,
    )
    return raw.hex()


def _make_password_pair(password: str) -> tuple[str, str]:
    salt_hex = secrets.token_hex(16)
    return salt_hex, _make_password_hash(password, salt_hex)


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
        self._execute("PRAGMA journal_mode=WAL")
        self._execute("PRAGMA foreign_keys=ON")
        self._execute(
            """
            CREATE TABLE IF NOT EXISTS accounts (
                account_id TEXT PRIMARY KEY,
                login TEXT NOT NULL UNIQUE,
                display_name TEXT NOT NULL,
                password_salt TEXT NOT NULL,
                password_hash TEXT NOT NULL,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                disabled INTEGER NOT NULL DEFAULT 0
            )
            """
        )
        try:
            self._execute("ALTER TABLE accounts ADD COLUMN role TEXT NOT NULL DEFAULT 'user'")
        except Exception:
            pass
        try:
            self._execute("ALTER TABLE accounts ADD COLUMN api_access_key_hash TEXT NOT NULL DEFAULT ''")
        except Exception:
            pass
        try:
            self._execute("UPDATE accounts SET role='admin' WHERE lower(login)='admin'")
        except Exception:
            pass
        self._execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                token_hash TEXT PRIMARY KEY,
                account_id TEXT NOT NULL,
                created_at REAL NOT NULL,
                last_seen_at REAL NOT NULL,
                revoked_at REAL,
                FOREIGN KEY(account_id) REFERENCES accounts(account_id) ON DELETE CASCADE
            )
            """
        )
        self._execute(
            """
            CREATE TABLE IF NOT EXISTS account_api_access_keys (
                key_hash TEXT PRIMARY KEY,
                account_id TEXT NOT NULL,
                label TEXT NOT NULL DEFAULT '',
                created_at REAL NOT NULL,
                revoked_at REAL,
                FOREIGN KEY(account_id) REFERENCES accounts(account_id) ON DELETE CASCADE
            )
            """
        )
        self._execute("CREATE INDEX IF NOT EXISTS idx_accounts_login ON accounts(login)")
        self._execute("CREATE INDEX IF NOT EXISTS idx_accounts_api_access_key_hash ON accounts(api_access_key_hash)")
        self._execute("CREATE INDEX IF NOT EXISTS idx_sessions_account ON sessions(account_id)")
        self._execute("CREATE INDEX IF NOT EXISTS idx_account_api_access_keys_account ON account_api_access_keys(account_id)")

    def _legacy_account_for_login(self, login: str) -> dict[str, str] | None:
        legacy_path = LEGACY_ACCOUNTS_DB_PATH
        if not legacy_path.exists():
            return None
        try:
            with sqlite3.connect(str(legacy_path)) as db:
                db.row_factory = sqlite3.Row
                row = db.execute(
                    """
                    SELECT account_id, login, display_name, password_hash
                    FROM accounts
                    WHERE lower(login)=? AND is_active=1
                    """,
                    (_norm_login(login),),
                ).fetchone()
        except Exception:
            return None
        if row is None:
            return None
        account_id = str(row["account_id"] or "").strip()
        login_value = str(row["login"] or "").strip()
        if not account_id or not login_value:
            return None
        return {
            "account_id": account_id,
            "login": _norm_login(login_value),
            "display_name": str(row["display_name"] or login_value).strip() or login_value,
            "password_hash": str(row["password_hash"] or ""),
        }

    def _can_reset_passwordless_legacy_account(self, login: str) -> bool:
        legacy = self._legacy_account_for_login(login)
        if not legacy:
            return False
        return not str(legacy.get("password_hash") or "").strip()

    def set_password(self, login: str, password: str) -> dict[str, Any]:
        clean_login = _norm_login(login)
        clean_password = str(password or "")
        if len(clean_password) < 4:
            raise ValueError("password_too_short")
        row = self._execute(
            "SELECT account_id, login, display_name, role FROM accounts WHERE login=? AND disabled=0",
            (clean_login,),
        ).fetchone()
        if row is None:
            raise ValueError("account_not_found")
        salt_hex, password_hash = _make_password_pair(clean_password)
        self._execute(
            "UPDATE accounts SET password_salt=?, password_hash=?, updated_at=? WHERE login=?",
            (salt_hex, password_hash, _now(), clean_login),
        )
        return {
            "account_id": str(row["account_id"]),
            "login": str(row["login"]),
            "display_name": str(row["display_name"]),
            "role": str(row["role"] or "user"),
        }

    def _ensure_local_account(self, login: str, display_name: str, password: str) -> dict[str, str]:
        manager = AccountManager(self.db_path.parent)
        try:
            existing = manager.get_by_login(login)
        except Exception:
            existing = None
        if existing is None:
            account = manager.create_account(login=login, display_name=display_name or login, password=password)
        else:
            account = existing
            manager._ensure_account_assets(account.to_context(), migrate_legacy=False)
        return {
            "account_id": account.account_id,
            "login": account.login,
            "display_name": account.display_name,
        }

    def create_account(self, login: str, password: str, display_name: str = "") -> dict[str, Any]:
        clean_login = _norm_login(login)
        clean_password = str(password or "")
        requested_display = str(display_name or "").strip()
        clean_display = requested_display or clean_login

        if len(clean_login) < 3:
            raise ValueError("login_too_short")
        if len(clean_password) < 4:
            raise ValueError("password_too_short")

        exists = self._execute(
            "SELECT account_id, login, display_name FROM accounts WHERE login=? AND disabled=0",
            (clean_login,),
        ).fetchone()
        if exists:
            if self._can_reset_passwordless_legacy_account(clean_login):
                return self.set_password(clean_login, clean_password)
            raise ValueError("account_exists")

        legacy = self._legacy_account_for_login(clean_login)
        account_id = str((legacy or {}).get("account_id") or "").strip()
        if not account_id:
            local = self._ensure_local_account(clean_login, clean_display, clean_password)
            account_id = str(local.get("account_id") or "").strip()
            clean_display = str(local.get("display_name") or clean_display).strip() or clean_display
        if legacy and not requested_display:
            clean_display = str(legacy.get("display_name") or clean_login).strip() or clean_login
        salt_hex, password_hash = _make_password_pair(clean_password)
        ts = _now()
        self._execute(
            """
            INSERT INTO accounts(account_id, login, display_name, password_salt, password_hash, created_at, updated_at, role)
            VALUES(?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                account_id,
                clean_login,
                clean_display,
                salt_hex,
                password_hash,
                ts,
                ts,
                "admin" if clean_login == "admin" else "user",
            ),
        )
        role = "admin" if clean_login == "admin" else "user"
        return {"account_id": account_id, "login": clean_login, "display_name": clean_display, "role": role}

    def verify_login(self, login: str, password: str) -> dict[str, Any] | None:
        clean_login = _norm_login(login)
        row = self._execute(
            "SELECT * FROM accounts WHERE login=? AND disabled=0",
            (clean_login,),
        ).fetchone()
        if row is None:
            return None
        expected = str(row["password_hash"] or "")
        actual = _make_password_hash(str(password or ""), str(row["password_salt"] or ""))
        if not hmac.compare_digest(expected, actual):
            return None
        return {
            "account_id": str(row["account_id"]),
            "login": str(row["login"]),
            "display_name": str(row["display_name"]),
            "role": str(row["role"] or "user"),
        }

    def create_session(self, account_id: str) -> str:
        token = secrets.token_urlsafe(48)
        token_hash = _hash_token(token)
        ts = _now()
        self._execute(
            """
            INSERT INTO sessions(token_hash, account_id, created_at, last_seen_at, revoked_at)
            VALUES(?, ?, ?, ?, NULL)
            """,
            (token_hash, str(account_id), ts, ts),
        )
        return token

    def get_account_by_token(self, token: str) -> dict[str, Any] | None:
        token_hash = _hash_token(token)
        row = self._execute(
            """
            SELECT a.account_id, a.login, a.display_name, a.role
            FROM sessions s
            JOIN accounts a ON a.account_id = s.account_id
            WHERE s.token_hash=? AND s.revoked_at IS NULL AND a.disabled=0
            """,
            (token_hash,),
        ).fetchone()
        if row is None:
            return None
        self._execute("UPDATE sessions SET last_seen_at=? WHERE token_hash=?", (_now(), token_hash))
        return {
            "account_id": str(row["account_id"]),
            "login": str(row["login"]),
            "display_name": str(row["display_name"]),
            "role": str(row["role"] or "user"),
        }

    def set_api_access_key_hash(self, login: str, key_hash: str, *, label: str = "default") -> dict[str, Any]:
        clean_login = _norm_login(login)
        clean_hash = str(key_hash or "").strip().lower()
        row = self._execute(
            "SELECT account_id, login, display_name, role FROM accounts WHERE login=? AND disabled=0",
            (clean_login,),
        ).fetchone()
        if row is None:
            raise ValueError("account_not_found")
        clean_label = str(label or "default").strip() or "default"
        self._execute(
            "UPDATE accounts SET api_access_key_hash=?, updated_at=? WHERE login=?",
            (clean_hash, _now(), clean_login),
        )
        if clean_hash:
            self._execute(
                """
                INSERT INTO account_api_access_keys(key_hash, account_id, label, created_at, revoked_at)
                VALUES(?, ?, ?, ?, NULL)
                ON CONFLICT(key_hash) DO UPDATE SET
                    account_id=excluded.account_id,
                    label=excluded.label,
                    revoked_at=NULL
                """,
                (clean_hash, str(row["account_id"]), clean_label, _now()),
            )
        return {
            "account_id": str(row["account_id"]),
            "login": str(row["login"]),
            "display_name": str(row["display_name"]),
            "role": str(row["role"] or "user"),
        }

    def get_account_by_api_access_key_hash(self, key_hash: str) -> dict[str, Any] | None:
        clean_hash = str(key_hash or "").strip().lower()
        if not clean_hash:
            return None
        row = self._execute(
            """
            SELECT a.account_id, a.login, a.display_name, a.role
            FROM account_api_access_keys k
            JOIN accounts a ON a.account_id = k.account_id
            WHERE k.key_hash=? AND k.revoked_at IS NULL AND a.disabled=0
            """,
            (clean_hash,),
        ).fetchone()
        if row is None:
            row = self._execute(
                """
            SELECT account_id, login, display_name, role
            FROM accounts
            WHERE api_access_key_hash=? AND disabled=0
            """,
                (clean_hash,),
            ).fetchone()
        if row is None:
            return None
        return {
            "account_id": str(row["account_id"]),
            "login": str(row["login"]),
            "display_name": str(row["display_name"]),
            "role": str(row["role"] or "user"),
        }

    def list_api_access_keys(self, login: str = "") -> list[dict[str, Any]]:
        clean_login = _norm_login(login)
        if clean_login:
            rows = self._execute(
                """
                SELECT a.login, a.display_name, k.key_hash, k.label, k.created_at, k.revoked_at
                FROM account_api_access_keys k
                JOIN accounts a ON a.account_id = k.account_id
                WHERE a.login=?
                ORDER BY a.login, k.created_at DESC
                """,
                (clean_login,),
            ).fetchall()
        else:
            rows = self._execute(
                """
                SELECT a.login, a.display_name, k.key_hash, k.label, k.created_at, k.revoked_at
                FROM account_api_access_keys k
                JOIN accounts a ON a.account_id = k.account_id
                ORDER BY a.login, k.created_at DESC
                """
            ).fetchall()
        return [
            {
                "login": str(row["login"]),
                "display_name": str(row["display_name"]),
                "key_hash": str(row["key_hash"]),
                "label": str(row["label"] or ""),
                "created_at": float(row["created_at"] or 0.0),
                "revoked": row["revoked_at"] is not None,
            }
            for row in rows
        ]

    def clear_api_access_keys(self, login: str) -> int:
        clean_login = _norm_login(login)
        row = self._execute(
            "SELECT account_id FROM accounts WHERE login=? AND disabled=0",
            (clean_login,),
        ).fetchone()
        if row is None:
            raise ValueError("account_not_found")
        account_id = str(row["account_id"])
        cur = self._execute(
            "UPDATE account_api_access_keys SET revoked_at=? WHERE account_id=? AND revoked_at IS NULL",
            (_now(), account_id),
        )
        self._execute(
            "UPDATE accounts SET api_access_key_hash='', updated_at=? WHERE account_id=?",
            (_now(), account_id),
        )
        try:
            return int(cur.rowcount or 0)
        except Exception:
            return 0

    def revoke_token(self, token: str) -> None:
        token_hash = _hash_token(token)
        self._execute("UPDATE sessions SET revoked_at=? WHERE token_hash=?", (_now(), token_hash))
