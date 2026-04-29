from __future__ import annotations

import base64
import hashlib
import hmac
import os
import shutil
import sqlite3
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from config.settings import BASE_DIR, DATA_DIR
from core.account_context import AccountContext


_PBKDF2_ITERATIONS = 210_000


@dataclass(frozen=True)
class Account:
    account_id: str
    login: str
    display_name: str
    password_hash: str
    created_at: float
    last_login_at: float | None
    is_active: bool
    data_path: Path

    def to_context(self) -> AccountContext:
        return AccountContext.from_data_dir(
            account_id=self.account_id,
            login=self.login,
            display_name=self.display_name,
            data_dir=self.data_path,
        )


class AccountManager:
    def __init__(self, accounts_dir: str | Path | None = None):
        self.accounts_dir = Path(accounts_dir or (DATA_DIR / "accounts")).expanduser().resolve()
        self.accounts_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.accounts_dir / "accounts.db"
        self.current_path = self.accounts_dir / "current_account.txt"
        self._init_db()

    def create_account(self, login: str, display_name: str, password: str | None = None) -> Account:
        login_norm = self._normalize_login(login)
        display = str(display_name or login_norm).strip() or login_norm
        is_first_account = not self.list_accounts()
        account_id = self._new_account_id(login_norm)
        data_path = self.accounts_dir / account_id
        ctx = AccountContext.from_data_dir(
            account_id=account_id,
            login=login_norm,
            display_name=display,
            data_dir=data_path,
        )
        ctx.ensure_dirs()
        self._seed_account_config(ctx, from_legacy=is_first_account)
        self._write_account_paths_to_config(ctx)
        self._seed_default_character_specs(ctx)
        if is_first_account:
            self._migrate_legacy_data(ctx)

        created_at = time.time()
        password_hash = _hash_password(password) if password else ""
        with self._connect() as db:
            db.execute(
                """
                INSERT INTO accounts (
                    account_id, login, display_name, password_hash,
                    created_at, last_login_at, is_active, data_path
                )
                VALUES (?, ?, ?, ?, ?, NULL, 1, ?)
                """,
                (account_id, login_norm, display, password_hash, created_at, str(data_path)),
            )
        account = self.get_account(account_id)
        if account is None:
            raise RuntimeError("Account was created but could not be loaded")
        if self.get_current_account() is None:
            self.set_current_account(account.account_id)
        return account

    def login(self, login: str, password: str | None = None) -> Account:
        login_norm = self._normalize_login(login)
        account = self.get_by_login(login_norm)
        if account is None or not account.is_active:
            raise ValueError("Account not found")
        if account.password_hash and not _verify_password(password or "", account.password_hash):
            raise ValueError("Invalid password")
        now = time.time()
        with self._connect() as db:
            db.execute("UPDATE accounts SET last_login_at = ? WHERE account_id = ?", (now, account.account_id))
        self.set_current_account(account.account_id)
        return self.get_account(account.account_id) or account

    def list_accounts(self) -> list[Account]:
        with self._connect() as db:
            rows = db.execute(
                """
                SELECT account_id, login, display_name, password_hash, created_at,
                       last_login_at, is_active, data_path
                FROM accounts
                WHERE is_active = 1
                ORDER BY last_login_at DESC, created_at ASC
                """
            ).fetchall()
        return [self._row_to_account(row) for row in rows]

    def get_current_account(self) -> Account | None:
        account_id = ""
        try:
            account_id = self.current_path.read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            return None
        except OSError:
            return None
        return self.get_account(account_id) if account_id else None

    def set_current_account(self, account_id: str) -> None:
        account = self.get_account(account_id)
        if account is None or not account.is_active:
            raise ValueError("Account not found")
        self.current_path.write_text(account.account_id, encoding="utf-8")

    def get_account(self, account_id: str) -> Account | None:
        with self._connect() as db:
            row = db.execute(
                """
                SELECT account_id, login, display_name, password_hash, created_at,
                       last_login_at, is_active, data_path
                FROM accounts
                WHERE account_id = ?
                """,
                (str(account_id),),
            ).fetchone()
        return self._row_to_account(row) if row is not None else None

    def get_by_login(self, login: str) -> Account | None:
        with self._connect() as db:
            row = db.execute(
                """
                SELECT account_id, login, display_name, password_hash, created_at,
                       last_login_at, is_active, data_path
                FROM accounts
                WHERE login = ?
                """,
                (self._normalize_login(login),),
            ).fetchone()
        return self._row_to_account(row) if row is not None else None

    def resolve_account_path(self, *parts: str) -> Path:
        account = self.get_current_account()
        if account is None:
            raise RuntimeError("No current account selected")
        root = account.data_path.expanduser().resolve()
        target = root.joinpath(*[str(part) for part in parts]).resolve()
        try:
            target.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"Path escapes account directory: {target}") from exc
        return target

    def ensure_default_account(self, login: str = "default", display_name: str = "Default") -> Account:
        current = self.get_current_account()
        if current is not None:
            self._ensure_account_assets(current.to_context(), migrate_legacy=True)
            return current
        accounts = self.list_accounts()
        if accounts:
            self.set_current_account(accounts[0].account_id)
            self._ensure_account_assets(accounts[0].to_context(), migrate_legacy=True)
            return accounts[0]
        return self.create_account(login=login, display_name=display_name)

    def ensure_all_account_assets(self) -> None:
        for account in self.list_accounts():
            self._ensure_account_assets(account.to_context(), migrate_legacy=False)

    def _ensure_account_assets(self, context: AccountContext, *, migrate_legacy: bool) -> None:
        context.ensure_dirs()
        self._seed_account_config(context, from_legacy=False)
        self._write_account_paths_to_config(context)
        self._seed_default_character_specs(context)
        if migrate_legacy:
            self._migrate_legacy_data(context)

    def _init_db(self) -> None:
        with self._connect() as db:
            db.execute(
                """
                CREATE TABLE IF NOT EXISTS accounts (
                    account_id TEXT PRIMARY KEY,
                    login TEXT NOT NULL UNIQUE,
                    display_name TEXT NOT NULL,
                    password_hash TEXT,
                    created_at REAL NOT NULL,
                    last_login_at REAL,
                    is_active INTEGER NOT NULL DEFAULT 1,
                    data_path TEXT NOT NULL
                )
                """
            )

    @contextmanager
    def _connect(self):
        db = sqlite3.connect(str(self.db_path))
        db.row_factory = sqlite3.Row
        try:
            yield db
            db.commit()
        finally:
            db.close()

    def _row_to_account(self, row: sqlite3.Row) -> Account:
        return Account(
            account_id=str(row["account_id"]),
            login=str(row["login"]),
            display_name=str(row["display_name"]),
            password_hash=str(row["password_hash"] or ""),
            created_at=float(row["created_at"] or 0.0),
            last_login_at=(float(row["last_login_at"]) if row["last_login_at"] is not None else None),
            is_active=bool(row["is_active"]),
            data_path=Path(str(row["data_path"])).expanduser().resolve(),
        )

    def _new_account_id(self, login: str) -> str:
        base = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in login.lower()).strip("_")
        base = base or "account"
        account_id = base
        while (self.accounts_dir / account_id).exists() or self.get_account(account_id) is not None:
            account_id = f"{base}_{uuid.uuid4().hex[:8]}"
        return account_id

    @staticmethod
    def _normalize_login(login: str) -> str:
        text = str(login or "").strip().lower()
        if not text:
            raise ValueError("Login is required")
        return text

    @staticmethod
    def _seed_account_config(account: AccountContext, *, from_legacy: bool = False) -> None:
        if account.config_path.exists():
            return
        source = BASE_DIR / "config" / "config.json"
        if from_legacy and source.exists():
            shutil.copy2(source, account.config_path)
        else:
            try:
                from config.settings import _default_config_tree

                import json

                account.config_path.write_text(
                    json.dumps(_default_config_tree(), ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            except Exception:
                account.config_path.write_text("{}\n", encoding="utf-8")

    def _migrate_legacy_data(self, account: AccountContext) -> None:
        marker = account.data_dir / ".legacy_migration_done"
        if marker.exists():
            return

        legacy_memory = DATA_DIR / "memory_core"
        legacy_cache = DATA_DIR / "cache"
        legacy_ui_state = BASE_DIR / "ui" / ".mmis_client" / "ui_state.json"
        legacy_ui_chats = BASE_DIR / "ui" / ".mmis_client" / "ui_chats"

        _copy_file_if_missing(legacy_memory / "memory.db", account.memory_db_path)
        _copy_dir_if_missing(legacy_memory / "vector", account.memory_vector_path)
        _copy_file_if_missing(legacy_memory / "brain_state.json", account.data_dir / "state" / "brain_state.json")
        _copy_dir_if_missing(legacy_memory / "brain_state_store", account.data_dir / "state" / "brain_state_store")
        _copy_dir_if_missing(legacy_memory / "characters_runtime", account.data_dir / "characters_runtime")
        _copy_dir_if_missing(DATA_DIR / "specs" / "characters", account.data_dir / "specs" / "characters")
        _copy_file_if_missing(DATA_DIR / "specs" / "performance_profiles.json", account.performance_profiles_path)
        _copy_dir_if_missing(legacy_cache, account.data_dir / "cache")
        _copy_file_if_missing(legacy_ui_state, account.data_dir / "ui_state.json")
        _copy_dir_if_missing(legacy_ui_chats, account.data_dir / "chats" / "ui_chats")

        marker.write_text(f"migrated_at={time.time()}\n", encoding="utf-8")

    @staticmethod
    def _seed_default_character_specs(account: AccountContext) -> None:
        for cid in ("default", "asya"):
            _copy_dir_if_missing(DATA_DIR / "specs" / "characters" / cid, account.data_dir / "specs" / "characters" / cid)
        _copy_file_if_missing(DATA_DIR / "specs" / "performance_profiles.json", account.performance_profiles_path)

    def _write_account_paths_to_config(self, account: AccountContext) -> None:
        try:
            import json

            payload = json.loads(account.config_path.read_text(encoding="utf-8-sig"))
            if not isinstance(payload, dict):
                payload = {}
        except Exception:
            payload = {}

        _set_dotted(payload, "account.account_id", account.account_id)
        _set_dotted(payload, "paths.accounts_dir", str(self.accounts_dir))
        _set_dotted(payload, "memory.memory_dir", str(account.memory_db_path.parent))
        _set_dotted(payload, "memory.cache_dir", str(account.data_dir / "cache"))
        _set_dotted(payload, "memory.db_path", str(account.memory_db_path))
        _set_dotted(payload, "memory_core.memory_dir", str(account.memory_db_path.parent))
        _set_dotted(payload, "memory_core.cache_dir", str(account.data_dir / "cache"))
        _set_dotted(payload, "memory_core.db_path", str(account.memory_db_path))
        _set_dotted(payload, "memory_core.vector_path", str(account.memory_vector_path))
        _set_dotted(payload, "memory_core.default_workspace", account.account_id)
        _set_dotted(payload, "paths.character_specs_dir", str(account.data_dir / "specs" / "characters"))
        _set_dotted(payload, "paths.performance_profiles_file", str(account.performance_profiles_path))

        account.config_path.parent.mkdir(parents=True, exist_ok=True)
        account.config_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def prepare_local_account(settings) -> AccountContext:
    manager = AccountManager(getattr(settings, "accounts_dir", None))
    account_id = str(getattr(settings, "account_id", "") or "").strip()
    account = manager.get_account(account_id) if account_id else manager.get_current_account()
    if account is None:
        account = manager.ensure_default_account()

    context = account.to_context()
    manager.ensure_all_account_assets()
    manager._ensure_account_assets(context, migrate_legacy=False)
    os.environ["MMIS_ACTIVE_ACCOUNT_ID"] = account.account_id
    os.environ["MMIS_ACCOUNTS_DIR"] = str(manager.accounts_dir)
    try:
        from core.spec_registry import get_spec_registry

        get_spec_registry(force_reload=True)
    except Exception:
        pass
    return context


def _copy_file_if_missing(source: Path, target: Path) -> None:
    try:
        if not source.exists() or target.exists():
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    except Exception:
        pass


def _copy_dir_if_missing(source: Path, target: Path) -> None:
    try:
        if not source.exists() or not source.is_dir():
            return
        target.mkdir(parents=True, exist_ok=True)
        for item in source.rglob("*"):
            rel = item.relative_to(source)
            dst = target / rel
            if item.is_dir():
                dst.mkdir(parents=True, exist_ok=True)
            elif item.is_file() and not dst.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(item, dst)
    except Exception:
        pass


def _set_dotted(payload: dict, dotted_path: str, value) -> None:
    cursor = payload
    parts = [part for part in str(dotted_path).split(".") if part]
    for part in parts[:-1]:
        next_value = cursor.get(part)
        if not isinstance(next_value, dict):
            next_value = {}
            cursor[part] = next_value
        cursor = next_value
    if parts:
        cursor[parts[-1]] = value


def _hash_password(password: str) -> str:
    salt = os.urandom(16)
    digest = hashlib.pbkdf2_hmac("sha256", str(password).encode("utf-8"), salt, _PBKDF2_ITERATIONS)
    return "pbkdf2_sha256${}${}${}".format(
        _PBKDF2_ITERATIONS,
        base64.b64encode(salt).decode("ascii"),
        base64.b64encode(digest).decode("ascii"),
    )


def _verify_password(password: str, encoded: str) -> bool:
    try:
        scheme, iterations_raw, salt_raw, digest_raw = str(encoded).split("$", 3)
        if scheme != "pbkdf2_sha256":
            return False
        iterations = int(iterations_raw)
        salt = base64.b64decode(salt_raw.encode("ascii"))
        expected = base64.b64decode(digest_raw.encode("ascii"))
        actual = hashlib.pbkdf2_hmac("sha256", str(password).encode("utf-8"), salt, iterations)
        return hmac.compare_digest(actual, expected)
    except Exception:
        return False
