from __future__ import annotations

import argparse
import shutil
import sqlite3
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api.auth_store import AuthStore
from config.settings import DATA_DIR


def _norm_login(value: str) -> str:
    return str(value or "").strip().lower()


def _legacy_account(login: str) -> dict[str, str] | None:
    db_path = Path(DATA_DIR) / "accounts" / "accounts.db"
    if not db_path.exists():
        return None
    try:
        with sqlite3.connect(str(db_path)) as db:
            db.row_factory = sqlite3.Row
            row = db.execute(
                "SELECT account_id, login, data_path FROM accounts WHERE login=?",
                (_norm_login(login),),
            ).fetchone()
    except Exception:
        return None
    if row is None:
        return None
    return {
        "account_id": str(row["account_id"] or ""),
        "login": str(row["login"] or ""),
        "data_path": str(row["data_path"] or ""),
    }


def _disable_legacy_account(login: str) -> dict[str, str] | None:
    db_path = Path(DATA_DIR) / "accounts" / "accounts.db"
    account = _legacy_account(login)
    if account is None or not db_path.exists():
        return account
    try:
        with sqlite3.connect(str(db_path)) as db:
            db.execute("UPDATE accounts SET is_active=0 WHERE login=?", (_norm_login(login),))
            db.commit()
    except Exception:
        pass
    return account


def _safe_remove_account_dir(path_value: str) -> bool:
    if not path_value:
        return False
    target = Path(path_value).expanduser().resolve()
    accounts_root = (Path(DATA_DIR) / "accounts").expanduser().resolve()
    try:
        target.relative_to(accounts_root)
    except ValueError:
        raise SystemExit(f"refusing to delete path outside accounts dir: {target}")
    if target == accounts_root or not target.exists():
        return False
    shutil.rmtree(target)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Delete or disable an MMis account locally.")
    parser.add_argument("login", help="Account login to delete, for example: oleg")
    parser.add_argument("--yes", action="store_true", help="Do not ask for confirmation.")
    parser.add_argument("--keep-data", action="store_true", help="Keep data/accounts/<account_id> directory.")
    parser.add_argument("--hard", action="store_true", help="Physically delete auth.db row instead of disabling it.")
    args = parser.parse_args()

    login = _norm_login(args.login)
    if not login:
        raise SystemExit("login is required")

    store = AuthStore()
    legacy = _legacy_account(login)
    if not args.yes:
        print(f"Account to delete: {login}")
        print("This will revoke sessions and API keys, disable the account in auth.db and accounts.db.")
        if not args.keep_data:
            print("It will also delete the account data directory.")
        answer = input("Type DELETE to continue: ").strip()
        if answer != "DELETE":
            print("Cancelled.")
            return 1

    account = store.delete_account(login, soft=not bool(args.hard))
    legacy = _disable_legacy_account(login) or legacy

    removed_data = False
    if not args.keep_data:
        data_path = str((legacy or {}).get("data_path") or "").strip()
        if not data_path:
            account_id = str(account.get("account_id") or "").strip()
            data_path = str((Path(DATA_DIR) / "accounts" / account_id).resolve()) if account_id else ""
        removed_data = _safe_remove_account_dir(data_path)

    print(f"Deleted account: {account['login']} ({account['account_id']})")
    print(f"auth_db_mode={'hard-delete' if args.hard else 'disabled'}")
    print(f"data_dir_removed={str(bool(removed_data)).lower()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
