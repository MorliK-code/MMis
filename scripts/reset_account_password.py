from __future__ import annotations

import argparse
import getpass
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api.auth_store import AuthStore


def main() -> int:
    parser = argparse.ArgumentParser(description="Reset an MMis account password locally.")
    parser.add_argument("login", help="Account login, for example: admin")
    parser.add_argument("--password", default="", help="New password. If omitted, prompt securely.")
    args = parser.parse_args()

    login = str(args.login or "").strip().lower()
    if not login:
        raise SystemExit("login is required")

    password = str(args.password or "")
    if not password:
        password = getpass.getpass(f"New password for {login}: ")
        confirm = getpass.getpass("Confirm password: ")
        if password != confirm:
            raise SystemExit("passwords do not match")

    if len(password) < 4:
        raise SystemExit("password must be at least 4 characters")

    store = AuthStore()
    account = store.set_password(login, password)
    print(f"Password reset for account: {account['login']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
