from __future__ import annotations

import argparse
import hashlib
import secrets
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api.auth_store import AuthStore


def _make_key(prefix: str, token_bytes: int) -> str:
    clean_prefix = str(prefix or "").strip() or "mmis_"
    return clean_prefix + secrets.token_urlsafe(max(16, int(token_bytes)))


def _sha256(value: str) -> str:
    return hashlib.sha256(str(value or "").encode("utf-8")).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate or set a per-account MMis API access key.",
    )
    parser.add_argument("login", help="Account login, for example: admin")
    parser.add_argument("--key", default="", help="Use an existing raw key instead of generating a new one.")
    parser.add_argument("--label", default="default", help="Human label for this key hash. Default: default")
    parser.add_argument("--prefix", default="mmis_", help="Generated raw key prefix. Default: mmis_")
    parser.add_argument("--bytes", type=int, default=48, help="Random token byte length. Default: 48")
    parser.add_argument("--clear", action="store_true", help="Clear this account's API access key hash.")
    parser.add_argument("--list", action="store_true", help="List stored API access key hashes. Use login 'all' for all accounts.")
    args = parser.parse_args()

    login = str(args.login or "").strip().lower()
    if not login:
        raise SystemExit("login is required")

    store = AuthStore()
    if args.list:
        rows = store.list_api_access_keys("" if login == "all" else login)
        if not rows:
            print("No API access key hashes stored.")
            return 0
        for row in rows:
            status = "revoked" if row["revoked"] else "active"
            key = str(row.get("key") or "")
            key_text = key if key else "<raw key not stored>"
            print(f"{row['login']}\t{status}\t{row['label']}\t{row['key_hash']}\t{key_text}")
        return 0

    if args.clear:
        count = store.clear_api_access_keys(login)
        print(f"Cleared API access keys for account: {login} ({count} revoked)")
        return 0

    key = str(args.key or "").strip() or _make_key(args.prefix, args.bytes)
    digest = _sha256(key)
    account = store.set_api_access_key_hash(login, digest, label=args.label, raw_key=key)

    print("MMis per-account API access key")
    print()
    print(f"ACCOUNT={account['login']}")
    print()
    print("CLIENT / Desktop UI for this account -> Main config -> Подключение к API -> API access key")
    print(f"KEY={key}")
    print()
    print("SERVER / stored in auth.db for this account")
    print(f"SHA256={digest}")
    print()
    print("PowerShell test header:")
    print(f'curl -H "X-MMis-Access-Key: {key}" http://127.0.0.1:8027/health')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
