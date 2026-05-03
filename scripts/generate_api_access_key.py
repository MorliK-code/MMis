from __future__ import annotations

import argparse
import hashlib
import secrets


def _make_key(prefix: str, token_bytes: int) -> str:
    clean_prefix = str(prefix or "").strip()
    if not clean_prefix:
        clean_prefix = "mmis_"
    return clean_prefix + secrets.token_urlsafe(max(16, int(token_bytes)))


def _sha256(value: str) -> str:
    return hashlib.sha256(str(value or "").encode("utf-8")).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate an MMis API access key and its SHA256 hash.",
    )
    parser.add_argument(
        "--prefix",
        default="mmis_",
        help="Raw key prefix. Default: mmis_",
    )
    parser.add_argument(
        "--bytes",
        type=int,
        default=48,
        help="Random token byte length before URL-safe encoding. Default: 48",
    )
    parser.add_argument(
        "--key",
        default="",
        help="Hash an existing raw key instead of generating a new one.",
    )
    args = parser.parse_args()

    key = str(args.key or "").strip() or _make_key(args.prefix, args.bytes)
    digest = _sha256(key)

    print("MMis API access key")
    print()
    print("CLIENT / Desktop UI -> Main config -> Подключение к API -> API access key")
    print(f"KEY={key}")
    print()
    print("SERVER / Main config -> API -> API access key SHA256")
    print(f"SHA256={digest}")
    print()
    print("PowerShell test header:")
    print(f'curl -H "X-MMis-Access-Key: {key}" http://127.0.0.1:8027/health')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
