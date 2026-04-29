from __future__ import annotations

from core.account_context import AccountContext


def memory_paths_for_account(account: AccountContext) -> dict[str, str]:
    return {
        "db_path": str(account.memory_db_path),
        "vector_path": str(account.memory_vector_path),
    }
