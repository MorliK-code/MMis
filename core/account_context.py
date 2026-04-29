from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AccountContext:
    account_id: str
    login: str
    display_name: str
    data_dir: Path
    config_path: Path
    memory_db_path: Path
    memory_vector_path: Path
    chats_db_path: Path
    performance_profiles_path: Path

    @classmethod
    def from_data_dir(
        cls,
        *,
        account_id: str,
        login: str,
        display_name: str,
        data_dir: str | Path,
    ) -> "AccountContext":
        root = Path(data_dir).expanduser().resolve()
        return cls(
            account_id=str(account_id),
            login=str(login),
            display_name=str(display_name),
            data_dir=root,
            config_path=root / "config" / "config.json",
            memory_db_path=root / "memory_core" / "memory.db",
            memory_vector_path=root / "memory_core" / "vector",
            chats_db_path=root / "chats" / "chats.db",
            performance_profiles_path=root / "specs" / "performance_profiles.json",
        )

    def ensure_dirs(self) -> None:
        for path in (
            self.config_path.parent,
            self.memory_db_path.parent,
            self.memory_vector_path,
            self.chats_db_path.parent,
            self.data_dir / "attachments",
            self.data_dir / "cache",
            self.data_dir / "state",
            self.data_dir / "specs",
            self.data_dir / "specs" / "characters",
        ):
            path.mkdir(parents=True, exist_ok=True)
