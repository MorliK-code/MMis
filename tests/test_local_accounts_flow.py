from __future__ import annotations

from pathlib import Path

from core.account_manager import AccountManager, prepare_local_account
from core.chat_store import ChatStore
from memory_core.account_paths import memory_paths_for_account


def test_account_manager_creates_account_context(tmp_path, monkeypatch):
    monkeypatch.setenv("MMIS_ACCOUNTS_DIR", str(tmp_path / "accounts"))
    manager = AccountManager(tmp_path / "accounts")

    account = manager.create_account("Alice", "Alice")
    context = account.to_context()

    assert account.account_id == "alice"
    assert context.config_path.exists()
    assert context.performance_profiles_path.exists()
    assert context.memory_db_path.parent == tmp_path / "accounts" / "alice" / "memory_core"
    assert memory_paths_for_account(context) == {
        "db_path": str(context.memory_db_path),
        "vector_path": str(context.memory_vector_path),
    }


def test_prepare_local_account_updates_runtime_paths(tmp_path, monkeypatch):
    monkeypatch.setenv("MMIS_ACCOUNTS_DIR", str(tmp_path / "accounts"))
    monkeypatch.delenv("MMIS_ACTIVE_ACCOUNT_ID", raising=False)

    class Settings:
        accounts_dir = tmp_path / "accounts"
        account_id = ""
        memory_dir = Path("unused")
        cache_dir = Path("unused")
        db_path = Path("unused")
        memory_core_db_path = ""
        memory_core_vector_path = ""
        memory_core_default_workspace = "global"

    settings = Settings()
    context = prepare_local_account(settings)

    assert context.account_id == "default"
    assert context.memory_db_path == context.data_dir / "memory_core" / "memory.db"
    assert context.memory_vector_path == context.data_dir / "memory_core" / "vector"
    assert context.config_path.exists()
    assert context.performance_profiles_path.exists()


def test_prepare_local_account_seeds_performance_profiles_for_all_accounts(tmp_path, monkeypatch):
    monkeypatch.setenv("MMIS_ACCOUNTS_DIR", str(tmp_path / "accounts"))
    manager = AccountManager(tmp_path / "accounts")
    admin = manager.create_account("Admin", "Admin")
    user = manager.create_account("User", "User")
    admin.to_context().performance_profiles_path.unlink()
    user.to_context().performance_profiles_path.unlink()

    class Settings:
        accounts_dir = tmp_path / "accounts"
        account_id = admin.account_id

    prepare_local_account(Settings())

    assert admin.to_context().performance_profiles_path.exists()
    assert user.to_context().performance_profiles_path.exists()


def test_settings_loads_account_performance_profiles(tmp_path):
    import json

    from config.settings import _settings_from_payload

    account_root = tmp_path / "accounts" / "alice"
    profiles_path = account_root / "specs" / "performance_profiles.json"
    profiles_path.parent.mkdir(parents=True)
    profiles_path.write_text(
        json.dumps(
            {
                "profiles": {
                    "BALANCED": {
                        "generation": {"temperature": 0.33},
                        "ollama": {"num_ctx": 1234, "keep_alive": "account"},
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    payload = {
        "paths": {
            "accounts_dir": str(tmp_path / "accounts"),
            "performance_profiles_file": str(profiles_path),
        },
        "account": {"account_id": "alice"},
        "memory_core": {"memory_dir": str(account_root / "memory_core")},
    }

    settings = _settings_from_payload(payload, config_file=account_root / "config" / "config.json")

    assert settings.performance_profiles_file == profiles_path.resolve()
    assert settings.llm_profiles["BALANCED"]["ollama"]["num_ctx"] == 1234


def test_chat_store_round_trip(tmp_path):
    store = ChatStore(tmp_path / "account" / "chats" / "chats.db")

    chat = store.upsert_chat("chat-1", "Main", persona_id="asya")
    store.replace_messages(
        chat.chat_id,
        [
            {"role": "user", "text": "hello", "metadata": {"account_id": "default"}},
            {"role": "assistant", "text": "hi", "metadata": {"persona_id": "asya"}},
        ],
    )

    assert store.list_chats()[0].chat_id == "chat-1"
    messages = store.list_messages("chat-1")
    assert [message.role for message in messages] == ["user", "assistant"]
    assert messages[0].metadata["account_id"] == "default"


def test_chat_store_archives_chats_except_active(tmp_path):
    store = ChatStore(tmp_path / "account" / "chats" / "chats.db")
    store.upsert_chat("visible-main-chat", "Main")
    store.upsert_chat("old-chat", "Old")

    store.archive_chats_except({"visible-main-chat"})

    assert [chat.chat_id for chat in store.list_chats()] == ["visible-main-chat"]
    assert {chat.chat_id for chat in store.list_chats(include_archived=True)} == {"visible-main-chat", "old-chat"}


def test_legacy_directory_migration_copies_into_existing_account_dirs(tmp_path, monkeypatch):
    import core.account_manager as account_manager_module

    data_dir = tmp_path / "data"
    legacy_vector = data_dir / "memory_core" / "vector"
    legacy_cache = data_dir / "cache"
    legacy_vector.mkdir(parents=True)
    legacy_cache.mkdir(parents=True)
    (legacy_vector / "index.bin").write_text("vector", encoding="utf-8")
    (legacy_cache / "cache.json").write_text("cache", encoding="utf-8")

    monkeypatch.setattr(account_manager_module, "DATA_DIR", data_dir)
    monkeypatch.setattr(account_manager_module, "BASE_DIR", tmp_path)

    manager = AccountManager(data_dir / "accounts")
    account = manager.create_account("Default", "Default")
    context = account.to_context()

    assert (context.memory_vector_path / "index.bin").read_text(encoding="utf-8") == "vector"
    assert (context.data_dir / "cache" / "cache.json").read_text(encoding="utf-8") == "cache"


def test_new_accounts_start_with_only_seed_characters_after_first_migration(tmp_path, monkeypatch):
    import core.account_manager as account_manager_module

    data_dir = tmp_path / "data"
    for cid in ("default", "asya", "custom"):
        char_dir = data_dir / "specs" / "characters" / cid
        char_dir.mkdir(parents=True)
        (char_dir / "character.json").write_text(f'{{"id": "{cid}"}}', encoding="utf-8")

    monkeypatch.setattr(account_manager_module, "DATA_DIR", data_dir)
    monkeypatch.setattr(account_manager_module, "BASE_DIR", tmp_path)

    manager = AccountManager(data_dir / "accounts")
    first = manager.create_account("first", "First")
    second = manager.create_account("second", "Second")

    first_chars = {path.name for path in (first.to_context().data_dir / "specs" / "characters").iterdir() if path.is_dir()}
    second_chars = {path.name for path in (second.to_context().data_dir / "specs" / "characters").iterdir() if path.is_dir()}

    assert {"default", "asya", "custom"}.issubset(first_chars)
    assert second_chars == {"default", "asya"}
