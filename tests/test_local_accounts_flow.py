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


def test_server_auth_store_sessions_round_trip(tmp_path):
    from api.auth_store import AuthStore

    store = AuthStore(tmp_path / "auth.db")
    account = store.create_account("MorliK", "pass1234", "MorliK")
    token = store.create_session(account["account_id"])

    assert account["login"] == "morlik"
    assert store.verify_login("morlik", "bad") is None
    assert store.verify_login("MorliK", "pass1234")["account_id"] == account["account_id"]
    assert store.get_account_by_token(token)["display_name"] == "MorliK"

    store.revoke_token(token)

    assert store.get_account_by_token(token) is None


def test_server_auth_can_claim_legacy_admin_without_password(tmp_path, monkeypatch):
    import sqlite3

    import api.auth_store as auth_store_module
    from api.auth_store import AuthStore

    legacy_db = tmp_path / "accounts.db"
    with sqlite3.connect(str(legacy_db)) as db:
        db.execute(
            """
            CREATE TABLE accounts (
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
        db.execute(
            """
            INSERT INTO accounts(account_id, login, display_name, password_hash, created_at, data_path)
            VALUES('admin', 'admin', 'Admin', '', 1.0, 'unused')
            """
        )

    monkeypatch.setattr(auth_store_module, "LEGACY_ACCOUNTS_DB_PATH", legacy_db)
    store = AuthStore(tmp_path / "auth.db")

    account = store.create_account("admin", "pass1234", "")

    assert account["account_id"] == "admin"
    assert store.verify_login("admin", "pass1234")["account_id"] == "admin"

    account = store.create_account("admin", "newpass123", "")

    assert account["account_id"] == "admin"
    assert store.verify_login("admin", "pass1234") is None
    assert store.verify_login("admin", "newpass123")["account_id"] == "admin"


def test_ui_auth_state_is_root_client_state(tmp_path, monkeypatch):
    import ui.client_config_store as client_config_store
    import ui.auth_client_store as auth_client_store

    root = tmp_path / ".mmis_client"
    monkeypatch.setattr(client_config_store, "CLIENT_DATA_DIR", root)
    monkeypatch.setattr(client_config_store, "CLIENT_CONFIG_PATH", root / "client_config.json")
    monkeypatch.setattr(client_config_store, "UI_STATE_PATH", root / "ui_state.json")
    monkeypatch.setattr(auth_client_store, "AUTH_STATE_PATH", root / "auth.json")

    auth_client_store.save_auth_state(
        {
            "token": "tok",
            "account_id": "acc_1",
            "login": "morlik",
            "display_name": "MorliK",
        }
    )

    assert auth_client_store.get_auth_token() == "tok"
    assert auth_client_store.get_auth_display_name() == "MorliK"
    assert (root / "auth.json").exists()

    auth_client_store.clear_auth_state()

    assert auth_client_store.get_auth_token() == ""


def test_ui_auth_state_keeps_switchable_sessions(tmp_path, monkeypatch):
    import ui.client_config_store as client_config_store
    import ui.auth_client_store as auth_client_store

    root = tmp_path / ".mmis_client"
    monkeypatch.setattr(client_config_store, "CLIENT_DATA_DIR", root)
    monkeypatch.setattr(client_config_store, "CLIENT_CONFIG_PATH", root / "client_config.json")
    monkeypatch.setattr(client_config_store, "UI_STATE_PATH", root / "ui_state.json")
    monkeypatch.setattr(auth_client_store, "AUTH_STATE_PATH", root / "auth.json")

    auth_client_store.save_auth_state({"token": "admin-token", "account_id": "admin", "login": "admin"})
    auth_client_store.save_auth_state({"token": "user-token", "account_id": "user", "login": "user"})

    assert {row["account_id"] for row in auth_client_store.list_auth_sessions()} == {"admin", "user"}

    auth_client_store.activate_auth_session("admin")

    assert auth_client_store.load_auth_state()["account_id"] == "admin"
    assert auth_client_store.get_auth_token() == "admin-token"

    auth_client_store.clear_auth_state(forget_current=True)

    assert auth_client_store.get_auth_token() == ""
    assert [row["account_id"] for row in auth_client_store.list_auth_sessions()] == ["user"]


def test_ui_auth_state_preserves_legacy_current_session_when_logging_into_next_account(tmp_path, monkeypatch):
    import json

    import ui.client_config_store as client_config_store
    import ui.auth_client_store as auth_client_store

    root = tmp_path / ".mmis_client"
    monkeypatch.setattr(client_config_store, "CLIENT_DATA_DIR", root)
    monkeypatch.setattr(client_config_store, "CLIENT_CONFIG_PATH", root / "client_config.json")
    monkeypatch.setattr(client_config_store, "UI_STATE_PATH", root / "ui_state.json")
    monkeypatch.setattr(auth_client_store, "AUTH_STATE_PATH", root / "auth.json")

    root.mkdir(parents=True)
    (root / "auth.json").write_text(
        json.dumps({"token": "admin-token", "account_id": "admin", "login": "admin"}),
        encoding="utf-8",
    )

    auth_client_store.save_auth_state({"token": "user-token", "account_id": "user", "login": "user"})

    assert {row["account_id"] for row in auth_client_store.list_auth_sessions()} == {"admin", "user"}


def test_ui_client_data_dir_follows_auth_account(tmp_path, monkeypatch):
    import ui.auth_client_store as auth_client_store
    import ui.client_config_store as client_config_store

    root = tmp_path / ".mmis_client"
    project_root = tmp_path / "project"
    accounts_dir = project_root / "data" / "accounts"
    monkeypatch.setattr(client_config_store, "CLIENT_DATA_DIR", root)
    monkeypatch.setattr(client_config_store, "LEGACY_CLIENT_DATA_DIR", root)
    monkeypatch.setattr(client_config_store, "PROJECT_ROOT", project_root)
    monkeypatch.setattr(client_config_store, "ACCOUNTS_DIR", accounts_dir)
    monkeypatch.setattr(client_config_store, "CLIENT_CONFIG_PATH", root / "client_config.json")
    monkeypatch.setattr(client_config_store, "UI_STATE_PATH", root / "ui_state.json")
    monkeypatch.setattr(auth_client_store, "AUTH_STATE_PATH", root / "auth.json")

    assert client_config_store.get_client_data_dir() == root

    auth_client_store.save_auth_state({"token": "a", "account_id": "admin", "login": "admin"})
    assert client_config_store.get_active_account_data_dir() == accounts_dir / "admin"
    assert client_config_store.get_client_data_dir() == accounts_dir / "admin" / "ui"

    client_config_store.save_ui_state({"last_persona_name": "AdminPersona"})

    auth_client_store.save_auth_state({"token": "b", "account_id": "user", "login": "user"})
    assert client_config_store.load_ui_state()["last_persona_name"] == "Default"

    auth_client_store.save_auth_state({"token": "a", "account_id": "admin", "login": "admin"})
    assert client_config_store.load_ui_state()["last_persona_name"] == "AdminPersona"
    assert (accounts_dir / "admin" / "ui_state.json").exists()


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
