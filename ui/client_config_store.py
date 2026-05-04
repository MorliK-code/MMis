import json
import os
import re
from pathlib import Path
from typing import Any

UI_DIR = Path(__file__).parent
PROJECT_ROOT = UI_DIR.parent
LEGACY_CLIENT_DATA_DIR = UI_DIR / ".mmis_client"


def _default_client_data_dir() -> Path:
    portable = str(os.environ.get("MMIS_UI_PORTABLE") or "").strip().lower()
    if portable in {"1", "true", "yes", "on"}:
        return LEGACY_CLIENT_DATA_DIR
    override = str(os.environ.get("MMIS_UI_DATA_DIR") or "").strip()
    if override:
        return Path(override).expanduser()
    if os.name == "nt":
        root = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA")
        if root:
            return Path(root) / "MMis" / "ui_client"
    return Path.home() / ".mmis" / "ui_client"


CLIENT_DATA_DIR = _default_client_data_dir()
ACCOUNTS_DIR = CLIENT_DATA_DIR / "accounts"
CLIENT_CONFIG_PATH = CLIENT_DATA_DIR / "client_config.json"
ACCOUNT_DATA_DIR = None


def _safe_account_id(value: str) -> str:
    token = str(value or "").strip()
    token = re.sub(r"[^A-Za-z0-9_.-]+", "-", token)
    token = token.strip("._-")
    return token[:80] or "guest"


def _active_auth_account_id() -> str:
    auth_path = CLIENT_DATA_DIR / "auth.json"
    try:
        payload = json.loads(auth_path.read_text(encoding="utf-8-sig") or "{}")
    except Exception:
        return ""
    if not isinstance(payload, dict):
        return ""
    if not str(payload.get("token") or "").strip():
        return ""
    return str(payload.get("account_id") or "").strip()


def get_active_account_data_dir() -> Path | None:
    account_id = _safe_account_id(_active_auth_account_id())
    if not account_id or account_id == "guest":
        return None
    return ACCOUNTS_DIR / account_id


def get_client_data_dir() -> Path:
    account_dir = get_active_account_data_dir()
    path = account_dir / "ui" if account_dir is not None else CLIENT_DATA_DIR
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_client_config_path() -> Path:
    return get_client_data_dir() / "client_config.json"


def get_account_client_config_path(account_id: str) -> Path:
    safe = _safe_account_id(account_id)
    return ACCOUNTS_DIR / safe / "ui" / "client_config.json"


def _client_config_path() -> Path:
    return get_client_config_path()


def get_ui_state_path() -> Path:
    account_dir = get_active_account_data_dir()
    if account_dir is not None:
        return account_dir / "ui_state.json"
    return get_client_data_dir() / "ui_state.json"

DEFAULT_CONFIG = {
    "connection": {
        "active_endpoint": "local",
        "local_base_url": "http://127.0.0.1:8027",
        "public_base_url": "",
        "api_access_key": "",
    },
    "values": {},
    "server_snapshot": {},
    "pending_updates": {},
    "last_sync_at": None,
    "last_error": None
}

def load_client_config() -> dict[str, Any]:
    """Loads the client configuration from the local JSON file."""
    path = _client_config_path()
    if not path.exists():
        return save_client_config(DEFAULT_CONFIG)
    
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
            # Ensure basic structure exists
            for key, default in DEFAULT_CONFIG.items():
                if key not in data:
                    data[key] = default
            if not isinstance(data.get("connection"), dict):
                data["connection"] = dict(DEFAULT_CONFIG["connection"])
            else:
                for key, value in DEFAULT_CONFIG["connection"].items():
                    data["connection"].setdefault(key, value)
            return data
    except Exception:
        return DEFAULT_CONFIG

def save_client_config(data: dict[str, Any]) -> dict[str, Any]:
    """Saves the client configuration to the local JSON file."""
    try:
        client_dir = get_client_data_dir()
        path = _client_config_path()
        client_dir.mkdir(parents=True, exist_ok=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception:
        pass
    return data


def _load_client_config_at(path: Path) -> dict[str, Any]:
    if not path.exists():
        return dict(DEFAULT_CONFIG)
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig") or "{}")
        if not isinstance(data, dict):
            return dict(DEFAULT_CONFIG)
        out = dict(DEFAULT_CONFIG)
        out.update(data)
        conn = out.get("connection")
        if not isinstance(conn, dict):
            out["connection"] = dict(DEFAULT_CONFIG["connection"])
        else:
            merged = dict(DEFAULT_CONFIG["connection"])
            merged.update(conn)
            out["connection"] = merged
        for key in ("values", "server_snapshot", "pending_updates"):
            if not isinstance(out.get(key), dict):
                out[key] = {}
        return out
    except Exception:
        return dict(DEFAULT_CONFIG)


def save_account_client_config_updates(
    account_id: str,
    *,
    connection: dict[str, Any] | None = None,
    values: dict[str, Any] | None = None,
) -> None:
    safe = _safe_account_id(account_id)
    if not safe or safe == "guest":
        return
    path = get_account_client_config_path(safe)
    cfg = _load_client_config_at(path)
    if isinstance(connection, dict):
        cfg.setdefault("connection", {}).update(connection)
    if isinstance(values, dict):
        cfg.setdefault("values", {}).update(values)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")

def get_connection_config() -> dict[str, Any]:
    """Returns the connection-related part of the config."""
    cfg = load_client_config()
    conn = cfg.get("connection", DEFAULT_CONFIG["connection"])
    if not isinstance(conn, dict):
        conn = {}
    out = dict(DEFAULT_CONFIG["connection"])
    out.update(conn)
    return out


def get_api_access_key() -> str:
    conn = get_connection_config()
    return str(conn.get("api_access_key") or "").strip()

def update_connection_config(values: dict[str, Any]) -> None:
    """Updates only the connection-related settings."""
    cfg = load_client_config()
    cfg.setdefault("connection", {}).update(values)
    save_client_config(cfg)

def _normalize_url(url: str) -> str:
    if not url: return ""
    url = url.strip().rstrip("/")
    if not (url.startswith("http://") or url.startswith("https://")):
        url = "http://" + url
    return url

def get_selected_base_url() -> str:
    """Returns the currently active API base URL."""
    # Environment variable has highest priority for portable/debug modes
    env_url = os.environ.get("MMIS_API_URL")
    if env_url:
        return _normalize_url(env_url)

    conn = get_connection_config()
    active = str(conn.get("active_endpoint", "local")).lower()
    if active == "public":
        return _normalize_url(str(conn.get("public_base_url") or ""))
    return _normalize_url(str(conn.get("local_base_url") or "http://127.0.0.1:8027"))

def merge_local_values(updates: dict[str, Any]) -> None:
    """Merges new values into the local values cache."""
    cfg = load_client_config()
    cfg.setdefault("values", {}).update(updates)
    save_client_config(cfg)

def merge_pending_updates(updates: dict[str, Any]) -> None:
    """Adds updates to the pending queue for server synchronization."""
    cfg = load_client_config()
    cfg.setdefault("pending_updates", {}).update(updates)
    save_client_config(cfg)

def clear_pending_updates(paths: list[str] | None = None) -> None:
    """Clears pending updates, optionally only for specific paths."""
    cfg = load_client_config()
    pending = cfg.get("pending_updates", {})
    if paths is None:
        cfg["pending_updates"] = {}
    else:
        for path in paths:
            pending.pop(path, None)
    save_client_config(cfg)

def save_server_snapshot(payload: dict[str, Any]) -> None:
    """Saves a snapshot of the server configuration and updates sync time."""
    import time
    cfg = load_client_config()
    cfg["server_snapshot"] = payload
    cfg["last_sync_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    cfg["last_error"] = None
    save_client_config(cfg)

def set_last_error(error: str | None) -> None:
    """Logs the last synchronization error."""
    cfg = load_client_config()
    cfg["last_error"] = error
    save_client_config(cfg)


# ---------------------------------------------------------------------------
# Local UI state.
# ---------------------------------------------------------------------------

UI_STATE_PATH = CLIENT_DATA_DIR / "ui_state.json"


def _ui_state_path() -> Path:
    return get_ui_state_path()

DEFAULT_UI_STATE: dict[str, Any] = {
    "think_enabled": True,
    "verbose_enabled": False,
    "json_mode_enabled": False,
    "screen_enabled": False,
    "web_mode": "auto",
    "active_topic_title": "",
    "last_persona_name": "Default",
    "ollama_models_cache": [],
}


def load_ui_state() -> dict[str, Any]:
    """Loads local UI state from this machine's UI data directory."""
    path = _ui_state_path()
    if not path.exists():
        return dict(DEFAULT_UI_STATE)

    try:
        with path.open("r", encoding="utf-8-sig") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return dict(DEFAULT_UI_STATE)
        out = dict(DEFAULT_UI_STATE)
        out.update(data)
        return out
    except Exception:
        return dict(DEFAULT_UI_STATE)


def save_ui_state(data: dict[str, Any]) -> dict[str, Any]:
    """Saves local UI state to this machine's UI data directory."""
    out = dict(DEFAULT_UI_STATE)
    if isinstance(data, dict):
        out.update(data)

    try:
        client_dir = get_client_data_dir()
        path = _ui_state_path()
        client_dir.mkdir(parents=True, exist_ok=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
    except Exception:
        pass
    return out


def merge_ui_state(updates: dict[str, Any]) -> dict[str, Any]:
    """Merges updates into the current UI state and persists."""
    state = load_ui_state()
    if isinstance(updates, dict):
        state.update(updates)
    return save_ui_state(state)


def get_last_persona_name() -> str:
    """Returns the last known persona name, defaulting to 'Default'."""
    state = load_ui_state()
    name = str(state.get("last_persona_name") or "").strip()
    return name or "Default"


def set_last_persona_name(name: str) -> None:
    """Persists the last known persona name to the portable UI state."""
    clean = str(name or "").strip()
    if not clean:
        return
    merge_ui_state({"last_persona_name": clean})


def get_ollama_models_cache() -> list[str]:
    """Returns last known Ollama model list from portable UI state."""
    state = load_ui_state()
    raw = state.get("ollama_models_cache") or []
    out: list[str] = []
    seen: set[str] = set()
    for item in raw:
        name = str(item or "").strip()
        if not name or name in seen:
            continue
        seen.add(name)
        out.append(name)
    return out


def set_ollama_models_cache(models: list[str]) -> None:
    """Persists last known Ollama model list to portable UI state."""
    out: list[str] = []
    seen: set[str] = set()
    for item in models or []:
        name = str(item or "").strip()
        if not name or name in seen:
            continue
        seen.add(name)
        out.append(name)
    merge_ui_state({"ollama_models_cache": out})
