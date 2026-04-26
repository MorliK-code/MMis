import json
import os
from pathlib import Path
from typing import Any

# Path to the local config file relative to the ui/ directory
# It should be inside ui/.mmis_client/
UI_DIR = Path(__file__).parent
CLIENT_DATA_DIR = UI_DIR / ".mmis_client"
CLIENT_CONFIG_PATH = CLIENT_DATA_DIR / "client_config.json"

DEFAULT_CONFIG = {
    "connection": {
        "active_endpoint": "local",
        "local_base_url": "http://127.0.0.1:8027",
        "public_base_url": ""
    },
    "values": {},
    "server_snapshot": {},
    "pending_updates": {},
    "last_sync_at": None,
    "last_error": None
}

def load_client_config() -> dict[str, Any]:
    """Loads the client configuration from the local JSON file."""
    if not CLIENT_CONFIG_PATH.exists():
        return save_client_config(DEFAULT_CONFIG)
    
    try:
        with CLIENT_CONFIG_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
            # Ensure basic structure exists
            for key, default in DEFAULT_CONFIG.items():
                if key not in data:
                    data[key] = default
            return data
    except Exception:
        return DEFAULT_CONFIG

def save_client_config(data: dict[str, Any]) -> dict[str, Any]:
    """Saves the client configuration to the local JSON file."""
    try:
        CLIENT_DATA_DIR.mkdir(parents=True, exist_ok=True)
        with CLIENT_CONFIG_PATH.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception:
        pass
    return data

def get_connection_config() -> dict[str, Any]:
    """Returns the connection-related part of the config."""
    cfg = load_client_config()
    return cfg.get("connection", DEFAULT_CONFIG["connection"])

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
# Portable UI state (ui/.mmis_client/ui_state.json)
# ---------------------------------------------------------------------------

UI_STATE_PATH = CLIENT_DATA_DIR / "ui_state.json"

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
    """Loads local portable UI state from ui/.mmis_client/ui_state.json."""
    if not UI_STATE_PATH.exists():
        return dict(DEFAULT_UI_STATE)

    try:
        with UI_STATE_PATH.open("r", encoding="utf-8-sig") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return dict(DEFAULT_UI_STATE)
        out = dict(DEFAULT_UI_STATE)
        out.update(data)
        return out
    except Exception:
        return dict(DEFAULT_UI_STATE)


def save_ui_state(data: dict[str, Any]) -> dict[str, Any]:
    """Saves local portable UI state to ui/.mmis_client/ui_state.json."""
    out = dict(DEFAULT_UI_STATE)
    if isinstance(data, dict):
        out.update(data)

    try:
        CLIENT_DATA_DIR.mkdir(parents=True, exist_ok=True)
        with UI_STATE_PATH.open("w", encoding="utf-8") as f:
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
