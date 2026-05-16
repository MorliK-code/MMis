import logging
from typing import Any

from ui.api_client import ApiClient, ApiClientError
from ui.client_config_store import (
    clear_pending_updates,
    get_selected_base_url,
    load_client_config,
    merge_local_values,
    merge_pending_updates,
    save_server_snapshot,
    update_connection_config,
)
from ui.settings_schema import SETTINGS_CATEGORIES, dotted_get

LOGGER = logging.getLogger(__name__)

def dotted_set(payload: dict, path: str, value: Any):
    """Sets a value in a nested dictionary using a dotted path."""
    parts = path.split(".")
    current = payload
    for part in parts[:-1]:
        current = current.setdefault(part, {})
    current[parts[-1]] = value

def default_value_for_path(path: str, kind: str):
    """Provides default values for settings based on their path and type."""
    if path == "ui.api.active_endpoint":
        return "local"
    if path == "ui.api.local_base_url":
        return "http://127.0.0.1:8027"
    if path == "ui.api.public_base_url":
        return ""
    if path == "ui.api.api_access_key":
        return ""
    if path == "ui.ollama.start_mode":
        return "serve"
    if path == "ui.ollama.serve_exe":
        return ""
    if path == "ui.ollama.models_dir":
        return ""
    if path == "llm.providers.ollama.keep_alive":
        return "5m"
    if path == "memory_core.memory_llm.keep_alive":
        return "30m"
    
    if kind in {"text", "select", "json"}:
        return ""
    if kind in {"int", "float"}:
        return ""
    if kind == "bool":
        return False
    return ""

def _is_client_only_path(path: str) -> bool:
    path = str(path or "")
    return path.startswith("ui.api.") or path.startswith("ui.ollama.")

def _load_project_config_snapshot() -> dict:
    """Loads the actual config snapshot from the project's config manager."""
    try:
        from config.settings import get_config_payload
        payload = get_config_payload(force_reload=True)
        return payload if isinstance(payload, dict) else {}
    except Exception as exc:
        LOGGER.debug("Failed to load local project config snapshot: %s", exc)
        return {}

def resolve_value(spec, local_config: dict, project_snapshot: dict | None = None, *, prefer_server_snapshot: bool = False) -> Any:
    """Resolves the value for a specific setting based on priority."""
    path = spec.path
    pending = local_config.get("pending_updates", {})
    values = local_config.get("values", {})
    snapshot = local_config.get("server_snapshot", {})
    
    # 1. Pending updates (not yet synced to server)
    if path in pending:
        return pending[path]
    
    # 2. Special handling for connection settings (stored in separate block)
    if path.startswith("ui.api."):
        conn = local_config.get("connection", {})
        key = path.replace("ui.api.", "")
        if key in conn:
            return conn[key]

    # 3. Client-only values live only in the UI cache.
    if _is_client_only_path(path) and path in values:
        return values[path]
    
    # 4. Fresh server snapshot, only when this load successfully reached API.
    if prefer_server_snapshot:
        val = dotted_get(snapshot, path)
        if val is not None:
            return val

    # 5. Local project/account config (current on-disk source of truth).
    val = dotted_get(project_snapshot or {}, path)
    if val is not None:
        return val

    # 6. Stale server snapshot as a fallback only.
    val = dotted_get(snapshot, path)
    if val is not None:
        return val

    # 7. Local values are only a fallback for paths missing from real config.
    if path in values:
        return values[path]

    # 8. Fallback (schema defaults)
    return default_value_for_path(path, spec.kind)

def build_payload_from_schema(local_config: dict, project_snapshot: dict | None = None, *, prefer_server_snapshot: bool = False) -> dict:
    """Constructs the full settings payload using the schema and local config."""
    payload = {}
    for category in SETTINGS_CATEGORIES:
        for card in category.cards:
            for spec in card.settings:
                value = resolve_value(spec, local_config, project_snapshot, prefer_server_snapshot=prefer_server_snapshot)
                dotted_set(payload, spec.path, value)
    return payload

def load_settings_payload(
    api_client: ApiClient | None = None,
    *,
    allow_remote: bool = False,
) -> tuple[dict, dict]:
    """
    Loads the settings payload, optionally syncing with the API.
    Returns (payload, meta).
    """
    local_cfg = load_client_config()
    project_snapshot = _load_project_config_snapshot()
    payload = build_payload_from_schema(local_cfg, project_snapshot, prefer_server_snapshot=False)
    
    meta = {
        "online": False,
        "source": "cache" if local_cfg.get("server_snapshot") else "project",
        "pending_count": len(local_cfg.get("pending_updates", {})),
        "last_error": local_cfg.get("last_error")
    }
    
    if not allow_remote:
        return payload, meta

    client = api_client or ApiClient()
    
    # Try to fetch from server
    try:
        # GET /config is expected to return the full nested config
        server_config = client._request_json("GET", "/config", timeout=1.5)
        if server_config:
            save_server_snapshot(server_config)
            # Re-build payload with new snapshot
            local_cfg = load_client_config()
            payload = build_payload_from_schema(local_cfg, project_snapshot, prefer_server_snapshot=True)
            meta["online"] = True
            meta["source"] = "server"
            meta["pending_count"] = len(local_cfg.get("pending_updates", {}))
    except Exception as e:
        LOGGER.debug("Server sync failed (offline mode): %s", e)
        meta["online"] = False
        meta["source"] = "cache" if local_cfg.get("server_snapshot") else "project"
        meta["last_error"] = str(e)

    return payload, meta

def save_settings_updates(
    updates: dict, 
    api_client: ApiClient | None = None,
    *,
    sync_remote: bool = False,
) -> tuple[bool, str]:
    """
    Saves setting updates locally and optionally tries to sync them with the API.
    Returns (online, message).
    """
    if not updates:
        return True, "No changes"
        
    # 1. Always save locally first (instantly)
    merge_local_values(updates)
    merge_pending_updates(updates)
    
    # 2. Handle connection updates immediately
    conn_updates = {}
    for path, val in updates.items():
        if str(path).startswith("ui.api."):
            key = str(path).replace("ui.api.", "")
            conn_updates[key] = val
    
    if conn_updates:
        update_connection_config(conn_updates)
        
    # 3. If no remote sync requested, stop here
    if not sync_remote:
        return False, "Saved locally"

    # 4. Filter out UI-only settings before sending to server
    remote_updates = {
        path: value
        for path, value in updates.items()
        if not (
            str(path).startswith("ui.api.")
            or str(path).startswith("ui.ollama.")
        )
    }

    if not remote_updates:
        clear_pending_updates(list(updates.keys()))
        return True, "Local UI settings saved"

    client = api_client or ApiClient()
    
    try:
        LOGGER.info("settings sync target=%s updates=%s", client.base_url, list(remote_updates.keys()))
        client._request_json("PATCH", "/config", payload=remote_updates, timeout=5.0)
        clear_pending_updates(list(updates.keys()))
        return True, "Synced with server"
    except Exception as e:
        LOGGER.warning("Settings sync failed: %s", e)
        return False, str(e)
