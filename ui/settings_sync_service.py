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
    
    if kind in {"text", "select", "json"}:
        return ""
    if kind in {"int", "float"}:
        return ""
    if kind == "bool":
        return False
    return ""

def resolve_value(spec, local_config: dict) -> Any:
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

    # 3. Local values (cached from UI changes)
    if path in values:
        return values[path]
    
    # 4. Server snapshot (last known values from API)
    val = dotted_get(snapshot, path)
    if val is not None:
        return val
        
    # 5. Default UI values for connection
    return default_value_for_path(path, spec.kind)

def build_payload_from_schema(local_config: dict) -> dict:
    """Constructs the full settings payload using the schema and local config."""
    payload = {}
    for category in SETTINGS_CATEGORIES:
        for card in category.cards:
            for spec in card.settings:
                value = resolve_value(spec, local_config)
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
    payload = build_payload_from_schema(local_cfg)
    
    meta = {
        "online": False,
        "source": "cache" if local_cfg.get("server_snapshot") else "schema",
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
            payload = build_payload_from_schema(local_cfg)
            meta["online"] = True
            meta["source"] = "server"
            meta["pending_count"] = len(local_cfg.get("pending_updates", {}))
    except Exception as e:
        LOGGER.debug("Server sync failed (offline mode): %s", e)
        meta["online"] = False
        meta["source"] = "cache" if local_cfg.get("server_snapshot") else "schema"
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
        if not str(path).startswith("ui.api.")
    }

    if not remote_updates:
        clear_pending_updates(list(updates.keys()))
        return True, "Local UI settings saved"

    client = api_client or ApiClient()
    
    try:
        # Try to send to server
        client._request_json("PATCH", "/config", payload=remote_updates, timeout=1.0)
        clear_pending_updates(list(remote_updates.keys()))
        return True, "Synced with server"
    except Exception as e:
        LOGGER.warning("Offline save: %s", e)
        return False, str(e)
