"""Background workers used by the desktop UI."""

from __future__ import annotations

import traceback
from dataclasses import dataclass

from PySide6.QtCore import QThread, Signal

from ui.api_client import ApiClient, ApiClientError


@dataclass
class ReplyResult:
    text: str
    stats: dict
    thinking: str = ""
    thinking_generated: bool = False
    model: str = ""
    debug_trace: dict | None = None
    memory_debug_snapshot: dict | None = None


class ReplyWorker(QThread):
    finished = Signal(object)
    errored = Signal(str)
    chunk = Signal(str)
    thinking_chunk = Signal(str)
    debug_event = Signal(object)

    def __init__(
        self,
        api: ApiClient,
        user_text: str,
        store_turn: bool = True,
        think: bool | None = None,
        verbose: bool | None = None,
        json_mode: bool | None = None,
        attachments: list[dict] | None = None,
    ):
        super().__init__()
        self.api = api
        self.user_text = user_text
        self.store_turn = bool(store_turn)
        self.think = think
        self.verbose = verbose
        self.json_mode = json_mode
        self.attachments = [dict(item) for item in list(attachments or []) if isinstance(item, dict)]
        self._cancel_requested = False

    def request_cancel(self):
        self._cancel_requested = True

    def _emit_stream_piece(self, piece: str, signal: Signal) -> None:
        text = str(piece or "")
        signal.emit(text)

    def run(self):
        payload = {
            "api_ok": False,
            "model": "",
            "model_status": {},
            "memory_status": {},
            "server_resources": {},
            "error": "",
        }
        try:
            health = None
            try:
                ping_resp = self.api.ping(timeout=1.2)
                payload["api_ok"] = str(ping_resp.get("status") or "").strip().lower() == "ok"
            except Exception:
                # fallback for older server versions without /ping
                health = self.api.health(timeout=5.0)
                payload["api_ok"] = str(health.get("status") or "").strip().lower() == "ok"

            if payload["api_ok"]:
                try:
                    if health is None:
                        health = self.api.health(timeout=8.0)
                    
                    payload["model"] = str(health.get("model") or self.api.get_runtime_model() or "")
                    payload["thinking_enabled"] = bool(health.get("thinking_enabled", False))
                    payload["verbose_enabled"] = bool(health.get("verbose_enabled", False))
                    payload["json_mode_enabled"] = bool(health.get("json_mode_enabled", False))
                    payload["web_mode"] = str(health.get("web_mode") or "")
                    if "persona_name" in health:
                        payload["persona_name"] = str(health.get("persona_name") or "").strip()
                    payload["model_status"] = dict(health.get("model_status") or {})
                    payload["memory_status"] = dict(health.get("memory_status") or {})
                    payload["server_resources"] = dict(health.get("server_resources") or {})
                except Exception as exc:
                    payload["error"] = f"health details unavailable: {exc}"
        except ApiClientError as exc:
            self.errored.emit(str(exc))
        except Exception as exc:
            msg = str(exc or "").strip()
            self.errored.emit(msg if msg else traceback.format_exc())


class StatusPollWorker(QThread):
    status_ready = Signal(object)

    def __init__(self, api: ApiClient):
        super().__init__()
        self.api = api

    def run(self):
        payload = {
            "api_ok": False,
            "model": "",
            "model_status": {},
            "memory_status": {},
            "server_resources": {},
            "error": "",
        }
        try:
            health = None
            try:
                ping_resp = self.api.ping(timeout=1.2)
                payload["api_ok"] = str(ping_resp.get("status") or "").strip().lower() == "ok"
            except Exception:
                # fallback for older server versions without /ping
                health = self.api.health(timeout=5.0)
                payload["api_ok"] = str(health.get("status") or "").strip().lower() == "ok"

            if payload["api_ok"]:
                try:
                    if health is None:
                        health = self.api.health(timeout=8.0)
                    
                    payload["model"] = str(health.get("model") or self.api.get_runtime_model() or "")
                    payload["thinking_enabled"] = bool(health.get("thinking_enabled", False))
                    payload["verbose_enabled"] = bool(health.get("verbose_enabled", False))
                    payload["json_mode_enabled"] = bool(health.get("json_mode_enabled", False))
                    payload["web_mode"] = str(health.get("web_mode") or "")
                    if "persona_name" in health:
                        payload["persona_name"] = str(health.get("persona_name") or "").strip()
                    payload["model_status"] = dict(health.get("model_status") or {})
                    payload["memory_status"] = dict(health.get("memory_status") or {})
                    payload["server_resources"] = dict(health.get("server_resources") or {})
                except Exception as exc:
                    payload["error"] = f"health details unavailable: {exc}"
        except ApiClientError as exc:
            payload["error"] = str(exc)
        except Exception as exc:
            payload["error"] = str(exc or "").strip() or traceback.format_exc()
        self.status_ready.emit(payload)
