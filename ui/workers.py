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
        try:
            from ui.settings_sync_service import load_settings_payload, dotted_get
            from utils.ollama_runtime import ensure_ollama_started

            settings_payload, _meta = load_settings_payload(allow_remote=False)
            provider = str(dotted_get(settings_payload, "llm.provider", "ollama") or "ollama").lower()
            auto_start = bool(dotted_get(settings_payload, "ui.console.auto_start_ollama", True))

            if provider in {"ollama", "auto"}:
                ensure_ollama_started(
                    base_url=str(dotted_get(settings_payload, "llm.providers.ollama.base_url", "http://127.0.0.1:11434") or "http://127.0.0.1:11434"),
                    enabled=auto_start,
                    start_mode=str(dotted_get(settings_payload, "ui.ollama.start_mode", "serve") or "serve"),
                    serve_exe=str(dotted_get(settings_payload, "ui.ollama.serve_exe", "") or ""),
                    models_dir=str(dotted_get(settings_payload, "ui.ollama.models_dir", "") or ""),
                    wait_sec=8.0,
                )

            reply = self.api.stream_chat(
                self.user_text,
                store_turn=self.store_turn,
                think=self.think,
                verbose=self.verbose,
                json_mode=self.json_mode,
                attachments=self.attachments,
                on_chunk=lambda piece: self._emit_stream_piece(piece, self.chunk),
                on_thinking_chunk=lambda piece: self._emit_stream_piece(piece, self.thinking_chunk),
                on_debug_event=lambda event: self.debug_event.emit(event),
                cancel_requested=lambda: self._cancel_requested,
            )

            self.finished.emit(
                ReplyResult(
                    text=reply.answer,
                    stats=reply.stats or {},
                    thinking=reply.thinking or "",
                    thinking_generated=bool(reply.thinking_generated),
                    model=reply.model or "",
                    debug_trace=reply.debug_trace,
                    memory_debug_snapshot=reply.memory_debug_snapshot,
                )
            )
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


class CharacterListWorker(QThread):
    """Фоновый поток для получения списка персонажей."""
    finished = Signal(object)
    errored = Signal(str)

    def __init__(self, api: ApiClient):
        super().__init__()
        self.api = api

    def run(self):
        try:
            manifest = self.api.list_characters()
            active = self.api.get_active_character()
            self.finished.emit({
                "manifest": manifest,
                "active": active
            })
        except Exception as e:
            self.errored.emit(str(e))


class CharacterActionWorker(QThread):
    """Фоновый поток для выполнения действий с персонажами (создание, удаление, смена активного)."""
    finished = Signal(object)
    errored = Signal(str)

    def __init__(self, api: ApiClient, action: str, **kwargs):
        super().__init__()
        self.api = api
        self.action = action
        self.kwargs = kwargs

    def run(self):
        try:
            result = None
            if self.action == "get":
                result = self.api.get_character(self.kwargs.get("id"))
            elif self.action == "set_active":
                result = self.api.set_active_character(self.kwargs.get("id"))
            elif self.action == "create":
                result = self.api.create_character(
                    self.kwargs.get("id"), 
                    self.kwargs.get("name"),
                    llm_profile=self.kwargs.get("llm_profile"),
                    default_mood=self.kwargs.get("default_mood")
                )
            elif self.action == "update":
                result = self.api.update_character(self.kwargs.get("id"), self.kwargs.get("updates"))
            elif self.action == "delete":
                result = self.api.delete_character(self.kwargs.get("id"))
            
            self.finished.emit(result)
        except Exception as e:
            self.errored.emit(str(e))
