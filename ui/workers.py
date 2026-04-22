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
        attachments: list[dict] | None = None,
    ):
        super().__init__()
        self.api = api
        self.user_text = user_text
        self.store_turn = bool(store_turn)
        self.think = think
        self.verbose = verbose
        self.attachments = [dict(item) for item in list(attachments or []) if isinstance(item, dict)]
        self._cancel_requested = False

    def request_cancel(self):
        self._cancel_requested = True

    def _emit_stream_piece(self, piece: str, signal: Signal) -> None:
        text = str(piece or "")
        signal.emit(text)

    def run(self):
        try:
            reply = self.api.stream_chat(
                text=self.user_text,
                store_turn=self.store_turn,
                think=self.think,
                verbose=self.verbose,
                attachments=self.attachments,
                on_chunk=lambda piece: self._emit_stream_piece(piece, self.chunk),
                on_thinking_chunk=lambda piece: self._emit_stream_piece(piece, self.thinking_chunk),
                on_debug_event=self.debug_event.emit,
                cancel_requested=lambda: self._cancel_requested,
            )
            if self._cancel_requested:
                return
            self.finished.emit(
                ReplyResult(
                    text=reply.answer,
                    stats=reply.stats,
                    thinking=reply.thinking,
                    thinking_generated=bool(reply.thinking_generated),
                    model=reply.model,
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
            "error": "",
        }
        try:
            health = self.api.health()
            payload["api_ok"] = str(health.get("status") or "").strip().lower() == "ok"
            payload["model"] = str(health.get("model") or self.api.get_runtime_model() or "")
            payload["model_status"] = dict(health.get("model_status") or {})
            payload["memory_status"] = dict(health.get("memory_status") or {})
        except ApiClientError as exc:
            payload["error"] = str(exc)
        except Exception as exc:
            payload["error"] = str(exc or "").strip() or traceback.format_exc()
        self.status_ready.emit(payload)
