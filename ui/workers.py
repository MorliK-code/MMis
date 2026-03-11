"""Background workers used by the desktop UI."""

from __future__ import annotations

import traceback
from dataclasses import dataclass

from PySide6.QtCore import QObject, Signal, Slot

from ui.api_client import ApiClient, ApiClientError


@dataclass
class ReplyResult:
    text: str
    stats: dict
    thinking: str = ""
    model: str = ""


class ReplyWorker(QObject):
    finished = Signal(object)
    errored = Signal(str)
    chunk = Signal(str)
    thinking_chunk = Signal(str)

    def __init__(
        self,
        api: ApiClient,
        user_text: str,
        store_turn: bool = True,
        think: bool | None = None,
    ):
        super().__init__()
        self.api = api
        self.user_text = user_text
        self.store_turn = bool(store_turn)
        self.think = think
        self._cancel_requested = False

    def request_cancel(self):
        self._cancel_requested = True

    @Slot()
    def run(self):
        try:
            reply = self.api.stream_chat(
                text=self.user_text,
                store_turn=self.store_turn,
                think=self.think,
                on_chunk=self.chunk.emit,
                on_thinking_chunk=self.thinking_chunk.emit,
                cancel_requested=lambda: self._cancel_requested,
            )
            if self._cancel_requested:
                return
            self.finished.emit(
                ReplyResult(
                    text=reply.answer,
                    stats=reply.stats,
                    thinking=reply.thinking,
                    model=reply.model,
                )
            )
        except ApiClientError as exc:
            self.errored.emit(str(exc))
        except Exception as exc:
            msg = str(exc or "").strip()
            self.errored.emit(msg if msg else traceback.format_exc())


