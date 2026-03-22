"""Background workers used by the desktop UI."""

from __future__ import annotations

import re
import time
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
    debug_trace: dict | None = None
    memory_debug_snapshot: dict | None = None


def _split_stream_display_piece(text: str, *, max_chars: int = 12) -> list[str]:
    src = str(text or "")
    if not src:
        return []
    tokens = re.findall(r"\S+\s*|\s+", src, flags=re.UNICODE)
    if not tokens:
        return [src]

    out: list[str] = []
    carry = ""
    limit = max(1, int(max_chars))

    def _flush_carry() -> None:
        nonlocal carry
        if carry:
            out.append(carry)
            carry = ""

    for token in tokens:
        if len(token) > limit:
            _flush_carry()
            for start in range(0, len(token), limit):
                out.append(token[start : start + limit])
            continue
        if carry and (len(carry) + len(token)) > limit:
            _flush_carry()
        carry += token

    _flush_carry()
    return out or [src]


class ReplyWorker(QObject):
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
    ):
        super().__init__()
        self.api = api
        self.user_text = user_text
        self.store_turn = bool(store_turn)
        self.think = think
        self._cancel_requested = False
        # A tiny pause keeps split display pieces visually progressive instead
        # of collapsing a large provider chunk into one instant repaint.
        self._stream_emit_pause_sec = 0.012
        self._stream_emit_chunk_chars = 12

    def request_cancel(self):
        self._cancel_requested = True

    def _emit_stream_piece(self, piece: str, signal: Signal) -> None:
        text = str(piece or "")
        if not text:
            signal.emit("")
            return

        parts = _split_stream_display_piece(text, max_chars=self._stream_emit_chunk_chars)
        if len(parts) <= 1:
            signal.emit(text)
            return

        pause_s = max(0.0, float(self._stream_emit_pause_sec or 0.0))
        last_index = len(parts) - 1
        for index, part in enumerate(parts):
            if self._cancel_requested:
                return
            signal.emit(part)
            if pause_s > 0.0 and index < last_index:
                time.sleep(pause_s)
            elif index < last_index:
                time.sleep(0)

    @Slot()
    def run(self):
        try:
            reply = self.api.stream_chat(
                text=self.user_text,
                store_turn=self.store_turn,
                think=self.think,
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


