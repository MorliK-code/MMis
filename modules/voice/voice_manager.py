from __future__ import annotations

import queue
import threading
import time
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Callable

from modules.voice.stt import STTConfig, STTResult, STTService, transcribe_result
from modules.voice.tts import AudioChunk, TTSConfig, TTSService
from utils.logger import get_logger


LOGGER = get_logger(__name__)


class VoiceState(str, Enum):
    IDLE = "IDLE"
    LISTENING = "LISTENING"
    PROCESSING = "PROCESSING"
    SPEAKING = "SPEAKING"


@dataclass(frozen=True)
class VoiceTask:
    text: str
    lang: str
    config: TTSConfig


class VoiceManager:
    """Coordinates STT/TTS lifecycle without making product decisions."""

    def __init__(
        self,
        *,
        tts: TTSService | None = None,
        stt: STTService | None = None,
        on_partial: Callable[[str], None] | None = None,
        on_final: Callable[[str], None] | None = None,
        on_state: Callable[[VoiceState], None] | None = None,
    ):
        self.tts = tts or TTSService()
        self.stt = stt or STTService()
        self._on_partial = on_partial
        self._on_final = on_final
        self._on_state = on_state

        self._state = VoiceState.IDLE
        self._state_lock = threading.RLock()
        self._tts_queue: queue.Queue[VoiceTask] = queue.Queue()
        self._worker_stop = threading.Event()
        self._stt_paused = False

        self._worker = threading.Thread(target=self._worker_loop, name="voice-tts-worker", daemon=True)
        self._worker.start()

    def set_callbacks(
        self,
        *,
        on_partial: Callable[[str], None] | None = None,
        on_final: Callable[[str], None] | None = None,
        on_state: Callable[[VoiceState], None] | None = None,
    ) -> None:
        if on_partial is not None:
            self._on_partial = on_partial
        if on_final is not None:
            self._on_final = on_final
        if on_state is not None:
            self._on_state = on_state

    def get_state(self) -> VoiceState:
        with self._state_lock:
            return self._state

    def start_listening(self) -> VoiceState:
        with self._state_lock:
            if self._state == VoiceState.SPEAKING:
                self.barge_in()
            return self._set_state_locked(VoiceState.LISTENING)

    def stop_listening(self) -> VoiceState:
        with self._state_lock:
            if self._state == VoiceState.LISTENING:
                self._set_state_locked(VoiceState.IDLE)
            return self._state

    def barge_in(self) -> None:
        """Stop TTS playback if user starts speaking over assistant audio."""
        with self._state_lock:
            if self._state == VoiceState.SPEAKING:
                self.tts.stop()
                _drain_queue(self._tts_queue)
                self._stt_paused = False
                self._set_state_locked(VoiceState.LISTENING)

    def enqueue_speak(self, text: str, *, lang: str = "", config: TTSConfig | None = None) -> None:
        payload = str(text or "").strip()
        if not payload:
            return
        self._tts_queue.put(VoiceTask(text=payload, lang=lang, config=config or TTSConfig()))

    def speak(self, text: str, *, lang: str = "", config: TTSConfig | None = None) -> AudioChunk:
        payload = str(text or "").strip()
        if not payload:
            raise ValueError("speak expects non-empty text")

        with self._state_lock:
            self._set_state_locked(VoiceState.SPEAKING)
            self._stt_paused = True

        try:
            audio = self.tts.speak(payload, lang=lang, config=config or TTSConfig())
            return audio
        finally:
            with self._state_lock:
                self._stt_paused = False
                self._set_state_locked(VoiceState.IDLE)

    def transcribe(self, audio: str | bytes, *, config: STTConfig | None = None) -> STTResult:
        if self._stt_paused:
            return STTResult(text="", conf=0.0, metadata={"paused": True})

        with self._state_lock:
            self._set_state_locked(VoiceState.PROCESSING)

        try:
            result = self.stt.transcribe(audio=audio, config=config or STTConfig())
            if result.text:
                self._emit_final(result.text)
            return result
        finally:
            with self._state_lock:
                self._set_state_locked(VoiceState.IDLE)

    def on_partial(self, text: str) -> None:
        payload = str(text or "").strip()
        if not payload:
            return

        if self.get_state() == VoiceState.SPEAKING:
            self.barge_in()
        self._emit_partial(payload)

    def on_final(self, text: str) -> None:
        payload = str(text or "").strip()
        if not payload:
            return

        if self.get_state() == VoiceState.SPEAKING:
            self.barge_in()
        self._emit_final(payload)

    def shutdown(self) -> None:
        self._worker_stop.set()
        _drain_queue(self._tts_queue)
        try:
            self._worker.join(timeout=1.0)
        except Exception:
            pass
        with self._state_lock:
            self._set_state_locked(VoiceState.IDLE)
            self._stt_paused = False

    def _worker_loop(self) -> None:
        while not self._worker_stop.is_set():
            try:
                task = self._tts_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            try:
                self.speak(text=task.text, lang=task.lang, config=task.config)
            except Exception as exc:
                LOGGER.warning("voice queue speak failed: %s", exc)
            finally:
                self._tts_queue.task_done()

    def _emit_partial(self, text: str) -> None:
        if callable(self._on_partial):
            try:
                self._on_partial(text)
            except Exception as exc:
                LOGGER.warning("on_partial callback failed: %s", exc)

    def _emit_final(self, text: str) -> None:
        if callable(self._on_final):
            try:
                self._on_final(text)
            except Exception as exc:
                LOGGER.warning("on_final callback failed: %s", exc)

    def _set_state_locked(self, state: VoiceState) -> VoiceState:
        if self._state == state:
            return self._state
        self._state = state
        callback = self._on_state
        if callable(callback):
            try:
                callback(state)
            except Exception as exc:
                LOGGER.warning("on_state callback failed: %s", exc)
        return self._state


def _drain_queue(q: queue.Queue) -> None:
    while True:
        try:
            q.get_nowait()
            q.task_done()
        except queue.Empty:
            break


# Backward-compatible helper style API.
_DEFAULT_MANAGER: VoiceManager | None = None


def _get_default_manager() -> VoiceManager:
    global _DEFAULT_MANAGER
    if _DEFAULT_MANAGER is None:
        _DEFAULT_MANAGER = VoiceManager()
    return _DEFAULT_MANAGER


def speak(text: str) -> None:
    _get_default_manager().enqueue_speak(text)


def transcribe(path: str) -> str:
    return transcribe_result(path).text
