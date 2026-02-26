from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


class STTEngine(Protocol):
    def transcribe_file(self, audio_path: str | Path) -> str:
        """Convert recorded audio file to text."""


class TTSEngine(Protocol):
    def synthesize_to_file(self, text: str, output_path: str | Path) -> Path:
        """Convert text reply to audio file."""


@dataclass
class VoiceModule:
    stt: STTEngine
    tts: TTSEngine

    def speech_to_text(self, audio_path: str | Path) -> str:
        return (self.stt.transcribe_file(audio_path) or "").strip()

    def text_to_speech(self, text: str, output_path: str | Path) -> Path:
        return self.tts.synthesize_to_file((text or "").strip(), output_path)
