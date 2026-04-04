from __future__ import annotations

from modules.voice.stt import STTConfig, STTResult, STTSegment, STTService, transcribe, transcribe_result
from modules.voice.tts import AudioChunk, TTSConfig, TTSService, speak, synthesize
from modules.voice.voice_manager import VoiceManager, VoiceState

__all__ = [
    "TTSConfig",
    "AudioChunk",
    "TTSService",
    "synthesize",
    "speak",
    "STTConfig",
    "STTSegment",
    "STTResult",
    "STTService",
    "transcribe",
    "transcribe_result",
    "VoiceState",
    "VoiceManager",
]
