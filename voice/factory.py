from __future__ import annotations

import logging

from config import (
    MMIS_VOICE_LANGUAGE,
    MMIS_VOICE_STT_BACKEND,
    MMIS_VOICE_STT_COMPUTE_TYPE,
    MMIS_VOICE_STT_DEVICE,
    MMIS_VOICE_STT_MODEL,
    MMIS_VOICE_TTS_BACKEND,
    MMIS_VOICE_TTS_RATE,
    MMIS_VOICE_TTS_VOICE,
    MMIS_VOICE_TTS_VOLUME,
)

from .base import VoiceModule
from .dummy import DummySTT, DummyTTS

logger = logging.getLogger(__name__)


def build_voice_module() -> VoiceModule:
    stt = build_stt_engine()
    tts = build_tts_engine()
    return VoiceModule(stt=stt, tts=tts)


def build_stt_engine():
    backend = MMIS_VOICE_STT_BACKEND
    if backend == "dummy":
        return DummySTT()
    if backend == "faster_whisper":
        try:
            from .engines import FasterWhisperSTT

            return FasterWhisperSTT(
                model_size=MMIS_VOICE_STT_MODEL,
                device=MMIS_VOICE_STT_DEVICE,
                compute_type=MMIS_VOICE_STT_COMPUTE_TYPE,
                language=MMIS_VOICE_LANGUAGE,
            )
        except Exception as exc:
            logger.warning("Failed to init faster_whisper STT, fallback to dummy: %s", exc)
            return DummySTT()

    logger.warning("Unknown STT backend '%s', fallback to dummy", backend)
    return DummySTT()


def build_tts_engine(
    tts_voice: str | None = None,
    tts_rate: str | None = None,
    tts_volume: str | None = None,
):
    voice = (tts_voice if tts_voice is not None else MMIS_VOICE_TTS_VOICE)
    rate = (tts_rate if tts_rate is not None else MMIS_VOICE_TTS_RATE)
    volume = (tts_volume if tts_volume is not None else MMIS_VOICE_TTS_VOLUME)
    backend = MMIS_VOICE_TTS_BACKEND
    if backend == "dummy":
        return DummyTTS()
    if backend == "pyttsx3":
        try:
            from .engines import Pyttsx3TTSEngine

            return Pyttsx3TTSEngine(
                rate=rate,
                volume=volume,
                language_hint=MMIS_VOICE_LANGUAGE,
            )
        except Exception as exc:
            logger.warning("Failed to init pyttsx3 TTS, fallback to dummy: %s", exc)
            return DummyTTS()
    if backend == "edge_tts":
        try:
            from .engines import EdgeTTSEngine

            return EdgeTTSEngine(
                voice=voice,
                rate=rate,
                volume=volume,
            )
        except Exception as exc:
            logger.warning("Failed to init edge_tts, trying pyttsx3: %s", exc)
            try:
                from .engines import Pyttsx3TTSEngine

                return Pyttsx3TTSEngine(
                    rate=rate,
                    volume=volume,
                    language_hint=MMIS_VOICE_LANGUAGE,
                )
            except Exception as pyttsx_exc:
                logger.warning("Failed to init pyttsx3 TTS, fallback to dummy: %s", pyttsx_exc)
                return DummyTTS()

    logger.warning("Unknown TTS backend '%s', fallback to dummy", backend)
    return DummyTTS()
