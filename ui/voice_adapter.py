from __future__ import annotations

import shutil
from pathlib import Path

from ui.settings_sync_service import load_settings_payload
from modules.voice.stt import STTConfig, STTService
from modules.voice.tts import TTSConfig, TTSService
from modules.voice.voice_manager import VoiceManager


def _percent_to_speed(value: str, default: float = 1.0) -> float:
    raw = str(value or "").strip().replace("%", "")
    if not raw:
        return float(default)
    try:
        pct = float(raw)
    except Exception:
        return float(default)
    speed = 1.0 + (pct / 100.0)
    return max(0.4, min(2.0, speed))


class _STTEngineAdapter:
    def __init__(self):
        self._svc = STTService()

    def transcribe_file(self, path: str | Path) -> str:
        voice = _voice_runtime_config()
        cfg = STTConfig(
            vad=True,
            engine=str(voice.get("stt_engine") or "auto"),
            model=str(voice.get("stt_model") or "small"),
            device=str(voice.get("stt_device") or "default"),
            compute_type=str(voice.get("stt_compute_type") or "default"),
            language_hint=str(voice.get("stt_language_hint") or ""),
        )
        result = self._svc.transcribe(audio=str(path), config=cfg)
        return str(result.text or "").strip()


class _TTSEngineAdapter:
    def __init__(self, *, voice: str, rate: str, volume: str):
        self._svc = TTSService()
        self._voice = str(voice or "default").strip() or "default"
        self._speed = _percent_to_speed(rate, default=1.0)
        self._volume = str(volume or "").strip()  # reserved for future engines

    def synthesize_to_file(self, text: str, out_path: str | Path) -> Path:
        _ = self._volume
        voice_cfg = _voice_runtime_config()
        dst = Path(out_path).expanduser()
        dst.parent.mkdir(parents=True, exist_ok=True)

        cfg = TTSConfig(
            voice=self._voice,
            speed=self._speed,
            cache_enabled=True,
            engine=str(voice_cfg.get("tts_engine") or "auto"),
            model=str(voice_cfg.get("tts_model") or "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"),
            device=str(voice_cfg.get("tts_device") or "default"),
        )
        audio = self._svc.synthesize(text=str(text or ""), lang="", config=cfg)
        src = Path(audio.path).expanduser()

        if src.resolve() != dst.resolve():
            shutil.copy2(src, dst)
        return dst


def build_stt_engine():
    return _STTEngineAdapter()


def build_tts_engine(*, tts_voice: str, tts_rate: str, tts_volume: str):
    return _TTSEngineAdapter(voice=tts_voice, rate=tts_rate, volume=tts_volume)


def build_voice_manager() -> VoiceManager:
    return VoiceManager(tts=TTSService(), stt=STTService())


def build_stt_config() -> STTConfig:
    voice = _voice_runtime_config()
    return STTConfig(
        vad=True,
        engine=str(voice.get("stt_engine") or "auto"),
        model=str(voice.get("stt_model") or "small"),
        device=str(voice.get("stt_device") or "default"),
        compute_type=str(voice.get("stt_compute_type") or "default"),
        language_hint=str(voice.get("stt_language_hint") or ""),
    )


def build_tts_config(*, tts_voice: str = "", tts_rate: str = "") -> TTSConfig:
    voice = _voice_runtime_config()
    speed = _percent_to_speed(tts_rate, default=1.0)
    return TTSConfig(
        voice=str(tts_voice or voice.get("tts_voice") or "default"),
        speed=speed,
        engine=str(voice.get("tts_engine") or "auto"),
        model=str(voice.get("tts_model") or "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"),
        device=str(voice.get("tts_device") or "default"),
        cache_enabled=True,
    )


def _voice_runtime_config() -> dict:
    try:
        payload, _ = load_settings_payload()
        voice = dict(payload.get("voice") or {})
    except Exception:
        voice = {}
    return voice

