from __future__ import annotations

import shutil
from pathlib import Path

from modules.voice.stt import STTConfig, STTService
from modules.voice.tts import TTSConfig, TTSService


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
        cfg = STTConfig(vad=True)
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
        dst = Path(out_path).expanduser()
        dst.parent.mkdir(parents=True, exist_ok=True)

        cfg = TTSConfig(voice=self._voice, speed=self._speed, cache_enabled=True)
        audio = self._svc.synthesize(text=str(text or ""), lang="", config=cfg)
        src = Path(audio.path).expanduser()

        if src.resolve() != dst.resolve():
            shutil.copy2(src, dst)
        return dst


def build_stt_engine():
    return _STTEngineAdapter()


def build_tts_engine(*, tts_voice: str, tts_rate: str, tts_volume: str):
    return _TTSEngineAdapter(voice=tts_voice, rate=tts_rate, volume=tts_volume)

