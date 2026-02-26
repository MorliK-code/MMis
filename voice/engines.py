from __future__ import annotations

import asyncio
import time
from pathlib import Path


class FasterWhisperSTT:
    def __init__(
        self,
        model_size: str = "small",
        device: str = "auto",
        compute_type: str = "int8",
        language: str = "ru",
        beam_size: int = 5,
    ):
        from faster_whisper import WhisperModel

        self._whisper_model_cls = WhisperModel
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
        self.language = (language or "ru").strip()
        self.beam_size = int(beam_size)

    def _reload_model(self, device: str) -> None:
        self.device = device
        self.model = self._whisper_model_cls(self.model_size, device=device, compute_type=self.compute_type)

    def transcribe_file(self, audio_path: str | Path) -> str:
        path = Path(audio_path)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")

        def _run_transcribe():
            return self.model.transcribe(
                str(path),
                language=self.language,
                task="transcribe",
                beam_size=self.beam_size,
                vad_filter=True,
            )

        try:
            segments, _info = _run_transcribe()
        except RuntimeError as exc:
            msg = str(exc).lower()
            if ("cublas" in msg or "cuda" in msg) and self.device != "cpu":
                self._reload_model("cpu")
                segments, _info = _run_transcribe()
            else:
                raise
        text = " ".join((seg.text or "").strip() for seg in segments).strip()
        return text


class EdgeTTSEngine:
    def __init__(
        self,
        voice: str,
        rate: str = "+0%",
        volume: str = "+0%",
        max_retries: int = 2,
        retry_delay_sec: float = 0.6,
    ):
        self.voice = voice
        self.rate = rate
        self.volume = volume
        self.max_retries = max(0, int(max_retries))
        self.retry_delay_sec = max(0.0, float(retry_delay_sec))

    async def _save(self, text: str, out_path: Path) -> None:
        import edge_tts

        communicate = edge_tts.Communicate(
            text=text,
            voice=self.voice,
            rate=self.rate,
            volume=self.volume,
        )
        await communicate.save(str(out_path))

    def synthesize_to_file(self, text: str, output_path: str | Path) -> Path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        text = (text or "").strip()
        if not text:
            out.write_bytes(b"")
            return out

        last_exc = None
        for attempt in range(self.max_retries + 1):
            loop = asyncio.new_event_loop()
            try:
                loop.run_until_complete(self._save(text, out))
                return out
            except Exception as exc:
                last_exc = exc
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay_sec)
            finally:
                loop.close()
        if last_exc is not None:
            raise last_exc
        return out


class Pyttsx3TTSEngine:
    def __init__(self, rate: str = "+0%", volume: str = "+0%", language_hint: str = "ru"):
        self._rate_delta = self._parse_rate_delta(rate)
        self._volume = self._parse_volume(volume)
        self._language_hint = (language_hint or "ru").strip().lower()

    @staticmethod
    def _parse_rate_delta(rate: str) -> int:
        s = str(rate or "").strip()
        if not s.endswith("%"):
            return 0
        try:
            return int(float(s[:-1]) * 1.5)
        except Exception:
            return 0

    @staticmethod
    def _parse_volume(volume: str) -> float:
        s = str(volume or "").strip()
        if not s.endswith("%"):
            return 1.0
        try:
            v = 1.0 + (float(s[:-1]) / 100.0)
            return max(0.0, min(1.0, v))
        except Exception:
            return 1.0

    def _pick_voice_id(self, engine):
        voices = engine.getProperty("voices") or []
        hint = self._language_hint
        for voice in voices:
            langs = getattr(voice, "languages", None) or []
            for lang in langs:
                try:
                    norm = str(lang).lower()
                except Exception:
                    continue
                if hint in norm:
                    return getattr(voice, "id", None)
            name = str(getattr(voice, "name", "")).lower()
            if hint in name:
                return getattr(voice, "id", None)
        return None

    def synthesize_to_file(self, text: str, output_path: str | Path) -> Path:
        import pyttsx3

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        text = (text or "").strip()
        if not text:
            out.write_bytes(b"")
            return out
        if out.suffix.lower() != ".wav":
            out = out.with_suffix(".wav")

        engine = pyttsx3.init()
        voice_id = self._pick_voice_id(engine)
        if voice_id:
            engine.setProperty("voice", voice_id)
        try:
            base_rate = int(engine.getProperty("rate") or 180)
        except Exception:
            base_rate = 180
        engine.setProperty("rate", max(80, min(300, base_rate + self._rate_delta)))
        engine.setProperty("volume", self._volume)
        engine.save_to_file(text, str(out))
        engine.runAndWait()
        engine.stop()
        return out
