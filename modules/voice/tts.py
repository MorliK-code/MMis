from __future__ import annotations

import hashlib
import math
import re
import shutil
import subprocess
import threading
import wave
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

from config.settings import load_config
from utils.logger import get_logger


LOGGER = get_logger(__name__)


_EMOJI_WORDS = {
    ":)": "smile",
    ":(": "sad",
    ":D": "laugh",
    ";)": "wink",
    "<3": "love",
    "😂": "laugh",
    "🤣": "laugh",
    "😊": "smile",
    "🙂": "smile",
    "🙃": "playful",
    "😢": "sad",
    "😭": "sad",
    "😡": "angry",
    "🤔": "thinking",
    "👍": "ok",
    "❤️": "love",
}


@dataclass(frozen=True)
class TTSConfig:
    voice: str = "default"
    speed: float = 1.0
    pitch: float = 1.0
    sample_rate: int = 22050
    device: str = "default"
    engine: str = "auto"
    cache_enabled: bool = True
    max_chunk_chars: int = 320


@dataclass(frozen=True)
class AudioChunk:
    path: str
    duration_s: float
    sample_rate: int
    text: str
    lang: str
    cached: bool = False
    engine: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


class TTSEngine(Protocol):
    name: str

    def synthesize(self, text: str, lang: str, config: TTSConfig, out_path: Path) -> AudioChunk:
        raise NotImplementedError


class ToneTTSEngine:
    name = "tone"

    def synthesize(self, text: str, lang: str, config: TTSConfig, out_path: Path) -> AudioChunk:
        sample_rate = max(8000, int(config.sample_rate))
        duration_s = max(0.25, min(12.0, len(text) / 28.0))
        total_frames = int(sample_rate * duration_s)
        amp = 9000
        base_freq = 440.0 if str(lang).lower().startswith("en") else 380.0

        out_path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(out_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            frames = bytearray()
            for i in range(total_frames):
                # Basic synthetic fallback audio when no real TTS backend is installed.
                sample = int(amp * math.sin(2.0 * math.pi * base_freq * (i / sample_rate)))
                frames.extend(sample.to_bytes(2, byteorder="little", signed=True))
            wf.writeframes(bytes(frames))

        return AudioChunk(
            path=str(out_path),
            duration_s=duration_s,
            sample_rate=sample_rate,
            text=text,
            lang=lang,
            cached=False,
            engine=self.name,
        )


class Pyttsx3Engine:
    name = "pyttsx3"
    _lock = threading.RLock()

    def __init__(self) -> None:
        import pyttsx3  # type: ignore

        self._pyttsx3 = pyttsx3

    @classmethod
    def available(cls) -> bool:
        try:
            import pyttsx3  # noqa: F401

            return True
        except Exception:
            return False

    def synthesize(self, text: str, lang: str, config: TTSConfig, out_path: Path) -> AudioChunk:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            engine = self._pyttsx3.init()
            try:
                rate = int(180 * max(0.5, min(2.0, float(config.speed))))
                engine.setProperty("rate", rate)
                voice_name = str(config.voice or "").strip().lower()
                if voice_name and voice_name != "default":
                    for v in list(engine.getProperty("voices") or []):
                        raw = f"{getattr(v, 'id', '')} {getattr(v, 'name', '')}".lower()
                        if voice_name in raw:
                            engine.setProperty("voice", getattr(v, "id", ""))
                            break
                engine.save_to_file(text, str(out_path))
                engine.runAndWait()
            finally:
                engine.stop()

        duration = _wav_duration(out_path)
        return AudioChunk(
            path=str(out_path),
            duration_s=duration,
            sample_rate=max(8000, int(config.sample_rate)),
            text=text,
            lang=lang,
            cached=False,
            engine=self.name,
        )


class TTSService:
    def __init__(self, cache_dir: str | Path | None = None):
        cfg = load_config()
        default_cache = cfg.memory_dir / "tts_cache"
        self.cache_dir = Path(cache_dir).expanduser() if cache_dir is not None else default_cache
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._engine_cache: dict[str, TTSEngine] = {}

    def synthesize(self, text: str, lang: str = "", config: TTSConfig | None = None) -> AudioChunk:
        cfg = config or TTSConfig()
        normalized = normalize_tts_text(text)
        if not normalized:
            raise ValueError("TTS text is empty after normalization")

        cache_key = _cache_key(normalized, lang, cfg)
        output_path = self.cache_dir / f"{cache_key}.wav"
        if cfg.cache_enabled and output_path.exists():
            return AudioChunk(
                path=str(output_path),
                duration_s=_wav_duration(output_path),
                sample_rate=cfg.sample_rate,
                text=normalized,
                lang=lang,
                cached=True,
                engine="cache",
            )

        chunks = split_text_for_tts(normalized, max_chars=max(80, int(cfg.max_chunk_chars)))
        engine = self._resolve_engine(cfg.engine)

        if len(chunks) == 1:
            audio = engine.synthesize(chunks[0], lang, cfg, output_path)
            return audio

        parts: list[Path] = []
        durations = 0.0
        for idx, chunk in enumerate(chunks):
            part_path = self.cache_dir / f"{cache_key}.part{idx}.wav"
            audio = engine.synthesize(chunk, lang, cfg, part_path)
            durations += float(audio.duration_s)
            parts.append(part_path)
        _concat_wav(parts, output_path)
        for part in parts:
            try:
                part.unlink(missing_ok=True)
            except Exception:
                pass

        return AudioChunk(
            path=str(output_path),
            duration_s=durations,
            sample_rate=cfg.sample_rate,
            text=normalized,
            lang=lang,
            cached=False,
            engine=getattr(engine, "name", "unknown"),
        )

    def speak(self, text: str, lang: str = "", config: TTSConfig | None = None) -> AudioChunk:
        cfg = config or TTSConfig()
        audio = self.synthesize(text=text, lang=lang, config=cfg)
        _play_audio(Path(audio.path), device=cfg.device)
        return audio

    def stop(self) -> None:
        _stop_audio_playback()

    def _resolve_engine(self, requested: str) -> TTSEngine:
        name = str(requested or "auto").strip().lower()
        if name in {"", "auto"}:
            if Pyttsx3Engine.available():
                name = "pyttsx3"
            else:
                name = "tone"

        with self._lock:
            if name in self._engine_cache:
                return self._engine_cache[name]
            if name == "pyttsx3" and Pyttsx3Engine.available():
                eng: TTSEngine = Pyttsx3Engine()
            else:
                eng = ToneTTSEngine()
            self._engine_cache[name] = eng
            return eng


def normalize_tts_text(text: str) -> str:
    src = str(text or "")
    if not src.strip():
        return ""

    out = src
    for token, word in _EMOJI_WORDS.items():
        out = out.replace(token, f" {word} ")

    out = re.sub(r"\){3,}", " smile ", out)
    out = re.sub(r"\({3,}", " sad ", out)
    out = re.sub(r"\s+", " ", out)
    out = out.replace("\ufeff", " ")
    out = "".join(ch for ch in out if ch.isprintable() or ch in "\n\t")
    return out.strip()


def split_text_for_tts(text: str, max_chars: int = 320) -> list[str]:
    src = normalize_tts_text(text)
    if not src:
        return []
    if len(src) <= max_chars:
        return [src]

    chunks: list[str] = []
    sentence_parts = re.split(r"(?<=[.!?])\s+", src)
    current: list[str] = []
    current_len = 0

    for sentence in sentence_parts:
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(sentence) > max_chars:
            for i in range(0, len(sentence), max_chars):
                piece = sentence[i : i + max_chars].strip()
                if piece:
                    chunks.append(piece)
            continue

        if current_len + len(sentence) + 1 > max_chars:
            if current:
                chunks.append(" ".join(current).strip())
            current = [sentence]
            current_len = len(sentence)
        else:
            current.append(sentence)
            current_len += len(sentence) + 1

    if current:
        chunks.append(" ".join(current).strip())

    return [x for x in chunks if x]


def synthesize(text: str, lang: str = "", config: TTSConfig | None = None) -> AudioChunk:
    return _DEFAULT_TTS.synthesize(text=text, lang=lang, config=config)


def speak(text: str, lang: str = "", config: TTSConfig | None = None) -> AudioChunk:
    return _DEFAULT_TTS.speak(text=text, lang=lang, config=config)


def tts_speak(text: str) -> None:
    try:
        speak(text=text)
    except Exception as exc:
        LOGGER.warning("tts_speak failed: %s", exc)


def _cache_key(text: str, lang: str, config: TTSConfig) -> str:
    payload = {
        "text": text,
        "lang": lang,
        "cfg": asdict(config),
    }
    raw = repr(payload).encode("utf-8", errors="ignore")
    return hashlib.sha1(raw).hexdigest()  # noqa: S324


def _concat_wav(parts: list[Path], out_path: Path) -> None:
    if not parts:
        raise ValueError("No parts to concat")

    params = None
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(out_path), "wb") as wf_out:
        for idx, part in enumerate(parts):
            with wave.open(str(part), "rb") as wf_in:
                if idx == 0:
                    params = wf_in.getparams()
                    wf_out.setparams(params)
                frames = wf_in.readframes(wf_in.getnframes())
                wf_out.writeframes(frames)


def _wav_duration(path: Path) -> float:
    try:
        with wave.open(str(path), "rb") as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            return float(frames) / float(rate or 1)
    except Exception:
        return 0.0


def _play_audio(path: Path, device: str = "default") -> None:
    _ = device
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    try:
        import winsound  # type: ignore

        winsound.PlaySound(str(path), winsound.SND_FILENAME)
        return
    except Exception:
        pass

    if shutil.which("afplay"):
        subprocess.run(["afplay", str(path)], check=False)
        return
    if shutil.which("aplay"):
        subprocess.run(["aplay", str(path)], check=False)
        return


def _stop_audio_playback() -> None:
    try:
        import winsound  # type: ignore

        winsound.PlaySound(None, winsound.SND_PURGE)
    except Exception:
        return


_DEFAULT_TTS = TTSService()
