from __future__ import annotations

import io
import itertools
import re
import tempfile
import wave
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Protocol

from utils.logger import get_logger


LOGGER = get_logger(__name__)


@dataclass(frozen=True)
class STTConfig:
    device: str = "default"
    vad: bool = True
    language_hint: str = ""
    beam_size: int = 5
    speed_mode: str = "balanced"
    engine: str = "auto"
    silence_ms: int = 900
    min_speech_ms: int = 200


@dataclass(frozen=True)
class STTSegment:
    start_s: float
    end_s: float
    text: str
    conf: float


@dataclass(frozen=True)
class STTResult:
    text: str
    conf: float
    segments: list[STTSegment] = field(default_factory=list)
    duration_s: float = 0.0
    avg_conf: float = 0.0
    language_detected: str = "unknown"
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "conf": float(self.conf),
            "segments": [asdict(s) for s in self.segments],
            "duration_s": float(self.duration_s),
            "avg_conf": float(self.avg_conf),
            "language_detected": self.language_detected,
            "metadata": dict(self.metadata or {}),
        }


class STTEngine(Protocol):
    name: str

    def transcribe(self, audio: str | Path | bytes, config: STTConfig) -> STTResult:
        raise NotImplementedError


class NullSTTEngine:
    """Fallback STT engine that extracts text from sidecar files if available."""

    name = "null"

    def transcribe(self, audio: str | Path | bytes, config: STTConfig) -> STTResult:
        _ = config
        duration = _estimate_duration(audio)

        if isinstance(audio, (str, Path)):
            path = Path(audio)
            if path.exists():
                sidecar = path.with_suffix(".txt")
                if sidecar.exists():
                    text = sidecar.read_text(encoding="utf-8-sig", errors="ignore").strip()
                    lang = _detect_language(text)
                    segment = STTSegment(start_s=0.0, end_s=max(duration, 0.01), text=text, conf=0.35)
                    return STTResult(
                        text=text,
                        conf=0.35,
                        segments=[segment],
                        duration_s=duration,
                        avg_conf=0.35,
                        language_detected=lang,
                        metadata={"engine": self.name, "source": "sidecar_txt"},
                    )

        if isinstance(audio, (bytes, bytearray)):
            guessed = _try_decode_text(bytes(audio))
            if guessed:
                lang = _detect_language(guessed)
                return STTResult(
                    text=guessed,
                    conf=0.2,
                    segments=[STTSegment(start_s=0.0, end_s=max(duration, 0.01), text=guessed, conf=0.2)],
                    duration_s=duration,
                    avg_conf=0.2,
                    language_detected=lang,
                    metadata={"engine": self.name, "source": "decoded_bytes"},
                )

        return STTResult(
            text="",
            conf=0.0,
            segments=[],
            duration_s=duration,
            avg_conf=0.0,
            language_detected=(config.language_hint or "unknown"),
            metadata={"engine": self.name, "reason": "no_backend_or_no_text"},
        )


class STTService:
    def __init__(self, engine: STTEngine | None = None):
        self._engine: STTEngine = engine or NullSTTEngine()

    def transcribe(self, audio: str | Path | bytes, config: STTConfig | None = None) -> STTResult:
        cfg = config or STTConfig()
        result = self._engine.transcribe(audio=audio, config=cfg)
        if not result.language_detected or result.language_detected == "unknown":
            lang = _detect_language(result.text)
            result = STTResult(
                text=result.text,
                conf=result.conf,
                segments=result.segments,
                duration_s=result.duration_s,
                avg_conf=result.avg_conf,
                language_detected=lang,
                metadata=result.metadata,
            )
        return result

    def transcribe_stream(self, chunks: Iterable[bytes], config: STTConfig | None = None) -> Iterable[STTResult]:
        cfg = config or STTConfig()
        merged = bytearray()
        total = 0
        for idx, chunk in enumerate(chunks):
            if not isinstance(chunk, (bytes, bytearray)):
                continue
            merged.extend(chunk)
            total += len(chunk)
            partial = _try_decode_text(bytes(chunk))
            if partial:
                text = partial.strip()
                if text:
                    yield STTResult(
                        text=text,
                        conf=0.15,
                        segments=[STTSegment(start_s=0.0, end_s=0.0, text=text, conf=0.15)],
                        duration_s=0.0,
                        avg_conf=0.15,
                        language_detected=_detect_language(text),
                        metadata={"engine": "stream", "chunk_index": idx, "bytes": len(chunk)},
                    )

        final = self.transcribe(bytes(merged), cfg)
        meta = dict(final.metadata or {})
        meta.update({"stream_total_bytes": total})
        yield STTResult(
            text=final.text,
            conf=final.conf,
            segments=final.segments,
            duration_s=final.duration_s,
            avg_conf=final.avg_conf,
            language_detected=final.language_detected,
            metadata=meta,
        )


def transcribe(audio: str | Path | bytes, config: STTConfig | None = None) -> tuple[str, float, list[STTSegment]]:
    result = _DEFAULT_STT.transcribe(audio=audio, config=config)
    return result.text, result.conf, result.segments


def transcribe_result(audio: str | Path | bytes, config: STTConfig | None = None) -> STTResult:
    return _DEFAULT_STT.transcribe(audio=audio, config=config)


def stt_transcribe(path: str) -> str:
    try:
        result = _DEFAULT_STT.transcribe(path, STTConfig())
        return result.text
    except Exception as exc:
        LOGGER.warning("stt_transcribe failed: %s", exc)
        return ""


def _estimate_duration(audio: str | Path | bytes) -> float:
    if isinstance(audio, (str, Path)):
        path = Path(audio)
        if path.exists() and path.suffix.lower() == ".wav":
            try:
                with wave.open(str(path), "rb") as wf:
                    rate = float(wf.getframerate() or 1)
                    frames = float(wf.getnframes() or 0)
                    return frames / rate
            except Exception:
                return 0.0
        return 0.0

    if isinstance(audio, (bytes, bytearray)):
        blob = bytes(audio)
        try:
            with wave.open(io.BytesIO(blob), "rb") as wf:
                rate = float(wf.getframerate() or 1)
                frames = float(wf.getnframes() or 0)
                return frames / rate
        except Exception:
            # raw PCM rough guess at 16kHz/16bit mono
            if not blob:
                return 0.0
            return len(blob) / 32000.0
    return 0.0


def _try_decode_text(blob: bytes) -> str:
    if not blob:
        return ""
    for enc in ("utf-8", "utf-16", "cp1251", "latin-1"):
        try:
            decoded = blob.decode(enc, errors="ignore").strip()
            decoded = re.sub(r"\s+", " ", decoded)
            if decoded and re.search(r"[A-Za-zА-Яа-яЁё]", decoded):
                return decoded
        except Exception:
            continue
    return ""


def _detect_language(text: str) -> str:
    src = str(text or "")
    if not src.strip():
        return "unknown"

    cyr = len(re.findall(r"[А-Яа-яЁё]", src))
    lat = len(re.findall(r"[A-Za-z]", src))
    if cyr > 0 and lat > 0:
        return "mixed"
    if cyr > 0:
        return "ru"
    if lat > 0:
        return "en"
    return "unknown"


_DEFAULT_STT = STTService()
