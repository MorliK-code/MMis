from __future__ import annotations

import importlib.util
import tempfile
import wave
from pathlib import Path
from typing import Any

from modules.voice.stt import STTConfig, STTResult, STTSegment
from utils.logger import get_logger


LOGGER = get_logger(__name__)


class FasterWhisperSTTEngine:
    name = "faster-whisper"

    def __init__(self, model_size: str = "small", *, device: str = "auto", compute_type: str = "default"):
        self.model_size = str(model_size or "small").strip() or "small"
        self.device = str(device or "auto").strip().lower() or "auto"
        self.compute_type = str(compute_type or "default").strip() or "default"
        self._model: Any | None = None

    @classmethod
    def available(cls) -> bool:
        return importlib.util.find_spec("faster_whisper") is not None

    def transcribe(self, audio: str | Path | bytes, config: STTConfig) -> STTResult:
        path = _audio_to_path(audio)
        model = self._load_model(config)
        kwargs = self._transcribe_kwargs(config)
        segments_iter, info = model.transcribe(str(path), **kwargs)

        segments: list[STTSegment] = []
        parts: list[str] = []
        confidences: list[float] = []
        for item in segments_iter:
            text = str(getattr(item, "text", "") or "").strip()
            if text:
                parts.append(text)
            conf = _segment_confidence(item)
            confidences.append(conf)
            segments.append(
                STTSegment(
                    start_s=float(getattr(item, "start", 0.0) or 0.0),
                    end_s=float(getattr(item, "end", 0.0) or 0.0),
                    text=text,
                    conf=conf,
                )
            )

        text = " ".join(parts).strip()
        avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
        duration = float(getattr(info, "duration", 0.0) or _wav_duration(path))
        lang = str(getattr(info, "language", "") or config.language_hint or "unknown")
        language_probability = float(getattr(info, "language_probability", 0.0) or 0.0)
        result_conf = max(avg_conf, language_probability if text else 0.0)
        return STTResult(
            text=text,
            conf=result_conf,
            segments=segments,
            duration_s=duration,
            avg_conf=avg_conf,
            language_detected=lang,
            metadata={
                "engine": self.name,
                "model": self.model_size,
                "device": self._resolved_device(config),
                "compute_type": self._resolved_compute_type(config),
                "language_probability": language_probability,
            },
        )

    def _load_model(self, config: STTConfig):
        if self._model is not None:
            return self._model
        from faster_whisper import WhisperModel  # type: ignore

        self._model = WhisperModel(
            self.model_size,
            device=self._resolved_device(config),
            compute_type=self._resolved_compute_type(config),
        )
        return self._model

    def _resolved_device(self, config: STTConfig) -> str:
        raw = str(config.device or self.device or "auto").strip().lower()
        if raw in {"", "default", "auto"}:
            return "auto"
        if raw.startswith("cuda") or raw in {"gpu", "cu"}:
            return "cuda"
        return "cpu"

    def _resolved_compute_type(self, config: STTConfig) -> str:
        raw = str(getattr(config, "compute_type", "") or self.compute_type or "default").strip()
        if raw and raw.lower() != "default":
            return raw
        device = self._resolved_device(config)
        return "int8_float16" if device == "cuda" else "int8"

    def _transcribe_kwargs(self, config: STTConfig) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "beam_size": max(1, int(config.beam_size or 1)),
            "vad_filter": bool(config.vad),
        }
        lang = str(config.language_hint or "").strip()
        if lang:
            kwargs["language"] = lang
        if str(config.speed_mode or "").strip().lower() in {"fast", "speed"}:
            kwargs["beam_size"] = 1
        return kwargs


def _audio_to_path(audio: str | Path | bytes) -> Path:
    if isinstance(audio, (str, Path)):
        path = Path(audio).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")
        return path
    suffix = ".wav" if _looks_like_wav(bytes(audio)) else ".raw"
    tmp = tempfile.NamedTemporaryFile(prefix="mmis_stt_", suffix=suffix, delete=False)
    with tmp:
        tmp.write(bytes(audio))
    return Path(tmp.name)


def _looks_like_wav(blob: bytes) -> bool:
    return len(blob) >= 12 and blob[:4] == b"RIFF" and blob[8:12] == b"WAVE"


def _segment_confidence(segment: Any) -> float:
    if hasattr(segment, "avg_logprob"):
        try:
            # faster-whisper exposes log probability; map it to a bounded rough confidence.
            return max(0.0, min(1.0, 1.0 + (float(segment.avg_logprob) / 5.0)))
        except Exception:
            pass
    if hasattr(segment, "no_speech_prob"):
        try:
            return max(0.0, min(1.0, 1.0 - float(segment.no_speech_prob)))
        except Exception:
            pass
    return 0.0


def _wav_duration(path: Path) -> float:
    try:
        with wave.open(str(path), "rb") as wf:
            return float(wf.getnframes()) / float(wf.getframerate() or 1)
    except Exception:
        return 0.0
