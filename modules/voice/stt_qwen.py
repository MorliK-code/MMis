from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from typing import Any

from modules.voice.stt import STTConfig, STTResult, STTSegment
from utils.logger import get_logger


LOGGER = get_logger(__name__)


class QwenASRSTTEngine:
    name = "qwen-asr"

    def __init__(self, model_name: str = "Qwen/Qwen3-ASR-1.7B", *, device: str = "auto"):
        self.model_name = _normalize_model_name(model_name)
        self.device = str(device or "auto").strip().lower() or "auto"
        self._model: Any | None = None

    @classmethod
    def available(cls) -> bool:
        return importlib.util.find_spec("qwen_asr") is not None

    def transcribe(self, audio: str | Path | bytes, config: STTConfig) -> STTResult:
        path = _audio_to_path(audio)
        model = self._load_model(config)
        language = str(config.language_hint or "").strip() or None
        items = model.transcribe(str(path), language=language)
        item = items[0] if items else None
        text = str(getattr(item, "text", "") or "").strip()
        lang = str(getattr(item, "language", "") or language or "unknown").strip() or "unknown"
        return STTResult(
            text=text,
            conf=1.0 if text else 0.0,
            segments=[STTSegment(start_s=0.0, end_s=0.0, text=text, conf=1.0)] if text else [],
            duration_s=0.0,
            avg_conf=1.0 if text else 0.0,
            language_detected=lang,
            metadata={
                "engine": self.name,
                "model": self.model_name,
                "device": self._resolved_device(config),
            },
        )

    def _load_model(self, config: STTConfig):
        if self._model is not None:
            return self._model

        import torch  # type: ignore
        _ensure_sox_on_path()
        from qwen_asr import Qwen3ASRModel  # type: ignore

        device = self._resolved_device(config)
        kwargs: dict[str, Any] = {
            "device_map": "cuda:0" if device == "cuda" else "cpu",
            "max_inference_batch_size": 1,
        }
        kwargs["dtype"] = torch.bfloat16 if device == "cuda" else torch.float32
        self._model = Qwen3ASRModel.from_pretrained(self.model_name, **kwargs)
        return self._model

    def _resolved_device(self, config: STTConfig) -> str:
        raw = str(config.device or self.device or "auto").strip().lower()
        wants_cuda = raw.startswith("cuda") or raw in {"gpu", "cu"}
        if raw in {"", "default", "auto"}:
            wants_cuda = _torch_cuda_available()
        if wants_cuda and _torch_cuda_available():
            return "cuda"
        if wants_cuda:
            LOGGER.warning("Qwen ASR requested CUDA, but torch CUDA is not available; using CPU")
        return "cpu"


def _normalize_model_name(value: str) -> str:
    raw = str(value or "").strip()
    lowered = raw.lower()
    if lowered in {"", "qwen3-asr", "qwen3-asr-1.5b", "qwen/qwen3-asr-1.5b"}:
        return "Qwen/Qwen3-ASR-1.7B"
    if lowered in {"qwen3-asr-1.7b", "qwen/qwen3-asr-1.7b"}:
        return "Qwen/Qwen3-ASR-1.7B"
    if lowered in {"qwen3-asr-0.6b", "qwen/qwen3-asr-0.6b"}:
        return "Qwen/Qwen3-ASR-0.6B"
    return raw


def _torch_cuda_available() -> bool:
    try:
        import torch  # type: ignore

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _ensure_sox_on_path() -> None:
    if _which_sox():
        return
    roots = [
        Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "WinGet" / "Packages",
        Path(os.environ.get("ProgramFiles", "")),
    ]
    for root in roots:
        if not root.exists():
            continue
        try:
            match = next(root.rglob("sox.exe"), None)
        except Exception:
            match = None
        if match is not None:
            os.environ["PATH"] = f"{match.parent}{os.pathsep}{os.environ.get('PATH', '')}"
            return


def _which_sox() -> bool:
    for item in os.environ.get("PATH", "").split(os.pathsep):
        if (Path(item) / "sox.exe").exists():
            return True
    return False


def _audio_to_path(audio: str | Path | bytes) -> Path:
    if isinstance(audio, (str, Path)):
        path = Path(audio).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")
        return path
    raise TypeError("Qwen ASR expects an audio file path")
