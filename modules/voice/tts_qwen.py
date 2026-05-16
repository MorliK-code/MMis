from __future__ import annotations

import importlib.util
import os
import wave
from pathlib import Path
from typing import Any

import numpy as np

from modules.voice.tts import AudioChunk, TTSConfig, split_text_for_tts
from utils.logger import get_logger


LOGGER = get_logger(__name__)


class QwenTTSEngine:
    name = "qwen"

    def __init__(self, model_name: str = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice", *, device: str = "auto"):
        self.model_name = _normalize_model_name(model_name)
        self.device = str(device or "auto").strip() or "auto"
        self._model: Any | None = None

    @classmethod
    def available(cls) -> bool:
        return importlib.util.find_spec("qwen_tts") is not None and importlib.util.find_spec("soundfile") is not None

    def synthesize(self, text: str, lang: str, config: TTSConfig, out_path: Path) -> AudioChunk:
        payload = str(text or "").strip()
        if not payload:
            raise ValueError("Qwen TTS text is empty")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        chunks = split_text_for_tts(payload, max_chars=max(80, int(config.max_chunk_chars or 320)))
        parts: list[Path] = []
        for idx, chunk in enumerate(chunks or [payload]):
            part = out_path if len(chunks) <= 1 else out_path.with_name(f"{out_path.stem}.qwen{idx}{out_path.suffix}")
            self._synthesize_one(chunk, lang, config, part)
            parts.append(part)
        if len(parts) > 1:
            _concat_wav(parts, out_path)
            for part in parts:
                try:
                    part.unlink(missing_ok=True)
                except Exception:
                    pass
        return AudioChunk(
            path=str(out_path),
            duration_s=_wav_duration(out_path),
            sample_rate=max(8000, int(config.sample_rate or 22050)),
            text=payload,
            lang=lang,
            cached=False,
            engine=self.name,
        )

    def _synthesize_one(self, text: str, lang: str, config: TTSConfig, out_path: Path) -> None:
        import soundfile as sf  # type: ignore

        model = self._load_model(config)
        language = _qwen_language(lang)
        if "customvoice" in self.model_name.lower() or "custom_voice" in self.model_name.lower():
            wavs, sample_rate = model.generate_custom_voice(
                text=text,
                language=language,
                speaker=_qwen_speaker(config.voice),
                instruct=_qwen_instruct(config.voice),
            )
        else:
            raise RuntimeError(
                "Qwen TTS Base requires reference audio. Use Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice "
                "or add reference-audio support before selecting a Base checkpoint."
            )
        if not wavs:
            raise RuntimeError("Qwen TTS backend returned no audio")
        audio = np.asarray(wavs[0])
        sf.write(str(out_path), audio, int(sample_rate or config.sample_rate or 22050))

    def _load_model(self, config: TTSConfig):
        if self._model is not None:
            return self._model

        import torch  # type: ignore
        _ensure_sox_on_path()
        from qwen_tts import Qwen3TTSModel  # type: ignore

        device = _resolved_device(config.device or self.device)
        kwargs: dict[str, Any] = {"device_map": "cuda:0" if device == "cuda" else "cpu"}
        kwargs["dtype"] = torch.bfloat16 if device == "cuda" else torch.float32
        self._model = Qwen3TTSModel.from_pretrained(self.model_name, **kwargs)
        return self._model


def _normalize_model_name(value: str) -> str:
    raw = str(value or "").strip()
    lowered = raw.lower()
    if lowered in {"", "qwen3-tts", "qwen3-tts-0.6b", "qwen/qwen3-tts-0.6b"}:
        return "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
    if lowered in {"qwen3-tts-12hz-0.6b-customvoice", "qwen/qwen3-tts-12hz-0.6b-customvoice"}:
        return "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
    return raw


def _resolved_device(device: str) -> str:
    raw = str(device or "auto").strip().lower()
    wants_cuda = raw.startswith("cuda") or raw in {"gpu", "cu"}
    if raw in {"", "default", "auto"}:
        wants_cuda = _torch_cuda_available()
    if wants_cuda and _torch_cuda_available():
        return "cuda"
    if wants_cuda:
        LOGGER.warning("Qwen TTS requested CUDA, but torch CUDA is not available; using CPU")
    return "cpu"


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
    paths = os.environ.get("PATH", "").split(os.pathsep)
    for item in paths:
        path = Path(item) / "sox.exe"
        if path.exists():
            return True
    return False


def _qwen_language(lang: str) -> str:
    raw = str(lang or "").strip().lower()
    mapping = {
        "ru": "Russian",
        "rus": "Russian",
        "russian": "Russian",
        "en": "English",
        "eng": "English",
        "english": "English",
        "zh": "Chinese",
        "chinese": "Chinese",
    }
    return mapping.get(raw, "Russian")


def _qwen_speaker(voice: str) -> str:
    raw = str(voice or "").strip()
    if raw and raw.lower() not in {"default", "ru-ru-dmitryneural"}:
        return raw
    return "Ryan"


def _qwen_instruct(voice: str) -> str:
    _ = voice
    return "Speak naturally and calmly."


def _concat_wav(parts: list[Path], out_path: Path) -> None:
    if not parts:
        raise ValueError("No Qwen TTS chunks to concat")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(out_path), "wb") as wf_out:
        for idx, part in enumerate(parts):
            with wave.open(str(part), "rb") as wf_in:
                if idx == 0:
                    wf_out.setparams(wf_in.getparams())
                wf_out.writeframes(wf_in.readframes(wf_in.getnframes()))


def _wav_duration(path: Path) -> float:
    try:
        with wave.open(str(path), "rb") as wf:
            return float(wf.getnframes()) / float(wf.getframerate() or 1)
    except Exception:
        return 0.0
