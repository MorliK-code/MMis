from __future__ import annotations

import importlib.util
import wave
from pathlib import Path
from typing import Any

from modules.voice.tts import AudioChunk, TTSConfig, split_text_for_tts
from utils.logger import get_logger


LOGGER = get_logger(__name__)


class QwenTTSEngine:
    name = "qwen"

    def __init__(self, model_name: str = "Qwen3-TTS-0.6B", *, device: str = "auto"):
        self.model_name = str(model_name or "Qwen3-TTS-0.6B").strip() or "Qwen3-TTS-0.6B"
        self.device = str(device or "auto").strip() or "auto"
        self._pipeline: Any | None = None

    @classmethod
    def available(cls) -> bool:
        return (
            importlib.util.find_spec("transformers") is not None
            and importlib.util.find_spec("soundfile") is not None
            and importlib.util.find_spec("torch") is not None
        )

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
        from transformers import pipeline  # type: ignore

        pipe = self._pipeline
        if pipe is None:
            device = _pipeline_device(config.device or self.device)
            kwargs: dict[str, Any] = {"model": self.model_name}
            if device is not None:
                kwargs["device"] = device
            pipe = pipeline("text-to-speech", **kwargs)
            self._pipeline = pipe

        prompt = text
        if lang:
            prompt = f"[{lang}] {text}"
        if config.voice and config.voice != "default":
            prompt = f"{config.voice}: {prompt}"

        result = pipe(prompt)
        audio = result.get("audio") if isinstance(result, dict) else None
        sample_rate = result.get("sampling_rate") if isinstance(result, dict) else None
        if audio is None:
            raise RuntimeError("Qwen TTS backend returned no audio")
        sf.write(str(out_path), audio, int(sample_rate or config.sample_rate or 22050))


def _pipeline_device(device: str) -> int | str | None:
    raw = str(device or "auto").strip().lower()
    if raw in {"", "default", "auto"}:
        return None
    if raw.startswith("cuda") or raw in {"gpu", "cu"}:
        return 0
    return -1


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
