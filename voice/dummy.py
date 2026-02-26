from __future__ import annotations

from pathlib import Path


class DummySTT:
    def transcribe_file(self, audio_path: str | Path) -> str:
        path = Path(audio_path)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")
        return ""


class DummyTTS:
    def synthesize_to_file(self, text: str, output_path: str | Path) -> Path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        # Placeholder output: keeps integration path stable before real TTS backend is selected.
        out.write_bytes(b"")
        return out
