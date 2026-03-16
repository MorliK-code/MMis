from __future__ import annotations

from metadata.emotion_detector import detect
from memory.ingest_analyzer import IngestEmotion


_EMOTION_MAP = {
    "neutral": "calm",
    "happy": "positive",
    "excited": "excited",
    "frustrated": "frustrated",
    "angry": "irritated",
    "sad": "sad",
    "anxious": "uncertain",
    "tired": "calm",
}


def detect_memory_emotion(text: str) -> IngestEmotion | None:
    src = str(text or "").strip()
    if not src:
        return None
    result = detect(src, lang="unknown")
    primary = _EMOTION_MAP.get(str(result.label or "").strip().lower(), "calm")
    intensity = max(0.0, min(1.0, float(result.intensity)))
    arousal = max(0.0, min(1.0, float(result.arousal)))
    confidence = max(0.35, min(1.0, (0.55 * intensity) + (0.25 * arousal) + 0.20))
    return IngestEmotion(
        primary=primary,
        intensity=intensity,
        arousal=arousal,
        confidence=confidence,
    )
