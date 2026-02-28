from __future__ import annotations

import re
from dataclasses import dataclass


EMOTION_LABELS = (
    "neutral",
    "positive_excited",
    "frustrated_angry",
    "sad_tired",
    "playful_ironic",
    "confused",
)


@dataclass(frozen=True)
class EmotionResult:
    label: str
    scores: dict[str, float]
    intensity: float
    arousal: float


_POS_RE = re.compile(r"\b(класс|отлично|спасибо|love|great|nice|cool|awesome)\b", re.I)
_ANGRY_RE = re.compile(r"\b(бесит|задолб|ненавиж|wtf|damn|fuck|shit)\b", re.I)
_CONFUSED_RE = re.compile(r"\b(не понял|непонятно|я туплю|что за|confused|dont understand)\b", re.I)
_SAD_RE = re.compile(r"\b(грустно|устал|устала|плохо|sad|tired|depressed)\b", re.I)
_PLAYFUL_RE = re.compile(r"(хаха|аха|lol|lmao|ирони|сарказ|:d|;\)|\^\^)", re.I)
_CAPS_RE = re.compile(r"[A-ZА-ЯЁ]{3,}")


def detect(text: str, lang: str = "unknown") -> EmotionResult:
    src = str(text or "").strip()
    lower = src.lower()
    exclam = src.count("!")
    qmarks = src.count("?")
    caps_hits = len(_CAPS_RE.findall(src))

    scores = {k: 0.05 for k in EMOTION_LABELS}
    scores["neutral"] = 0.35

    if _POS_RE.search(src):
        scores["positive_excited"] += 0.35
    if _ANGRY_RE.search(src):
        scores["frustrated_angry"] += 0.45
    if _CONFUSED_RE.search(lower):
        scores["confused"] += 0.4
    if _SAD_RE.search(src):
        scores["sad_tired"] += 0.42
    if _PLAYFUL_RE.search(src):
        scores["playful_ironic"] += 0.34

    if exclam >= 2:
        scores["positive_excited"] += 0.08
        scores["frustrated_angry"] += 0.12
    if caps_hits > 0:
        scores["frustrated_angry"] += 0.2
    if qmarks >= 2:
        scores["confused"] += 0.08

    if lang in {"ru", "uk"} and any(token in lower for token in ("блин", "черт", "чёрт")):
        scores["frustrated_angry"] += 0.18

    label = max(scores, key=scores.get)
    peak = max(0.0, min(1.0, scores[label]))
    intensity = max(0.0, min(1.0, peak))
    arousal = max(
        0.0,
        min(
            1.0,
            0.2 + exclam * 0.08 + caps_hits * 0.15 + (0.15 if label in {"frustrated_angry", "positive_excited"} else 0.0),
        ),
    )
    return EmotionResult(label=label, scores=scores, intensity=intensity, arousal=arousal)


def detect_emotion(text: str) -> dict:
    # Backward-compat helper used by existing call sites.
    result = detect(text=text, lang="unknown")
    return {
        "primary": result.label,
        "score": float(result.intensity),
        "scores": dict(result.scores),
        "arousal": float(result.arousal),
    }
