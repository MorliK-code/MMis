from __future__ import annotations

import re
from dataclasses import dataclass

from metadata.taxonomy import normalize_emotion


EMOTION_LABELS = (
    "neutral",
    "frustrated",
    "happy",
    "anxious",
    "angry",
    "sad",
    "excited",
    "tired",
)


@dataclass(frozen=True)
class EmotionResult:
    label: str
    scores: dict[str, float]
    intensity: float
    arousal: float


_POS_RE = re.compile(r"\b(класс|отлично|спасибо|love|great|nice|cool|awesome|happy)\b", re.I)
_EXCITED_RE = re.compile(r"\b(вау|ура|супер|awesome|excited|let's go|круто)\b", re.I)
_ANGRY_RE = re.compile(r"\b(бесит|задолб|ненавиж|wtf|damn|fuck|shit|сука)\b", re.I)
_FRUSTRATED_RE = re.compile(r"\b(не работает|сломалось|опять|пофикси|can't fix|stuck)\b", re.I)
_ANXIOUS_RE = re.compile(r"\b(тревож|волнуюсь|anxious|nervous|confused|не понял|непонятно)\b", re.I)
_SAD_RE = re.compile(r"\b(грустно|плохо|sad|depressed|разочарован)\b", re.I)
_TIRED_RE = re.compile(r"\b(устал|измотан|tired|sleepy|бессил)\b", re.I)
_CAPS_RE = re.compile(r"[A-ZА-ЯЁ]{3,}")
_CODE_HINT_RE = re.compile(r"```|`[^`]+`|\bdef\s+\w+\(|\bclass\s+\w+\s*:|Traceback|Exception", re.I | re.S)
_BRACKET_SMILE_RUN_RE = re.compile(r"\){2,}")
_BRACKET_SAD_RUN_RE = re.compile(r"\({2,}")
_BRACKET_SMILE_SINGLE_RE = re.compile(r"[A-Za-zА-Яа-яЁёІіЇїЄє0-9]\)(?=$|[\s.!?,;:])")
_BRACKET_SAD_SINGLE_RE = re.compile(r"[A-Za-zА-Яа-яЁёІіЇїЄє0-9]\((?=$|[\s.!?,;:])")
_BRACKET_SMILE_QMARK_RE = re.compile(r"[!?]+\)(?=$|[\s.!?,;:])")
_BRACKET_SAD_QMARK_RE = re.compile(r"[!?]+\((?=$|[\s.!?,;:])")


def detect(text: str, lang: str = "unknown") -> EmotionResult:
    src = str(text or "").strip()
    lower = src.lower()
    exclam = src.count("!")
    qmarks = src.count("?")
    caps_hits = len(_CAPS_RE.findall(src))

    scores = {k: 0.05 for k in EMOTION_LABELS}
    scores["neutral"] = 0.35

    if _POS_RE.search(src):
        scores["happy"] += 0.35
    if _EXCITED_RE.search(src):
        scores["excited"] += 0.35
    if _ANGRY_RE.search(src):
        scores["angry"] += 0.42
    if _FRUSTRATED_RE.search(src):
        scores["frustrated"] += 0.40
    if _ANXIOUS_RE.search(lower):
        scores["anxious"] += 0.38
    if _SAD_RE.search(src):
        scores["sad"] += 0.40
    if _TIRED_RE.search(src):
        scores["tired"] += 0.40

    if exclam >= 2:
        scores["excited"] += 0.10
        scores["angry"] += 0.12
    if caps_hits > 0:
        scores["angry"] += 0.18
    if qmarks >= 2:
        scores["anxious"] += 0.08

    if lang in {"ru", "uk"} and any(token in lower for token in ("блин", "черт", "чёрт")):
        scores["frustrated"] += 0.16

    smile_qmark = bool(_BRACKET_SMILE_QMARK_RE.search(src))
    sad_qmark = bool(_BRACKET_SAD_QMARK_RE.search(src))
    smile_units, sad_units = _bracket_emotion_units(src)
    if smile_units > 0:
        scores["happy"] += min(0.34, 0.10 * float(smile_units))
        if smile_units >= 2:
            scores["excited"] += min(0.18, 0.06 * float(smile_units))
        scores["neutral"] = max(0.0, float(scores.get("neutral", 0.0)) - min(0.20, 0.10 * float(smile_units)))
        scores["sad"] = max(0.0, float(scores.get("sad", 0.0)) - 0.06)
        scores["angry"] = max(0.0, float(scores.get("angry", 0.0)) - 0.04)
    if sad_units > 0:
        scores["sad"] += min(0.42, 0.14 * float(sad_units))
        scores["frustrated"] += min(0.22, 0.06 * float(sad_units))
        scores["neutral"] = max(0.0, float(scores.get("neutral", 0.0)) - min(0.22, 0.12 * float(sad_units)))
        scores["happy"] = max(0.0, float(scores.get("happy", 0.0)) - 0.05)
        scores["excited"] = max(0.0, float(scores.get("excited", 0.0)) - 0.04)
    if smile_qmark:
        scores["happy"] += 0.18
        scores["excited"] += 0.08
        scores["neutral"] = max(0.0, float(scores.get("neutral", 0.0)) - 0.12)
    if sad_qmark:
        scores["sad"] += 0.22
        scores["anxious"] += 0.10
        scores["frustrated"] += 0.06
        scores["neutral"] = max(0.0, float(scores.get("neutral", 0.0)) - 0.14)

    label = max(scores, key=scores.get)
    normalized_label = normalize_emotion(label)
    peak = max(0.0, min(1.0, scores[label]))
    intensity = max(0.0, min(1.0, peak))
    arousal = max(
        0.0,
        min(
            1.0,
            0.2 + exclam * 0.08 + caps_hits * 0.15 + (0.15 if normalized_label in {"angry", "excited"} else 0.0),
        ),
    )
    return EmotionResult(label=normalized_label, scores=scores, intensity=intensity, arousal=arousal)


def detect_emotion(text: str) -> dict:
    result = detect(text=text, lang="unknown")
    return {
        "primary": result.label,
        "score": float(result.intensity),
        "intensity": float(result.intensity),
        "scores": dict(result.scores),
        "arousal": float(result.arousal),
    }


def _bracket_emotion_units(text: str) -> tuple[int, int]:
    src = str(text or "")
    if not src or _CODE_HINT_RE.search(src):
        return (0, 0)

    smile_units = 0
    sad_units = 0
    for m in _BRACKET_SMILE_RUN_RE.finditer(src):
        smile_units += max(1, int(len(m.group(0)) // 2))
    for m in _BRACKET_SAD_RUN_RE.finditer(src):
        sad_units += max(1, int(len(m.group(0)) // 2))

    if _BRACKET_SMILE_SINGLE_RE.search(src):
        smile_units += 1
    if _BRACKET_SAD_SINGLE_RE.search(src):
        sad_units += 1
    if _BRACKET_SMILE_QMARK_RE.search(src):
        smile_units += 1
    if _BRACKET_SAD_QMARK_RE.search(src):
        sad_units += 1

    return (smile_units, sad_units)
