from __future__ import annotations

import re

from .normalizer import normalize_text, tokenize_text
from .types import Emotion, Segment


class EmotionDetector:
    """Fast rule-based detector of user's current emotional state."""

    _ALLOWED_SEGMENT_KINDS = {"emotion", "commentary", "smalltalk"}

    _LEXICON: dict[str, set[str]] = {
        "tired": {
            "устал",
            "устала",
            "устали",
            "вымотан",
            "вымотана",
            "измотан",
            "измотана",
            "муторно",
            "сонный",
            "сонная",
            "нетсил",
            "tired",
            "exhausted",
        },
        "frustrated": {
            "бесит",
            "раздражает",
            "достало",
            "задолбало",
            "фрустрация",
            "frustrated",
            "annoyed",
        },
        "sad": {
            "грустно",
            "печально",
            "тоскливо",
            "уныло",
            "подавлен",
            "подавлена",
            "расстроен",
            "расстроена",
            "sad",
            "unhappy",
        },
        "happy": {
            "рад",
            "рада",
            "счастлив",
            "счастлива",
            "доволен",
            "довольна",
            "круто",
            "классно",
            "happy",
            "great",
        },
        "angry": {
            "злюсь",
            "злой",
            "злая",
            "злость",
            "бешусь",
            "ярость",
            "ненавижу",
            "angry",
            "furious",
        },
        "anxious": {
            "тревожно",
            "тревога",
            "волнуюсь",
            "переживаю",
            "нервничаю",
            "беспокоюсь",
            "anxious",
            "worried",
        },
        "neutral": {
            "норм",
            "нормально",
            "ок",
            "ладно",
            "спокойно",
            "нейтрально",
            "обычно",
            "ровно",
            "neutral",
            "fine",
        },
    }

    _PATTERNS: dict[str, tuple[re.Pattern[str], ...]] = {
        "tired": (
            re.compile(r"\b(так\s+муторно|без\s+сил|очень\s+устал[а]?)\b"),
            re.compile(r"\bмуторно\b"),
            re.compile(r"\b(выжат\s+как\s+лимон)\b"),
        ),
        "frustrated": (
            re.compile(r"\b(меня\s+это\s+бесит|это\s+достало)\b"),
            re.compile(r"\b(как\s+же\s+раздражает)\b"),
        ),
        "sad": (
            re.compile(r"\b(мне\s+грустно|очень\s+печально)\b"),
        ),
        "happy": (
            re.compile(r"\b(я\s+рад[а]?|очень\s+рад[а]?)\b"),
            re.compile(r"\b(все\s+классно|все\s+отлично)\b"),
        ),
        "angry": (
            re.compile(r"\b(я\s+злюсь|я\s+в\s+ярости)\b"),
            re.compile(r"\b(это\s+прям\s+бесит)\b"),
        ),
        "anxious": (
            re.compile(r"\b(что-?то\s+тревожно|мне\s+тревожно)\b"),
            re.compile(r"\b(очень\s+волнуюсь|сильно\s+переживаю)\b"),
        ),
        "neutral": (
            re.compile(r"\b(в\s+целом\s+нормально|все\s+ок)\b"),
            re.compile(r"\b(ну\s+такое|так\s+себе)\b"),
        ),
    }

    _INTENSIFIERS = {"очень", "сильно", "прям", "капец", "слишком"}
    _NEGATIONS = {"не", "нет"}
    _NEUTRAL_IDIOM_RE = re.compile(r"\b(ну\s+такое|так\s+себе)\b")
    _PRIMARY_THRESHOLD = 0.58
    _NEUTRAL_THRESHOLD = 0.52

    def detect(self, segments: list[Segment]) -> list[Emotion]:
        emotion_scores: dict[str, float] = {}

        for segment in segments:
            if segment.kind not in self._ALLOWED_SEGMENT_KINDS:
                continue

            normalized = normalize_text(segment.text)
            if not normalized:
                continue

            segment_scores = self._score_segment(segment, normalized)
            for emotion_name, score in segment_scores.items():
                current = emotion_scores.get(emotion_name, 0.0)
                if score > current:
                    emotion_scores[emotion_name] = score

        if not emotion_scores:
            return []

        return self._finalize_emotions(emotion_scores)

    def _score_segment(self, segment: Segment, normalized_text: str) -> dict[str, float]:
        tokens = tokenize_text(normalized_text)
        token_set = set(tokens)
        scores = {name: 0.0 for name in self._LEXICON}
        intensifier_bonus = 0.08 if token_set.intersection(self._INTENSIFIERS) else 0.0

        for emotion_name, vocabulary in self._LEXICON.items():
            lexical_hits = sum(1 for token in tokens if token in vocabulary)
            if lexical_hits:
                scores[emotion_name] += min(0.36 * lexical_hits, 0.72)

            for pattern in self._PATTERNS.get(emotion_name, ()):
                if pattern.search(normalized_text):
                    scores[emotion_name] += 0.24

            if emotion_name == "neutral" and self._NEUTRAL_IDIOM_RE.search(normalized_text):
                scores[emotion_name] += 0.3

            if emotion_name != "neutral":
                scores[emotion_name] += intensifier_bonus

            scores[emotion_name] = self._apply_negation_adjustment(
                emotion_name=emotion_name,
                score=scores[emotion_name],
                tokens=tokens,
            )

            # Bind emotion confidence to segment classification confidence.
            scores[emotion_name] = self._mix_with_segment_confidence(scores[emotion_name], segment.confidence)

        return scores

    def _apply_negation_adjustment(self, emotion_name: str, score: float, tokens: list[str]) -> float:
        if score <= 0.0:
            return score

        vocabulary = self._LEXICON[emotion_name]
        for index, token in enumerate(tokens):
            if token not in vocabulary:
                continue
            if index > 0 and tokens[index - 1] in self._NEGATIONS:
                return max(0.0, score - 0.25)
        return score

    def _finalize_emotions(self, scores: dict[str, float]) -> list[Emotion]:
        sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        best_name, best_score = sorted_scores[0]

        if best_name == "neutral":
            if best_score < self._NEUTRAL_THRESHOLD:
                return []
            return [Emotion(name="neutral", confidence=round(best_score, 3))]

        if best_score < self._PRIMARY_THRESHOLD:
            neutral_score = scores.get("neutral", 0.0)
            if neutral_score >= self._NEUTRAL_THRESHOLD:
                return [Emotion(name="neutral", confidence=round(neutral_score, 3))]
            return []

        emotions = [Emotion(name=best_name, confidence=round(best_score, 3))]

        # Keep one secondary emotion only if it is close enough and meaningful.
        if len(sorted_scores) > 1:
            second_name, second_score = sorted_scores[1]
            if (
                second_name != "neutral"
                and second_score >= self._PRIMARY_THRESHOLD
                and (best_score - second_score) <= 0.12
            ):
                emotions.append(Emotion(name=second_name, confidence=round(second_score, 3)))

        return emotions

    @staticmethod
    def _mix_with_segment_confidence(raw_score: float, segment_confidence: float) -> float:
        blended = (raw_score * 0.72) + (segment_confidence * 0.28)
        if blended < 0.0:
            return 0.0
        if blended > 1.0:
            return 1.0
        return blended


if __name__ == "__main__":
    detector = EmotionDetector()

    examples = [
        Segment(text="я сегодня муторно поработал", kind="emotion", confidence=0.92),
        Segment(text="меня это бесит", kind="emotion", confidence=0.94),
        Segment(text="что-то тревожно", kind="emotion", confidence=0.9),
        Segment(text="я рад", kind="emotion", confidence=0.89),
        Segment(text="ну такое", kind="commentary", confidence=0.82),
        Segment(text="что там по погодке", kind="request", confidence=0.95),
    ]

    for segment in examples:
        detected = detector.detect([segment])
        print(f"Input: {segment.text}")
        print(f"Detected: {detected}")
        print("-" * 56)
