from __future__ import annotations

import re
from typing import Iterable

from .normalizer import normalize_text, tokenize_text
from .types import Segment


class SegmentClassifier:
    """Rule-based classifier that maps semantic chunks to lightweight NLU kinds."""

    _QUESTION_WORDS = {
        "что",
        "кто",
        "где",
        "когда",
        "как",
        "какой",
        "какая",
        "какие",
        "какую",
        "каким",
        "сколько",
        "почему",
        "зачем",
        "куда",
        "откуда",
        "чем",
    }
    _REQUEST_VERBS = {
        "подскажи",
        "скажи",
        "расскажи",
        "объясни",
        "покажи",
        "дай",
        "помоги",
        "сравни",
        "проверь",
        "уточни",
    }
    _FIRST_PERSON = {"я", "мне", "меня", "мой", "моя", "мои", "мое", "мы", "нам", "наш"}
    _PERSONAL_FACT_VERBS = {
        "живу",
        "работаю",
        "учусь",
        "люблю",
        "занимаюсь",
        "интересуюсь",
        "пишу",
        "использую",
        "из",
        "есть",
    }
    _EMOTION_WORDS = {
        "муторно",
        "грустно",
        "рад",
        "рада",
        "устал",
        "устала",
        "устали",
        "злюсь",
        "бесит",
        "тревожно",
        "волнительно",
        "обидно",
        "страшно",
        "тяжело",
        "плохо",
        "хорошо",
        "супер",
    }
    _COMMENTARY_WORDS = {
        "норм",
        "нормально",
        "такое",
        "неплохо",
        "странно",
        "интересно",
        "ясно",
        "понятно",
        "логично",
        "круто",
        "жесть",
        "капец",
        "хз",
        "ок",
        "ладно",
    }
    _SMALLTALK_WORDS = {
        "привет",
        "здарова",
        "здравствуйте",
        "здравствуй",
        "хай",
        "пока",
        "спасибо",
        "благодарю",
        "доброе",
        "добрый",
    }
    _SMALLTALK_PHRASES = {
        "доброе утро",
        "добрый день",
        "добрый вечер",
        "как дела",
        "все хорошо",
        "все ок",
    }

    _REQUEST_PATTERN = re.compile(
        r"\b(что по|как там|есть ли|можешь|можно|подскажи|скажи|помоги|нужно|хочу узнать)\b"
    )
    _PERSONAL_FACT_PATTERN = re.compile(
        r"\b(меня зовут|я живу|я из|я работаю|я учусь|я люблю|я занимаюсь|у меня)\b"
    )
    _EMOTION_PATTERN = re.compile(
        r"\b(чувствую|ощущаю|муторно|грустно|устал|устала|бесит|злит|тревожно|рад)\b"
    )
    _COMMENTARY_PATTERN = re.compile(
        r"\b(ну такое|так себе|не очень|в целом|по сути|короче|вообще)\b"
    )
    _AGE_PATTERN = re.compile(r"\b(?:мне\s+\d{1,3}|\d{1,3}\s+лет)\b")

    def classify(self, segments: list[str]) -> list[Segment]:
        classified: list[Segment] = []

        for raw_text in segments:
            text = raw_text.strip()
            if not text:
                continue

            kind, confidence = self._classify_single(text)
            classified.append(
                Segment(
                    text=text,
                    kind=kind,
                    confidence=round(confidence, 3),
                    meta={},
                )
            )

        return classified

    def _classify_single(self, text: str) -> tuple[str, float]:
        normalized = normalize_text(text)
        tokens = tokenize_text(normalized)
        if not tokens:
            return "unknown", 0.25

        scores = {
            "request": self._score_request(normalized, tokens, text),
            "smalltalk": self._score_smalltalk(normalized, tokens),
            "emotion": self._score_emotion(normalized, tokens),
            "personal_fact": self._score_personal_fact(normalized, tokens),
            "commentary": self._score_commentary(normalized, tokens),
            "unknown": 0.2,
        }

        return self._resolve_kind(scores)

    def _score_request(self, normalized: str, tokens: list[str], original: str) -> float:
        score = 0.0

        if "?" in original:
            score += 0.2
        if tokens and tokens[0] in self._QUESTION_WORDS:
            score += 0.55
        if tokens and tokens[0] in self._REQUEST_VERBS:
            score += 0.35
        if len(tokens) > 1 and tokens[0] == "а" and tokens[1] in self._QUESTION_WORDS:
            score += 0.3
        if any(token in self._REQUEST_VERBS for token in tokens):
            score += 0.35
        if self._REQUEST_PATTERN.search(normalized):
            score += 0.35

        return min(score, 1.0)

    def _score_smalltalk(self, normalized: str, tokens: list[str]) -> float:
        score = 0.0

        if normalized in self._SMALLTALK_PHRASES:
            score += 0.9
        if tokens and tokens[0] in self._SMALLTALK_WORDS:
            score += 0.65
        if self._contains_any(tokens, {"спасибо", "благодарю"}):
            score += 0.25
        if len(tokens) <= 3 and self._contains_any(tokens, self._SMALLTALK_WORDS):
            score += 0.15
        if tokens and tokens[0] in self._QUESTION_WORDS:
            score -= 0.2

        return self._clamp_score(score)

    def _score_emotion(self, normalized: str, tokens: list[str]) -> float:
        score = 0.0
        emotion_hits = sum(1 for token in tokens if token in self._EMOTION_WORDS)

        if emotion_hits:
            score += min(0.28 * emotion_hits, 0.62)
        if self._EMOTION_PATTERN.search(normalized):
            score += 0.25
        if self._contains_any(tokens, self._FIRST_PERSON):
            score += 0.15
        if self._contains_any(tokens, {"очень", "сильно", "прям"}) and emotion_hits > 0:
            score += 0.1

        return self._clamp_score(score)

    def _score_personal_fact(self, normalized: str, tokens: list[str]) -> float:
        score = 0.0

        if self._contains_any(tokens, self._FIRST_PERSON):
            score += 0.25
        if self._contains_any(tokens, self._PERSONAL_FACT_VERBS):
            score += 0.3
        if self._PERSONAL_FACT_PATTERN.search(normalized):
            score += 0.45
        if self._AGE_PATTERN.search(normalized):
            score += 0.45
        if tokens and tokens[0] in self._QUESTION_WORDS:
            score -= 0.3
        if self._contains_any(tokens, self._EMOTION_WORDS):
            score -= 0.15

        return self._clamp_score(score)

    def _score_commentary(self, normalized: str, tokens: list[str]) -> float:
        score = 0.0
        commentary_hits = sum(1 for token in tokens if token in self._COMMENTARY_WORDS)

        if normalized in {"ну такое", "так себе", "не очень"}:
            score += 0.8
        if commentary_hits:
            score += min(0.22 * commentary_hits, 0.55)
        if self._COMMENTARY_PATTERN.search(normalized):
            score += 0.25
        if self._contains_any(tokens, self._FIRST_PERSON):
            score -= 0.08
        if tokens and tokens[0] in self._QUESTION_WORDS:
            score -= 0.2

        return self._clamp_score(score)

    def _resolve_kind(self, scores: dict[str, float]) -> tuple[str, float]:
        request_score = scores["request"]
        if request_score >= 0.65:
            return "request", request_score

        smalltalk_score = scores["smalltalk"]
        if smalltalk_score >= 0.8 and request_score < 0.6:
            return "smalltalk", smalltalk_score

        emotion_score = scores["emotion"]
        commentary_score = scores["commentary"]
        if emotion_score >= 0.7 and emotion_score >= commentary_score + 0.1:
            return "emotion", emotion_score

        fact_score = scores["personal_fact"]
        if fact_score >= 0.68 and request_score < 0.6 and emotion_score < 0.72:
            return "personal_fact", fact_score

        if commentary_score >= 0.62 and request_score < 0.6:
            return "commentary", commentary_score

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        best_kind, best_score = ranked[0]
        if best_kind != "unknown" and best_score >= 0.5:
            return best_kind, best_score

        return "unknown", max(0.3, best_score)

    @staticmethod
    def _contains_any(tokens: Iterable[str], vocabulary: set[str]) -> bool:
        return any(token in vocabulary for token in tokens)

    @staticmethod
    def _clamp_score(score: float) -> float:
        if score < 0.0:
            return 0.0
        if score > 1.0:
            return 1.0
        return score


if __name__ == "__main__":
    classifier = SegmentClassifier()

    examples = [
        "я живу в Украине",
        "что там по погодке",
        "я сегодня муторно поработал",
        "ну такое",
        "привет, как дела",
        "мне 29 лет",
        "покажи последние новости по gpu",
    ]

    for text in examples:
        result = classifier.classify([text])
        segment = result[0] if result else None
        print(f"Input: {text}")
        print(f"Classified: {segment}")
        print("-" * 56)
