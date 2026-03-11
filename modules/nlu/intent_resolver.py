from __future__ import annotations

import re
from dataclasses import dataclass

from .normalizer import canonicalize_tokens, normalize_text, tokenize_text
from .types import Concept, Intent, Segment


@dataclass(slots=True)
class _SegmentFeatures:
    segment: Segment
    normalized_text: str
    canonical_tokens: list[str]
    canonical_text: str
    token_set: set[str]
    is_question_like: bool
    has_request_marker: bool


class IntentResolver:
    """Resolve high-level intents and concepts from semantic segments."""

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
        "зачем",
        "почему",
        "сколько",
        "куда",
        "откуда",
        "почем",
    }
    _REQUEST_MARKERS = {
        "подскажи",
        "скажи",
        "расскажи",
        "объясни",
        "покажи",
        "дай",
        "помоги",
        "сравни",
        "найди",
        "поищи",
        "проверить",
        "узнать",
    }

    _WEATHER_ANCHORS = {"weather", "погода", "погоде", "прогноз", "температура", "дождь", "снег", "ветер", "градус"}
    _WEATHER_DESCRIPTORS = {"жарко", "холодно", "тепло", "прохладно"}

    _SEARCH_ANCHORS = {"search", "поиск", "найди", "найти", "поищи", "искать", "загугли", "гугли"}
    _FINANCE_ANCHORS = {
        "currency",
        "currency_rate",
        "доллар",
        "usd",
        "eur",
        "евро",
        "курс",
        "рубль",
        "гривна",
        "биткоин",
        "btc",
    }
    _NEWS_ANCHORS = {"news", "новости", "новость", "сми", "headline", "хедлайны"}
    _SMALLTALK_ANCHORS = {"привет", "здравствуй", "здравствуйте", "спасибо", "пока", "добрый", "доброе"}

    _EMBEDDINGS_ANCHORS = {"embeddings", "embedding", "эмбеддинги", "эмбеддинг", "векторизация", "vector"}
    _GPU_ANCHORS = {"gpu", "видеокарта", "видеокарты", "cuda", "nvidia", "rtx"}
    _PROGRAMMING_ANCHORS = {
        "programming",
        "код",
        "кодинг",
        "кодить",
        "python",
        "java",
        "javascript",
        "backend",
        "frontend",
        "api",
        "sdk",
        "llm",
        "rag",
        "prompt",
    }
    _MEMORY_ANCHORS = {"memory", "память", "контекст", "контекста", "запомни", "кэш", "cache"}

    _DEFAULT_ALIAS_MAP: dict[str, str] = {
        "погодка": "weather",
        "погодке": "weather",
        "видюха": "gpu",
        "видюху": "gpu",
        "долларчик": "currency",
        "долларчику": "currency",
        "бакс": "currency",
        "баксы": "currency",
        "эмбеддинг": "embeddings",
        "эмбеддинги": "embeddings",
    }

    _WEATHER_QUERY_RE = re.compile(r"\b(что\s+по\s+weather|какая\s+погода|прогноз\s+погоды)\b")
    _SEARCH_RE = re.compile(r"\b(поиск|найди|поищи|search|загугли|гугли)\b")
    _FINANCE_RE = re.compile(r"\b(курс|почем|сколько\s+стоит|доллар|евро|currency|currency_rate)\b")
    _NEWS_RE = re.compile(r"\b(новости|news|что\s+нового|хедлайны)\b")
    _CODING_RE = re.compile(r"\b(код|python|llm|rag|api|sdk|gpu|embedding|embeddings)\b")
    _GENERAL_QUESTION_RE = re.compile(r"\b(что|как|когда|почему|зачем|какой|какие|сколько)\b")
    _WEATHER_FALSE_POSITIVE_RE = re.compile(
        r"\b(жарко|холодно|тепло)\b.*\b(танцевал|бегал|работал|тренировался|играл)\b"
    )

    def resolve(
        self,
        segments: list[Segment],
        alias_map: dict[str, str] | None = None,
    ) -> tuple[list[Intent], list[Concept]]:
        merged_alias_map = self._merge_alias_map(alias_map)
        features = [self._build_features(segment, merged_alias_map) for segment in segments if segment.text.strip()]

        intent_scores = {name: 0.0 for name in self._supported_intents()}
        concept_scores = {name: 0.0 for name in self._supported_concepts()}

        for feature in features:
            segment_intent_scores = self._score_intents(feature)
            segment_concept_scores = self._score_concepts(feature)

            for name, score in segment_intent_scores.items():
                if score > intent_scores[name]:
                    intent_scores[name] = score

            for name, score in segment_concept_scores.items():
                if score > concept_scores[name]:
                    concept_scores[name] = score

        intents = self._finalize_intents(intent_scores)
        concepts = self._finalize_concepts(concept_scores)
        return intents, concepts

    def _build_features(self, segment: Segment, alias_map: dict[str, str]) -> _SegmentFeatures:
        normalized = normalize_text(segment.text)
        tokens = tokenize_text(normalized)
        canonical_tokens, _ = canonicalize_tokens(tokens, alias_map)
        canonical_text = " ".join(canonical_tokens)

        question_like = (
            segment.kind == "request"
            or "?" in segment.text
            or (canonical_tokens and canonical_tokens[0] in self._QUESTION_WORDS)
            or bool(self._GENERAL_QUESTION_RE.search(canonical_text))
        )
        has_request_marker = (
            any(token in self._REQUEST_MARKERS for token in canonical_tokens)
            or bool(self._SEARCH_RE.search(canonical_text))
        )

        return _SegmentFeatures(
            segment=segment,
            normalized_text=normalized,
            canonical_tokens=canonical_tokens,
            canonical_text=canonical_text,
            token_set=set(canonical_tokens),
            is_question_like=question_like,
            has_request_marker=has_request_marker,
        )

    def _score_intents(self, feature: _SegmentFeatures) -> dict[str, float]:
        scores = {
            "weather_query": self._score_weather_query(feature),
            "search_query": self._score_search_query(feature),
            "coding_question": self._score_coding_question(feature),
            "finance_query": self._score_finance_query(feature),
            "news_query": self._score_news_query(feature),
            "smalltalk": self._score_smalltalk(feature),
            "general_question": self._score_general_question(feature),
        }
        return {k: self._clamp(v) for k, v in scores.items()}

    def _score_concepts(self, feature: _SegmentFeatures) -> dict[str, float]:
        return {
            "weather": self._clamp(self._score_weather_concept(feature)),
            "programming": self._clamp(self._score_programming_concept(feature)),
            "embeddings": self._clamp(self._score_embeddings_concept(feature)),
            "gpu": self._clamp(self._score_gpu_concept(feature)),
            "currency": self._clamp(self._score_currency_concept(feature)),
            "news": self._clamp(self._score_news_concept(feature)),
            "memory": self._clamp(self._score_memory_concept(feature)),
            "search": self._clamp(self._score_search_concept(feature)),
        }

    def _score_weather_query(self, feature: _SegmentFeatures) -> float:
        weather_concept = self._score_weather_concept(feature)
        if weather_concept < 0.45:
            return 0.0

        score = weather_concept * 0.6
        if feature.is_question_like:
            score += 0.28
        if feature.has_request_marker:
            score += 0.12
        if self._WEATHER_QUERY_RE.search(feature.canonical_text):
            score += 0.3

        if feature.segment.kind == "personal_fact" and not feature.is_question_like:
            score -= 0.35

        return score

    def _score_search_query(self, feature: _SegmentFeatures) -> float:
        if not feature.token_set.intersection(self._SEARCH_ANCHORS):
            if not self._SEARCH_RE.search(feature.canonical_text):
                return 0.0

        score = 0.45
        if feature.is_question_like:
            score += 0.2
        if feature.has_request_marker:
            score += 0.2
        if feature.segment.kind == "request":
            score += 0.1
        return score

    def _score_coding_question(self, feature: _SegmentFeatures) -> float:
        tech_score = self._score_programming_concept(feature)
        tech_score = max(tech_score, self._score_embeddings_concept(feature))
        tech_score = max(tech_score, self._score_gpu_concept(feature))
        if tech_score < 0.4:
            return 0.0

        score = tech_score * 0.65
        if feature.is_question_like:
            score += 0.24
        if self._CODING_RE.search(feature.canonical_text):
            score += 0.16
        return score

    def _score_finance_query(self, feature: _SegmentFeatures) -> float:
        currency_score = self._score_currency_concept(feature)
        if currency_score < 0.45:
            return 0.0

        score = currency_score * 0.62
        if feature.is_question_like:
            score += 0.25
        if self._FINANCE_RE.search(feature.canonical_text):
            score += 0.2
        return score

    def _score_news_query(self, feature: _SegmentFeatures) -> float:
        news_score = self._score_news_concept(feature)
        if news_score < 0.45:
            return 0.0

        score = news_score * 0.62
        if feature.is_question_like:
            score += 0.23
        if self._NEWS_RE.search(feature.canonical_text):
            score += 0.22
        return score

    def _score_smalltalk(self, feature: _SegmentFeatures) -> float:
        score = 0.0
        if feature.segment.kind == "smalltalk":
            score += 0.65
        if feature.token_set.intersection(self._SMALLTALK_ANCHORS):
            score += 0.25
        if feature.is_question_like and not feature.token_set.intersection(self._SMALLTALK_ANCHORS):
            score -= 0.2
        if self._has_any_domain_concept(feature):
            score -= 0.25
        return score

    def _score_general_question(self, feature: _SegmentFeatures) -> float:
        if not feature.is_question_like:
            return 0.0

        score = 0.52
        if self._GENERAL_QUESTION_RE.search(feature.canonical_text):
            score += 0.12
        if self._has_any_domain_concept(feature):
            score -= 0.12
        if feature.segment.kind == "request":
            score += 0.08
        return score

    def _score_weather_concept(self, feature: _SegmentFeatures) -> float:
        anchors = feature.token_set.intersection(self._WEATHER_ANCHORS)
        has_anchor = bool(anchors)
        has_descriptors = bool(feature.token_set.intersection(self._WEATHER_DESCRIPTORS))

        if self._WEATHER_FALSE_POSITIVE_RE.search(feature.canonical_text):
            return 0.0

        if not has_anchor and not (has_descriptors and feature.is_question_like):
            return 0.0

        score = 0.0
        if has_anchor:
            score += 0.58
        if "weather" in feature.token_set:
            score += 0.2
        if has_descriptors and feature.is_question_like:
            score += 0.12
        if feature.segment.kind == "request":
            score += 0.1
        if feature.segment.kind == "personal_fact" and not feature.is_question_like:
            score -= 0.28
        return score

    def _score_programming_concept(self, feature: _SegmentFeatures) -> float:
        matches = feature.token_set.intersection(self._PROGRAMMING_ANCHORS)
        if not matches:
            return 0.0
        score = 0.45 + min(len(matches), 3) * 0.1
        if feature.segment.kind == "request":
            score += 0.08
        return score

    def _score_embeddings_concept(self, feature: _SegmentFeatures) -> float:
        matches = feature.token_set.intersection(self._EMBEDDINGS_ANCHORS)
        if not matches:
            return 0.0
        score = 0.6 + min(len(matches), 2) * 0.1
        if "модели" in feature.token_set or "models" in feature.token_set:
            score += 0.08
        return score

    def _score_gpu_concept(self, feature: _SegmentFeatures) -> float:
        matches = feature.token_set.intersection(self._GPU_ANCHORS)
        if not matches:
            return 0.0
        score = 0.58 + min(len(matches), 2) * 0.1
        return score

    def _score_currency_concept(self, feature: _SegmentFeatures) -> float:
        matches = feature.token_set.intersection(self._FINANCE_ANCHORS)
        if not matches:
            return 0.0
        score = 0.56 + min(len(matches), 3) * 0.1
        return score

    def _score_news_concept(self, feature: _SegmentFeatures) -> float:
        matches = feature.token_set.intersection(self._NEWS_ANCHORS)
        if not matches:
            return 0.0
        score = 0.56 + min(len(matches), 2) * 0.1
        return score

    def _score_memory_concept(self, feature: _SegmentFeatures) -> float:
        matches = feature.token_set.intersection(self._MEMORY_ANCHORS)
        if not matches:
            return 0.0
        score = 0.52 + min(len(matches), 2) * 0.1
        return score

    def _score_search_concept(self, feature: _SegmentFeatures) -> float:
        matches = feature.token_set.intersection(self._SEARCH_ANCHORS)
        if not matches and not self._SEARCH_RE.search(feature.canonical_text):
            return 0.0
        score = 0.54 + min(len(matches), 2) * 0.1
        return score

    def _finalize_intents(self, intent_scores: dict[str, float]) -> list[Intent]:
        specialized = ("weather_query", "search_query", "coding_question", "finance_query", "news_query")

        intents: list[Intent] = []
        for name in specialized:
            score = intent_scores[name]
            if score >= 0.58:
                intents.append(Intent(name=name, confidence=round(score, 3)))

        if intent_scores["smalltalk"] >= 0.62 and not intents:
            intents.append(Intent(name="smalltalk", confidence=round(intent_scores["smalltalk"], 3)))

        if intent_scores["general_question"] >= 0.58 and not intents:
            intents.append(Intent(name="general_question", confidence=round(intent_scores["general_question"], 3)))

        if not intents:
            best_name, best_score = max(intent_scores.items(), key=lambda item: item[1])
            if best_score >= 0.45:
                intents.append(Intent(name=best_name, confidence=round(best_score, 3)))

        intents.sort(key=lambda item: item.confidence, reverse=True)
        return intents

    def _finalize_concepts(self, concept_scores: dict[str, float]) -> list[Concept]:
        concepts: list[Concept] = []
        for name, score in concept_scores.items():
            if score >= 0.55:
                concepts.append(Concept(name=name, confidence=round(score, 3)))
        concepts.sort(key=lambda item: item.confidence, reverse=True)
        return concepts

    def _has_any_domain_concept(self, feature: _SegmentFeatures) -> bool:
        return any(
            [
                bool(feature.token_set.intersection(self._WEATHER_ANCHORS)),
                bool(feature.token_set.intersection(self._PROGRAMMING_ANCHORS)),
                bool(feature.token_set.intersection(self._FINANCE_ANCHORS)),
                bool(feature.token_set.intersection(self._NEWS_ANCHORS)),
                bool(feature.token_set.intersection(self._SEARCH_ANCHORS)),
                bool(feature.token_set.intersection(self._GPU_ANCHORS)),
                bool(feature.token_set.intersection(self._EMBEDDINGS_ANCHORS)),
            ]
        )

    def _merge_alias_map(self, alias_map: dict[str, str] | None) -> dict[str, str]:
        merged = dict(self._DEFAULT_ALIAS_MAP)
        if not alias_map:
            return merged
        merged.update(alias_map)
        return merged

    @staticmethod
    def _supported_intents() -> tuple[str, ...]:
        return (
            "weather_query",
            "search_query",
            "coding_question",
            "finance_query",
            "news_query",
            "smalltalk",
            "general_question",
        )

    @staticmethod
    def _supported_concepts() -> tuple[str, ...]:
        return (
            "weather",
            "programming",
            "embeddings",
            "gpu",
            "currency",
            "news",
            "memory",
            "search",
        )

    @staticmethod
    def _clamp(score: float) -> float:
        if score < 0.0:
            return 0.0
        if score > 1.0:
            return 1.0
        return score


if __name__ == "__main__":
    resolver = IntentResolver()
    test_segments = [
        Segment(text="что там по погодке", kind="request", confidence=0.92),
        Segment(text="какие embedding модели сейчас норм", kind="request", confidence=0.9),
        Segment(text="почем долларчик", kind="request", confidence=0.88),
        Segment(text="я сегодня так жарко танцевал", kind="personal_fact", confidence=0.95),
        Segment(text="привет, как дела", kind="smalltalk", confidence=0.89),
    ]

    for segment in test_segments:
        intents, concepts = resolver.resolve([segment])
        print(f"Segment: {segment.text}")
        print(f"Intents: {intents}")
        print(f"Concepts: {concepts}")
        print("-" * 56)
