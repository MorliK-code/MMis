from __future__ import annotations

from dataclasses import dataclass

from .normalizer import canonicalize_tokens, normalize_text, token_variants, tokenize_text
from .types import Concept, Intent, Segment


@dataclass(slots=True)
class _SegmentFeatures:
    segment: Segment
    normalized_text: str
    canonical_tokens: list[str]
    canonical_text: str
    token_set: set[str]
    token_forms: set[str]
    is_question_like: bool
    has_request_marker: bool
    has_negation: bool
    is_short: bool


class IntentResolver:
    """Resolve high-level intents and concepts from semantic segments."""

    _QUESTION_WORDS = {
        "\u0447\u0442\u043e",
        "\u043a\u0442\u043e",
        "\u0433\u0434\u0435",
        "\u043a\u043e\u0433\u0434\u0430",
        "\u043a\u0430\u043a",
        "\u043a\u0430\u043a\u043e\u0439",
        "\u043a\u0430\u043a\u0430\u044f",
        "\u043a\u0430\u043a\u0438\u0435",
        "\u043a\u0430\u043a\u0443\u044e",
        "\u043a\u0430\u043a\u0438\u043c",
        "\u0437\u0430\u0447\u0435\u043c",
        "\u043f\u043e\u0447\u0435\u043c\u0443",
        "\u0441\u043a\u043e\u043b\u044c\u043a\u043e",
        "\u043a\u0443\u0434\u0430",
        "\u043e\u0442\u043a\u0443\u0434\u0430",
        "\u043b\u0438",
        "\u0440\u0430\u0437\u0432\u0435",
    }
    _REQUEST_MARKERS = {
        "\u043f\u043e\u0434\u0441\u043a\u0430\u0436\u0438",
        "\u0441\u043a\u0430\u0436\u0438",
        "\u0440\u0430\u0441\u0441\u043a\u0430\u0436\u0438",
        "\u043e\u0431\u044a\u044f\u0441\u043d\u0438",
        "\u043f\u043e\u043a\u0430\u0436\u0438",
        "\u0434\u0430\u0439",
        "\u043f\u043e\u043c\u043e\u0433\u0438",
        "\u0441\u0440\u0430\u0432\u043d\u0438",
        "\u043d\u0430\u0439\u0434\u0438",
        "\u043f\u043e\u0438\u0449\u0438",
        "\u043f\u0440\u043e\u0432\u0435\u0440\u044c",
        "\u0443\u0437\u043d\u0430\u0442\u044c",
        "search",
        "google",
        "googl",
    }
    _NEGATION_TOKENS = {"\u043d\u0435", "\u043d\u0438", "no", "not"}

    _SMALLTALK_EXACT = {
        "\u043f\u0440\u0438\u0432\u0435\u0442",
        "\u0437\u0434\u043e\u0440\u043e\u0432\u0430",
        "\u0437\u0434\u0440\u0430\u0432\u0441\u0442\u0432\u0443\u0439",
        "\u0437\u0434\u0440\u0430\u0432\u0441\u0442\u0432\u0443\u0439\u0442\u0435",
        "\u0441\u043f\u0430\u0441\u0438\u0431\u043e",
        "\u043f\u043e\u043a\u0430",
        "hello",
        "hi",
        "thanks",
    }
    _SMALLTALK_STEMS = (
        "\u043f\u0440\u0438\u0432\u0435\u0442",
        "\u0437\u0434\u043e\u0440\u043e\u0432",
        "\u0434\u043e\u0431\u0440",
        "\u0441\u043f\u0430\u0441\u0438\u0431",
        "hello",
        "thank",
    )

    _WEATHER_EXACT = {
        "weather",
        "\u043f\u043e\u0433\u043e\u0434\u0430",
        "\u043f\u0440\u043e\u0433\u043d\u043e\u0437",
        "\u0434\u043e\u0436\u0434\u044c",
        "\u0441\u043d\u0435\u0433",
        "\u0432\u0435\u0442\u0435\u0440",
        "\u0437\u043e\u043d\u0442",
        "\u0437\u043e\u043d\u0442\u0438\u043a",
    }
    _WEATHER_STEMS = (
        "weather",
        "\u043f\u043e\u0433\u043e\u0434",
        "\u043f\u0440\u043e\u0433\u043d\u043e\u0437",
        "\u0442\u0435\u043c\u043f\u0435\u0440\u0430\u0442\u0443\u0440",
        "\u0434\u043e\u0436\u0434",
        "\u0441\u043d\u0435\u0433",
        "\u0432\u0435\u0442\u0435\u0440",
        "\u0433\u0440\u0430\u0434\u0443\u0441",
        "\u0436\u0430\u0440",
        "\u0436\u0430\u0440\u043a",
        "\u0437\u043d\u043e\u0439",
        "\u0445\u043e\u043b\u043e\u0434",
        "\u043c\u043e\u0440\u043e\u0437",
        "\u0442\u0435\u043f\u043b",
        "\u0434\u0443\u0448\u043d",
        "\u043b\u0438\u0432\u043d",
        "\u0433\u0440\u043e\u0437",
        "\u0437\u043e\u043d\u0442",
    )
    _WEATHER_CONTEXT_STEMS = (
        "\u0443\u043b\u0438\u0446",
        "\u043d\u0430\u0440\u0443\u0436",
        "\u0441\u0435\u0439\u0447\u0430\u0441",
        "\u0441\u0435\u0433\u043e\u0434\u043d\u044f",
        "\u0437\u0430\u0432\u0442\u0440\u0430",
        "\u0432\u0435\u0447\u0435\u0440",
        "\u0443\u0442\u0440",
        "\u0431\u0440\u0430\u0442",
        "\u043e\u0434\u0435\u0432",
        "\u043a\u0443\u0440\u0442",
        "\u043c\u0435\u0440\u0437",
        "\u0436\u0430\u0440\u044b",
    )
    _WEATHER_DESCRIPTOR_STEMS = (
        "\u0436\u0430\u0440",
        "\u0436\u0430\u0440\u043a",
        "\u0445\u043e\u043b\u043e\u0434",
        "\u0445\u043e\u043b\u043e\u0434\u043d",
        "\u0442\u0435\u043f\u043b",
        "\u0434\u0443\u0448\u043d",
        "\u0437\u043d\u043e\u0439",
        "\u043b\u0438\u0432\u0435\u043d",
        "\u043c\u043e\u0440\u043e\u0437",
    )
    _WEATHER_ACTIVITY_NEGATIVE_STEMS = (
        "\u0442\u0430\u043d\u0446",
        "\u0431\u0435\u0433",
        "\u0442\u0440\u0435\u043d\u0438\u0440",
        "\u0438\u0433\u0440",
        "\u0440\u0430\u0431\u043e\u0442\u0430",
        "\u043f\u043e\u0442\u0435\u043b",
    )
    _WEATHER_BODY_NEGATIVE_STEMS = (
        "\u0433\u043e\u043b\u043e\u0432",
        "\u0442\u0435\u043b",
        "\u0434\u0430\u0432\u043b",
        "\u0431\u043e\u043b\u0438\u0442",
    )
    _WEATHER_RISK_STEMS = (
        "\u0441\u0434\u043e\u0445\u043d",
        "\u0437\u0430\u043c\u0435\u0440\u0437",
        "\u043f\u0435\u0440\u0435\u0436\u0438\u0432",
        "\u043f\u043b\u0430\u0432\u043b",
        "\u0436\u0430\u0440",
    )

    _SEARCH_EXACT = {"search", "\u043f\u043e\u0438\u0441\u043a", "\u043d\u0430\u0439\u0434\u0438", "\u043d\u0430\u0439\u0442\u0438", "\u043f\u043e\u0438\u0449\u0438"}
    _SEARCH_STEMS = ("search", "\u043f\u043e\u0438\u0441\u043a", "\u043d\u0430\u0439\u0434", "\u0438\u0449", "\u0433\u0443\u0433\u043b")
    _FINANCE_EXACT = {"currency", "usd", "eur", "btc", "\u043a\u0443\u0440\u0441", "\u0434\u043e\u043b\u043b\u0430\u0440", "\u0435\u0432\u0440\u043e"}
    _FINANCE_STEMS = ("currency", "usd", "eur", "btc", "\u043a\u0443\u0440\u0441", "\u0434\u043e\u043b\u043b\u0430\u0440", "\u0435\u0432\u0440", "\u0433\u0440\u0438\u0432\u043d", "\u0440\u0443\u0431\u043b", "\u0431\u0438\u0442\u043a\u043e")
    _NEWS_EXACT = {"news", "\u043d\u043e\u0432\u043e\u0441\u0442\u0438", "\u0441\u043c\u0438", "headline"}
    _NEWS_STEMS = ("news", "\u043d\u043e\u0432\u043e\u0441\u0442", "\u0445\u0435\u0434\u043b\u0430\u0439\u043d", "\u0441\u043c\u0438")
    _PROGRAMMING_EXACT = {"python", "java", "javascript", "backend", "frontend", "api", "sdk", "llm", "rag", "prompt", "\u043a\u043e\u0434"}
    _PROGRAMMING_STEMS = ("python", "java", "javascript", "backend", "frontend", "api", "sdk", "llm", "rag", "prompt", "\u043a\u043e\u0434", "\u043f\u0440\u043e\u0433\u0440\u0430\u043c", "\u0440\u0430\u0437\u0440\u0430\u0431\u043e\u0442")
    _EMBEDDINGS_EXACT = {"embeddings", "embedding", "vector", "\u044d\u043c\u0431\u0435\u0434\u0434\u0438\u043d\u0433"}
    _EMBEDDINGS_STEMS = ("embedding", "embeddings", "vector", "\u044d\u043c\u0431\u0435\u0434", "\u0432\u0435\u043a\u0442\u043e\u0440")
    _GPU_EXACT = {"gpu", "cuda", "nvidia", "rtx"}
    _GPU_STEMS = ("gpu", "cuda", "nvidia", "rtx", "\u0432\u0438\u0434\u0435\u043e\u043a\u0430\u0440\u0442", "\u0432\u0438\u0434\u044e\u0445")
    _MEMORY_EXACT = {"memory", "cache", "\u043f\u0430\u043c\u044f\u0442\u044c", "\u043a\u043e\u043d\u0442\u0435\u043a\u0441\u0442"}
    _MEMORY_STEMS = ("memory", "cache", "\u043f\u0430\u043c\u044f\u0442", "\u043a\u043e\u043d\u0442\u0435\u043a\u0441\u0442", "\u0437\u0430\u043f\u043e\u043c\u043d", "\u043a\u044d\u0448")

    _DEFAULT_ALIAS_MAP: dict[str, str] = {
        "\u0432\u0438\u0434\u044e\u0445\u0430": "gpu",
        "\u0432\u0438\u0434\u044e\u0445\u0443": "gpu",
        "\u0431\u0430\u043a\u0441": "currency",
        "\u0431\u0430\u043a\u0441\u044b": "currency",
        "\u044d\u043c\u0431\u0435\u0434\u044b": "embeddings",
    }

    def resolve(
        self,
        segments: list[Segment],
        alias_map: dict[str, str] | None = None,
    ) -> tuple[list[Intent], list[Concept]]:
        merged_alias_map = self._merge_alias_map(alias_map)
        features = [self._build_features(segment, merged_alias_map) for segment in segments if segment.text.strip()]

        intent_scores = {name: (0.0, "rule") for name in self._supported_intents()}
        concept_scores = {name: (0.0, "rule") for name in self._supported_concepts()}

        for feature in features:
            for name, (score, reasons) in self._score_intents(feature).items():
                if score > intent_scores[name][0]:
                    intent_scores[name] = (score, self._format_source(name, reasons))

            for name, (score, reasons) in self._score_concepts(feature).items():
                if score > concept_scores[name][0]:
                    concept_scores[name] = (score, self._format_source(name, reasons))

        intents = self._finalize_intents(intent_scores)
        concepts = self._finalize_concepts(concept_scores)
        return intents, concepts

    def _build_features(self, segment: Segment, alias_map: dict[str, str]) -> _SegmentFeatures:
        normalized = normalize_text(segment.text)
        tokens = tokenize_text(normalized)
        canonical_tokens, _ = canonicalize_tokens(tokens, alias_map)
        canonical_text = " ".join(canonical_tokens)
        forms: set[str] = set()
        for token in canonical_tokens:
            forms.update(token_variants(token))

        is_question_like = (
            segment.kind == "request"
            or "?" in segment.text
            or any(token in self._QUESTION_WORDS for token in canonical_tokens[:2])
            or "\u043b\u0438" in canonical_tokens
        )
        has_request_marker = segment.kind == "request" or any(token in self._REQUEST_MARKERS for token in canonical_tokens)
        has_negation = any(token in self._NEGATION_TOKENS for token in canonical_tokens)

        return _SegmentFeatures(
            segment=segment,
            normalized_text=normalized,
            canonical_tokens=canonical_tokens,
            canonical_text=canonical_text,
            token_set=set(canonical_tokens),
            token_forms=forms,
            is_question_like=is_question_like,
            has_request_marker=has_request_marker,
            has_negation=has_negation,
            is_short=len(canonical_tokens) <= 3,
        )

    def _score_intents(self, feature: _SegmentFeatures) -> dict[str, tuple[float, list[str]]]:
        return {
            "weather_query": self._score_weather_query(feature),
            "search_query": self._score_search_query(feature),
            "coding_question": self._score_coding_question(feature),
            "finance_query": self._score_finance_query(feature),
            "news_query": self._score_news_query(feature),
            "smalltalk": self._score_smalltalk(feature),
            "general_question": self._score_general_question(feature),
        }

    def _score_concepts(self, feature: _SegmentFeatures) -> dict[str, tuple[float, list[str]]]:
        return {
            "weather": self._score_weather_concept(feature),
            "programming": self._score_programming_concept(feature),
            "embeddings": self._score_embeddings_concept(feature),
            "gpu": self._score_gpu_concept(feature),
            "currency": self._score_currency_concept(feature),
            "news": self._score_news_concept(feature),
            "memory": self._score_memory_concept(feature),
            "search": self._score_search_concept(feature),
        }

    def _score_weather_query(self, feature: _SegmentFeatures) -> tuple[float, list[str]]:
        concept_score, concept_reasons = self._score_weather_concept(feature)
        if concept_score < 0.34:
            return 0.0, []

        score = concept_score * 0.52
        reasons = list(concept_reasons)
        colloquial_query = self._weather_colloquial_query_score(feature)
        risk_score = self._match_forms(feature, set(), self._WEATHER_RISK_STEMS)[0]

        if feature.is_question_like:
            score += 0.24
            reasons.append("question")
        if feature.has_request_marker:
            score += 0.12
            reasons.append("request")
        if feature.segment.kind == "request":
            score += 0.08
        if colloquial_query > 0.0:
            score += colloquial_query
            reasons.append("colloquial")
        if risk_score > 0.0 and feature.is_question_like:
            score += 0.14
            reasons.append("risk")
        if feature.is_short and not feature.is_question_like and colloquial_query <= 0.0:
            score -= 0.20
        if feature.segment.kind == "personal_fact" and not feature.is_question_like and colloquial_query <= 0.0:
            score -= 0.28

        return self._clamp(score), reasons

    def _score_search_query(self, feature: _SegmentFeatures) -> tuple[float, list[str]]:
        concept_score, reasons = self._score_search_concept(feature)
        if concept_score < 0.36:
            return 0.0, []

        score = concept_score * 0.58
        out_reasons = list(reasons)
        if feature.is_question_like:
            score += 0.18
            out_reasons.append("question")
        if feature.has_request_marker:
            score += 0.18
            out_reasons.append("request")
        if feature.segment.kind == "request":
            score += 0.08
        return self._clamp(score), out_reasons

    def _score_coding_question(self, feature: _SegmentFeatures) -> tuple[float, list[str]]:
        programming_score, programming_reasons = self._score_programming_concept(feature)
        embeddings_score, embeddings_reasons = self._score_embeddings_concept(feature)
        gpu_score, gpu_reasons = self._score_gpu_concept(feature)
        best_score = programming_score
        reasons = list(programming_reasons)
        for candidate_score, candidate_reasons in ((embeddings_score, embeddings_reasons), (gpu_score, gpu_reasons)):
            if candidate_score > best_score:
                best_score = candidate_score
                reasons = list(candidate_reasons)
        if best_score < 0.34:
            return 0.0, []

        score = best_score * 0.6
        if feature.is_question_like:
            score += 0.22
            reasons.append("question")
        if feature.has_request_marker:
            score += 0.08
            reasons.append("request")
        return self._clamp(score), reasons

    def _score_finance_query(self, feature: _SegmentFeatures) -> tuple[float, list[str]]:
        concept_score, reasons = self._score_currency_concept(feature)
        if concept_score < 0.38:
            return 0.0, []

        score = concept_score * 0.6
        if feature.is_question_like:
            score += 0.24
            reasons.append("question")
        if feature.has_request_marker:
            score += 0.08
        return self._clamp(score), reasons

    def _score_news_query(self, feature: _SegmentFeatures) -> tuple[float, list[str]]:
        concept_score, reasons = self._score_news_concept(feature)
        if concept_score < 0.38:
            return 0.0, []

        score = concept_score * 0.6
        if feature.is_question_like:
            score += 0.22
            reasons.append("question")
        if feature.has_request_marker:
            score += 0.06
        return self._clamp(score), reasons

    def _score_smalltalk(self, feature: _SegmentFeatures) -> tuple[float, list[str]]:
        smalltalk_signal, reasons = self._match_forms(feature, self._SMALLTALK_EXACT, self._SMALLTALK_STEMS)
        if smalltalk_signal <= 0.0 and feature.segment.kind != "smalltalk":
            return 0.0, []

        score = 0.0
        out_reasons = list(reasons)
        if feature.segment.kind == "smalltalk":
            score += 0.58
            out_reasons.append("segment_kind")
        if smalltalk_signal > 0.0:
            score += 0.22 + (0.3 * smalltalk_signal)
            out_reasons.append("greeting")
        if feature.is_question_like and not smalltalk_signal:
            score -= 0.18
        if self._has_any_domain_concept(feature):
            score -= 0.26
            out_reasons.append("domain_overlap")
        return self._clamp(score), out_reasons

    def _score_general_question(self, feature: _SegmentFeatures) -> tuple[float, list[str]]:
        if not feature.is_question_like:
            return 0.0, []

        score = 0.5
        reasons = ["question"]
        if feature.has_request_marker:
            score += 0.1
            reasons.append("request")
        if self._has_any_domain_concept(feature):
            score -= 0.12
            reasons.append("domain_overlap")
        return self._clamp(score), reasons

    def _score_weather_concept(self, feature: _SegmentFeatures) -> tuple[float, list[str]]:
        anchor_score, anchor_reasons = self._match_forms(feature, self._WEATHER_EXACT, self._WEATHER_STEMS)
        context_score, context_reasons = self._match_forms(feature, set(), self._WEATHER_CONTEXT_STEMS)
        descriptor_score, descriptor_reasons = self._match_forms(feature, set(), self._WEATHER_DESCRIPTOR_STEMS)
        risk_score, risk_reasons = self._match_forms(feature, set(), self._WEATHER_RISK_STEMS)
        negative_score = self._weather_negative_score(feature, anchor_score, context_score)

        if max(anchor_score, context_score, descriptor_score, risk_score) <= 0.0:
            return 0.0, []

        score = 0.0
        reasons: list[str] = []
        if anchor_score > 0.0:
            score += 0.44 + (0.22 * anchor_score)
            reasons.extend(anchor_reasons)
        if context_score > 0.0:
            score += 0.08 + (0.16 * context_score)
            reasons.extend(context_reasons)
        if descriptor_score > 0.0:
            score += 0.08 + (0.18 * descriptor_score)
            reasons.extend(descriptor_reasons)
        if risk_score > 0.0:
            score += 0.12
            reasons.extend(risk_reasons)
        colloquial_query = self._weather_colloquial_query_score(feature)
        if colloquial_query > 0.0:
            score += colloquial_query
            reasons.append("colloquial")
        if colloquial_query > 0.0 and context_score > 0.0:
            score += 0.08
            reasons.append("contextual_query")
        if feature.is_question_like:
            score += 0.05
        score -= negative_score

        descriptor_only = descriptor_score > 0.0 and anchor_score < 0.35 and context_score < 0.25
        if descriptor_only and not feature.is_question_like and colloquial_query <= 0.0:
            score *= 0.55
            reasons.append("descriptor_only")
        weak_single_token = (
            len(feature.canonical_tokens) == 1
            and anchor_score > 0.0
            and context_score <= 0.0
            and descriptor_score <= 0.0
            and risk_score <= 0.0
            and colloquial_query <= 0.0
            and not feature.is_question_like
            and feature.canonical_tokens[0] not in self._WEATHER_EXACT
        )
        if weak_single_token:
            score *= 0.68
            reasons.append("single_token_weak")

        return self._clamp(score), reasons

    def _score_programming_concept(self, feature: _SegmentFeatures) -> tuple[float, list[str]]:
        return self._score_generic_concept(feature, self._PROGRAMMING_EXACT, self._PROGRAMMING_STEMS, base=0.42)

    def _score_embeddings_concept(self, feature: _SegmentFeatures) -> tuple[float, list[str]]:
        return self._score_generic_concept(feature, self._EMBEDDINGS_EXACT, self._EMBEDDINGS_STEMS, base=0.52)

    def _score_gpu_concept(self, feature: _SegmentFeatures) -> tuple[float, list[str]]:
        return self._score_generic_concept(feature, self._GPU_EXACT, self._GPU_STEMS, base=0.5)

    def _score_currency_concept(self, feature: _SegmentFeatures) -> tuple[float, list[str]]:
        return self._score_generic_concept(feature, self._FINANCE_EXACT, self._FINANCE_STEMS, base=0.48)

    def _score_news_concept(self, feature: _SegmentFeatures) -> tuple[float, list[str]]:
        return self._score_generic_concept(feature, self._NEWS_EXACT, self._NEWS_STEMS, base=0.48)

    def _score_memory_concept(self, feature: _SegmentFeatures) -> tuple[float, list[str]]:
        return self._score_generic_concept(feature, self._MEMORY_EXACT, self._MEMORY_STEMS, base=0.44)

    def _score_search_concept(self, feature: _SegmentFeatures) -> tuple[float, list[str]]:
        return self._score_generic_concept(feature, self._SEARCH_EXACT, self._SEARCH_STEMS, base=0.46)

    def _score_generic_concept(
        self,
        feature: _SegmentFeatures,
        exact_anchors: set[str],
        stem_anchors: tuple[str, ...],
        *,
        base: float,
    ) -> tuple[float, list[str]]:
        match_score, reasons = self._match_forms(feature, exact_anchors, stem_anchors)
        if match_score <= 0.0:
            return 0.0, []

        score = base + (0.24 * match_score)
        if feature.segment.kind == "request":
            score += 0.06
            reasons.append("request")
        return self._clamp(score), reasons

    def _weather_negative_score(self, feature: _SegmentFeatures, anchor_score: float, context_score: float) -> float:
        activity_score = self._match_forms(feature, set(), self._WEATHER_ACTIVITY_NEGATIVE_STEMS)[0]
        body_score = self._match_forms(feature, set(), self._WEATHER_BODY_NEGATIVE_STEMS)[0]
        penalty = 0.0
        if feature.segment.kind in {"personal_fact", "emotion"} and not feature.is_question_like:
            penalty += 0.18
        if activity_score > 0.0 and anchor_score < 0.45 and context_score < 0.3:
            penalty += 0.22
        if body_score > 0.0 and anchor_score < 0.45 and context_score < 0.3:
            penalty += 0.18
        return penalty

    def _weather_colloquial_query_score(self, feature: _SegmentFeatures) -> float:
        tokens = feature.canonical_tokens
        forms = feature.token_forms
        if self._sequence_contains(tokens, ("\u0437\u043e\u043d\u0442\u0438\u043a", "\u0431\u0440\u0430\u0442\u044c")):
            return 0.36
        if self._sequence_contains(tokens, ("\u0447\u0442\u043e", "\u043f\u043e")) and any(form.startswith("\u0443\u043b\u0438\u0446") for form in forms):
            return 0.34
        if self._sequence_contains(tokens, ("\u0447\u0442\u043e", "\u043f\u043e")) and any(form.startswith("\u043f\u043e\u0433\u043e\u0434") for form in forms):
            return 0.32
        if feature.is_question_like and any(form.startswith("\u0437\u043e\u043d\u0442") for form in forms):
            return 0.22
        return 0.0

    def _match_forms(
        self,
        feature: _SegmentFeatures,
        exact_anchors: set[str],
        stem_anchors: tuple[str, ...],
    ) -> tuple[float, list[str]]:
        exact_hits = {form for form in feature.token_forms if form in exact_anchors}
        stem_hits = {
            form
            for form in feature.token_forms
            if any(form.startswith(stem) or stem in form for stem in stem_anchors if stem)
        }
        fuzzy_score = 0.0
        if not exact_hits:
            for form in feature.token_forms:
                for anchor in exact_anchors:
                    fuzzy_score = max(fuzzy_score, self._soft_similarity(form, anchor))

        score = 0.0
        reasons: list[str] = []
        if exact_hits:
            score += 0.34 + (0.08 * min(len(exact_hits), 3))
            reasons.append("anchor")
        if stem_hits:
            score += 0.22 + (0.05 * min(len(stem_hits), 3))
            reasons.append("stem")
        if fuzzy_score >= 0.78:
            score += 0.16 + (0.08 * fuzzy_score)
            reasons.append("fuzzy")
        return self._clamp(score), reasons

    def _has_any_domain_concept(self, feature: _SegmentFeatures) -> bool:
        concept_scores = (
            self._score_weather_concept(feature)[0],
            self._score_programming_concept(feature)[0],
            self._score_currency_concept(feature)[0],
            self._score_news_concept(feature)[0],
            self._score_search_concept(feature)[0],
            self._score_gpu_concept(feature)[0],
            self._score_embeddings_concept(feature)[0],
            self._score_memory_concept(feature)[0],
        )
        return max(concept_scores) >= 0.4

    def _finalize_intents(self, intent_scores: dict[str, tuple[float, str]]) -> list[Intent]:
        specialized = ("weather_query", "search_query", "coding_question", "finance_query", "news_query")

        intents: list[Intent] = []
        for name in specialized:
            score, source = intent_scores[name]
            if score >= 0.56:
                intents.append(Intent(name=name, confidence=round(score, 3), source=source))

        smalltalk_score, smalltalk_source = intent_scores["smalltalk"]
        if smalltalk_score >= 0.62 and not intents:
            intents.append(Intent(name="smalltalk", confidence=round(smalltalk_score, 3), source=smalltalk_source))

        general_score, general_source = intent_scores["general_question"]
        if general_score >= 0.58 and not intents:
            intents.append(Intent(name="general_question", confidence=round(general_score, 3), source=general_source))

        if not intents:
            best_name, (best_score, best_source) = max(intent_scores.items(), key=lambda item: item[1][0])
            if best_score >= 0.45:
                intents.append(Intent(name=best_name, confidence=round(best_score, 3), source=best_source))

        intents.sort(key=lambda item: item.confidence, reverse=True)
        return intents

    def _finalize_concepts(self, concept_scores: dict[str, tuple[float, str]]) -> list[Concept]:
        concepts: list[Concept] = []
        for name, (score, source) in concept_scores.items():
            if score >= 0.52:
                concepts.append(Concept(name=name, confidence=round(score, 3), source=source))
        concepts.sort(key=lambda item: item.confidence, reverse=True)
        return concepts

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
    def _sequence_contains(tokens: list[str], pattern: tuple[str, ...]) -> bool:
        if not tokens or not pattern or len(tokens) < len(pattern):
            return False
        width = len(pattern)
        for index in range(0, len(tokens) - width + 1):
            if tuple(tokens[index : index + width]) == pattern:
                return True
        return False

    @staticmethod
    def _char_bigrams(value: str) -> set[str]:
        src = str(value or "")
        if len(src) < 2:
            return {src} if src else set()
        return {src[index : index + 2] for index in range(len(src) - 1)}

    def _soft_similarity(self, left: str, right: str) -> float:
        a = str(left or "")
        b = str(right or "")
        if len(a) < 4 or len(b) < 4:
            return 0.0
        a_bigrams = self._char_bigrams(a)
        b_bigrams = self._char_bigrams(b)
        if not a_bigrams or not b_bigrams:
            return 0.0
        inter = len(a_bigrams.intersection(b_bigrams))
        union = max(1, len(a_bigrams.union(b_bigrams)))
        return inter / union

    @staticmethod
    def _format_source(name: str, reasons: list[str]) -> str:
        unique: list[str] = []
        for reason in reasons:
            item = str(reason or "").strip().lower()
            if item and item not in unique:
                unique.append(item)
        if not unique:
            return "rule"
        summary = "+".join(unique[:5])
        return f"rule:{name}({summary})"[:96]

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
        Segment(text="\u0447\u0442\u043e \u0442\u0430\u043c \u043f\u043e \u043f\u043e\u0433\u043e\u0434\u043a\u0435", kind="request", confidence=0.92),
        Segment(text="\u0448\u043e \u043f\u043e \u0443\u043b\u0438\u0446\u0435", kind="request", confidence=0.9),
        Segment(text="\u0437\u043e\u043d\u0442\u0438\u043a \u0431\u0440\u0430\u0442\u044c?", kind="request", confidence=0.88),
        Segment(text="\u0436\u0430\u0440\u043a\u043e", kind="commentary", confidence=0.82),
        Segment(text="\u043f\u0440\u0438\u0432\u0435\u0442, \u043a\u0430\u043a \u0434\u0435\u043b\u0430", kind="smalltalk", confidence=0.89),
    ]

    for segment in test_segments:
        intents, concepts = resolver.resolve([segment])
        print(f"Segment: {segment.text}")
        print(f"Intents: {intents}")
        print(f"Concepts: {concepts}")
        print("-" * 56)
