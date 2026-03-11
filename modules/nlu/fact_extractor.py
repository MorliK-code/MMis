from __future__ import annotations

import re

from .normalizer import normalize_text
from .types import Fact, Segment


class FactExtractor:
    """Rule-based extractor for stable user facts from personal segments."""

    _CLAUSE_END = r"(?=\b(?:и|а|но)\b|[,.;!?]|$)"

    _LOCATION_PATTERNS = (
        re.compile(rf"\b(?:я\s+)?живу\s+в\s+(.+?){_CLAUSE_END}"),
        re.compile(rf"\b(?:я\s+)?из\s+(.+?){_CLAUSE_END}"),
    )
    _INTEREST_PATTERNS = (
        re.compile(rf"\bлюблю\s+(.+?){_CLAUSE_END}"),
        re.compile(rf"\bинтересуюсь\s+(.+?){_CLAUSE_END}"),
        re.compile(rf"\bмне\s+нравится\s+(.+?){_CLAUSE_END}"),
    )
    _WORK_PATTERNS = (
        re.compile(rf"\b(?:я\s+)?работаю\s+(.+?){_CLAUSE_END}"),
    )
    _ROLE_PATTERNS: dict[str, re.Pattern[str]] = {
        "programmer": re.compile(r"\b(?:я\s+)?(?:программист|разработчик)\b"),
    }

    _LEADING_GARBAGE_RE = re.compile(
        r"^(?:как\s+бы|типа|ну|кстати|вообще|вроде|по\s+сути)\s+"
    )
    _TRAILING_GARBAGE_RE = re.compile(r"(?:\s+(?:тоже|наверное|короче|вроде))+$")
    _VALUE_CLEAN_RE = re.compile(r"[^a-zа-я0-9+\-_.\s]", re.IGNORECASE)
    _MULTI_SPACE_RE = re.compile(r"\s+")

    _MIN_VALUE_LEN = 2
    _MAX_VALUE_LEN = 48
    _MAX_VALUE_TOKENS = 6

    _EPHEMERAL_MARKERS = {
        "сегодня",
        "вчера",
        "завтра",
        "утром",
        "вечером",
        "днем",
        "ночью",
        "час",
        "часа",
        "часов",
        "день",
        "дня",
        "дней",
        "неделю",
        "неделя",
        "недели",
        "месяц",
        "месяца",
        "месяцев",
        "разово",
        "однажды",
    }
    _EPHEMERAL_SEGMENT_MARKERS = {
        "сегодня",
        "вчера",
        "завтра",
        "утром",
        "вечером",
        "днем",
        "ночью",
        "временно",
        "разово",
        "сейчас",
    }

    _STOP_VALUES = {
        "это",
        "такое",
        "все",
        "всё",
        "что",
        "ничего",
        "много",
        "немного",
        "вроде",
        "что-то",
        "где-то",
    }

    _CANONICAL_VALUE_MAP = {
        "ukraine": "Ukraine",
        "украина": "Ukraine",
        "украине": "Ukraine",
        "украины": "Ukraine",
        "python": "Python",
        "питон": "Python",
        "py": "Python",
        "программировать": "programming",
        "программирование": "programming",
        "программирую": "programming",
        "кодить": "programming",
        "кодинг": "programming",
    }

    def extract(self, segments: list[Segment]) -> list[Fact]:
        raw_facts: list[Fact] = []

        for segment in segments:
            if segment.kind != "personal_fact":
                continue
            if segment.confidence < 0.5:
                continue

            raw_facts.extend(self._extract_from_personal_segment(segment))

        return self._deduplicate(raw_facts)

    def _extract_from_personal_segment(self, segment: Segment) -> list[Fact]:
        normalized = normalize_text(segment.text)
        if not normalized:
            return []

        facts: list[Fact] = []
        has_ephemeral_context = self._has_ephemeral_context(normalized)
        facts.extend(
            self._extract_by_patterns(
                normalized=normalized,
                base_confidence=segment.confidence,
                key="location_place",
                patterns=self._LOCATION_PATTERNS,
                confidence_delta=0.0,
                stable=True,
            )
        )
        facts.extend(
            self._extract_by_patterns(
                normalized=normalized,
                base_confidence=segment.confidence,
                key="interest",
                patterns=self._INTEREST_PATTERNS,
                confidence_delta=0.01,
                stable=True,
            )
        )
        if not has_ephemeral_context:
            facts.extend(
                self._extract_by_patterns(
                    normalized=normalized,
                    base_confidence=segment.confidence,
                    key="work_context",
                    patterns=self._WORK_PATTERNS,
                    confidence_delta=-0.03,
                    stable=False,
                )
            )
        facts.extend(self._extract_roles(normalized, segment.confidence))
        return facts

    def _extract_by_patterns(
        self,
        *,
        normalized: str,
        base_confidence: float,
        key: str,
        patterns: tuple[re.Pattern[str], ...],
        confidence_delta: float,
        stable: bool,
    ) -> list[Fact]:
        extracted: list[Fact] = []
        seen_values: set[str] = set()

        for pattern in patterns:
            for match in pattern.finditer(normalized):
                raw_value = match.group(1).strip()
                canonical_value = self._normalize_value(key, raw_value)
                if not canonical_value:
                    continue
                if canonical_value in seen_values:
                    continue

                seen_values.add(canonical_value)
                extracted.append(
                    Fact(
                        key=key,
                        value=canonical_value,
                        confidence=self._clamp_confidence(base_confidence + confidence_delta),
                        stable=stable,
                    )
                )

        return extracted

    def _extract_roles(self, normalized: str, base_confidence: float) -> list[Fact]:
        facts: list[Fact] = []
        for role_value, pattern in self._ROLE_PATTERNS.items():
            if not pattern.search(normalized):
                continue
            facts.append(
                Fact(
                    key="role",
                    value=role_value,
                    confidence=self._clamp_confidence(base_confidence + 0.02),
                    stable=True,
                )
            )
        return facts

    def _normalize_value(self, key: str, raw_value: str) -> str:
        value = normalize_text(raw_value)
        if not value:
            return ""

        value = self._LEADING_GARBAGE_RE.sub("", value).strip()
        value = self._TRAILING_GARBAGE_RE.sub("", value).strip()
        value = self._VALUE_CLEAN_RE.sub(" ", value)
        value = self._MULTI_SPACE_RE.sub(" ", value).strip(" -_,.")

        if not self._is_valid_value(value):
            return ""
        if self._is_ephemeral(value):
            return ""

        canonical = self._CANONICAL_VALUE_MAP.get(value)
        if canonical:
            return canonical

        if key == "interest":
            if value in {"python", "питон"}:
                return "Python"
            if value in {"программировать", "программирование", "программирую", "кодить"}:
                return "programming"

        if key == "location_place":
            if value in {"ukraine", "украина", "украине"}:
                return "Ukraine"

        return value

    def _is_valid_value(self, value: str) -> bool:
        if not value:
            return False
        if len(value) < self._MIN_VALUE_LEN:
            return False
        if len(value) > self._MAX_VALUE_LEN:
            return False

        tokens = value.split()
        if len(tokens) > self._MAX_VALUE_TOKENS:
            return False
        if value in self._STOP_VALUES:
            return False
        if all(token in self._STOP_VALUES for token in tokens):
            return False

        return True

    def _is_ephemeral(self, value: str) -> bool:
        tokens = set(value.split())
        return any(token in self._EPHEMERAL_MARKERS for token in tokens)

    def _has_ephemeral_context(self, normalized_segment: str) -> bool:
        tokens = set(normalized_segment.split())
        return any(token in self._EPHEMERAL_SEGMENT_MARKERS for token in tokens)

    def _deduplicate(self, facts: list[Fact]) -> list[Fact]:
        if not facts:
            return []

        dedup: dict[tuple[str, str], Fact] = {}
        for fact in facts:
            dedup_key = (fact.key, fact.value)
            existing = dedup.get(dedup_key)
            if existing is None:
                dedup[dedup_key] = fact
                continue

            existing.confidence = max(existing.confidence, fact.confidence)
            existing.evidence_count += 1
            existing.stable = existing.stable and fact.stable

        return list(dedup.values())

    @staticmethod
    def _clamp_confidence(value: float) -> float:
        if value < 0.05:
            return 0.05
        if value > 0.99:
            return 0.99
        return round(value, 3)


if __name__ == "__main__":
    extractor = FactExtractor()

    test_segments = [
        Segment(text="я живу в Украине", kind="personal_fact", confidence=0.95),
        Segment(text="люблю программировать", kind="personal_fact", confidence=0.92),
        Segment(text="я из Kyiv", kind="personal_fact", confidence=0.88),
        Segment(text="мне нравится питон", kind="personal_fact", confidence=0.91),
        Segment(text="я работаю в продуктовой команде", kind="personal_fact", confidence=0.86),
        Segment(text="я программист", kind="personal_fact", confidence=0.94),
        Segment(text="сегодня работаю в офисе", kind="personal_fact", confidence=0.89),
        Segment(text="что там по погодке", kind="request", confidence=0.96),
    ]

    extracted = extractor.extract(test_segments)
    print("Extracted facts:")
    for fact in extracted:
        print(f"- {fact}")
