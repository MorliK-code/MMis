from __future__ import annotations

import re
from dataclasses import dataclass

from .normalizer import normalize_text, tokenize_text
from .types import Fact, Location, Segment


@dataclass(slots=True)
class _LocationMatch:
    name: str
    kind: str
    confidence: float


class LocationResolver:
    """
    Resolve geographic context from message segments and memory facts.

    Sources:
    - explicit_message
    - fact_memory
    - dialogue_context
    """

    _PATTERNS: tuple[tuple[re.Pattern[str], float], ...] = (
        (re.compile(r"\bпо\s+погоде\s+в\s+([a-zа-яіїєґ0-9\- ]{2,60})"), 0.96),
        (re.compile(r"\bпогода\s+в\s+([a-zа-яіїєґ0-9\- ]{2,60})"), 0.95),
        (re.compile(r"\b(?:я\s+)?живу\s+в\s+([a-zа-яіїєґ0-9\- ]{2,60})"), 0.93),
        (re.compile(r"\b(?:я\s+)?из\s+([a-zа-яіїєґ0-9\- ]{2,60})"), 0.92),
        (re.compile(r"\b(?:в|во|из)\s+([a-zа-яіїєґ0-9\- ]{2,60})"), 0.74),
    )

    _TRAILING_SPLIT_RE = re.compile(r"\b(?:и|а|но|что|который|которая|которые)\b|[,;!?]")
    _NOISE_PREFIX_RE = re.compile(
        r"^(?:город(?:е)?|стране|страны|области|районе|регионе|города)\s+"
    )
    _NOISE_SUFFIX_RE = re.compile(r"\s+(?:сейчас|сегодня|завтра|вообще)$")

    _NON_GEO_TOKENS = {
        "офисе",
        "офис",
        "доме",
        "работе",
        "команде",
        "проекте",
        "коде",
        "погоде",
        "погодка",
        "погодке",
        "мире",
        "целом",
        "стране",
        "городе",
        "долларе",
        "курсе",
    }

    _LOCATION_CANONICAL_MAP: dict[str, str] = {
        "киев": "Kyiv",
        "киеве": "Kyiv",
        "kyiv": "Kyiv",
        "kiev": "Kyiv",
        "львов": "Lviv",
        "львове": "Lviv",
        "lviv": "Lviv",
        "одесса": "Odesa",
        "одессе": "Odesa",
        "odesa": "Odesa",
        "odessa": "Odesa",
        "харьков": "Kharkiv",
        "харькове": "Kharkiv",
        "kharkiv": "Kharkiv",
        "украина": "Ukraine",
        "украине": "Ukraine",
        "ukraine": "Ukraine",
        "польша": "Poland",
        "польше": "Poland",
        "poland": "Poland",
        "германия": "Germany",
        "германии": "Germany",
        "germany": "Germany",
        "сша": "USA",
        "usa": "USA",
        "америка": "USA",
        "америке": "USA",
        "united states": "USA",
        "великобритания": "UK",
        "англия": "UK",
        "uk": "UK",
        "лондон": "London",
        "london": "London",
        "нью йорк": "New York",
        "new york": "New York",
    }

    _CITY_CANONICAL = {
        "Kyiv",
        "Lviv",
        "Odesa",
        "Kharkiv",
        "London",
        "New York",
    }
    _COUNTRY_CANONICAL = {
        "Ukraine",
        "Poland",
        "Germany",
        "USA",
        "UK",
    }

    _FACT_LOCATION_KEYS = {
        "location_place",
        "location_city",
        "location_country",
        "location",
    }

    def extract_from_segments(self, segments: list[Segment]) -> list[Location]:
        locations: list[Location] = []

        for segment in segments:
            normalized = normalize_text(segment.text)
            if not normalized:
                continue

            matched = self._extract_from_text(normalized)
            if not matched:
                continue

            for match in matched:
                confidence = self._clamp(match.confidence * max(segment.confidence, 0.5))
                locations.append(
                    Location(
                        name=match.name,
                        kind=match.kind,
                        confidence=confidence,
                        source="explicit_message",
                    )
                )

        return self._deduplicate_locations(locations)

    def extract_from_facts(self, facts: list[Fact]) -> list[Location]:
        locations: list[Location] = []

        for fact in facts:
            if fact.key not in self._FACT_LOCATION_KEYS:
                continue
            canonical_name = self._canonicalize_location_name(fact.value)
            if not canonical_name:
                continue
            kind = self._infer_location_kind(canonical_name)
            locations.append(
                Location(
                    name=canonical_name,
                    kind=kind,
                    confidence=self._clamp(fact.confidence * 0.96),
                    source="fact_memory",
                )
            )

        return self._deduplicate_locations(locations)

    def resolve_search_location(
        self,
        message_locations: list[Location],
        fact_locations: list[Location],
        dialogue_location: str | None = None,
        fallback_location: str | None = None,
    ) -> str | None:
        best_message = self._best_location(message_locations)
        if best_message is not None:
            return best_message.name

        if dialogue_location:
            canonical_dialogue = self._canonicalize_location_name(dialogue_location)
            if canonical_dialogue:
                return canonical_dialogue

        best_fact = self._best_location(fact_locations)
        if best_fact is not None:
            return best_fact.name

        if fallback_location:
            canonical_fallback = self._canonicalize_location_name(fallback_location)
            if canonical_fallback:
                return canonical_fallback

        return None

    def _extract_from_text(self, normalized_text: str) -> list[_LocationMatch]:
        matches: list[_LocationMatch] = []

        for pattern, base_confidence in self._PATTERNS:
            for hit in pattern.finditer(normalized_text):
                candidate = hit.group(1).strip()
                resolved = self._resolve_candidate(candidate)
                if not resolved:
                    continue
                matches.append(
                    _LocationMatch(
                        name=resolved.name,
                        kind=resolved.kind,
                        confidence=base_confidence,
                    )
                )

        return matches

    def _resolve_candidate(self, candidate: str) -> _LocationMatch | None:
        candidate = normalize_text(candidate)
        if not candidate:
            return None

        candidate = self._TRAILING_SPLIT_RE.split(candidate, maxsplit=1)[0]
        candidate = self._NOISE_PREFIX_RE.sub("", candidate).strip()
        candidate = self._NOISE_SUFFIX_RE.sub("", candidate).strip()
        candidate = candidate.strip(" -_.,:;")

        if not candidate:
            return None

        tokens = tokenize_text(candidate)
        if not tokens:
            return None

        # Prefer longest n-gram from the beginning of the candidate.
        max_len = min(3, len(tokens))
        for ngram_size in range(max_len, 0, -1):
            ngram = " ".join(tokens[:ngram_size])
            canonical_name = self._LOCATION_CANONICAL_MAP.get(ngram)
            if canonical_name:
                return _LocationMatch(
                    name=canonical_name,
                    kind=self._infer_location_kind(canonical_name),
                    confidence=1.0,
                )

        # Fallback: one-token location guess if token is not clearly non-geographic.
        first_token = tokens[0]
        if first_token in self._NON_GEO_TOKENS:
            return None
        if len(first_token) < 3:
            return None

        guessed = first_token.title()
        return _LocationMatch(
            name=guessed,
            kind="place",
            confidence=0.65,
        )

    def _canonicalize_location_name(self, raw_name: str) -> str | None:
        normalized = normalize_text(raw_name)
        if not normalized:
            return None

        normalized = self._NOISE_PREFIX_RE.sub("", normalized).strip()
        normalized = self._NOISE_SUFFIX_RE.sub("", normalized).strip()
        normalized = normalized.strip(" -_.,:;")
        if not normalized:
            return None

        direct = self._LOCATION_CANONICAL_MAP.get(normalized)
        if direct:
            return direct

        tokens = tokenize_text(normalized)
        if not tokens:
            return None
        if len(tokens) == 1 and tokens[0] in self._NON_GEO_TOKENS:
            return None

        joined = " ".join(tokens[:3])
        mapped = self._LOCATION_CANONICAL_MAP.get(joined)
        if mapped:
            return mapped

        return " ".join(token.capitalize() for token in tokens[:3])

    def _infer_location_kind(self, canonical_name: str) -> str:
        if canonical_name in self._CITY_CANONICAL:
            return "city"
        if canonical_name in self._COUNTRY_CANONICAL:
            return "country"
        return "place"

    @staticmethod
    def _deduplicate_locations(locations: list[Location]) -> list[Location]:
        dedup: dict[tuple[str, str, str], Location] = {}
        for location in locations:
            key = (location.name.lower(), location.kind, location.source)
            existing = dedup.get(key)
            if existing is None or location.confidence > existing.confidence:
                dedup[key] = location
        return list(dedup.values())

    @staticmethod
    def _best_location(locations: list[Location]) -> Location | None:
        if not locations:
            return None
        return max(locations, key=lambda item: item.confidence)

    @staticmethod
    def _clamp(value: float) -> float:
        if value < 0.05:
            return 0.05
        if value > 0.99:
            return 0.99
        return round(value, 3)


if __name__ == "__main__":
    resolver = LocationResolver()

    segments = [
        Segment(text="что по погоде в киеве", kind="request", confidence=0.93),
        Segment(text="я живу в украине", kind="personal_fact", confidence=0.95),
        Segment(text="что по погодке", kind="request", confidence=0.92),
    ]
    facts = [
        Fact(key="location_place", value="Ukraine", confidence=0.9),
        Fact(key="interest", value="Python", confidence=0.88),
    ]

    message_locations = resolver.extract_from_segments(segments)
    fact_locations = resolver.extract_from_facts(facts)

    print("Message locations:", message_locations)
    print("Fact locations:", fact_locations)
    print(
        "Resolved:",
        resolver.resolve_search_location(
            message_locations=message_locations,
            fact_locations=fact_locations,
            dialogue_location="kyiv",
            fallback_location="Poland",
        ),
    )
