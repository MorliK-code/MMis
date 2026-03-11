from __future__ import annotations

from dataclasses import replace

from modules.nlu.normalizer import normalize_text
from modules.nlu.types import Fact


class MemoryPolicy:
    """Policy for selecting durable user facts for long-term memory."""

    _CORE_LONG_TERM_KEYS = {
        "location_place",
        "interest",
        "role",
    }
    _WORK_CONTEXT_KEYS = {
        "work_context",
        "work_domain",
        "work_stack",
        "work_role",
    }
    _EXPLICIT_PREFERENCE_KEYS = {
        "preference",
        "preferred_tool",
        "preferred_language",
        "preferred_editor",
        "preferred_style",
    }
    _DENY_KEYS = {
        "today_status",
        "emotion",
        "mood",
        "smalltalk",
        "commentary",
        "event",
        "last_event",
        "temporary_state",
        "one_time_event",
    }
    _EPHEMERAL_MARKERS = {
        "today",
        "yesterday",
        "tomorrow",
        "сегодня",
        "вчера",
        "завтра",
        "утром",
        "вечером",
        "ночью",
        "сейчас",
        "разово",
        "одноразово",
        "временно",
        "temporary",
        "once",
    }
    _NOISE_VALUES = {
        "",
        "ok",
        "норм",
        "нормально",
        "такое",
        "не знаю",
        "unknown",
        "none",
    }

    def select_facts_for_long_term(
        self,
        facts: list[Fact],
        min_confidence: float = 0.75,
    ) -> list[Fact]:
        if not facts:
            return []

        deduped = self.dedupe_facts(facts)
        selected: list[Fact] = []

        for fact in deduped:
            if not self._passes_confidence(fact, min_confidence):
                continue
            if not self._is_allowed_key(fact.key):
                continue
            if not self._is_stable_enough(fact):
                continue
            if self._is_ephemeral_or_noise(fact):
                continue
            selected.append(fact)

        return selected

    def dedupe_facts(self, facts: list[Fact]) -> list[Fact]:
        if not facts:
            return []

        dedup: dict[tuple[str, str], Fact] = {}

        for fact in facts:
            normalized_key = self._norm_key(fact.key)
            normalized_value = self._norm_value(fact.value)
            if not normalized_key or not normalized_value:
                continue

            dedupe_key = (normalized_key, normalized_value)
            existing = dedup.get(dedupe_key)
            if existing is None:
                dedup[dedupe_key] = replace(
                    fact,
                    key=normalized_key,
                    value=self._canonical_value_preserve_case(fact.value),
                    confidence=round(float(fact.confidence), 3),
                )
                continue

            merged_confidence = max(existing.confidence, fact.confidence)
            merged_evidence = max(1, int(existing.evidence_count)) + max(1, int(fact.evidence_count))
            merged_stable = existing.stable or fact.stable or merged_evidence >= 3

            # Keep best-looking value formatting from the higher-confidence fact.
            if fact.confidence > existing.confidence:
                merged_value = self._canonical_value_preserve_case(fact.value)
                merged_source = fact.source
            else:
                merged_value = existing.value
                merged_source = existing.source

            dedup[dedupe_key] = replace(
                existing,
                value=merged_value,
                confidence=round(float(merged_confidence), 3),
                stable=merged_stable,
                evidence_count=merged_evidence,
                source=merged_source,
            )

        return list(dedup.values())

    def _is_allowed_key(self, key: str) -> bool:
        key_norm = self._norm_key(key)
        if not key_norm:
            return False
        if key_norm in self._DENY_KEYS:
            return False
        if key_norm in self._CORE_LONG_TERM_KEYS:
            return True
        if key_norm in self._WORK_CONTEXT_KEYS:
            return True
        if key_norm in self._EXPLICIT_PREFERENCE_KEYS:
            return True
        if key_norm.startswith("preference_") or key_norm.startswith("preferred_"):
            return True
        if key_norm.startswith("work_"):
            return True
        return False

    @staticmethod
    def _passes_confidence(fact: Fact, min_confidence: float) -> bool:
        return float(fact.confidence) >= float(min_confidence)

    def _is_stable_enough(self, fact: Fact) -> bool:
        key_norm = self._norm_key(fact.key)

        # Core personal facts should be explicitly stable.
        if key_norm in self._CORE_LONG_TERM_KEYS:
            return bool(fact.stable)

        # Preferences/work context can become durable either by explicit stability
        # or repeated confirmations in evidence.
        if key_norm in self._WORK_CONTEXT_KEYS or key_norm in self._EXPLICIT_PREFERENCE_KEYS:
            return bool(fact.stable) or int(fact.evidence_count) >= 3

        if key_norm.startswith("preference_") or key_norm.startswith("preferred_"):
            return bool(fact.stable) or int(fact.evidence_count) >= 3

        if key_norm.startswith("work_"):
            return bool(fact.stable) or int(fact.evidence_count) >= 3

        return False

    def _is_ephemeral_or_noise(self, fact: Fact) -> bool:
        value_norm = self._norm_value(fact.value)
        if not value_norm:
            return True
        if value_norm in self._NOISE_VALUES:
            return True

        tokens = set(value_norm.split())
        if tokens.intersection(self._EPHEMERAL_MARKERS):
            return True

        # Very long values are usually sentence fragments, not compact facts.
        if len(value_norm) > 96:
            return True
        if len(tokens) > 10:
            return True

        return False

    @staticmethod
    def _norm_key(key: str) -> str:
        return normalize_text(str(key or ""))

    @staticmethod
    def _norm_value(value: str) -> str:
        return normalize_text(str(value or ""))

    @staticmethod
    def _canonical_value_preserve_case(value: str) -> str:
        clean = str(value or "").strip()
        return clean

