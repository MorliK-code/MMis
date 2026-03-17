from __future__ import annotations

"""Legacy semantic-to-search planner.

This module is kept only for compatibility with older internet search flows.
The active web pipeline lives under ``modules.internet.web.*`` and should be
preferred for new routing or retrieval work.
"""

from dataclasses import dataclass

from modules.nlu.location_resolver import LocationResolver
from modules.nlu.normalizer import normalize_text
from modules.nlu.types import Concept, Fact, Intent, Location, SearchTask


@dataclass(slots=True)
class _PlanContext:
    intent_scores: dict[str, float]
    concept_scores: dict[str, float]
    facts_by_key: dict[str, list[Fact]]
    resolved_location: str | None


class SearchPlanner:
    """Legacy planner for structured search tasks from semantic NLU outputs."""

    _FINANCE_CURRENCY_HINTS = {"usd", "eur", "gbp", "uah", "btc"}
    _CODING_WEB_REQUIRED_FACT_KEYS = {
        "requires_web",
        "internet_required",
        "requires_freshness",
        "needs_latest",
        "version_check",
        "release_check",
    }
    _CODING_WEB_TRIGGER_CONCEPTS = {"embeddings", "gpu", "search"}

    def __init__(self, location_resolver: LocationResolver | None = None) -> None:
        self._location_resolver = location_resolver or LocationResolver()

    def plan(
        self,
        *,
        intents: list[Intent],
        concepts: list[Concept],
        facts: list[Fact],
        locations: list[Location],
        dialogue_location: str | None = None,
        fallback_location: str | None = None,
    ) -> list[SearchTask]:
        context = self._build_context(
            intents=intents,
            concepts=concepts,
            facts=facts,
            locations=locations,
            dialogue_location=dialogue_location,
            fallback_location=fallback_location,
        )

        tasks: list[SearchTask] = []

        if self._has_intent(context, "weather_query", min_conf=0.55):
            weather_task = self._build_weather_task(context)
            if weather_task is not None:
                tasks.append(weather_task)

        if self._has_intent(context, "finance_query", min_conf=0.55) and self._has_concept(
            context, "currency", min_conf=0.5
        ):
            currency_task = self._build_currency_task(context)
            if currency_task is not None:
                tasks.append(currency_task)

        if self._has_intent(context, "news_query", min_conf=0.55):
            news_task = self._build_news_task(context)
            if news_task is not None:
                tasks.append(news_task)

        if self._has_intent(context, "coding_question", min_conf=0.6) and self._needs_web_for_coding(context):
            coding_task = self._build_coding_web_task(context)
            if coding_task is not None:
                tasks.append(coding_task)

        return self._deduplicate_tasks(tasks)

    def _build_context(
        self,
        *,
        intents: list[Intent],
        concepts: list[Concept],
        facts: list[Fact],
        locations: list[Location],
        dialogue_location: str | None,
        fallback_location: str | None,
    ) -> _PlanContext:
        intent_scores = self._collect_max_scores(intents)
        concept_scores = self._collect_max_scores(concepts)
        facts_by_key = self._group_facts(facts)

        fact_locations = self._location_resolver.extract_from_facts(facts)
        resolved_location = self._location_resolver.resolve_search_location(
            message_locations=locations,
            fact_locations=fact_locations,
            dialogue_location=dialogue_location,
            fallback_location=fallback_location,
        )

        return _PlanContext(
            intent_scores=intent_scores,
            concept_scores=concept_scores,
            facts_by_key=facts_by_key,
            resolved_location=resolved_location,
        )

    def _build_weather_task(self, context: _PlanContext) -> SearchTask | None:
        location = context.resolved_location or ""
        query = "weather today"
        if location:
            query = f"weather {location} today"

        return SearchTask(
            kind="weather",
            query=query,
            priority=1,
            location=location,
            requires_freshness=True,
            geo_hint=location,
            meta={"intent": "weather_query"},
        )

    def _build_currency_task(self, context: _PlanContext) -> SearchTask | None:
        currency = self._resolve_currency_symbol(context) or "USD"
        location = context.resolved_location or "Ukraine"
        query = f"{currency} exchange rate {location} today"

        return SearchTask(
            kind="currency_rate",
            query=query,
            priority=1,
            location=location,
            requires_freshness=True,
            geo_hint=location,
            meta={"intent": "finance_query", "concept": "currency", "currency": currency},
        )

    def _build_news_task(self, context: _PlanContext) -> SearchTask | None:
        location = context.resolved_location or ""
        query = "latest news"
        if location:
            query = f"{location} latest news"

        return SearchTask(
            kind="news",
            query=query,
            priority=1,
            location=location,
            requires_freshness=True,
            geo_hint=location,
            meta={"intent": "news_query"},
        )

    def _build_coding_web_task(self, context: _PlanContext) -> SearchTask | None:
        focus = self._pick_coding_focus(context)
        query = "latest programming best practices"
        if focus:
            query = f"latest {focus} best practices"

        return SearchTask(
            kind="web_search",
            query=query,
            priority=2,
            location=context.resolved_location or "",
            requires_freshness=True,
            geo_hint=context.resolved_location or "",
            meta={"intent": "coding_question", "focus": focus or "programming"},
        )

    def _pick_coding_focus(self, context: _PlanContext) -> str | None:
        ranking = sorted(context.concept_scores.items(), key=lambda item: item[1], reverse=True)
        for concept_name, score in ranking:
            if score < 0.5:
                continue
            if concept_name in {"embeddings", "gpu", "programming", "search", "memory"}:
                return concept_name
        return None

    def _resolve_currency_symbol(self, context: _PlanContext) -> str | None:
        for fact in context.facts_by_key.get("currency", []):
            normalized = normalize_text(fact.value).upper()
            if normalized in self._FINANCE_CURRENCY_HINTS:
                return normalized

        for fact in context.facts_by_key.get("currency_code", []):
            normalized = normalize_text(fact.value).upper()
            if normalized in self._FINANCE_CURRENCY_HINTS:
                return normalized

        if "currency" in context.concept_scores:
            return "USD"
        return None

    def _needs_web_for_coding(self, context: _PlanContext) -> bool:
        if context.concept_scores.get("search", 0.0) >= 0.55:
            return True

        for concept_name in self._CODING_WEB_TRIGGER_CONCEPTS:
            if context.concept_scores.get(concept_name, 0.0) >= 0.55:
                return True

        for key in self._CODING_WEB_REQUIRED_FACT_KEYS:
            for fact in context.facts_by_key.get(key, []):
                value = normalize_text(fact.value)
                if value in {"1", "true", "yes", "required", "need", "needed"}:
                    return True
                if fact.confidence >= 0.75:
                    return True

        return False

    @staticmethod
    def _group_facts(facts: list[Fact]) -> dict[str, list[Fact]]:
        grouped: dict[str, list[Fact]] = {}
        for fact in facts:
            grouped.setdefault(fact.key, []).append(fact)
        return grouped

    @staticmethod
    def _collect_max_scores(items: list[Intent] | list[Concept]) -> dict[str, float]:
        scores: dict[str, float] = {}
        for item in items:
            current = scores.get(item.name, 0.0)
            if item.confidence > current:
                scores[item.name] = item.confidence
        return scores

    @staticmethod
    def _has_intent(context: _PlanContext, name: str, min_conf: float) -> bool:
        return context.intent_scores.get(name, 0.0) >= min_conf

    @staticmethod
    def _has_concept(context: _PlanContext, name: str, min_conf: float) -> bool:
        return context.concept_scores.get(name, 0.0) >= min_conf

    @staticmethod
    def _deduplicate_tasks(tasks: list[SearchTask]) -> list[SearchTask]:
        dedup: dict[tuple[str, str, str], SearchTask] = {}
        for task in tasks:
            key = (task.kind, task.query.lower(), task.location.lower())
            existing = dedup.get(key)
            if existing is None or task.priority < existing.priority:
                dedup[key] = task
        return list(dedup.values())


if __name__ == "__main__":
    planner = SearchPlanner()

    sample_intents = [
        Intent(name="weather_query", confidence=0.92),
        Intent(name="finance_query", confidence=0.81),
        Intent(name="news_query", confidence=0.78),
    ]
    sample_concepts = [
        Concept(name="weather", confidence=0.89),
        Concept(name="currency", confidence=0.9),
        Concept(name="news", confidence=0.84),
    ]
    sample_facts = [
        Fact(key="location_place", value="Ukraine", confidence=0.93),
        Fact(key="currency", value="USD", confidence=0.95),
    ]
    sample_locations = [Location(name="Ukraine", kind="country", confidence=0.91, source="explicit_message")]

    planned = planner.plan(
        intents=sample_intents,
        concepts=sample_concepts,
        facts=sample_facts,
        locations=sample_locations,
        dialogue_location=None,
        fallback_location="Poland",
    )

    for task in planned:
        print(task)
