from __future__ import annotations

import re
from typing import Iterable

from modules.internet.web.query_text import normalize_search_text
from modules.internet.web.web_models import QueryClassification


_WORD_RE = re.compile(r"[A-Za-zА-Яа-яЁёІіЇїЄєҐґ0-9_]+")

_TEMPORAL_MARKERS = (
    "today",
    "now",
    "latest",
    "current",
    "breaking",
    "new",
    "2024",
    "2025",
    "2026",
    "сегодня",
    "сейчас",
    "актуаль",
    "последн",
    "новости",
    "что нового",
    "сьогодні",
    "зараз",
    "останні",
    "актуальні",
    "завтра",
    "tomorrow",
)

_EXTERNAL_FACT_MARKERS = (
    "version",
    "release",
    "changelog",
    "docs",
    "documentation",
    "reference",
    "price",
    "cost",
    "pricing",
    "availability",
    "news",
    "market",
    "benchmark",
    "weather",
    "forecast",
    "temperature",
    "api",
    "sdk",
    "exchange rate",
    "currency",
    "usd",
    "eur",
    "uah",
    "btc",
    "eth",
    "документация",
    "версия",
    "релиз",
    "цена",
    "сколько стоит",
    "доступно",
    "новости",
    "рынок",
    "бенчмарк",
    "курс",
    "лимит",
    "погода",
    "прогноз",
    "температура",
    "доллар",
    "евро",
    "гривн",
    "упомин",
)

_LOCAL_PROJECT_MARKERS = (
    "this project",
    "repo",
    "repository",
    "codebase",
    "refactor",
    "debug",
    "function",
    "module",
    "class",
    "pipeline",
    "memory",
    "context",
    "local file",
    "в проекте",
    "в репозитории",
    "в коде",
    "рефактор",
    "архитектур",
    "файл",
    "функци",
    "модул",
    "пайплайн",
    "локальн",
)

_MIXED_MARKERS = (
    "best",
    "what to choose",
    "compare",
    "options",
    "какой выбрать",
    "что выбрать",
    "лучше взять",
    "сравни",
    "варианты",
    "практики",
)

_SEARCH_INTENT_MARKERS = (
    "find",
    "search",
    "lookup",
    "verify",
    "check",
    "look up",
    "source",
    "найди",
    "поищи",
    "подскажи",
    "скажи",
    "глянь",
    "поиск",
    "проверь",
    "проверка",
    "источник",
    "посмотри",
)

_SMALLTALK_MARKERS = (
    "hello",
    "hi",
    "how are you",
    "just chat",
    "привет",
    "как дела",
    "что нового",
    "поговори",
)

_STAKES_HIGH_MARKERS = (
    "medical",
    "health",
    "legal",
    "law",
    "finance",
    "money",
    "security",
    "врач",
    "медицина",
    "закон",
    "юрид",
    "финанс",
    "безопас",
)

_WEATHER_MARKERS = (
    "weather",
    "forecast",
    "temperature",
    "rain",
    "snow",
    "umbrella",
    "storm",
    "wind",
    "hot",
    "cold",
    "погода",
    "прогноз",
    "температура",
    "дожд",
    "снег",
    "зонтик",
    "ветер",
    "жарко",
    "холодно",
)

_FINANCE_MARKERS = (
    "exchange rate",
    "fx",
    "forex",
    "currency",
    "usd",
    "eur",
    "uah",
    "gbp",
    "btc",
    "eth",
    "курс",
    "доллар",
    "евро",
    "гривн",
    "грн",
    "биткоин",
    "эфир",
    "акции",
    "stock",
    "stocks",
    "ticker",
)

_PRICE_MARKERS = (
    "price",
    "pricing",
    "cost",
    "сколько стоит",
    "цена",
    "стоимость",
)

_VERSION_MARKERS = (
    "version",
    "release",
    "release notes",
    "changelog",
    "latest stable version",
    "версия",
    "релиз",
    "обновление",
)

_DOCS_MARKERS = (
    "docs",
    "documentation",
    "reference",
    "manual",
    "guide",
    "api reference",
    "документация",
    "справочник",
    "гайд",
)

_NEWS_MARKERS = (
    "news",
    "mention",
    "mentioned",
    "mentions",
    "reported",
    "report",
    "recent event",
    "last time mentioned",
    "новост",
    "упомин",
    "писали",
    "сообщали",
    "последний раз",
)


def classify_query(text: str, *, metadata_tags: Iterable[str] | None = None) -> QueryClassification:
    src = str(text or "").strip()
    normalized = normalize_search_text(src) or src
    low = normalized.lower()
    raw_low = src.lower()
    tags = [str(x or "").strip().lower() for x in list(metadata_tags or []) if str(x or "").strip()]

    temporal_hits = _find_markers(low, _TEMPORAL_MARKERS)
    external_hits = _find_markers(low, _EXTERNAL_FACT_MARKERS)
    local_hits = _find_markers(low, _LOCAL_PROJECT_MARKERS)
    mixed_hits = _find_markers(low, _MIXED_MARKERS)
    search_hits = _find_markers(raw_low, _SEARCH_INTENT_MARKERS)
    category_hits = _find_markers(low, _WEATHER_MARKERS + _FINANCE_MARKERS + _PRICE_MARKERS + _VERSION_MARKERS + _DOCS_MARKERS + _NEWS_MARKERS)

    has_question = "?" in src
    has_year = bool(re.search(r"\b20[2-4][0-9]\b", low))
    is_temporal = bool(temporal_hits or has_year)

    tokens = _WORD_RE.findall(normalized)
    is_smalltalk = bool(_find_markers(raw_low, _SMALLTALK_MARKERS))
    is_local = bool(local_hits or ("project" in tags) or ("code" in tags))
    primary_category = _detect_primary_category(low, query_type="")
    is_external = bool(
        external_hits
        or is_temporal
        or primary_category in {"weather", "finance", "price", "version", "news", "docs", "external"}
    )

    short_non_fact = bool(len(tokens) <= 4 and not is_external and not mixed_hits and not search_hits and not is_temporal)
    long_weird = any(len(t) >= 18 for t in tokens)
    mixed_alnum = any(bool(re.search(r"[A-Za-z]", t) and re.search(r"[0-9]", t)) for t in tokens)
    too_short_unknown = len(tokens) <= 2 and not is_local and not is_external and not mixed_hits
    is_ambiguous = bool((too_short_unknown and (long_weird or mixed_alnum)) or "??" in src)

    if is_smalltalk:
        query_type = "local_logical"
    elif is_ambiguous:
        query_type = "ambiguous"
    elif is_local and not is_external and not is_temporal:
        query_type = "local_logical"
    elif is_external and not is_local:
        query_type = "external_factual"
    elif mixed_hits or (is_local and is_external):
        query_type = "mixed"
    elif short_non_fact:
        query_type = "local_logical"
    else:
        query_type = "local_logical"

    primary_category = _detect_primary_category(low, query_type=query_type)
    if is_smalltalk:
        primary_category = "chitchat"

    requires_freshness = bool(is_temporal or _needs_freshness_from_markers(external_hits, primary_category))
    stakes_level = _stakes_level(low)

    if query_type == "local_logical" and not requires_freshness:
        expected = "NO_SEARCH"
    elif query_type == "external_factual" and requires_freshness:
        expected = "VERIFY_ONLY"
    elif query_type == "mixed":
        expected = "SOFT_SEARCH"
    elif query_type == "ambiguous":
        expected = "SOFT_SEARCH"
    else:
        expected = "VERIFY_ONLY"

    return QueryClassification(
        query_type=query_type,
        primary_category=primary_category,
        is_temporal=bool(is_temporal),
        is_local_project_question=bool(is_local),
        is_external_fact_question=bool(is_external),
        is_ambiguous=bool(is_ambiguous),
        requires_freshness=bool(requires_freshness),
        stakes_level=stakes_level,
        expected_search_need=expected,
        explicit_search_intent=bool(search_hits or has_question and is_external),
        temporal_markers=list(temporal_hits),
        category_hits=list(dict.fromkeys(local_hits + external_hits + mixed_hits + category_hits)),
    )


def _find_markers(text: str, markers: Iterable[str]) -> list[str]:
    out: list[str] = []
    for marker in markers:
        token = str(marker or "").strip().lower()
        if not token:
            continue
        if len(token) <= 3 and token.isalnum():
            pattern = rf"(?<!\w){re.escape(token)}(?!\w)"
            if re.search(pattern, text):
                out.append(token)
            continue
        if token in text:
            out.append(token)
    return out


def _needs_freshness_from_markers(external_hits: list[str], primary_category: str) -> bool:
    if str(primary_category or "").strip().lower() in {"weather", "finance", "price", "version", "news"}:
        return True
    freshness_tokens = (
        "version",
        "release",
        "changelog",
        "price",
        "cost",
        "availability",
        "news",
        "market",
        "weather",
        "forecast",
        "temperature",
        "sdk",
        "api",
        "exchange rate",
        "currency",
        "верс",
        "релиз",
        "цена",
        "доступно",
        "новост",
        "рынок",
        "курс",
        "погод",
        "прогноз",
        "температур",
    )
    for token in external_hits:
        if any(x in token for x in freshness_tokens):
            return True
    return False


def _detect_primary_category(text: str, *, query_type: str) -> str:
    if any(x in text for x in ("refactor", "рефактор", "rewrite", "перепиши")):
        return "refactor"
    if any(x in text for x in ("architecture", "архитектур")):
        return "architecture"
    if any(x in text for x in ("reasoning", "объясни", "explain", "почему")):
        return "reasoning"
    if any(x in text for x in _WEATHER_MARKERS):
        return "weather"
    if any(x in text for x in _FINANCE_MARKERS):
        return "finance"
    if any(x in text for x in _PRICE_MARKERS):
        return "price"
    if any(x in text for x in _VERSION_MARKERS):
        return "version"
    if any(x in text for x in _NEWS_MARKERS):
        return "news"
    if any(x in text for x in _DOCS_MARKERS):
        return "docs"
    if query_type == "local_logical":
        return "local"
    if query_type == "external_factual":
        return "external"
    if query_type == "ambiguous":
        return "ambiguous"
    return "mixed"


def _stakes_level(text: str) -> str:
    if any(token in text for token in _STAKES_HIGH_MARKERS):
        return "high"
    if any(token in text for token in ("price", "cost", "цена", "курс", "money", "market", "рынок", "usd", "eur", "uah")):
        return "medium"
    return "normal"
