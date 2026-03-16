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

_FINANCE_REGEX_MARKERS: tuple[tuple[str, str], ...] = (
    ("доллар", r"(?<!\w)доллар\w*(?!\w)"),
    ("евро", r"(?<!\w)евро\w*(?!\w)"),
    ("гривна", r"(?<!\w)(?:гривн\w*|грн)(?!\w)"),
    ("рубль", r"(?<!\w)рубл\w*(?!\w)"),
    ("валюта", r"(?<!\w)валют\w*(?!\w)"),
    ("обмен", r"(?<!\w)(?:обмен\w*|котировк\w*)(?!\w)"),
    ("бакс", r"(?<!\w)бакс\w*(?!\w)"),
    (
        "currency_pair",
        r"(?<!\w)(?:usd|eur|uah|gbp|btc|eth|доллар\w*|евро\w*|гривн\w*|грн|рубл\w*|бакс\w*)"
        r"\s*(?:/|в|к|to)\s*"
        r"(?:usd|eur|uah|gbp|btc|eth|доллар\w*|евро\w*|гривн\w*|грн|рубл\w*|бакс\w*)(?!\w)",
    ),
    (
        "colloquial_fx",
        r"(?<!\w)(?:что\s+по|how\s+about)\s+(?:usd|eur|uah|доллар\w*|евро\w*|гривн\w*|грн|бакс\w*)(?!\w)",
    ),
)

_FALSE_FINANCE_CONTEXT_PATTERNS: tuple[tuple[str, str], ...] = (
    (
        "евро",
        r"(?<!\w)(?:евросоюз\w*|еврозон\w*|european\s+union|eurozone)(?!\w)",
    ),
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

_DATE_MARKERS = (
    "date",
    "exact date",
    "when exactly",
    "what date",
    "\u0434\u0430\u0442\u0430",
    "\u0442\u043e\u0447\u043d\u0443\u044e \u0434\u0430\u0442\u0443",
    "\u0442\u043e\u0447\u043d\u0430\u044f \u0434\u0430\u0442\u0430",
    "\u043a\u0430\u043a\u043e\u0433\u043e \u0447\u0438\u0441\u043b\u0430",
    "\u043a\u043e\u0433\u0434\u0430 \u0438\u043c\u0435\u043d\u043d\u043e",
    "\u0432 \u043a\u0430\u043a\u043e\u043c \u0433\u043e\u0434\u0443",
    "\u0438\u0441\u0442\u043e\u0440\u0438\u0447\u0435\u0441",
    "historical",
)

_CORRECTION_CHALLENGE_MARKERS = (
    "\u043d\u0435\u043f\u0440\u0430\u0432\u0438\u043b\u044c\u043d\u043e",
    "\u043d\u0435 \u0442\u0430\u043a",
    "\u043d\u0435\u0432\u0435\u0440\u043d\u043e",
    "\u0432\u0440\u0435\u0448\u044c",
    "\u0432\u0440\u0451\u0448\u044c",
    "\u0442\u044b \u043e\u0448\u0438\u0431\u043b\u0430\u0441\u044c",
    "\u0442\u044b \u043e\u0448\u0438\u0431\u0430\u0435\u0448\u044c\u0441\u044f",
    "\u0442\u044b \u043d\u0435 \u043f\u0443\u0442\u0430\u0435\u0448\u044c",
    "\u043f\u0435\u0440\u0435\u043f\u0440\u043e\u0432\u0435\u0440\u044c",
    "\u043f\u0440\u043e\u0432\u0435\u0440\u044c \u0435\u0449\u0451 \u0440\u0430\u0437",
    "\u043f\u0440\u043e\u0432\u0435\u0440\u044c \u0435\u0449\u0435 \u0440\u0430\u0437",
    "\u0442\u043e\u0447\u043d\u043e \u0441\u043c\u043e\u0442\u0440\u0438\u0448\u044c",
    "\u043d\u0430\u0437\u043e\u0432\u0438 \u0442\u043e\u0447\u043d\u0443\u044e \u0434\u0430\u0442\u0443",
    "\u0442\u043e\u0447\u043d\u0443\u044e \u0434\u0430\u0442\u0443",
    "wrong",
    "you're wrong",
    "you are wrong",
    "check again",
    "re-check",
    "recheck",
    "verify again",
    "exact date",
)


def classify_query(
    text: str,
    *,
    metadata_tags: Iterable[str] | None = None,
    original_text: str | None = None,
) -> QueryClassification:
    src = str(text or "").strip()
    normalized = normalize_search_text(src) or src
    low = normalized.lower()
    raw_src = str(original_text or src).strip() or src
    raw_low = raw_src.lower()
    tags = [str(x or "").strip().lower() for x in list(metadata_tags or []) if str(x or "").strip()]

    temporal_hits = _find_markers(low, _TEMPORAL_MARKERS)
    finance_hits = _find_finance_markers(low)
    external_hits = list(dict.fromkeys(_find_markers(low, _EXTERNAL_FACT_MARKERS) + finance_hits))
    local_hits = _find_markers(low, _LOCAL_PROJECT_MARKERS)
    mixed_hits = _find_markers(low, _MIXED_MARKERS)
    search_hits = _find_markers(raw_low, _SEARCH_INTENT_MARKERS)
    challenge_hits = _find_markers(raw_low, _CORRECTION_CHALLENGE_MARKERS)
    date_hits = list(dict.fromkeys(_find_markers(low, _DATE_MARKERS) + _find_markers(raw_low, _DATE_MARKERS)))
    category_hits = list(
        dict.fromkeys(
            _find_markers(low, _WEATHER_MARKERS)
            + finance_hits
            + _find_markers(low, _PRICE_MARKERS)
            + _find_markers(low, _VERSION_MARKERS)
            + _find_markers(low, _DOCS_MARKERS)
            + _find_markers(low, _NEWS_MARKERS)
            + date_hits
        )
    )

    has_question = "?" in src or "?" in raw_src
    has_year = bool(re.search(r"\b20[2-4][0-9]\b", low))
    is_temporal = bool(temporal_hits or has_year)

    tokens = _WORD_RE.findall(normalized)
    is_smalltalk = bool(_find_markers(raw_low, _SMALLTALK_MARKERS))
    is_local = bool(local_hits or ("project" in tags) or ("code" in tags))
    primary_category = _detect_primary_category(low, query_type="", finance_hits=finance_hits)
    has_date_like_request = bool(date_hits)
    is_external = bool(
        external_hits
        or is_temporal
        or has_date_like_request
        or primary_category in {"weather", "finance", "price", "version", "news", "docs", "external"}
    )
    is_correction_challenge = bool(
        challenge_hits
        and (
            is_external
            or has_date_like_request
            or has_question
            or primary_category in {"weather", "finance", "price", "version", "news", "docs", "external"}
        )
    )

    short_non_fact = bool(
        len(tokens) <= 4
        and not is_external
        and not mixed_hits
        and not search_hits
        and not is_temporal
        and not is_correction_challenge
    )
    long_weird = any(len(t) >= 18 for t in tokens)
    mixed_alnum = any(bool(re.search(r"[A-Za-z]", t) and re.search(r"[0-9]", t)) for t in tokens)
    too_short_unknown = len(tokens) <= 2 and not is_local and not is_external and not mixed_hits
    is_ambiguous = bool((too_short_unknown and (long_weird or mixed_alnum)) or "??" in src)

    if is_smalltalk:
        query_type = "local_logical"
    elif is_correction_challenge and (is_external or has_date_like_request):
        query_type = "external_factual"
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

    primary_category = _detect_primary_category(low, query_type=query_type, finance_hits=finance_hits)
    if is_smalltalk:
        primary_category = "chitchat"

    short_fx_query = _is_short_clear_fx_query(
        normalized=normalized,
        tokens=tokens,
        finance_hits=finance_hits,
        has_question=has_question,
        primary_category=primary_category,
    )

    requires_freshness = bool(
        is_temporal
        or _needs_freshness_from_markers(external_hits + date_hits, primary_category)
        or (
            is_correction_challenge
            and primary_category in {"weather", "finance", "price", "news", "version"}
        )
    )
    stakes_level = _stakes_level(low)

    if is_correction_challenge and (
        primary_category in {"finance", "price", "weather", "news"}
        or has_date_like_request
        or has_year
    ):
        expected = "TARGETED_SEARCH"
    elif is_correction_challenge:
        expected = "VERIFY_ONLY"
    elif query_type == "local_logical" and not requires_freshness:
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
        is_correction_challenge=bool(is_correction_challenge),
        requires_freshness=bool(requires_freshness),
        stakes_level=stakes_level,
        expected_search_need=expected,
        explicit_search_intent=bool(
            search_hits
            or is_correction_challenge
            or (has_question and is_external)
            or short_fx_query
        ),
        temporal_markers=list(temporal_hits),
        category_hits=list(dict.fromkeys(local_hits + external_hits + mixed_hits + category_hits + date_hits)),
        challenge_markers=list(dict.fromkeys(challenge_hits)),
    )


def _find_markers(text: str, markers: Iterable[str]) -> list[str]:
    out: list[str] = []
    for marker in markers:
        token = str(marker or "").strip().lower()
        if not token:
            continue
        if re.fullmatch(r"[\wа-яёіїєґ-]+", token, flags=re.I):
            pattern = rf"(?<!\w){re.escape(token)}(?!\w)"
            if re.search(pattern, text):
                out.append(token)
            continue
        if token in text:
            out.append(token)
    return out


def _find_finance_markers(text: str) -> list[str]:
    out = list(_find_markers(text, _FINANCE_MARKERS))
    for label, pattern in _FINANCE_REGEX_MARKERS:
        if re.search(pattern, text, flags=re.I):
            out.append(label)
    filtered: list[str] = []
    for label in list(dict.fromkeys(out)):
        skip = False
        for false_label, pattern in _FALSE_FINANCE_CONTEXT_PATTERNS:
            if label != false_label:
                continue
            if re.search(pattern, text, flags=re.I):
                skip = True
                break
        if not skip:
            filtered.append(label)
    return filtered


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


def _detect_primary_category(text: str, *, query_type: str, finance_hits: list[str] | None = None) -> str:
    if any(x in text for x in ("refactor", "рефактор", "rewrite", "перепиши")):
        return "refactor"
    if any(x in text for x in ("architecture", "архитектур")):
        return "architecture"
    if any(x in text for x in ("reasoning", "объясни", "explain", "почему")):
        return "reasoning"
    if _find_markers(text, _WEATHER_MARKERS):
        return "weather"
    if finance_hits is None:
        finance_hits = _find_finance_markers(text)
    if finance_hits:
        return "finance"
    if _find_markers(text, _PRICE_MARKERS):
        return "price"
    if _find_markers(text, _VERSION_MARKERS):
        return "version"
    if _find_markers(text, _NEWS_MARKERS):
        return "news"
    if _find_markers(text, _DOCS_MARKERS):
        return "docs"
    if _find_markers(text, _DATE_MARKERS):
        return "external"
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
    if _find_finance_markers(text) or any(
        token in text for token in ("price", "cost", "цена", "курс", "money", "market", "рынок", "usd", "eur", "uah")
    ):
        return "medium"
    return "normal"


def _is_short_clear_fx_query(
    *,
    normalized: str,
    tokens: list[str],
    finance_hits: list[str],
    has_question: bool,
    primary_category: str,
) -> bool:
    if str(primary_category or "").strip().lower() != "finance":
        return False
    if not finance_hits:
        return False
    short = len(tokens) <= 6
    if not short and not has_question:
        return False
    low = str(normalized or "").strip().lower()
    if re.search(
        r"(?<!\w)(?:что\s+по|how\s+about)\s+(?:usd|eur|uah|доллар\w*|евро\w*|гривн\w*|грн|бакс\w*)(?!\w)",
        low,
        flags=re.I,
    ):
        return True
    if re.search(
        r"(?<!\w)(?:usd|eur|uah|доллар\w*|евро\w*|гривн\w*|грн|бакс\w*)\s*(?:/|в|к|to)\s*"
        r"(?:usd|eur|uah|доллар\w*|евро\w*|гривн\w*|грн)(?!\w)",
        low,
        flags=re.I,
    ):
        return True
    return bool(has_question or short)
