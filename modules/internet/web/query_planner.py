from __future__ import annotations

import datetime as dt
import re
from typing import Iterable

from modules.internet.web.query_text import analyze_search_text
from modules.internet.web.web_models import QueryClassification, WebPolicyDecision, WebQueryPlan, WebSearchMode


_WORD_RE = re.compile(r"[A-Za-zА-Яа-яЁёІіЇїЄєҐґ0-9\-\._]+")
_QUOTED_RE = re.compile(r"\"([^\"]{2,120})\"|“([^”]{2,120})”|«([^»]{2,120})»")

_GENERIC_STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "from",
    "this",
    "that",
    "what",
    "which",
    "when",
    "where",
    "how",
    "in",
    "on",
    "at",
    "of",
    "to",
    "me",
    "please",
    "latest",
    "current",
    "now",
    "today",
    "tomorrow",
    "a",
    "an",
    "и",
    "в",
    "во",
    "на",
    "по",
    "о",
    "об",
    "про",
    "для",
    "мне",
    "нам",
    "пожалуйста",
    "это",
    "этот",
    "эта",
    "эти",
    "сегодня",
    "сейчас",
    "завтра",
}
_ENTITY_NOISE_WORDS = _GENERIC_STOPWORDS | {
    "official",
    "officially",
    "docs",
    "documentation",
    "document",
    "reference",
    "manual",
    "guide",
    "source",
    "news",
    "release",
    "releases",
    "notes",
    "version",
    "versions",
    "changelog",
    "pricing",
    "price",
    "cost",
    "weather",
    "forecast",
    "temperature",
    "market",
    "mention",
    "mentions",
    "mentioned",
    "last",
    "time",
    "official-source",
    "documentation",
    "exchange",
    "rate",
    "library",
    "package",
    "lib",
    "документация",
    "справочник",
    "источник",
    "источники",
    "библиотека",
    "библиотеки",
    "пакет",
    "новости",
    "релиз",
    "релизы",
    "версия",
    "версии",
    "обновление",
    "обновления",
    "цена",
    "курс",
    "погода",
    "прогноз",
    "температура",
    "рынок",
    "когда",
    "какой",
    "какая",
    "какие",
    "где",
    "что",
    "сколько",
    "последний",
    "последняя",
    "последние",
    "раз",
    "интернет",
    "интернете",
    "сеть",
    "сети",
    "сегодня",
    "сейчас",
    "завтра",
    "упоминался",
    "упоминалась",
    "упоминались",
}
_ENTITY_NOISE_PREFIXES = (
    "mention",
    "упомин",
    "news",
    "новост",
    "релиз",
    "верс",
    "release",
    "version",
    "doc",
    "докум",
    "official",
    "официал",
    "источн",
    "price",
    "pricing",
    "цен",
    "weather",
    "forecast",
    "погод",
    "прогноз",
    "курс",
    "find",
    "search",
    "lookup",
    "найд",
    "поищ",
    "подскаж",
    "скажи",
    "глян",
    "посмотр",
    "проверь",
)
_FOCUS_STOPWORDS = _GENERIC_STOPWORDS | {
    "what",
    "when",
    "where",
    "how",
    "which",
    "когда",
    "где",
    "что",
    "какой",
    "какая",
    "какие",
    "сколько",
    "internet",
    "web",
    "online",
    "интернет",
    "интернете",
    "веб",
    "онлайн",
}
_DOCS_QUERY_MARKERS = (
    "docs",
    "documentation",
    "reference",
    "manual",
    "guide",
    "api",
    "sdk",
    "release notes",
    "changelog",
    "migration guide",
    "документация",
    "справочник",
    "гайд",
    "api reference",
)
_RELEASE_QUERY_MARKERS = (
    "version",
    "release",
    "release notes",
    "changelog",
    "latest stable version",
    "версия",
    "релиз",
    "обновление",
)
_NON_DOCS_OVERRIDES = (
    "pricing",
    "price",
    "cost",
    "weather",
    "forecast",
    "temperature",
    "news",
    "mention",
    "mentioned",
    "mentions",
    "exchange rate",
    "currency",
    "usd",
    "eur",
    "uah",
    "btc",
    "eth",
    "цена",
    "курс",
    "погода",
    "прогноз",
    "новости",
    "упомин",
    "доллар",
    "евро",
    "гривн",
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
    "stock",
    "stocks",
    "ticker",
)
_PRICE_MARKERS = (
    "price",
    "pricing",
    "cost",
    "quote",
    "цена",
    "стоимость",
    "стоит",
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
_HISTORICAL_MARKERS = (
    "historical",
    "history",
    "timeline",
    "archive",
    "exact date",
    "what date",
    "when exactly",
    "дата",
    "точную дату",
    "точная дата",
    "истор",
    "архив",
    "хронолог",
    "когда именно",
    "в каком году",
)
_TEMPORAL_TERMS = (
    "today",
    "tomorrow",
    "tonight",
    "weekend",
    "сегодня",
    "завтра",
    "сейчас",
    "вечером",
    "ночью",
    "на выходных",
)
_CURRENCY_ALIASES = {
    "usd": "USD",
    "доллар": "USD",
    "доллара": "USD",
    "долларов": "USD",
    "eur": "EUR",
    "евро": "EUR",
    "uah": "UAH",
    "гривна": "UAH",
    "гривны": "UAH",
    "гривен": "UAH",
    "грн": "UAH",
    "gbp": "GBP",
    "фунт": "GBP",
    "btc": "BTC",
    "биткоин": "BTC",
    "eth": "ETH",
    "эфир": "ETH",
}
_DOCS_DOMAIN_HINTS = ("docs.", "developer.", "readthedocs", "github.com", "gitlab.com", "pypi.org", "npmjs.com")
_FINANCE_DOMAIN_HINTS = ("bank.", "bank.gov", "minfin", "finance.", "forex", "fx", "kurs", "invest", "marketwatch")
_WEATHER_DOMAIN_HINTS = ("weather", "meteo", "forecast", "sinoptik", "accuweather", "gismeteo")
_NEWS_DOMAIN_HINTS = ("news", "reuters", "apnews", "bbc", "ukrinform", "cnn", "nytimes", "wsj")
_HISTORICAL_DOMAIN_HINTS = ("wikipedia", "britannica", "history", "archive", "museum", ".gov", ".edu", "reuters", "apnews", "bbc")


def build_query_plan(
    *,
    query: str,
    classification: QueryClassification,
    decision: WebPolicyDecision,
    preferred_domains: list[str] | None = None,
    geo_hint: str = "",
) -> WebQueryPlan:
    query_debug = analyze_search_text(query)
    text = str(
        query_debug.get("extracted_search_core")
        or query_debug.get("search_core")
        or query_debug.get("normalized_query")
        or ""
    ).strip()
    if not text:
        return WebQueryPlan(mode=decision.mode, debug={"query_text": query_debug})

    budget_queries = max(0, int(decision.budget.max_queries))
    now_year = str(dt.datetime.now(dt.timezone.utc).year)
    category = str(classification.primary_category or "").strip().lower()
    strategy = _query_strategy(text=text, classification=classification)
    region_bias = _planner_region_bias(text=text, strategy=strategy)
    domains = _normalize_domains(preferred_domains, strategy=strategy, region_bias=region_bias)
    category_hint = _category_hint(strategy=strategy, category=category)
    entity = _extract_primary_entity(text)
    focus_terms = _extract_focus_terms(text, entity=entity)

    geo = _normalize_geo_hint(geo_hint)
    geo_bias = _should_bias_geo(text=text, classification=classification, geo_hint=geo)
    geo_tail = geo if geo_bias else ""

    primary_query = _build_primary_query(strategy=strategy, text=text, geo_tail=geo_tail)
    validation_queries = _build_validation_queries(
        strategy=strategy,
        text=text,
        entity=entity,
        focus_terms=focus_terms,
        category_hint=category_hint,
        geo_tail=geo_tail,
    )
    exact_entity_query = _build_exact_entity_query(
        strategy=strategy,
        text=text,
        entity=entity,
        focus_terms=focus_terms,
        geo_tail=geo_tail,
        category_hint=category_hint,
    )

    release_note_queries: list[str] = []
    if _needs_release_notes(text=text, classification=classification, strategy=strategy):
        release_target = entity or _extract_primary_product(text)
        release_note_queries = _dedupe_keep_order(
            [
                _join_parts(release_target, "release notes", now_year),
                _join_parts(release_target, "changelog", now_year),
                _join_parts(release_target, "latest stable version official"),
            ]
        )

    domain_queries = [_join_parts(primary_query, f"site:{domain}") for domain in domains[:3]]
    fallback_queries = _build_fallback_queries(
        strategy=strategy,
        text=text,
        entity=entity,
        focus_terms=focus_terms,
        category_hint=category_hint,
        geo_tail=geo_tail,
    )

    query_roles = {
        "primary": _dedupe_keep_order([primary_query]),
        "validation": _dedupe_keep_order(validation_queries),
        "release_notes": release_note_queries,
        "exact_entity": _dedupe_keep_order([exact_entity_query]),
        "domain_constrained": _dedupe_keep_order(domain_queries),
    }

    ordered_roles = _role_order_for_mode(decision.mode)
    prioritized: list[str] = []
    for role in ordered_roles:
        prioritized.extend(query_roles.get(role) or [])
    prioritized.extend(fallback_queries)
    prioritized = _dedupe_keep_order(prioritized)

    if budget_queries <= 0:
        return WebQueryPlan(
            mode=decision.mode,
            scout_queries=[],
            focused_queries=[],
            fallback_queries=[],
            preferred_domains=domains,
            query_roles=query_roles,
            debug={"query_text": query_debug, "strategy": strategy, "category": category, "region_bias": region_bias},
        )

    bounded = prioritized[:budget_queries]
    if (
        domains
        and decision.mode in {WebSearchMode.TARGETED_SEARCH, WebSearchMode.DEEP_SEARCH}
        and not any("site:" in str(x or "").lower() for x in list(bounded))
    ):
        domain_seed = str((query_roles.get("domain_constrained") or [""])[0] or "").strip()
        if domain_seed:
            if len(bounded) < budget_queries:
                bounded.append(domain_seed)
            else:
                bounded[-1] = domain_seed
            bounded = _dedupe_keep_order(bounded)[:budget_queries]

    scout_cap, focused_cap = _phase_caps(decision.mode)
    scout = bounded[: max(1, min(scout_cap, len(bounded)))]
    remaining = bounded[len(scout) :]
    focused = remaining[: max(0, focused_cap)]
    fallback = remaining[len(focused) :]

    return WebQueryPlan(
        mode=decision.mode,
        scout_queries=scout,
        focused_queries=focused,
        fallback_queries=fallback,
        preferred_domains=domains,
        query_roles=query_roles,
        debug={"query_text": query_debug, "strategy": strategy, "category": category, "region_bias": region_bias},
    )
def _query_strategy(*, text: str, classification: QueryClassification) -> str:
    if _uses_docs_strategy(text=text, classification=classification):
        return "docs"
    category = str(classification.primary_category or "").strip().lower()
    low = str(text or "").strip().lower()
    if category == "weather" or any(marker in low for marker in _WEATHER_MARKERS):
        return "weather"
    if category == "price" or any(marker in low for marker in _PRICE_MARKERS):
        return "price"
    if category == "finance" or any(marker in low for marker in _FINANCE_MARKERS):
        return "finance"
    if category == "external" and any(marker in low for marker in _HISTORICAL_MARKERS):
        return "historical"
    if category == "news" or any(marker in low for marker in _NEWS_MARKERS):
        return "news"
    return "generic"


def _uses_docs_strategy(*, text: str, classification: QueryClassification) -> bool:
    category = str(classification.primary_category or "").strip().lower()
    if category in {"version", "docs"}:
        return True

    low = str(text or "").strip().lower()
    if any(marker in low for marker in _RELEASE_QUERY_MARKERS):
        return True
    if any(marker in low for marker in _DOCS_QUERY_MARKERS) and not any(marker in low for marker in _NON_DOCS_OVERRIDES):
        return True
    return False


def _build_primary_query(*, strategy: str, text: str, geo_tail: str) -> str:
    if strategy == "weather":
        return _join_parts(text, geo_tail)
    if strategy == "finance":
        finance_queries = _finance_primary_queries(text=text, geo_tail=geo_tail)
        if finance_queries:
            return finance_queries[0]
        return _join_parts(text, geo_tail)
    if strategy == "price":
        target = _extract_primary_entity(text) or text
        return _join_parts(target, "price", _extract_temporal_focus(text), geo_tail)
    if strategy == "historical":
        target = _extract_primary_entity(text) or text
        return _join_parts(target, "history", _extract_temporal_focus(text), geo_tail)
    return _join_parts(text, geo_tail)


def _extract_primary_entity(text: str) -> str:
    src = str(text or "").strip()
    if not src:
        return ""

    quoted = _extract_quoted_entities(src)
    if quoted:
        return quoted[0]

    tokens = _WORD_RE.findall(src)
    if not tokens:
        return ""

    spans = _candidate_entity_spans(tokens)
    if spans:
        return spans[0]

    strong = [x for x in tokens if any(ch.isupper() for ch in x) or any(ch.isdigit() for ch in x)]
    if strong:
        return " ".join(strong[:4]).strip()

    filtered = [t for t in tokens if not _is_entity_noise_token(t)]
    return " ".join(filtered[:4]).strip()


def _candidate_entity_spans(tokens: list[str]) -> list[str]:
    spans: list[list[tuple[int, str]]] = []
    current: list[tuple[int, str]] = []
    for idx, token in enumerate(list(tokens or [])):
        if _is_entity_noise_token(token):
            if current:
                spans.append(current)
                current = []
            continue
        current.append((idx, str(token)))
    if current:
        spans.append(current)

    ranked = sorted(
        spans,
        key=lambda span: (
            1 if not str(span[0][1]).isdigit() else 0,
            sum(1 for _, token in span if any(ch.isalpha() for ch in token)),
            -sum(1 for _, token in span if str(token).isdigit()),
            len(span),
            sum(1 for _, token in span if any(ch.isupper() for ch in token) or any(ch.isdigit() for ch in token)),
            sum(len(token) for _, token in span),
            -int(span[0][0]),
        ),
        reverse=True,
    )
    out: list[str] = []
    for span in ranked:
        value = " ".join(token for _, token in span).strip()
        if value and value not in out:
            out.append(value)
    return out


def _is_entity_noise_token(token: str) -> bool:
    low = str(token or "").strip().lower()
    if not low:
        return True
    if low in _ENTITY_NOISE_WORDS:
        return True
    if low.isdigit() and len(low) == 4 and low.startswith("20"):
        return True
    return any(low.startswith(prefix) for prefix in _ENTITY_NOISE_PREFIXES)


def _extract_quoted_entities(text: str) -> list[str]:
    out: list[str] = []
    for match in _QUOTED_RE.findall(str(text or "")):
        for group in list(match):
            token = str(group or "").strip()
            if token and token not in out:
                out.append(token)
    return out


def _extract_focus_terms(text: str, *, entity: str) -> list[str]:
    src = str(text or "").strip()
    entity_tokens = {x.lower() for x in _WORD_RE.findall(str(entity or ""))}
    out: list[str] = []
    for token in _WORD_RE.findall(src):
        item = str(token or "").strip().lower()
        if not item or item in _FOCUS_STOPWORDS or item in entity_tokens:
            continue
        if item not in out:
            out.append(item)
    return out[:10]


def _build_validation_queries(
    *,
    strategy: str,
    text: str,
    entity: str,
    focus_terms: list[str],
    category_hint: str,
    geo_tail: str,
) -> list[str]:
    focus_tail = " ".join(focus_terms[:4]).strip()
    target = entity or _extract_primary_product(text) or text

    if strategy == "docs":
        return _dedupe_keep_order(
            [
                _join_parts(target, focus_tail, "official documentation", geo_tail),
                _join_parts(target, "official source", category_hint or "documentation"),
            ]
        )

    if strategy == "finance":
        return _finance_validation_queries(text=text, geo_tail=geo_tail)

    if strategy == "price":
        time_tail = _extract_temporal_focus(text)
        return _dedupe_keep_order(
            [
                _join_parts(target, "price", time_tail, geo_tail),
                _join_parts(f"\"{target}\"", "pricing", time_tail, geo_tail),
            ]
        )

    if strategy == "weather":
        time_tail = _extract_temporal_focus(text)
        return _dedupe_keep_order(
            [
                _join_parts("weather forecast", time_tail, geo_tail),
                _join_parts("погода прогноз", time_tail, geo_tail),
            ]
        )

    if strategy == "news":
        return _dedupe_keep_order(
            [
                _join_parts(f"\"{target}\"", "latest mention"),
                _join_parts(target, "recent news", geo_tail),
            ]
        )

    if strategy == "historical":
        return _dedupe_keep_order(
            [
                _join_parts(f"\"{target}\"", "exact date"),
                _join_parts(target, "history timeline"),
                _join_parts(target, "archive", geo_tail),
            ]
        )

    return _dedupe_keep_order(
        [
            _join_parts(target, focus_tail or category_hint, geo_tail),
            _join_parts(f"\"{target}\"", category_hint if category_hint else ""),
        ]
    )


def _build_exact_entity_query(
    *,
    strategy: str,
    text: str,
    entity: str,
    focus_terms: list[str],
    geo_tail: str,
    category_hint: str,
) -> str:
    target = entity or _extract_primary_product(text)

    if strategy == "docs":
        if not target:
            return ""
        return _join_parts(f"\"{target}\"", category_hint or (focus_terms[0] if focus_terms else "official"))

    if strategy == "finance":
        pairs = _extract_finance_pairs(text)
        subject = _extract_finance_subject(text) or (pairs[0] if pairs else target)
        if not subject:
            return ""
        return _join_parts(f"\"{subject}\"", "exchange rate")

    if strategy == "price":
        if not target:
            return ""
        return _join_parts(f"\"{target}\"", "price")

    if strategy == "weather":
        if geo_tail:
            return _join_parts(f"\"{geo_tail}\"", "weather")
        if target:
            return _join_parts(f"\"{target}\"", "weather")
        return ""

    if strategy == "news":
        if not target:
            return ""
        return _join_parts(f"\"{target}\"")

    if strategy == "historical":
        if not target:
            return ""
        return _join_parts(f"\"{target}\"", "history")

    if not target:
        return ""
    return _join_parts(f"\"{target}\"")


def _build_fallback_queries(
    *,
    strategy: str,
    text: str,
    entity: str,
    focus_terms: list[str],
    category_hint: str,
    geo_tail: str,
) -> list[str]:
    target = entity or _extract_primary_product(text) or text

    if strategy == "docs":
        return _dedupe_keep_order(
            [
                _join_parts(target, category_hint or "documentation", geo_tail),
                _join_parts(target, "official", geo_tail),
            ]
        )

    if strategy == "finance":
        subject = _extract_finance_subject(text) or target
        return _dedupe_keep_order(
            [
                _join_parts(subject, "rate", geo_tail),
                _join_parts(subject, "market rate", geo_tail),
                _join_parts(text, geo_tail),
            ]
        )

    if strategy == "price":
        return _dedupe_keep_order(
            [
                _join_parts(target, "price", geo_tail),
                _join_parts(target, "cost", geo_tail),
                _join_parts(text, geo_tail),
            ]
        )

    if strategy == "weather":
        time_tail = _extract_temporal_focus(text)
        return _dedupe_keep_order(
            [
                _join_parts("forecast", time_tail, geo_tail),
                _join_parts("weather", time_tail, geo_tail),
                _join_parts(text, geo_tail),
            ]
        )

    if strategy == "news":
        return _dedupe_keep_order(
            [
                _join_parts(target, "news", geo_tail),
                _join_parts(target, "recent mention", geo_tail),
                _join_parts(target, "reported", geo_tail),
            ]
        )

    if strategy == "historical":
        return _dedupe_keep_order(
            [
                _join_parts(target, "history", geo_tail),
                _join_parts(target, "timeline", geo_tail),
                _join_parts(target, "archive", geo_tail),
            ]
        )

    return _dedupe_keep_order(
        [
            _join_parts(target, category_hint, geo_tail),
            _join_parts(target, " ".join(focus_terms[:2]), geo_tail),
            _join_parts(text, geo_tail),
        ]
    )


def _extract_finance_subject(text: str) -> str:
    pairs = _extract_finance_pairs(text)
    if pairs:
        return pairs[0]
    codes: list[str] = []
    for token in _WORD_RE.findall(str(text or "").lower()):
        code = _CURRENCY_ALIASES.get(str(token or "").strip().lower())
        if code and code not in codes:
            codes.append(code)
    if len(codes) >= 2:
        return " ".join(codes[:2])
    if codes:
        return codes[0]
    return ""


def _extract_finance_pairs(text: str) -> list[str]:
    low = str(text or "").strip().lower()
    codes: list[str] = []
    for token in _WORD_RE.findall(low):
        code = _CURRENCY_ALIASES.get(str(token or "").strip().lower())
        if code and code not in codes:
            codes.append(code)
    out: list[str] = []
    has_uah_context = any(token in low for token in ("грн", "грив", "uah"))
    for code in list(codes):
        if code != "UAH" and has_uah_context:
            pair = f"{code}/UAH"
            if pair not in out:
                out.append(pair)
    if not out and len(codes) >= 2:
        out.append(f"{codes[0]}/{codes[1]}")
    return out[:3]


def _finance_validation_queries(*, text: str, geo_tail: str) -> list[str]:
    time_tail = _extract_temporal_focus(text)
    pairs = _extract_finance_pairs(text)
    if pairs:
        queries: list[str] = []
        for pair in list(pairs):
            base, _, quote = pair.partition("/")
            queries.append(_join_parts(pair, "exchange rate", time_tail, geo_tail))
            queries.append(_join_parts(base, quote, "rate", time_tail, geo_tail))
        return _dedupe_keep_order(queries)
    subject = _extract_finance_subject(text) or text
    return _dedupe_keep_order(
        [
            _join_parts(subject, "exchange rate", time_tail, geo_tail),
            _join_parts(f"\"{subject}\"", "rate", time_tail, geo_tail),
        ]
    )


def _finance_primary_queries(*, text: str, geo_tail: str) -> list[str]:
    time_tail = _extract_temporal_focus(text)
    pairs = _extract_finance_pairs(text)
    if pairs:
        out: list[str] = []
        for pair in list(pairs):
            base, _, quote = pair.partition("/")
            out.append(_join_parts(base, quote, "exchange rate", time_tail, geo_tail))
        return _dedupe_keep_order(out)
    return []


def _extract_temporal_focus(text: str) -> str:
    low = str(text or "").strip().lower()
    for marker in _TEMPORAL_TERMS:
        if marker in low:
            return marker
    return ""


def _extract_primary_product(text: str) -> str:
    tokens = [token for token in _WORD_RE.findall(str(text or "")) if not _is_entity_noise_token(token)]
    if not tokens:
        return ""
    return " ".join(tokens[:4]).strip()


def _normalize_domains(domains: list[str] | None, *, strategy: str = "", region_bias: str = "") -> list[str]:
    out: list[str] = []
    for row in list(domains or []):
        item = str(row or "").strip().lower()
        if not item:
            continue
        if item.startswith("www."):
            item = item[4:]
        if item not in out:
            out.append(item)
    return _filter_domains_for_strategy(out, strategy=strategy, region_bias=region_bias)


def _filter_domains_for_strategy(domains: list[str], *, strategy: str, region_bias: str = "") -> list[str]:
    if not domains:
        return []
    if strategy == "docs":
        return _sort_domains_for_region_bias(list(domains), region_bias=region_bias)
    if strategy == "finance":
        return _sort_domains_for_region_bias(
            [domain for domain in domains if _matches_domain_hints(domain, _FINANCE_DOMAIN_HINTS)],
            region_bias=region_bias,
        )
    if strategy == "price":
        return _sort_domains_for_region_bias(
            [domain for domain in domains if _matches_domain_hints(domain, _FINANCE_DOMAIN_HINTS + _NEWS_DOMAIN_HINTS)],
            region_bias=region_bias,
        )
    if strategy == "weather":
        return _sort_domains_for_region_bias(
            [domain for domain in domains if _matches_domain_hints(domain, _WEATHER_DOMAIN_HINTS)],
            region_bias=region_bias,
        )
    if strategy == "news":
        return _sort_domains_for_region_bias(
            [domain for domain in domains if _matches_domain_hints(domain, _NEWS_DOMAIN_HINTS)],
            region_bias=region_bias,
        )
    if strategy == "historical":
        return _sort_domains_for_region_bias(
            [domain for domain in domains if _matches_domain_hints(domain, _HISTORICAL_DOMAIN_HINTS)],
            region_bias=region_bias,
        )
    return _sort_domains_for_region_bias(
        [domain for domain in domains if not _matches_domain_hints(domain, _DOCS_DOMAIN_HINTS)],
        region_bias=region_bias,
    )


def _matches_domain_hints(domain: str, hints: tuple[str, ...]) -> bool:
    src = str(domain or "").strip().lower()
    if not src:
        return False
    return any(str(hint or "").strip().lower() in src for hint in hints)


def _category_hint(*, strategy: str, category: str) -> str:
    if strategy == "docs":
        if category == "version":
            return "version"
        return "documentation"
    if strategy == "finance":
        return "exchange rate"
    if strategy == "price":
        return "price"
    if strategy == "weather":
        return "forecast"
    if strategy == "news":
        return "news"
    if strategy == "historical":
        return "history"
    value = str(category or "").strip().lower()
    mapping = {
        "price": "price",
        "refactor": "best practices",
        "architecture": "architecture",
    }
    return str(mapping.get(value, "")).strip()


def _needs_release_notes(*, text: str, classification: QueryClassification, strategy: str) -> bool:
    if strategy != "docs":
        return False
    if str(classification.primary_category or "").strip().lower() == "version":
        return True
    low = str(text or "").strip().lower()
    return any(marker in low for marker in _RELEASE_QUERY_MARKERS)


def _normalize_geo_hint(value: str) -> str:
    src = str(value or "").strip()
    low = src.lower()
    if low in {"internet", "интернет", "интернете", "web", "веб", "site", "сайт"}:
        return ""
    return src


def _should_bias_geo(*, text: str, classification: QueryClassification, geo_hint: str) -> bool:
    geo = str(geo_hint or "").strip()
    if not geo:
        return False
    low = str(text or "").strip().lower()
    if geo.lower() in low:
        return False
    category = str(classification.primary_category or "").strip().lower()
    if category == "finance":
        return any(token in low for token in ("bank", "cash", "buy", "sell", "обмен", "межбанк", "наличн"))
    if category in {"weather", "news"}:
        return True
    if any(token in low for token in _WEATHER_MARKERS + _FINANCE_MARKERS + _NEWS_MARKERS):
        if category == "finance":
            return any(token in low for token in ("bank", "cash", "buy", "sell", "обмен", "межбанк", "наличн"))
        return True
    if classification.is_external_fact_question and classification.requires_freshness:
        return True
    return False


def _planner_region_bias(*, text: str, strategy: str) -> str:
    low = str(text or "").strip().lower()
    if strategy == "finance" and any(token in low for token in ("uah", "грн", "грив")):
        return "ua"
    if strategy == "weather" and any(token in low for token in ("kyiv", "kiev", "ukraine", "киев", "київ", "украин", "україн")):
        return "ua"
    return ""


def _sort_domains_for_region_bias(domains: list[str], *, region_bias: str) -> list[str]:
    bias = str(region_bias or "").strip().lower()
    if bias != "ua":
        return list(domains or [])

    def _key(domain: str) -> tuple[int, str]:
        host = str(domain or "").strip().lower()
        if host.endswith(".ua") or host.endswith(".com.ua"):
            return (0, host)
        if host.endswith(".ru") or host.endswith(".su"):
            return (2, host)
        return (1, host)

    return sorted([str(x or "").strip().lower() for x in list(domains or []) if str(x or "").strip()], key=_key)


def _role_order_for_mode(mode: WebSearchMode) -> list[str]:
    if mode == WebSearchMode.VERIFY_ONLY:
        return ["primary", "validation", "exact_entity", "domain_constrained", "release_notes"]
    if mode == WebSearchMode.SOFT_SEARCH:
        return ["primary", "exact_entity", "validation", "domain_constrained", "release_notes"]
    if mode == WebSearchMode.TARGETED_SEARCH:
        return ["primary", "validation", "exact_entity", "domain_constrained", "release_notes"]
    if mode == WebSearchMode.DEEP_SEARCH:
        return ["primary", "validation", "domain_constrained", "exact_entity", "release_notes"]
    return ["primary", "validation", "exact_entity", "release_notes", "domain_constrained"]


def _phase_caps(mode: WebSearchMode) -> tuple[int, int]:
    if mode == WebSearchMode.VERIFY_ONLY:
        return (1, 1)
    if mode == WebSearchMode.SOFT_SEARCH:
        return (1, 2)
    if mode == WebSearchMode.TARGETED_SEARCH:
        return (2, 2)
    if mode == WebSearchMode.DEEP_SEARCH:
        return (2, 3)
    return (1, 1)


def _join_parts(*parts: Iterable[str] | str) -> str:
    out: list[str] = []
    for part in parts:
        if isinstance(part, str):
            token = str(part or "").strip()
            if token:
                out.append(token)
            continue
        for row in list(part or []):
            token = str(row or "").strip()
            if token:
                out.append(token)
    return " ".join(out).strip()


def _dedupe_keep_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for row in list(items or []):
        value = str(row or "").strip()
        if not value:
            continue
        key = value.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(value)
    return out
