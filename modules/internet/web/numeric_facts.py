from __future__ import annotations

import re
from dataclasses import dataclass, field


_NUM_RE = re.compile(r"(?<![\w/])-?\d+(?:[\.,]\d+)?\b")
_DATE_RE = re.compile(r"\b(?:\d{4}-\d{2}-\d{2}|\d{1,2}[./-]\d{1,2}[./-]\d{2,4}|20\d{2}|19\d{2})\b")
_SENTENCE_SPLIT_RE = re.compile(r"[.!?\n;]+")

_FX_MARKERS = (
    "exchange rate",
    "currency",
    "forex",
    "fx",
    "official rate",
    "market rate",
    "cash rate",
    "buy",
    "sell",
    "межбанк",
    "налич",
    "курс",
    "валют",
    "обмен",
    "грн",
    "грив",
    "доллар",
    "евро",
)
_PRICE_MARKERS = (
    "price",
    "pricing",
    "cost",
    "quote",
    "price tag",
    "стоит",
    "стоим",
    "цена",
    "прайс",
)
_HISTORY_MARKERS = (
    "historical",
    "history",
    "archive",
    "timeline",
    "recorded",
    "when exactly",
    "exact date",
    "what date",
    "histor",
    "истор",
    "архив",
    "хронолог",
    "дата",
    "точн",
    "когда именно",
    "в каком году",
    "в ",
)
_DATE_QUERY_MARKERS = (
    "date",
    "exact date",
    "year",
    "дата",
    "дату",
    "год",
    "году",
    "когда",
    "когда именно",
)
_DATE_CONTEXT_MARKERS = (
    "founded",
    "signed",
    "launched",
    "announced",
    "created",
    "основан",
    "подписан",
    "запущен",
    "объявлен",
    "дата",
    "год",
    "в году",
)
_NOISE_MARKERS = (
    "views",
    "view",
    "subscribers",
    "comments",
    "likes",
    "reply",
    "thread",
    "article id",
    "post id",
    "order",
    "sku",
    "captcha",
    "support",
    "help",
    "cookie",
    "login",
    "watch",
    "playlist",
    "подписчик",
    "просмотр",
    "коммент",
    "лайк",
    "id",
    "артикул",
    "заказ",
    "серия",
)
_CURRENT_QUERY_MARKERS = (
    "today",
    "current",
    "right now",
    "live",
    "latest",
    "\u0441\u0435\u0433\u043e\u0434\u043d\u044f",
    "\u043d\u0430 \u0441\u0435\u0433\u043e\u0434\u043d\u044f",
    "\u0441\u0435\u0439\u0447\u0430\u0441",
    "\u0442\u0435\u043a\u0443\u0449",
)
_EXACTNESS_MARKERS = (
    "exact",
    "precise",
    "to the cent",
    "\u0442\u043e\u0447\u043d",
    "\u0434\u043e \u043a\u043e\u043f\u0435\u0435\u043a",
    "\u0434\u043e \u043a\u043e\u043f\u0456\u0439\u043a\u0438",
)
_CASH_RATE_MARKERS = (
    "cash rate",
    "cash exchange",
    "cash usd",
    "cash eur",
    "exchange office",
    "black market",
    "auction",
    "\u043d\u0430\u043b\u0438\u0447",
    "\u043d\u0430\u043b\u0456\u0447",
    "\u0433\u043e\u0442\u0456\u0432",
    "\u043e\u0431\u043c\u0435\u043d",
)
_NBU_RATE_MARKERS = (
    "nbu",
    "national bank",
    "central bank",
    "official rate",
    "official exchange rate",
    "bank.gov",
    "\u043d\u0431\u0443",
    "\u043d\u0430\u0446\u0456\u043e\u043d\u0430\u043b\u044c\u043d\u0438\u0439 \u0431\u0430\u043d\u043a",
    "\u043d\u0430\u0446\u0431\u0430\u043d\u043a",
    "\u043e\u0444\u0438\u0446\u0438\u0430\u043b\u044c\u043d",
)
_BANK_RATE_MARKERS = (
    "bank rate",
    "rates in banks",
    "bank rates",
    "banks rate",
    "/banks",
    "in banks",
    "\u0431\u0430\u043d\u043a\u0438",
    "\u0432 \u0431\u0430\u043d\u043a\u0430\u0445",
    "\u043a\u0443\u0440\u0441 \u0432 \u0431\u0430\u043d\u043a\u0430\u0445",
)
_HISTORICAL_RATE_MARKERS = (
    "historical",
    "history",
    "archive",
    "archived",
    "chart",
    "charts",
    "dynamics",
    "timeline",
    "/history",
    "/historical",
    "/archive",
    "\u0438\u0441\u0442\u043e\u0440",
    "\u0430\u0440\u0445\u0438\u0432",
    "\u0434\u0438\u043d\u0430\u043c\u0438\u043a",
    "\u043d\u0430 \u0434\u0430\u0442\u0443",
)
_INDEX_RATE_MARKERS = (
    "index",
    "average rate",
    "average buy",
    "average sell",
    "aggregate",
    "aggregated",
    "currency index",
    "\u0441\u0440\u0435\u0434\u043d\u0438\u0439",
    "\u0430\u0433\u0440\u0435\u0433",
    "\u0438\u043d\u0434\u0435\u043a\u0441",
)
_OVERVIEW_RATE_MARKERS = (
    "exchange rate today",
    "currency rate today",
    "currency overview",
    "currency converter",
    "market rate",
    "usd/uah",
    "eur/uah",
    "gbp/uah",
    "/currency/",
    "\u043a\u0443\u0440\u0441 \u0432\u0430\u043b\u044e\u0442",
    "\u043a\u043e\u043d\u0432\u0435\u0440\u0442\u0435\u0440",
)
_OFFICIAL_RATE_MARKERS = (
    "official rate",
    "official exchange rate",
    "reference rate",
    "reference price",
    "official price",
    "\u043e\u0444\u0438\u0446\u0438\u0430\u043b\u044c\u043d",
    "\u043e\u0444\u0456\u0446\u0456\u0439\u043d",
    "\u0434\u043e\u0432\u0456\u0434\u043a\u043e\u0432",
)
_NEWS_ANALYSIS_MARKERS = (
    "analysis",
    "analyst",
    "market update",
    "market review",
    "commentary",
    "opinion",
    "news",
    "\u0430\u043d\u0430\u043b\u0438\u0437",
    "\u043e\u0431\u0437\u043e\u0440",
    "\u043d\u043e\u0432\u043e\u0441\u0442",
    "\u0441\u0442\u0430\u0442\u044c",
)
_REFERENCE_PAGE_MARKERS = (
    "wikipedia",
    "britannica",
    "encyclopedia",
    "reference",
    "museum",
    "archive.org",
    ".gov",
    ".edu",
    "\u0441\u043f\u0440\u0430\u0432\u043a",
    "\u044d\u043d\u0446\u0438\u043a\u043b\u043e\u043f",
    "\u0432\u0456\u043a\u0456\u043f\u0435\u0434",
)
_WEATHER_QUERY_MARKERS = (
    "weather",
    "forecast",
    "temperature",
    "degrees",
    "rain",
    "wind",
    "humidity",
    "\u043f\u043e\u0433\u043e\u0434",
    "\u043f\u0440\u043e\u0433\u043d\u043e\u0437",
    "\u0442\u0435\u043c\u043f\u0435\u0440\u0430\u0442\u0443\u0440",
    "\u0433\u0440\u0430\u0434\u0443\u0441",
    "\u0434\u043e\u0436\u0434",
    "\u0432\u0435\u0442\u0435\u0440",
)
_WEATHER_NUMERIC_QUERY_MARKERS = (
    "temperature",
    "degrees",
    "high",
    "low",
    "feels like",
    "hot",
    "cold",
    "\u0442\u0435\u043c\u043f\u0435\u0440\u0430\u0442\u0443\u0440",
    "\u0433\u0440\u0430\u0434\u0443\u0441",
    "\u0436\u0430\u0440\u043a",
    "\u0445\u043e\u043b\u043e\u0434",
)
_WEATHER_VALUE_MARKERS = (
    "temperature",
    "feels like",
    "high",
    "low",
    "celsius",
    "fahrenheit",
    "\u0442\u0435\u043c\u043f\u0435\u0440\u0430\u0442\u0443\u0440",
    "\u0433\u0440\u0430\u0434\u0443\u0441",
    "\u043c\u0438\u043d\u0438\u043c\u0443\u043c",
    "\u043c\u0430\u043a\u0441\u0438\u043c\u0443\u043c",
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
_CURRENCY_UNIT_MARKERS = {
    "USD": ("usd", "$", "доллар", "доллара", "долларов"),
    "EUR": ("eur", "€", "евро"),
    "UAH": ("uah", "грн", "грив", "₴"),
    "GBP": ("gbp", "£", "фунт"),
    "BTC": ("btc", "биткоин"),
    "ETH": ("eth", "эфир"),
}


@dataclass(frozen=True)
class NumericCandidate:
    value: float
    value_text: str
    unit: str = ""
    pair: str = ""
    target_entity: str = ""
    page_type: str = ""
    source_url: str = ""
    time_scope: str = ""
    confidence: float = 0.0
    source_snippet: str = ""
    context: str = ""
    reason: str = ""
    matched_tokens: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "value": float(self.value),
            "value_text": str(self.value_text or ""),
            "unit": str(self.unit or ""),
            "pair": str(self.pair or ""),
            "entity_pair": str(self.pair or ""),
            "target_entity": str(self.target_entity or ""),
            "page_type": str(self.page_type or ""),
            "source_url": str(self.source_url or ""),
            "time_scope": str(self.time_scope or ""),
            "confidence": round(float(self.confidence), 6),
            "evidence_confidence": round(float(self.confidence), 6),
            "source_snippet": str(self.source_snippet or ""),
            "context": str(self.context or ""),
            "reason": str(self.reason or ""),
            "matched_tokens": [str(x or "").strip() for x in list(self.matched_tokens or []) if str(x or "").strip()],
        }


@dataclass(frozen=True)
class NumericExtractionResult:
    profile: str
    selected: list[NumericCandidate] = field(default_factory=list)
    rejected: list[NumericCandidate] = field(default_factory=list)
    target_pairs: list[str] = field(default_factory=list)
    target_units: list[str] = field(default_factory=list)
    strict_required: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "profile": str(self.profile or ""),
            "selected": [row.to_dict() for row in list(self.selected or [])],
            "rejected": [row.to_dict() for row in list(self.rejected or [])],
            "target_pairs": [str(x or "").strip() for x in list(self.target_pairs or []) if str(x or "").strip()],
            "target_units": [str(x or "").strip() for x in list(self.target_units or []) if str(x or "").strip()],
            "strict_required": bool(self.strict_required),
        }


def infer_numeric_profile(*, query_category: str, query_text: str) -> str:
    category = str(query_category or "").strip().lower()
    low = str(query_text or "").strip().lower()
    if category == "external" and (
        _contains_any(low, _HISTORY_MARKERS)
        or _contains_any(low, _DATE_QUERY_MARKERS)
        or bool(_DATE_RE.search(low))
    ):
        return "historical"
    if category == "finance" or _contains_any(low, _FX_MARKERS):
        return "fx_rate"
    if category == "price" or _contains_any(low, _PRICE_MARKERS):
        return "price"
    if category == "weather" or (_contains_any(low, _WEATHER_QUERY_MARKERS) and _contains_any(low, _WEATHER_NUMERIC_QUERY_MARKERS)):
        return "weather"
    return "generic"


def requires_strict_numeric_evidence(*, query_category: str, query_text: str) -> bool:
    return infer_numeric_profile(query_category=query_category, query_text=query_text) in {"fx_rate", "price", "historical", "weather"}


def extract_numeric_candidates(
    *,
    text: str,
    snippet: str = "",
    title: str = "",
    url: str = "",
    key_facts: list[str] | None = None,
    query_category: str = "",
    query_text: str = "",
    max_selected: int = 6,
    max_rejected: int = 8,
) -> NumericExtractionResult:
    profile = infer_numeric_profile(query_category=query_category, query_text=query_text)
    strict_required = profile in {"fx_rate", "price", "historical"}
    target_units, target_pairs = _query_targets(query_text, profile=profile)
    factual_page_type = detect_factual_page_type(
        url=url,
        title=title,
        snippet=snippet,
        text=text,
        query_category=query_category,
        query_text=query_text,
    )
    requested_rate_type = detect_requested_currency_rate_type(query_text)
    query_time_scope = detect_numeric_time_scope(query_text, profile=profile)

    candidates: list[NumericCandidate] = []
    rejected: list[NumericCandidate] = []
    seen = set()
    ordered_segments = _segments(text=text, snippet=snippet, title=title, key_facts=key_facts)

    for source_label, segment in ordered_segments:
        context = str(segment or "").strip()
        if not context:
            continue
        context_low = context.lower()
        if source_label != "key_fact" and len(context) > 320:
            context = context[:320].strip()
            context_low = context.lower()
        for match in list(_NUM_RE.finditer(context)):
            raw_value = str(match.group(0) or "").strip()
            value = _parse_number(raw_value)
            if value is None:
                continue
            if profile != "weather" and value <= 0:
                continue
            candidate = _assess_candidate(
                value=value,
                value_text=raw_value,
                context=context,
                context_low=context_low,
                source_label=source_label,
                profile=profile,
                target_units=target_units,
                target_pairs=target_pairs,
                page_type=factual_page_type,
                source_url=url,
                requested_rate_type=requested_rate_type,
                query_time_scope=query_time_scope,
            )
            if candidate is None:
                continue
            dedupe_key = (
                round(float(candidate.value), 6),
                str(candidate.pair or ""),
                str(candidate.unit or ""),
                str(candidate.context or "").lower(),
            )
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            if candidate.confidence >= _selection_threshold(profile=profile):
                candidates.append(candidate)
            else:
                rejected.append(candidate)

    candidates.sort(key=lambda row: (-float(row.confidence), len(str(row.context or ""))))
    rejected.sort(key=lambda row: (-float(row.confidence), len(str(row.context or ""))))
    return NumericExtractionResult(
        profile=profile,
        selected=candidates[: max(1, int(max_selected))],
        rejected=rejected[: max(1, int(max_rejected))],
        target_pairs=target_pairs,
        target_units=target_units,
        strict_required=strict_required,
    )


def numeric_values(candidates: list[dict[str, object]] | list[NumericCandidate]) -> list[float]:
    out: list[float] = []
    for row in list(candidates or []):
        if isinstance(row, NumericCandidate):
            out.append(float(row.value))
            continue
        try:
            out.append(float(dict(row or {}).get("value") or 0.0))
        except Exception:
            continue
    return [value for value in out if value > 0]


def candidate_group_key(row: dict[str, object] | NumericCandidate) -> str:
    if isinstance(row, NumericCandidate):
        pair = str(row.pair or "").strip().upper()
        unit = str(row.unit or "").strip().upper()
        page_type = str(row.page_type or "").strip().lower()
    else:
        data = dict(row or {})
        pair = str(data.get("pair") or "").strip().upper()
        unit = str(data.get("unit") or "").strip().upper()
        page_type = str(data.get("page_type") or "").strip().lower()
    if pair and page_type:
        return f"{pair}|{page_type}"
    if isinstance(row, NumericCandidate):
        target_entity = str(row.target_entity or "").strip().upper()
    else:
        target_entity = str(dict(row or {}).get("target_entity") or "").strip().upper()
    if target_entity and page_type:
        return f"{target_entity}|{page_type}"
    if pair:
        return pair
    if target_entity:
        return target_entity
    if unit:
        return unit
    return "generic"


def detect_numeric_time_scope(query_text: str, *, profile: str = "") -> str:
    low = str(query_text or "").strip().lower()
    if profile == "historical" or _contains_any(low, _HISTORICAL_RATE_MARKERS):
        return "historical"
    if _contains_any(low, _CURRENT_QUERY_MARKERS):
        return "current"
    return ""


def detect_requested_currency_rate_type(query_text: str) -> str:
    low = str(query_text or "").strip().lower()
    if not low:
        return ""
    if _contains_any(low, _HISTORICAL_RATE_MARKERS):
        return "historical_rate"
    if _contains_any(low, _OFFICIAL_RATE_MARKERS):
        return "official_rate"
    if _contains_any(low, _CASH_RATE_MARKERS):
        return "cash_rate"
    if _contains_any(low, _NBU_RATE_MARKERS):
        return "nbu_rate"
    if _contains_any(low, _BANK_RATE_MARKERS):
        return "bank_rate"
    if _contains_any(low, _INDEX_RATE_MARKERS):
        return "currency_index"
    return ""


def detect_factual_page_type(*, url: str = "", title: str = "", snippet: str = "", text: str = "", query_category: str = "", query_text: str = "") -> str:
    blob = " ".join(
        part
        for part in (str(url or "").strip(), str(title or "").strip(), str(snippet or "").strip(), str(text or "").strip())
        if str(part or "").strip()
    ).lower()
    if not blob:
        return ""
    profile = infer_numeric_profile(query_category=query_category, query_text=query_text)
    if profile == "fx_rate" or _contains_any(blob, _FX_MARKERS):
        if _contains_any(blob, _HISTORICAL_RATE_MARKERS):
            return "historical_rate"
        if _contains_any(blob, _NBU_RATE_MARKERS):
            return "nbu_rate"
        if _contains_any(blob, _OFFICIAL_RATE_MARKERS):
            return "official_rate"
        if _contains_any(blob, _CASH_RATE_MARKERS):
            return "cash_rate"
        if _contains_any(blob, _BANK_RATE_MARKERS):
            return "bank_rate"
        if _contains_any(blob, _INDEX_RATE_MARKERS):
            return "index_or_aggregate"
        if _contains_any(blob, _NEWS_ANALYSIS_MARKERS):
            return "news_or_analysis"
        if _contains_any(blob, _OVERVIEW_RATE_MARKERS):
            return "overview_page"
        if _contains_any(blob, _REFERENCE_PAGE_MARKERS):
            return "generic_reference"
        return "overview_page"
    if profile == "historical":
        if _contains_any(blob, _REFERENCE_PAGE_MARKERS):
            return "generic_reference"
        if _contains_any(blob, _NEWS_ANALYSIS_MARKERS):
            return "news_or_analysis"
        if _contains_any(blob, _HISTORICAL_RATE_MARKERS) and _contains_any(blob, ("chart", "archive", "history", "timeline", "historical", "\u0430\u0440\u0445\u0438\u0432")):
            return "historical_rate"
        return "generic_reference"
    if profile == "weather":
        if _contains_any(blob, _HISTORICAL_RATE_MARKERS):
            return "historical_rate"
        if _contains_any(blob, _NEWS_ANALYSIS_MARKERS):
            return "news_or_analysis"
        if _contains_any(blob, _WEATHER_QUERY_MARKERS):
            return "overview_page"
        return "generic_reference"
    if profile == "price":
        if _contains_any(blob, _NEWS_ANALYSIS_MARKERS):
            return "news_or_analysis"
        if _contains_any(blob, _OFFICIAL_RATE_MARKERS):
            return "official_rate"
        if _contains_any(blob, _REFERENCE_PAGE_MARKERS):
            return "generic_reference"
        if _contains_any(blob, _PRICE_MARKERS):
            return "overview_page"
        return "overview_page"
    if _contains_any(blob, _NEWS_ANALYSIS_MARKERS):
        return "news_or_analysis"
    if _contains_any(blob, _REFERENCE_PAGE_MARKERS):
        return "generic_reference"
    return ""


def detect_currency_page_type(*, url: str = "", title: str = "", snippet: str = "", text: str = "") -> str:
    factual = detect_factual_page_type(
        url=url,
        title=title,
        snippet=snippet,
        text=text,
        query_category="finance",
        query_text="currency",
    )
    mapping = {
        "cash_rate": "cash_rate",
        "official_rate": "official_rate",
        "nbu_rate": "nbu_rate",
        "bank_rate": "bank_rate",
        "historical_rate": "historical_rate",
        "index_or_aggregate": "currency_index",
        "overview_page": "currency_overview",
        "news_or_analysis": "news_or_analysis",
        "generic_reference": "currency_page",
    }
    return str(mapping.get(factual, "currency_page"))


def is_current_currency_page_type(page_type: str) -> bool:
    return str(page_type or "").strip().lower() in {
        "cash_rate",
        "official_rate",
        "nbu_rate",
        "bank_rate",
        "currency_overview",
        "currency_index",
        "currency_page",
    }


def currency_page_type_priority(*, page_type: str, requested_rate_type: str = "", query_time_scope: str = "") -> float:
    current_page = str(page_type or "").strip().lower()
    requested = str(requested_rate_type or "").strip().lower()
    time_scope = str(query_time_scope or "").strip().lower()
    if requested and current_page == requested:
        return 1.0
    if time_scope == "current" and current_page == "historical_rate":
        return 0.02
    priorities = {
        "currency_overview": 0.92,
        "cash_rate": 0.88,
        "official_rate": 0.86,
        "bank_rate": 0.85,
        "nbu_rate": 0.83,
        "currency_index": 0.62,
        "currency_page": 0.56,
        "news_or_analysis": 0.24,
        "historical_rate": 0.08,
    }
    return float(priorities.get(current_page, 0.35))


def representative_value(candidates: list[dict[str, object]] | list[NumericCandidate]) -> float | None:
    rows = []
    for row in list(candidates or []):
        if isinstance(row, NumericCandidate):
            rows.append((float(row.confidence), float(row.value)))
        else:
            data = dict(row or {})
            try:
                rows.append((float(data.get("confidence") or 0.0), float(data.get("value") or 0.0)))
            except Exception:
                continue
    rows = [row for row in rows if row[1] > 0]
    if not rows:
        return None
    rows.sort(key=lambda item: (-item[0], item[1]))
    top = rows[: min(3, len(rows))]
    return float(sum(value for _conf, value in top) / max(1, len(top)))


def conflict_severity_for_values(*, left_value: float, right_value: float, profile: str, group_key: str = "") -> float:
    a = abs(float(left_value or 0.0))
    b = abs(float(right_value or 0.0))
    if a <= 0 or b <= 0:
        return 0.0
    if str(group_key or "").strip().upper() in {"YEAR", "DATE"} or profile == "historical":
        spread = abs(a - b)
        if spread <= 0.5:
            return 0.0
        if spread <= 1.0:
            return 0.35
        if spread <= 3.0:
            return 0.72
        return 0.95
    rel = abs(a - b) / max(1.0, a, b)
    if rel <= 0.05:
        return 0.0
    if rel <= 0.10:
        return 0.25
    if rel <= 0.18:
        return 0.58
    return 0.92


def format_candidate_label(row: dict[str, object] | NumericCandidate) -> str:
    if isinstance(row, NumericCandidate):
        value = float(row.value)
        pair = str(row.pair or "").strip()
        unit = str(row.unit or "").strip()
    else:
        data = dict(row or {})
        value = float(data.get("value") or 0.0)
        pair = str(data.get("pair") or "").strip()
        unit = str(data.get("unit") or "").strip()
    suffix = pair or unit
    if suffix:
        return f"{value:.3f} {suffix}".strip()
    return f"{value:.3f}"


def _selection_threshold(*, profile: str) -> float:
    if profile == "fx_rate":
        return 0.64
    if profile == "price":
        return 0.60
    if profile == "historical":
        return 0.58
    if profile == "weather":
        return 0.56
    return 0.55


def _segments(*, text: str, snippet: str, title: str, key_facts: list[str] | None) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    for fact in list(key_facts or []):
        row = str(fact or "").strip()
        if row and row.casefold() not in seen:
            seen.add(row.casefold())
            out.append(("key_fact", row))
    for source_label, blob in (("body", text), ("snippet", snippet), ("title", title)):
        raw = str(blob or "").strip()
        if not raw:
            continue
        for part in _SENTENCE_SPLIT_RE.split(raw):
            row = str(part or "").strip()
            if len(row) < 6:
                continue
            key = row.casefold()
            if key in seen:
                continue
            seen.add(key)
            out.append((source_label, row))
    return out


def _query_targets(query_text: str, *, profile: str) -> tuple[list[str], list[str]]:
    units: list[str] = []
    pairs: list[str] = []
    low = str(query_text or "").strip().lower()
    codes: list[str] = []
    for token in re.findall(r"[A-Za-zА-Яа-яЁёІіЇїЄєҐґ0-9_/-]+", low):
        code = _CURRENCY_ALIASES.get(str(token or "").strip().lower())
        if code and code not in codes:
            codes.append(code)
    if "/" in low:
        for left, right in re.findall(r"\b([A-Za-z]{3})\s*/\s*([A-Za-z]{3})\b", low, flags=re.I):
            pair = f"{left.upper()}/{right.upper()}"
            if pair not in pairs:
                pairs.append(pair)
    has_uah_context = bool(re.search(r"\b(?:uah|грн|грив)\b", low))
    if profile == "fx_rate":
        for code in list(codes):
            if code != "UAH" and has_uah_context:
                pair = f"{code}/UAH"
                if pair not in pairs:
                    pairs.append(pair)
        if not pairs and len(codes) >= 2:
            pair = f"{codes[0]}/{codes[1]}"
            if pair not in pairs:
                pairs.append(pair)
        for code in list(codes):
            if code not in units:
                units.append(code)
        if has_uah_context and "UAH" not in units:
            units.append("UAH")
    elif profile == "price":
        for code in list(codes):
            if code not in units:
                units.append(code)
        if re.search(r"[$€£₴]", low):
            for unit in ("USD", "EUR", "GBP", "UAH"):
                symbol = _CURRENCY_UNIT_MARKERS.get(unit, ())
                if any(marker in low for marker in symbol):
                    if unit not in units:
                        units.append(unit)
    elif profile == "historical":
        if _contains_any(low, ("дата", "год", "year", "date")):
            units.append("YEAR")
        for code in list(codes):
            if code not in units:
                units.append(code)
        if has_uah_context and "UAH" not in units:
            units.append("UAH")
    return (units, pairs)


def _assess_candidate(
    *,
    value: float,
    value_text: str,
    context: str,
    context_low: str,
    source_label: str,
    profile: str,
    target_units: list[str],
    target_pairs: list[str],
    page_type: str,
    source_url: str,
    requested_rate_type: str,
    query_time_scope: str,
) -> NumericCandidate | None:
    effective_time_scope = query_time_scope or ("historical" if page_type == "historical_rate" else "")
    if _contains_any(context_low, _NOISE_MARKERS):
        return NumericCandidate(
            value=value,
            value_text=value_text,
            target_entity=_candidate_target_entity(profile=profile, pair="", unit=""),
            page_type=page_type,
            source_url=source_url,
            time_scope=effective_time_scope,
            confidence=0.08,
            source_snippet=context[:180],
            context=context[:220],
            reason="noise_context",
            matched_tokens=[],
        )

    matched_tokens: list[str] = []
    pair = _match_pair(context_low=context_low, target_pairs=target_pairs, target_units=target_units)
    unit = _match_unit(context_low=context_low, target_units=target_units)
    if profile == "fx_rate" and not pair:
        pair = _detect_context_pair(context_low)
    if profile in {"fx_rate", "price", "historical"} and not unit:
        unit = _detect_any_unit(context_low)
    if pair:
        matched_tokens.append(pair)
    if unit and unit not in matched_tokens:
        matched_tokens.append(unit)
    target_entity = _candidate_target_entity(profile=profile, pair=pair, unit=unit)

    if profile == "fx_rate":
        finance_hit = _contains_any(context_low, _FX_MARKERS)
        if finance_hit:
            matched_tokens.append("fx_marker")
        if page_type and page_type not in matched_tokens:
            matched_tokens.append(page_type)
        if target_pairs and not pair:
            return NumericCandidate(
                value=value,
                value_text=value_text,
                unit=unit,
                pair="",
                target_entity=target_entity,
                page_type=page_type,
                source_url=source_url,
                time_scope=effective_time_scope,
                confidence=0.20 if finance_hit else 0.06,
                source_snippet=context[:180],
                context=context[:220],
                reason="pair_mismatch",
                matched_tokens=matched_tokens,
            )
        if requested_rate_type and page_type and requested_rate_type != page_type:
            return NumericCandidate(
                value=value,
                value_text=value_text,
                unit=unit,
                pair=pair,
                target_entity=target_entity,
                page_type=page_type,
                source_url=source_url,
                time_scope=effective_time_scope,
                confidence=0.16 if finance_hit else 0.08,
                source_snippet=context[:180],
                context=context[:220],
                reason="rate_type_mismatch",
                matched_tokens=matched_tokens,
            )
        if effective_time_scope == "current" and page_type == "historical_rate":
            return NumericCandidate(
                value=value,
                value_text=value_text,
                unit=unit,
                pair=pair,
                target_entity=target_entity,
                page_type=page_type,
                source_url=source_url,
                time_scope="historical",
                confidence=0.08,
                source_snippet=context[:180],
                context=context[:220],
                reason="historical_rate_page",
                matched_tokens=matched_tokens,
            )
        if value > 500 or value < 0.01:
            return NumericCandidate(
                value=value,
                value_text=value_text,
                unit=unit,
                pair=pair,
                target_entity=target_entity,
                page_type=page_type,
                source_url=source_url,
                time_scope=effective_time_scope,
                confidence=0.10,
                source_snippet=context[:180],
                context=context[:220],
                reason="implausible_fx_value",
                matched_tokens=matched_tokens,
            )
        score = 0.18
        if pair:
            score += 0.34
        elif unit:
            score += 0.18
        if finance_hit:
            score += 0.18
        if "." in value_text or "," in value_text:
            score += 0.10
        if source_label == "key_fact":
            score += 0.12
        if _contains_any(context_low, ("official", "nbu", "national bank", "банк")):
            score += 0.08
        if page_type == "currency_overview":
            score += 0.12
        elif page_type in {"cash_rate", "bank_rate", "nbu_rate"}:
            score += 0.10
        elif page_type == "currency_index":
            score += 0.03
        reason = "matched_fx_context" if score >= _selection_threshold(profile=profile) else "weak_fx_context"
        return NumericCandidate(
            value=value,
            value_text=value_text,
            unit=unit or _pair_quote_unit(pair),
            pair=pair,
            target_entity=target_entity,
            page_type=page_type,
            source_url=source_url,
            time_scope=effective_time_scope or "current",
            confidence=min(0.98, score),
            source_snippet=context[:180],
            context=context[:220],
            reason=reason,
            matched_tokens=matched_tokens,
        )

    if profile == "price":
        price_hit = _contains_any(context_low, _PRICE_MARKERS) or bool(re.search(r"[$€£₴]", context))
        if value > 1_000_000 or value < 0.01:
            return NumericCandidate(
                value=value,
                value_text=value_text,
                unit=unit,
                target_entity=_candidate_target_entity(profile=profile, pair="", unit=unit),
                page_type=page_type,
                source_url=source_url,
                time_scope=effective_time_scope,
                confidence=0.08,
                source_snippet=context[:180],
                context=context[:220],
                reason="implausible_price_value",
                matched_tokens=matched_tokens,
            )
        score = 0.16
        if price_hit:
            score += 0.28
            matched_tokens.append("price_marker")
        if unit:
            score += 0.18
        if "." in value_text or "," in value_text:
            score += 0.06
        if source_label == "key_fact":
            score += 0.10
        reason = "matched_price_context" if score >= _selection_threshold(profile=profile) else "weak_price_context"
        return NumericCandidate(
            value=value,
            value_text=value_text,
            unit=unit,
            target_entity=_candidate_target_entity(profile=profile, pair="", unit=unit),
            page_type=page_type,
            source_url=source_url,
            time_scope=effective_time_scope,
            confidence=min(0.96, score),
            source_snippet=context[:180],
            context=context[:220],
            reason=reason,
            matched_tokens=matched_tokens,
        )

    if profile == "historical":
        historical_hit = _contains_any(context_low, _HISTORY_MARKERS) or _contains_any(context_low, _DATE_CONTEXT_MARKERS)
        if _DATE_RE.search(context_low):
            matched_tokens.append("date_marker")
        if 1800 <= value <= 2100:
            score = 0.30
            if historical_hit:
                score += 0.34
            if source_label == "key_fact":
                score += 0.10
            reason = "matched_historical_date" if score >= _selection_threshold(profile=profile) else "weak_historical_date"
            return NumericCandidate(
                value=value,
                value_text=value_text,
                unit="YEAR",
                target_entity="date",
                page_type=page_type or "historical_rate",
                source_url=source_url,
                time_scope="historical",
                confidence=min(0.94, score),
                source_snippet=context[:180],
                context=context[:220],
                reason=reason,
                matched_tokens=matched_tokens,
            )
        if target_pairs and not pair:
            return NumericCandidate(
                value=value,
                value_text=value_text,
                unit=unit,
                pair="",
                target_entity=_candidate_target_entity(profile=profile, pair="", unit=unit),
                page_type=page_type or "historical_rate",
                source_url=source_url,
                time_scope="historical",
                confidence=0.12,
                source_snippet=context[:180],
                context=context[:220],
                reason="historical_pair_mismatch",
                matched_tokens=matched_tokens,
            )
        score = 0.18
        if pair:
            score += 0.22
        if unit:
            score += 0.10
        if historical_hit:
            score += 0.20
        if source_label == "key_fact":
            score += 0.10
        reason = "matched_historical_numeric" if score >= _selection_threshold(profile=profile) else "weak_historical_numeric"
        return NumericCandidate(
            value=value,
            value_text=value_text,
            unit=unit or _pair_quote_unit(pair),
            pair=pair,
            target_entity=_candidate_target_entity(profile=profile, pair=pair, unit=unit or _pair_quote_unit(pair)),
            page_type=page_type or "historical_rate",
            source_url=source_url,
            time_scope="historical",
            confidence=min(0.92, score),
            source_snippet=context[:180],
            context=context[:220],
            reason=reason,
            matched_tokens=matched_tokens,
        )

    if profile == "weather":
        weather_hit = _contains_any(context_low, _WEATHER_VALUE_MARKERS)
        weather_unit = _weather_unit(context=context, context_low=context_low)
        if not weather_hit and not weather_unit:
            return NumericCandidate(
                value=value,
                value_text=value_text,
                unit=weather_unit,
                target_entity="temperature",
                page_type=page_type,
                source_url=source_url,
                time_scope=effective_time_scope or "current",
                confidence=0.10,
                source_snippet=context[:180],
                context=context[:220],
                reason="weather_context_mismatch",
                matched_tokens=matched_tokens,
            )
        if value < -90 or value > 70:
            return NumericCandidate(
                value=value,
                value_text=value_text,
                unit=weather_unit,
                target_entity="temperature",
                page_type=page_type,
                source_url=source_url,
                time_scope=effective_time_scope or "current",
                confidence=0.08,
                source_snippet=context[:180],
                context=context[:220],
                reason="implausible_weather_value",
                matched_tokens=matched_tokens,
            )
        score = 0.22
        if weather_hit:
            score += 0.26
            matched_tokens.append("weather_marker")
        if weather_unit:
            score += 0.18
        if source_label == "key_fact":
            score += 0.10
        if page_type == "overview_page":
            score += 0.12
        elif page_type == "historical_rate":
            score += 0.04 if effective_time_scope == "historical" else -0.08
        elif page_type == "news_or_analysis":
            score -= 0.08
        reason = "matched_weather_context" if score >= _selection_threshold(profile=profile) else "weak_weather_context"
        return NumericCandidate(
            value=value,
            value_text=value_text,
            unit=weather_unit or "C",
            target_entity="temperature",
            page_type=page_type or "overview_page",
            source_url=source_url,
            time_scope=effective_time_scope or "current",
            confidence=min(0.95, score),
            source_snippet=context[:180],
            context=context[:220],
            reason=reason,
            matched_tokens=matched_tokens,
        )

    generic_hit = bool(target_units or target_pairs)
    if not generic_hit:
        return NumericCandidate(
            value=value,
            value_text=value_text,
            target_entity=_candidate_target_entity(profile=profile, pair=pair, unit=unit),
            page_type=page_type,
            source_url=source_url,
            time_scope=effective_time_scope,
            confidence=0.10,
            source_snippet=context[:180],
            context=context[:220],
            reason="generic_numeric",
            matched_tokens=[],
        )
    return NumericCandidate(
        value=value,
        value_text=value_text,
        unit=unit,
        pair=pair,
        target_entity=_candidate_target_entity(profile=profile, pair=pair, unit=unit),
        page_type=page_type,
        source_url=source_url,
        time_scope=effective_time_scope,
        confidence=0.40,
        source_snippet=context[:180],
        context=context[:220],
        reason="generic_numeric_match",
        matched_tokens=matched_tokens,
    )


def _match_pair(*, context_low: str, target_pairs: list[str], target_units: list[str]) -> str:
    for pair in list(target_pairs or []):
        left, _, right = str(pair or "").partition("/")
        if not left or not right:
            continue
        left_markers = _CURRENCY_UNIT_MARKERS.get(left.upper(), ())
        right_markers = _CURRENCY_UNIT_MARKERS.get(right.upper(), ())
        if any(marker.lower() in context_low for marker in left_markers) and any(marker.lower() in context_low for marker in right_markers):
            return f"{left.upper()}/{right.upper()}"
    if "UAH" in list(target_units or []):
        for unit in list(target_units or []):
            if unit == "UAH":
                continue
            left_markers = _CURRENCY_UNIT_MARKERS.get(unit.upper(), ())
            right_markers = _CURRENCY_UNIT_MARKERS.get("UAH", ())
            if any(marker.lower() in context_low for marker in left_markers) and any(marker.lower() in context_low for marker in right_markers):
                return f"{unit.upper()}/UAH"
    return ""


def _match_unit(*, context_low: str, target_units: list[str]) -> str:
    for unit in list(target_units or []):
        markers = _CURRENCY_UNIT_MARKERS.get(str(unit or "").upper(), ())
        if any(marker.lower() in context_low for marker in markers):
            return str(unit or "").upper()
    if re.search(r"[$€£₴]", context_low):
        if "$" in context_low:
            return "USD"
        if "€" in context_low:
            return "EUR"
        if "£" in context_low:
            return "GBP"
        if "₴" in context_low:
            return "UAH"
    return ""


def _pair_quote_unit(pair: str) -> str:
    text = str(pair or "").strip().upper()
    if "/" in text:
        return str(text.split("/", 1)[1] or "")
    return ""


def _detect_any_unit(context_low: str) -> str:
    for unit, markers in _CURRENCY_UNIT_MARKERS.items():
        if any(str(marker or "").strip().lower() in context_low for marker in markers):
            return str(unit or "").strip().upper()
    return ""


def _detect_context_pair(context_low: str) -> str:
    explicit = re.findall(r"\b([A-Za-z]{3})\s*/\s*([A-Za-z]{3})\b", context_low, flags=re.I)
    if explicit:
        left, right = explicit[0]
        return f"{left.upper()}/{right.upper()}"
    units: list[str] = []
    for unit, markers in _CURRENCY_UNIT_MARKERS.items():
        if any(str(marker or "").strip().lower() in context_low for marker in markers):
            units.append(str(unit or "").strip().upper())
    units = [unit for unit in units if unit]
    if "UAH" in units:
        for unit in units:
            if unit != "UAH":
                return f"{unit}/UAH"
    if len(units) >= 2:
        return f"{units[0]}/{units[1]}"
    return ""


def _candidate_target_entity(*, profile: str, pair: str, unit: str) -> str:
    if pair:
        return str(pair or "").strip().upper()
    if profile == "weather":
        return "temperature"
    if profile == "price":
        return "price"
    if profile == "historical" and str(unit or "").strip().upper() == "YEAR":
        return "date"
    if unit:
        return str(unit or "").strip().upper()
    return str(profile or "").strip().lower()


def _weather_unit(*, context: str, context_low: str) -> str:
    raw = str(context or "")
    low = str(context_low or "").strip().lower()
    if any(token in low for token in ("fahrenheit", "\u0444\u0430\u0440\u0435\u043d\u0433", "\u00b0f")):
        return "F"
    if any(token in low for token in ("celsius", "\u0446\u0435\u043b\u044c\u0441", "\u00b0c", "\u00b0")):
        return "C"
    if "\u00b0" in raw:
        return "C"
    return ""


def _contains_any(text: str, markers: tuple[str, ...]) -> bool:
    src = str(text or "").strip().lower()
    if not src:
        return False
    return any(str(marker or "").strip().lower() in src for marker in markers)


def _parse_number(value: str) -> float | None:
    text = str(value or "").strip().replace(",", ".")
    if not text:
        return None
    try:
        return float(text)
    except Exception:
        return None
