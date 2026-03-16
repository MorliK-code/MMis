from __future__ import annotations

import re
from dataclasses import dataclass, field


_WORD_RE = re.compile(r"[A-Za-z\u0400-\u04ff0-9_]{3,}")
_NUM_RE = re.compile(r"\b\d+(?:[\.,]\d+)?\b")
_DATE_RE = re.compile(r"\b(?:\d{4}-\d{2}-\d{2}|\d{1,2}[./-]\d{1,2}[./-]\d{2,4}|20\d{2}|19\d{2})\b")

_FINANCE_MARKERS = (
    "exchange rate",
    "fx",
    "forex",
    "currency",
    "usd",
    "eur",
    "uah",
    "курс",
    "валют",
    "грив",
    "грн",
    "доллар",
    "евро",
    "межбанк",
    "налич",
    "buy",
    "sell",
)
_PRICE_MARKERS = (
    "price",
    "pricing",
    "cost",
    "quote",
    "rate card",
    "цена",
    "стоимость",
    "сколько стоит",
    "прайс",
)
_HISTORICAL_MARKERS = (
    "historical",
    "history",
    "timeline",
    "archive",
    "archived",
    "recorded",
    "exact date",
    "what date",
    "when exactly",
    "which year",
    "histor",
    "истор",
    "дата",
    "точную дату",
    "точная дата",
    "когда именно",
    "в каком году",
    "архив",
    "хронолог",
    "упомин",
    "последний раз",
)

_FINANCE_DOMAIN_HINTS = ("bank.", "bank.gov", "minfin", "finance.", "forex", "fx", "kurs", "invest", "marketwatch", "xe.com")
_HISTORY_DOMAIN_HINTS = ("wikipedia.org", "britannica.com", "history.com", "archive", "museum", "encyclopedia", "reuters.com", "apnews.com", "bbc.", ".gov", ".edu")

_VIDEO_HINTS = ("youtube.com", "youtu.be", "vimeo.com", "tiktok.com", "/watch", "watch?v=", "/shorts", "/playlist", "/channel", "/video")
_SUPPORT_HINTS = ("support.", "/support", "help.", "/help", "answer/", "/answer?", "support.google", "support.microsoft")
_FORUM_HINTS = ("community.", "forum.", "discuss.", "reddit.com", "quora.com", "/thread", "/threads")
_DOCS_HINTS = ("docs.", "developer.", "readthedocs", "pypi.org", "npmjs.com")


@dataclass(frozen=True)
class SourceTopicalAssessment:
    profile: str
    score: float
    bonus: float
    penalty: float
    page_type: str
    allow_selection: bool
    reasons: list[str] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)


def infer_query_profile(*, query_category: str, query_text: str) -> str:
    category = str(query_category or "").strip().lower()
    low = str(query_text or "").strip().lower()
    if category in {"docs", "version"}:
        return "docs"
    if category == "weather":
        return "weather"
    if category == "news":
        if _contains_any(low, _HISTORICAL_MARKERS):
            return "historical"
        return "news"
    if category == "finance" or _contains_any(low, _FINANCE_MARKERS):
        return "finance"
    if category == "price" or _contains_any(low, _PRICE_MARKERS):
        return "price"
    if category == "external" and _contains_any(low, _HISTORICAL_MARKERS):
        return "historical"
    return "generic"


def assess_source_topical_relevance(
    *,
    domain: str,
    url: str,
    title: str,
    snippet: str,
    text: str = "",
    key_facts: list[str] | None = None,
    query_category: str = "",
    query_text: str = "",
) -> SourceTopicalAssessment:
    profile = infer_query_profile(query_category=query_category, query_text=query_text)
    page_type = _detect_page_type(domain=domain, url=url, title=title)
    strict = profile in {"finance", "price", "historical"}
    blob = " ".join(
        part
        for part in (
            str(domain or "").strip(),
            str(url or "").strip(),
            str(title or "").strip(),
            str(snippet or "").strip(),
            str(text or "").strip(),
            " ".join(list(key_facts or [])),
        )
        if str(part or "").strip()
    ).lower()

    reasons: list[str] = []
    flags: list[str] = []
    score = 0.55
    bonus = 0.0
    penalty = 0.0
    allow_selection = True

    if strict and page_type in {"video", "support", "forum"}:
        allow_selection = False
        penalty = 0.95
        flags.append(f"source_type:{page_type}")
        reasons.append(f"unsupported_{page_type}_source_for_{profile}")
    elif strict and page_type == "docs":
        penalty = max(penalty, 0.78)
        flags.append("source_type:docs")
        reasons.append(f"docs_style_source_for_{profile}")

    if profile == "finance":
        score = _finance_signal_score(domain=domain, blob=blob)
    elif profile == "price":
        score = _price_signal_score(domain=domain, blob=blob)
    elif profile == "historical":
        score = _historical_signal_score(domain=domain, blob=blob)

    if strict:
        if score >= 0.78:
            bonus = 0.12
            reasons.append("strong_topical_match")
        elif score >= 0.58:
            bonus = 0.06
            reasons.append("usable_topical_match")
        elif score <= 0.18:
            penalty = max(penalty, 0.92 if allow_selection else penalty)
            allow_selection = False
            flags.append("topical_mismatch")
            reasons.append("strong_topical_mismatch")
        elif score <= 0.34:
            penalty = max(penalty, 0.60)
            flags.append("weak_topical_match")
            reasons.append("weak_topical_match")

    return SourceTopicalAssessment(
        profile=profile,
        score=max(0.0, min(1.0, float(score))),
        bonus=max(0.0, min(0.18, float(bonus))),
        penalty=max(0.0, min(1.0, float(penalty))),
        page_type=page_type,
        allow_selection=bool(allow_selection),
        reasons=reasons,
        flags=flags,
    )


def is_strict_factual_profile(*, query_category: str, query_text: str) -> bool:
    return infer_query_profile(query_category=query_category, query_text=query_text) in {"finance", "price", "historical"}


def _finance_signal_score(*, domain: str, blob: str) -> float:
    hits = _count_hits(blob, _FINANCE_MARKERS)
    score = 0.08
    if _contains_any(domain, _FINANCE_DOMAIN_HINTS):
        score += 0.34
    if hits:
        score += min(0.34, 0.10 * hits)
    if _contains_any(blob, ("usd/uah", "eur/uah", "gbp/uah", "exchange rate", "official rate", "rate")):
        score += 0.18
    if _NUM_RE.search(blob):
        score += 0.08
    if _contains_any(blob, _HISTORICAL_MARKERS):
        score += 0.02
    return max(0.0, min(1.0, score))


def _price_signal_score(*, domain: str, blob: str) -> float:
    hits = _count_hits(blob, _PRICE_MARKERS)
    score = 0.08
    if _contains_any(domain, _FINANCE_DOMAIN_HINTS):
        score += 0.12
    if hits:
        score += min(0.40, 0.12 * hits)
    if _NUM_RE.search(blob):
        score += 0.14
    if _contains_any(blob, ("usd", "eur", "uah", "$", "€", "грн")):
        score += 0.10
    return max(0.0, min(1.0, score))


def _historical_signal_score(*, domain: str, blob: str) -> float:
    hits = _count_hits(blob, _HISTORICAL_MARKERS)
    dates = len(_DATE_RE.findall(blob))
    score = 0.08
    if _contains_any(domain, _HISTORY_DOMAIN_HINTS):
        score += 0.28
    if hits:
        score += min(0.30, 0.10 * hits)
    if dates:
        score += min(0.20, 0.08 * dates)
    if _contains_any(blob, ("reuters", "ap news", "archive", "timeline", "wikipedia", "encyclopedia")):
        score += 0.12
    return max(0.0, min(1.0, score))


def _detect_page_type(*, domain: str, url: str, title: str) -> str:
    blob = " ".join((str(domain or "").strip().lower(), str(url or "").strip().lower(), str(title or "").strip().lower()))
    if _contains_any(blob, _VIDEO_HINTS):
        return "video"
    if _contains_any(blob, _SUPPORT_HINTS):
        return "support"
    if _contains_any(blob, _FORUM_HINTS):
        return "forum"
    if _contains_any(blob, _DOCS_HINTS):
        return "docs"
    return "page"


def _contains_any(text: str, tokens: tuple[str, ...]) -> bool:
    low = str(text or "").strip().lower()
    if not low:
        return False
    return any(str(token or "").strip().lower() in low for token in tokens)


def _count_hits(text: str, tokens: tuple[str, ...]) -> int:
    low = str(text or "").strip().lower()
    if not low:
        return 0
    hits = 0
    for token in tokens:
        cur = str(token or "").strip().lower()
        if cur and cur in low:
            hits += 1
    return hits
