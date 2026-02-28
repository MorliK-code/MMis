from __future__ import annotations

import re
from dataclasses import dataclass


_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.I)
_CODE_RE = re.compile(
    r"```|`[^`]+`|Traceback|Exception|def\s+\w+\s*\(|class\s+\w+\s*:|"
    r"import\s+\w+|#include\s*<|function\s+\w+\s*\(|\bSELECT\b.+\bFROM\b|"
    r"[A-Za-z]:\\|\\b(?:npm|pip|pytest|uvicorn)\\b",
    re.I | re.S,
)
_EMOJI_RE = re.compile(
    "["
    "\U0001F300-\U0001F5FF"
    "\U0001F600-\U0001F64F"
    "\U0001F680-\U0001F6FF"
    "\U0001F900-\U0001FAFF"
    "]+",
    flags=re.UNICODE,
)
_CYR_RE = re.compile(r"[\u0400-\u04FF]")
_LAT_RE = re.compile(r"[A-Za-z]")
_UK_ONLY_RE = re.compile(r"[ЇїЄєҐґІі]")


@dataclass(frozen=True)
class LanguageResult:
    lang: str
    conf: float
    is_code_like: bool
    has_emoji: bool
    has_url: bool
    cyrillic_ratio: float
    latin_ratio: float
    ukrainian_hint_ratio: float


def analyze(text: str) -> LanguageResult:
    src = str(text or "").strip()
    if not src:
        return LanguageResult(
            lang="unknown",
            conf=0.0,
            is_code_like=False,
            has_emoji=False,
            has_url=False,
            cyrillic_ratio=0.0,
            latin_ratio=0.0,
            ukrainian_hint_ratio=0.0,
        )

    has_url = bool(_URL_RE.search(src))
    has_emoji = bool(_EMOJI_RE.search(src))
    is_code_like = bool(_CODE_RE.search(src))

    cyr = len(_CYR_RE.findall(src))
    lat = len(_LAT_RE.findall(src))
    uk = len(_UK_ONLY_RE.findall(src))
    letters = max(1, cyr + lat)
    cyr_ratio = cyr / letters
    lat_ratio = lat / letters
    uk_ratio = uk / max(1, cyr)

    short_text = len(src) <= 4
    if short_text and src.lower() in {"ok", "ага", "ок", "угу", "yes", "no"}:
        base_lang = "unknown"
        conf = 0.35
    elif cyr and lat:
        base_lang = "mixed"
        conf = 0.55 + 0.25 * min(cyr_ratio, lat_ratio)
    elif cyr:
        base_lang = "uk" if uk_ratio >= 0.06 else "ru"
        conf = 0.62 + min(0.33, cyr_ratio * 0.3 + uk_ratio * 0.4)
    elif lat:
        base_lang = "en"
        conf = 0.6 + min(0.35, lat_ratio * 0.35)
    else:
        base_lang = "unknown"
        conf = 0.3

    if short_text:
        conf = min(conf, 0.58)
    if is_code_like and base_lang in {"ru", "uk", "en"}:
        # Code snippets often include mixed symbols and keywords.
        conf = min(conf, 0.72)
    conf = max(0.0, min(1.0, conf))

    return LanguageResult(
        lang=base_lang,
        conf=conf,
        is_code_like=is_code_like,
        has_emoji=has_emoji,
        has_url=has_url,
        cyrillic_ratio=cyr_ratio,
        latin_ratio=lat_ratio,
        ukrainian_hint_ratio=uk_ratio,
    )


def detect(text: str) -> tuple[str, float]:
    result = analyze(text)
    return result.lang, result.conf


def detect_language(text: str) -> str:
    # Backward-compat helper used by existing call sites.
    return detect(text)[0]

