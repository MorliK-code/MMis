"""
Summary Quality — утилиты проверки качества summary.

Перенесено из memory/summary_quality.py
"""
from __future__ import annotations

import re
from typing import Any


_SPACE_RE = re.compile(r"\s+")
_TRANSCRIPT_LINE_RE = re.compile(r"^\s*[-*]\s*(?:user|assistant|system|tool)\s*:", re.I)
_NOISE_ONLY_RE = re.compile(
    r"^(?:"
    r"hi|hello|hey|thanks?|thank you|"
    r"привет|здравствуй(?:те)?|ок(?:ей)?|ладно|ага|угу|да|нет|"
    r"понял(?:а)?|ясно|спасибо"
    r")[.!?\s]*$",
    re.I,
)
_GREETING_RE = re.compile(r"\b(?:hi|hello|hey|привет|здравствуй(?:те)?)\b", re.I)
_INTRO_RE = re.compile(
    r"(?:"
    r"what(?:'s| is)\s+your\s+name|"
    r"как\s+тебя\s+зовут|"
    r"меня\s+[a-zа-яёіїє' -]{2,40}"
    r")",
    re.I,
)


def _normalize_summary_text(value: Any) -> str:
    text = str(value or "").replace("\r", "\n")
    lines = [_SPACE_RE.sub(" ", line.strip()) for line in text.splitlines()]
    compact = "\n".join([line for line in lines if line])
    return compact.strip()


def looks_like_transcript_summary(value: Any) -> bool:
    text = _normalize_summary_text(value)
    if not text:
        return False
    lines = [line for line in text.splitlines() if line]
    if not lines:
        return False
    transcript_lines = [line for line in lines if _TRANSCRIPT_LINE_RE.match(line)]
    if transcript_lines and len(transcript_lines) >= max(1, len(lines) - 1):
        return True
    return bool(_TRANSCRIPT_LINE_RE.match(lines[0]))


def is_summary_noise_text(value: Any) -> bool:
    text = _SPACE_RE.sub(" ", _normalize_summary_text(value).replace("\n", " ")).strip()
    if not text:
        return True
    if len(text) <= 32 and _NOISE_ONLY_RE.fullmatch(text):
        return True
    if len(text) <= 96 and _GREETING_RE.search(text) and _INTRO_RE.search(text):
        return True
    return False


def is_low_quality_session_summary(value: Any) -> bool:
    text = _normalize_summary_text(value)
    if not text or text == "- none":
        return True
    if looks_like_transcript_summary(text):
        return True
    if is_summary_noise_text(text):
        return True
    return False


def sanitize_session_summary_text(value: Any, *, max_chars: int = 2400) -> str:
    text = _SPACE_RE.sub(" ", _normalize_summary_text(value).replace("\n", " ")).strip()
    if is_low_quality_session_summary(text):
        return ""
    if len(text) <= max(16, int(max_chars)):
        return text
    clipped = text[: max(16, int(max_chars))]
    cut = clipped.rsplit(" ", 1)[0].strip()
    return cut if cut else clipped.strip()


def is_meaningful_summary_turn(value: Any) -> bool:
    text = _SPACE_RE.sub(" ", _normalize_summary_text(value).replace("\n", " ")).strip()
    if not text:
        return False
    if len(text) <= 32 and _NOISE_ONLY_RE.fullmatch(text):
        return False
    if len(text) <= 96 and _GREETING_RE.search(text) and _INTRO_RE.search(text):
        return False
    return True
