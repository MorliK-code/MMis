from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class IntentResult:
    label: str
    conf: float
    alt_labels: list[str] = field(default_factory=list)


_PATTERNS = {
    "coding_help": [
        r"\btraceback\b",
        r"\bexception\b",
        r"\berror\b",
        r"\bdebug\b",
        r"\bstack trace\b",
        r"```",
        r"[A-Za-z]:\\",
        r"\bpython\b|\bjs\b|\btypescript\b|\bjava\b|\bc\+\+\b",
    ],
    "ui_request": [
        r"\bui\b",
        r"\bbutton\b|\bicon\b|\bstyle\b|\bcss\b|\btheme\b",
        r"\bинтерфейс\b|\bкнопк[аиу]\b|\bиконк[аиу]\b|\bцвет\b|\bстил[ья]\b",
    ],
    "memory_update": [
        r"\bзапомни\b|\bудали из памяти\b|\bзабудь\b|\bисправь факт\b",
        r"\bremember\b|\bforget\b|\bupdate memory\b",
    ],
    "search": [
        r"\bпоищи\b|\bнайди\b|\bузнай\b|\bпоиск\b",
        r"\bsearch\b|\blook up\b|\bgoogle\b",
    ],
    "complaint": [
        r"\bне работает\b|\bсломал[оа]\b|\bбаг\b|\bфигня\b",
        r"\bdoesn't work\b|\bbroken\b|\bbug\b|\bwtf\b",
    ],
    "nsfw_flirt": [
        r"\bsexy\b|\bkiss\b|\bhot\b|\bflirt\b",
        r"\bпоцел\b|\bсексуал\b|\bфлирт\b|\bинтим\b",
    ],
    "task_request": [
        r"\bсделай\b|\bвыполни\b|\bнастрой\b|\bсоздай\b",
        r"\bdo\b|\bmake\b|\bbuild\b|\bconfigure\b|\bplease\b",
    ],
    "question": [
        r"\?$",
        r"\bкак\b|\bчто\b|\bпочему\b|\bзачем\b|\bгде\b",
        r"\bhow\b|\bwhat\b|\bwhy\b|\bwhere\b|\bwhen\b",
    ],
}


def classify(text: str, lang: str, context_tags: dict[str, Any] | None = None) -> IntentResult:
    src = str(text or "").strip()
    lower = src.lower()
    tags = dict(context_tags or {})

    scores = {
        "chat": 0.22,
        "question": 0.16,
        "task_request": 0.16,
        "coding_help": 0.12,
        "ui_request": 0.08,
        "memory_update": 0.07,
        "search": 0.07,
        "complaint": 0.06,
        "nsfw_flirt": 0.02,
    }

    if not src:
        return IntentResult(label="chat", conf=0.35, alt_labels=["question"])

    for label, patterns in _PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, src, re.I | re.S):
                scores[label] += 0.28

    if "?" in src:
        scores["question"] += 0.15
    if any(token in lower for token in ("help", "помоги")):
        scores["task_request"] += 0.14
    if any(token in lower for token in ("ui", "css", "frontend", "пиксель", "layout")):
        scores["ui_request"] += 0.18
    if any(token in lower for token in ("git", "branch", "commit", "merge", "pytest", "traceback")):
        scores["coding_help"] += 0.2
    if "```" in src:
        scores["coding_help"] += 0.24
    if lang in {"mixed", "unknown"} and len(src) > 40:
        scores["question"] += 0.04

    # Context bias.
    mode = str(tags.get("mode") or "").lower()
    if mode == "coding":
        scores["coding_help"] += 0.16
    if mode == "task":
        scores["task_request"] += 0.12
    if str(tags.get("last_intent") or "") == "complaint":
        scores["complaint"] += 0.08

    ordered = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    best_label, best_score = ordered[0]
    second_score = ordered[1][1] if len(ordered) > 1 else 0.0
    conf = best_score / max(0.001, best_score + second_score)
    conf = max(0.0, min(1.0, conf))

    alt = [name for name, _score in ordered[1:4] if _score >= 0.18]
    return IntentResult(label=best_label, conf=conf, alt_labels=alt)


def classify_intent(text: str) -> str:
    # Backward-compat helper used by existing call sites.
    result = classify(text=text, lang="unknown", context_tags=None)
    return result.label

