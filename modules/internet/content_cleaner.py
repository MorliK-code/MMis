from __future__ import annotations

import html
import re
from dataclasses import dataclass

try:  # Optional dependency; graceful fallback is kept for local/dev envs.
    import trafilatura
except Exception:  # pragma: no cover
    trafilatura = None

try:  # Optional dependency; graceful fallback is kept for local/dev envs.
    from bs4 import BeautifulSoup
except Exception:  # pragma: no cover
    BeautifulSoup = None


_REMOVE_TAGS = {"script", "style", "noscript", "svg", "canvas", "iframe"}
_NOISE_HINTS = {
    "cookie",
    "consent",
    "banner",
    "popup",
    "advert",
    "sponsor",
    "promo",
    "comment",
    "reply",
    "share",
    "subscribe",
    "newsletter",
    "disqus",
    "reply-list",
}
_NOISE_LINE_RX = [
    re.compile(r"\b(we use cookies|cookie policy|accept all cookies)\b", flags=re.I),
    re.compile(r"\b(consent preferences|manage cookies)\b", flags=re.I),
    re.compile(r"\b(subscribe|newsletter|turn on notifications)\b", flags=re.I),
    re.compile(r"\b(leave a comment|sign in to comment)\b", flags=re.I),
    re.compile(r"\b(sponsored|promoted)\b", flags=re.I),
    re.compile(r"(файлы cookie|согласие на обработку|принять все cookies)", flags=re.I),
    re.compile(r"(оставьте комментарий|подпишитесь|включите уведомления)", flags=re.I),
    re.compile(r"(спонсорский материал|промо)", flags=re.I),
]
_AD_TOKEN_RX = re.compile(r"(^|[\s_\-])(ad|ads|adblock|advert|adslot)($|[\s_\-])", flags=re.I)


@dataclass(frozen=True)
class CleanResult:
    text: str
    title: str
    method: str
    removed_blocks: int
    raw_len: int
    clean_len: int


def clean_web_content(
    html_text: str,
    *,
    base_url: str = "",
    max_chars: int = 4000,
    min_chars: int = 200,
    language_hint: str = "",
) -> CleanResult:
    raw = str(html_text or "")
    raw_len = len(raw)
    cap = max(256, int(max_chars))
    min_size = max(40, int(min_chars))

    title = _extract_title(raw)
    removed_blocks = 0
    method = "regex"
    text = ""

    trafilatura_text = _extract_with_trafilatura(raw, base_url=base_url, language_hint=language_hint)
    if trafilatura_text and len(trafilatura_text) >= min_size:
        text = trafilatura_text
        method = "trafilatura"

    bs4_text, bs4_removed, bs4_title = _extract_with_bs4(raw)
    removed_blocks += int(bs4_removed)
    if not title and bs4_title:
        title = bs4_title
    if not text or len(text) < min_size:
        if bs4_text:
            text = bs4_text
            method = "bs4"

    if not text:
        text = _extract_with_regex(raw)
        method = "regex"

    before_line_count = _line_count(text)
    text = _post_filter_lines(text)
    removed_blocks += max(0, before_line_count - _line_count(text))
    if len(text) < min_size:
        fallback_raw = _extract_with_regex(raw)
        fallback_before = _line_count(fallback_raw)
        fallback = _post_filter_lines(fallback_raw)
        removed_blocks += max(0, fallback_before - _line_count(fallback))
        if len(fallback) > len(text):
            text = fallback
            method = f"{method}+regex_fallback"

    text = _clip(text, cap)
    clean_len = len(text)
    return CleanResult(
        text=text,
        title=title,
        method=method,
        removed_blocks=max(0, int(removed_blocks)),
        raw_len=int(raw_len),
        clean_len=int(clean_len),
    )


def _extract_with_trafilatura(raw: str, *, base_url: str, language_hint: str) -> str:
    if trafilatura is None or not raw.strip():
        return ""
    try:
        return str(
            trafilatura.extract(
                raw,
                include_comments=False,
                include_tables=False,
                favor_precision=True,
                output_format="txt",
                url=str(base_url or ""),
                target_language=str(language_hint or "").strip() or None,
            )
            or ""
        ).strip()
    except Exception:
        return ""


def _extract_with_bs4(raw: str) -> tuple[str, int, str]:
    if BeautifulSoup is None or not raw.strip():
        return "", 0, ""
    try:
        soup = BeautifulSoup(raw, "lxml")
    except Exception:
        return "", 0, ""

    title = ""
    if getattr(soup, "title", None) and getattr(soup.title, "string", None):
        title = str(soup.title.string or "").strip()

    for tag_name in _REMOVE_TAGS:
        for tag in list(soup.find_all(tag_name)):
            tag.decompose()

    removed = 0
    for tag in list(soup.find_all(True)):
        if _is_noise_node(tag):
            tag.decompose()
            removed += 1

    main = soup.find("main") or soup.find("article") or soup.body or soup
    text = main.get_text(separator="\n", strip=True) if main is not None else soup.get_text(separator="\n", strip=True)
    return str(text or "").strip(), int(removed), title


def _is_noise_node(tag) -> bool:
    if tag is None:
        return False
    attrs = []
    for key in ("id", "class", "role", "aria-label", "aria-labelledby", "data-testid"):
        val = tag.get(key)
        if isinstance(val, list):
            attrs.extend(str(x or "") for x in val)
        elif val is not None:
            attrs.append(str(val))
    hay = " ".join(attrs).strip().lower()
    if not hay:
        return False
    if _AD_TOKEN_RX.search(hay):
        return True
    return any(hint in hay for hint in _NOISE_HINTS)


def _extract_with_regex(raw: str) -> str:
    text = re.sub(r"<!--.*?-->", " ", raw, flags=re.S)
    text = re.sub(r"<script[^>]*>.*?</script>", " ", text, flags=re.S | re.I)
    text = re.sub(r"<style[^>]*>.*?</style>", " ", text, flags=re.S | re.I)
    text = re.sub(r"<noscript[^>]*>.*?</noscript>", " ", text, flags=re.S | re.I)
    text = re.sub(r"<[^>]+>", "\n", text)
    text = html.unescape(text)
    return str(text or "").strip()


def _post_filter_lines(text: str) -> str:
    lines = [str(x or "").strip() for x in str(text or "").replace("\r", "\n").split("\n")]
    out: list[str] = []
    seen: set[str] = set()
    for row in lines:
        line = re.sub(r"\s+", " ", row).strip()
        if not line:
            continue
        low = line.lower()
        if any(rx.search(low) for rx in _NOISE_LINE_RX):
            continue
        if low in seen:
            continue
        seen.add(low)
        out.append(line)
    return "\n".join(out).strip()


def _extract_title(raw: str) -> str:
    m = re.search(r"<title[^>]*>(.*?)</title>", str(raw or ""), flags=re.S | re.I)
    if not m:
        return ""
    value = re.sub(r"<[^>]+>", " ", str(m.group(1) or ""))
    value = html.unescape(value)
    return re.sub(r"\s+", " ", value).strip()


def _clip(text: str, cap: int) -> str:
    src = str(text or "").strip()
    if len(src) <= cap:
        return src
    return src[: cap - 1].rstrip() + "..."


def _line_count(text: str) -> int:
    rows = [str(x or "").strip() for x in str(text or "").replace("\r", "\n").split("\n")]
    return sum(1 for row in rows if row)
