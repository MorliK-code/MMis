from __future__ import annotations

import re
import unicodedata


_WS_RE = re.compile(r"\s+")
_REPEAT_PUNCT_RE = re.compile(r"([!?.,:;])\1+")
_SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([!?.,:;])")
_NOISE_SYMBOL_RE = re.compile(r"[~*_=]{2,}")
_TOKEN_RE = re.compile(r"[a-z\u0430-\u044f0-9]+(?:[._+#-][a-z\u0430-\u044f0-9]+)*", re.IGNORECASE)
_EDGE_NOISE_RE = re.compile(r"^[^a-z\u0430-\u044f0-9]+|[^a-z\u0430-\u044f0-9]+$", re.IGNORECASE)
_LETTER_RUN_RE = re.compile(r"([a-z\u0430-\u044f])\1{2,}", re.IGNORECASE)
_LETTER_RUN_SINGLE_RE = re.compile(r"([a-z\u0430-\u044f])\1+", re.IGNORECASE)

_CONVERSATIONAL_RULES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\b(?:\u0447\u0435|\u0447\u0451|\u0447\u043e|\u0448\u043e)\b"), "\u0447\u0442\u043e"),
    (re.compile(r"\b(?:\u0449\u0430|\u0449\u0430\u0441)\b"), "\u0441\u0435\u0439\u0447\u0430\u0441"),
    (re.compile(r"\b(?:\u0432\u0430\u0449\u0435)\b"), "\u0432\u043e\u043e\u0431\u0449\u0435"),
    (re.compile(r"\b(?:\u043a\u0430\u0434\u0430)\b"), "\u043a\u043e\u0433\u0434\u0430"),
    (re.compile(r"\b(?:\u0441\u043a\u043e\u043a)\b"), "\u0441\u043a\u043e\u043b\u044c\u043a\u043e"),
)

_TOKEN_SUFFIXES = (
    "\u043e\u0447\u043a\u0430",
    "\u0435\u0447\u043a\u0430",
    "\u0435\u043d\u044c\u043a\u0430",
    "\u0443\u0448\u043a\u0430",
    "\u044e\u0448\u043a\u0430",
    "\u0447\u0438\u043a",
    "\u0449\u0438\u043a",
    "\u0430\u043c\u0438",
    "\u044f\u043c\u0438",
    "\u043e\u0433\u043e",
    "\u0435\u043c\u0443",
    "\u044b\u043c\u0438",
    "\u0438\u043c\u0438",
    "\u044f\u0445",
    "\u0430\u0445",
    "\u044b\u0435",
    "\u043e\u0435",
    "\u0430\u044f",
    "\u043e\u0439",
    "\u043e\u043c",
    "\u0430\u043c",
    "\u044f\u043c",
    "\u0438\u0435",
    "\u0438\u0439",
    "\u044b\u0439",
    "\u043e\u0433\u043e",
    "\u0443\u044e",
    "\u044e\u044e",
    "\u043a\u0430",
    "\u043a\u0443",
    "\u043a\u0435",
    "\u043a\u0438",
    "\u0442\u044c\u0441\u044f",
    "\u0442\u044c",
    "\u0442\u0438",
    "ing",
    "ed",
    "es",
    "s",
)


def _clean_noise_chars(text: str) -> str:
    cleaned: list[str] = []
    append = cleaned.append

    for char in text:
        category = unicodedata.category(char)
        if category.startswith("C"):
            if char in "\t\n\r":
                append(" ")
            continue
        append(char)

    return "".join(cleaned)


def _collapse_letter_runs(text: str) -> str:
    return _LETTER_RUN_RE.sub(r"\1\1", text)


def _collapse_letter_runs_single(text: str) -> str:
    return _LETTER_RUN_SINGLE_RE.sub(r"\1", text)


def _normalize_conversational_forms(text: str) -> str:
    out = text
    for pattern, replacement in _CONVERSATIONAL_RULES:
        out = pattern.sub(replacement, out)
    return out


def _normalize_token(token: str) -> str:
    value = str(token or "").strip().lower()
    if not value:
        return ""
    value = value.replace("\u0451", "\u0435")
    value = _clean_noise_chars(value)
    value = _collapse_letter_runs(value)
    value = _normalize_conversational_forms(value)
    value = _EDGE_NOISE_RE.sub("", value)
    return value


def _token_root(token: str) -> str:
    base = _normalize_token(token)
    if len(base) < 4:
        return base
    for suffix in _TOKEN_SUFFIXES:
        if base.endswith(suffix) and len(base) - len(suffix) >= 3:
            return base[: -len(suffix)]
    return base


def normalize_text(text: str) -> str:
    """Normalize user text for deterministic NLU routing."""
    if not text:
        return ""

    normalized = str(text or "").lower().strip()
    normalized = normalized.replace("\u0451", "\u0435")
    normalized = _clean_noise_chars(normalized)
    normalized = _NOISE_SYMBOL_RE.sub(" ", normalized)
    normalized = _REPEAT_PUNCT_RE.sub(r"\1", normalized)
    normalized = _SPACE_BEFORE_PUNCT_RE.sub(r"\1", normalized)
    normalized = _collapse_letter_runs(normalized)
    normalized = _normalize_conversational_forms(normalized)
    normalized = _WS_RE.sub(" ", normalized)
    return normalized.strip()


def tokenize_text(text: str) -> list[str]:
    """Tokenize normalized text into alphanumeric routing tokens."""
    normalized = normalize_text(text)
    if not normalized:
        return []

    tokens: list[str] = []
    for raw_token in _TOKEN_RE.findall(normalized):
        token = _normalize_token(raw_token)
        if token:
            tokens.append(token)
    return tokens


def token_variants(token: str) -> set[str]:
    """
    Produce a compact set of token forms for fuzzy and subword routing.

    Variants include:
        - normalized surface form
        - de-stretched form with repeated letters collapsed to one
        - a lightweight stem/root form
    """
    base = _normalize_token(token)
    if not base:
        return set()

    variants = {base}
    collapsed = _collapse_letter_runs_single(base)
    if len(collapsed) >= 3:
        variants.add(collapsed)

    root = _token_root(base)
    if len(root) >= 3:
        variants.add(root)

    root_collapsed = _token_root(collapsed)
    if len(root_collapsed) >= 3:
        variants.add(root_collapsed)

    return {variant for variant in variants if variant}


def canonicalize_tokens(
    tokens: list[str],
    alias_map: dict[str, str],
) -> tuple[list[str], list[dict[str, str]]]:
    """
    Replace known user aliases with canonical routing tokens.

    Returns:
        - canonicalized token list
        - replacement records [{"alias": "...", "canonical": "..."}]
    """
    if not tokens:
        return [], []

    if not alias_map:
        return [_normalize_token(token) for token in tokens], []

    normalized_alias_map: dict[str, str] = {}
    for alias, canonical in alias_map.items():
        alias_key = _normalize_token(alias)
        canonical_value = _normalize_token(canonical)
        if alias_key and canonical_value:
            normalized_alias_map[alias_key] = canonical_value

    result_tokens: list[str] = []
    replacements: list[dict[str, str]] = []

    for token in tokens:
        normalized_token = _normalize_token(token)
        matched_alias = normalized_token if normalized_token in normalized_alias_map else ""
        if not matched_alias:
            for variant in token_variants(normalized_token):
                if variant in normalized_alias_map:
                    matched_alias = variant
                    break

        canonical = normalized_alias_map.get(matched_alias)
        if canonical is None:
            result_tokens.append(normalized_token)
            continue

        result_tokens.append(canonical)
        replacements.append({"alias": normalized_token, "canonical": canonical})

    return result_tokens, replacements


def canonicalize_text(
    text: str,
    alias_map: dict[str, str],
) -> tuple[str, list[dict[str, str]]]:
    """Normalize, tokenize, canonicalize, and return canonical text + alias usage."""
    normalized = normalize_text(text)
    tokens = tokenize_text(normalized) if normalized else []
    canonical_tokens, replacements = canonicalize_tokens(tokens, alias_map)
    return " ".join(canonical_tokens), replacements


if __name__ == "__main__":
    demo_alias_map = {
        "\u0447\u0435": "\u0447\u0442\u043e",
        "\u0448\u043e": "\u0447\u0442\u043e",
        "\u043f\u043e\u0433\u043e\u0434\u043a\u0430": "weather",
        "\u043f\u043e\u0433\u043e\u0434\u043a\u0435": "weather",
        "\u0432\u0438\u0434\u044e\u0445\u0430": "gpu",
        "\u0434\u043e\u043b\u043b\u0430\u0440\u0447\u0438\u043a\u0443": "usd",
    }

    examples = [
        "\u0447\u0435 \u0442\u0430\u043c \u043f\u043e \u043f\u043e\u0433\u043e\u0434\u043a\u0435???",
        "\u0448\u043e \u043f\u043e \u0443\u043b\u0438\u0446\u0435",
        "\u0436\u0430\u0430\u0430\u0440\u043a\u043e\u043e\u043e",
    ]

    for sample in examples:
        normalized_sample = normalize_text(sample)
        canonical_text, used_aliases = canonicalize_text(sample, demo_alias_map)
        print(f"Input:      {sample}")
        print(f"Normalized: {normalized_sample}")
        print(f"Canonical:  {canonical_text}")
        print(f"Aliases:    {used_aliases}")
        print("-" * 48)
