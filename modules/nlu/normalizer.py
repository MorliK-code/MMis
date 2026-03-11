from __future__ import annotations

import re
import unicodedata


_WS_RE = re.compile(r"\s+")
_REPEAT_PUNCT_RE = re.compile(r"([!?.,:;])\1+")
_SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([!?.,:;])")
_TOKEN_RE = re.compile(r"[a-zа-я0-9]+(?:[._+#-][a-zа-я0-9]+)*", re.IGNORECASE)


def _clean_noise_chars(text: str) -> str:
    """Remove control/format noise while preserving semantic characters."""
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


def normalize_text(text: str) -> str:
    """Normalize user text for deterministic NLU routing."""
    if not text:
        return ""

    normalized = text.lower().strip()
    normalized = normalized.replace("ё", "е")
    normalized = _clean_noise_chars(normalized)
    normalized = _REPEAT_PUNCT_RE.sub(r"\1", normalized)
    normalized = _SPACE_BEFORE_PUNCT_RE.sub(r"\1", normalized)
    normalized = _WS_RE.sub(" ", normalized)
    return normalized.strip()


def tokenize_text(text: str) -> list[str]:
    """Tokenize normalized text into alphanumeric routing tokens."""
    normalized = normalize_text(text)
    if not normalized:
        return []
    return _TOKEN_RE.findall(normalized)


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
        return [normalize_text(token) for token in tokens], []

    normalized_alias_map: dict[str, str] = {}
    for alias, canonical in alias_map.items():
        alias_key = normalize_text(alias)
        canonical_value = normalize_text(canonical)
        if alias_key and canonical_value:
            normalized_alias_map[alias_key] = canonical_value

    result_tokens: list[str] = []
    replacements: list[dict[str, str]] = []

    for token in tokens:
        normalized_token = normalize_text(token)
        canonical = normalized_alias_map.get(normalized_token)
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
    tokens = _TOKEN_RE.findall(normalized) if normalized else []
    canonical_tokens, replacements = canonicalize_tokens(tokens, alias_map)
    return " ".join(canonical_tokens), replacements


if __name__ == "__main__":
    demo_alias_map = {
        "че": "что",
        "шо": "что",
        "погодке": "weather",
        "погодка": "weather",
        "видюха": "gpu",
        "долларчику": "usd",
    }

    examples = [
        "че там по погодке?",
        "видюха шумит",
        "шо по долларчику",
    ]

    for sample in examples:
        normalized_sample = normalize_text(sample)
        canonical_text, used_aliases = canonicalize_text(sample, demo_alias_map)
        print(f"Input:      {sample}")
        print(f"Normalized: {normalized_sample}")
        print(f"Canonical:  {canonical_text}")
        print(f"Aliases:    {used_aliases}")
        print("-" * 48)
