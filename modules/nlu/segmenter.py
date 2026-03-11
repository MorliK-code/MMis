from __future__ import annotations

import re

from .normalizer import normalize_text, tokenize_text


class SemanticSegmenter:
    """Heuristic semantic segmenter for short conversational user messages."""

    _STRONG_SPLIT_RE = re.compile(r"[!?;]+|(?:\.(?=\s|$))")
    _MULTI_COMMA_RE = re.compile(r"\s*,\s*")
    _WORD_RE = re.compile(r"[a-zа-я0-9]+(?:[._+#-][a-zа-я0-9]+)*", re.IGNORECASE)
    _BOUNDARY_STRIP_RE = re.compile(r"^[^\wа-яa-z0-9]+|[^\wа-яa-z0-9]+$")
    _LEADING_PARTICLE_RE = re.compile(r"^(?:а|ну)\s+")
    _INTRO_PREFIX_RE = re.compile(
        r"^(?:(?:кстати|слушай|короче|вообще|между прочим|значит|типа|в общем)\s*,?\s*)+"
    )
    _VERB_HINT_RE = re.compile(
        r"(ться|ить|ать|ять|еть|уть|ти|ть|"
        r"юсь|ешь|ишь|ем|им|ете|ите|"
        r"ют|ут|ат|ят|"
        r"ал|ала|али|ил|ила|или|"
        r"ит|ится|ился|илась|ились|"
        r"ю|у)$"
    )

    _CONJUNCTIONS = {"и", "а", "но"}
    _QUESTION_WORDS = {
        "что",
        "кто",
        "где",
        "когда",
        "как",
        "какой",
        "какая",
        "какие",
        "какую",
        "зачем",
        "почему",
        "сколько",
        "чем",
        "по",
    }
    _SUBJECT_TOKENS = {"я", "мы", "ты", "вы", "он", "она", "они", "мне", "нам", "тебе"}
    _KNOWN_VERBS = {
        "люблю",
        "живу",
        "хочу",
        "могу",
        "буду",
        "нравится",
        "устал",
        "поработал",
        "шумит",
        "кодить",
        "программировать",
    }

    def split(self, text: str) -> list[str]:
        normalized = normalize_text(text)
        if not normalized:
            return []

        rough_chunks = self._rough_split_by_punctuation(normalized)

        intro_cleaned: list[str] = []
        for chunk in rough_chunks:
            cleaned = self._cleanup_intro_words(chunk)
            if cleaned:
                intro_cleaned.append(cleaned)

        clause_chunks: list[str] = []
        for chunk in intro_cleaned:
            clause_chunks.extend(self._split_compound_clauses(chunk))

        return self._final_cleanup(clause_chunks)

    def _rough_split_by_punctuation(self, text: str) -> list[str]:
        parts = self._STRONG_SPLIT_RE.split(text)
        return [part.strip(" ,:-") for part in parts if part and part.strip(" ,:-")]

    def _cleanup_intro_words(self, text: str) -> str:
        cleaned = self._INTRO_PREFIX_RE.sub("", text).strip(" ,:-")
        if self._LEADING_PARTICLE_RE.match(cleaned):
            tokens = tokenize_text(cleaned)
            if len(tokens) > 1 and tokens[1] in self._QUESTION_WORDS:
                cleaned = self._LEADING_PARTICLE_RE.sub("", cleaned, count=1).strip()
        return cleaned

    def _split_compound_clauses(self, text: str) -> list[str]:
        comma_parts = self._split_by_comma_rules(text)
        output: list[str] = []
        for part in comma_parts:
            output.extend(self._split_by_conjunction_rules(part))
        return output

    def _split_by_comma_rules(self, text: str) -> list[str]:
        chunks = [part.strip() for part in self._MULTI_COMMA_RE.split(text) if part.strip()]
        if len(chunks) <= 1:
            return chunks

        result: list[str] = []
        current = chunks[0]

        for nxt in chunks[1:]:
            if self._should_split(current, nxt, trigger="comma"):
                result.append(current)
                current = nxt
            else:
                current = f"{current} {nxt}".strip()

        result.append(current)
        return result

    def _split_by_conjunction_rules(self, text: str) -> list[str]:
        tokens = tokenize_text(text)
        if not tokens:
            return []
        if len(tokens) == 1:
            return tokens

        segments: list[str] = []
        current_tokens: list[str] = []

        for index, token in enumerate(tokens):
            if token in self._CONJUNCTIONS and current_tokens and index + 1 < len(tokens):
                left_text = " ".join(current_tokens)
                right_text = " ".join(tokens[index + 1 :])
                if self._should_split(left_text, right_text, trigger=token):
                    segments.append(left_text)
                    current_tokens = []
                    continue

            current_tokens.append(token)

        if current_tokens:
            segments.append(" ".join(current_tokens))

        return segments

    def _should_split(self, left: str, right: str, trigger: str) -> bool:
        left_tokens = tokenize_text(left)
        right_tokens = tokenize_text(right)
        if not left_tokens or not right_tokens:
            return False

        if trigger == "comma":
            right_first = right_tokens[0]
            if right_first in self._QUESTION_WORDS:
                return True
            if self._looks_like_clause(left_tokens) and self._looks_like_clause(right_tokens):
                return True
            return False

        if trigger in {"но", "а"}:
            return self._looks_like_clause(left_tokens) and self._looks_like_clause(right_tokens)

        if trigger == "и":
            if not (self._looks_like_clause(left_tokens) and self._looks_like_clause(right_tokens)):
                return False
            if self._looks_like_enumeration(left_tokens, right_tokens):
                return False
            return True

        return False

    def _looks_like_clause(self, tokens: list[str]) -> bool:
        if not tokens:
            return False
        if any(token in self._QUESTION_WORDS for token in tokens):
            return True
        if self._contains_verb(tokens):
            return True
        if len(tokens) >= 4:
            return True
        if len(tokens) >= 3 and tokens[0] in self._SUBJECT_TOKENS:
            return True
        return False

    def _contains_verb(self, tokens: list[str]) -> bool:
        for token in tokens:
            if token in self._KNOWN_VERBS:
                return True
            if len(token) >= 3 and self._VERB_HINT_RE.search(token):
                return True
        return False

    def _looks_like_enumeration(self, left_tokens: list[str], right_tokens: list[str]) -> bool:
        if len(right_tokens) == 1 and not self._contains_verb(right_tokens):
            return True
        if len(left_tokens) <= 2 and len(right_tokens) <= 2:
            if not self._contains_verb(left_tokens) and not self._contains_verb(right_tokens):
                return True
        return False

    def _final_cleanup(self, chunks: list[str]) -> list[str]:
        result: list[str] = []

        for chunk in chunks:
            cleaned = normalize_text(chunk)
            cleaned = self._cleanup_intro_words(cleaned)
            cleaned = re.sub(r"^(?:и|а|но)\s+", "", cleaned)
            cleaned = self._BOUNDARY_STRIP_RE.sub("", cleaned).strip()
            cleaned = " ".join(self._WORD_RE.findall(cleaned))
            if cleaned:
                result.append(cleaned)

        return result


if __name__ == "__main__":
    segmenter = SemanticSegmenter()

    examples = [
        "я живу в Украине и люблю программировать",
        "кстати, что там по погодке?",
        "я сегодня устал, но всё равно хочу кодить",
        "слушай, а какие сейчас embedding модели норм?",
        "я из Киева и мне нравится python, что по погоде?",
    ]

    for index, example in enumerate(examples, start=1):
        print(f"{index}. {example}")
        print(segmenter.split(example))
        print("-" * 56)
