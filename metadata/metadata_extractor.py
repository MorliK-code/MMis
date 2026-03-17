from __future__ import annotations

import hashlib
import re
import time
from collections import OrderedDict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.spec_registry import load_spec
from metadata.entity_extractor import (
    extract_entities as extract_catalog_entities,
)
from metadata.emotion_detector import EmotionResult, detect as detect_emotion
from metadata.intent_classifier import IntentResult, classify as classify_intent
from metadata.language_detector import LanguageResult, analyze as analyze_language
from metadata.tagger import Tagger
from metadata.taxonomy import (
    STRUCTURAL_TAGS,
    normalize_emotion,
    normalize_intent,
    normalize_lang,
    normalize_topic_list,
)
from utils.cache import DiskTTLCache
from utils.logger import get_logger


PROFILE_FAST = "FAST"
PROFILE_BALANCED = "BALANCED"
PROFILE_QUALITY = "QUALITY"
_PROFILES = {PROFILE_FAST, PROFILE_BALANCED, PROFILE_QUALITY}
LOGGER = get_logger(__name__)

_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.I)
_CODE_RE = re.compile(r"```|`[^`]+`|Traceback|Exception|def\s+\w+\s*\(|class\s+\w+\s*:", re.I | re.S)
_DATE_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}[./-]\d{1,2}(?:[./-]\d{2,4})?\b")
_NUM_RE = re.compile(r"\b\d+(?:[.,]\d+)?\b")
_WINDOWS_PATH_RE = re.compile(r"[A-Za-z]:\\")
_PERSON_FULLNAME_RE = re.compile(r"\b([A-ZА-ЯЁ][a-zа-яё]{2,24})\s+([A-ZА-ЯЁ][a-zа-яё]{2,24})\b")
_PERSON_INTRO_RE = re.compile(
    r"(?:меня\s+зовут|мо[её]\s+имя|my\s+name\s+is|i\s+am|i'm)\s+([A-ZА-ЯЁ][a-zа-яё]{2,24})",
    re.I,
)
_PERSON_STOPWORDS = {
    "hello",
    "hi",
    "how",
    "what",
    "traceback",
    "exception",
    "internal",
    "server",
    "error",
    "docker",
    "python",
    "windows",
    "linux",
    "vscode",
    "ollama",
    "chromadb",
    "git",
    "json",
    "api",
    "code",
    "plan",
    "debug",
    "assistant",
    "привет",
    "здравствуй",
    "здравствуйте",
    "как",
    "что",
    "трейсбек",
    "ошибка",
    "сервер",
    "докер",
    "питон",
    "виндовс",
    "линукс",
    "код",
}

_TOXIC_RE = re.compile(r"\b(fuck|shit|идиот|тупой|дебил|сука|wtf)\b", re.I)
_SELF_HARM_RE = re.compile(r"\b(kill myself|самоубий|суицид)\b", re.I)


@dataclass(frozen=True)
class IntentMeta:
    label: str
    conf: float
    alt_labels: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class EmotionMeta:
    label: str
    intensity: float
    arousal: float
    scores: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class Metadata:
    lang: str
    lang_conf: float
    intent: IntentMeta
    emotion: EmotionMeta
    tags: list[str] = field(default_factory=list)
    runtime_entities: dict[str, Any] = field(default_factory=dict)
    safety_flags: dict[str, Any] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["runtime_entities"] = dict(payload.get("runtime_entities") or {})
        return payload


class MetadataExtractor:
    def __init__(
        self,
        *,
        cache_size: int = 200,
        cache_ttl_s: int = 24 * 3600,
        cache_dir: str | Path | None = None,
        use_disk_cache: bool = True,
        prompt_registry=None,
    ):
        self.cache_size = max(50, int(cache_size))
        self._cache: OrderedDict[str, Metadata] = OrderedDict()
        _ = prompt_registry  # legacy arg kept for compatibility
        self._metadata_spec = load_spec("metadata", required=False)
        self._tagger = Tagger(metadata_spec=self._metadata_spec)
        tagging = dict(self._metadata_spec.get("tagging") or {})
        self._allowed_tags = {
            str(x).strip().lower()
            for x in list(tagging.get("allowed_tags") or [])
            if str(x).strip()
        }
        self._tag_soft_filter = bool(tagging.get("soft_filter", True))
        self._allowed_intents = {
            normalize_intent(x)
            for x in list(self._metadata_spec.get("allowed_intents") or [])
            if str(x).strip()
        }
        self._allowed_emotions = {
            normalize_emotion(x)
            for x in list(self._metadata_spec.get("allowed_emotions") or [])
            if str(x).strip()
        }
        self._intent_rules = [dict(x) for x in list(self._metadata_spec.get("intent_rules") or []) if isinstance(x, dict)]
        self._emotion_rules = [dict(x) for x in list(self._metadata_spec.get("emotion_rules") or []) if isinstance(x, dict)]
        self._disk_cache = DiskTTLCache(
            namespace="metadata_extractor",
            root=cache_dir,
            default_ttl_s=max(60, int(cache_ttl_s)),
            max_memory_entries=max(64, min(2048, self.cache_size * 2)),
            enabled=bool(use_disk_cache),
        )
        try:
            self._disk_cache.purge_expired(max_files=800)
        except Exception as exc:
            LOGGER.debug("metadata cache purge skipped: %s", exc)

    def extract(self, text: str, state, last_messages=None) -> Metadata:
        src = str(text or "").strip()
        state_map = _as_dict(state)
        profile = _resolve_profile(state_map)
        cache_key = self._cache_key(src, profile=profile, state=state_map)
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached

        disk_hit = self._disk_cache.get(cache_key)
        if isinstance(disk_hit, dict):
            restored = _metadata_from_dict(disk_hit)
            if restored is not None:
                self._cache_set(cache_key, restored, write_disk=False)
                return restored

        lang_result = analyze_language(src)
        lang_value = normalize_lang(lang_result.lang)
        context_tags = _context_tags(state_map=state_map, last_messages=last_messages)

        if profile == PROFILE_FAST:
            intent = IntentResult(label="chat", conf=0.4, alt_labels=["question"])
            emotion = EmotionResult(label="neutral", scores={"neutral": 0.6}, intensity=0.3, arousal=0.2)
            tags = self._tagger.build(
                text=src,
                lang=lang_result,
                intent=intent,
                emotion=emotion,
                extra_flags=self._extra_flags(src, lang_result, context_tags),
            )
            entities = {}
            safety_flags = self._safety_flags(src, lightweight=True)
        else:
            intent_raw = classify_intent(src, lang=lang_value, context_tags=context_tags)
            emotion_raw = detect_emotion(src, lang=lang_value)
            intent = IntentResult(
                label=normalize_intent(intent_raw.label),
                conf=float(intent_raw.conf),
                alt_labels=[normalize_intent(x) for x in list(intent_raw.alt_labels or [])],
            )
            emotion = EmotionResult(
                label=normalize_emotion(emotion_raw.label),
                scores=dict(emotion_raw.scores),
                intensity=float(emotion_raw.intensity),
                arousal=float(emotion_raw.arousal),
            )
            tags = self._tagger.build(
                text=src,
                lang=lang_result,
                intent=intent,
                emotion=emotion,
                extra_flags=self._extra_flags(src, lang_result, context_tags),
            )
            entities = self._entities(src, quality=(profile == PROFILE_QUALITY))
            safety_flags = self._safety_flags(src, lightweight=False)

        intent = IntentResult(
            label=normalize_intent(intent.label),
            conf=float(intent.conf),
            alt_labels=[normalize_intent(x) for x in list(intent.alt_labels or [])],
        )
        emotion = EmotionResult(
            label=normalize_emotion(emotion.label),
            scores=dict(emotion.scores),
            intensity=float(emotion.intensity),
            arousal=float(emotion.arousal),
        )

        intent = self._apply_intent_rules(src=src, tags=tags, current=intent)
        emotion = self._apply_emotion_rules(src=src, tags=tags, current=emotion)

        if self._allowed_intents and intent.label not in self._allowed_intents:
            intent = IntentResult(label="chat", conf=min(0.49, float(intent.conf)), alt_labels=list(intent.alt_labels))
        if self._allowed_emotions and emotion.label not in self._allowed_emotions:
            emotion = EmotionResult(
                label="neutral",
                scores=dict(emotion.scores),
                intensity=float(emotion.intensity),
                arousal=float(emotion.arousal),
            )

        tags = _normalize_tags(tags)
        lang_from_tags = _lang_from_tags(tags)
        if lang_value == "unknown" and lang_from_tags != "unknown":
            lang_value = lang_from_tags
        if f"intent_{intent.label}" not in tags:
            tags.append(f"intent_{intent.label}")
        if f"emotion_{emotion.label}" not in tags:
            tags.append(f"emotion_{emotion.label}")
        if f"lang_{lang_value}" not in tags:
            tags.append(f"lang_{lang_value}")
        topics = normalize_topic_list([x.replace("topic_", "", 1) for x in tags if x.startswith("topic_")])
        for topic in topics:
            topic_tag = f"topic_{topic}"
            if topic_tag not in tags:
                tags.append(topic_tag)

        tags = _soft_filter_tags(
            tags=tags,
            allowed_tags=self._allowed_tags,
            enabled=self._tag_soft_filter,
        )
        tags = _canonicalize_lang_tags(tags=tags, lang=lang_value)

        meta_payload = self._meta(src, state_map, lang_result, profile=profile)
        meta_payload.setdefault("spec_keys", ["metadata_spec", "taxonomy"])
        meta_payload["topics"] = topics

        metadata = Metadata(
            lang=lang_value,
            lang_conf=float(lang_result.conf),
            intent=IntentMeta(label=intent.label, conf=float(intent.conf), alt_labels=list(intent.alt_labels)),
            emotion=EmotionMeta(
                label=emotion.label,
                intensity=float(emotion.intensity),
                arousal=float(emotion.arousal),
                scores=dict(emotion.scores),
            ),
            tags=tags,
            runtime_entities=entities,
            safety_flags=safety_flags,
            meta=meta_payload,
        )
        self._cache_set(cache_key, metadata, write_disk=True)
        return metadata

    def _cache_key(self, text: str, *, profile: str, state: dict[str, Any]) -> str:
        normalized = str(text or "").strip().lower()
        mode = str(state.get("mode") or "")
        source = str(state.get("source") or state.get("input_source") or "")
        payload = f"taxonomy_v4|{profile}|{mode}|{source}|{normalized}"
        return hashlib.sha1(payload.encode("utf-8", errors="ignore")).hexdigest()

    def _cache_get(self, key: str) -> Metadata | None:
        hit = self._cache.get(key)
        if hit is None:
            return None
        self._cache.move_to_end(key)
        return hit

    def _cache_set(self, key: str, value: Metadata, *, write_disk: bool = True) -> None:
        self._cache[key] = value
        self._cache.move_to_end(key)
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        if write_disk:
            try:
                self._disk_cache.set(key, value.to_dict())
            except Exception as exc:
                LOGGER.debug("metadata disk cache set failed: %s", exc)

    def _entities(self, text: str, *, quality: bool) -> dict[str, Any]:
        people = _extract_people_entities(text, quality=quality)
        dates = _dedupe(_DATE_RE.findall(text))[:12]
        numbers = _dedupe(_NUM_RE.findall(text))[:20]
        links = _dedupe(_URL_RE.findall(text))[:8]
        out = {
            "people": people if quality else people[:5],
            "places": [],
            "dates": dates,
            "numbers": numbers,
            "links": links,
        }
        catalog = extract_catalog_entities(text)
        for kind, values in dict(catalog or {}).items():
            key = str(kind or "").strip().lower()
            if not key:
                continue
            existing = [str(x).strip() for x in list(out.get(key) or []) if str(x).strip()]
            seen = {x.lower() for x in existing}
            for value in list(values or []):
                item = str(value or "").strip()
                if not item:
                    continue
                low = item.lower()
                if low in seen:
                    continue
                seen.add(low)
                existing.append(item)
            max_items = 12 if quality else 8
            out[key] = existing[:max_items]
        return out

    @staticmethod
    def _safety_flags(text: str, *, lightweight: bool) -> dict[str, Any]:
        lowered = str(text or "")
        toxicity = bool(_TOXIC_RE.search(lowered))
        self_harm = bool(_SELF_HARM_RE.search(lowered))
        flags = {
            "toxicity": toxicity,
            "self_harm": self_harm,
            "unsafe": bool(toxicity or self_harm),
        }
        if not lightweight:
            flags["caps_excess"] = sum(1 for ch in lowered if ch.isupper()) >= 8
            flags["has_nsfw_hint"] = bool(re.search(r"\b(sex|nsfw|эрот|интим)\b", lowered, re.I))
        return flags

    @staticmethod
    def _meta(text: str, state: dict[str, Any], lang: LanguageResult, *, profile: str) -> dict[str, Any]:
        src = str(text or "")
        now = datetime.now(timezone.utc).isoformat()
        source = str(state.get("source") or state.get("input_source") or "text")
        words = len([x for x in re.split(r"\s+", src.strip()) if x])
        return {
            "ts": now,
            "source": source,
            "chars": len(src),
            "words": words,
            "has_code": bool(lang.is_code_like or _CODE_RE.search(src)),
            "has_link": bool(lang.has_url or _URL_RE.search(src)),
            "has_emoji": bool(lang.has_emoji),
            "profile": profile,
            "elapsed_ms": 0.0,
        }

    @staticmethod
    def _extra_flags(text: str, lang: LanguageResult, context_tags: dict[str, Any]) -> dict[str, Any]:
        return {
            "is_code_like": bool(lang.is_code_like or _CODE_RE.search(text)),
            "has_url": bool(lang.has_url or _URL_RE.search(text)),
            "has_emoji": bool(lang.has_emoji),
            "has_windows_path": bool(_WINDOWS_PATH_RE.search(text)),
            "context_tags": dict(context_tags),
        }

    def _apply_intent_rules(self, *, src: str, tags: list[str], current: IntentResult) -> IntentResult:
        tag_set = {str(x).strip().lower() for x in list(tags or []) if str(x).strip()}
        lower = str(src or "").lower()
        for row in list(self._intent_rules or []):
            if not _rule_matches_text_and_tags(text=lower, tags=tag_set, rule=row):
                continue
            forced = normalize_intent(row.get("intent"))
            if forced:
                return IntentResult(label=forced, conf=max(0.8, float(current.conf)), alt_labels=list(current.alt_labels))
        return current

    def _apply_emotion_rules(self, *, src: str, tags: list[str], current: EmotionResult) -> EmotionResult:
        tag_set = {str(x).strip().lower() for x in list(tags or []) if str(x).strip()}
        lower = str(src or "").lower()
        for row in list(self._emotion_rules or []):
            if not _rule_matches_text_and_tags(text=lower, tags=tag_set, rule=row):
                continue
            forced = normalize_emotion(row.get("emotion"))
            if forced:
                scores = dict(current.scores)
                scores[forced] = max(float(scores.get(forced) or 0.0), 0.75)
                return EmotionResult(
                    label=forced,
                    scores=scores,
                    intensity=max(0.55, float(current.intensity)),
                    arousal=float(current.arousal),
                )
        return current


_DEFAULT_EXTRACTOR = MetadataExtractor(cache_size=240)


def extract(text: str, state, last_messages=None) -> Metadata:
    t0 = time.perf_counter()
    item = _DEFAULT_EXTRACTOR.extract(text=text, state=state, last_messages=last_messages)
    elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
    meta = dict(item.meta)
    meta["elapsed_ms"] = elapsed_ms
    return Metadata(
        lang=normalize_lang(item.lang),
        lang_conf=item.lang_conf,
        intent=IntentMeta(
            label=normalize_intent(item.intent.label),
            conf=float(item.intent.conf),
            alt_labels=[normalize_intent(x) for x in list(item.intent.alt_labels or [])],
        ),
        emotion=EmotionMeta(
            label=normalize_emotion(item.emotion.label),
            intensity=float(item.emotion.intensity),
            arousal=float(item.emotion.arousal),
            scores=dict(item.emotion.scores),
        ),
        tags=list(item.tags),
        runtime_entities=dict(item.runtime_entities),
        safety_flags=dict(item.safety_flags),
        meta=meta,
    )


def extract_message_metadata(text: str, state=None, last_messages=None) -> dict[str, Any]:
    return extract(text=text, state=state or {}, last_messages=last_messages).to_dict()


def _resolve_profile(state_map: dict[str, Any]) -> str:
    profile = str(
        state_map.get("quality_profile")
        or state_map.get("profile")
        or state_map.get("metadata_profile")
        or PROFILE_BALANCED
    ).strip().upper()
    return profile if profile in _PROFILES else PROFILE_BALANCED


def _context_tags(state_map: dict[str, Any], last_messages=None) -> dict[str, Any]:
    tags = dict(state_map.get("context_tags") or {})
    if last_messages:
        tags["history_len"] = len(list(last_messages))
    tags.setdefault("mode", str(state_map.get("mode") or "chat"))
    return tags


def _as_dict(value) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if value is None:
        return {}
    try:
        return dict(vars(value))
    except Exception:
        return {}


def _dedupe(values) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        row = str(value or "").strip()
        if not row:
            continue
        low = row.lower()
        if low in seen:
            continue
        seen.add(low)
        out.append(row)
    return out


def _extract_people_entities(text: str, *, quality: bool) -> list[str]:
    src = str(text or "")
    if not src.strip():
        return []

    candidates: list[str] = []

    for match in _PERSON_INTRO_RE.finditer(src):
        token = _normalize_person_token(match.group(1))
        if not token or _is_person_stopword(token):
            continue
        candidates.append(token)

    for match in _PERSON_FULLNAME_RE.finditer(src):
        first = _normalize_person_token(match.group(1))
        last = _normalize_person_token(match.group(2))
        if not first or not last:
            continue
        if _is_person_stopword(first) or _is_person_stopword(last):
            continue
        candidates.append(f"{first} {last}")

    cleaned: list[str] = []
    seen: set[str] = set()
    limit = 12 if quality else 5
    for raw in candidates:
        row = str(raw or "").strip()
        if not row:
            continue
        key = row.lower()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(row)
        if len(cleaned) >= limit:
            break
    return cleaned


def _normalize_person_token(value: str) -> str:
    token = str(value or "").strip().strip(".,!?;:()[]{}\"'`")
    if not token:
        return ""
    if not re.fullmatch(r"[A-ZА-ЯЁ][a-zа-яё'-]{1,24}", token):
        return ""
    return token


def _is_person_stopword(token: str) -> bool:
    return str(token or "").strip().lower() in _PERSON_STOPWORDS


def _soft_filter_tags(*, tags: list[str], allowed_tags: set[str], enabled: bool = True) -> list[str]:
    if not enabled or not allowed_tags:
        return _dedupe(tags)
    out: list[str] = []
    for item in list(tags or []):
        tag = str(item or "").strip().lower()
        if not tag:
            continue
        if (
            tag in allowed_tags
            or tag.startswith(("lang_", "intent_", "emotion_", "topic_", "mode_hint_", "is_", "tone_", "needs_"))
            or tag in STRUCTURAL_TAGS
        ):
            out.append(tag)
    return _dedupe(out)


def _rule_matches_text_and_tags(*, text: str, tags: set[str], rule: dict[str, Any]) -> bool:
    if not isinstance(rule, dict):
        return False
    contains = [str(x).strip().lower() for x in list(rule.get("if_contains") or []) if str(x).strip()]
    if contains and not any(token in text for token in contains):
        return False
    tags_any = {str(x).strip().lower() for x in list(rule.get("if_tags_any") or []) if str(x).strip()}
    if tags_any and not (tags_any & tags):
        return False
    pattern = str(rule.get("if_regex") or "").strip()
    if pattern:
        try:
            if not bool(re.search(pattern, text, flags=re.I | re.S)):
                return False
        except re.error:
            return False
    return bool(contains or tags_any or pattern)


def _normalize_tags(tags) -> list[str]:
    out: list[str] = []
    for raw in list(tags or []):
        tag = str(raw or "").strip().lower()
        if not tag:
            continue
        if tag == "contains_link":
            tag = "has_link"
        if tag.startswith("intent_"):
            suffix = normalize_intent(tag.replace("intent_", "", 1))
            tag = f"intent_{suffix}"
        elif tag.startswith("emotion_"):
            suffix = normalize_emotion(tag.replace("emotion_", "", 1))
            tag = f"emotion_{suffix}"
        out.append(tag)
    return _dedupe(out)


def _lang_from_tags(tags: list[str]) -> str:
    for item in list(tags or []):
        token = str(item or "").strip().lower()
        if not token.startswith("lang_"):
            continue
        value = normalize_lang(token.replace("lang_", "", 1))
        if value != "unknown":
            return value
    return "unknown"


def _canonicalize_lang_tags(*, tags: list[str], lang: str) -> list[str]:
    normalized_lang = normalize_lang(lang)
    out: list[str] = []
    seen: set[str] = set()
    for item in list(tags or []):
        token = str(item or "").strip().lower()
        if not token or token.startswith("lang_"):
            continue
        if token in seen:
            continue
        seen.add(token)
        out.append(token)
    if normalized_lang != "unknown":
        lang_tag = f"lang_{normalized_lang}"
        if lang_tag not in seen:
            out.append(lang_tag)
    elif "lang_unknown" not in seen:
        out.append("lang_unknown")
    return out


def _metadata_from_dict(payload: dict[str, Any]) -> Metadata | None:
    if not isinstance(payload, dict):
        return None
    try:
        intent_row = dict(payload.get("intent") or {})
        emotion_row = dict(payload.get("emotion") or {})
        tags_raw = [str(x).strip().lower() for x in list(payload.get("tags") or []) if str(x).strip()]
        tags = _normalize_tags(tags_raw)
        lang_value = normalize_lang(str(payload.get("lang") or ""))
        if lang_value == "unknown":
            lang_value = _lang_from_tags(tags)
        tags = _canonicalize_lang_tags(tags=tags, lang=lang_value)
        return Metadata(
            lang=lang_value,
            lang_conf=float(payload.get("lang_conf") or 0.0),
            intent=IntentMeta(
                label=normalize_intent(str(intent_row.get("label") or "chat")),
                conf=float(intent_row.get("conf") or 0.0),
                alt_labels=[normalize_intent(str(x)) for x in list(intent_row.get("alt_labels") or []) if str(x).strip()],
            ),
            emotion=EmotionMeta(
                label=normalize_emotion(str(emotion_row.get("label") or "neutral")),
                intensity=float(emotion_row.get("intensity") or 0.0),
                arousal=float(emotion_row.get("arousal") or 0.0),
                scores={str(k): float(v) for k, v in dict(emotion_row.get("scores") or {}).items()},
            ),
            tags=_dedupe(tags),
            runtime_entities=dict(payload.get("runtime_entities") or payload.get("entities") or {}),
            safety_flags=dict(payload.get("safety_flags") or {}),
            meta=dict(payload.get("meta") or {}),
        )
    except Exception:
        return None
