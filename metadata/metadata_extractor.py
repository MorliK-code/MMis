from __future__ import annotations

import hashlib
import re
import time
from collections import OrderedDict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from metadata.emotion_detector import EmotionResult, detect as detect_emotion
from metadata.intent_classifier import IntentResult, classify as classify_intent
from metadata.language_detector import LanguageResult, analyze as analyze_language
from metadata.tagger import Tagger
from prompt_engine.prompt_registry import PromptRegistry
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
_PERSON_RE = re.compile(r"\b[А-ЯЁ][а-яё]{2,}\b|\b[A-Z][a-z]{2,}\b")
_WINDOWS_PATH_RE = re.compile(r"[A-Za-z]:\\")

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
    entities: dict[str, Any] = field(default_factory=dict)
    safety_flags: dict[str, Any] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class MetadataExtractor:
    def __init__(
        self,
        *,
        cache_size: int = 200,
        cache_ttl_s: int = 24 * 3600,
        cache_dir: str | Path | None = None,
        use_disk_cache: bool = True,
        prompt_registry: PromptRegistry | None = None,
    ):
        self.cache_size = max(50, int(cache_size))
        self._cache: OrderedDict[str, Metadata] = OrderedDict()
        self._tagger = Tagger()
        self._prompt_registry = prompt_registry or PromptRegistry()
        self._prompt_text_cache: dict[str, str] = {}
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
        context_tags = _context_tags(state_map=state_map, last_messages=last_messages)
        intent_prompt = self._prompt_text("metadata.intent")
        emotion_prompt = self._prompt_text("metadata.emotion")
        tagging_prompt = self._prompt_text("metadata.tagging")
        allowed_intents = _extract_bullet_values(intent_prompt)
        allowed_emotions = _extract_bullet_values(emotion_prompt)
        allowed_tags = _extract_bullet_values(tagging_prompt)

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
            intent = classify_intent(src, lang=lang_result.lang, context_tags=context_tags)
            emotion = detect_emotion(src, lang=lang_result.lang)
            tags = self._tagger.build(
                text=src,
                lang=lang_result,
                intent=intent,
                emotion=emotion,
                extra_flags=self._extra_flags(src, lang_result, context_tags),
            )
            entities = self._entities(src, quality=(profile == PROFILE_QUALITY))
            safety_flags = self._safety_flags(src, lightweight=False)

        if allowed_intents and intent.label not in allowed_intents:
            intent = IntentResult(label="chat", conf=min(0.49, float(intent.conf)), alt_labels=list(intent.alt_labels))
        if allowed_emotions and emotion.label not in allowed_emotions:
            emotion = EmotionResult(label="neutral", scores=dict(emotion.scores), intensity=float(emotion.intensity), arousal=float(emotion.arousal))
        if allowed_tags:
            tags = [
                x
                for x in list(tags or [])
                if (
                    str(x).lower() in allowed_tags
                    or str(x).startswith("lang_")
                    or str(x).startswith("intent_")
                    or str(x).startswith("emotion_")
                )
            ]

        meta_payload = self._meta(src, state_map, lang_result, profile=profile)
        meta_payload.setdefault("prompt_keys", ["metadata.intent", "metadata.emotion", "metadata.tagging"])
        metadata = Metadata(
            lang=lang_result.lang,
            lang_conf=float(lang_result.conf),
            intent=IntentMeta(label=intent.label, conf=float(intent.conf), alt_labels=list(intent.alt_labels)),
            emotion=EmotionMeta(
                label=emotion.label,
                intensity=float(emotion.intensity),
                arousal=float(emotion.arousal),
                scores=dict(emotion.scores),
            ),
            tags=tags,
            entities=entities,
            safety_flags=safety_flags,
            meta=meta_payload,
        )
        self._cache_set(cache_key, metadata, write_disk=True)
        return metadata

    def _prompt_text(self, key: str) -> str:
        name = str(key or "").strip()
        if not name:
            return ""
        if name in self._prompt_text_cache:
            return self._prompt_text_cache[name]
        try:
            text = str(self._prompt_registry.get_text(name) or "")
        except Exception:
            text = ""
        self._prompt_text_cache[name] = text
        return text

    def _cache_key(self, text: str, *, profile: str, state: dict[str, Any]) -> str:
        normalized = str(text or "").strip().lower()
        mode = str(state.get("mode") or "")
        source = str(state.get("source") or state.get("input_source") or "")
        payload = f"{profile}|{mode}|{source}|{normalized}"
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
        people = _dedupe(_PERSON_RE.findall(text))[:12]
        dates = _dedupe(_DATE_RE.findall(text))[:12]
        numbers = _dedupe(_NUM_RE.findall(text))[:20]
        links = _dedupe(_URL_RE.findall(text))[:8]
        out = {
            "people": people if quality else people[:5],
            "places": [],  # placeholder for future NER model
            "dates": dates,
            "numbers": numbers,
            "links": links,
        }
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
            "elapsed_ms": 0.0,  # filled by wrapper if needed
        }

    @staticmethod
    def _extra_flags(text: str, lang: LanguageResult, context_tags: dict[str, Any]) -> dict[str, Any]:
        return {
            "is_code_like": bool(lang.is_code_like or _CODE_RE.search(text)),
            "has_url": bool(lang.has_url or _URL_RE.search(text)),
            "has_emoji": bool(lang.has_emoji),
            "has_path_windows": bool(_WINDOWS_PATH_RE.search(text)),
            "context_tags": dict(context_tags),
        }


_DEFAULT_EXTRACTOR = MetadataExtractor(cache_size=240)


def extract(text: str, state, last_messages=None) -> Metadata:
    t0 = time.perf_counter()
    item = _DEFAULT_EXTRACTOR.extract(text=text, state=state, last_messages=last_messages)
    # update elapsed in a new object (dataclass is frozen)
    elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
    meta = dict(item.meta)
    meta["elapsed_ms"] = elapsed_ms
    return Metadata(
        lang=item.lang,
        lang_conf=item.lang_conf,
        intent=item.intent,
        emotion=item.emotion,
        tags=list(item.tags),
        entities=dict(item.entities),
        safety_flags=dict(item.safety_flags),
        meta=meta,
    )


def extract_message_metadata(text: str, state=None, last_messages=None) -> dict[str, Any]:
    # Backward-compatible wrapper for callers expecting dict.
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


def _extract_bullet_values(text: str) -> set[str]:
    src = str(text or "")
    out: set[str] = set()
    for line in src.splitlines():
        raw = str(line or "").strip()
        if not raw.startswith("-"):
            continue
        value = raw[1:].strip().lower()
        if not value:
            continue
        # Keep first token-like part before additional descriptions.
        value = value.split(":", 1)[0].strip()
        value = value.split(" ", 1)[0].strip()
        value = value.replace("|", "").replace(",", "").strip()
        if not value:
            continue
        out.add(value)
        norm = re.sub(r"[^a-z0-9_]+", "_", value.lower()).strip("_")
        if norm:
            out.add(norm)
            out.add(norm.replace("__", "_"))
    return out


def _metadata_from_dict(payload: dict[str, Any]) -> Metadata | None:
    if not isinstance(payload, dict):
        return None
    try:
        intent_row = dict(payload.get("intent") or {})
        emotion_row = dict(payload.get("emotion") or {})
        return Metadata(
            lang=str(payload.get("lang") or ""),
            lang_conf=float(payload.get("lang_conf") or 0.0),
            intent=IntentMeta(
                label=str(intent_row.get("label") or "chat"),
                conf=float(intent_row.get("conf") or 0.0),
                alt_labels=[str(x) for x in list(intent_row.get("alt_labels") or []) if str(x).strip()],
            ),
            emotion=EmotionMeta(
                label=str(emotion_row.get("label") or "neutral"),
                intensity=float(emotion_row.get("intensity") or 0.0),
                arousal=float(emotion_row.get("arousal") or 0.0),
                scores={str(k): float(v) for k, v in dict(emotion_row.get("scores") or {}).items()},
            ),
            tags=[str(x) for x in list(payload.get("tags") or []) if str(x).strip()],
            entities=dict(payload.get("entities") or {}),
            safety_flags=dict(payload.get("safety_flags") or {}),
            meta=dict(payload.get("meta") or {}),
        )
    except Exception:
        return None
