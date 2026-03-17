from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class IngestAnchor:
    kind: str
    surface: str
    normalized: str = ""
    confidence: float = 0.0
    hints: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": str(self.kind or "").strip(),
            "surface": str(self.surface or "").strip(),
            "normalized": str(self.normalized or "").strip(),
            "confidence": float(self.confidence or 0.0),
            "hints": dict(self.hints or {}),
        }


@dataclass(frozen=True)
class EntityItem:
    type: str
    surface: str
    canonical: str
    confidence: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class NumericFact:
    kind: str
    value: float | int | str
    unit: str = ""
    canonical_key: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class StableFact:
    subject: str
    predicate: str
    value: Any
    confidence: float
    relation: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class IngestEmotion:
    primary: str
    intensity: float
    arousal: float
    confidence: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class IngestAnalysis:
    raw_text: str
    normalized_text: str
    canonical_text: str
    search_text: str
    anchors: list[IngestAnchor] = field(default_factory=list)
    contextual_resolution: dict[str, Any] = field(default_factory=dict)
    entities: list[EntityItem] = field(default_factory=list)
    numeric_facts: list[NumericFact] = field(default_factory=list)
    stable_facts: list[StableFact] = field(default_factory=list)
    emotion: IngestEmotion | None = None
    tags: list[str] = field(default_factory=list)
    memory_views: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw_text": str(self.raw_text or ""),
            "normalized_text": str(self.normalized_text or ""),
            "canonical_text": str(self.canonical_text or ""),
            "search_text": str(self.search_text or ""),
            "anchors": [item.to_dict() for item in list(self.anchors or [])],
            "contextual_resolution": dict(self.contextual_resolution or {}),
            "entities": [item.to_dict() for item in list(self.entities or [])],
            "numeric_facts": [item.to_dict() for item in list(self.numeric_facts or [])],
            "stable_facts": [item.to_dict() for item in list(self.stable_facts or [])],
            "emotion": (self.emotion.to_dict() if self.emotion is not None else None),
            "tags": [str(x).strip().lower() for x in list(self.tags or []) if str(x).strip()],
            "memory_views": dict(self.memory_views or {}),
        }


_SELF_CUES = (
    "у меня",
    "я использую",
    "мы используем",
    "работаю на",
    "сижу на",
    "запускаю на",
    "my ",
    "my name is",
    "i am",
    "i'm",
    "i use",
    "we use",
    "running on",
)
_PROJECT_CUES = (
    "проект",
    "project",
    "мой проект",
    "наш проект",
    "в проекте",
    "для проекта",
)
_ENVIRONMENT_ENTITY_TYPES = {
    "gpu_model",
    "cpu_model",
    "ram_size",
    "vram_size",
    "python_version",
    "llm_model",
    "tool_name",
    "os_name",
}


def analyze_message_for_memory(text: str, *, metadata: dict | None = None) -> IngestAnalysis:
    from memory.anchor_extractor import extract_anchors
    from memory.contextual_resolver import resolve_anchor_context
    from memory.emotion_profile import detect_memory_emotion
    from memory.entity_resolver import resolve_entities
    from memory.fact_synthesizer import synthesize_stable_facts
    from memory.numeric_extractor import extract_numeric_facts
    from memory.retrieval_projection import build_memory_views

    raw = str(text or "")
    meta = dict(metadata or {})
    views = build_memory_views(raw, metadata=meta)

    anchors = extract_anchors(raw, metadata=meta)
    contextual = resolve_anchor_context(raw, anchors=anchors)
    entities = resolve_entities(raw, metadata=meta, anchors=anchors, context=contextual.to_dict())
    numeric_facts = extract_numeric_facts(raw, entities=entities, anchors=anchors, context=contextual.to_dict())
    stable_facts = synthesize_stable_facts(
        raw,
        entities=entities,
        numeric_facts=numeric_facts,
        contextual_resolution=contextual.to_dict(),
        metadata=meta,
    )
    emotion = detect_memory_emotion(raw) if raw.strip() else None
    views = build_memory_views(
        raw,
        metadata={
            **meta,
            "anchors": [item.to_dict() for item in list(anchors or [])],
            "contextual_resolution": dict(contextual.to_dict()),
            "memory_entities": [item.to_dict() for item in list(entities or [])],
            "numeric_facts": [item.to_dict() for item in list(numeric_facts or [])],
            "memory_views": dict(views or {}),
        },
    )
    tags = build_tags_from_analysis(
        entities=entities,
        numeric_facts=numeric_facts,
        stable_facts=stable_facts,
        emotion=emotion,
    )

    return IngestAnalysis(
        raw_text=raw,
        normalized_text=str(views.get("normalized_text") or ""),
        canonical_text=str(views.get("canonical_text") or ""),
        search_text=str(views.get("search_text") or raw),
        anchors=anchors,
        contextual_resolution=contextual.to_dict(),
        entities=entities,
        numeric_facts=numeric_facts,
        stable_facts=stable_facts,
        emotion=emotion,
        tags=tags,
        memory_views=views,
    )


def build_tags_from_analysis(
    *,
    entities: list[EntityItem],
    numeric_facts: list[NumericFact],
    stable_facts: list[StableFact] | None = None,
    emotion: IngestEmotion | None = None,
) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()

    def _add(tag: str) -> None:
        token = str(tag or "").strip().lower()
        if not token or token in seen:
            return
        seen.add(token)
        out.append(token)

    for item in list(entities or []):
        entity_type = str(item.type or "").strip().lower()
        canonical = _slug(item.canonical)
        if entity_type in {"gpu_model", "cpu_model", "ram_size", "vram_size"}:
            _add("topic_hardware")
        if entity_type in {"python_version", "llm_model", "tool_name", "os_name"}:
            _add("topic_environment")
        if entity_type == "project_name":
            _add("topic_project")
        if entity_type == "person_name":
            _add("topic_identity")
        if canonical:
            _add(f"entity_{entity_type}_{canonical}")

    for item in list(numeric_facts or []):
        kind = _slug(item.kind)
        if not kind:
            continue
        _add(f"numeric_{kind}")
        if kind in {"vram_gb", "ram_gb", "memory_gb"}:
            _add("topic_hardware")
        elif kind in {"python_version"}:
            _add("topic_environment")
        elif kind in {"age_years"}:
            _add("topic_identity")

    for item in list(stable_facts or []):
        predicate = _slug(item.predicate)
        relation = _slug(item.relation)
        if predicate:
            _add(f"fact_{predicate}")
        if relation:
            _add(f"relation_{relation}")

    if emotion is not None and str(emotion.primary or "").strip():
        _add(f"emotion_{emotion.primary}")

    return out


def _build_stable_facts(
    text: str,
    *,
    entities: list[EntityItem],
    numeric_facts: list[NumericFact],
    metadata: dict[str, Any] | None = None,
) -> list[StableFact]:
    lower = str(text or "").lower()
    meta = dict(metadata or {})
    self_context = _has_any(lower, _SELF_CUES)
    project_context = _has_any(lower, _PROJECT_CUES) or bool(meta.get("project_name"))

    out: list[StableFact] = []
    seen: set[tuple[str, str, str]] = set()

    def _push(subject: str, predicate: str, value: Any, confidence: float, relation: str) -> None:
        key = (str(subject).lower(), str(predicate).lower(), str(value).strip().lower())
        if key in seen:
            return
        seen.add(key)
        out.append(
            StableFact(
                subject=str(subject or "").strip().lower(),
                predicate=str(predicate or "").strip().lower(),
                value=value,
                confidence=max(0.0, min(1.0, float(confidence))),
                relation=str(relation or "").strip().lower(),
            )
        )

    for item in list(entities or []):
        entity_type = str(item.type or "").strip().lower()
        value = str(item.canonical or item.surface or "").strip()
        confidence = float(item.confidence or 0.0)
        if not value:
            continue
        if entity_type == "person_name" and self_context:
            _push("identity", "name", value, confidence, "identity")
        elif entity_type == "project_name" and project_context:
            _push("project", "name", value, confidence, "project")
        elif entity_type in _ENVIRONMENT_ENTITY_TYPES and self_context:
            mapping = {
                "gpu_model": ("hardware", "gpu_model"),
                "cpu_model": ("hardware", "cpu_model"),
                "ram_size": ("hardware", "ram_size"),
                "vram_size": ("hardware", "gpu_vram_size"),
                "python_version": ("environment", "python_version"),
                "llm_model": ("environment", "llm_model"),
                "tool_name": ("environment", "tool_name"),
                "os_name": ("environment", "os_name"),
            }
            target = mapping.get(entity_type)
            if target is not None:
                _push(target[0], target[1], value, confidence, "environment")

    for item in list(numeric_facts or []):
        kind = str(item.kind or "").strip().lower()
        confidence = 0.94
        if kind == "age_years" and self_context:
            _push("identity", "age_years", item.value, confidence, "identity")
        elif kind in {"vram_gb", "ram_gb", "memory_gb", "python_version"} and self_context:
            mapping = {
                "vram_gb": ("hardware", "gpu_vram_gb"),
                "ram_gb": ("hardware", "ram_gb"),
                "memory_gb": ("hardware", "memory_gb"),
                "python_version": ("environment", "python_version"),
            }
            target = mapping.get(kind)
            if target is not None:
                _push(target[0], target[1], item.value, confidence, "environment")

    return out


def _has_any(text: str, markers: tuple[str, ...]) -> bool:
    src = str(text or "")
    if not src:
        return False
    return any(marker in src for marker in markers)


def _slug(value: str) -> str:
    out = []
    for ch in str(value or "").strip().lower():
        if ch.isalnum():
            out.append(ch)
        else:
            out.append("_")
    return "".join(out).strip("_")
