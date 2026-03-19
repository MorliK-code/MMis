from __future__ import annotations

from typing import Any

from memory.ingest_analyzer import EntityItem, NumericFact, StableFact


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
    "i have",
)
_PROJECT_CUES = (
    "проект",
    "project",
    "мой проект",
    "наш проект",
    "в проекте",
    "для проекта",
)


def synthesize_stable_facts(
    text: str,
    *,
    entities: list[EntityItem],
    numeric_facts: list[NumericFact],
    contextual_resolution: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> list[StableFact]:
    lower = str(text or "").lower()
    resolution = dict(contextual_resolution or {})
    signals = dict(resolution.get("signals") or {})
    current_os_values = {
        str(x or "").strip().lower()
        for x in list(resolution.get("current_os_values") or [])
        if str(x or "").strip()
    }
    past_os_values = {
        str(x or "").strip().lower()
        for x in list(resolution.get("past_os_values") or [])
        if str(x or "").strip()
    }

    self_context = _has_any(lower, _SELF_CUES) or bool(signals.get("age_present"))
    project_context = _has_any(lower, _PROJECT_CUES)

    out: list[StableFact] = []
    seen: set[tuple[str, str, str]] = set()

    def _push(subject: str, predicate: str, value: Any, confidence: float, relation: str) -> None:
        key = (str(subject).strip().lower(), str(predicate).strip().lower(), str(value).strip().lower())
        if not key[0] or not key[1] or not key[2] or key in seen:
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
        elif entity_type == "gpu_model" and self_context:
            _push("hardware", "gpu_model", value, confidence, "environment")
        elif entity_type == "cpu_model" and self_context:
            _push("hardware", "cpu_model", value, confidence, "environment")
        elif entity_type == "ram_size" and self_context:
            _push("hardware", "ram_size", value, confidence, "environment")
        elif entity_type == "vram_size" and self_context:
            _push("hardware", "gpu_vram_size", value, confidence, "environment")
        elif entity_type == "python_version" and self_context:
            _push("environment", "python_version", value, confidence, "environment")
        elif entity_type == "llm_model" and self_context:
            _push("environment", "llm_model", value, confidence, "environment")
        elif entity_type == "tool_name" and self_context:
            _push("environment", "tool_name", value, confidence, "environment")
        elif entity_type == "os_name" and self_context:
            value_key = str(value or "").strip().lower()
            if current_os_values and value_key not in current_os_values:
                continue
            if not current_os_values and past_os_values and value_key in past_os_values:
                continue
            _push("environment", "os_name", value, confidence, "environment")

    for item in list(numeric_facts or []):
        kind = str(item.kind or "").strip().lower()
        confidence = 0.94
        if kind == "age_years" and self_context:
            _push("identity", "age_years", item.value, confidence, "identity")
        elif kind == "vram_gb" and self_context:
            _push("hardware", "gpu_vram_gb", item.value, confidence, "environment")
        elif kind == "ram_gb" and self_context:
            _push("hardware", "ram_gb", item.value, confidence, "environment")
        elif kind == "memory_gb" and self_context:
            _push("hardware", "memory_gb", item.value, confidence, "environment")
        elif kind == "python_version" and self_context:
            _push("environment", "python_version", item.value, confidence, "environment")

    return out


def _has_any(text: str, markers: tuple[str, ...]) -> bool:
    src = str(text or "")
    if not src:
        return False
    return any(marker in src for marker in markers)
