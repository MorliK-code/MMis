from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from memory.ingest_analyzer import IngestAnchor


_GPU_MARKERS = ("rtx", "gtx", "rx", "gpu", "видюх", "видеокарт", "nvidia", "geforce", "radeon")
_RAM_MARKERS = (" ram", "озу", "оператив")
_PYTHON_MARKERS = ("python", "питон", "пайтон", "удав")
_OS_MARKERS = ("windows", "linux", "ubuntu", "debian", "macos", "винда", "винды", "ос", "операционк", "виндовс", "система")
_ACTUAL_OS_VALUES = {"windows", "linux", "ubuntu", "debian", "macos"}


@dataclass(frozen=True)
class ContextualResolution:
    domains: tuple[str, ...] = ()
    signals: dict[str, Any] = field(default_factory=dict)
    memory_kind_by_surface: dict[str, str] = field(default_factory=dict)
    version_kind_by_surface: dict[str, str] = field(default_factory=dict)
    os_state_by_surface: dict[str, str] = field(default_factory=dict)
    current_os_values: tuple[str, ...] = ()
    past_os_values: tuple[str, ...] = ()
    age_surfaces: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "domains": list(self.domains or ()),
            "signals": dict(self.signals or {}),
            "memory_kind_by_surface": dict(self.memory_kind_by_surface or {}),
            "version_kind_by_surface": dict(self.version_kind_by_surface or {}),
            "os_state_by_surface": dict(self.os_state_by_surface or {}),
            "current_os_values": list(self.current_os_values or ()),
            "past_os_values": list(self.past_os_values or ()),
            "age_surfaces": list(self.age_surfaces or ()),
        }


def resolve_anchor_context(text: str, *, anchors: list[IngestAnchor] | None = None) -> ContextualResolution:
    src = str(text or "")
    lower = src.lower()
    items = list(anchors or [])

    gpu_present = any(str(item.kind or "").strip().lower() == "gpu_model_candidate" for item in items) or _has_any(lower, _GPU_MARKERS)
    ram_present = any(str(item.kind or "").strip().lower() == "ram_candidate" for item in items) or _has_any(lower, _RAM_MARKERS)
    python_present = any(
        str(item.kind or "").strip().lower() in {"python_version_candidate", "python_present_candidate"}
        for item in items
    ) or _has_any(lower, _PYTHON_MARKERS)
    os_present = any(str(item.kind or "").strip().lower() in {"os_candidate", "os_mention_candidate"} for item in items) or _has_any(lower, _OS_MARKERS)
    age_present = False

    memory_kind_by_surface: dict[str, str] = {}
    version_kind_by_surface: dict[str, str] = {}
    os_state_by_surface: dict[str, str] = {}
    current_os_values: list[str] = []
    past_os_values: list[str] = []
    age_surfaces: list[str] = []

    for item in items:
        kind = str(item.kind or "").strip().lower()
        surface = str(item.surface or "").strip()
        normalized = str(item.normalized or "").strip()
        hints = dict(item.hints or {})

        if kind == "gpu_model_candidate":
            gpu_present = True
            continue

        if kind == "ram_candidate":
            ram_present = True
            if surface:
                memory_kind_by_surface.setdefault(surface, "ram")
            continue

        if kind == "memory_size_candidate":
            memory_kind = str(hints.get("memory_kind") or "").strip().lower()
            if not memory_kind:
                if gpu_present or _has_any(lower, _GPU_MARKERS):
                    memory_kind = "vram"
                elif ram_present or _has_any(lower, _RAM_MARKERS):
                    memory_kind = "ram"
                elif normalized:
                    memory_kind = "memory"
            if surface and memory_kind:
                memory_kind_by_surface[surface] = memory_kind
            if memory_kind == "vram":
                gpu_present = True
            elif memory_kind == "ram":
                ram_present = True
            continue

        if kind == "python_version_candidate":
            python_present = True
            if surface:
                version_kind_by_surface[surface] = "runtime_python"
            continue

        if kind == "python_present_candidate":
            python_present = True
            continue

        if kind in {"person_age_candidate", "age_candidate"}:
            age_present = True
            if surface:
                age_surfaces.append(surface)
            continue

        if kind in {"os_candidate", "os_mention_candidate"}:
            os_present = True
            state = str(hints.get("temporal_state") or "").strip().lower()
            normalized_key = _norm_token(normalized)
            if surface and state:
                os_state_by_surface[surface] = state
            if normalized_key in _ACTUAL_OS_VALUES:
                if state == "past":
                    if normalized not in past_os_values:
                        past_os_values.append(normalized)
                else:
                    if normalized not in current_os_values:
                        current_os_values.append(normalized)

    domains: list[str] = []
    if gpu_present or ram_present:
        domains.append("hardware")
    if python_present:
        domains.append("runtime")
    if os_present:
        domains.append("environment")
    if age_present:
        domains.append("identity")

    return ContextualResolution(
        domains=tuple(domains),
        signals={
            "gpu_present": gpu_present,
            "ram_present": ram_present,
            "python_present": python_present,
            "os_present": os_present,
            "current_os_present": bool(current_os_values),
            "past_os_present": bool(past_os_values),
            "age_present": age_present,
        },
        memory_kind_by_surface=memory_kind_by_surface,
        version_kind_by_surface=version_kind_by_surface,
        os_state_by_surface=os_state_by_surface,
        current_os_values=tuple(current_os_values),
        past_os_values=tuple(past_os_values),
        age_surfaces=tuple(age_surfaces),
    )


def _has_any(text: str, markers: tuple[str, ...]) -> bool:
    src = str(text or "")
    if not src:
        return False
    return any(marker in src for marker in markers)


def _norm_token(value: str) -> str:
    return str(value or "").strip().lower()
