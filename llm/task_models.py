from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from config.settings import AppSettings, VALID_PROVIDERS, load_config


def _normalize_task_name(value: str | None) -> str:
    return str(value or "").strip().lower()


@dataclass(frozen=True)
class TaskModelProfile:
    name: str
    provider: str
    model: str
    temperature: float
    max_tokens: int
    timeout: float
    enabled: bool = True
    fallback_profile: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TaskModelRequest:
    task_name: str
    prompt: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TaskModelResult:
    task_name: str
    status: str
    enabled: bool
    reason: str = ""
    profile: TaskModelProfile | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        if self.profile is not None:
            payload["profile"] = self.profile.to_dict()
        return payload


class TaskModelRegistry:
    def __init__(self, profiles: dict[str, TaskModelProfile]):
        self._profiles = dict(profiles or {})
        self._validate_profiles()

    @classmethod
    def from_settings(cls, settings: AppSettings | None = None) -> "TaskModelRegistry":
        cfg = settings or load_config()
        rows = dict(getattr(cfg, "task_model_profiles", {}) or {})
        profiles: dict[str, TaskModelProfile] = {}
        for task_name, raw in rows.items():
            profile = cls._coerce_profile(task_name, raw)
            profiles[profile.name] = profile
        return cls(profiles)

    @staticmethod
    def _coerce_profile(task_name: str, raw: Any) -> TaskModelProfile:
        if isinstance(raw, TaskModelProfile):
            return raw
        payload = dict(raw or {})
        name = _normalize_task_name(payload.get("name") or task_name)
        provider = _normalize_task_name(payload.get("provider"))
        model = str(payload.get("model") or "").strip()
        fallback_profile = _normalize_task_name(payload.get("fallback_profile"))
        try:
            temperature = float(payload.get("temperature"))
        except Exception as exc:
            raise ValueError(f"Task model profile '{name}' has invalid temperature") from exc
        try:
            max_tokens = int(payload.get("max_tokens"))
        except Exception as exc:
            raise ValueError(f"Task model profile '{name}' has invalid max_tokens") from exc
        try:
            timeout = float(payload.get("timeout"))
        except Exception as exc:
            raise ValueError(f"Task model profile '{name}' has invalid timeout") from exc
        enabled = bool(payload.get("enabled", True))
        return TaskModelProfile(
            name=name,
            provider=provider,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            enabled=enabled,
            fallback_profile=fallback_profile,
        )

    def _validate_profiles(self) -> None:
        known_names = set(self._profiles.keys())
        for name, profile in self._profiles.items():
            if not name:
                raise ValueError("Task model profile name cannot be empty")
            if profile.provider not in VALID_PROVIDERS:
                raise ValueError(
                    f"Task model profile '{name}' has unsupported provider '{profile.provider}'"
                )
            if not profile.model:
                raise ValueError(f"Task model profile '{name}' must define model")
            if not (0.0 <= float(profile.temperature) <= 2.0):
                raise ValueError(f"Task model profile '{name}' temperature must be in [0, 2]")
            if int(profile.max_tokens) < 1:
                raise ValueError(f"Task model profile '{name}' max_tokens must be >= 1")
            if float(profile.timeout) <= 0.0:
                raise ValueError(f"Task model profile '{name}' timeout must be > 0")
            if profile.fallback_profile:
                if profile.fallback_profile == name:
                    raise ValueError(f"Task model profile '{name}' fallback_profile cannot point to itself")
                if profile.fallback_profile not in known_names:
                    raise ValueError(
                        f"Task model profile '{name}' fallback_profile '{profile.fallback_profile}' is not defined"
                    )

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._profiles.keys()))

    def get_task_profile(self, task_name: str) -> TaskModelProfile | None:
        return self._profiles.get(_normalize_task_name(task_name))

    def is_task_enabled(self, task_name: str) -> bool:
        profile = self.get_task_profile(task_name)
        return bool(profile and profile.enabled)

    def resolve_task(self, task_name: str) -> TaskModelResult:
        name = _normalize_task_name(task_name)
        profile = self.get_task_profile(name)
        if profile is None:
            return TaskModelResult(
                task_name=name,
                status="not_found",
                enabled=False,
                reason="profile_not_found",
                profile=None,
            )
        if not profile.enabled:
            return TaskModelResult(
                task_name=name,
                status="disabled",
                enabled=False,
                reason="profile_disabled",
                profile=profile,
            )
        return TaskModelResult(
            task_name=name,
            status="ok",
            enabled=True,
            reason="profile_enabled",
            profile=profile,
        )

def get_task_model_registry(*, force_reload: bool = False) -> TaskModelRegistry:
    settings = load_config(force_reload=force_reload)
    return TaskModelRegistry.from_settings(settings)


def get_task_profile(task_name: str, *, force_reload: bool = False) -> TaskModelProfile | None:
    return get_task_model_registry(force_reload=force_reload).get_task_profile(task_name)


def is_task_enabled(task_name: str, *, force_reload: bool = False) -> bool:
    return get_task_model_registry(force_reload=force_reload).is_task_enabled(task_name)
