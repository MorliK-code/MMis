"""
Конфигурация Memory Core.

Загружает настройки из memory_core/config.json
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from config.settings import BASE_DIR


@dataclass(slots=True)
class MemoryLLMConfig:
    """Конфигурация Memory LLM."""
    provider: str = "ollama"
    model: str = "qwen3.5:9b"
    temperature: float = 0.1
    max_tokens: int = 1024
    timeout: float = 60.0
    system_prompt: str = ""


@dataclass(slots=True)
class ArtifactTypeConfig:
    """Конфигурация типа артефакта."""
    description: str = ""
    default_decay: str = "slow"
    default_scope: str = "profile"
    priority: int = 5
    protected: bool = False


@dataclass(slots=True)
class GovernorConfig:
    """Конфигурация Governor."""
    confidence_high: float = 0.8
    confidence_medium: float = 0.5
    confidence_low: float = 0.3
    max_artifacts_per_type: int = 100
    protected_types: list[str] = field(default_factory=lambda: ["identity_core"])
    merge_min_overlap: int = 3


@dataclass(slots=True)
class RetrievalConfig:
    """Конфигурация Retrieval."""
    always_load: list[str] = field(default_factory=lambda: ["identity_core", "profile_fact", "task_state"])
    sometimes_load: list[str] = field(default_factory=lambda: ["preference", "episode_event", "emotional_state"])
    never_load_if_decay: str = "immediate"
    never_load_if_status: str = "superseded"
    never_load_if_confidence_below: float = 0.3


@dataclass(slots=True)
class MemoryCoreConfig:
    """
    Конфигурация Memory Core.
    """
    enabled: bool = True
    db_path: str = "data/memory_core/memory.db"
    vector_path: str = "data/memory_core/vector"
    default_workspace: str = "global"
    default_namespace: str = "default"
    top_k: int = 8
    enable_background_worker: bool = True
    worker_poll_interval: float = 2.0
    
    # Настройки паузы worker во время обработки запросов API
    enable_worker_pause_during_api_request: bool = True
    worker_pause_timeout: float = 0.0  # 0 = без ограничения, >0 = макс. время паузы в секундах
    memory_llm_scheduler_mode: str = "strict"  # strict | cooperative
    
    # Таймаут завершения worker при простое
    worker_shutdown_idle_timeout: float = 300.0  # 5 минут по умолчанию

    llm: MemoryLLMConfig = field(default_factory=MemoryLLMConfig)
    artifact_types: dict[str, ArtifactTypeConfig] = field(default_factory=dict)
    governor: GovernorConfig = field(default_factory=GovernorConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    
    # Профили task model (основной источник настроек LLM)
    task_model_profiles: dict[str, dict[str, Any]] = field(default_factory=dict)

    # Полные пути
    db_path_full: str = ""
    vector_path_full: str = ""

    @classmethod
    def load(cls, config_path: str | None = None) -> "MemoryCoreConfig":
        """
        Загружает конфигурацию из файла.

        Args:
            config_path: Путь к файлу конфигурации.

        Returns:
            Конфигурация MemoryCoreConfig.
        """
        if config_path is None:
            config_path = str(Path(BASE_DIR) / "memory_core" / "config.json")

        config_file = Path(config_path)
        if not config_file.exists():
            return cls()  # Возвращаем конфигурацию по умолчанию

        try:
            with config_file.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return cls()

        return cls._from_dict(data)

    @classmethod
    def _from_dict(cls, data: dict[str, Any]) -> "MemoryCoreConfig":
        """Создаёт конфигурацию из словаря."""
        config = cls()

        # Простые поля
        config.enabled = bool(data.get("enabled", True))
        config.db_path = str(data.get("db_path", "data/memory_core/memory.db"))
        config.vector_path = str(data.get("vector_path", "data/memory_core/vector"))
        config.default_workspace = str(data.get("default_workspace", "global"))
        config.default_namespace = str(data.get("default_namespace", "default"))
        config.top_k = int(data.get("top_k", 8))
        config.enable_background_worker = bool(data.get("enable_background_worker", True))
        config.worker_poll_interval = float(data.get("worker_poll_interval", 2.0))
        config.enable_worker_pause_during_api_request = bool(data.get("enable_worker_pause_during_api_request", True))
        config.worker_pause_timeout = float(data.get("worker_pause_timeout", 0.0))
        scheduler_mode = str(data.get("memory_llm_scheduler_mode", "strict")).strip().lower()
        config.memory_llm_scheduler_mode = scheduler_mode if scheduler_mode in {"strict", "cooperative"} else "strict"
        config.worker_shutdown_idle_timeout = float(data.get("worker_shutdown_idle_timeout", 300.0))
        
        # Task model profiles (основной источник настроек LLM)
        config.task_model_profiles = dict(data.get("task_model_profiles", {}))

        # Полные пути
        config.db_path_full = str(Path(BASE_DIR) / config.db_path)
        config.vector_path_full = str(Path(BASE_DIR) / config.vector_path)

        # Memory LLM
        llm_data = data.get("llm", {})
        config.llm = MemoryLLMConfig(
            provider=str(llm_data.get("provider", "ollama")),
            model=str(llm_data.get("model", "qwen3:1.7b")),
            temperature=float(llm_data.get("temperature", 0.1)),
            max_tokens=int(llm_data.get("max_tokens", 1024)),
            timeout=float(llm_data.get("timeout", 60.0)),
            system_prompt=str(llm_data.get("system_prompt", "")),
        )

        # Task Model Profiles читаются из memory_core/config.json
        # Для использования в TaskRouter нужно загружать их отдельно
        task_profiles_data = data.get("task_model_profiles", {})
        # Не регистрируем в глобальном конфиге, используем только внутри memory_core

        # Artifact Types
        artifact_types_data = data.get("artifact_types", {})
        config.artifact_types = {}
        for type_name, type_data in artifact_types_data.items():
            config.artifact_types[type_name] = ArtifactTypeConfig(
                description=str(type_data.get("description", "")),
                default_decay=str(type_data.get("default_decay", "slow")),
                default_scope=str(type_data.get("default_scope", "profile")),
                priority=int(type_data.get("priority", 5)),
                protected=bool(type_data.get("protected", False)),
            )

        # Governor
        governor_data = data.get("governor", {})
        confidence_data = governor_data.get("confidence_thresholds", {})
        config.governor = GovernorConfig(
            confidence_high=float(confidence_data.get("high", 0.8)),
            confidence_medium=float(confidence_data.get("medium", 0.5)),
            confidence_low=float(confidence_data.get("low", 0.3)),
            max_artifacts_per_type=int(governor_data.get("max_artifacts_per_type", 100)),
            protected_types=list(governor_data.get("protected_types", ["identity_core"])),
            merge_min_overlap=int(governor_data.get("merge_min_overlap", 3)),
        )

        # Retrieval
        retrieval_data = data.get("retrieval", {})
        never_load_data = retrieval_data.get("never_load_if", {})
        config.retrieval = RetrievalConfig(
            always_load=list(retrieval_data.get("always_load", [])),
            sometimes_load=list(retrieval_data.get("sometimes_load", [])),
            never_load_if_decay=str(never_load_data.get("decay", "immediate")),
            never_load_if_status=str(never_load_data.get("status", "superseded")),
            never_load_if_confidence_below=float(never_load_data.get("confidence_below", 0.3)),
        )

        return config

    def to_dict(self) -> dict[str, Any]:
        """Преобразует конфигурацию в словарь."""
        return {
            "enabled": self.enabled,
            "db_path": self.db_path,
            "vector_path": self.vector_path,
            "default_workspace": self.default_workspace,
            "default_namespace": self.default_namespace,
            "top_k": self.top_k,
            "enable_background_worker": self.enable_background_worker,
            "worker_poll_interval": self.worker_poll_interval,
            "enable_worker_pause_during_api_request": self.enable_worker_pause_during_api_request,
            "worker_pause_timeout": self.worker_pause_timeout,
            "memory_llm_scheduler_mode": self.memory_llm_scheduler_mode,
            "worker_shutdown_idle_timeout": self.worker_shutdown_idle_timeout,
            "llm": {
                "provider": self.llm.provider,
                "model": self.llm.model,
                "temperature": self.llm.temperature,
                "max_tokens": self.llm.max_tokens,
                "timeout": self.llm.timeout,
                "system_prompt": self.llm.system_prompt,
            },
            "task_model_profiles": self.task_model_profiles,
            "governor": {
                "confidence_thresholds": {
                    "high": self.governor.confidence_high,
                    "medium": self.governor.confidence_medium,
                    "low": self.governor.confidence_low,
                },
                "max_artifacts_per_type": self.governor.max_artifacts_per_type,
                "protected_types": self.governor.protected_types,
            },
            "retrieval": {
                "always_load": self.retrieval.always_load,
                "sometimes_load": self.retrieval.sometimes_load,
                "never_load_if": {
                    "decay": self.retrieval.never_load_if_decay,
                    "status": self.retrieval.never_load_if_status,
                    "confidence_below": self.retrieval.never_load_if_confidence_below,
                },
            },
        }


# Глобальный кэш конфигурации
_config_cache: MemoryCoreConfig | None = None


def get_memory_core_config(config_path: str | None = None) -> MemoryCoreConfig:
    """
    Получает конфигурацию Memory Core.

    Args:
        config_path: Путь к файлу конфигурации (опционально).

    Returns:
        Конфигурация MemoryCoreConfig.
    """
    global _config_cache
    if _config_cache is None:
        _config_cache = MemoryCoreConfig.load(config_path)
    return _config_cache


def reload_memory_core_config(config_path: str | None = None) -> MemoryCoreConfig:
    """
    Перезагружает конфигурацию.

    Args:
        config_path: Путь к файлу конфигурации.

    Returns:
        Обновлённая конфигурация.
    """
    global _config_cache
    _config_cache = MemoryCoreConfig.load(config_path)
    return _config_cache
