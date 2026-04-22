"""
Memory Core Adapter - адаптер для интеграции memory_core в существующий MMis.

Этот файл позволяет постепенно переводить проект на новую память,
сохраняя обратную совместимость со старыми интерфейсами.
"""

from typing import Any
from memory_core.bootstrap.service_factory import build_memory_service, MemoryServiceConfig
from memory_core.schemas import MemoryEnvelope, MemoryQuery
from memory_core.facade import MemoryService
from memory_core.config_manager import get_memory_core_config
from utils.logger import get_logger
import re
import threading
import time


LOGGER = get_logger(__name__)

_DURATION_RE = re.compile(r"^\s*([+-]?\d+(?:\.\d+)?)\s*([a-zA-Z]*)\s*$")

# Глобальная блокировка для Memory LLM
# Используется для приостановки Memory LLM во время ответа основной модели
_memory_llm_lock = threading.Lock()
_memory_llm_interrupt_epoch = 0
_memory_llm_interrupt_epoch_lock = threading.Lock()


def _mark_memory_llm_interrupt() -> int:
    global _memory_llm_interrupt_epoch
    with _memory_llm_interrupt_epoch_lock:
        _memory_llm_interrupt_epoch += 1
        return int(_memory_llm_interrupt_epoch)


def _get_memory_llm_interrupt_epoch() -> int:
    with _memory_llm_interrupt_epoch_lock:
        return int(_memory_llm_interrupt_epoch)


class MemoryCoreAdapter:
    """
    Адаптер memory_core для использования в main.py и brain.py.

    Предоставляет упрощённый интерфейс для миграции со старого MemoryManager.
    """

    def __init__(
        self,
        db_path: str | None = None,
        vector_path: str | None = None,
        default_workspace: str | None = None,
        default_namespace: str | None = None,
        top_k: int | None = None,
        enable_background_worker: bool | None = None,
        worker_poll_interval: float | None = None,
        config_path: str | None = None,
    ):
        """
        Инициализирует адаптер.

        Args:
            db_path: Путь к SQLite базе данных.
            vector_path: Путь к векторному индексу.
            default_workspace: Workspace по умолчанию.
            default_namespace: Namespace по умолчанию.
            top_k: Количество результатов по умолчанию.
            enable_background_worker: Включить фоновый воркер.
            worker_poll_interval: Интервал опроса очереди (сек).
            config_path: Путь к файлу конфигурации.
        """
        # Загружаем конфигурацию из файла
        config_file = get_memory_core_config(config_path)

        # Переопределяем параметрами из конструктора
        self.config = MemoryServiceConfig(
            db_path=db_path or config_file.db_path,
            vector_path=vector_path or config_file.vector_path,
            default_workspace=default_workspace or config_file.default_workspace,
            default_namespace=default_namespace or config_file.default_namespace,
            top_k=top_k or config_file.top_k,
            enable_background_worker=(
                enable_background_worker
                if enable_background_worker is not None
                else config_file.enable_background_worker
            ),
            worker_poll_interval=(
                worker_poll_interval
                if worker_poll_interval is not None
                else config_file.worker_poll_interval
            ),
            worker_shutdown_idle_timeout=config_file.worker_shutdown_idle_timeout,
        )

        self.service = build_memory_service(self.config)
        self._current_workspace = self.config.default_workspace
        self._current_session = "default"
        self._config_file = config_file
        self._scheduler_mode = str(getattr(config_file, "memory_llm_scheduler_mode", "strict") or "strict").strip().lower()
        if self._scheduler_mode not in {"strict", "cooperative"}:
            self._scheduler_mode = "strict"
        
        # Настройки паузы worker из конфига
        self._memory_wake_delay_after_main_sec = max(
            0.0,
            float(getattr(config_file, "memory_llm_wake_delay_after_main_sec", 0.0) or 0.0),
        )
        self._unload_memory_llm_before_main_request = bool(
            getattr(config_file, "unload_memory_llm_before_main_request", False)
        )
        requires_api_arbitration = (
            self.is_strict_scheduler_mode()
            or self._unload_memory_llm_before_main_request
            or self._memory_wake_delay_after_main_sec > 0
        )
        self._enable_pause = bool(config_file.enable_worker_pause_during_api_request) and requires_api_arbitration
        self._pause_timeout = float(config_file.worker_pause_timeout or 0.0)
        
        # Таймер для автоматического возобновления Memory LLM
        self._auto_resume_timer: threading.Timer | None = None
        self._main_sleep_timer: threading.Timer | None = None
        self._main_sleep_deadline_at = 0.0
        self._main_lease_holds_memory = False
        self._auto_resume_delay = self._memory_wake_delay_after_main_sec
        self._resume_epoch = 0
        self._resume_epoch_lock = threading.Lock()
        self._memory_drain_lock = threading.Lock()
        self._memory_drained_epoch: int | None = None
        self._install_worker_lifecycle_hooks()
        LOGGER.info(
            "MemoryCoreAdapter scheduler mode: %s (worker pause enabled=%s, memory wake delay=%.1fs, unload before main=%s)",
            self._scheduler_mode,
            self._enable_pause,
            self._memory_wake_delay_after_main_sec,
            self._unload_memory_llm_before_main_request,
        )

    def _install_worker_lifecycle_hooks(self) -> None:
        worker = self._get_worker()
        if worker is None:
            return
        try:
            setattr(worker, "on_memory_queue_idle", self._on_memory_worker_queue_idle)
        except Exception as exc:
            LOGGER.debug(f"MemoryCoreAdapter: failed to install worker lifecycle hooks: {exc}")

    def scheduler_mode(self) -> str:
        mode = str(getattr(self, "_scheduler_mode", "strict") or "strict").strip().lower()
        return mode if mode in {"strict", "cooperative"} else "strict"

    def is_strict_scheduler_mode(self) -> bool:
        return self.scheduler_mode() == "strict"

    def should_pause_worker_for_api_request(self) -> bool:
        return self.worker_pause_enabled

    @staticmethod
    def _duration_to_seconds(value: Any) -> float:
        if value is None:
            return 0.0
        if isinstance(value, (int, float)):
            return max(0.0, float(value))
        text = str(value or "").strip().lower()
        if not text:
            return 0.0
        match = _DURATION_RE.match(text)
        if not match:
            return 0.0
        try:
            number = float(match.group(1))
        except Exception:
            return 0.0
        if number <= 0:
            return 0.0
        unit = str(match.group(2) or "s").strip().lower()
        if unit in {"", "s", "sec", "secs", "second", "seconds"}:
            multiplier = 1.0
        elif unit in {"m", "min", "mins", "minute", "minutes"}:
            multiplier = 60.0
        elif unit in {"h", "hr", "hrs", "hour", "hours"}:
            multiplier = 3600.0
        elif unit in {"ms", "msec", "millisecond", "milliseconds"}:
            multiplier = 0.001
        else:
            return 0.0
        return max(0.0, number * multiplier)

    def _main_model_keep_alive_delay_sec(self) -> float:
        runtime_ollama = self._runtime_main_ollama_settings()
        if runtime_ollama:
            return self._duration_to_seconds(runtime_ollama.get("keep_alive"))
        try:
            from config.settings import get_profile, load_config

            settings = load_config()
            profile = get_profile(getattr(settings, "active_profile", "BALANCED"))
            return self._duration_to_seconds(getattr(profile.ollama, "keep_alive", ""))
        except Exception as exc:
            LOGGER.debug(f"MemoryCoreAdapter: failed to resolve main keep_alive delay: {exc}")
            return 0.0

    def _memory_handoff_delay_sec(self) -> float:
        return max(
            0.0,
            float(getattr(self, "_memory_wake_delay_after_main_sec", 0.0) or 0.0),
        )

    def _runtime_main_ollama_settings(self) -> dict[str, Any]:
        try:
            from api.app import _resolved_profile_payload

            _profile, payload = _resolved_profile_payload()
            ollama = dict(dict(payload or {}).get("ollama") or {})
            return ollama
        except Exception:
            return {}

    def _main_model_warmup_options(self) -> dict[str, Any]:
        settings = self._runtime_main_ollama_settings()
        if not settings:
            try:
                from config.settings import get_profile, load_config

                app_settings = load_config()
                profile = get_profile(getattr(app_settings, "active_profile", "BALANCED"))
                settings = {
                    "num_thread": int(profile.ollama.num_thread),
                    "num_ctx": int(profile.ollama.num_ctx),
                    "num_gpu": int(profile.ollama.num_gpu),
                    "num_batch": int(profile.ollama.num_batch),
                }
            except Exception:
                settings = {}
        out: dict[str, Any] = {}
        for key, value in dict(settings or {}).items():
            if key not in {"num_ctx", "num_thread", "num_gpu", "num_batch"} or value is None:
                continue
            try:
                out[key] = int(value)
            except Exception:
                continue
        return out

    def _main_sleep_remaining_sec(self) -> float:
        deadline = float(getattr(self, "_main_sleep_deadline_at", 0.0) or 0.0)
        if deadline <= 0:
            return 0.0
        return max(0.0, deadline - time.monotonic())

    def ingest_event(
        self,
        text: str,
        source_kind: str = "user",
        payload_type: str = "message",
        metadata: dict[str, Any] | None = None,
        session_id: str | None = None,
        workspace_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Добавляет событие в память.

        Args:
            text: Текст события.
            source_kind: Тип источника (user | assistant | tool | system).
            payload_type: Тип контента (message | tool_result | state_update).
            metadata: Дополнительные метаданные.
            session_id: ID сессии.
            workspace_id: ID workspace.

        Returns:
            Результат ingest.
        """
        envelope = MemoryEnvelope(
            source_kind=source_kind,
            payload_type=payload_type,
            text=text,
            metadata=metadata or {},
            workspace_id=workspace_id or self._current_workspace,
            session_id=session_id or self._current_session,
        )

        return self.service.ingest_event(envelope)

    def query(
        self,
        text: str,
        workspace_id: str | None = None,
        session_id: str | None = None,
        top_k: int | None = None,
        include_citations: bool = True,
        topic_thread_id: str | None = None,
        related_topic_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Выполняет запрос к памяти.

        Args:
            text: Текст запроса.
            workspace_id: ID workspace.
            session_id: ID сессии.
            top_k: Количество результатов.
            include_citations: Включать ли цитаты.

        Returns:
            Результат запроса с context_blocks.
        """
        query = MemoryQuery(
            text=text,
            workspace_id=workspace_id or self._current_workspace,
            session_id=session_id or self._current_session,
            top_k=top_k or self.config.top_k,
            include_citations=include_citations,
            topic_thread_id=topic_thread_id,
            related_topic_ids=list(related_topic_ids or []),
        )

        result = self.service.query(query)
        runtime_state: dict[str, Any] = {}
        runtime_store = getattr(self.service, "runtime_session_store", None)
        if runtime_store is not None and hasattr(runtime_store, "get_state"):
            try:
                runtime_state = dict(runtime_store.get_state(query.workspace_id, query.session_id) or {})
            except Exception:
                runtime_state = {}

        return {
            "context_blocks": result.context_blocks,
            "hits": result.hits,
            "citations": result.citations,
            "blocks": dict(result.blocks or {}),
            "selected": list(result.selected or []),
            "dropped": list(result.dropped or []),
            "recent_user_state": dict(result.recent_user_state or {}),
            "response_bias": dict(result.response_bias or {}),
            "debug": dict(result.debug or {}),
            "open_questions": [
                str(dict(item).get("text") or "").strip()
                for item in list(runtime_state.get("open_questions") or [])
                if isinstance(item, dict) and str(dict(item).get("text") or "").strip()
            ],
            "current_decisions": [
                str(dict(item).get("text") or "").strip()
                for item in list(runtime_state.get("recent_decisions") or [])
                if isinstance(item, dict) and str(dict(item).get("text") or "").strip()
            ],
            "task_continuity": {
                "active_task": dict(runtime_state.get("active_task") or {}),
                "current_episode_id": str(runtime_state.get("current_episode_id") or ""),
                "active_topic": str(runtime_state.get("active_topic") or ""),
            },
        }

    def get_context(
        self,
        query_text: str,
        workspace_id: str | None = None,
    ) -> str:
        """
        Получает контекст для LLM.

        Args:
            query_text: Текст запроса.
            workspace_id: ID workspace.

        Returns:
            Текстовый контекст.
        """
        result = self.query(query_text, workspace_id)

        if not result["context_blocks"]:
            return ""

        return "\n\n".join(result["context_blocks"])

    def retrieve(
        self,
        query: str,
        top_k: int = 8,
        workspace_id: str | None = None,
        session_id: str | None = None,
        topic_thread_id: str | None = None,
        related_topic_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Compatibility shim для старого response_pipeline.

        Args:
            query: Текст запроса.
            top_k: Количество результатов.
            workspace_id: ID workspace.
            session_id: ID сессии.

        Returns:
            Результат в старом формате.
        """
        result = self.query(
            text=query,
            workspace_id=workspace_id,
            session_id=session_id,
            top_k=top_k,
            include_citations=True,
            topic_thread_id=topic_thread_id,
            related_topic_ids=related_topic_ids,
        )

        selected = [dict(x) for x in list(result.get("selected") or []) if isinstance(x, dict)]
        if not selected:
            for hit in result.get("hits", []):
                selected.append({
                    "artifact_id": hit.get("artifact_id"),
                    "artifact_type": hit.get("artifact_type"),
                    "text": hit.get("text"),
                    "summary": hit.get("summary", ""),
                    "prompt_view": hit.get("prompt_view", ""),
                    "exposure_mode": hit.get("exposure_mode", ""),
                    "score": hit.get("score", 1.0),
                    "confidence": hit.get("confidence", 0.5),
                    "metadata": hit.get("metadata", {}),
                })

        return {
            "selected": selected,
            "blocks": dict(result.get("blocks") or {"context_blocks": result.get("context_blocks", [])}),
            "dropped": list(result.get("dropped") or []),
            "recent_user_state": dict(result.get("recent_user_state") or {}),
            "response_bias": dict(result.get("response_bias") or {}),
            "debug": dict(result.get("debug") or {}),
            "recall_mode": "memory_core_query",
            "citations": result.get("citations", []),
            "open_questions": list(result.get("open_questions") or []),
            "current_decisions": list(result.get("current_decisions") or []),
            "task_continuity": dict(result.get("task_continuity") or {}),
        }

    def set_current_workspace(self, workspace_id: str) -> None:
        """Устанавливает текущий workspace."""
        self._current_workspace = workspace_id
        self.service.set_current_workspace(workspace_id)

    def get_current_workspace(self) -> str:
        """Получает текущий workspace."""
        return self._current_workspace

    def set_current_session(self, session_id: str) -> None:
        """Устанавливает текущую сессию."""
        self._current_session = session_id

    def get_stats(self) -> dict[str, Any]:
        """Получает статистику памяти."""
        return self.service.get_stats()

    def inspect(
        self,
        kind: str = "events",
        limit: int = 50,
        workspace_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Инспектирует память.

        Args:
            kind: Тип инспекции (events | artifacts | workspaces | profile | stats).
            limit: Максимальное количество результатов.

        Returns:
            Результат инспекции.
        """
        from memory_core.schemas import MemoryInspectRequest

        request = MemoryInspectRequest(
            kind=kind,
            limit=limit,
            workspace_id=workspace_id or getattr(self, "_current_workspace", "global"),
        )
        return self.service.inspect(request)

    def debug_snapshot(self, limit: int = 50) -> dict[str, Any]:
        """
        Метод для совместимости со старым debug_snapshot.

        Args:
            limit: Максимальное количество результатов.

        Returns:
            Снимок состояния памяти.
        """
        return {
            "events": self.inspect(kind="events", limit=limit),
            "artifacts": self.inspect(kind="artifacts", limit=limit),
            "runtime": self.inspect(kind="runtime", limit=limit),
            "episodes": self.inspect(kind="episodes", limit=limit),
            "stats": self.inspect(kind="stats"),
        }

    def get_identity_core_snapshot(
        self,
        workspace_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Получает snapshot Identity Core из памяти.

        Args:
            workspace_id: ID workspace.

        Returns:
            Identity Core snapshot в формате memory_core.
        """
        from memory_core.identity.identity_core import IdentityCore
        
        workspace = str(workspace_id or self._current_workspace or "global").strip() or "global"
        identity = IdentityCore(self.service.artifact_store)
        profile = identity.get_profile(workspace)
        return profile.to_dict()

    def get_episode_continuity(
        self,
        session_id: str,
        workspace_id: str | None = None,
        topic_thread_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Получает continuity pack для эпизода.

        Args:
            session_id: ID сессии.
            workspace_id: ID workspace.

        Returns:
            Continuity pack с active episode и recent summaries.
        """
        from memory_core.planner.episode_planner import EpisodePlanner
        
        workspace = str(workspace_id or self._current_workspace or "global").strip() or "global"
        
        planner = EpisodePlanner(
            artifact_store=self.service.artifact_store,
            event_store=self.service.event_store,
        )
        
        # Получаем активный эпизод
        active_episode = planner.get_or_create_episode(
            session_id=session_id,
            workspace_id=workspace,
            topic_thread_id=topic_thread_id,
        )
        
        # Получаем недавние эпизоды для контекста
        recent_episodes = [
            ep
            for ep in planner.get_active_episodes(session_id=session_id)
            if str((ep.metadata or {}).get("topic_thread_id") or "default").strip()
            == str(topic_thread_id or "default").strip()
        ][:3]
        
        return {
            "active_episode": {
                "episode_id": active_episode.episode_id,
                "title": active_episode.title,
                "summary": active_episode.summary,
                "context_tags": active_episode.context_tags,
                "status": active_episode.status,
                "session_id": active_episode.session_id,
                "workspace_id": active_episode.workspace_id,
                "topic_thread_id": str((active_episode.metadata or {}).get("topic_thread_id") or ""),
            },
            "recent_episode_summaries": [
                {
                    "episode_id": ep.episode_id,
                    "title": ep.title,
                    "summary": ep.summary,
                    "context_tags": ep.context_tags,
                }
                for ep in recent_episodes
            ],
            "continuity_hint": "This looks like a follow-up in the same episode." if active_episode else "New episode started.",
        }

    def apply_stabilizer(
        self,
        character_id: str,
        active_profile_snapshot: dict[str, Any] | None = None,
        memory_reasoning_snapshot: dict[str, Any] | None = None,
        session_stats: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Применяет стабилизатор для медленной эволюции личности.

        Args:
            character_id: ID персонажа.
            active_profile_snapshot: Активный профиль из retrieval.
            memory_reasoning_snapshot: Snapshot из memory reasoning.
            session_stats: Статистика сессии.

        Returns:
            Результат стабилизации с promoted patches.
        """
        from modules.character.storage import CharacterStorage
        from modules.character.profile_stabilizer import ProfileStabilizer
        
        storage = CharacterStorage()
        stabilizer = ProfileStabilizer()
        
        # Загружаем текущее состояние
        persona_state = storage.load_persona_state(character_id)
        identity_core = storage.load_identity_core(character_id)
        
        # Принимаем решение о promotion
        decision = stabilizer.decide(
            current_persona_state=persona_state,
            current_identity_core=identity_core,
            active_profile_snapshot=active_profile_snapshot,
            memory_reasoning_snapshot=memory_reasoning_snapshot,
            session_stats=session_stats,
        )
        
        result = {
            "promoted_to_identity_core": decision.promote_to_identity_core,
            "promoted_to_persona_baseline": decision.promote_to_persona_baseline,
            "kept_runtime_only": decision.keep_runtime_only,
            "rejected_count": len(decision.rejected),
            "debug": decision.debug,
        }
        
        # Сохраняем promoted patches если есть
        if decision.promote_to_identity_core:
            # Обновляем identity_core
            updated_identity_core = self._merge_identity_core_updates(
                identity_core,
                decision.promote_to_identity_core,
            )
            storage.save_identity_core(character_id, updated_identity_core)
            result["identity_core_saved"] = True
        
        if decision.promote_to_persona_baseline:
            # Обновляем persona_state.learned.baseline_traits
            updated_persona_state = self._merge_persona_baseline_updates(
                persona_state,
                decision.promote_to_persona_baseline,
            )
            storage.save_persona_state(character_id, updated_persona_state)
            result["persona_baseline_saved"] = True
        
        # Decay counters
        if persona_state.get("stabilizer", {}).get("counters"):
            from datetime import datetime
            stabilizer_state = persona_state.get("stabilizer", {})
            cleaned_counters = stabilizer.decay_counters(stabilizer_state.get("counters", {}))
            
            if cleaned_counters != stabilizer_state.get("counters"):
                persona_state["stabilizer"]["counters"] = cleaned_counters
                persona_state["stabilizer"]["last_decay_at"] = datetime.now().isoformat()
                storage.save_persona_state(character_id, persona_state)
                result["counters_decayed"] = True
        
        return result

    def _merge_identity_core_updates(
        self,
        current_identity_core: dict[str, Any],
        patch: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Объединяет updates в identity_core.

        Args:
            current_identity_core: Текущий identity core.
            patch: Patch для применения.

        Returns:
            Обновлённый identity core.
        """
        updated = dict(current_identity_core)
        
        for key, value in patch.items():
            if key in updated and isinstance(updated[key], dict) and isinstance(value, dict):
                updated[key].update(value)
            else:
                updated[key] = value
        
        return updated

    def _merge_persona_baseline_updates(
        self,
        current_persona_state: dict[str, Any],
        baseline_patch: dict[str, float],
    ) -> dict[str, Any]:
        """
        Объединяет updates в persona baseline.

        Args:
            current_persona_state: Текущее persona state.
            baseline_patch: Patch для baseline_traits.

        Returns:
            Обновлённое persona state.
        """
        updated = dict(current_persona_state)
        learned = dict(updated.get("learned") or {})
        baseline_traits = dict(learned.get("baseline_traits") or {})
        
        # Применяем smoothing
        for key, new_value in baseline_patch.items():
            old_value = baseline_traits.get(key, new_value)
            # Slow drift: old * 0.85 + new * 0.15
            smoothed_value = old_value * 0.85 + new_value * 0.15
            baseline_traits[key] = max(0.0, min(1.0, smoothed_value))
        
        learned["baseline_traits"] = baseline_traits
        updated["learned"] = learned
        
        return updated

    def close(self) -> None:
        """Закрывает background worker, Memory LLM provider и базу данных."""
        self._cancel_auto_resume_timer()
        self._bump_resume_epoch()

        worker = self._get_worker()
        if worker is not None:
            pause = getattr(worker, "pause", None)
            if callable(pause):
                try:
                    pause()
                except Exception as exc:
                    LOGGER.debug(f"MemoryCoreAdapter.close(): worker.pause() failed: {exc}")

            is_running = getattr(worker, "is_running", None)
            stop = getattr(worker, "stop", None)
            if callable(stop):
                try:
                    if callable(is_running) and is_running():
                        stop(timeout_sec=3.0)
                except Exception as exc:
                    LOGGER.debug(f"MemoryCoreAdapter.close(): worker.stop() failed: {exc}")

            shutdown_memory_provider = getattr(worker, "_shutdown_memory_llm_provider", None)
            if callable(shutdown_memory_provider):
                try:
                    shutdown_memory_provider()
                except Exception as exc:
                    LOGGER.debug(f"MemoryCoreAdapter.close(): memory provider shutdown failed: {exc}")
            else:
                memory_llm_processor = getattr(worker, "memory_llm_processor", None)
                shutdown = getattr(memory_llm_processor, "shutdown", None)
                if callable(shutdown):
                    try:
                        shutdown()
                    except Exception as exc:
                        LOGGER.debug(f"MemoryCoreAdapter.close(): memory processor shutdown failed: {exc}")

        try:
            self._release_memory_llm_lock()
        except Exception as exc:
            LOGGER.debug(f"MemoryCoreAdapter.close(): releasing memory lock failed: {exc}")

        db = getattr(getattr(self.service, "event_store", None), "db", None)
        close_db = getattr(db, "close", None)
        if callable(close_db):
            try:
                close_db()
            except Exception as exc:
                LOGGER.debug(f"MemoryCoreAdapter.close(): db.close() failed: {exc}")

    def _bump_resume_epoch(self) -> int:
        if not hasattr(self, "_resume_epoch_lock"):
            self._resume_epoch_lock = threading.Lock()
        if not hasattr(self, "_resume_epoch"):
            self._resume_epoch = 0
        with self._resume_epoch_lock:
            self._resume_epoch += 1
            return self._resume_epoch

    def _resume_epoch_matches(self, epoch: int) -> bool:
        if not hasattr(self, "_resume_epoch_lock"):
            self._resume_epoch_lock = threading.Lock()
        if not hasattr(self, "_resume_epoch"):
            self._resume_epoch = 0
        with self._resume_epoch_lock:
            return int(self._resume_epoch) == int(epoch)

    def _current_resume_epoch(self) -> int:
        if not hasattr(self, "_resume_epoch_lock"):
            self._resume_epoch_lock = threading.Lock()
        if not hasattr(self, "_resume_epoch"):
            self._resume_epoch = 0
        with self._resume_epoch_lock:
            return int(self._resume_epoch)

    def _mark_memory_drained_once(self, epoch: int) -> bool:
        if not hasattr(self, "_memory_drain_lock"):
            self._memory_drain_lock = threading.Lock()
        with self._memory_drain_lock:
            if getattr(self, "_memory_drained_epoch", None) == int(epoch):
                return False
            self._memory_drained_epoch = int(epoch)
            return True

    def _reset_memory_drain_epoch(self) -> None:
        if not hasattr(self, "_memory_drain_lock"):
            self._memory_drain_lock = threading.Lock()
        with self._memory_drain_lock:
            self._memory_drained_epoch = None

    def _on_memory_worker_queue_idle(self) -> None:
        epoch = self._current_resume_epoch()
        if not self._resume_epoch_matches(epoch):
            return
        if self._pending_memory_job_count() > 0:
            return
        self._on_memory_jobs_drained(epoch)

    def _get_worker(self):
        service = getattr(self, "service", None)
        if service is not None and hasattr(service, 'worker') and service.worker is not None:
            return service.worker
        return None

    def _pending_memory_job_count(self) -> int:
        worker = self._get_worker()
        if worker is None:
            return 0
        job_queue = getattr(worker, "job_queue", None)
        get_stats = getattr(job_queue, "get_stats", None)
        if not callable(get_stats):
            return 0
        try:
            stats = dict(get_stats() or {})
        except Exception:
            return 0
        by_type = dict(stats.get("by_type") or {})
        try:
            return max(0, int(by_type.get("memory_llm_process") or 0))
        except Exception:
            return 0

    def _complete_memory_resume(self, epoch: int) -> None:
        if not self._resume_epoch_matches(epoch):
            return
        self._release_memory_llm_lock()
        worker = self._get_worker()
        if worker is None:
            return
        wake = getattr(worker, "wake", None)
        if callable(wake):
            wake()

    def _resume_worker_after_memory_ready(self, epoch: int) -> None:
        if not self._resume_epoch_matches(epoch):
            return
        self._release_memory_llm_lock()
        worker = self._get_worker()
        if worker is None:
            return
        try:
            if hasattr(worker, "is_running") and not worker.is_running():
                LOGGER.info("MemoryCoreAdapter: Worker is stopped, starting it...")
                worker.start()
                LOGGER.info("MemoryCoreAdapter: Worker started")
            else:
                LOGGER.info("MemoryCoreAdapter: Resuming worker after Memory LLM is ready...")
                resume_idle = getattr(worker, "resume_idle", None)
                if callable(resume_idle):
                    resume_idle()
                else:
                    worker.resume()
                LOGGER.info("MemoryCoreAdapter: Worker resumed")
        except Exception as exc:
            LOGGER.warning(f"MemoryCoreAdapter: Failed to resume/start worker: {exc}")
        wake = getattr(worker, "wake", None)
        if callable(wake):
            wake()

    def _schedule_memory_prewarm(self, epoch: int) -> bool:
        worker = self._get_worker()
        if worker is None:
            return False
        if self._pending_memory_job_count() <= 0:
            return False

        memory_llm_processor = getattr(worker, "memory_llm_processor", None)
        warmup = getattr(memory_llm_processor, "warmup", None)
        if not callable(warmup):
            return False

        def _run_prewarm() -> None:
            try:
                warmup()
            except Exception as exc:
                LOGGER.debug(f"MemoryCoreAdapter: Memory LLM prewarm failed: {exc}")
            finally:
                self._resume_worker_after_memory_ready(epoch)

        thread = threading.Thread(
            target=_run_prewarm,
            name="memory-llm-prewarm",
            daemon=True,
        )
        thread.start()
        LOGGER.info("MemoryCoreAdapter: Memory LLM prewarm scheduled")
        return True

    def _schedule_main_sleep_unload(self, epoch: int, delay_sec: float) -> None:
        delay = max(0.0, float(delay_sec or 0.0))
        if delay <= 0:
            return

        def _sleep_deadline() -> None:
            self._main_sleep_timer = None
            if not self._resume_epoch_matches(epoch):
                return
            self._main_sleep_deadline_at = 0.0
            LOGGER.info("MemoryCoreAdapter: main LLM sleep deadline reached, unloading main model")
            self._unload_main_model_from_vram()
            auto_resume_timer = getattr(self, "_auto_resume_timer", None)
            if auto_resume_timer is not None:
                auto_resume_timer.cancel()
                self._auto_resume_timer = None
            if bool(getattr(self, "_main_lease_holds_memory", False)):
                self._main_lease_holds_memory = False
                self._resume_worker_now(epoch)

        self._main_sleep_timer = threading.Timer(delay, _sleep_deadline)
        self._main_sleep_timer.daemon = True
        self._main_sleep_timer.start()
        LOGGER.info("MemoryCoreAdapter: main LLM sleep deadline scheduled for %.1fs", delay)

    def _schedule_delayed_memory_resume(self, epoch: int, delay_sec: float) -> None:
        delay = max(0.0, float(delay_sec or 0.0))

        def _delayed_resume() -> None:
            self._auto_resume_timer = None
            if not self._resume_epoch_matches(epoch):
                return
            self._resume_worker_now(epoch)

        self._auto_resume_timer = threading.Timer(delay, _delayed_resume)
        self._auto_resume_timer.daemon = True
        self._auto_resume_timer.start()
        LOGGER.info("MemoryCoreAdapter: Memory LLM handoff delayed for %.1fs", delay)

    def _schedule_memory_drain_monitor(self, epoch: int) -> None:
        def _monitor() -> None:
            while self._resume_epoch_matches(epoch):
                if self._pending_memory_job_count() <= 0:
                    self._on_memory_jobs_drained(epoch)
                    return
                if self._main_sleep_remaining_sec() <= 0:
                    return
                time.sleep(0.25)

        thread = threading.Thread(
            target=_monitor,
            name="memory-llm-drain-monitor",
            daemon=True,
        )
        thread.start()

    def _on_memory_jobs_drained(self, epoch: int) -> None:
        if not self._resume_epoch_matches(epoch):
            return
        if self._pending_memory_job_count() > 0:
            return
        remaining = self._main_sleep_remaining_sec()
        if remaining <= 0:
            LOGGER.info("MemoryCoreAdapter: memory jobs drained after main sleep deadline; main stays unloaded")
            return
        if not self._mark_memory_drained_once(epoch):
            return

        LOGGER.info(
            "MemoryCoreAdapter: memory jobs drained with %.1fs main lease left; returning main LLM",
            remaining,
        )
        self._hold_memory_for_main_lease()
        self._unload_memory_llm()
        self._warm_main_model_in_vram(remaining)

    def _hold_memory_for_main_lease(self) -> None:
        self._main_lease_holds_memory = True
        self._acquire_memory_llm_lock()
        worker = self._get_worker()
        if worker is None:
            return
        pause = getattr(worker, "pause", None)
        if callable(pause):
            try:
                pause()
            except Exception as exc:
                LOGGER.debug(f"MemoryCoreAdapter: failed to pause worker for main lease: {exc}")

    def _resume_worker_idle_without_memory_llm(self, resume_epoch: int) -> None:
        if self._main_sleep_remaining_sec() > 0:
            self._main_lease_holds_memory = True
            LOGGER.info(
                "MemoryCoreAdapter: no pending Memory LLM jobs; main lease keeps memory paused until sleep deadline"
            )
            return

        self._main_lease_holds_memory = False
        worker = self._get_worker()
        if worker is not None:
            try:
                is_running = getattr(worker, "is_running", None)
                running = bool(is_running()) if callable(is_running) else True
                if running:
                    resume_idle = getattr(worker, "resume_idle", None)
                    if callable(resume_idle):
                        resume_idle()
                    else:
                        pause_event = getattr(worker, "_pause_event", None)
                        clear = getattr(pause_event, "clear", None)
                        if callable(clear):
                            clear()
                else:
                    LOGGER.debug("MemoryCoreAdapter: no pending memory jobs; stopped worker stays stopped")
            except Exception as exc:
                LOGGER.debug(f"MemoryCoreAdapter: failed to resume idle worker: {exc}")

        LOGGER.info(
            "MemoryCoreAdapter: no pending Memory LLM jobs; Memory LLM stays unloaded and main keep_alive keeps priority"
        )
        self._complete_memory_resume(resume_epoch)

    def _resume_worker_now(self, resume_epoch: int) -> None:
        pending_jobs = self._pending_memory_job_count()
        if pending_jobs <= 0:
            self._resume_worker_idle_without_memory_llm(resume_epoch)
            return

        self._main_lease_holds_memory = False
        self._unload_main_model_from_vram()
        self._resume_memory_llm()
        self._schedule_memory_drain_monitor(resume_epoch)
        if pending_jobs > 0 and self._schedule_memory_prewarm(resume_epoch):
            return

        self._resume_worker_after_memory_ready(resume_epoch)

    def pause_worker(self) -> None:
        """
        Ставит background worker на паузу и уступает управление Memory LLM.
        
        Полезно для приостановки обработки памяти во время ответа пользователю.
        Работает только если enable_worker_pause_during_api_request=True в конфиге.
        """
        if not self._enable_pause:
            return  # Пауза отключена в конфиге
        
        # Отменяем предыдущий таймер если есть
        self._cancel_auto_resume_timer()
        self._bump_resume_epoch()
        
        # Сначала блокируем Memory LLM
        self._acquire_memory_llm_lock()
        
        if hasattr(self.service, 'worker') and self.service.worker is not None:
            LOGGER.info("MemoryCoreAdapter: Pausing worker...")
            self.service.worker.pause()
            LOGGER.info("MemoryCoreAdapter: Worker paused")
        
        # В cooperative lifecycle pause/yield не выгружает модель из VRAM.
        if bool(getattr(self, "_unload_memory_llm_before_main_request", False)):
            self._unload_memory_llm()
        else:
            self._pause_memory_llm()
        
        # Важно: не используем auto-resume timer во время активного ответа API.
        # Таймер мог сработать посреди генерации основной LLM и вернуть memory worker
        # слишком рано, что ломало приоритет основной модели и приводило к гонкам.

    def _auto_resume_worker(self) -> None:
        """Автоматически возобновляет worker после таймаута."""
        LOGGER.info("MemoryCoreAdapter: Auto-resume timer triggered, resuming worker...")
        self.resume_worker()

    def _cancel_auto_resume_timer(self) -> None:
        """Отменяет таймер автоматического возобновления."""
        auto_resume_timer = getattr(self, "_auto_resume_timer", None)
        if auto_resume_timer is not None:
            auto_resume_timer.cancel()
            self._auto_resume_timer = None
            LOGGER.debug("MemoryCoreAdapter: Auto-resume timer cancelled")
        if getattr(self, "_main_sleep_timer", None) is not None:
            self._main_sleep_timer.cancel()
            self._main_sleep_timer = None
            LOGGER.debug("MemoryCoreAdapter: Main sleep timer cancelled")
        self._main_sleep_deadline_at = 0.0
        self._main_lease_holds_memory = False

    def resume_worker(self) -> None:
        """
        Снимает background worker с паузы и возобновляет Memory LLM.
        
        Возобновляет обработку задач памяти.
        """
        if not self._enable_pause:
            return  # Пауза отключена в конфиге
        
        # Отменяем таймер если ещё активен
        self._cancel_auto_resume_timer()
        resume_epoch = self._bump_resume_epoch()
        self._reset_memory_drain_epoch()
        main_sleep_delay_sec = self._main_model_keep_alive_delay_sec()
        if main_sleep_delay_sec > 0:
            self._main_sleep_deadline_at = time.monotonic() + main_sleep_delay_sec
            self._main_lease_holds_memory = True
            self._schedule_main_sleep_unload(resume_epoch, main_sleep_delay_sec)
        else:
            self._main_sleep_deadline_at = 0.0
            self._main_lease_holds_memory = False

        handoff_delay_sec = self._memory_handoff_delay_sec()
        if handoff_delay_sec > 0:
            self._schedule_delayed_memory_resume(resume_epoch, handoff_delay_sec)
            return

        self._resume_worker_now(resume_epoch)

    def _runtime_main_provider(self):
        try:
            from api.app import _runtime

            return getattr(_runtime, "provider", None)
        except Exception:
            return None

    def _runtime_main_model_name(self) -> str:
        try:
            from api.app import _runtime

            model = str(getattr(_runtime, "model", "") or "").strip()
            if model:
                return model
        except Exception:
            pass
        try:
            provider = self._runtime_main_provider()
            model = str(getattr(provider, "_current_model", "") or getattr(provider, "default_model", "") or "").strip()
            if model:
                return model
        except Exception:
            pass
        try:
            from config.settings import load_config

            return str(getattr(load_config(), "model_name", "") or "").strip()
        except Exception:
            return ""

    def _unload_main_model_from_vram(self) -> None:
        """Unload the main LLM when memory needs VRAM or the main sleep deadline expires."""
        try:
            provider = self._runtime_main_provider()
            unload = getattr(provider, "unload_model", None)
            if not callable(unload):
                unload = getattr(provider, "unload", None)
            if callable(unload):
                LOGGER.info("MemoryCoreAdapter: Unloading main model from VRAM...")
                unload()
                LOGGER.info("MemoryCoreAdapter: Main model unloaded from VRAM")
        except Exception as exc:
            LOGGER.debug(f"Failed to unload main model from VRAM: {exc}")

    def _warm_main_model_in_vram(self, keep_alive_remaining_sec: float) -> bool:
        """Warm the main LLM back while its post-answer sleep timer is still alive."""
        remaining = max(0.0, float(keep_alive_remaining_sec or 0.0))
        if remaining <= 0:
            return False
        try:
            provider = self._runtime_main_provider()
            warmup = getattr(provider, "warmup", None)
            if not callable(warmup):
                return False
            keep_alive = f"{max(1, int(round(remaining)))}s"
            model = self._runtime_main_model_name()
            options = self._main_model_warmup_options()
            LOGGER.info(
                "MemoryCoreAdapter: Warming main model back model=%s keep_alive=%s options=%s",
                model,
                keep_alive,
                options,
            )
            try:
                return bool(warmup(keep_alive=keep_alive, model=model or None, options=options))
            except TypeError:
                try:
                    return bool(warmup(keep_alive=keep_alive))
                except TypeError:
                    return bool(warmup())
        except Exception as exc:
            LOGGER.debug(f"Failed to warm main model in VRAM: {exc}")
            return False

    def _acquire_memory_llm_lock(self) -> None:
        """Блокирует Memory LLM для предотвращения обработки."""
        try:
            # СНАЧАЛА устанавливаем флаг прерывания — это немедленно!
            _mark_memory_llm_interrupt()
            try:
                from llm.ollama_provider import _memory_llm_interrupt
                _memory_llm_interrupt.set()
                LOGGER.debug("MemoryCoreAdapter: Memory LLM interrupt flag set (IMMEDIATE)")
            except Exception:
                pass
            
            # ЗАТЕМ захватываем lock — это предотвратит новые задачи
            acquired = _memory_llm_lock.acquire(timeout=0.1)
            if acquired:
                LOGGER.debug("MemoryCoreAdapter: Memory LLM lock acquired")
        except Exception as exc:
            LOGGER.debug(f"Failed to acquire Memory LLM lock: {exc}")

    def _release_memory_llm_lock(self) -> None:
        """Разблокирует Memory LLM."""
        try:
            # СНАЧАЛА освобождаем lock
            if _memory_llm_lock.locked():
                _memory_llm_lock.release()
                LOGGER.debug("MemoryCoreAdapter: Memory LLM lock released")
            
            # ЗАТЕМ сбрасываем флаг прерывания
            try:
                from llm.ollama_provider import _memory_llm_interrupt
                _memory_llm_interrupt.clear()
                LOGGER.debug("MemoryCoreAdapter: Memory LLM interrupt flag cleared")
            except Exception:
                pass
        except Exception as exc:
            LOGGER.debug(f"Failed to release Memory LLM lock: {exc}")

    def _pause_memory_llm(self) -> None:
        """Yield Memory LLM without unloading it."""
        try:
            # Получаем memory_llm_processor из worker
            if hasattr(self.service, 'worker') and self.service.worker is not None:
                worker = self.service.worker
                memory_llm_processor = getattr(worker, 'memory_llm_processor', None)
                yield_control = getattr(memory_llm_processor, 'yield_control', None)
                if callable(yield_control):
                    LOGGER.info("MemoryCoreAdapter: Yielding Memory LLM...")
                    yield_control()
                    LOGGER.info("MemoryCoreAdapter: Memory LLM yielded")
                    return
                if memory_llm_processor:
                    # Получаем task_router и provider
                    task_router = getattr(memory_llm_processor, 'task_router', None)
                    if task_router:
                        # Получаем provider из task_router
                        provider = getattr(task_router, '_provider', None)
                        if provider and hasattr(provider, 'yield_control'):
                            LOGGER.info("MemoryCoreAdapter: Yielding Memory LLM...")
                            provider.yield_control()
                            LOGGER.info("MemoryCoreAdapter: Memory LLM yielded")
        except Exception as exc:
            LOGGER.debug(f"Failed to pause Memory LLM: {exc}")

    def _unload_memory_llm(self) -> None:
        """Unload Memory LLM so the main LLM can answer with minimal contention."""
        try:
            worker = self._get_worker()
            if worker is None:
                return

            unload_worker = getattr(worker, "_unload_memory_llm_provider", None)
            if callable(unload_worker):
                LOGGER.info("MemoryCoreAdapter: Unloading Memory LLM before main request...")
                unload_worker()
                LOGGER.info("MemoryCoreAdapter: Memory LLM unloaded before main request")
                return

            memory_llm_processor = getattr(worker, "memory_llm_processor", None)
            unload = getattr(memory_llm_processor, "unload", None)
            if callable(unload):
                LOGGER.info("MemoryCoreAdapter: Unloading Memory LLM before main request...")
                unload()
                try:
                    setattr(worker, "_memory_llm_provider_unloaded", True)
                except Exception:
                    pass
                LOGGER.info("MemoryCoreAdapter: Memory LLM unloaded before main request")
        except Exception as exc:
            LOGGER.debug(f"Failed to unload Memory LLM: {exc}")

    def _resume_memory_llm(self) -> None:
        """Возобновляет Memory LLM после паузы."""
        try:
            if hasattr(self.service, 'worker') and self.service.worker is not None:
                worker = self.service.worker
                memory_llm_processor = getattr(worker, 'memory_llm_processor', None)
                resume = getattr(memory_llm_processor, 'resume', None)
                if callable(resume):
                    LOGGER.info("MemoryCoreAdapter: Resuming Memory LLM...")
                    resume()
                    LOGGER.info("MemoryCoreAdapter: Memory LLM resumed")
                    return
                if memory_llm_processor:
                    task_router = getattr(memory_llm_processor, 'task_router', None)
                    if task_router:
                        provider = getattr(task_router, '_provider', None)
                        if provider and hasattr(provider, 'resume'):
                            LOGGER.info("MemoryCoreAdapter: Resuming Memory LLM...")
                            provider.resume()
                            LOGGER.info("MemoryCoreAdapter: Memory LLM resumed")
        except Exception as exc:
            LOGGER.debug(f"Failed to resume Memory LLM: {exc}")

    def is_worker_paused(self) -> bool:
        """Проверяет, на паузе ли worker."""
        if not self._enable_pause:
            return False
        if hasattr(self.service, 'worker') and self.service.worker is not None:
            return self.service.worker.is_paused()
        return False
    
    @property
    def worker_pause_enabled(self) -> bool:
        """Проверяет, включена ли пауза worker."""
        return self._enable_pause
    
    @property
    def worker_pause_timeout(self) -> float:
        """Получает таймаут паузы worker (сек)."""
        return self._pause_timeout


# Глобальный экземпляр для использования в main.py
_memory_core_adapter: MemoryCoreAdapter | None = None


def get_memory_core_adapter() -> MemoryCoreAdapter:
    """Получает глобальный экземпляр адаптера."""
    global _memory_core_adapter
    if _memory_core_adapter is None:
        _memory_core_adapter = MemoryCoreAdapter()
    return _memory_core_adapter


def init_memory_core(
    db_path: str = "data/memory_core/memory.db",
    vector_path: str = "data/memory_core/vector",
    **kwargs,
) -> MemoryCoreAdapter:
    """
    Инициализирует глобальный экземпляр memory_core.

    Args:
        db_path: Путь к базе данных.
        vector_path: Путь к векторному индексу.
        **kwargs: Дополнительные аргументы для MemoryCoreAdapter.

    Returns:
        Инициализированный адаптер.
    """
    global _memory_core_adapter
    _memory_core_adapter = MemoryCoreAdapter(
        db_path=db_path,
        vector_path=vector_path,
        **kwargs,
    )
    return _memory_core_adapter
