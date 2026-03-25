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
import threading


LOGGER = get_logger(__name__)

# Глобальная блокировка для Memory LLM
# Используется для приостановки Memory LLM во время ответа основной модели
_memory_llm_lock = threading.Lock()


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
        
        # Настройки паузы worker из конфига
        self._enable_pause = bool(config_file.enable_worker_pause_during_api_request)
        self._pause_timeout = float(config_file.worker_pause_timeout or 0.0)
        
        # Таймер для автоматического возобновления Memory LLM
        self._auto_resume_timer: threading.Timer | None = None
        self._auto_resume_delay = float(config_file.worker_pause_timeout or 0.0)

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
        )

        result = self.service.query(query)

        return {
            "context_blocks": result.context_blocks,
            "hits": result.hits,
            "citations": result.citations,
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
        )

        selected = []
        for hit in result.get("hits", []):
            selected.append({
                "artifact_id": hit.get("artifact_id"),
                "artifact_type": hit.get("artifact_type"),
                "text": hit.get("text"),
                "score": hit.get("score", 1.0),
            })

        return {
            "selected": selected,
            "blocks": {"context_blocks": result.get("context_blocks", [])},
            "recall_mode": "memory_core_query",
            "citations": result.get("citations", []),
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

    def inspect(self, kind: str = "events", limit: int = 50) -> dict[str, Any]:
        """
        Инспектирует память.

        Args:
            kind: Тип инспекции (events | artifacts | workspaces | profile | stats).
            limit: Максимальное количество результатов.

        Returns:
            Результат инспекции.
        """
        from memory_core.schemas import MemoryInspectRequest

        request = MemoryInspectRequest(kind=kind, limit=limit)
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
        )
        
        # Получаем недавние эпизоды для контекста
        recent_episodes = planner.get_active_episodes(session_id=session_id)[:3]
        
        return {
            "active_episode": {
                "episode_id": active_episode.episode_id,
                "title": active_episode.title,
                "summary": active_episode.summary,
                "context_tags": active_episode.context_tags,
                "status": active_episode.status,
                "session_id": active_episode.session_id,
                "workspace_id": active_episode.workspace_id,
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
        """Закрывает соединения (если требуется)."""
        # SQLite не требует явного закрытия в большинстве случаев
        pass

    def pause_worker(self) -> None:
        """
        Ставит background worker на паузу и освобождает VRAM.
        
        Полезно для приостановки обработки памяти во время ответа пользователю.
        Работает только если enable_worker_pause_during_api_request=True в конфиге.
        """
        if not self._enable_pause:
            return  # Пауза отключена в конфиге
        
        # Отменяем предыдущий таймер если есть
        self._cancel_auto_resume_timer()
        
        # Сначала блокируем Memory LLM
        self._acquire_memory_llm_lock()
        
        if hasattr(self.service, 'worker') and self.service.worker is not None:
            LOGGER.info("MemoryCoreAdapter: Pausing worker...")
            self.service.worker.pause()
            LOGGER.info("MemoryCoreAdapter: Worker paused")
        
        # Также ставим на паузу Memory LLM для освобождения VRAM
        self._pause_memory_llm()
        
        # Планируем автоматическое возобновление через worker_pause_timeout
        # Если timeout=0, то возобновление только после явного вызова resume_worker()
        pause_timeout = self._pause_timeout
        if pause_timeout > 0:
            LOGGER.info(f"MemoryCoreAdapter: Auto-resume timer set for {pause_timeout}s")
            self._auto_resume_timer = threading.Timer(
                pause_timeout,
                self._auto_resume_worker
            )
            self._auto_resume_timer.daemon = True
            self._auto_resume_timer.start()

    def _auto_resume_worker(self) -> None:
        """Автоматически возобновляет worker после таймаута."""
        LOGGER.info("MemoryCoreAdapter: Auto-resume timer triggered, resuming worker...")
        self.resume_worker()

    def _cancel_auto_resume_timer(self) -> None:
        """Отменяет таймер автоматического возобновления."""
        if self._auto_resume_timer is not None:
            self._auto_resume_timer.cancel()
            self._auto_resume_timer = None
            LOGGER.debug("MemoryCoreAdapter: Auto-resume timer cancelled")

    def resume_worker(self) -> None:
        """
        Снимает background worker с паузы и возобновляет Memory LLM.
        
        Возобновляет обработку задач памяти.
        """
        if not self._enable_pause:
            return  # Пауза отключена в конфиге
        
        # Отменяем таймер если ещё активен
        self._cancel_auto_resume_timer()
        
        # СНАЧАЛА выгружаем модель основной модели из VRAM
        self._unload_main_model_from_vram()
        
        if hasattr(self.service, 'worker') and self.service.worker is not None:
            LOGGER.info("MemoryCoreAdapter: Resuming worker...")
            self.service.worker.resume()
            LOGGER.info("MemoryCoreAdapter: Worker resumed")
        
        # Также возобновляем Memory LLM
        self._resume_memory_llm()
        
        # Разблокируем Memory LLM
        self._release_memory_llm_lock()

    def _unload_main_model_from_vram(self) -> None:
        """Выгружает модель основной модели из VRAM для освобождения памяти."""
        try:
            # Получаем provider из _runtime (основная модель)
            from api.app import _runtime
            provider = getattr(_runtime, "provider", None)
            if provider and hasattr(provider, 'unload_model'):
                LOGGER.info("MemoryCoreAdapter: Unloading main model from VRAM...")
                provider.unload_model()
                LOGGER.info("MemoryCoreAdapter: Main model unloaded from VRAM")
        except Exception as exc:
            LOGGER.debug(f"Failed to unload main model from VRAM: {exc}")

    def _acquire_memory_llm_lock(self) -> None:
        """Блокирует Memory LLM для предотвращения обработки."""
        try:
            # СНАЧАЛА устанавливаем флаг прерывания — это немедленно!
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
        """Ставит Memory LLM на паузу для освобождения VRAM."""
        try:
            # Получаем memory_llm_processor из worker
            if hasattr(self.service, 'worker') and self.service.worker is not None:
                worker = self.service.worker
                memory_llm_processor = getattr(worker, 'memory_llm_processor', None)
                if memory_llm_processor:
                    # Получаем task_router и provider
                    task_router = getattr(memory_llm_processor, 'task_router', None)
                    if task_router:
                        # Получаем provider из task_router
                        provider = getattr(task_router, '_provider', None)
                        if provider and hasattr(provider, 'pause'):
                            LOGGER.info("MemoryCoreAdapter: Pausing Memory LLM...")
                            provider.pause()
                            LOGGER.info("MemoryCoreAdapter: Memory LLM paused")
        except Exception as exc:
            LOGGER.debug(f"Failed to pause Memory LLM: {exc}")

    def _resume_memory_llm(self) -> None:
        """Возобновляет Memory LLM после паузы."""
        try:
            if hasattr(self.service, 'worker') and self.service.worker is not None:
                worker = self.service.worker
                memory_llm_processor = getattr(worker, 'memory_llm_processor', None)
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
