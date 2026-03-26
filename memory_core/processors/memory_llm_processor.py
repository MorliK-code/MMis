"""
Memory LLM Processor - процессор памяти на основе LLM.

Анализирует сырые события и генерирует предложения (proposals) по созданию,
обновлению или удалению артефактов памяти.

Типы артефактов:
- profile_fact: стабильный факт о пользователе
- preference: предпочтение
- episode_event: событие эпизода
- task_state: состояние задачи
- emotional_state: эмоциональное состояние
- identity_core: ядро личности (максимальный приоритет)
"""

from __future__ import annotations

import json
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from memory_core.schemas import MemoryEnvelope
from llm.task_router import TaskModelRouter
from utils.logger import get_logger


LOGGER = get_logger(__name__)

# Импортируем глобальную блокировку из adapter
_memory_llm_lock: threading.Lock | None = None
try:
    from memory_core.adapter import _memory_llm_lock
except ImportError:
    pass

# Импортируем флаг прерывания из ollama_provider
_memory_llm_interrupt: threading.Event | None = None
try:
    from llm.ollama_provider import _memory_llm_interrupt
except ImportError:
    pass


def _get_memory_llm_lock():
    if _memory_llm_lock is not None:
        return _memory_llm_lock
    try:
        from memory_core.adapter import _memory_llm_lock as runtime_lock
        return runtime_lock
    except Exception:
        return None


def _get_memory_llm_interrupt():
    if _memory_llm_interrupt is not None:
        return _memory_llm_interrupt
    try:
        from llm.ollama_provider import _memory_llm_interrupt as runtime_interrupt
        return runtime_interrupt
    except Exception:
        return None


def _get_memory_llm_interrupt_epoch() -> int:
    try:
        from memory_core.adapter import _get_memory_llm_interrupt_epoch as runtime_interrupt_epoch
        return int(runtime_interrupt_epoch())
    except Exception:
        return 0


def _get_memory_llm_preemption(*, since_epoch: int | None = None) -> InterruptedError | None:
    if since_epoch is not None and _get_memory_llm_interrupt_epoch() > int(since_epoch):
        return InterruptedError("Memory LLM interrupted by main model")

    memory_llm_lock = _get_memory_llm_lock()
    if memory_llm_lock is not None and memory_llm_lock.locked():
        return InterruptedError("Memory LLM paused by main model")

    memory_llm_interrupt = _get_memory_llm_interrupt()
    if memory_llm_interrupt is not None and memory_llm_interrupt.is_set():
        return InterruptedError("Memory LLM interrupted by main model")

    return None


@dataclass(slots=True)
class ArtifactProposal:
    """
    Предложение по созданию/обновлению артефакта.
    """
    artifact_type: str  # profile_fact | preference | episode_event | task_state | emotional_state | identity_core
    text: str
    summary: str
    confidence: float  # 0.0 - 1.0
    scope: str  # profile | episode | task | global
    decay: str  # none | slow | fast | immediate
    retrieve_when: list[str]  # теги для retrieval
    action: str  # create | update | merge | ignore | supersede
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Преобразует в словарь."""
        return {
            "artifact_type": self.artifact_type,
            "text": self.text,
            "summary": self.summary,
            "confidence": self.confidence,
            "scope": self.scope,
            "decay": self.decay,
            "retrieve_when": self.retrieve_when,
            "action": self.action,
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class MemoryLLMResult:
    """
    Результат обработки Memory LLM.
    """
    event_id: str
    importance: float  # 0.0 - 1.0
    should_process: bool
    proposals: list[ArtifactProposal]
    raw_response: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Преобразует в словарь."""
        return {
            "event_id": self.event_id,
            "importance": self.importance,
            "should_process": self.should_process,
            "proposals": [p.to_dict() for p in self.proposals],
            "raw_response": self.raw_response,
        }


# Системный промпт для Memory LLM (упрощённый)
MEMORY_LLM_SYSTEM_PROMPT = """Ты — Memory LLM процессор. Анализируй события и возвращай JSON.

Формат ответа:
{"event_id": "ID", "importance": 0.5, "should_process": true, "proposals": []}

Типы: profile_fact, preference, episode_event, task_state, emotional_state, identity_core
Действия: create, update, merge, ignore, supersede

Пример proposal:
{"artifact_type": "profile_fact", "text": "Факт", "summary": "Кратко", "confidence": 0.8, "scope": "profile", "decay": "none", "retrieve_when": ["tag"], "action": "create"}

Возвращай ТОЛЬКО JSON. Если нет предложений — proposals: []."""


class MemoryLLMProcessor:
    """
    Процессор памяти на основе LLM.
    """
    TASK_NAME = "memory_llm_process"

    def __init__(
        self,
        task_router: TaskModelRouter | None = None,
        model_profile: str = "memory_llm",
        timeout_sec: float = 60.0,
        keep_alive: Any | None = None,
    ):
        """
        Инициализирует процессор.

        Args:
            task_router: Маршрутизатор задач LLM.
            model_profile: Профиль модели для memory_llm.
            timeout_sec: Таймаут для LLM вызова (сек).
        """
        self.task_router = task_router
        self.model_profile = model_profile
        self.timeout_sec = float(timeout_sec or 60.0)
        self.keep_alive = keep_alive

    def process(self, envelope: MemoryEnvelope) -> MemoryLLMResult:
        """
        Обрабатывает событие и генерирует предложения.

        Args:
            envelope: Конверт события.

        Returns:
            Результат обработки с предложениями.
        """
        # Проверяем глобальную блокировку Memory LLM
        # Если блокировка установлена — основная модель отвечает, возвращаем задачу в очередь
        preemption = _get_memory_llm_preemption()
        if preemption is not None:
            LOGGER.debug(f"MemoryLLMProcessor: preempted before processing event {envelope.event_id[:8]}...")
            raise preemption
        
        # Быстрый фильтр: пропускаем неважные события
        if self._should_skip(envelope):
            return MemoryLLMResult(
                event_id=envelope.event_id,
                importance=0.0,
                should_process=False,
                proposals=[],
            )

        # Формируем промпт
        prompt = self._build_prompt(envelope)
        interrupt_epoch = _get_memory_llm_interrupt_epoch()

        # Вызываем LLM с таймаутом
        try:
            response = self._call_llm(prompt, timeout_sec=self.timeout_sec, interrupt_epoch=interrupt_epoch)
        except InterruptedError:
            LOGGER.debug(f"MemoryLLMProcessor: Event {envelope.event_id[:8]}... interrupted, requeueing")
            raise

        # Парсим ответ
        result = self._parse_response(envelope.event_id, response, interrupt_epoch=interrupt_epoch)

        return result

    def _should_skip(self, envelope: MemoryEnvelope) -> bool:
        """
        Проверяет, можно ли пропустить событие (Fast Path).

        Args:
            envelope: Конверт события.

        Returns:
            True если событие можно пропустить.
        """
        # Пропускаем пустые
        if not envelope.text or len(envelope.text.strip()) < 5:
            return True

        # Пропускаем технические сообщения
        skip_kinds = {"system", "tool"}
        if envelope.source_kind.lower() in skip_kinds:
            # Но не пропускаем важные tool results
            if envelope.payload_type == "tool_result" and len(envelope.text) > 50:
                return False
            return True

        # Пропускаем команды
        if envelope.text.strip().startswith("/"):
            return True

        return False

    def _build_prompt(self, envelope: MemoryEnvelope) -> str:
        """
        Строит промпт для LLM.

        Args:
            envelope: Конверт события.

        Returns:
            Промпт для LLM.
        """
        metadata_str = json.dumps(envelope.metadata, ensure_ascii=False, indent=2)

        prompt = f"""
Событие:
- ID: {envelope.event_id}
- Тип источника: {envelope.source_kind}
- Тип контента: {envelope.payload_type}
- Workspace: {envelope.workspace_id}
- Session: {envelope.session_id}
- Metadata: {metadata_str}

Текст:
{envelope.text}

Проанализируй событие и предложи артефакты для памяти.
"""
        return prompt

    def _call_llm(
        self,
        prompt: str,
        timeout_sec: float | None = None,
        interrupt_epoch: int | None = None,
    ) -> str:
        """
        Вызывает LLM для обработки.

        Args:
            prompt: Промпт для LLM.
            timeout_sec: Таймаут для LLM вызова (сек).

        Returns:
            Ответ LLM.
        """
        timeout = float(timeout_sec or self.timeout_sec or 60.0)
        if interrupt_epoch is None:
            interrupt_epoch = _get_memory_llm_interrupt_epoch()
        
        if self.task_router:
            try:
                # Проверяем прерывание ПЕРЕД вызовом LLM
                preemption = _get_memory_llm_preemption(since_epoch=interrupt_epoch)
                if preemption is not None:
                    LOGGER.debug("MemoryLLMProcessor: Interrupted before LLM call")
                    raise preemption

                metadata: dict[str, Any] = {
                    "source": self.TASK_NAME,
                    "think": False,
                }
                keep_alive = self.keep_alive
                if keep_alive is not None and str(keep_alive).strip() != "":
                    metadata["keep_alive"] = keep_alive
                
                result_queue: queue.Queue[tuple[str, Any]] = queue.Queue(maxsize=1)

                def _invoke_task_router() -> None:
                    try:
                        run_task_model_json = getattr(self.task_router, "run_task_model_json", None)
                        if callable(run_task_model_json):
                            result = run_task_model_json(
                                task_name=self.TASK_NAME,
                                prompt=prompt,
                                system_prompt=MEMORY_LLM_SYSTEM_PROMPT,
                                metadata=metadata,
                                timeout=timeout,
                                required_fields=("event_id", "importance", "should_process", "proposals"),
                                allow_array=False,
                                allow_fallback=False,
                            )
                        else:
                            result = self.task_router.run_task_model(
                                task_name=self.TASK_NAME,
                                prompt=prompt,
                                system_prompt=MEMORY_LLM_SYSTEM_PROMPT,
                                metadata=metadata,
                                timeout=timeout,
                                allow_fallback=False,
                            )
                    except Exception as exc:
                        result_queue.put(("error", exc))
                        return
                    result_queue.put(("ok", result))

                call_thread = threading.Thread(
                    target=_invoke_task_router,
                    name=f"memory-llm-call-{self.TASK_NAME}",
                    daemon=True,
                )
                call_thread.start()
                deadline_at = time.perf_counter() + max(0.1, timeout)

                while call_thread.is_alive():
                    preemption = _get_memory_llm_preemption(since_epoch=interrupt_epoch)
                    if preemption is not None:
                        LOGGER.warning("MemoryLLMProcessor: preempted while waiting for TaskModelRouter")
                        raise preemption
                    remaining = deadline_at - time.perf_counter()
                    if remaining <= 0.0:
                        break
                    call_thread.join(timeout=min(0.1, max(0.01, remaining)))

                if call_thread.is_alive():
                    LOGGER.error(f"TaskModelRouter hard-timeout after {timeout}s; resetting memory provider")
                    try:
                        self.shutdown()
                    except Exception as shutdown_exc:
                        LOGGER.warning(f"Failed to reset memory provider after timeout: {shutdown_exc}")
                    preemption = _get_memory_llm_preemption(since_epoch=interrupt_epoch)
                    if preemption is not None:
                        raise preemption
                    raise TimeoutError(f"TaskModelRouter hard-timeout after {timeout}s")

                try:
                    state, payload = result_queue.get_nowait()
                except queue.Empty as exc:
                    raise RuntimeError("TaskModelRouter finished without result") from exc

                if state == "error":
                    preemption = _get_memory_llm_preemption(since_epoch=interrupt_epoch)
                    if preemption is not None:
                        raise preemption from payload
                    raise payload

                result = payload

                # TaskModelExecutionResult имеет атрибут 'text', а не 'output'
                if result.text and len(result.text.strip()) >= 10:
                    return result.text
                else:
                    LOGGER.warning(f"TaskModelRouter returned empty/short text: {len(result.text or '')} chars")
                    preemption = _get_memory_llm_preemption(since_epoch=interrupt_epoch)
                    if preemption is not None:
                        raise preemption
                    raise ValueError(f"TaskModelRouter returned empty/short text: {len(result.text or '')} chars")
            except TimeoutError as e:
                LOGGER.error(f"TaskModelRouter timeout after {timeout}s: {e}")
                preemption = _get_memory_llm_preemption(since_epoch=interrupt_epoch)
                if preemption is not None:
                    raise preemption from e
                raise
            except InterruptedError as e:
                LOGGER.warning(f"MemoryLLMProcessor interrupted: {e}")
                raise  # Пробрасываем прерывание выше
            except Exception as e:
                LOGGER.error(f"TaskModelRouter error: {e}")
                preemption = _get_memory_llm_preemption(since_epoch=interrupt_epoch)
                if preemption is not None:
                    raise preemption from e
                raise
        else:
            # Fallback: простой ответ (для тестов)
            LOGGER.warning("TaskRouter not available, using fallback")
            return self._fallback_response(prompt)

    def get_provider(self):
        if self.task_router is None:
            return None
        getter = getattr(self.task_router, "get_cached_provider", None)
        if not callable(getter):
            return None
        try:
            return getter(self.TASK_NAME)
        except Exception:
            return None

    def pause(self) -> None:
        provider = self.get_provider()
        if provider is None:
            return
        method = getattr(provider, "pause", None)
        if callable(method):
            method()

    def resume(self) -> None:
        provider = self.get_provider()
        if provider is None:
            return
        method = getattr(provider, "resume", None)
        if callable(method):
            method()

    def warmup(self) -> bool:
        provider = self.get_provider()
        if provider is None:
            return False
        method = getattr(provider, "warmup", None)
        if not callable(method):
            return False
        try:
            return bool(method(keep_alive=self.keep_alive))
        except TypeError:
            return bool(method())

    def shutdown(self) -> None:
        if self.task_router is None:
            return
        shutdown = getattr(self.task_router, "shutdown_cached_provider", None)
        if callable(shutdown):
            shutdown(self.TASK_NAME)
            return

        provider = self.get_provider()
        if provider is None:
            return
        method = getattr(provider, "shutdown", None)
        if callable(method):
            method()

    def _fallback_response(self, prompt: str) -> str:
        """
        Fallback ответ для тестов.

        Args:
            prompt: Промпт.

        Returns:
            JSON ответ.
        """
        # Простая эвристика для тестов
        return json.dumps(
            {
                "event_id": "fallback",
                "importance": 0.0,
                "should_process": False,
                "proposals": [],
            },
            ensure_ascii=False,
        )

    def _parse_response(
        self,
        event_id: str,
        response: str,
        interrupt_epoch: int | None = None,
    ) -> MemoryLLMResult:
        """
        Парсит ответ LLM.

        Args:
            event_id: ID события.
            response: Ответ LLM.

        Returns:
            Результат обработки.
        """
        # Проверяем пустой ответ
        if not response or len(response.strip()) < 10:
            LOGGER.warning(f"Empty or too short response for {event_id[:8]}...")
            preemption = _get_memory_llm_preemption(since_epoch=interrupt_epoch)
            if preemption is not None:
                raise preemption
            raise ValueError(f"Empty or too short Memory LLM response for event {event_id}")
        
        try:
            data = json.loads(response)

            proposals = []
            for p in data.get("proposals", []):
                proposal = ArtifactProposal(
                    artifact_type=p.get("artifact_type", "fact"),
                    text=p.get("text", ""),
                    summary=p.get("summary", ""),
                    confidence=float(p.get("confidence", 0.5)),
                    scope=p.get("scope", "global"),
                    decay=p.get("decay", "slow"),
                    retrieve_when=p.get("retrieve_when", []),
                    action=p.get("action", "create"),
                    metadata=p.get("metadata", {}),
                )
                # Валидация confidence
                proposal.confidence = max(0.0, min(1.0, proposal.confidence))
                proposals.append(proposal)

            return MemoryLLMResult(
                event_id=event_id,
                importance=float(data.get("importance", 0.5)),
                should_process=bool(data.get("should_process", True)),
                proposals=proposals,
                raw_response=response,
            )

        except json.JSONDecodeError as e:
            LOGGER.error(f"Failed to parse Memory LLM response: {e}")
            preemption = _get_memory_llm_preemption(since_epoch=interrupt_epoch)
            if preemption is not None:
                raise preemption from e
            raise ValueError(f"Failed to parse Memory LLM response for event {event_id}") from e


def build_memory_llm_processor(
    task_router: TaskModelRouter | None = None,
    keep_alive: Any | None = None,
) -> MemoryLLMProcessor:
    """
    Строит процессор Memory LLM.

    Args:
        task_router: Маршрутизатор задач.

    Returns:
        Процессор Memory LLM.
    """
    return MemoryLLMProcessor(task_router=task_router, keep_alive=keep_alive)
