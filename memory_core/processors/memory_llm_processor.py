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
from dataclasses import dataclass, field
from typing import Any

from memory_core.schemas import MemoryEnvelope
from llm.task_router import TaskModelRouter
from utils.logger import get_logger


LOGGER = get_logger(__name__)


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

    def __init__(
        self,
        task_router: TaskModelRouter | None = None,
        model_profile: str = "memory_llm",
    ):
        """
        Инициализирует процессор.

        Args:
            task_router: Маршрутизатор задач LLM.
            model_profile: Профиль модели для memory_llm.
        """
        self.task_router = task_router
        self.model_profile = model_profile

    def process(self, envelope: MemoryEnvelope) -> MemoryLLMResult:
        """
        Обрабатывает событие и генерирует предложения.

        Args:
            envelope: Конверт события.

        Returns:
            Результат обработки с предложениями.
        """
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

        # Вызываем LLM
        response = self._call_llm(prompt)

        # Парсим ответ
        result = self._parse_response(envelope.event_id, response)

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

    def _call_llm(self, prompt: str) -> str:
        """
        Вызывает LLM для обработки.

        Args:
            prompt: Промпт для LLM.

        Returns:
            Ответ LLM.
        """
        if self.task_router:
            try:
                # Используем метод экземпляра task_router
                result = self.task_router.run_task_model(
                    task_name="memory_llm_process",
                    prompt=prompt,
                    system_prompt=MEMORY_LLM_SYSTEM_PROMPT,
                )

                # TaskModelExecutionResult имеет атрибут 'text', а не 'output'
                if result.text and len(result.text.strip()) >= 10:
                    return result.text
                else:
                    LOGGER.warning(f"TaskModelRouter returned empty/short text: {len(result.text or '')} chars")
                    return self._fallback_response(prompt)
            except Exception as e:
                LOGGER.error(f"TaskModelRouter error: {e}")
                return self._fallback_response(prompt)
        else:
            # Fallback: простой ответ (для тестов)
            LOGGER.warning("TaskRouter not available, using fallback")
            return self._fallback_response(prompt)

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
                "importance": 0.5,
                "should_process": True,
                "proposals": [
                    {
                        "artifact_type": "episode_event",
                        "text": "Событие из лога",
                        "summary": "Тестовое событие",
                        "confidence": 0.6,
                        "scope": "episode",
                        "decay": "fast",
                        "retrieve_when": ["test"],
                        "action": "create",
                    }
                ],
            },
            ensure_ascii=False,
        )

    def _parse_response(
        self,
        event_id: str,
        response: str,
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
            return MemoryLLMResult(
                event_id=event_id,
                importance=0.0,
                should_process=False,
                proposals=[],
                raw_response=response or "",
            )
        
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
            return MemoryLLMResult(
                event_id=event_id,
                importance=0.0,
                should_process=False,
                proposals=[],
                raw_response=response,
            )


def build_memory_llm_processor(
    task_router: TaskModelRouter | None = None,
) -> MemoryLLMProcessor:
    """
    Строит процессор Memory LLM.

    Args:
        task_router: Маршрутизатор задач.

    Returns:
        Процессор Memory LLM.
    """
    return MemoryLLMProcessor(task_router=task_router)
