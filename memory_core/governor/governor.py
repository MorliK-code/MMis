"""
Governor — управляющий компонент памяти.

Принимает решения о том, что записывать в память:
- создать новый артефакт
- обновить существующий
- объединить с существующим
- отклонить (игнорировать)
- заменить (supersede)

Governor — это логика, а не LLM.
LLM предлагает → Governor решает → память сохраняет
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from memory_core.schemas import MemoryEnvelope, MemoryArtifact
from memory_core.storage.artifact_store import ArtifactStore
from memory_core.processors.memory_llm_processor import ArtifactProposal, MemoryLLMResult
from memory_core.memory_types import enrich_metadata_with_memory_type
from utils.logger import get_logger


LOGGER = get_logger(__name__)


@dataclass(slots=True)
class GovernorDecision:
    """
    Решение Governor по артефакту.
    """
    action: str  # create | update | merge | ignore | supersede
    artifact: MemoryArtifact | None = None
    merged_artifact_ids: list[str] = field(default_factory=list)
    superseded_artifact_ids: list[str] = field(default_factory=list)
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Преобразует в словарь."""
        return {
            "action": self.action,
            "artifact": self.artifact.to_dict() if self.artifact else None,
            "merged_artifact_ids": self.merged_artifact_ids,
            "superseded_artifact_ids": self.superseded_artifact_ids,
            "reason": self.reason,
        }


@dataclass(slots=True)
class GovernorResult:
    """
    Результат работы Governor.
    """
    decisions: list[GovernorDecision] = field(default_factory=list)
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Преобразует в словарь."""
        return {
            "decisions": [d.to_dict() for d in self.decisions],
            "artifacts": self.artifacts,
            "stats": self.stats,
        }


class Governor:
    """
    Governor — управляющий компонент памяти.

    Отвечает за:
    - конфликт фактов
    - superseded (замена устаревших)
    - confidence decay
    - merge фактов
    - idempotency (не создавать дубли)
    - identity protection
    - persistent vs volatile
    - очистку устаревших данных
    """

    # Пороги confidence
    CONFIDENCE_HIGH = 0.8
    CONFIDENCE_MEDIUM = 0.5
    CONFIDENCE_LOW = 0.3

    # Типы артефактов с защитой от перезаписи
    PROTECTED_TYPES = {"identity_core"}

    # Максимальное количество артефактов одного типа
    MAX_ARTIFACTS_PER_TYPE = 100
    TOPIC_SCOPED_ARTIFACT_TYPES = {"episode_event", "task_state", "emotional_state"}

    def __init__(
        self,
        artifact_store: ArtifactStore,
    ):
        """
        Инициализирует Governor.

        Args:
            artifact_store: Хранилище артефактов.
        """
        self.artifact_store = artifact_store

    def decide(
        self,
        proposals: list[ArtifactProposal],
        envelope: MemoryEnvelope,
    ) -> GovernorResult:
        """
        Принимает решения по предложениям.

        Args:
            proposals: Список предложений от Memory LLM.
            envelope: Конверт события.

        Returns:
            Результат с решениями.
        """
        result = GovernorResult()
        stats = {
            "proposals_received": len(proposals),
            "artifacts_created": 0,
            "artifacts_updated": 0,
            "artifacts_merged": 0,
            "artifacts_superseded": 0,
            "proposals_ignored": 0,
        }

        for proposal in proposals:
            decision = self._decide_proposal(proposal, envelope)
            result.decisions.append(decision)

            if decision.action == "ignore":
                stats["proposals_ignored"] += 1
            elif decision.action == "create":
                stats["artifacts_created"] += 1
                if decision.artifact:
                    result.artifacts.append(decision.artifact.to_dict())
            elif decision.action == "update":
                stats["artifacts_updated"] += 1
                if decision.artifact:
                    result.artifacts.append(decision.artifact.to_dict())
            elif decision.action == "merge":
                stats["artifacts_merged"] += 1
                if decision.artifact:
                    result.artifacts.append(decision.artifact.to_dict())
            elif decision.action == "supersede":
                stats["artifacts_superseded"] += len(decision.superseded_artifact_ids)
                if decision.artifact:
                    result.artifacts.append(decision.artifact.to_dict())

        result.stats = stats
        return result

    def _decide_proposal(
        self,
        proposal: ArtifactProposal,
        envelope: MemoryEnvelope,
    ) -> GovernorDecision:
        """
        Принимает решение по одному предложению.

        Args:
            proposal: Предложение.
            envelope: Конверт события.

        Returns:
            Решение.
        """
        # Игнорируем предложения с низким confidence
        if proposal.confidence < self.CONFIDENCE_LOW:
            return GovernorDecision(
                action="ignore",
                reason=f"Low confidence: {proposal.confidence}",
            )

        # Игнорируем пустые
        if not proposal.text or len(proposal.text.strip()) < 3:
            return GovernorDecision(
                action="ignore",
                reason="Empty text",
            )

        # Проверяем существующие артефакты
        existing = self._find_similar_artifacts(proposal, envelope)

        if not existing:
            # Нет похожих → создаём новый
            return self._create_artifact(proposal, envelope)

        # Есть похожие → проверяем на конфликт/merge/supersede
        return self._resolve_conflicts(proposal, existing, envelope)

    def _find_similar_artifacts(
        self,
        proposal: ArtifactProposal,
        envelope: MemoryEnvelope,
    ) -> list[MemoryArtifact]:
        """
        Ищет похожие артефакты.

        Args:
            proposal: Предложение.
            envelope: Конверт события.

        Returns:
            Список похожих артефактов.
        """
        # Ищем по типу и workspace (list_artifacts возвращает list[MemoryArtifact])
        existing = self.artifact_store.list_artifacts(
            artifact_type=proposal.artifact_type,
            workspace_id=envelope.workspace_id,
            status="active",
            limit=100,
        )

        if not existing:
            return []

        # Фильтруем по схожести текста (простая эвристика)
        similar = []
        proposal_text_lower = proposal.text.lower()

        for artifact in existing:
            if self._is_topic_scoped_proposal(proposal) and not self._topic_matches(artifact, envelope):
                continue
            artifact_text_lower = artifact.text.lower()

            # Проверка на частичное совпадение
            if (
                proposal_text_lower in artifact_text_lower
                or artifact_text_lower in proposal_text_lower
            ):
                similar.append(artifact)
                continue

            # Проверка на схожесть по ключевым словам
            proposal_words = set(proposal_text_lower.split())
            artifact_words = set(artifact_text_lower.split())
            overlap = len(proposal_words & artifact_words)

            if overlap >= 3:  # Минимум 3 общих слова
                similar.append(artifact)

        return similar

    def _create_artifact(
        self,
        proposal: ArtifactProposal,
        envelope: MemoryEnvelope,
    ) -> GovernorDecision:
        """
        Создаёт новый артефакт.

        Args:
            proposal: Предложение.
            envelope: Конверт события.

        Returns:
            Решение с созданным артефактом.
        """
        import uuid
        now = time.time()

        # Генерируем детерминированный artifact_id на основе event_id + type
        # Это предотвратит дублирование при retry
        artifact_id = f"{envelope.event_id}_{proposal.artifact_type}"

        # Проверяем, нет ли уже артефакта с таким ID
        try:
            existing = self.artifact_store.get_by_id(artifact_id)
            if existing:
                # Такой артефакт уже есть, обновляем его
                return self._update_artifact(proposal, existing, envelope)
        except Exception:
            # Артефакт не найден, создаём новый
            pass

        artifact = MemoryArtifact(
            artifact_id=artifact_id,  # Используем детерминированный ID
            artifact_type=proposal.artifact_type,
            source_event_id=envelope.event_id,
            text=proposal.text,
            summary=proposal.summary,
            metadata={
                **self._merged_artifact_metadata(proposal=proposal, envelope=envelope),
                "confidence": proposal.confidence,
                "scope": proposal.scope,
                "decay": proposal.decay,
                "retrieve_when": proposal.retrieve_when,
                "session_id": envelope.session_id,  # Добавляем session_id
                # episode_id будет добавлен в ingest pipeline
            },
            namespace=envelope.namespace,
            workspace_id=envelope.workspace_id,
            status="active",
            created_at=now,
            updated_at=now,
        )

        # Сохраняем с обработкой UNIQUE constraint
        try:
            self.artifact_store.create(artifact)
        except Exception as e:
            if "UNIQUE constraint failed" in str(e):
                # Артефакт уже создан другим потоком, получаем его и обновляем
                existing = self.artifact_store.get_by_id(artifact_id)
                if existing:
                    return self._update_artifact(proposal, existing, envelope)
            # Пересоздаём ошибку если это не UNIQUE constraint
            raise

        return GovernorDecision(
            action="create",
            artifact=artifact,
            reason="New artifact created",
        )

    def _resolve_conflicts(
        self,
        proposal: ArtifactProposal,
        existing: list[MemoryArtifact],
        envelope: MemoryEnvelope,
    ) -> GovernorDecision:
        """
        Разрешает конфликты с существующими артефактами.

        Args:
            proposal: Предложение.
            existing: Существующие артефакты.
            envelope: Конверт события.

        Returns:
            Решение.
        """
        # Сортируем по confidence (в metadata)
        existing_sorted = sorted(
            existing,
            key=lambda a: float(a.metadata.get("confidence", 0.5)),
            reverse=True,
        )

        best_existing = existing_sorted[0]
        existing_confidence = float(best_existing.metadata.get("confidence", 0.5))

        # Если новый артефакт значительно увереннее → supersede
        if proposal.confidence > existing_confidence + 0.2:
            return self._supersede_artifacts(proposal, existing, envelope)

        # Если confidence примерно одинаковый → merge или update
        if abs(proposal.confidence - existing_confidence) < 0.1:
            # Проверяем, не дубликат ли это
            if self._is_duplicate(proposal, best_existing):
                return GovernorDecision(
                    action="ignore",
                    reason="Duplicate of existing artifact",
                )

            # Merge если тексты дополняют друг друга
            return self._merge_artifacts(proposal, best_existing, envelope)

        # Если существующий увереннее → update существующего
        return self._update_artifact(proposal, best_existing, envelope)

    def _is_duplicate(
        self,
        proposal: ArtifactProposal,
        existing: MemoryArtifact,
    ) -> bool:
        """
        Проверяет, является ли предложение дубликатом.

        Args:
            proposal: Предложение.
            existing: Существующий артефакт.

        Returns:
            True если дубликат.
        """
        # Точное совпадение текста
        if proposal.text.strip().lower() == existing.text.strip().lower():
            return True

        # Защищённые типы не дублируем
        if proposal.artifact_type in self.PROTECTED_TYPES:
            if proposal.text.strip().lower() in existing.text.strip().lower():
                return True

        return False

    def _update_artifact(
        self,
        proposal: ArtifactProposal,
        existing: MemoryArtifact,
        envelope: MemoryEnvelope,
    ) -> GovernorDecision:
        """
        Обновляет существующий артефакт.

        Args:
            proposal: Предложение.
            existing: Существующий артефакт.
            envelope: Конверт события.

        Returns:
            Решение с обновлённым артефактом.
        """
        now = time.time()

        # Обновляем текст и metadata
        existing.text = proposal.text
        existing.summary = proposal.summary
        existing.metadata.update({
            **self._merged_artifact_metadata(proposal=proposal, envelope=envelope),
            "confidence": proposal.confidence,
            "scope": proposal.scope,
            "decay": proposal.decay,
            "retrieve_when": proposal.retrieve_when,
            "last_source_event_id": envelope.event_id,
            "last_governor_action": "update",
        })
        existing.updated_at = now

        # Сохраняем через update() а не create()
        self.artifact_store.update(existing)

        return GovernorDecision(
            action="update",
            artifact=existing,
            reason="Existing artifact updated",
        )

    def _merge_artifacts(
        self,
        proposal: ArtifactProposal,
        existing: MemoryArtifact,
        envelope: MemoryEnvelope,
    ) -> GovernorDecision:
        """
        Объединяет артефакты.

        Args:
            proposal: Предложение.
            existing: Существующий артефакт.
            envelope: Конверт события.

        Returns:
            Решение с объединённым артефактом.
        """
        now = time.time()

        # Объединяем тексты
        merged_text = self._merge_texts(existing.text, proposal.text)
        merged_summary = self._merge_texts(existing.summary, proposal.summary)

        existing.text = merged_text
        existing.summary = merged_summary
        existing.metadata.update({
            **self._merged_artifact_metadata(proposal=proposal, envelope=envelope),
            "confidence": max(
                float(existing.metadata.get("confidence", 0.5)),
                proposal.confidence,
            ),
            "scope": proposal.scope,
            "decay": proposal.decay,
            "retrieve_when": proposal.retrieve_when,
            "merged_at": now,
            "merge_count": existing.metadata.get("merge_count", 0) + 1,
            "last_source_event_id": envelope.event_id,
            "last_governor_action": "merge",
        })
        existing.updated_at = now

        # Сохраняем через update() а не create()
        self.artifact_store.update(existing)

        return GovernorDecision(
            action="merge",
            artifact=existing,
            merged_artifact_ids=[existing.artifact_id],
            reason="Artifacts merged",
        )

    def _merge_texts(self, text1: str, text2: str) -> str:
        """
        Объединяет два текста.

        Args:
            text1: Первый текст.
            text2: Второй текст.

        Returns:
            Объединённый текст.
        """
        # Простая эвристика: если один текст содержит другой → возвращаем более полный
        if text1.lower() in text2.lower():
            return text2
        if text2.lower() in text1.lower():
            return text1

        # Иначе объединяем
        return f"{text1}; {text2}"

    def _supersede_artifacts(
        self,
        proposal: ArtifactProposal,
        existing: list[MemoryArtifact],
        envelope: MemoryEnvelope,
    ) -> GovernorDecision:
        """
        Заменяет устаревшие артефакты.

        Args:
            proposal: Предложение.
            existing: Существующие артефакты.
            envelope: Конверт события.

        Returns:
            Решение с новым артефактом и списком заменённых.
        """
        now = time.time()
        superseded_ids = [a.artifact_id for a in existing]

        # Создаём новый артефакт
        artifact = MemoryArtifact(
            artifact_type=proposal.artifact_type,
            source_event_id=envelope.event_id,
            text=proposal.text,
            summary=proposal.summary,
            metadata={
                **self._merged_artifact_metadata(proposal=proposal, envelope=envelope),
                "confidence": proposal.confidence,
                "scope": proposal.scope,
                "decay": proposal.decay,
                "retrieve_when": proposal.retrieve_when,
                "supersedes": superseded_ids,
                "last_governor_action": "supersede_new",
            },
            namespace=envelope.namespace,
            workspace_id=envelope.workspace_id,
            status="active",
            created_at=now,
            updated_at=now,
        )

        # Помечаем старые как superseded через update()
        for a in existing:
            a.status = "superseded"
            a.metadata.update({
                "superseded_by": artifact.artifact_id,
                "superseded_at": now,
                "last_governor_action": "supersede_old",
            })
            a.updated_at = now
            self.artifact_store.update(a)

        # Сохраняем новый
        self.artifact_store.create(artifact)

        return GovernorDecision(
            action="supersede",
            artifact=artifact,
            superseded_artifact_ids=superseded_ids,
            reason=f"Superseded {len(superseded_ids)} artifacts",
        )

    def _merged_artifact_metadata(
        self,
        *,
        proposal: ArtifactProposal,
        envelope: MemoryEnvelope,
    ) -> dict[str, Any]:
        metadata = dict(proposal.metadata or {})
        envelope_meta = dict(envelope.metadata or {})
        for key in (
            "topic_thread_id",
            "topic_key",
            "topic_title",
            "visible_chat_id",
            "topic_route_reason",
            "topic_route_score",
            "episode_id",
        ):
            value = envelope_meta.get(key)
            if value in (None, "", []):
                continue
            metadata.setdefault(key, value)
        related_topic_ids = [
            str(item).strip()
            for item in list(
                envelope_meta.get("related_topic_thread_ids")
                or envelope_meta.get("related_topic_ids")
                or []
            )
            if str(item).strip()
        ]
        if related_topic_ids:
            metadata.setdefault("related_topic_thread_ids", related_topic_ids)
        return enrich_metadata_with_memory_type(
            metadata,
            artifact_type=proposal.artifact_type,
        )

    def _is_topic_scoped_proposal(self, proposal: ArtifactProposal) -> bool:
        scope = str(proposal.scope or "").strip().lower()
        if scope in {"episode", "task"}:
            return True
        artifact_type = str(proposal.artifact_type or "").strip().lower()
        return artifact_type in self.TOPIC_SCOPED_ARTIFACT_TYPES

    @staticmethod
    def _topic_matches(artifact: MemoryArtifact, envelope: MemoryEnvelope) -> bool:
        envelope_topic = str(dict(envelope.metadata or {}).get("topic_thread_id") or "").strip()
        if not envelope_topic:
            return True
        artifact_topic = str(dict(artifact.metadata or {}).get("topic_thread_id") or "").strip()
        return artifact_topic == envelope_topic


def build_governor(artifact_store: ArtifactStore) -> Governor:
    """
    Строит Governor.

    Args:
        artifact_store: Хранилище артефактов.

    Returns:
        Governor.
    """
    return Governor(artifact_store=artifact_store)
