"""
IdentityCore — управление ядром личности ассистента.

Отдельный тип памяти с максимальным приоритетом.
Содержит:
- имя пользователя
- как обращаться
- стиль общения
- характер ассистента
- важные долгосрочные факты

Identity Core:
- без decay
- всегда подгружается
- защищён от перезаписи
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from memory_core.schemas import MemoryArtifact, MemoryEnvelope
from memory_core.storage.artifact_store import ArtifactStore
from memory_core.governor.governor import Governor
from utils.logger import get_logger


LOGGER = get_logger(__name__)


@dataclass(slots=True)
class IdentityProfile:
    """
    Профиль идентичности.
    """
    # Пользователь
    user_name: str | None = None
    user_address_form: str = "ты"  # ты | вы

    # Ассистент
    assistant_name: str = "MMis"
    assistant_personality: list[str] = field(default_factory=list)
    assistant_style: str = "friendly"  # friendly | formal | concise | detailed | empathetic
    assistant_tone: str = "warm"  # warm | neutral | professional

    # Ограничения
    always_do: list[str] = field(default_factory=list)
    never_do: list[str] = field(default_factory=list)

    # Мета
    workspace_id: str = "global"
    updated_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        """Преобразует в словарь."""
        return {
            "user_name": self.user_name,
            "user_address_form": self.user_address_form,
            "assistant_name": self.assistant_name,
            "assistant_personality": self.assistant_personality,
            "assistant_style": self.assistant_style,
            "assistant_tone": self.assistant_tone,
            "always_do": self.always_do,
            "never_do": self.never_do,
            "workspace_id": self.workspace_id,
            "updated_at": self.updated_at,
        }

    def to_artifact_text(self) -> str:
        """
        Формирует текст для артефакта.

        Returns:
            Текст идентичности.
        """
        lines = []

        if self.user_name:
            lines.append(f"Имя пользователя: {self.user_name}")
            lines.append(f"Форма обращения: {self.user_address_form}")

        lines.append(f"Имя ассистента: {self.assistant_name}")

        if self.assistant_personality:
            lines.append("Личность:")
            for trait in self.assistant_personality:
                lines.append(f"  - {trait}")

        lines.append(f"Стиль: {self.assistant_style}")
        lines.append(f"Тон: {self.assistant_tone}")

        if self.always_do:
            lines.append("Всегда делать:")
            for item in self.always_do:
                lines.append(f"  - {item}")

        if self.never_do:
            lines.append("Никогда не делать:")
            for item in self.never_do:
                lines.append(f"  - {item}")

        return "\n".join(lines)


class IdentityCore:
    """
    Управление ядром идентичности.

    Защищённое хранилище для важнейших фактов о личности
    ассистента и пользователя.
    """

    ARTIFACT_TYPE = "identity_core"
    IDENTITY_KEY = "identity_profile"

    # Максимальное количество артефактов identity
    MAX_IDENTITY_ARTIFACTS = 20

    def __init__(
        self,
        artifact_store: ArtifactStore,
        governor: Governor | None = None,
    ):
        """
        Инициализирует IdentityCore.

        Args:
            artifact_store: Хранилище артефактов.
            governor: Governor для принятия решений.
        """
        self.artifact_store = artifact_store
        self.governor = governor

    def get_profile(self, workspace_id: str = "global") -> IdentityProfile:
        """
        Получает профиль идентичности.

        Args:
            workspace_id: ID workspace.

        Returns:
            IdentityProfile.
        """
        artifacts = self.artifact_store.get_by_type(
            artifact_type=self.ARTIFACT_TYPE,
            workspace_id=workspace_id,
            status="active",
            limit=self.MAX_IDENTITY_ARTIFACTS,
        )

        profile = IdentityProfile(workspace_id=workspace_id)

        for artifact in artifacts:
            self._apply_artifact_to_profile(artifact, profile)

        # Сортируем артефакты по updated_at
        artifacts_sorted = sorted(artifacts, key=lambda a: a.updated_at, reverse=True)
        if artifacts_sorted:
            profile.updated_at = artifacts_sorted[0].updated_at

        return profile

    def _apply_artifact_to_profile(
        self,
        artifact: MemoryArtifact,
        profile: IdentityProfile,
    ) -> None:
        """
        Применяет артефакт к профилю.

        Args:
            artifact: Артефакт.
            profile: Профиль для обновления.
        """
        text = artifact.text.lower()
        metadata = artifact.metadata or {}

        # Извлекаем имя пользователя
        if "имя пользователя" in text or "user name" in text:
            name = text.split(":")[-1].strip() if ":" in text else text
            profile.user_name = name

        # Форма обращения
        if "форма обращения" in text or "address form" in text:
            if "вы" in text:
                profile.user_address_form = "вы"
            elif "ты" in text:
                profile.user_address_form = "ты"

        # Имя ассистента
        if "имя ассистента" in text or "assistant name" in text:
            name = text.split(":")[-1].strip() if ":" in text else text
            profile.assistant_name = name

        # Стиль
        if "стиль" in text or "style" in text:
            if "friendly" in text or "дружелюбный" in text:
                profile.assistant_style = "friendly"
            elif "formal" in text or "формальный" in text:
                profile.assistant_style = "formal"
            elif "concise" in text or "краткий" in text:
                profile.assistant_style = "concise"
            elif "detailed" in text or "подробный" in text:
                profile.assistant_style = "detailed"
            elif "empathetic" in text or "эмпатичный" in text:
                profile.assistant_style = "empathetic"

        # Тон
        if "тон" in text or "tone" in text:
            if "warm" in text or "тёплый" in text:
                profile.assistant_tone = "warm"
            elif "neutral" in text or "нейтральный" in text:
                profile.assistant_tone = "neutral"
            elif "professional" in text or "профессиональный" in text:
                profile.assistant_tone = "professional"

        # Черты личности
        if "личность" in text or "personality" in text:
            # Извлекаем список черт
            lines = artifact.text.split("\n")
            for line in lines:
                line = line.strip()
                if line.startswith("- ") or line.startswith("  - "):
                    trait = line.lstrip("- ").strip()
                    if trait and trait not in profile.assistant_personality:
                        profile.assistant_personality.append(trait)

        # Всегда делать
        if "всегда" in text or "always" in text:
            lines = artifact.text.split("\n")
            for line in lines:
                line = line.strip()
                if line.startswith("- ") or line.startswith("  - "):
                    item = line.lstrip("- ").strip()
                    if item and item not in profile.always_do:
                        profile.always_do.append(item)

        # Никогда не делать
        if "никогда" in text or "never" in text:
            lines = artifact.text.split("\n")
            for line in lines:
                line = line.strip()
                if line.startswith("- ") or line.startswith("  - "):
                    item = line.lstrip("- ").strip()
                    if item and item not in profile.never_do:
                        profile.never_do.append(item)

    def set_user_name(
        self,
        name: str,
        workspace_id: str = "global",
        envelope: MemoryEnvelope | None = None,
    ) -> MemoryArtifact:
        """
        Устанавливает имя пользователя.

        Args:
            name: Имя пользователя.
            workspace_id: ID workspace.
            envelope: Конверт события (опционально).

        Returns:
            Созданный/обновлённый артефакт.
        """
        return self._set_identity_field(
            field_text=f"Имя пользователя: {name}",
            workspace_id=workspace_id,
            envelope=envelope,
        )

    def set_assistant_name(
        self,
        name: str,
        workspace_id: str = "global",
        envelope: MemoryEnvelope | None = None,
    ) -> MemoryArtifact:
        """
        Устанавливает имя ассистента.

        Args:
            name: Имя ассистента.
            workspace_id: ID workspace.
            envelope: Конверт события (опционально).

        Returns:
            Созданный/обновлённый артефакт.
        """
        return self._set_identity_field(
            field_text=f"Имя ассистента: {name}",
            workspace_id=workspace_id,
            envelope=envelope,
        )

    def set_address_form(
        self,
        form: str,  # ты | вы
        workspace_id: str = "global",
        envelope: MemoryEnvelope | None = None,
    ) -> MemoryArtifact:
        """
        Устанавливает форму обращения.

        Args:
            form: Форма обращения (ты | вы).
            workspace_id: ID workspace.
            envelope: Конверт события (опционально).

        Returns:
            Созданный/обновлённый артефакт.
        """
        return self._set_identity_field(
            field_text=f"Форма обращения: {form}",
            workspace_id=workspace_id,
            envelope=envelope,
        )

    def set_style(
        self,
        style: str,
        workspace_id: str = "global",
        envelope: MemoryEnvelope | None = None,
    ) -> MemoryArtifact:
        """
        Устанавливает стиль общения.

        Args:
            style: Стиль (friendly | formal | concise | detailed | empathetic).
            workspace_id: ID workspace.
            envelope: Конверт события (опционально).

        Returns:
            Созданный/обновлённый артефакт.
        """
        return self._set_identity_field(
            field_text=f"Стиль общения: {style}",
            workspace_id=workspace_id,
            envelope=envelope,
        )

    def add_personality_trait(
        self,
        trait: str,
        workspace_id: str = "global",
        envelope: MemoryEnvelope | None = None,
    ) -> MemoryArtifact:
        """
        Добавляет черту личности.

        Args:
            trait: Черта личности.
            workspace_id: ID workspace.
            envelope: Конверт события (опционально).

        Returns:
            Созданный/обновлённый артефакт.
        """
        return self._set_identity_field(
            field_text=f"Черта личности: {trait}",
            workspace_id=workspace_id,
            envelope=envelope,
        )

    def _set_identity_field(
        self,
        field_text: str,
        workspace_id: str = "global",
        envelope: MemoryEnvelope | None = None,
    ) -> MemoryArtifact:
        """
        Устанавливает поле идентичности.

        Args:
            field_text: Текст поля.
            workspace_id: ID workspace.
            envelope: Конверт события.

        Returns:
            Созданный/обновлённый артефакт.
        """
        now = time.time()

        # Проверяем существующие артефакты
        existing = self.artifact_store.get_by_type(
            artifact_type=self.ARTIFACT_TYPE,
            workspace_id=workspace_id,
            status="active",
            limit=self.MAX_IDENTITY_ARTIFACTS,
        )

        # Ищем похожий артефакт
        field_lower = field_text.lower()
        for artifact in existing:
            if field_lower in artifact.text.lower():
                # Обновляем существующий
                artifact.text = field_text
                artifact.updated_at = now
                artifact.metadata["confidence"] = 1.0  # Максимальная уверенность
                artifact.metadata["decay"] = "none"  # Без устаревания
                self.artifact_store.save(artifact)
                LOGGER.info(f"Updated identity artifact: {artifact.artifact_id}")
                return artifact

        # Создаём новый артефакт
        artifact = MemoryArtifact(
            artifact_type=self.ARTIFACT_TYPE,
            source_event_id=envelope.event_id if envelope else "system",
            text=field_text,
            summary=f"Identity: {field_text[:50]}",
            metadata={
                "confidence": 1.0,
                "decay": "none",
                "scope": "global",
                "retrieve_when": ["always", "identity"],
                "protected": True,
            },
            namespace="identity",
            workspace_id=workspace_id,
            status="active",
            created_at=now,
            updated_at=now,
        )

        self.artifact_store.save(artifact)
        LOGGER.info(f"Created identity artifact: {artifact.artifact_id}")

        return artifact

    def get_artifacts(self, workspace_id: str = "global") -> list[MemoryArtifact]:
        """
        Получает все артефакты Identity Core.

        Args:
            workspace_id: ID workspace.

        Returns:
            Список артефактов.
        """
        return self.artifact_store.get_by_type(
            artifact_type=self.ARTIFACT_TYPE,
            workspace_id=workspace_id,
            status="active",
            limit=self.MAX_IDENTITY_ARTIFACTS,
        )

    def to_prompt(self, workspace_id: str = "global") -> str:
        """
        Формирует промпт для LLM.

        Args:
            workspace_id: ID workspace.

        Returns:
            Текстовый промпт.
        """
        profile = self.get_profile(workspace_id)
        return profile.to_artifact_text()


def build_identity_core(
    artifact_store: ArtifactStore,
    governor: Governor | None = None,
) -> IdentityCore:
    """
    Строит IdentityCore.

    Args:
        artifact_store: Хранилище артефактов.
        governor: Governor.

    Returns:
        IdentityCore.
    """
    return IdentityCore(
        artifact_store=artifact_store,
        governor=governor,
    )
