"""
Episode Processor - сборка эпизодов диалога.
"""

import re
from memory_core.schemas import MemoryEnvelope, MemoryArtifact
from memory_core.constants import ARTIFACT_EPISODE, STATUS_ACTIVE


class EpisodeProcessor:
    """
    Процессор эпизодов.
    
    Собирает эпизоды диалога - не каждое сообщение,
    а смысловой кусок:
    - пользователь спросил про память
    - вы решили снести memory_v2
    - договорились делать memory_core
    - обсуждали workspace memory
    
    Это нужно, чтобы retrieval по истории не пытался
    вспоминать 100 отдельных сообщений вместо 1 осмысленного блока.
    """
    
    def __init__(self):
        """Инициализирует процессор эпизодов."""
        # Темы для эпизодов
        self._topic_patterns = [
            (r"(?:память|memory)", "memory"),
            (r"(?:база\s*данных|database|db|sqlite)", "database"),
            (r"(?:архитектура|architecture)", "architecture"),
            (r"(?:код|code|рефакторинг|refactor)", "code"),
            (r"(?:тесты|tests|testing)", "testing"),
            (r"(?:документ|document|файл|file)", "documents"),
            (r"(?:задача|task|проблема|problem|issue)", "tasks"),
            (r"(?:персонаж|character|роль|role)", "character"),
        ]
    
    def process(self, envelope: MemoryEnvelope) -> list[MemoryArtifact]:
        """
        Обрабатывает событие и определяет тему эпизода.
        
        Args:
            envelope: Конверт события.
            
        Returns:
            Список артефактов эпизодов.
        """
        artifacts = []
        text = envelope.text
        
        # Определяем тему
        topics = self._detect_topics(text)
        
        if topics:
            # Создаём эпизод для каждой темы
            for topic in topics:
                episode = self._create_episode_artifact(
                    topic=topic,
                    text=text,
                    envelope=envelope,
                )
                artifacts.append(episode)
        
        return artifacts
    
    def _detect_topics(self, text: str) -> list[str]:
        """Определяет темы в тексте."""
        topics = []
        text_lower = text.lower()
        
        for pattern, topic in self._topic_patterns:
            if re.search(pattern, text_lower):
                topics.append(topic)
        
        # Убираем дубликаты
        return list(set(topics))
    
    def _create_episode_artifact(
        self,
        topic: str,
        text: str,
        envelope: MemoryEnvelope,
    ) -> MemoryArtifact:
        """Создаёт артефакт эпизода."""
        # Создаём краткое summary
        summary = self._summarize_episode(text, topic)
        
        return MemoryArtifact(
            artifact_type=ARTIFACT_EPISODE,
            source_event_id=envelope.event_id,
            text=text[:500],  # Ограничиваем длину
            summary=summary,
            metadata={
                "extracted_by": "episode_processor",
                "topic": topic,
                "source_kind": envelope.source_kind,
            },
            namespace=envelope.namespace,
            workspace_id=envelope.workspace_id,
            status=STATUS_ACTIVE,
        )
    
    def _summarize_episode(self, text: str, topic: str) -> str:
        """Создаёт краткое summary эпизода."""
        # Берём первые 2-3 предложения
        sentences = re.split(r'[.!?]', text)
        summary_sentences = [s.strip() for s in sentences[:2] if s.strip()]
        
        if summary_sentences:
            summary = ". ".join(summary_sentences) + "."
        else:
            summary = text[:150] + "..." if len(text) > 150 else text
        
        return f"[{topic}] {summary}"
    
    @property
    def name(self) -> str:
        """Возвращает имя процессора."""
        return "episode_processor"
    
    @property
    def priority(self) -> int:
        """Возвращает приоритет."""
        return 20
