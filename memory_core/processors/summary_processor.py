"""
Summary Processor - создание кратких содержимого.
"""

import re
from memory_core.schemas import MemoryEnvelope, MemoryArtifact
from memory_core.constants import STATUS_ACTIVE


class SummaryProcessor:
    """
    Процессор summary.
    
    Делает компактные summaries по:
    - эпизоду
    - документу
    - workspace
    - длинной ветке диалога
    
    Но summary — это производный артефакт, а не истина.
    """
    
    def __init__(self, max_summary_length: int = 300):
        """
        Инициализирует процессор summary.
        
        Args:
            max_summary_length: Максимальная длина summary.
        """
        self.max_summary_length = max_summary_length
    
    def process(self, envelope: MemoryEnvelope) -> list[MemoryArtifact]:
        """
        Обработка для summary (на этапе ingest не применяется).
        
        Args:
            envelope: Конверт события.
            
        Returns:
            Пустой список (summary применяется к существующим артефактам).
        """
        # Summary применяется к уже существующим артефактам
        # через метод create_summary
        return []
    
    def create_summary(
        self,
        artifacts: list[MemoryArtifact],
        summary_type: str = "episode",
    ) -> MemoryArtifact | None:
        """
        Создаёт summary для группы артефактов.
        
        Args:
            artifacts: Список артефактов для суммаризации.
            summary_type: Тип summary (episode | document | conversation).
            
        Returns:
            Артефакт summary или None.
        """
        if not artifacts:
            return None
        
        # Собираем тексты
        texts = [a.text for a in artifacts if a.text]
        
        if not texts:
            return None
        
        # Создаём summary
        summary_text = self._summarize_texts(texts, summary_type)
        
        if not summary_text:
            return None
        
        # Берём первый артефакт как основу для метаданных
        base_artifact = artifacts[0]
        
        return MemoryArtifact(
            artifact_type=f"summary_{summary_type}",
            source_event_id=base_artifact.source_event_id,
            text="\n\n".join(texts)[:2000],  # Исходные тексты
            summary=summary_text,
            metadata={
                "extracted_by": "summary_processor",
                "summary_type": summary_type,
                "source_artifacts_count": len(artifacts),
                "source_artifact_ids": [a.artifact_id for a in artifacts],
            },
            namespace=base_artifact.namespace,
            workspace_id=base_artifact.workspace_id,
            status=STATUS_ACTIVE,
        )
    
    def _summarize_texts(
        self,
        texts: list[str],
        summary_type: str,
    ) -> str:
        """
        Создаёт summary для списка текстов.
        
        Args:
            texts: Список текстов.
            summary_type: Тип summary.
            
        Returns:
            Краткое summary.
        """
        # Объединяем тексты
        combined = " ".join(texts)
        
        # Разбиваем на предложения
        sentences = re.split(r'[.!?]', combined)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Берём первые N предложений
        num_sentences = min(5, len(sentences))
        summary_sentences = sentences[:num_sentences]
        
        if not summary_sentences:
            return combined[:self.max_summary_length] + "..."
        
        summary = ". ".join(summary_sentences)
        
        # Добавляем точку в конце
        if not summary.endswith("."):
            summary += "."
        
        # Ограничиваем длину
        if len(summary) > self.max_summary_length:
            summary = summary[:self.max_summary_length - 3] + "..."
        
        return summary
    
    def compress_episode(
        self,
        episode_artifacts: list[MemoryArtifact],
    ) -> str:
        """
        Сжимает эпизод до краткого описания.
        
        Args:
            episode_artifacts: Артефакты эпизода.
            
        Returns:
            Краткое описание эпизода.
        """
        if not episode_artifacts:
            return ""
        
        # Собираем ключевую информацию
        topics = set()
        for artifact in episode_artifacts:
            topic = artifact.metadata.get("topic", "")
            if topic:
                topics.add(topic)
        
        # Создаём описание
        if topics:
            return f"Episode about: {', '.join(sorted(topics))}"
        
        # Если нет тем, берём первое summary
        for artifact in episode_artifacts:
            if artifact.summary:
                return artifact.summary[:self.max_summary_length]
        
        return ""
    
    def extract_key_points(
        self,
        text: str,
        max_points: int = 5,
    ) -> list[str]:
        """
        Извлекает ключевые моменты из текста.
        
        Args:
            text: Текст.
            max_points: Максимальное количество пунктов.
            
        Returns:
            Список ключевых моментов.
        """
        # Разбиваем на предложения
        sentences = re.split(r'[.!?]', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Берём первые N предложений как ключевые моменты
        key_points = sentences[:max_points]
        
        return key_points
    
    @property
    def name(self) -> str:
        """Возвращает имя процессора."""
        return "summary_processor"
    
    @property
    def priority(self) -> int:
        """Возвращает приоритет."""
        return 90  # Применяется после основных процессоров
