"""
Ingest Analyzer - предварительная классификация событий.
"""

from memory_core.schemas import MemoryEnvelope, MemoryArtifact
from memory_core.processors.base import MemoryProcessor
from memory_core.constants import SOURCE_USER, SOURCE_ASSISTANT, SOURCE_TOOL, SOURCE_DOCUMENT


class IngestAnalyzer:
    """
    Анализатор входящих событий.
    
    Определяет:
    - стоит ли событие обрабатывать
    - тип события: message/tool/document/system
    - базовые теги
    - важность (importance)
    - какие процессоры нужно запустить
    """
    
    def __init__(self):
        """Инициализирует анализатор."""
        self._skip_patterns = [
            # Системные сообщения, которые не стоит запоминать
            "/help",
            "/status",
            "/clear",
        ]
    
    def analyze(self, envelope: MemoryEnvelope) -> dict:
        """
        Анализирует событие.
        
        Args:
            envelope: Конверт события.
            
        Returns:
            Словарь с результатами анализа:
            - should_process: bool
            - event_category: str
            - tags: list[str]
            - importance: float (0.0 - 1.0)
            - processors: list[str]
        """
        result = {
            "should_process": True,
            "event_category": "message",
            "tags": [],
            "importance": 0.5,
            "processors": ["fact_processor"],
        }
        
        text = envelope.text.lower().strip()
        
        # Проверка на skip
        for pattern in self._skip_patterns:
            if text.startswith(pattern.lower()):
                result["should_process"] = False
                result["importance"] = 0.0
                return result
        
        # Классификация по типу источника
        if envelope.source_kind == SOURCE_USER:
            result["event_category"] = "user_message"
            result["tags"].append("user_input")
            
            # Вопросы более важны
            if "?" in envelope.text:
                result["tags"].append("question")
                result["importance"] = 0.7
            
            # Команды менее важны
            if text.startswith("/"):
                result["event_category"] = "command"
                result["importance"] = 0.2
                result["should_process"] = False
        
        elif envelope.source_kind == SOURCE_ASSISTANT:
            result["event_category"] = "assistant_response"
            result["tags"].append("assistant_output")
            result["processors"] = ["episode_processor", "task_processor"]
        
        elif envelope.source_kind == SOURCE_TOOL:
            result["event_category"] = "tool_result"
            result["tags"].append("tool_execution")
            result["importance"] = 0.6
            result["processors"] = ["fact_processor"]
        
        elif envelope.source_kind == SOURCE_DOCUMENT:
            result["event_category"] = "document"
            result["tags"].append("document_ingest")
            result["importance"] = 0.8
            result["processors"] = ["document_processor"]
        
        # Определение важности по длине
        if len(envelope.text) > 500:
            result["importance"] = min(result["importance"] + 0.1, 1.0)
        
        # Теги для специальных паттернов
        if any(word in text for word in ["помни", "запомни", "remember"]):
            result["tags"].append("memory_instruction")
            result["importance"] = 0.9
        
        if any(word in text for word in ["факт", "fact", "информация", "information"]):
            result["tags"].append("factual_content")
        
        return result
    
    def get_processors_for_event(
        self,
        envelope: MemoryEnvelope,
        analysis: dict,
    ) -> list[str]:
        """
        Возвращает список процессоров для события.
        
        Args:
            envelope: Конверт события.
            analysis: Результаты анализа.
            
        Returns:
            Список имён процессоров.
        """
        if not analysis["should_process"]:
            return []
        
        return analysis["processors"]
