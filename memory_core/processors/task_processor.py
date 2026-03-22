"""
Task Processor - выделение задач и открытых петель.
"""

import re
from memory_core.schemas import MemoryEnvelope, MemoryArtifact
from memory_core.constants import ARTIFACT_TASK, STATUS_ACTIVE


class TaskProcessor:
    """
    Процессор задач.
    
    Выделяет текущие задачи и открытые петли:
    - "надо переделать память"
    - "надо улучшить characters/specs"
    - "надо сделать extractor всех фактов"
    
    Это уже ближе к planner/task-state, чем к facts.
    """
    
    def __init__(self):
        """Инициализирует процессор задач."""
        # Паттерны для задач
        self._task_patterns = [
            # "надо X" / "need to X"
            r"(?:надо|нужно|need\s+to|must|should)\s+([^\.\n]+)",
            
            # "планирую X" / "plan to X"
            r"(?:планирую|plan\s+to|хочу|want\s+to)\s+([^\.\n]+)",
            
            # "буду X" / "will X"
            r"(?:буду|i\s+will|i'll)\s+([^\.\n]+)",
            
            # "осталось X" / "remaining: X"
            r"(?:осталось|remaining)\s*(?:сделать|to\s+do)?\s*[:\-]?\s*([^\.\n]+)",
            
            # "открытая задача: X" / "open task: X"
            r"(?:открытая\s*задача|open\s*task)\s*[:\-]?\s*([^\.\n]+)",
        ]
        
        # Ключевые слова задач
        self._task_keywords = [
            "задача", "task", "надо", "need", "должен", "must",
            "планирую", "plan", "хочу", "want", "цель", "goal",
            "сделать", "do", "реализовать", "implement", "исправить", "fix",
        ]
    
    def process(self, envelope: MemoryEnvelope) -> list[MemoryArtifact]:
        """
        Обрабатывает событие и извлекает задачи.
        
        Args:
            envelope: Конверт события.
            
        Returns:
            Список артефактов задач.
        """
        artifacts = []
        text = envelope.text
        
        # Извлекаем задачи
        extracted_tasks = self._extract_tasks(text)
        
        for task_text in extracted_tasks:
            task = self._create_task_artifact(task_text, envelope)
            if task:
                artifacts.append(task)
        
        return artifacts
    
    def _extract_tasks(self, text: str) -> list[str]:
        """Извлекает задачи из текста."""
        tasks = []
        
        for pattern in self._task_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                task = str(match).strip()
                if 5 <= len(task) <= 300:
                    tasks.append(task)
        
        # Если не нашли по паттернам, ищем по ключевым словам
        if not tasks and self._has_task_keywords(text):
            # Пробуем извлечь предложения с ключевыми словами
            sentences = re.split(r'[.!?]', text)
            for sentence in sentences:
                sentence = sentence.strip()
                if self._has_task_keywords(sentence) and 10 <= len(sentence) <= 200:
                    tasks.append(sentence)
        
        return tasks[:5]  # Ограничиваем количество
    
    def _has_task_keywords(self, text: str) -> bool:
        """Проверяет наличие ключевых слов задач."""
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in self._task_keywords)
    
    def _create_task_artifact(
        self,
        task_text: str,
        envelope: MemoryEnvelope,
    ) -> MemoryArtifact | None:
        """Создаёт артефакт задачи."""
        # Очищаем текст
        task_text = re.sub(r'\s+', ' ', task_text).strip()
        
        if not task_text:
            return None
        
        # Определяем статус (открытая/закрытая)
        status = self._detect_task_status(task_text)
        
        # Создаём summary
        summary = task_text[:100] + "..." if len(task_text) > 100 else task_text
        
        return MemoryArtifact(
            artifact_type=ARTIFACT_TASK,
            source_event_id=envelope.event_id,
            text=task_text,
            summary=summary,
            metadata={
                "extracted_by": "task_processor",
                "task_status": status,
                "priority": "normal",
            },
            namespace=envelope.namespace,
            workspace_id=envelope.workspace_id,
            status=STATUS_ACTIVE,
        )
    
    def _detect_task_status(self, task_text: str) -> str:
        """Определяет статус задачи."""
        text_lower = task_text.lower()
        
        # Закрытые задачи
        closed_patterns = [
            "сделал", "done", "завершил", "completed",
            "решил", "resolved", "исправил", "fixed",
        ]
        
        if any(pattern in text_lower for pattern in closed_patterns):
            return "completed"
        
        # Открытые задачи
        return "open"
    
    @property
    def name(self) -> str:
        """Возвращает имя процессора."""
        return "task_processor"
    
    @property
    def priority(self) -> int:
        """Возвращает приоритет."""
        return 15
