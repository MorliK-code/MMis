"""
Fact Processor - извлечение фактов из событий.
"""

import re
from memory_core.schemas import MemoryEnvelope, MemoryArtifact
from memory_core.constants import ARTIFACT_FACT, STATUS_ACTIVE


class FactProcessor:
    """
    Процессор извлечения фактов.
    
    Извлекает только проверяемые и полезные факты:
    - "Паша использует VS Code"
    - "Паша работает на Windows"
    - "Проект называется MMis"
    
    НЕ тащит:
    - домыслы
    - стиль речи
    - настроение одного сообщения
    - неустойчивую чепуху
    """
    
    def __init__(self):
        """Инициализирует процессор фактов."""
        # Паттерны для извлечения фактов
        self._fact_patterns = [
            # "Я работаю в X" / "I work in X"
            r"(?:я|i)\s+(?:работаю|work)\s+(?:в|на|with|using)\s+([^\.\n]+)",
            
            # "Я использую X" / "I use X"
            r"(?:я|i)\s+(?:использую|use|юзаю)\s+([^\.\n]+)",
            
            # "Мой X это Y" / "My X is Y"
            r"(?:мой|my)\s+(\w+)\s+(?:это|is)\s+([^\.\n]+)",
            
            # "Проект называется X"
            r"(?:проект|project)\s+(?:называется|called|named)\s+([^\.\n]+)",
            
            # "Я предпочитаю X" / "I prefer X"
            r"(?:я|i)\s+(?:предпочитаю|prefer)\s+([^\.\n]+)",
            
            # "У меня X" / "I have X"
            r"(?:у\s+меня|i\s+have)\s+([^\.\n]+)",
        ]
        
        # Ключевые слова для фактов
        self._fact_keywords = [
            "использует", "uses", "работает", "works",
            "предпочитает", "prefers", "установлен", "installed",
            "настроено", "configured", "версия", "version",
        ]
    
    def process(self, envelope: MemoryEnvelope) -> list[MemoryArtifact]:
        """
        Обрабатывает событие и извлекает факты.
        
        Args:
            envelope: Конверт события.
            
        Returns:
            Список артефактов-фактов.
        """
        artifacts = []
        text = envelope.text
        
        # Извлекаем факты по паттернам
        extracted_facts = self._extract_facts(text)
        
        # Проверяем на ключевые слова
        if not extracted_facts and self._has_fact_keywords(text):
            # Если есть ключевые слова, но не сработали паттерны,
            # можно попробовать извлечь короткие утверждения
            extracted_facts = self._extract_simple_statements(text)
        
        for fact_text in extracted_facts:
            fact = self._normalize_fact(fact_text, envelope)
            if fact:
                artifacts.append(fact)
        
        return artifacts
    
    def _extract_facts(self, text: str) -> list[str]:
        """Извлекает факты по паттернам."""
        facts = []
        
        for pattern in self._fact_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    # Если паттерн с группами, объединяем
                    fact = " ".join(str(m) for m in match if m)
                else:
                    fact = str(match)
                
                # Очищаем и проверяем
                fact = fact.strip()
                if len(fact) > 5 and len(fact) < 200:
                    facts.append(fact)
        
        return facts
    
    def _has_fact_keywords(self, text: str) -> bool:
        """Проверяет наличие ключевых слов фактов."""
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in self._fact_keywords)
    
    def _extract_simple_statements(self, text: str) -> list[str]:
        """Извлекает простые утверждения."""
        statements = []
        
        # Разбиваем на предложения
        sentences = re.split(r'[.!?]', text)
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 10 and len(sentence) < 150:
                # Проверяем на наличие ключевых слов
                if self._has_fact_keywords(sentence):
                    statements.append(sentence)
        
        return statements[:3]  # Ограничиваем количество
    
    def _normalize_fact(self, fact_text: str, envelope: MemoryEnvelope) -> MemoryArtifact | None:
        """Нормализует факт и создаёт артефакт."""
        # Очищаем текст
        fact_text = re.sub(r'\s+', ' ', fact_text).strip()
        
        if not fact_text:
            return None
        
        # Создаём краткое summary
        summary = fact_text[:100] + "..." if len(fact_text) > 100 else fact_text
        
        return MemoryArtifact(
            artifact_type=ARTIFACT_FACT,
            source_event_id=envelope.event_id,
            text=fact_text,
            summary=summary,
            metadata={
                "extracted_by": "fact_processor",
                "confidence": 0.8,
            },
            namespace=envelope.namespace,
            workspace_id=envelope.workspace_id,
            status=STATUS_ACTIVE,
        )
    
    @property
    def name(self) -> str:
        """Возвращает имя процессора."""
        return "fact_processor"
    
    @property
    def priority(self) -> int:
        """Возвращает приоритет (меньше = выше приоритет)."""
        return 10
