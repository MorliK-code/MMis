"""
Profile Processor - обработка профиля пользователя и ассистента.
"""

import re
from memory_core.schemas import MemoryEnvelope, MemoryArtifact
from memory_core.constants import ARTIFACT_PROFILE_FACT, STATUS_ACTIVE


class ProfileProcessor:
    """
    Процессор профиля.
    
    Отвечает за профиль пользователя / ассистента:
    - имя пользователя
    - предпочитаемый язык
    - железо (GPU, CPU, RAM)
    - editor (VS Code, PyCharm, etc.)
    - любимый стиль общения
    - роль персонажа
    
    Это должно жить отдельно от "обычных эпизодов".
    """
    
    def __init__(self):
        """Инициализирует процессор профиля."""
        # Паттерны для профиля пользователя
        self._user_patterns = [
            # "Меня зовут X" / "My name is X"
            (r"(?:меня зовут|my name is)\s+([^\.\n]+)", "user_name"),
            
            # "Я предпочитаю X" / "I prefer X"
            (r"(?:я|i)\s+(?:предпочитаю|prefer)\s+([^\.\n]+)", "preference"),
            
            # "Мой язык X" / "My language is X"
            (r"(?:мой язык|my language)\s+(?:это|is)\s+([^\.\n]+)", "language"),
            
            # "У меня X RAM" / "I have X RAM"
            (r"(\d+\s*(?:gb|гб)\s*(?:ram|оперативки|памяти))", "ram"),
            
            # "У меня GPU X" / "I have GPU X"
            (r"(?:gpu|видеокарта|видяха)\s+(?:это|is|:)?\s*([^\.\n]+)", "gpu"),
            
            # "Я использую X редактор" / "I use X editor"
            (r"(?:редактор|editor|ide)\s+(?:это|is|:)?\s*([^\.\n]+)", "editor"),
        ]
        
        # Ключевые слова профиля
        self._profile_keywords = [
            "предпочитаю", "prefer", "люблю", "like",
            "обычно", "usually", "всегда", "always",
            "никогда", "never", "стараюсь", "try to",
        ]
        
        # Категории профиля
        self._profile_categories = {
            "editor": ["vs code", "pycharm", "vim", "neovim", "sublime", "cursor"],
            "os": ["windows", "linux", "macos", "ubuntu", "arch"],
            "language": ["russian", "english", "русский", "английский"],
        }
    
    def process(self, envelope: MemoryEnvelope) -> list[MemoryArtifact]:
        """
        Обрабатывает событие и извлекает профильные факты.
        
        Args:
            envelope: Конверт события.
            
        Returns:
            Список профильных артефактов.
        """
        artifacts = []
        text = envelope.text
        
        # Извлекаем профильные факты
        extracted = self._extract_profile_facts(text)
        
        for category, value in extracted:
            artifact = self._create_profile_artifact(
                category=category,
                value=value,
                envelope=envelope,
            )
            if artifact:
                artifacts.append(artifact)
        
        return artifacts
    
    def _extract_profile_facts(self, text: str) -> list[tuple[str, str]]:
        """Извлекает профильные факты."""
        facts = []
        
        for pattern, category in self._user_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                value = str(match).strip()
                if 2 <= len(value) <= 100:
                    facts.append((category, value))
        
        # Проверяем категории
        for category, keywords in self._profile_categories.items():
            for keyword in keywords:
                if keyword.lower() in text.lower():
                    facts.append((category, keyword))
        
        return facts
    
    def _create_profile_artifact(
        self,
        category: str,
        value: str,
        envelope: MemoryEnvelope,
    ) -> MemoryArtifact | None:
        """Создаёт профильный артефакт."""
        fact_text = f"User {category}: {value}"
        
        return MemoryArtifact(
            artifact_type=ARTIFACT_PROFILE_FACT,
            source_event_id=envelope.event_id,
            text=fact_text,
            summary=f"{category}: {value}",
            metadata={
                "extracted_by": "profile_processor",
                "category": category,
                "value": value,
                "confidence": 0.85,
            },
            namespace=envelope.namespace,
            workspace_id=envelope.workspace_id,
            status=STATUS_ACTIVE,
        )
    
    @property
    def name(self) -> str:
        """Возвращает имя процессора."""
        return "profile_processor"
    
    @property
    def priority(self) -> int:
        """Возвращает приоритет."""
        return 5  # Высокий приоритет для профиля
