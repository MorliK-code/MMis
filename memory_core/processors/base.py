"""
Базовый интерфейс процессора памяти.
"""

from typing import Protocol
from memory_core.schemas import MemoryEnvelope, MemoryArtifact


class MemoryProcessor(Protocol):
    """
    Протокол процессора памяти.
    
    Процессор принимает MemoryEnvelope и возвращает список MemoryArtifact.
    """
    
    def process(self, envelope: MemoryEnvelope) -> list[MemoryArtifact]:
        """
        Обрабатывает событие и создаёт артефакты.
        
        Args:
            envelope: Конверт события.
            
        Returns:
            Список созданных артефактов.
        """
        ...
    
    @property
    def name(self) -> str:
        """Возвращает имя процессора."""
        ...
    
    @property
    def priority(self) -> int:
        """
        Возвращает приоритет процессора.
        
        Меньшее число = выше приоритет.
        """
        ...
