"""
Исключения memory_core.
"""


class MemoryError(Exception):
    """Базовое исключение для всех ошибок памяти."""
    pass


class EventStoreError(MemoryError):
    """Ошибка при работе с хранилищем событий."""
    pass


class ArtifactStoreError(MemoryError):
    """Ошибка при работе с хранилищем артефактов."""
    pass


class WorkspaceStoreError(MemoryError):
    """Ошибка при работе с хранилищем workspace."""
    pass


class StateStoreError(MemoryError):
    """Ошибка при работе с хранилищем состояния."""
    pass


class RetrievalError(MemoryError):
    """Ошибка при поиске и извлечении памяти."""
    pass


class IngestError(MemoryError):
    """Ошибка при обработке входящего события."""
    pass


class EmbeddingError(MemoryError):
    """Ошибка при работе с embeddings."""
    pass


class VectorIndexError(MemoryError):
    """Ошибка при работе с векторным индексом."""
    pass
