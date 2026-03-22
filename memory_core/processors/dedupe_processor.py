"""
Dedupe Processor - снятие дубликатов.
"""

import hashlib
from memory_core.schemas import MemoryEnvelope, MemoryArtifact
from memory_core.constants import STATUS_ACTIVE, STATUS_SUPERSEDED


class DedupeProcessor:
    """
    Процессор дедупликации.
    
    Снимает дубли, чтобы память не превращалась в мусорку
    из 40 одинаковых фактов:
    - "Пользователь работает в VS Code"
    - "Пользователь юзает VS Code"
    - "Основной редактор VS Code"
    - "Использует Visual Studio Code"
    
    Должна остаться 1 нормализованная запись.
    """
    
    def __init__(self, similarity_threshold: float = 0.85):
        """
        Инициализирует процессор дедупликации.
        
        Args:
            similarity_threshold: Порог схожести для дедупликации.
        """
        self.similarity_threshold = similarity_threshold
    
    def process(self, envelope: MemoryEnvelope) -> list[MemoryArtifact]:
        """
        Обработка для dedupe (на этапе ingest не применяется).
        
        Args:
            envelope: Конверт события.
            
        Returns:
            Пустой список (dedupe применяется к существующим артефактам).
        """
        # Dedupe применяется к уже существующим артефактам
        # через метод deduplicate_artifacts
        return []
    
    def deduplicate_artifacts(
        self,
        artifacts: list[MemoryArtifact],
    ) -> list[MemoryArtifact]:
        """
        Убирает дубликаты из списка артефактов.
        
        Args:
            artifacts: Список артефактов.
            
        Returns:
            Список без дубликатов.
        """
        if not artifacts:
            return []
        
        # Группируем по типу
        by_type: dict[str, list[MemoryArtifact]] = {}
        for artifact in artifacts:
            if artifact.artifact_type not in by_type:
                by_type[artifact.artifact_type] = []
            by_type[artifact.artifact_type].append(artifact)
        
        result = []
        
        for artifact_type, type_artifacts in by_type.items():
            # Дедуплицируем внутри типа
            deduped = self._dedupe_by_type(type_artifacts)
            result.extend(deduped)
        
        return result
    
    def _dedupe_by_type(
        self,
        artifacts: list[MemoryArtifact],
    ) -> list[MemoryArtifact]:
        """
        Убирает дубликаты внутри одного типа.
        
        Args:
            artifacts: Список артефактов одного типа.
            
        Returns:
            Список без дубликатов.
        """
        if len(artifacts) <= 1:
            return artifacts
        
        # Используем хэш нормализованного текста для быстрого сравнения
        seen_hashes: dict[str, MemoryArtifact] = {}
        result = []
        
        for artifact in artifacts:
            normalized = self._normalize_text(artifact.text)
            hash_key = self._hash_text(normalized)
            
            if hash_key in seen_hashes:
                # Дубликат - помечаем как superseded
                existing = seen_hashes[hash_key]
                
                # Оставляем более новый или более длинный
                if artifact.created_at > existing.created_at:
                    # Новый заменяет старый
                    result.remove(existing)
                    result.append(artifact)
                    seen_hashes[hash_key] = artifact
                # Иначе оставляем старый, новый игнорируем
            else:
                seen_hashes[hash_key] = artifact
                result.append(artifact)
        
        return result
    
    def _normalize_text(self, text: str) -> str:
        """
        Нормализует текст для сравнения.
        
        Args:
            text: Исходный текст.
            
        Returns:
            Нормализованный текст.
        """
        # Приводим к нижнему регистру
        text = text.lower()
        
        # Удаляем лишние пробелы
        text = " ".join(text.split())
        
        # Удаляем пунктуацию
        import string
        text = text.translate(str.maketrans("", "", string.punctuation))
        
        # Сортируем слова для нечувствительности к порядку
        words = sorted(text.split())
        
        return " ".join(words)
    
    def _hash_text(self, text: str) -> str:
        """
        Создаёт хэш текста.
        
        Args:
            text: Текст.
            
        Returns:
            MD5 хэш.
        """
        return hashlib.md5(text.encode()).hexdigest()
    
    def find_similar(
        self,
        artifact: MemoryArtifact,
        candidates: list[MemoryArtifact],
    ) -> list[MemoryArtifact]:
        """
        Находит похожие артефакты.
        
        Args:
            artifact: Артефакт для поиска.
            candidates: Кандидаты для сравнения.
            
        Returns:
            Список похожих артефактов.
        """
        similar = []
        normalized_ref = self._normalize_text(artifact.text)
        
        for candidate in candidates:
            normalized_cand = self._normalize_text(candidate.text)
            
            # Простая метрика схожести по Jaccard
            similarity = self._jaccard_similarity(
                set(normalized_ref.split()),
                set(normalized_cand.split()),
            )
            
            if similarity >= self.similarity_threshold:
                similar.append(candidate)
        
        return similar
    
    def _jaccard_similarity(
        self,
        set1: set[str],
        set2: set[str],
    ) -> float:
        """
        Вычисляет схожесть Jaccard между множествами.
        
        Args:
            set1: Первое множество.
            set2: Второе множество.
            
        Returns:
            Коэффициент схожести (0.0 - 1.0).
        """
        if not set1 and not set2:
            return 1.0
        
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        return intersection / union if union > 0 else 0.0
    
    @property
    def name(self) -> str:
        """Возвращает имя процессора."""
        return "dedupe_processor"
    
    @property
    def priority(self) -> int:
        """Возвращает приоритет."""
        return 100  # Низкий приоритет, применяется последним
