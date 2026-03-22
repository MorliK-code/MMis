"""
Vector Index - векторный индекс для поиска.
"""

import os
import json
from typing import Any
from memory_core.indexing.embeddings import EmbeddingProvider
from memory_core.errors import VectorIndexError


class VectorIndex:
    """
    Векторный индекс для semantic search.
    
    Индекс можно:
    - удалить
    - пересобрать
    - переэмбеддить
    
    Потому что истина не тут, а в raw events.
    """
    
    def __init__(
        self,
        index_path: str,
        embedding_provider: EmbeddingProvider | None = None,
    ):
        """
        Инициализирует векторный индекс.
        
        Args:
            index_path: Путь к директории индекса.
            embedding_provider: Провайдер embeddings.
        """
        self.index_path = index_path
        self.embedding_provider = embedding_provider or EmbeddingProvider()
        
        self._index: Any = None
        self._metadata: dict[str, Any] = {}
        
        # Создаём директорию
        if not os.path.exists(index_path):
            os.makedirs(index_path, exist_ok=True)
        
        self._load_or_create_index()
    
    def _load_or_create_index(self) -> None:
        """Загружает или создаёт индекс."""
        # Пока используем простой ин-memory индекс
        # В будущем можно подключить Chroma/Faiss
        self._index = []
        self._metadata = {
            "version": "1.0",
            "model": self.embedding_provider.get_model_name(),
            "vectors": 0,
        }
    
    def add(
        self,
        artifact_id: str,
        text: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Добавляет вектор в индекс.
        
        Args:
            artifact_id: ID артефакта.
            text: Текст для индексации.
            metadata: Метаданные.
        """
        # Создаём embedding
        embedding = self.embedding_provider.embed(text)
        
        # Добавляем в индекс
        self._index.append({
            "artifact_id": artifact_id,
            "embedding": embedding,
            "text": text,
            "metadata": metadata or {},
        })
        
        self._metadata["vectors"] = len(self._index)
    
    def add_batch(
        self,
        items: list[dict[str, Any]],
    ) -> None:
        """
        Добавляет пакет векторов в индекс.
        
        Args:
            items: Список items с artifact_id, text, metadata.
        """
        texts = [item["text"] for item in items]
        
        # Создаём embeddings batch
        embeddings = self.embedding_provider.embed_batch(texts)
        
        # Добавляем в индекс
        for item, embedding in zip(items, embeddings):
            self._index.append({
                "artifact_id": item["artifact_id"],
                "embedding": embedding,
                "text": item["text"],
                "metadata": item.get("metadata", {}),
            })
        
        self._metadata["vectors"] = len(self._index)
    
    def search(
        self,
        query: str,
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        """
        Ищет ближайшие векторы.
        
        Args:
            query: Поисковый запрос.
            top_k: Количество результатов.
            
        Returns:
            Список результатов.
        """
        if not self._index:
            return []
        
        # Создаём embedding запроса
        query_embedding = self.embedding_provider.embed(query)
        
        # Вычисляем схожесть (cosine similarity)
        scores = []
        for item in self._index:
            similarity = self._cosine_similarity(
                query_embedding,
                item["embedding"],
            )
            scores.append((item, similarity))
        
        # Сортируем по схожести
        scores.sort(key=lambda x: x[1], reverse=True)
        
        # Возвращаем top_k
        results = []
        for item, score in scores[:top_k]:
            results.append({
                "artifact_id": item["artifact_id"],
                "text": item["text"],
                "metadata": item["metadata"],
                "score": score,
            })
        
        return results
    
    def _cosine_similarity(
        self,
        vec1: list[float],
        vec2: list[float],
    ) -> float:
        """
        Вычисляет cosine similarity между векторами.
        
        Args:
            vec1: Первый вектор.
            vec2: Второй вектор.
            
        Returns:
            Cosine similarity.
        """
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a * a for a in vec1) ** 0.5
        norm2 = sum(b * b for b in vec2) ** 0.5
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def delete(self, artifact_id: str) -> None:
        """
        Удаляет вектор из индекса.
        
        Args:
            artifact_id: ID артефакта.
        """
        self._index = [
            item for item in self._index
            if item["artifact_id"] != artifact_id
        ]
        self._metadata["vectors"] = len(self._index)
    
    def clear(self) -> None:
        """Очищает индекс."""
        self._index = []
        self._metadata["vectors"] = 0
    
    def rebuild(
        self,
        artifacts: list[dict[str, Any]],
    ) -> None:
        """
        Перестраивает индекс из артефактов.
        
        Args:
            artifacts: Список артефактов для индексации.
        """
        self.clear()
        self.add_batch(artifacts)
    
    def get_metadata(self) -> dict[str, Any]:
        """Возвращает метаданные индекса."""
        return self._metadata.copy()
    
    def save(self) -> None:
        """Сохраняет индекс на диск."""
        # Сохраняем метаданные
        metadata_path = os.path.join(self.index_path, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(self._metadata, f, indent=2)
        
        # Сохраняем индекс (в реальном проекте лучше использовать специализированное хранилище)
        index_path = os.path.join(self.index_path, "index.json")
        # Не сохраняем embeddings для экономии места, пересоздаём при загрузке
        index_data = [
            {
                "artifact_id": item["artifact_id"],
                "text": item["text"],
                "metadata": item["metadata"],
            }
            for item in self._index
        ]
        with open(index_path, "w") as f:
            json.dump(index_data, f, indent=2)
    
    def load(self) -> None:
        """Загружает индекс с диска."""
        index_path = os.path.join(self.index_path, "index.json")
        
        if not os.path.exists(index_path):
            return
        
        with open(index_path, "r") as f:
            index_data = json.load(f)
        
        # Пересоздаём embeddings
        self._index = []
        for item in index_data:
            embedding = self.embedding_provider.embed(item["text"])
            self._index.append({
                "artifact_id": item["artifact_id"],
                "embedding": embedding,
                "text": item["text"],
                "metadata": item.get("metadata", {}),
            })
        
        self._metadata["vectors"] = len(self._index)
