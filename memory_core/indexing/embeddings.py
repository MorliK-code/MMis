"""
Embeddings provider для memory_core.
"""

from typing import Any


class EmbeddingProvider:
    """
    Провайдер embeddings.
    
    Умеет:
    - embed text
    - batch embed
    - выдавать версию embedding model
    """
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Инициализирует провайдер embeddings.
        
        Args:
            model_name: Название модели для embeddings.
        """
        self.model_name = model_name
        self._model: Any = None
        self._initialized = False
    
    def _ensure_initialized(self) -> None:
        """Гарантирует инициализацию модели."""
        if self._initialized:
            return
        
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
            self._initialized = True
        except ImportError:
            # Если sentence-transformers не установлен, используем заглушку
            self._model = None
            self._initialized = True
    
    def embed(self, text: str) -> list[float]:
        """
        Создаёт embedding для текста.
        
        Args:
            text: Текст для embedding.
            
        Returns:
            Вектор embedding.
        """
        self._ensure_initialized()
        
        if self._model is None:
            # Заглушка: простой хэш-вектор
            return self._fallback_embed(text)
        
        embedding = self._model.encode(text, convert_to_numpy=True)
        return embedding.tolist()
    
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Создаёт embeddings для списка текстов.
        
        Args:
            texts: Список текстов.
            
        Returns:
            Список векторов embeddings.
        """
        self._ensure_initialized()
        
        if self._model is None:
            # Заглушка
            return [self._fallback_embed(text) for text in texts]
        
        embeddings = self._model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()
    
    def _fallback_embed(self, text: str) -> list[float]:
        """
        Создаёт fallback embedding (если модель не доступна).
        
        Args:
            text: Текст.
            
        Returns:
            Простой вектор на основе хэша.
        """
        # Простой хэш-вектор для демонстрации
        import hashlib
        
        # Создаём 384-мерный вектор (как MiniLM)
        vector = [0.0] * 384
        
        # Используем хэш для детерминированности
        for i, char in enumerate(text):
            idx = (ord(char) + i) % 384
            vector[idx] += 0.1
        
        # Нормализуем
        norm = sum(v * v for v in vector) ** 0.5
        if norm > 0:
            vector = [v / norm for v in vector]
        
        return vector
    
    def get_model_name(self) -> str:
        """Возвращает название модели."""
        return self.model_name
    
    def get_embedding_dimension(self) -> int:
        """Возвращает размерность embeddings."""
        if self._model is not None:
            try:
                return self._model.get_sentence_embedding_dimension()
            except Exception:
                pass
        return 384  # Default для MiniLM
