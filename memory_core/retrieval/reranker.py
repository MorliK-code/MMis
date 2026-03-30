"""
Reranker - переупорядочивание результатов retrieval.
"""

from memory_core.schemas import MemoryArtifact
from memory_core.retrieval.query_models import RetrievalFilters


class Reranker:
    """
    Переупорядочиватель результатов поиска.
    
    Улучшает ранжирование найденных артефактов
    на основе релевантности запросу.
    """
    
    def __init__(self):
        """Инициализирует reranker."""
        # Веса для различных факторов
        self.weights = {
            "text_similarity": 0.20,
            "semantic_score": 0.30,
            "recency": 0.15,
            "confidence": 0.15,
            "session_match": 0.10,
            "retrieve_when_match": 0.05,
            "profile_boost": 0.03,
            "task_boost": 0.02,
        }
    
    def rerank(
        self,
        artifacts: list[MemoryArtifact],
        query: str,
        top_k: int = 8,
        session_id: str | None = None,
    ) -> list[MemoryArtifact]:
        """
        Переупорядочивает артефакты по релевантности.

        Args:
            artifacts: Список артефактов.
            query: Поисковый запрос.
            top_k: Количество лучших результатов.
            session_id: ID сессии для session match.

        Returns:
            Переупорядоченный список.
        """
        if not artifacts:
            return []

        # Вычисляем scores для каждого артефакта
        scored = []
        for artifact in artifacts:
            score = self._compute_score(artifact, query, session_id)
            scored.append((artifact, score))

        # Сортируем по score
        scored.sort(key=lambda x: x[1], reverse=True)

        # Возвращаем top_k
        return [a for a, _ in scored[:top_k]]

    def _compute_score(
        self,
        artifact: MemoryArtifact,
        query: str,
        session_id: str | None = None,
    ) -> float:
        """
        Вычисляет score для артефакта.

        Args:
            artifact: Артефакт.
            query: Поисковый запрос.
            session_id: ID сессии.

        Returns:
            Score (0.0 - 1.0).
        """
        import time

        score = 0.0
        meta = artifact.metadata or {}

        # Text similarity
        text_sim = self._text_similarity(artifact.text, query)
        score += self.weights["text_similarity"] * text_sim

        # Semantic score (из vector index)
        semantic_score = float(meta.get("semantic_score", 0.0))
        score += self.weights["semantic_score"] * semantic_score

        # Recency (более новые выше)
        recency = self._recency_score(artifact.created_at)
        score += self.weights["recency"] * recency

        # Confidence
        confidence = float(meta.get("confidence", 0.5))
        score += self.weights["confidence"] * confidence

        # Session match
        if session_id and meta.get("session_id") == session_id:
            score += self.weights["session_match"]

        # Topic-aware reranking hook attached by RetrievalService.
        score += max(0.0, min(float(meta.get("_topic_boost", 0.0) or 0.0), 0.5))

        # retrieve_when overlap
        retrieve_when = {str(x).lower() for x in meta.get("retrieve_when", [])}
        query_words = {w for w in query.lower().split() if len(w) >= 3}
        if retrieve_when & query_words:
            score += self.weights["retrieve_when_match"]

        # Boost для profile facts
        if artifact.artifact_type == "profile_fact":
            score += self.weights["profile_boost"]

        # Boost для tasks
        if artifact.artifact_type in {"task", "task_state"}:
            score += self.weights["task_boost"]

        return min(score, 1.0)
    
    def _text_similarity(self, text: str, query: str) -> float:
        """
        Вычисляет схожесть текста с запросом.
        
        Args:
            text: Текст артефакта.
            query: Поисковый запрос.
            
        Returns:
            Схожесть (0.0 - 1.0).
        """
        if not text or not query:
            return 0.0
        
        text_lower = text.lower()
        query_lower = query.lower()
        
        # Простая метрика: наличие слов запроса в тексте
        query_words = query_lower.split()
        if not query_words:
            return 0.0
        
        matches = sum(1 for word in query_words if word in text_lower)
        return matches / len(query_words)
    
    def _recency_score(self, created_at: float) -> float:
        """
        Вычисляет score на основе давности.
        
        Args:
            created_at: Время создания артефакта.
            
        Returns:
            Recency score (0.0 - 1.0).
        """
        import time
        
        now = time.time()
        age_seconds = now - created_at
        
        # exponential decay с half-life 1 день
        half_life = 24 * 60 * 60  # 1 день в секундах
        decay = 0.5 ** (age_seconds / half_life)
        
        return decay
