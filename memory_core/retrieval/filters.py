"""
Фильтры для retrieval.
"""

import time
from typing import Any
from memory_core.schemas import MemoryArtifact
from memory_core.retrieval.query_models import RetrievalFilters


def apply_filters(
    artifacts: list[MemoryArtifact],
    filters: RetrievalFilters,
) -> list[MemoryArtifact]:
    """
    Применяет фильтры к списку артефактов.
    
    Args:
        artifacts: Список артефактов.
        filters: Фильтры.
        
    Returns:
        Отфильтрованный список артефактов.
    """
    result = artifacts
    
    # Фильтр по типу
    if filters.artifact_types:
        result = [a for a in result if a.artifact_type in filters.artifact_types]
    
    # Фильтр по workspace
    if filters.workspace_id:
        result = [a for a in result if a.workspace_id == filters.workspace_id]
    
    # Фильтр по namespace
    if filters.namespace:
        result = [a for a in result if a.namespace == filters.namespace]
    
    # Фильтр по статусу
    if filters.status:
        result = [a for a in result if a.status == filters.status]
    
    # Фильтр по source event
    if filters.source_event_id:
        result = [a for a in result if a.source_event_id == filters.source_event_id]
    
    # Фильтр по confidence
    if filters.min_confidence is not None:
        result = [
            a for a in result
            if a.metadata.get("confidence", 0) >= filters.min_confidence
        ]
    
    # Фильтр по возрасту
    if filters.max_age_days is not None:
        now = time.time()
        max_age_seconds = filters.max_age_days * 24 * 60 * 60
        result = [
            a for a in result
            if (now - a.created_at) <= max_age_seconds
        ]
    
    return result


def filter_by_text_similarity(
    artifacts: list[MemoryArtifact],
    query: str,
    threshold: float = 0.3,
) -> list[tuple[MemoryArtifact, float]]:
    """
    Фильтрует артефакты по схожести с запросом.
    
    Args:
        artifacts: Список артефактов.
        query: Поисковый запрос.
        threshold: Порог схожести.
        
    Returns:
        Список кортежей (артефакт, схожесть).
    """
    results = []
    query_words = set(query.lower().split())
    
    for artifact in artifacts:
        text = artifact.text.lower()
        summary = artifact.summary.lower()
        
        # Простая метрика: пересечение слов
        text_words = set(text.split())
        summary_words = set(summary.split())
        
        all_words = text_words | summary_words
        common_words = query_words & all_words
        
        if not all_words:
            continue
        
        similarity = len(common_words) / len(all_words)
        
        if similarity >= threshold:
            results.append((artifact, similarity))
    
    # Сортируем по схожести
    results.sort(key=lambda x: x[1], reverse=True)
    
    return results


def filter_by_keyword_match(
    artifacts: list[MemoryArtifact],
    keywords: list[str],
) -> list[MemoryArtifact]:
    """
    Фильтрует артефакты по наличию ключевых слов.
    
    Args:
        artifacts: Список артефактов.
        keywords: Ключевые слова.
        
    Returns:
        Отфильтрованный список.
    """
    if not keywords:
        return artifacts
    
    keywords_lower = [k.lower() for k in keywords]
    
    result = []
    for artifact in artifacts:
        text_lower = artifact.text.lower()
        summary_lower = artifact.summary.lower()
        
        # Проверяем наличие любого ключевого слова
        if any(kw in text_lower or kw in summary_lower for kw in keywords_lower):
            result.append(artifact)
    
    return result
