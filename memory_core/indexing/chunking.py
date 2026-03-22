"""
Chunking - разбиение текста на части.
"""

import re
from typing import Iterator


def chunk_text(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> list[str]:
    """
    Разбивает текст на чанки.
    
    Args:
        text: Текст для разбиения.
        chunk_size: Размер чанка в символах.
        chunk_overlap: Перекрытие между чанками.
        
    Returns:
        Список чанков.
    """
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        # Пытаемся разбить по предложению
        if end < len(text):
            # Ищем ближайший конец предложения
            sentence_end = max(
                text.rfind(". ", start, end),
                text.rfind("! ", start, end),
                text.rfind("? ", start, end),
                text.rfind("\n", start, end),
            )
            
            if sentence_end > start + chunk_size // 2:
                end = sentence_end + 1
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        start = end - chunk_overlap
    
    return chunks


def chunk_by_sentences(text: str) -> list[str]:
    """
    Разбивает текст на предложения.
    
    Args:
        text: Текст.
        
    Returns:
        Список предложений.
    """
    # Разбиваем по концам предложений
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    # Очищаем и фильтруем пустые
    return [s.strip() for s in sentences if s.strip()]


def chunk_by_paragraphs(text: str) -> list[str]:
    """
    Разбивает текст на параграфы.
    
    Args:
        text: Текст.
        
    Returns:
        Список параграфов.
    """
    # Разбиваем по двойным newline
    paragraphs = re.split(r'\n\s*\n', text)
    
    # Очищаем и фильтруем пустые
    return [p.strip() for p in paragraphs if p.strip()]


def chunk_iterative(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> Iterator[str]:
    """
    Итеративно разбивает текст на чанки.
    
    Args:
        text: Текст.
        chunk_size: Размер чанка.
        chunk_overlap: Перекрытие.
        
    Yields:
        Чанки текста.
    """
    if len(text) <= chunk_size:
        yield text
        return
    
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        # Пытаемся разбить по предложению
        if end < len(text):
            sentence_end = max(
                text.rfind(". ", start, end),
                text.rfind("! ", start, end),
                text.rfind("? ", start, end),
                text.rfind("\n", start, end),
            )
            
            if sentence_end > start + chunk_size // 2:
                end = sentence_end + 1
        
        chunk = text[start:end].strip()
        if chunk:
            yield chunk
        
        start = end - chunk_overlap
