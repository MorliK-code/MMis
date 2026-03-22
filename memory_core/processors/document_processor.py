"""
Document Processor - обработка документов.
"""

import re
from typing import Any
from memory_core.schemas import MemoryEnvelope, MemoryArtifact, DocumentIngestRequest
from memory_core.constants import (
    ARTIFACT_DOCUMENT_CHUNK,
    ARTIFACT_DOCUMENT_SUMMARY,
    STATUS_ACTIVE,
)


class DocumentProcessor:
    """
    Процессор документов.
    
    Для ingest документов:
    - chunking (разбиение на части)
    - summary (краткое содержание)
    - document-level metadata
    - запись document_chunk артефактов
    - запись document_summary артефактов
    - регистрация source в workspace
    """
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """
        Инициализирует процессор документов.
        
        Args:
            chunk_size: Размер чанка в символах.
            chunk_overlap: Перекрытие между чанками.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def process(self, envelope: MemoryEnvelope) -> list[MemoryArtifact]:
        """
        Обрабатывает событие с документом.
        
        Args:
            envelope: Конверт события.
            
        Returns:
            Список артефактов документа.
        """
        artifacts = []
        
        # Проверяем, что это документ
        if envelope.source_kind != "document":
            return artifacts
        
        content = envelope.text
        metadata = envelope.metadata
        
        # Разбиваем на чанки
        chunks = self._chunk_text(content)
        
        # Создаём артефакты для каждого чанка
        for i, chunk in enumerate(chunks):
            chunk_artifact = MemoryArtifact(
                artifact_type=ARTIFACT_DOCUMENT_CHUNK,
                source_event_id=envelope.event_id,
                text=chunk,
                summary=f"Chunk {i+1}/{len(chunks)}",
                metadata={
                    "extracted_by": "document_processor",
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "document_id": metadata.get("document_id", ""),
                    "document_title": metadata.get("title", ""),
                },
                namespace=envelope.namespace,
                workspace_id=envelope.workspace_id,
                status=STATUS_ACTIVE,
            )
            artifacts.append(chunk_artifact)
        
        # Создаём summary документа
        summary = self._summarize_document(content)
        if summary:
            summary_artifact = MemoryArtifact(
                artifact_type=ARTIFACT_DOCUMENT_SUMMARY,
                source_event_id=envelope.event_id,
                text=content[:1000],  # Первые 1000 символов
                summary=summary,
                metadata={
                    "extracted_by": "document_processor",
                    "document_id": metadata.get("document_id", ""),
                    "document_title": metadata.get("title", ""),
                    "content_length": len(content),
                },
                namespace=envelope.namespace,
                workspace_id=envelope.workspace_id,
                status=STATUS_ACTIVE,
            )
            artifacts.append(summary_artifact)
        
        return artifacts
    
    def process_document(
        self,
        request: DocumentIngestRequest,
        envelope: MemoryEnvelope,
    ) -> list[MemoryArtifact]:
        """
        Обрабатывает документ из запроса.
        
        Args:
            request: Запрос на ingest документа.
            envelope: Конверт события.
            
        Returns:
            Список артефактов документа.
        """
        # Добавляем метаданные документа в envelope
        envelope.metadata["document_id"] = request.document_id
        envelope.metadata["title"] = request.title
        envelope.metadata["source_kind"] = request.source_kind
        envelope.metadata.update(request.metadata)
        
        # Обновляем текст документа
        envelope.text = request.content
        
        return self.process(envelope)
    
    def _chunk_text(self, text: str) -> list[str]:
        """
        Разбивает текст на чанки.
        
        Args:
            text: Текст для разбиения.
            
        Returns:
            Список чанков.
        """
        if len(text) <= self.chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            # Пытаемся разбить по предложению
            if end < len(text):
                # Ищем ближайший конец предложения
                sentence_end = max(
                    text.rfind(". ", start, end),
                    text.rfind("! ", start, end),
                    text.rfind("? ", start, end),
                    text.rfind("\n", start, end),
                )
                
                if sentence_end > start + self.chunk_size // 2:
                    end = sentence_end + 1
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end - self.chunk_overlap
        
        return chunks
    
    def _summarize_document(self, content: str) -> str:
        """
        Создаёт краткое summary документа.
        
        Args:
            content: Содержимое документа.
            
        Returns:
            Краткое summary.
        """
        if not content:
            return ""
        
        # Берём первые несколько предложений
        sentences = re.split(r'[.!?]', content)
        summary_sentences = [s.strip() for s in sentences[:3] if s.strip()]
        
        if not summary_sentences:
            return content[:200] + "..." if len(content) > 200 else content
        
        summary = ". ".join(summary_sentences) + "."
        
        # Ограничиваем длину
        if len(summary) > 500:
            summary = summary[:500] + "..."
        
        return summary
    
    @property
    def name(self) -> str:
        """Возвращает имя процессора."""
        return "document_processor"
    
    @property
    def priority(self) -> int:
        """Возвращает приоритет."""
        return 25
