"""
Indexing layer для memory_core.
"""

from memory_core.indexing.embeddings import EmbeddingProvider
from memory_core.indexing.vector_index import VectorIndex
from memory_core.indexing.chunking import chunk_text

__all__ = [
    "EmbeddingProvider",
    "VectorIndex",
    "chunk_text",
]
