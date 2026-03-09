from __future__ import annotations

import re
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from memory.memory_models import (
    ChunkRecord,
    DocumentIngestRequest,
    DocumentIngestResult,
    DocumentRecord,
    MemoryLevel,
    MemoryRecord,
    MemoryType,
)
from memory.vector_store import VectorStore


_CODE_EXT_LANG = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".cs": "csharp",
}


@dataclass
class ChunkingConfig:
    chunk_size: int = 1200
    chunk_overlap: int = 160


class DocumentMemory:
    def __init__(self, *, store: VectorStore, chunking: ChunkingConfig | None = None):
        self.store = store
        self.chunking = chunking or ChunkingConfig()

    def ingest_document(self, request: DocumentIngestRequest) -> DocumentIngestResult:
        text = str(request.text or "").strip()
        if not text:
            raise ValueError("Document text must be non-empty")

        now_ts = float(time.time())
        doc_id = f"doc:{uuid.uuid4().hex[:18]}"
        summary = self._summarize(text)

        document = DocumentRecord(
            id=doc_id,
            source=str(request.source or "document"),
            text=text,
            summary=summary,
            scope=request.scope,
            namespace=str(request.namespace or "default"),
            metadata=dict(request.metadata or {}),
            created_at=now_ts,
            updated_at=now_ts,
        )

        chunk_rows = self.chunk_document(
            text=text,
            metadata=dict(request.metadata or {}),
            chunk_size=int(self.chunking.chunk_size),
            chunk_overlap=int(self.chunking.chunk_overlap),
            namespace=str(request.namespace or "default"),
            scope=request.scope,
            document_id=doc_id,
        )

        records: list[MemoryRecord] = [
            MemoryRecord(
                id=doc_id,
                text=summary,
                memory_type=MemoryType.DOCUMENT,
                level=MemoryLevel.L4_DOCUMENT,
                scope=request.scope,
                namespace=str(request.namespace or "default"),
                metadata={
                    "document_source": document.source,
                    "summary": summary,
                    **dict(request.metadata or {}),
                },
                importance=0.72,
                confidence=0.86,
                created_at=now_ts,
                updated_at=now_ts,
            )
        ]

        for chunk in chunk_rows:
            records.append(
                MemoryRecord(
                    id=chunk.id,
                    text=chunk.text,
                    memory_type=MemoryType.DOCUMENT_CHUNK,
                    level=MemoryLevel.L4_DOCUMENT,
                    scope=chunk.scope,
                    namespace=chunk.namespace,
                    metadata={
                        "document_id": chunk.document_id,
                        "chunk_index": chunk.chunk_index,
                        **dict(chunk.metadata or {}),
                    },
                    importance=0.66,
                    confidence=0.84,
                    created_at=now_ts,
                    updated_at=now_ts,
                    parent_id=chunk.document_id,
                    chunk_index=chunk.chunk_index,
                )
            )

        self.store.batch_upsert(records)
        return DocumentIngestResult(document=document, chunks=chunk_rows)

    def chunk_document(
        self,
        *,
        text: str,
        metadata: dict[str, Any],
        chunk_size: int,
        chunk_overlap: int,
        namespace: str,
        scope,
        document_id: str,
    ) -> list[ChunkRecord]:
        src = str(text or "")
        language = self._detect_language(metadata=metadata, text=src)
        chunks: list[str]
        if language:
            chunks = self._chunk_code(src, language=language, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        else:
            chunks = self._chunk_plain(src, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        out: list[ChunkRecord] = []
        for idx, chunk_text in enumerate(chunks):
            piece = str(chunk_text or "").strip()
            if not piece:
                continue
            out.append(
                ChunkRecord(
                    id=f"chunk:{document_id}:{idx}",
                    document_id=document_id,
                    chunk_index=idx,
                    text=piece,
                    scope=scope,
                    namespace=namespace,
                    metadata={"language": language},
                )
            )
        return out

    @staticmethod
    def _summarize(text: str, max_chars: int = 360) -> str:
        src = str(text or "").strip()
        if not src:
            return ""
        lines = [x.strip() for x in src.splitlines() if x.strip()]
        if not lines:
            return src[:max_chars]
        head = " ".join(lines[:3]).strip()
        if len(head) <= max_chars:
            return head
        return head[: max_chars - 3].rstrip() + "..."

    @staticmethod
    def _chunk_plain(text: str, *, chunk_size: int, chunk_overlap: int) -> list[str]:
        src = str(text or "")
        n = max(256, int(chunk_size))
        overlap = max(0, min(n // 2, int(chunk_overlap)))
        if len(src) <= n:
            return [src]

        out: list[str] = []
        i = 0
        while i < len(src):
            end = min(len(src), i + n)
            piece = src[i:end]
            if end < len(src):
                split = max(piece.rfind("\n\n"), piece.rfind("\n"), piece.rfind(". "), piece.rfind(" "))
                if split > 120:
                    piece = piece[:split]
                    end = i + split
            out.append(piece.strip())
            if end >= len(src):
                break
            i = max(0, end - overlap)
        return [x for x in out if x]

    def _chunk_code(self, text: str, *, language: str, chunk_size: int, chunk_overlap: int) -> list[str]:
        sections = self._sections_from_tree_sitter(text=text, language=language)
        if not sections:
            sections = self._sections_from_regex(text=text, language=language)
        if not sections:
            return self._chunk_plain(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        chunks: list[str] = []
        cur = ""
        limit = max(280, int(chunk_size))
        overlap = max(0, min(limit // 2, int(chunk_overlap)))

        for section in sections:
            part = str(section or "").strip()
            if not part:
                continue
            if not cur:
                cur = part
                continue
            if len(cur) + len(part) + 2 <= limit:
                cur = f"{cur}\n\n{part}".strip()
            else:
                chunks.append(cur)
                if overlap > 0 and len(cur) > overlap:
                    tail = cur[-overlap:]
                    cur = f"{tail}\n{part}".strip()
                else:
                    cur = part
        if cur:
            chunks.append(cur)
        return [x for x in chunks if x]

    def _sections_from_tree_sitter(self, *, text: str, language: str) -> list[str]:
        # Optional path: if tree-sitter grammars are installed we can improve sectioning.
        # On missing grammars/runtime we safely fall back to regex-based sectioning.
        _ = (text, language)
        try:
            import tree_sitter  # type: ignore  # noqa: F401
        except Exception:
            return []
        return []

    @staticmethod
    def _sections_from_regex(*, text: str, language: str) -> list[str]:
        src = str(text or "")
        if not src.strip():
            return []

        patterns: list[str] = []
        if language == "python":
            patterns = [r"^\s*class\s+\w+", r"^\s*def\s+\w+"]
        elif language in {"javascript", "typescript"}:
            patterns = [
                r"^\s*export\s+class\s+\w+",
                r"^\s*class\s+\w+",
                r"^\s*function\s+\w+",
                r"^\s*const\s+\w+\s*=\s*\(",
            ]
        elif language == "go":
            patterns = [r"^\s*type\s+\w+\s+struct", r"^\s*func\s+\(", r"^\s*func\s+\w+"]
        elif language == "rust":
            patterns = [r"^\s*struct\s+\w+", r"^\s*impl\s+\w+", r"^\s*fn\s+\w+"]
        elif language == "java":
            patterns = [r"^\s*(public\s+)?class\s+\w+", r"^\s*(public|private|protected)\s+.*\(" ]
        elif language == "csharp":
            patterns = [r"^\s*(public\s+)?class\s+\w+", r"^\s*(public|private|protected)\s+.*\(" ]

        if not patterns:
            return []

        lines = src.splitlines()
        indexes = [0]
        for idx, line in enumerate(lines):
            for pattern in patterns:
                if re.search(pattern, line):
                    indexes.append(idx)
                    break
        indexes.append(len(lines))
        indexes = sorted(set(indexes))

        sections: list[str] = []
        for i in range(len(indexes) - 1):
            start = indexes[i]
            end = indexes[i + 1]
            if start >= end:
                continue
            part = "\n".join(lines[start:end]).strip()
            if part:
                sections.append(part)
        return sections

    @staticmethod
    def _detect_language(*, metadata: dict[str, Any], text: str) -> str:
        hint = str(metadata.get("language") or metadata.get("lang") or "").strip().lower()
        if hint in {"python", "javascript", "typescript", "go", "rust", "java", "csharp"}:
            return hint

        source_path = str(metadata.get("path") or metadata.get("source_path") or "").strip()
        ext = Path(source_path).suffix.lower()
        if ext in _CODE_EXT_LANG:
            return _CODE_EXT_LANG[ext]

        src = str(text or "")
        if "def " in src and "import " in src:
            return "python"
        if "function " in src or "=>" in src:
            return "javascript"
        if "package " in src and "func " in src:
            return "go"
        if "impl " in src and "fn " in src:
            return "rust"
        if "public class" in src and ";" in src:
            return "java"
        if "namespace " in src and "class " in src:
            return "csharp"
        return ""
