"""Standalone document chunking helpers for Document Memory."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class ChunkingConfig:
    chunk_size: int = 1200
    chunk_overlap: int = 160


@dataclass
class DocumentChunker:
    config: ChunkingConfig | None = None

    def __post_init__(self) -> None:
        self.config = self.config or ChunkingConfig()

    def chunk_spans(
        self,
        text: str,
        *,
        language: str = "",
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> list[tuple[str, int, int]]:
        src = str(text or "")
        size = int(chunk_size or self.config.chunk_size)
        overlap = int(chunk_overlap or self.config.chunk_overlap)
        lang = str(language or "").strip().lower()
        if lang:
            raw = self._chunk_code(src, language=lang, chunk_size=size, chunk_overlap=overlap)
            spans = self._approximate_spans(src, raw)
            if spans:
                return spans
        return self._chunk_plain_spans(src, chunk_size=size, chunk_overlap=overlap)

    @staticmethod
    def _chunk_plain(text: str, *, chunk_size: int, chunk_overlap: int) -> list[str]:
        return [
            row[0]
            for row in DocumentChunker._chunk_plain_spans(
                text,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        ]

    @staticmethod
    def _chunk_plain_spans(text: str, *, chunk_size: int, chunk_overlap: int) -> list[tuple[str, int, int]]:
        src = str(text or "")
        n = max(256, int(chunk_size))
        overlap = max(0, min(n // 2, int(chunk_overlap)))
        if len(src) <= n:
            return [(src, 0, len(src))]

        out: list[tuple[str, int, int]] = []
        i = 0
        while i < len(src):
            end = min(len(src), i + n)
            piece = src[i:end]
            if end < len(src):
                split = max(piece.rfind("\n\n"), piece.rfind("\n"), piece.rfind(". "), piece.rfind(" "))
                if split > 120:
                    piece = piece[:split]
                    end = i + split
            out.append((piece.strip(), i, end))
            if end >= len(src):
                break
            i = max(0, end - overlap)
        return [x for x in out if str(x[0]).strip()]

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

    @staticmethod
    def _approximate_spans(text: str, chunks: list[str]) -> list[tuple[str, int, int]]:
        src = str(text or "")
        out: list[tuple[str, int, int]] = []
        cursor = 0
        for piece in list(chunks or []):
            part = str(piece or "")
            if not part:
                continue
            idx = src.find(part, cursor)
            if idx < 0:
                idx = max(0, cursor)
            end = min(len(src), idx + len(part))
            out.append((part, idx, end))
            cursor = max(cursor, end)
        return out

    def _sections_from_tree_sitter(self, *, text: str, language: str) -> list[str]:
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
            patterns = [r"^\s*(public\s+)?class\s+\w+", r"^\s*(public|private|protected)\s+.*\("]
        elif language == "csharp":
            patterns = [r"^\s*(public\s+)?class\s+\w+", r"^\s*(public|private|protected)\s+.*\("]

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
