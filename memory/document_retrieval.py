from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from memory.document_models import DocumentClaim, DocumentSummary
from memory.memory_models import MemoryRecord, MemoryScope, MemoryType, RetrievalQuery
from memory.vector_store import VectorStore
from modules.nlu.normalizer import normalize_text


_WHERE_QUERY_RE = re.compile(r"(?:\b(?:where|locate|find|called|declared)\b|где|вызывается|объявляется)", re.I)
_CODE_QUERY_RE = re.compile(r"(?:\b(?:code|class|function|method|call|import|declared)\b|код|класс|функц|метод|вызывается|объявляется)", re.I)
_CHAPTER_QUERY_RE = re.compile(r"(?:\bchapter\s+(\d+)\b|глава\s+(\d+)|chapter\s+([ivxlcdm]+))", re.I)
_DOCUMENT_QUERY_STOPWORDS = {
    "a",
    "about",
    "and",
    "call",
    "called",
    "chapter",
    "class",
    "code",
    "declared",
    "find",
    "function",
    "in",
    "is",
    "method",
    "of",
    "the",
    "what",
    "where",
    "было",
    "в",
    "вызове",
    "где",
    "глава",
    "главе",
    "как",
    "класс",
    "коде",
    "что",
}


@dataclass(frozen=True)
class DocumentRetrievalHints:
    query_text: str
    query_tokens: set[str]
    chapter_number: str = ""
    location_query: bool = False
    code_query: bool = False


@dataclass(frozen=True)
class DocumentRetrievalHit:
    document_id: str
    score: float
    document_record: MemoryRecord | None = None
    relevant_chunks: list[MemoryRecord] = field(default_factory=list)
    section_summary_record: MemoryRecord | None = None
    section_summary: DocumentSummary | None = None
    document_claim_records: list[MemoryRecord] = field(default_factory=list)
    document_claims: list[DocumentClaim] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "document_id": str(self.document_id or ""),
            "score": float(self.score),
            "document_record": (self.document_record.to_dict() if self.document_record is not None else None),
            "relevant_chunks": [row.to_dict() for row in list(self.relevant_chunks or [])],
            "section_summary_record": (self.section_summary_record.to_dict() if self.section_summary_record is not None else None),
            "section_summary": (self.section_summary.to_dict() if self.section_summary is not None else None),
            "document_claim_records": [row.to_dict() for row in list(self.document_claim_records or [])],
            "document_claims": [row.to_dict() for row in list(self.document_claims or [])],
        }


@dataclass
class DocumentRetriever:
    store: VectorStore
    chunk_limit: int = 4
    claim_limit: int = 3

    def retrieve(
        self,
        *,
        query: RetrievalQuery,
        query_text: str | None = None,
        namespace: str | None = None,
        scopes: list[MemoryScope] | None = None,
        top_k: int | None = None,
        filters: dict[str, object] | None = None,
    ) -> list[DocumentRetrievalHit]:
        query_text_value = str(query_text or query.search_text or query.query_text or "").strip()
        namespace_value = str(namespace or query.namespace or "default")
        scope_values = list(scopes or query.scopes or [])
        top_k_value = max(1, int(top_k or query.top_k or 4))
        hints = self.build_query_hints(query)

        chunk_rows = self._retrieve_channel(
            query_text=query_text_value,
            namespace=namespace_value,
            scopes=scope_values,
            top_k=top_k_value,
            filters={**dict(filters or {}), "memory_type": [MemoryType.DOCUMENT_CHUNK.value]},
            source="document_chunk_channel",
        )
        summary_rows = self._retrieve_channel(
            query_text=query_text_value,
            namespace=namespace_value,
            scopes=scope_values,
            top_k=top_k_value,
            filters={**dict(filters or {}), "memory_type": [MemoryType.SUMMARY.value]},
            source="document_summary_channel",
        )
        claim_rows = self._retrieve_channel(
            query_text=query_text_value,
            namespace=namespace_value,
            scopes=scope_values,
            top_k=top_k_value,
            filters={**dict(filters or {}), "memory_type": [MemoryType.CLAIM.value]},
            source="document_claim_channel",
        )

        document_index = self._document_records(namespace=namespace_value)
        grouped: dict[str, dict[str, Any]] = {}

        for row in list(chunk_rows or []):
            record = row.get("record")
            if not isinstance(record, MemoryRecord):
                continue
            doc_id = self._record_document_id(record)
            if not doc_id:
                continue
            bucket = grouped.setdefault(
                doc_id,
                {
                    "document_record": document_index.get(doc_id),
                    "chunks": [],
                    "summaries": [],
                    "claims": [],
                    "score": 0.0,
                },
            )
            score = self._chunk_score(record=record, hints=hints, base=self._base_score(row))
            bucket["chunks"].append((score, record))
            bucket["score"] = max(float(bucket["score"]), score)

        for row in list(summary_rows or []):
            record = row.get("record")
            if not isinstance(record, MemoryRecord) or not self._is_document_summary_record(record):
                continue
            doc_id = self._record_document_id(record)
            if not doc_id:
                continue
            bucket = grouped.setdefault(
                doc_id,
                {
                    "document_record": document_index.get(doc_id),
                    "chunks": [],
                    "summaries": [],
                    "claims": [],
                    "score": 0.0,
                },
            )
            score = self._summary_score(record=record, hints=hints, base=self._base_score(row))
            bucket["summaries"].append((score, record))
            bucket["score"] = max(float(bucket["score"]), score)

        for row in list(claim_rows or []):
            record = row.get("record")
            if not isinstance(record, MemoryRecord) or not self._is_document_claim_record(record):
                continue
            doc_id = self._record_document_id(record)
            if not doc_id:
                continue
            bucket = grouped.setdefault(
                doc_id,
                {
                    "document_record": document_index.get(doc_id),
                    "chunks": [],
                    "summaries": [],
                    "claims": [],
                    "score": 0.0,
                },
            )
            score = self._claim_score(record=record, hints=hints, base=self._base_score(row))
            bucket["claims"].append((score, record))
            bucket["score"] = max(float(bucket["score"]), score)

        out: list[DocumentRetrievalHit] = []
        for doc_id, bucket in grouped.items():
            chunk_rows_sorted = sorted(list(bucket.get("chunks") or []), key=lambda item: item[0], reverse=True)
            summary_rows_sorted = sorted(list(bucket.get("summaries") or []), key=lambda item: item[0], reverse=True)
            claim_rows_sorted = sorted(list(bucket.get("claims") or []), key=lambda item: item[0], reverse=True)
            best_summary_record = summary_rows_sorted[0][1] if summary_rows_sorted else None
            relevant_chunks = [item[1] for item in chunk_rows_sorted[: max(1, int(self.chunk_limit))]]
            claim_records = [item[1] for item in claim_rows_sorted[: max(1, int(self.claim_limit))]]
            bonus = min(
                0.18,
                (0.03 * len(relevant_chunks))
                + (0.05 if best_summary_record is not None else 0.0)
                + (0.02 * len(claim_records)),
            )
            out.append(
                DocumentRetrievalHit(
                    document_id=doc_id,
                    score=float(bucket.get("score") or 0.0) + bonus,
                    document_record=bucket.get("document_record"),
                    relevant_chunks=relevant_chunks,
                    section_summary_record=best_summary_record,
                    section_summary=self._summary_from_record(best_summary_record) if best_summary_record is not None else None,
                    document_claim_records=claim_records,
                    document_claims=[row for row in (self._claim_from_record(item) for item in claim_records) if row is not None],
                )
            )

        out.sort(key=lambda row: float(row.score), reverse=True)
        return out[:top_k_value]

    def build_query_hints(self, query: RetrievalQuery) -> DocumentRetrievalHints:
        query_text = normalize_text(
            " ".join(
                part
                for part in (
                    str(query.query_text or "").strip(),
                    str(query.search_text or "").strip(),
                )
                if part
            )
        ).lower()
        query_tokens: set[str] = set()
        for raw in re.split(r"\s+", query_text):
            token = str(raw or "").strip().lower()
            if not token or token in _DOCUMENT_QUERY_STOPWORDS or len(token) < 2:
                continue
            query_tokens.add(token)
        for raw in list(query.entity_keys or []):
            token = str(raw or "").strip().lower()
            if token:
                query_tokens.add(token)

        chapter_match = _CHAPTER_QUERY_RE.search(query_text)
        chapter_number = ""
        if chapter_match:
            chapter_number = next((str(part).strip().lower() for part in chapter_match.groups() if str(part or "").strip()), "")

        return DocumentRetrievalHints(
            query_text=query_text,
            query_tokens=query_tokens,
            chapter_number=chapter_number,
            location_query=bool(_WHERE_QUERY_RE.search(query_text)),
            code_query=bool(_CODE_QUERY_RE.search(query_text)),
        )

    def _retrieve_channel(
        self,
        *,
        query_text: str,
        namespace: str,
        scopes: list[MemoryScope],
        top_k: int,
        filters: dict[str, object],
        source: str,
    ) -> list[dict[str, object]]:
        semantic_hits = self.store.semantic_search(
            query_text=query_text,
            top_k=max(1, int(top_k * 3)),
            namespace=namespace,
            scopes=scopes,
            include_stale=False,
            metadata_filters=filters,
        )
        lexical_hits = self.store.lexical_search(
            query_text=query_text,
            top_k=max(1, int(top_k * 3)),
            namespace=namespace,
            scopes=scopes,
            include_stale=False,
            metadata_filters=filters,
        )
        merged: dict[str, dict[str, object]] = {}
        for record, score in semantic_hits:
            merged[record.id] = {"record": record, "semantic_score": float(score), "lexical_score": 0.0, "source": source}
        for record, score in lexical_hits:
            row = merged.setdefault(record.id, {"record": record, "semantic_score": 0.0, "lexical_score": 0.0, "source": source})
            row["lexical_score"] = max(float(row.get("lexical_score") or 0.0), float(score))
        out = list(merged.values())
        out.sort(key=lambda row: self._base_score(row), reverse=True)
        return out

    def _document_records(self, *, namespace: str) -> dict[str, MemoryRecord]:
        if not hasattr(self.store, "iter_records"):
            return {}
        out: dict[str, MemoryRecord] = {}
        for record in list(self.store.iter_records(namespace=namespace) or []):
            if not isinstance(record, MemoryRecord) or record.memory_type != MemoryType.DOCUMENT:
                continue
            out[str(record.id or "")] = record
        return out

    @staticmethod
    def _base_score(row: dict[str, object]) -> float:
        return max(float(row.get("semantic_score") or 0.0), float(row.get("lexical_score") or 0.0))

    def _chunk_score(self, *, record: MemoryRecord, hints: DocumentRetrievalHints, base: float) -> float:
        meta = dict(record.metadata or {})
        text = normalize_text(str(record.text or "")).lower()
        tokens = self._record_tokens(record)
        score = base + 0.08
        if hints.code_query and str(meta.get("language") or "").strip():
            score += 0.10
        if hints.location_query and any(marker in text for marker in ("class ", "def ", "function ", "import ", "ollama", "class", "класс")):
            score += 0.10
        if hints.chapter_number and hints.chapter_number in normalize_text(str(meta.get("section_label") or text)).lower():
            score += 0.16
        score += min(0.18, 0.04 * self._token_overlap(tokens, hints.query_tokens))
        return score

    def _summary_score(self, *, record: MemoryRecord, hints: DocumentRetrievalHints, base: float) -> float:
        meta = dict(record.metadata or {})
        summary_kind = str(meta.get("summary_kind") or dict(meta.get("document_summary") or {}).get("summary_kind") or "").strip().lower()
        section_label = normalize_text(str(meta.get("section_label") or dict(meta.get("document_summary") or {}).get("metadata", {}).get("section_label") or "")).lower()
        text = normalize_text(str(record.text or "")).lower()
        tokens = self._record_tokens(record)
        score = base + 0.06
        if summary_kind == "section":
            score += 0.06
        elif summary_kind == "outline":
            score += 0.03
        if hints.chapter_number and hints.chapter_number and hints.chapter_number in section_label:
            score += 0.18
        if hints.code_query and any(marker in text for marker in ("class", "function", "method", "ollama")):
            score += 0.06
        score += min(0.14, 0.04 * self._token_overlap(tokens, hints.query_tokens))
        return score

    def _claim_score(self, *, record: MemoryRecord, hints: DocumentRetrievalHints, base: float) -> float:
        claim = dict(dict(record.metadata or {}).get("document_claim") or {})
        text = normalize_text(" ".join((str(claim.get("predicate") or ""), str(claim.get("obj") or ""), str(claim.get("evidence_text") or "")))).lower()
        tokens = {part for part in re.split(r"\s+", text) if part}
        score = base + 0.05
        score += min(0.16, 0.04 * self._token_overlap(tokens, hints.query_tokens))
        if hints.code_query and any(marker in text for marker in ("class", "function", "method", "ollama")):
            score += 0.08
        return score

    @staticmethod
    def _record_document_id(record: MemoryRecord) -> str:
        meta = dict(record.metadata or {})
        return str(meta.get("document_id") or meta.get("doc_id") or record.parent_id or "").strip()

    @staticmethod
    def _is_document_summary_record(record: MemoryRecord) -> bool:
        if record.memory_type != MemoryType.SUMMARY:
            return False
        meta = dict(record.metadata or {})
        return bool(meta.get("document_summary") or meta.get("document_id") or meta.get("doc_id"))

    @staticmethod
    def _is_document_claim_record(record: MemoryRecord) -> bool:
        if record.memory_type != MemoryType.CLAIM:
            return False
        meta = dict(record.metadata or {})
        return bool(meta.get("document_claim") or meta.get("document_id") or meta.get("doc_id"))

    @staticmethod
    def _summary_from_record(record: MemoryRecord | None) -> DocumentSummary | None:
        if record is None:
            return None
        meta = dict(record.metadata or {})
        payload = dict(meta.get("document_summary") or {})
        if payload:
            return DocumentSummary.from_dict(payload)
        if not (meta.get("document_id") or meta.get("doc_id")):
            return None
        return DocumentSummary(
            id=str(record.id or ""),
            document_id=str(meta.get("document_id") or meta.get("doc_id") or ""),
            text=str(record.text or ""),
            summary_kind=str(meta.get("summary_kind") or "section"),
            source_chunk_ids=[str(x).strip() for x in list(meta.get("source_chunk_ids") or []) if str(x).strip()],
            topic_keys=[str(x).strip().lower() for x in list(meta.get("topic_keys") or []) if str(x).strip()],
            entity_keys=[str(x).strip().lower() for x in list(meta.get("entity_keys") or []) if str(x).strip()],
            confidence=float(record.confidence or 0.0),
            salience=float(record.importance or 0.0),
            metadata={k: v for k, v in meta.items() if k not in {"document_id", "doc_id", "summary_kind", "source_chunk_ids", "topic_keys", "entity_keys"}},
            created_at=float(record.created_at or 0.0),
            updated_at=float(record.updated_at or record.created_at or 0.0),
            status=record.status,
        )

    @staticmethod
    def _claim_from_record(record: MemoryRecord | None) -> DocumentClaim | None:
        if record is None:
            return None
        meta = dict(record.metadata or {})
        payload = dict(meta.get("document_claim") or {})
        if payload:
            return DocumentClaim.from_dict(payload)
        if not (meta.get("document_id") or meta.get("doc_id")):
            return None
        claim = dict(meta.get("claim") or {})
        return DocumentClaim(
            id=str(record.id or ""),
            document_id=str(meta.get("document_id") or meta.get("doc_id") or ""),
            chunk_id=str(meta.get("chunk_id") or ""),
            subject=str(claim.get("subject") or ""),
            predicate=str(claim.get("predicate") or ""),
            obj=str(claim.get("obj") or ""),
            object_surface=str(claim.get("object_surface") or ""),
            qualifiers={},
            confidence=float(record.confidence or 0.0),
            salience=float(record.importance or 0.0),
            evidence_text=str(record.text or ""),
            topic_keys=[str(x).strip().lower() for x in list(claim.get("topic_keys") or []) if str(x).strip()],
            trigger_keys=[str(x).strip().lower() for x in list(claim.get("trigger_keys") or []) if str(x).strip()],
            metadata={k: v for k, v in meta.items() if k not in {"document_id", "doc_id", "chunk_id", "claim"}},
            created_at=float(record.created_at or 0.0),
            updated_at=float(record.updated_at or record.created_at or 0.0),
            status=record.status,
        )

    @staticmethod
    def _record_tokens(record: MemoryRecord) -> set[str]:
        meta = dict(record.metadata or {})
        values: list[str] = [str(record.text or "")]
        for key in ("section_label", "title", "summary_kind"):
            values.append(str(meta.get(key) or ""))
        values.extend([str(x) for x in list(meta.get("topic_keys") or [])])
        values.extend([str(x) for x in list(meta.get("entity_keys") or [])])
        document_summary = dict(meta.get("document_summary") or {})
        values.extend([str(x) for x in list(document_summary.get("topic_keys") or [])])
        values.extend([str(x) for x in list(document_summary.get("entity_keys") or [])])
        text = normalize_text(" ".join(values)).lower()
        return {token for token in re.split(r"\s+", text) if token and token not in _DOCUMENT_QUERY_STOPWORDS}

    @staticmethod
    def _token_overlap(left: set[str], right: set[str]) -> int:
        if not left or not right:
            return 0
        count = 0
        for a in set(left or set()):
            for b in set(right or set()):
                if DocumentRetriever._token_match(a, b):
                    count += 1
                    break
        return count

    @staticmethod
    def _token_match(a: str, b: str) -> bool:
        left = str(a or "").strip().lower()
        right = str(b or "").strip().lower()
        if not left or not right:
            return False
        if left == right:
            return True
        if len(left) >= 4 and len(right) >= 4:
            if left.startswith(right[:4]) or right.startswith(left[:4]):
                return True
        if len(left) >= 3 and len(right) >= 3:
            if left.startswith(right[:3]) or right.startswith(left[:3]):
                return True
        return False
