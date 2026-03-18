from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from memory.document_memory import ChunkingConfig, DocumentMemory
from memory.document_models import DocumentClaim, DocumentChunk, DocumentRecord, DocumentSummary
from memory.ingest_analyzer import IngestAnalysis, analyze_message_for_memory
from memory.memory_models import DocumentIngestRequest, MemoryLevel, MemoryRecord, MemoryScope, MemoryStatus, MemoryType
from memory.storage_profile import DEFAULT_STORAGE_PROFILE, normalize_storage_profile, sanitize_storage_metadata
from memory.vector_store import VectorStore


_FILE_ENCODINGS = ("utf-8", "utf-8-sig", "cp1251", "latin-1")


@dataclass(frozen=True)
class DocumentChunkAnalysis:
    chunk: DocumentChunk
    analysis: IngestAnalysis
    section_label: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk": self.chunk.to_dict(),
            "analysis": self.analysis.to_dict(),
            "section_label": str(self.section_label or ""),
        }


@dataclass(frozen=True)
class DocumentOutline:
    document_id: str
    title: str
    section_labels: list[str] = field(default_factory=list)
    text: str = ""
    topic_keys: list[str] = field(default_factory=list)
    entity_keys: list[str] = field(default_factory=list)

    def to_summary(self, *, summary_id: str, namespace: str, now_ts: float) -> DocumentSummary:
        return DocumentSummary(
            id=summary_id,
            document_id=str(self.document_id or ""),
            text=str(self.text or ""),
            summary_kind="outline",
            source_chunk_ids=[],
            topic_keys=list(self.topic_keys or []),
            entity_keys=list(self.entity_keys or []),
            confidence=0.84,
            salience=0.82,
            metadata={
                "namespace": str(namespace or "default"),
                "title": str(self.title or ""),
                "section_labels": [str(x).strip() for x in list(self.section_labels or []) if str(x).strip()],
            },
            created_at=now_ts,
            updated_at=now_ts,
            status=MemoryStatus.ACTIVE,
        )


@dataclass(frozen=True)
class DocumentIngestArtifacts:
    document: DocumentRecord
    chunks: list[DocumentChunk] = field(default_factory=list)
    chunk_analyses: list[DocumentChunkAnalysis] = field(default_factory=list)
    section_summaries: list[DocumentSummary] = field(default_factory=list)
    document_outline: DocumentSummary | None = None
    document_claims: list[DocumentClaim] = field(default_factory=list)
    stored_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "document": self.document.to_dict(),
            "chunks": [row.to_dict() for row in list(self.chunks or [])],
            "chunk_analyses": [row.to_dict() for row in list(self.chunk_analyses or [])],
            "section_summaries": [row.to_dict() for row in list(self.section_summaries or [])],
            "document_outline": (self.document_outline.to_dict() if self.document_outline is not None else None),
            "document_claims": [row.to_dict() for row in list(self.document_claims or [])],
            "stored_ids": [str(x).strip() for x in list(self.stored_ids or []) if str(x).strip()],
        }


@dataclass
class DocumentIngestPipeline:
    store: VectorStore
    chunking: ChunkingConfig | None = None
    storage_profile: str = DEFAULT_STORAGE_PROFILE

    def __post_init__(self) -> None:
        self.document_memory = DocumentMemory(store=self.store, chunking=self.chunking)
        self.storage_profile = normalize_storage_profile(self.storage_profile)

    def ingest_document(self, request: DocumentIngestRequest) -> DocumentIngestArtifacts:
        source = str(request.source or "document").strip() or "document"
        metadata = dict(request.metadata or {})
        parsed_text = self.parse_text(str(request.text or ""))
        document_request = DocumentIngestRequest(
            text=parsed_text,
            source=source,
            namespace=str(request.namespace or "default"),
            scope=request.scope,
            title=str(request.title or ""),
            metadata=metadata,
        )
        low_level = self.document_memory.ingest_document(document_request)
        now_ts = float(time.time())

        chunk_analyses = self._analyze_chunks(document=low_level.document, chunks=low_level.chunks)
        section_summaries = self.build_section_summaries(
            document=low_level.document,
            chunk_analyses=chunk_analyses,
            namespace=str(request.namespace or "default"),
            now_ts=now_ts,
        )
        outline = self.build_document_outline(
            document=low_level.document,
            chunk_analyses=chunk_analyses,
            section_summaries=section_summaries,
            namespace=str(request.namespace or "default"),
            now_ts=now_ts,
        )
        document_claims = self.build_document_claims(
            document=low_level.document,
            chunk_analyses=chunk_analyses,
            now_ts=now_ts,
        )
        stored_ids = self._store_artifacts(
            document=low_level.document,
            chunk_analyses=chunk_analyses,
            section_summaries=section_summaries,
            outline=outline,
            document_claims=document_claims,
            now_ts=now_ts,
        )
        return DocumentIngestArtifacts(
            document=low_level.document,
            chunks=list(low_level.chunks or []),
            chunk_analyses=chunk_analyses,
            section_summaries=section_summaries,
            document_outline=outline,
            document_claims=document_claims,
            stored_ids=stored_ids,
        )

    def ingest_file(
        self,
        path: str | Path,
        *,
        namespace: str = "default",
        scope: MemoryScope = MemoryScope.PROJECT,
        title: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> DocumentIngestArtifacts:
        file_path = Path(path)
        payload = self.load_file(file_path)
        meta = {
            "path": str(file_path),
            "source_path": str(file_path),
            **dict(metadata or {}),
            **dict(payload.get("metadata") or {}),
        }
        return self.ingest_document(
            DocumentIngestRequest(
                text=str(payload.get("text") or ""),
                source=str(file_path),
                namespace=str(namespace or "default"),
                scope=scope,
                title=str(title or payload.get("title") or ""),
                metadata=meta,
            )
        )

    @staticmethod
    def load_file(path: str | Path) -> dict[str, Any]:
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(str(file_path))

        raw_text = ""
        for encoding in _FILE_ENCODINGS:
            try:
                raw_text = file_path.read_text(encoding=encoding)
                break
            except Exception:
                continue
        else:
            raw_text = file_path.read_text(encoding="utf-8", errors="replace")

        parsed = DocumentIngestPipeline.parse_text(raw_text)
        title = file_path.stem.strip() or file_path.name.strip()
        return {
            "text": parsed,
            "title": title,
            "metadata": {
                "path": str(file_path),
                "source_path": str(file_path),
                "filename": file_path.name,
                "extension": file_path.suffix.lower(),
            },
        }

    @staticmethod
    def parse_text(text: str) -> str:
        src = str(text or "").replace("\ufeff", "")
        src = src.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
        raw_lines = [line.rstrip() for line in src.split("\n")]
        lines: list[str] = []
        blank_pending = False
        for line in raw_lines:
            if not line.strip():
                if blank_pending:
                    continue
                blank_pending = True
                lines.append("")
                continue
            blank_pending = False
            lines.append(line)
        while lines and not lines[0].strip():
            lines.pop(0)
        while lines and not lines[-1].strip():
            lines.pop()
        return "\n".join(lines).strip()

    def _analyze_chunks(self, *, document: DocumentRecord, chunks: list[DocumentChunk]) -> list[DocumentChunkAnalysis]:
        out: list[DocumentChunkAnalysis] = []
        doc_meta = dict(document.metadata or {})
        for chunk in list(chunks or []):
            chunk_meta = dict(chunk.metadata or {})
            section_label = self._section_label(chunk.text, fallback_title=str(doc_meta.get("title") or ""))
            analysis = analyze_message_for_memory(
                chunk.text,
                metadata={
                    **doc_meta,
                    **chunk_meta,
                    "document_id": document.id,
                    "chunk_id": chunk.id,
                    "section_label": section_label,
                    "document_source": document.source,
                },
            )
            out.append(DocumentChunkAnalysis(chunk=chunk, analysis=analysis, section_label=section_label))
        return out

    def build_section_summaries(
        self,
        *,
        document: DocumentRecord,
        chunk_analyses: list[DocumentChunkAnalysis],
        namespace: str,
        now_ts: float,
    ) -> list[DocumentSummary]:
        out: list[DocumentSummary] = []
        for item in list(chunk_analyses or []):
            analysis = item.analysis
            chunk = item.chunk
            label = str(item.section_label or f"section_{int(chunk.chunk_index)}").strip()
            summary_text = self._section_summary_text(label=label, analysis=analysis, chunk_text=chunk.text)
            out.append(
                DocumentSummary(
                    id=f"docsum:{document.id}:{int(chunk.chunk_index)}",
                    document_id=document.id,
                    text=summary_text,
                    summary_kind="section",
                    source_chunk_ids=[chunk.id],
                    topic_keys=self._clean_keys(list(analysis.memory_views.get("entity_keys") or []) + self._topic_tags_to_keys(list(analysis.tags or []))),
                    entity_keys=self._clean_keys(list(analysis.memory_views.get("entity_keys") or [])),
                    confidence=0.78,
                    salience=min(0.88, 0.52 + (0.04 * len(list(analysis.entities or []))) + (0.03 * len(list(analysis.claim_candidates or [])))),
                    metadata={
                        "section_label": label,
                        "namespace": str(namespace or "default"),
                        "chunk_index": int(chunk.chunk_index),
                        "chunk_id": chunk.id,
                    },
                    created_at=now_ts,
                    updated_at=now_ts,
                    status=MemoryStatus.ACTIVE,
                )
            )
        return out

    def build_document_outline(
        self,
        *,
        document: DocumentRecord,
        chunk_analyses: list[DocumentChunkAnalysis],
        section_summaries: list[DocumentSummary],
        namespace: str,
        now_ts: float,
    ) -> DocumentSummary | None:
        if not list(chunk_analyses or []):
            return None

        title = str(dict(document.metadata or {}).get("title") or document.source or document.id).strip()
        sections = [str(item.metadata.get("section_label") or "").strip() for item in list(section_summaries or []) if str(item.metadata.get("section_label") or "").strip()]
        topics: list[str] = []
        entities: list[str] = []
        for item in list(chunk_analyses or []):
            topics.extend(self._topic_tags_to_keys(list(item.analysis.tags or [])))
            entities.extend([str(x) for x in list(item.analysis.memory_views.get("entity_keys") or [])])

        outline = DocumentOutline(
            document_id=document.id,
            title=title,
            section_labels=self._dedupe_preserve(sections),
            text=self._outline_text(title=title, sections=self._dedupe_preserve(sections), topics=self._clean_keys(topics)),
            topic_keys=self._clean_keys(topics),
            entity_keys=self._clean_keys(entities),
        )
        return outline.to_summary(
            summary_id=f"docsum:{document.id}:outline",
            namespace=str(namespace or "default"),
            now_ts=now_ts,
        )

    def build_document_claims(
        self,
        *,
        document: DocumentRecord,
        chunk_analyses: list[DocumentChunkAnalysis],
        now_ts: float,
    ) -> list[DocumentClaim]:
        out: list[DocumentClaim] = []
        seen: set[tuple[str, str, str]] = set()
        for item in list(chunk_analyses or []):
            chunk = item.chunk
            for idx, candidate in enumerate(list(item.analysis.claim_candidates or [])):
                predicate = str(candidate.predicate or "").strip().lower()
                obj = str(candidate.normalized_object or candidate.object_surface or "").strip().lower()
                subject = str(candidate.subject or "document").strip().lower() or "document"
                if not predicate or not obj:
                    continue
                key = (subject, predicate, obj)
                if key in seen:
                    continue
                seen.add(key)
                out.append(
                    DocumentClaim(
                        id=f"docclaim:{document.id}:{int(chunk.chunk_index)}:{idx}",
                        document_id=document.id,
                        chunk_id=chunk.id,
                        subject=subject,
                        predicate=predicate,
                        obj=obj,
                        object_type=str(candidate.object_type or "").strip().lower(),
                        object_surface=str(candidate.object_surface or "").strip(),
                        qualifiers=dict(candidate.qualifiers or {}),
                        confidence=float(candidate.confidence or 0.0),
                        salience=min(
                            1.0,
                            (0.55 * float(candidate.confidence or 0.0))
                            + (0.35 * float(candidate.specificity or 0.0))
                            + (0.10 * (1.0 if str(candidate.object_type or "").strip() else 0.65)),
                        ),
                        evidence_text=str(candidate.evidence_text or chunk.text[:220]).strip(),
                        topic_keys=[str(x) for x in list(candidate.topic_keys or [])],
                        trigger_keys=[str(x) for x in list(candidate.trigger_keys or [])],
                        metadata={
                            "document_source": document.source,
                            "chunk_index": int(chunk.chunk_index),
                        },
                        created_at=now_ts,
                        updated_at=now_ts,
                        status=MemoryStatus.ACTIVE,
                    )
                )
        return out

    def _store_artifacts(
        self,
        *,
        document: DocumentRecord,
        chunk_analyses: list[DocumentChunkAnalysis],
        section_summaries: list[DocumentSummary],
        outline: DocumentSummary | None,
        document_claims: list[DocumentClaim],
        now_ts: float,
    ) -> list[str]:
        records: list[MemoryRecord] = []
        stored_ids: list[str] = []

        document_meta = {
            **dict(document.metadata or {}),
            "document_outline": (outline.to_dict() if outline is not None else None),
            "section_summary_ids": [row.id for row in list(section_summaries or [])],
            "document_claim_ids": [row.id for row in list(document_claims or [])],
        }
        records.append(
            MemoryRecord(
                id=document.id,
                text=f"{dict(document.metadata or {}).get('title') or document.source}\n{document.summary}".strip(),
                memory_type=MemoryType.DOCUMENT,
                level=MemoryLevel.L4_DOCUMENT,
                scope=document.scope,
                namespace=document.namespace,
                metadata=document_meta,
                importance=float(document.importance),
                confidence=float(document.confidence),
                created_at=float(document.created_at),
                updated_at=float(now_ts),
                status=document.status,
                version=int(document.version),
                parent_id=document.parent_id,
                chunk_index=document.chunk_index,
            )
        )
        stored_ids.append(document.id)

        for item in list(chunk_analyses or []):
            chunk = item.chunk
            analysis = item.analysis
            memory_entities = [row.to_dict() for row in list(analysis.entities or [])]
            numeric_facts = [row.to_dict() for row in list(analysis.numeric_facts or [])]
            claims = self._compact_claims(analysis)
            stable_facts = [row.to_dict() for row in list(analysis.stable_facts or [])]
            memory_tags = [str(x) for x in list(analysis.tags or []) if str(x).strip()]
            metadata = {
                **dict(chunk.metadata or {}),
                "document_id": document.id,
                "doc_id": document.id,
                "chunk_id": chunk.id,
                "chunk_index": int(chunk.chunk_index),
                "section_label": str(item.section_label or ""),
                "memory_views": dict(analysis.memory_views or {}),
            }
            if memory_entities:
                metadata["memory_entities"] = memory_entities
            if numeric_facts:
                metadata["numeric_facts"] = numeric_facts
            if claims:
                metadata["claims"] = claims
            if stable_facts:
                metadata["stable_facts"] = stable_facts
            if memory_tags:
                metadata["memory_tags"] = memory_tags
            records.append(
                MemoryRecord(
                    id=chunk.id,
                    text=chunk.text,
                    memory_type=MemoryType.DOCUMENT_CHUNK,
                    level=MemoryLevel.L4_DOCUMENT,
                    scope=chunk.scope,
                    namespace=chunk.namespace,
                    metadata=metadata,
                    importance=float(chunk.importance),
                    confidence=float(chunk.confidence),
                    created_at=float(chunk.created_at),
                    updated_at=float(now_ts),
                    status=chunk.status,
                    version=int(chunk.version),
                    parent_id=document.id,
                    chunk_index=int(chunk.chunk_index),
                )
            )
            stored_ids.append(chunk.id)

        for row in list(section_summaries or []):
            records.append(self._summary_record(document=document, summary=row, now_ts=now_ts))
            stored_ids.append(row.id)

        if outline is not None:
            records.append(self._summary_record(document=document, summary=outline, now_ts=now_ts))
            stored_ids.append(outline.id)

        for row in list(document_claims or []):
            records.append(self._claim_record(document=document, claim=row, now_ts=now_ts))
            stored_ids.append(row.id)

        compact_records: list[MemoryRecord] = []
        for record in list(records or []):
            compact_records.append(
                MemoryRecord(
                    id=record.id,
                    text=record.text,
                    memory_type=record.memory_type,
                    level=record.level,
                    scope=record.scope,
                    namespace=record.namespace,
                    metadata=sanitize_storage_metadata(
                        metadata=dict(record.metadata or {}),
                        storage_profile=self.storage_profile,
                    ),
                    embedding=list(record.embedding or []) if isinstance(record.embedding, list) else None,
                    importance=record.importance,
                    confidence=record.confidence,
                    created_at=record.created_at,
                    updated_at=record.updated_at,
                    expires_at=record.expires_at,
                    status=record.status,
                    version=record.version,
                    parent_id=record.parent_id,
                    chunk_index=record.chunk_index,
                    source_event_id=record.source_event_id,
                    embedding_model=record.embedding_model,
                    embedding_fingerprint=record.embedding_fingerprint,
                    embedding_version=record.embedding_version,
                )
            )

        self.store.batch_upsert(compact_records)
        return self._dedupe_preserve(stored_ids)

    def _summary_record(self, *, document: DocumentRecord, summary: DocumentSummary, now_ts: float) -> MemoryRecord:
        return MemoryRecord(
            id=summary.id,
            text=str(summary.text or "").strip(),
            memory_type=MemoryType.SUMMARY,
            level=MemoryLevel.L4_DOCUMENT,
            scope=document.scope,
            namespace=document.namespace,
            metadata={
                "document_id": document.id,
                "doc_id": document.id,
                "summary_kind": str(summary.summary_kind or "overview"),
                "source_chunk_ids": list(summary.source_chunk_ids or []),
                "topic_keys": list(summary.topic_keys or []),
                "entity_keys": list(summary.entity_keys or []),
                "document_summary": summary.to_dict(),
                **dict(summary.metadata or {}),
            },
            importance=float(summary.salience),
            confidence=float(summary.confidence),
            created_at=float(summary.created_at),
            updated_at=float(now_ts),
            status=summary.status,
            parent_id=document.id,
        )

    def _claim_record(self, *, document: DocumentRecord, claim: DocumentClaim, now_ts: float) -> MemoryRecord:
        canonical = ".".join(
            part
            for part in (
                str(document.id or "").strip().lower(),
                str(claim.subject or "").strip().lower(),
                str(claim.predicate or "").strip().lower(),
                self._slug(str(claim.obj or "").strip().lower()),
            )
            if part
        )
        return MemoryRecord(
            id=claim.id,
            text=f"{claim.subject}.{claim.predicate}={claim.obj}",
            memory_type=MemoryType.CLAIM,
            level=MemoryLevel.L4_DOCUMENT,
            scope=document.scope,
            namespace=document.namespace,
            metadata={
                "document_id": document.id,
                "doc_id": document.id,
                "canonical_key": canonical,
                "claim": {
                    "subject": str(claim.subject or "").strip().lower(),
                    "predicate": str(claim.predicate or "").strip().lower(),
                    "obj": str(claim.obj or "").strip().lower(),
                    "object_surface": str(claim.object_surface or "").strip(),
                    "topic_keys": list(claim.topic_keys or []),
                    "trigger_keys": list(claim.trigger_keys or []),
                    "recall_mode": "contextual",
                    "spontaneous_recall": False,
                },
                "document_claim": claim.to_dict(),
                **dict(claim.metadata or {}),
            },
            importance=float(claim.salience),
            confidence=float(claim.confidence),
            created_at=float(claim.created_at),
            updated_at=float(now_ts),
            status=claim.status,
            parent_id=document.id,
        )

    @staticmethod
    def _section_label(text: str, *, fallback_title: str = "") -> str:
        lines = [str(line or "").strip() for line in str(text or "").splitlines() if str(line or "").strip()]
        for line in lines[:8]:
            if line.startswith("#"):
                return line.lstrip("#").strip()[:96]
            if line.endswith(":") and len(line) <= 96:
                return line.rstrip(":").strip()
        if lines:
            return lines[0][:96].strip()
        return str(fallback_title or "section").strip() or "section"

    @staticmethod
    def _section_summary_text(*, label: str, analysis: IngestAnalysis, chunk_text: str) -> str:
        if list(analysis.claim_candidates or []):
            claim_bits = [str(row.object_surface or row.normalized_object or "").strip() for row in list(analysis.claim_candidates or [])[:2]]
            claim_bits = [row for row in claim_bits if row]
            if claim_bits:
                return f"{label}: discusses {', '.join(claim_bits)}."
        if list(analysis.entities or []):
            entity_bits = [str(row.canonical or row.surface or "").strip() for row in list(analysis.entities or [])[:3]]
            entity_bits = [row for row in entity_bits if row]
            if entity_bits:
                return f"{label}: mentions {', '.join(entity_bits)}."
        lines = [str(line or "").strip() for line in str(chunk_text or "").splitlines() if str(line or "").strip()]
        preview = " ".join(lines[:2]).strip()
        preview = preview[:220].strip()
        return f"{label}: {preview}" if preview else f"{label}: section content."

    @staticmethod
    def _outline_text(*, title: str, sections: list[str], topics: list[str]) -> str:
        parts: list[str] = []
        if title:
            parts.append(f"Document outline for {title}.")
        else:
            parts.append("Document outline.")
        if sections:
            parts.append("Sections: " + "; ".join(sections[:6]) + ".")
        if topics:
            parts.append("Key topics: " + ", ".join(topics[:8]) + ".")
        return " ".join(part.strip() for part in parts if part.strip())

    @staticmethod
    def _topic_tags_to_keys(tags: list[str]) -> list[str]:
        out: list[str] = []
        for raw in list(tags or []):
            token = str(raw or "").strip().lower()
            if token.startswith("topic_"):
                out.append(token.replace("topic_", "", 1))
        return out

    @staticmethod
    def _compact_claims(analysis: IngestAnalysis) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        seen: set[tuple[str, str, str]] = set()
        for item in list(analysis.claim_candidates or []):
            subject = str(item.subject or "").strip().lower()
            predicate = str(item.predicate or "").strip().lower()
            obj = str(item.normalized_object or item.object_surface or "").strip()
            if not subject or not predicate or not obj:
                continue
            key = (subject, predicate, obj.lower())
            if key in seen:
                continue
            seen.add(key)
            out.append(
                {
                    "subject": subject,
                    "predicate": predicate,
                    "obj": obj,
                    "object_surface": str(item.object_surface or "").strip(),
                    "object_type": str(item.object_type or "").strip().lower(),
                    "topic_keys": [str(x).strip().lower() for x in list(item.topic_keys or []) if str(x).strip()][:6],
                    "trigger_keys": [str(x).strip().lower() for x in list(item.trigger_keys or []) if str(x).strip()][:6],
                    "confidence": round(float(item.confidence or 0.0), 4),
                }
            )
        return out

    @staticmethod
    def _clean_keys(values: list[str]) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()
        for raw in list(values or []):
            token = str(raw or "").strip().lower()
            if not token or token in seen:
                continue
            seen.add(token)
            out.append(token)
        return out

    @staticmethod
    def _dedupe_preserve(values: list[str]) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()
        for raw in list(values or []):
            token = str(raw or "").strip()
            if not token or token in seen:
                continue
            seen.add(token)
            out.append(token)
        return out

    @staticmethod
    def _slug(value: str) -> str:
        out: list[str] = []
        for ch in str(value or "").strip().lower():
            if ch.isalnum():
                out.append(ch)
            else:
                out.append("_")
        return "".join(out).strip("_") or uuid.uuid4().hex[:10]
