from __future__ import annotations

import hashlib
import re
import time
from typing import Any

from memory.ingest_analyzer import EntityItem, IngestAnalysis, NumericFact, StableFact
from memory.memory_models import FactRecordV2, MemoryScope, MemoryStatus


MODE_FAST = "FAST"
MODE_BALANCED = "BALANCED"
MODE_QUALITY = "QUALITY"

_SPACE_RE = re.compile(r"\s+")
_SENTENCE_SPLIT_RE = re.compile(r"(?:[!?]+|\.(?=\s|$)|\n)+")
_CLAUSE_SPLIT_RE = re.compile(
    r",\s+(?=(?:а|но|и|and|but|в проекте|у меня|я |мы |мне |надо|нужно|проблема|problem|issue|решили|оставляем|"
    r"i use|we use|i want|need to)\b)",
    flags=re.I,
)
_TRIM_LEADING_RE = re.compile(
    r"^(?:что|that|to|про|about|это|is|are|будто|как будто)\s+",
    flags=re.I,
)
_TRIM_TRAILING_RE = re.compile(r"\s+(?:пожалуйста|please|pls)$", flags=re.I)
_GENERIC_VALUES = {
    "это",
    "так",
    "там",
    "тут",
    "ничего",
    "something",
    "anything",
    "everything",
    "this",
    "that",
}
_NAME_BLOCKLIST = {
    "люблю",
    "хочу",
    "использую",
    "работаю",
    "думаю",
    "считаю",
    "знаю",
    "нужно",
    "надо",
    "i",
    "prefer",
    "want",
    "use",
    "need",
}
_PROJECT_VALUE_BLOCKLIST = {
    "это",
    "так",
    "все",
    "ничего",
    "something",
    "anything",
}
_ACTION_KEYWORDS = (
    "fix",
    "add",
    "update",
    "remove",
    "support",
    "implement",
    "rewrite",
    "refactor",
    "check",
    "verify",
    "build",
    "ship",
    "почин",
    "исправ",
    "добав",
    "обнов",
    "убра",
    "поддерж",
    "провер",
    "сдела",
    "доработ",
    "перепис",
    "рефактор",
    "вынес",
    "настро",
    "почист",
)
_ENVIRONMENT_CUES = (
    "i use",
    "we use",
    "i am on",
    "i'm on",
    "running on",
    "run on",
    "using",
    "у меня",
    "я использую",
    "мы используем",
    "работаю на",
    "сижу на",
    "запускаю в",
    "поднимаю в",
)
_ENVIRONMENT_STATIC_RULES: tuple[tuple[re.Pattern[str], str, str], ...] = (
    (re.compile(r"\bwindows(?:\s+11|\s+10)?\b", re.I), "environment_os", "windows"),
    (re.compile(r"\blinux\b", re.I), "environment_os", "linux"),
    (re.compile(r"\bubuntu\b", re.I), "environment_os", "ubuntu"),
    (re.compile(r"\bdebian\b", re.I), "environment_os", "debian"),
    (re.compile(r"\bmac(?:os)?\b", re.I), "environment_os", "macos"),
    (re.compile(r"\bwsl\b", re.I), "environment_tool", "wsl"),
    (re.compile(r"\bdocker\b", re.I), "environment_tool", "docker"),
    (re.compile(r"\bpower\s?shell\b", re.I), "environment_shell", "powershell"),
    (re.compile(r"\bbash\b", re.I), "environment_shell", "bash"),
    (re.compile(r"\bzsh\b", re.I), "environment_shell", "zsh"),
    (re.compile(r"\bcmd\b", re.I), "environment_shell", "cmd"),
    (re.compile(r"\bpoetry\b", re.I), "environment_tool", "poetry"),
    (re.compile(r"\buv\b", re.I), "environment_tool", "uv"),
    (re.compile(r"\bvenv\b", re.I), "environment_tool", "venv"),
)
_ENVIRONMENT_VERSION_RULES: tuple[tuple[re.Pattern[str], str, str], ...] = (
    (re.compile(r"\bpython\s*([0-9]+(?:\.[0-9]+){0,2})\b", re.I), "environment_runtime_python", "python"),
    (re.compile(r"\bnode(?:\.js)?\s*v?([0-9]+(?:\.[0-9]+){0,2})\b", re.I), "environment_runtime_node", "node"),
)
_PROBLEM_MARKERS = (
    "error",
    "failed",
    "exception",
    "traceback",
    "bug",
    "problem",
    "issue",
    "ошибка",
    "не работает",
    "проблема",
    "сломал",
    "ломает",
    "падает",
    "не ловит",
)
_PLAN_SEQUENCE_RE = re.compile(
    r"\b(?:first|сначала)\s+([^.,;!?]{2,120})\s*(?:,|\s+)\s*(?:then|потом)\s+([^.,;!?]{2,120})",
    re.I,
)
_CANONICAL_TERM_MAP = {
    "search_text": ("search text", "search_text"),
    "canonical_text": ("canonical text", "canonical_text"),
    "assistant_thoughts": ("assistant thoughts", "assistant thought", "мысли ассистента", "thoughts of assistant"),
    "write_policy": ("write policy", "write-policy", "policy write", "политика записи", "write policy"),
    "memory": ("memory", "память"),
    "web": ("web", "веб"),
}


class FactExtractor:
    def extract_v2(
        self,
        *,
        text: str,
        metadata: dict | None,
        speaker: str,
        scope: MemoryScope = MemoryScope.CONVERSATION,
        mode: str = MODE_BALANCED,
        analysis: IngestAnalysis | None = None,
    ) -> list[FactRecordV2]:
        src = self._normalize_text(text)
        if not src:
            return []
        meta = dict(metadata or {})
        event_id = str(meta.get("event_id") or "")
        namespace = str(meta.get("namespace") or "default")
        subject = self._subject_for_speaker(speaker)
        profile = str(mode or MODE_BALANCED).strip().upper()
        if profile not in {MODE_FAST, MODE_BALANCED, MODE_QUALITY}:
            profile = MODE_BALANCED

        segments = self._segments(src)
        rows: list[FactRecordV2] = []
        rows.extend(
            self._analysis_facts(
                analysis=analysis,
                subject=subject,
                scope=scope,
                event_id=event_id,
                namespace=namespace,
            )
        )
        rows.extend(
            self._facts_from_entities(
                analysis=analysis,
                subject=subject,
                scope=scope,
                event_id=event_id,
                namespace=namespace,
            )
        )
        rows.extend(
            self._facts_from_numeric(
                analysis=analysis,
                subject=subject,
                scope=scope,
                event_id=event_id,
                namespace=namespace,
            )
        )
        rows.extend(self._identity_facts(src, segments=segments, subject=subject, scope=scope, event_id=event_id, namespace=namespace))
        rows.extend(self._project_facts(src, segments=segments, subject=subject, scope=scope, event_id=event_id, namespace=namespace))
        rows.extend(
            self._environment_facts(
                src,
                segments=segments,
                subject=subject,
                scope=scope,
                event_id=event_id,
                namespace=namespace,
                analysis=analysis,
            )
        )
        rows.extend(self._preference_facts(src, segments=segments, subject=subject, scope=scope, event_id=event_id, namespace=namespace))
        rows.extend(self._task_facts(src, segments=segments, subject=subject, scope=scope, event_id=event_id, namespace=namespace))
        rows.extend(self._decision_facts(src, segments=segments, subject=subject, scope=scope, event_id=event_id, namespace=namespace))
        rows.extend(self._issue_facts(src, segments=segments, subject=subject, scope=scope, event_id=event_id, namespace=namespace))
        rows.extend(self._relationship_facts(src, segments=segments, subject=subject, scope=scope, event_id=event_id, namespace=namespace))
        rows.extend(self._temporary_facts(src, segments=segments, subject=subject, scope=scope, event_id=event_id, namespace=namespace))
        rows.extend(self._resolution_facts(src, segments=segments, subject=subject, scope=scope, event_id=event_id, namespace=namespace))

        if profile == MODE_FAST:
            allowed = {"identity", "preference", "task", "issue", "decision"}
            rows = [x for x in rows if str(x.relation or "") in allowed]
        elif profile == MODE_QUALITY:
            rows = [
                FactRecordV2(
                    subject=row.subject,
                    predicate=row.predicate,
                    value=row.value,
                    scope=row.scope,
                    confidence=min(1.0, float(row.confidence) + 0.05),
                    importance=min(1.0, float(row.importance) + 0.03),
                    evidence=row.evidence,
                    source_event_id=row.source_event_id,
                    valid_from=row.valid_from,
                    valid_to=row.valid_to,
                    status=row.status,
                    canonical_key=row.canonical_key,
                    relation=row.relation,
                    id=row.id,
                    text=row.text,
                    memory_type=row.memory_type,
                    level=row.level,
                    namespace=row.namespace,
                    metadata=dict(row.metadata or {}),
                    created_at=row.created_at,
                    updated_at=row.updated_at,
                    parent_id=row.parent_id,
                    chunk_index=row.chunk_index,
                    version=row.version,
                )
                for row in rows
            ]

        rows = self._dedupe_v2(rows)
        source_role = str(speaker or "").strip().lower()
        default_source_kind = "structured_fact" if analysis is not None else "extracted_fact"
        out: list[FactRecordV2] = []
        for row in list(rows or []):
            out.append(
                FactRecordV2(
                    subject=row.subject,
                    predicate=row.predicate,
                    value=row.value,
                    scope=row.scope,
                    confidence=row.confidence,
                    importance=row.importance,
                    evidence=row.evidence,
                    source_event_id=row.source_event_id,
                    valid_from=row.valid_from,
                    valid_to=row.valid_to,
                    status=row.status,
                    canonical_key=row.canonical_key,
                    relation=row.relation,
                    id=row.id,
                    text=row.text,
                    memory_type=row.memory_type,
                    level=row.level,
                    namespace=row.namespace,
                    metadata={
                        **dict(row.metadata or {}),
                        "source_role": source_role,
                        "source_kind": str(dict(row.metadata or {}).get("source_kind") or default_source_kind),
                    },
                    created_at=row.created_at,
                    updated_at=row.updated_at,
                    parent_id=row.parent_id,
                    chunk_index=row.chunk_index,
                    version=row.version,
                )
            )
        return out

    @staticmethod
    def _subject_for_speaker(speaker: str) -> str:
        role = str(speaker or "").strip().lower()
        if role in {"assistant", "ai", "bot"}:
            return "assistant"
        if role in {"user", "human"}:
            return "user"
        return "other"

    def _analysis_facts(
        self,
        *,
        analysis: IngestAnalysis | None,
        subject: str,
        scope: MemoryScope,
        event_id: str,
        namespace: str,
    ) -> list[FactRecordV2]:
        if analysis is None:
            return []
        out: list[FactRecordV2] = []
        evidence = str(analysis.raw_text or analysis.search_text or analysis.canonical_text or "").strip()
        for item in list(analysis.stable_facts or []):
            if not isinstance(item, StableFact):
                continue
            predicate = self._analysis_predicate(item)
            if not predicate:
                continue
            value = self._analysis_value(predicate=predicate, value=item.value)
            if value in {"", None}:
                continue
            relation = str(item.relation or item.subject or "structured").strip().lower()
            importance = 0.82 if relation in {"identity", "project"} else 0.74
            metadata_extra = {
                "analysis_source": "ingest_analyzer",
                "stable_subject": str(item.subject or "").strip().lower(),
                "stable_predicate": str(item.predicate or "").strip().lower(),
                "structured": True,
            }
            self._append_fact(
                out,
                subject=subject,
                predicate=predicate,
                value=value,
                scope=scope,
                confidence=max(0.72, min(1.0, float(item.confidence))),
                importance=importance,
                evidence=evidence,
                event_id=event_id,
                relation=relation,
                namespace=namespace,
                allow_numeric=True,
                metadata_extra=metadata_extra,
            )
        return out

    def _facts_from_entities(
        self,
        *,
        analysis: IngestAnalysis | None,
        subject: str,
        scope: MemoryScope,
        event_id: str,
        namespace: str,
    ) -> list[FactRecordV2]:
        if analysis is None:
            return []
        out: list[FactRecordV2] = []
        evidence = str(analysis.raw_text or analysis.search_text or analysis.canonical_text or "").strip()
        mapping: dict[str, tuple[str, float, float, str]] = {
            "gpu_model": ("environment_gpu_model", 0.86, 0.78, "environment"),
            "cpu_model": ("environment_cpu_model", 0.84, 0.76, "environment"),
            "python_version": ("environment_runtime_python", 0.90, 0.80, "environment"),
            "os_name": ("environment_os", 0.82, 0.74, "environment"),
            "project_name": ("project_name", 0.84, 0.84, "project"),
            "person_name": ("identity_name", 0.90, 0.86, "identity"),
        }
        for item in list(analysis.entities or []):
            if not isinstance(item, EntityItem):
                continue
            entity_type = str(item.type or "").strip().lower()
            target = mapping.get(entity_type)
            value = str(item.canonical or "").strip()
            if target is None or not value:
                continue
            if entity_type == "os_name" and not self._structured_os_entity_allowed(analysis=analysis, value=value):
                continue
            row = self._make_fact(
                subject=subject,
                predicate=target[0],
                value=self._analysis_value(predicate=target[0], value=value),
                confidence=max(target[1], float(item.confidence or 0.0)),
                importance=target[2],
                scope=scope,
                evidence=evidence,
                event_id=event_id,
                relation=target[3],
                namespace=namespace,
                metadata_extra={
                    "analysis_source": "ingest_analyzer",
                    "structured": True,
                    "structured_source": "entities",
                    "entity_type": entity_type,
                },
            )
            if row is not None:
                out.append(row)
        return out

    @staticmethod
    def _structured_os_entity_allowed(*, analysis: IngestAnalysis | None, value: str) -> bool:
        resolution = dict(getattr(analysis, "contextual_resolution", {}) or {})
        current_os_values = {
            str(x or "").strip().lower()
            for x in list(resolution.get("current_os_values") or [])
            if str(x or "").strip()
        }
        past_os_values = {
            str(x or "").strip().lower()
            for x in list(resolution.get("past_os_values") or [])
            if str(x or "").strip()
        }
        token = str(value or "").strip().lower()
        if current_os_values:
            return token in current_os_values
        if past_os_values and token in past_os_values:
            return False
        return True

    def _facts_from_numeric(
        self,
        *,
        analysis: IngestAnalysis | None,
        subject: str,
        scope: MemoryScope,
        event_id: str,
        namespace: str,
    ) -> list[FactRecordV2]:
        if analysis is None:
            return []
        out: list[FactRecordV2] = []
        evidence = str(analysis.raw_text or analysis.search_text or analysis.canonical_text or "").strip()
        mapping: dict[str, tuple[str, float, float, str]] = {
            "age_years": ("identity_age_years", 0.93, 0.88, "identity"),
            "vram_gb": ("environment_gpu_vram_gb", 0.90, 0.80, "environment"),
            "ram_gb": ("environment_ram_gb", 0.89, 0.78, "environment"),
            "memory_gb": ("environment_memory_gb", 0.78, 0.70, "environment"),
        }
        for item in list(analysis.numeric_facts or []):
            if not isinstance(item, NumericFact):
                continue
            kind = str(item.kind or "").strip().lower()
            target = mapping.get(kind)
            if target is None:
                continue
            row = self._make_fact(
                subject=subject,
                predicate=target[0],
                value=item.value,
                confidence=target[1],
                importance=target[2],
                scope=scope,
                evidence=evidence,
                event_id=event_id,
                relation=target[3],
                namespace=namespace,
                allow_numeric=True,
                metadata_extra={
                    "analysis_source": "ingest_analyzer",
                    "structured": True,
                    "structured_source": "numeric_facts",
                    "numeric_kind": kind,
                },
            )
            if row is not None:
                out.append(row)
        return out

    @staticmethod
    def _analysis_predicate(item: StableFact) -> str:
        subject = str(item.subject or "").strip().lower()
        predicate = str(item.predicate or "").strip().lower()
        mapping = {
            ("identity", "name"): "identity_name",
            ("identity", "age_years"): "identity_age_years",
            ("project", "name"): "project_name",
            ("environment", "os_name"): "environment_os",
            ("environment", "tool_name"): "environment_tool",
            ("environment", "python_version"): "environment_runtime_python",
            ("environment", "llm_model"): "environment_llm_model",
            ("hardware", "gpu_model"): "environment_gpu_model",
            ("hardware", "cpu_model"): "environment_cpu_model",
            ("hardware", "gpu_vram_size"): "environment_gpu_vram_size",
            ("hardware", "gpu_vram_gb"): "environment_gpu_vram_gb",
            ("hardware", "ram_size"): "environment_ram_size",
            ("hardware", "ram_gb"): "environment_ram_gb",
            ("hardware", "memory_gb"): "environment_memory_gb",
        }
        value = mapping.get((subject, predicate))
        if value:
            return value
        if subject and predicate:
            return f"{subject}_{predicate}"
        return predicate

    @staticmethod
    def _analysis_value(*, predicate: str, value: Any) -> Any:
        pred = str(predicate or "").strip().lower()
        raw = str(value or "").strip()
        if not raw:
            return value
        if pred == "environment_runtime_python":
            return raw if raw.lower().startswith("python ") else f"python {raw}"
        if pred in {"environment_os", "environment_tool"}:
            return raw.lower()
        return value

    def _make_fact(
        self,
        *,
        subject: str,
        predicate: str,
        value: Any,
        confidence: float,
        importance: float,
        scope: MemoryScope,
        evidence: str,
        event_id: str,
        relation: str,
        namespace: str,
        allow_numeric: bool = False,
        metadata_extra: dict[str, Any] | None = None,
    ) -> FactRecordV2 | None:
        rows: list[FactRecordV2] = []
        self._append_fact(
            rows,
            subject=subject,
            predicate=predicate,
            value=value,
            scope=scope,
            confidence=confidence,
            importance=importance,
            evidence=evidence,
            event_id=event_id,
            relation=relation,
            namespace=namespace,
            allow_numeric=allow_numeric,
            metadata_extra=metadata_extra,
        )
        return rows[0] if rows else None

    @staticmethod
    def _mk(
        *,
        subject: str,
        predicate: str,
        value: Any,
        scope: MemoryScope,
        confidence: float,
        importance: float,
        evidence: str,
        event_id: str,
        relation: str,
        valid_to: float | None = None,
        namespace: str = "default",
    ) -> FactRecordV2:
        pred = str(predicate or "").strip().lower().replace(" ", "_")
        canonical_key = f"{subject}.{pred}" if subject and pred else pred
        fact_text = f"{subject}.{pred}={value}"
        digest = hashlib.blake2b(fact_text.encode("utf-8"), digest_size=6).hexdigest()
        return FactRecordV2(
            subject=subject,
            predicate=pred,
            value=value,
            scope=scope,
            confidence=max(0.0, min(1.0, float(confidence))),
            importance=max(0.0, min(1.0, float(importance))),
            evidence=str(evidence or "")[:280],
            source_event_id=str(event_id or ""),
            valid_from=float(time.time()),
            valid_to=valid_to,
            status=MemoryStatus.ACTIVE,
            canonical_key=canonical_key,
            relation=str(relation or ""),
            id=f"factv2:{event_id}:{pred}:{digest}",
            text=fact_text,
            namespace=namespace,
            metadata={"relation": str(relation or ""), "key": pred},
        )

    @staticmethod
    def _normalize_text(text: str) -> str:
        src = str(text or "").strip()
        if not src:
            return ""
        src = (
            src.replace("“", '"')
            .replace("”", '"')
            .replace("«", '"')
            .replace("»", '"')
            .replace("’", "'")
            .replace("`", "'")
            .replace("\u00a0", " ")
        )
        return _SPACE_RE.sub(" ", src).strip()

    def _segments(self, text: str) -> list[str]:
        src = self._normalize_text(text)
        if not src:
            return []
        out: list[str] = []

        def _push(value: str) -> None:
            item = self._normalize_text(value).strip(" ,;:-")
            if item and item not in out:
                out.append(item)

        for row in _SENTENCE_SPLIT_RE.split(src):
            _push(row)
        if not out:
            _push(src)
        for row in list(out):
            for part in _CLAUSE_SPLIT_RE.split(row):
                _push(part)
        return out

    def _clean_value(self, value: str) -> str:
        src = self._normalize_text(value)
        src = src.strip(" \"'()[]{}")
        src = _TRIM_LEADING_RE.sub("", src)
        src = _TRIM_TRAILING_RE.sub("", src)
        src = re.sub(r"\s+(?:и|and|но|but)$", "", src, flags=re.I)
        return src.strip(" ,;:-")

    def _is_informative(self, value: str, *, min_chars: int = 3) -> bool:
        src = self._clean_value(value)
        if len(src) < max(1, int(min_chars)):
            return False
        low = src.lower()
        if low in _GENERIC_VALUES:
            return False
        if re.fullmatch(r"[\d\W_]+", src):
            return False
        if re.fullmatch(r"\d{1,4}", src):
            return False
        return True

    def _looks_like_name(self, value: str) -> bool:
        src = self._clean_value(value)
        if not self._is_informative(src, min_chars=2):
            return False
        tokens = [x for x in src.split() if x]
        if not tokens or len(tokens) > 3:
            return False
        for token in tokens:
            low = token.lower()
            if low in _NAME_BLOCKLIST:
                return False
            if not re.fullmatch(r"[A-Za-zА-Яа-яЁёІіЇїЄєҐґ][A-Za-zА-Яа-яЁёІіЇїЄєҐґ' -]{1,30}", token):
                return False
            if token[:1].lower() == token[:1]:
                return False
        return True

    def _looks_project_name(self, value: str) -> bool:
        src = self._clean_value(value)
        if not self._is_informative(src, min_chars=2):
            return False
        return bool(re.fullmatch(r"[A-Za-zА-Яа-яЁёІіЇїЄєҐґ0-9_.\-]{2,48}", src))

    def _looks_project_value(self, value: str) -> bool:
        src = self._clean_value(value)
        if not self._is_informative(src, min_chars=3):
            return False
        low = src.lower()
        if low in _PROJECT_VALUE_BLOCKLIST:
            return False
        if low.startswith(("что ", "that ")):
            return False
        return True

    def _looks_actionable_task(self, value: str) -> bool:
        src = self._clean_value(value)
        if not self._is_informative(src, min_chars=4):
            return False
        low = src.lower()
        first = str((low.split() or [""])[0])
        if any(first.endswith(suffix) for suffix in ("ть", "ти", "ться", "ировать", "овать", "ать", "ить", "еть")):
            return True
        return any(token in low for token in _ACTION_KEYWORDS)

    def _looks_preference_value(self, value: str) -> bool:
        src = self._clean_value(value)
        if not self._is_informative(src, min_chars=4):
            return False
        return not self._looks_actionable_task(src)

    def _canonical_term_token(self, value: str) -> str:
        src = self._clean_value(value).lower()
        if not src:
            return ""
        for token, markers in _CANONICAL_TERM_MAP.items():
            if any(marker in src for marker in markers):
                return token
        slug = self._normalized_signature(src).replace(" ", "_")
        return slug[:80]

    def _canonical_task_goal(self, value: str) -> str:
        src = self._clean_value(value)
        low = src.lower()
        if "write policy" in low or "политик" in low and "запис" in low:
            return "next_write_policy"
        if "memory" in low and "web" in low and ("before" in low or "потом" in low or "сначала" in low):
            return "finish_memory_before_web"
        if "memory" in low:
            return "next_memory"
        if "web" in low:
            return "next_web"
        token = self._canonical_term_token(src)
        if token:
            return f"next_{token}"
        return self._normalized_signature(src).replace(" ", "_")[:96]

    def _canonical_decision_value(self, value: str) -> str:
        src = self._clean_value(value)
        low = src.lower()
        if "search_text" in low or "search text" in low:
            if "canonical_text" in low or "canonical text" in low:
                if any(word in low for word in ("store", "keep", "хран", "save")):
                    return "store_search_text_and_canonical_text"
        if "мысли ассистента" in low or "assistant thought" in low:
            if any(word in low for word in ("не", "not", "don't", "dont")) and any(word in low for word in ("сохраня", "store", "save", "persist")):
                return "do_not_store_assistant_thoughts"
        if "write policy" in low or ("политик" in low and "запис" in low):
            if any(word in low for word in ("next", "следующ", "сначала", "потом", "делаем", "do", "build")):
                return "next_write_policy"
            return "write_policy"
        if "memory" in low and "web" in low and ("before" in low or "потом" in low or "сначала" in low):
            return "finish_memory_before_web"
        token = self._canonical_term_token(src)
        return token or self._normalized_signature(src).replace(" ", "_")[:96]

    def _canonical_agreed_plan(self, first: str, second: str) -> str:
        left = self._canonical_term_token(first)
        right = self._canonical_term_token(second)
        if left and right:
            return f"finish_{left}_before_{right}"
        merged = f"{self._clean_value(first)} {self._clean_value(second)}"
        return self._normalized_signature(merged).replace(" ", "_")[:96]

    def _has_any(self, text: str, tokens: tuple[str, ...]) -> bool:
        low = str(text or "").lower()
        return any(token in low for token in tokens)

    def _normalized_signature(self, value: str) -> str:
        src = self._clean_value(value).lower()
        if not src:
            return ""
        return _SPACE_RE.sub(" ", re.sub(r"[\W_]+", " ", src)).strip()

    @staticmethod
    def _subject_allows_fact(*, subject: str, predicate: str) -> bool:
        owner = str(subject or "").strip().lower()
        pred = str(predicate or "").strip().lower()
        if owner != "user" and (pred.startswith("environment_") or pred.startswith("identity_")):
            return False
        return True

    def _append_fact(
        self,
        out: list[FactRecordV2],
        *,
        subject: str,
        predicate: str,
        value: Any,
        scope: MemoryScope,
        confidence: float,
        importance: float,
        evidence: str,
        event_id: str,
        relation: str,
        namespace: str,
        valid_to: float | None = None,
        allow_numeric: bool = False,
        metadata_extra: dict[str, Any] | None = None,
    ) -> None:
        if not self._subject_allows_fact(subject=subject, predicate=predicate):
            return
        cleaned = self._clean_value(str(value or ""))
        if allow_numeric:
            if not cleaned or re.fullmatch(r"[\W_]+", cleaned):
                return
        elif not self._is_informative(cleaned, min_chars=2):
            return
        row = self._mk(
            subject=subject,
            predicate=predicate,
            value=cleaned,
            scope=scope,
            confidence=confidence,
            importance=importance,
            evidence=evidence,
            event_id=event_id,
            relation=relation,
            valid_to=valid_to,
            namespace=namespace,
        )
        if metadata_extra:
            row = FactRecordV2(
                subject=row.subject,
                predicate=row.predicate,
                value=row.value,
                scope=row.scope,
                confidence=row.confidence,
                importance=row.importance,
                evidence=row.evidence,
                source_event_id=row.source_event_id,
                valid_from=row.valid_from,
                valid_to=row.valid_to,
                status=row.status,
                canonical_key=row.canonical_key,
                relation=row.relation,
                id=row.id,
                text=row.text,
                memory_type=row.memory_type,
                level=row.level,
                namespace=row.namespace,
                metadata={**dict(row.metadata or {}), **dict(metadata_extra or {})},
                created_at=row.created_at,
                updated_at=row.updated_at,
                parent_id=row.parent_id,
                chunk_index=row.chunk_index,
                version=row.version,
            )
        out.append(row)

    def _extract_pattern_facts(
        self,
        *,
        segments: list[str],
        subject: str,
        scope: MemoryScope,
        event_id: str,
        namespace: str,
        rules: list[dict[str, Any]],
    ) -> list[FactRecordV2]:
        out: list[FactRecordV2] = []
        for segment in list(segments or []):
            for rule in list(rules or []):
                pattern = rule.get("pattern")
                if not hasattr(pattern, "finditer"):
                    continue
                for match in pattern.finditer(segment):
                    group_index = int(rule.get("group", 1))
                    try:
                        raw_value = str(match.group(group_index) or "")
                    except Exception:
                        raw_value = str(match.group(0) or "")
                    value = self._clean_value(raw_value)
                    transform = rule.get("transform")
                    if callable(transform):
                        value = self._clean_value(str(transform(value, segment, match) or ""))
                    validator = rule.get("validator")
                    if callable(validator) and not bool(validator(value)):
                        continue
                    self._append_fact(
                        out,
                        subject=subject,
                        predicate=str(rule.get("predicate") or ""),
                        value=value,
                        scope=(rule.get("scope") or scope),
                        confidence=float(rule.get("confidence", 0.70)),
                        importance=float(rule.get("importance", 0.70)),
                        evidence=segment,
                        event_id=event_id,
                        relation=str(rule.get("relation") or ""),
                        namespace=namespace,
                        valid_to=rule.get("valid_to"),
                    )
        return out

    def _identity_facts(
        self, text: str, *, segments: list[str], subject: str, scope: MemoryScope, event_id: str, namespace: str
    ) -> list[FactRecordV2]:
        out: list[FactRecordV2] = []
        if subject != "user":
            return out
        rules = [
            {
                "pattern": re.compile(
                    r"\b(?:my name is)\s+([A-Za-zА-Яа-яЁёІіЇїЄєҐґ][A-Za-zА-Яа-яЁёІіЇїЄєҐґ' -]{1,40})",
                    re.I,
                ),
                "predicate": "identity_name",
                "relation": "identity",
                "confidence": 0.88,
                "importance": 0.82,
                "validator": self._looks_like_name,
            },
            {
                "pattern": re.compile(r"\b(?:меня зовут)\s+([A-Za-zА-Яа-яЁёІіЇїЄєҐґ][A-Za-zА-Яа-яЁёІіЇїЄєҐґ' -]{1,40})", re.I),
                "predicate": "identity_name",
                "relation": "identity",
                "confidence": 0.88,
                "importance": 0.82,
                "validator": self._looks_like_name,
            },
        ]
        out.extend(
            self._extract_pattern_facts(
                segments=segments,
                subject=subject,
                scope=scope,
                event_id=event_id,
                namespace=namespace,
                rules=rules,
            )
        )
        for segment in list(segments or []):
            match = re.match(r"^[Яя]\s+([A-ZА-ЯЁ][A-Za-zА-Яа-яЁёІіЇїЄєҐґ' -]{1,30})(?:\b|[,.;!?])", segment)
            if not match:
                continue
            name = self._clean_value(str(match.group(1) or ""))
            if not self._looks_like_name(name):
                continue
            self._append_fact(
                out,
                subject=subject,
                predicate="identity_name",
                value=name,
                scope=scope,
                confidence=0.90,
                importance=0.84,
                evidence=segment,
                event_id=event_id,
                relation="identity",
                namespace=namespace,
            )
        return out

    def _project_facts(
        self, text: str, *, segments: list[str], subject: str, scope: MemoryScope, event_id: str, namespace: str
    ) -> list[FactRecordV2]:
        rules = [
            {
                "pattern": re.compile(r"\b(?:мой|наш)\s+проект\s+(?:называется|это|called|is called)\s+([A-Za-zА-Яа-яЁёІіЇїЄєҐґ0-9_.\-]{2,48})\b", re.I),
                "predicate": "project_name",
                "relation": "project",
                "scope": MemoryScope.PROJECT,
                "confidence": 0.83,
                "importance": 0.84,
                "validator": self._looks_project_name,
            },
            {
                "pattern": re.compile(r"\b(?:project|проект)\s+([A-Za-z][A-Za-z0-9_.\-]{2,40})\b", re.I),
                "predicate": "project_name",
                "relation": "project",
                "scope": MemoryScope.PROJECT,
                "confidence": 0.74,
                "importance": 0.78,
                "validator": self._looks_project_name,
            },
            {
                "pattern": re.compile(r"\b(?:в|для)\s+проекте\s+(?:используется|используем|использую|стоит|есть|держим|храним|работает|поднят)\s+([^.!?;\n]{3,140})", re.I),
                "predicate": "project_fact",
                "relation": "project",
                "scope": MemoryScope.PROJECT,
                "confidence": 0.80,
                "importance": 0.82,
                "validator": self._looks_project_value,
            },
            {
                "pattern": re.compile(r"\b(?:мой|наш)\s+проект\s+(?:использует|собран на|работает на|построен на|на)\s+([^.!?;\n]{3,140})", re.I),
                "predicate": "project_fact",
                "relation": "project",
                "scope": MemoryScope.PROJECT,
                "confidence": 0.79,
                "importance": 0.82,
                "validator": self._looks_project_value,
            },
            {
                "pattern": re.compile(r"\b(?:я|мы)\s+использу(?:ю|ем)\s+([^.!?;\n]{2,100})\s+(?:в|для)\s+проекта\b", re.I),
                "predicate": "project_fact",
                "relation": "project",
                "scope": MemoryScope.PROJECT,
                "confidence": 0.76,
                "importance": 0.79,
                "validator": self._looks_project_value,
            },
            {
                "pattern": re.compile(r"\b(?:i|we)\s+use\s+([^.!?;\n]{2,100})\s+(?:in|for)\s+(?:the\s+)?project\b", re.I),
                "predicate": "project_fact",
                "relation": "project",
                "scope": MemoryScope.PROJECT,
                "confidence": 0.76,
                "importance": 0.79,
                "validator": self._looks_project_value,
            },
        ]
        return self._extract_pattern_facts(
            segments=segments,
            subject=subject,
            scope=scope,
            event_id=event_id,
            namespace=namespace,
            rules=rules,
        )

    def _environment_facts(
        self,
        text: str,
        *,
        segments: list[str],
        subject: str,
        scope: MemoryScope,
        event_id: str,
        namespace: str,
        analysis: IngestAnalysis | None = None,
    ) -> list[FactRecordV2]:
        out: list[FactRecordV2] = []
        if subject != "user":
            return out
        candidate_segments = [segment for segment in list(segments or []) if self._has_any(segment, _ENVIRONMENT_CUES)]
        if not candidate_segments:
            return out
        seen_pairs: set[tuple[str, str]] = set()
        for segment in candidate_segments:
            for pattern, predicate, value in _ENVIRONMENT_STATIC_RULES:
                if not pattern.search(segment):
                    continue
                if predicate == "environment_os" and not self._structured_os_entity_allowed(analysis=analysis, value=str(value)):
                    continue
                key = (predicate, str(value).lower())
                if key in seen_pairs:
                    continue
                seen_pairs.add(key)
                self._append_fact(
                    out,
                    subject=subject,
                    predicate=predicate,
                    value=value,
                    scope=scope,
                    confidence=0.70,
                    importance=0.66,
                    evidence=segment,
                    event_id=event_id,
                    relation="environment",
                    namespace=namespace,
                )
            for pattern, predicate, label in _ENVIRONMENT_VERSION_RULES:
                for match in pattern.finditer(segment):
                    version = str(match.group(1) or "").strip()
                    if not version:
                        continue
                    fact_value = f"{label} {version}"
                    key = (predicate, fact_value.lower())
                    if key in seen_pairs:
                        continue
                    seen_pairs.add(key)
                    self._append_fact(
                        out,
                        subject=subject,
                        predicate=predicate,
                        value=fact_value,
                        scope=scope,
                        confidence=0.74,
                        importance=0.70,
                        evidence=segment,
                        event_id=event_id,
                        relation="environment",
                        namespace=namespace,
                    )
        return out

    def _preference_facts(
        self, text: str, *, segments: list[str], subject: str, scope: MemoryScope, event_id: str, namespace: str
    ) -> list[FactRecordV2]:
        rules = [
            {
                "pattern": re.compile(r"\b(?:i prefer|prefer)\s+([^.,;!?]{2,120})", re.I),
                "predicate": "preference",
                "relation": "preference",
                "confidence": 0.74,
                "importance": 0.68,
                "validator": self._looks_preference_value,
            },
            {
                "pattern": re.compile(r"\b(?:i like|i love)\s+([^.,;!?]{2,120})", re.I),
                "predicate": "preference",
                "relation": "preference",
                "confidence": 0.72,
                "importance": 0.66,
                "validator": self._looks_preference_value,
            },
            {
                "pattern": re.compile(r"\b(?:предпочитаю|люблю|мне нравится|мне удобнее|мне важнее)\s+([^.,;!?]{2,120})", re.I),
                "predicate": "preference",
                "relation": "preference",
                "confidence": 0.76,
                "importance": 0.70,
                "validator": self._looks_preference_value,
            },
        ]
        return self._extract_pattern_facts(
            segments=segments,
            subject=subject,
            scope=scope,
            event_id=event_id,
            namespace=namespace,
            rules=rules,
        )

    def _task_facts(
        self, text: str, *, segments: list[str], subject: str, scope: MemoryScope, event_id: str, namespace: str
    ) -> list[FactRecordV2]:
        rules = [
            {
                "pattern": re.compile(r"\b(?:need to|need|todo|task)\s+([^\n.!?]{3,180})", re.I),
                "predicate": "task",
                "relation": "task",
                "confidence": 0.69,
                "importance": 0.82,
                "validator": self._looks_actionable_task,
                "transform": lambda value, _segment, _match: self._canonical_task_goal(value),
            },
            {
                "pattern": re.compile(r"\b(?:нужно|надо|надо бы|сделай|задача)\s+([^\n.!?]{3,180})", re.I),
                "predicate": "task",
                "relation": "task",
                "confidence": 0.73,
                "importance": 0.84,
                "validator": self._looks_actionable_task,
                "transform": lambda value, _segment, _match: self._canonical_task_goal(value),
            },
            {
                "pattern": re.compile(r"\b(?:i want to|i want|я хочу)\s+([^\n.!?]{3,180})", re.I),
                "predicate": "task_goal",
                "relation": "task",
                "confidence": 0.70,
                "importance": 0.80,
                "validator": self._looks_actionable_task,
                "transform": lambda value, _segment, _match: self._canonical_task_goal(value),
            },
            {
                "pattern": re.compile(r"\b(?:next|следующим делом|следующий шаг)\s+([^\n.!?]{3,180})", re.I),
                "predicate": "task_goal",
                "relation": "task",
                "confidence": 0.74,
                "importance": 0.82,
                "validator": self._looks_project_value,
                "transform": lambda value, _segment, _match: self._canonical_task_goal(value),
            },
        ]
        out = self._extract_pattern_facts(
            segments=segments,
            subject=subject,
            scope=scope,
            event_id=event_id,
            namespace=namespace,
            rules=rules,
        )
        for segment in list(segments or []):
            match = _PLAN_SEQUENCE_RE.search(segment)
            if not match:
                continue
            plan_value = self._canonical_agreed_plan(str(match.group(1) or ""), str(match.group(2) or ""))
            if not self._is_informative(plan_value, min_chars=6):
                continue
            self._append_fact(
                out,
                subject=subject,
                predicate="agreed_plan",
                value=plan_value,
                scope=scope,
                confidence=0.84,
                importance=0.86,
                evidence=segment,
                event_id=event_id,
                relation="decision",
                namespace=namespace,
            )
        return out

    def _decision_facts(
        self, text: str, *, segments: list[str], subject: str, scope: MemoryScope, event_id: str, namespace: str
    ) -> list[FactRecordV2]:
        rules = [
            {
                "pattern": re.compile(r"\b(?:we decided|decided to|let's use|lets use|we(?:'ll| will) use)\s+([^.,;!?]{2,160})", re.I),
                "predicate": "decision",
                "relation": "decision",
                "confidence": 0.78,
                "importance": 0.84,
                "validator": self._looks_project_value,
                "transform": lambda value, _segment, _match: self._canonical_decision_value(value),
            },
            {
                "pattern": re.compile(r"\b(?:we(?:'ll| will)|will|let's|lets)\s+(?:store|keep|save)\s+([^.,;!?]{2,160})", re.I),
                "predicate": "decision",
                "relation": "decision",
                "confidence": 0.80,
                "importance": 0.86,
                "validator": self._looks_project_value,
                "transform": lambda value, _segment, _match: self._canonical_decision_value(f"store {value}"),
            },
            {
                "pattern": re.compile(r"\b(?:мы решили|решили|решено|оставляем|выбрали|не будем|пусть будет)\s+([^.,;!?]{2,160})", re.I),
                "predicate": "decision",
                "relation": "decision",
                "confidence": 0.80,
                "importance": 0.86,
                "validator": self._looks_project_value,
                "transform": lambda value, _segment, _match: self._canonical_decision_value(value),
            },
            {
                "pattern": re.compile(r"\b(?:будем)\s+(?:хранить|сохранять)\s+([^.,;!?]{2,160})", re.I),
                "predicate": "decision",
                "relation": "decision",
                "confidence": 0.80,
                "importance": 0.86,
                "validator": self._looks_project_value,
                "transform": lambda value, _segment, _match: self._canonical_decision_value(f"store {value}"),
            },
            {
                "pattern": re.compile(r"^(?:мы\s+)?будем\s+([^.,;!?]{2,160})", re.I),
                "predicate": "decision",
                "relation": "decision",
                "confidence": 0.74,
                "importance": 0.82,
                "validator": self._looks_project_value,
                "transform": lambda value, _segment, _match: self._canonical_decision_value(value),
            },
            {
                "pattern": re.compile(r"\b(?:договорились|agreed)\s+([^.,;!?]{2,160})", re.I),
                "predicate": "decision",
                "relation": "decision",
                "confidence": 0.82,
                "importance": 0.86,
                "validator": self._looks_project_value,
                "transform": lambda value, _segment, _match: self._canonical_decision_value(value),
            },
        ]
        return self._extract_pattern_facts(
            segments=segments,
            subject=subject,
            scope=scope,
            event_id=event_id,
            namespace=namespace,
            rules=rules,
        )

    def _issue_facts(
        self, text: str, *, segments: list[str], subject: str, scope: MemoryScope, event_id: str, namespace: str
    ) -> list[FactRecordV2]:
        rules = [
            {
                "pattern": re.compile(r"\b(?:problem is(?: that)?|issue is(?: that)?)\s+([^.!?;\n]{4,220})", re.I),
                "predicate": "issue",
                "relation": "issue",
                "confidence": 0.79,
                "importance": 0.82,
                "validator": self._is_informative,
            },
            {
                "pattern": re.compile(r"\b(?:проблема(?:\s+в\s+том,\s+что)?|беда\s+в\s+том,\s+что)\s+([^.!?;\n]{4,220})", re.I),
                "predicate": "issue",
                "relation": "issue",
                "confidence": 0.82,
                "importance": 0.84,
                "validator": self._is_informative,
            },
            {
                "pattern": re.compile(r"\b(?:не работает|сломалось|ломается|падает|сыпется|не ловит)\s+([^.!?;\n]{3,180})", re.I),
                "predicate": "issue",
                "relation": "issue",
                "confidence": 0.80,
                "importance": 0.82,
                "validator": self._is_informative,
            },
        ]
        out = self._extract_pattern_facts(
            segments=segments,
            subject=subject,
            scope=scope,
            event_id=event_id,
            namespace=namespace,
            rules=rules,
        )
        captured_signatures = {self._normalized_signature(str(row.value or "")) for row in out if str(row.predicate) == "issue"}
        for segment in list(segments or []):
            if not self._has_any(segment, _PROBLEM_MARKERS):
                continue
            segment_value = segment[:200]
            segment_signature = self._normalized_signature(segment_value)
            if segment_signature and any(signature and signature in segment_signature for signature in captured_signatures):
                continue
            if segment_signature in captured_signatures:
                continue
            self._append_fact(
                out,
                subject=subject,
                predicate="issue",
                value=segment_value,
                scope=scope,
                confidence=0.74,
                importance=0.80,
                evidence=segment,
                event_id=event_id,
                relation="issue",
                namespace=namespace,
            )
            if segment_signature:
                captured_signatures.add(segment_signature)
        return out

    def _relationship_facts(
        self, text: str, *, segments: list[str], subject: str, scope: MemoryScope, event_id: str, namespace: str
    ) -> list[FactRecordV2]:
        rules = [
            {
                "pattern": re.compile(r"\bmy\s+(team|manager|colleague|client)\b", re.I),
                "predicate": "relationship",
                "relation": "relationship",
                "confidence": 0.64,
                "importance": 0.57,
            },
            {
                "pattern": re.compile(r"\b(команда|менеджер|коллега|клиент)\b", re.I),
                "predicate": "relationship",
                "relation": "relationship",
                "confidence": 0.64,
                "importance": 0.57,
            },
        ]
        return self._extract_pattern_facts(
            segments=segments,
            subject=subject,
            scope=scope,
            event_id=event_id,
            namespace=namespace,
            rules=rules,
        )

    def _temporary_facts(
        self, text: str, *, segments: list[str], subject: str, scope: MemoryScope, event_id: str, namespace: str
    ) -> list[FactRecordV2]:
        low = str(text or "").lower()
        _ = segments
        if not any(token in low for token in ("for now", "temporarily", "временно", "пока", "пока что")):
            return []
        return [
            self._mk(
                subject=subject,
                predicate="temporary_fact",
                value=text[:160],
                scope=MemoryScope.TEMPORARY,
                confidence=0.62,
                importance=0.44,
                evidence=text,
                event_id=event_id,
                relation="temporary",
                valid_to=float(time.time() + (4 * 3600)),
                namespace=namespace,
            )
        ]

    def _resolution_facts(
        self, text: str, *, segments: list[str], subject: str, scope: MemoryScope, event_id: str, namespace: str
    ) -> list[FactRecordV2]:
        low = str(text or "").lower()
        _ = segments
        out: list[FactRecordV2] = []
        if any(token in low for token in ("still", "unresolved", "не решено", "все еще", "всё ещё")):
            out.append(
                self._mk(
                    subject=subject,
                    predicate="issue_status",
                    value="unresolved",
                    scope=scope,
                    confidence=0.71,
                    importance=0.76,
                    evidence=text,
                    event_id=event_id,
                    relation="unresolved",
                    namespace=namespace,
                )
            )
        elif any(token in low for token in ("resolved", "fixed", "починил", "решено", "исправлено")):
            out.append(
                self._mk(
                    subject=subject,
                    predicate="issue_status",
                    value="resolved",
                    scope=scope,
                    confidence=0.73,
                    importance=0.70,
                    evidence=text,
                    event_id=event_id,
                    relation="resolved",
                    namespace=namespace,
                )
            )
        return out

    @staticmethod
    def _dedupe_fact_records(rows: list[FactRecordV2]) -> list[FactRecordV2]:
        out: list[FactRecordV2] = []
        seen: set[tuple[str, str, str]] = set()
        for row in list(rows or []):
            normalized_value = _SPACE_RE.sub(" ", str(row.value or "").strip().lower())
            key = (
                str(row.subject or "").strip().lower(),
                str(row.predicate or "").strip().lower(),
                normalized_value,
            )
            if key in seen:
                continue
            seen.add(key)
            out.append(row)
        return out

    @staticmethod
    def _dedupe_v2(rows: list[FactRecordV2]) -> list[FactRecordV2]:
        return FactExtractor._dedupe_fact_records(rows)

def extract_facts(text: str) -> dict[str, Any]:
    rows = FactExtractor().extract_v2(text=text, metadata={}, speaker="user", scope=MemoryScope.CONVERSATION)
    return {"facts": [x.to_dict() for x in rows]}

