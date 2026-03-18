from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from typing import Any

from modules.nlu.normalizer import normalize_text
from modules.nlu.types import Fact
from memory.memory_models import FactRecordV2, MemoryRecord, MemoryScope, MemorySourceKind, MemoryType
from memory.storage_profile import compact_metadata_payload, sanitize_storage_metadata


def _is_empty_value(value: Any) -> bool:
    """Check if a value is empty (None, empty string, empty list, empty dict)."""
    if value is None:
        return True
    if isinstance(value, str) and value.strip() == "":
        return True
    if isinstance(value, (list, dict)) and len(value) == 0:
        return True
    return False


@dataclass(frozen=True)
class AssistantWriteDecision:
    action: str = "allow"
    reason: str = ""
    target_scope: MemoryScope | None = None
    allow_fact_records: bool = True
    signals: dict[str, Any] = field(default_factory=dict)

    @property
    def allow_store(self) -> bool:
        return str(self.action or "").strip().lower() != "skip"

    @property
    def allow_long_term(self) -> bool:
        return self.allow_store and self.target_scope not in {MemoryScope.TEMPORARY, MemoryScope.PRIVATE_RUNTIME}

    def to_dict(self) -> dict[str, Any]:
        """Compact dict for storage (without signals)."""
        return {
            "action": str(self.action or "").strip().lower() or "allow",
            "reason": str(self.reason or "").strip(),
            "target_scope": (
                str(self.target_scope.value)
                if isinstance(self.target_scope, MemoryScope)
                else (str(self.target_scope or "").strip() or "")
            ),
            "allow_store": bool(self.allow_store),
            "allow_long_term": bool(self.allow_long_term),
            "allow_fact_records": bool(self.allow_fact_records),
        }

    def to_dict_debug(self) -> dict[str, Any]:
        """Full dict with signals for debug logging."""
        return {
            **self.to_dict(),
            "signals": dict(self.signals or {}),
        }


@dataclass(frozen=True)
class FactWriteDecision:
    action: str = "allow"
    reason: str = ""
    group: str = ""
    allow_write: bool = True
    allow_supersede: bool = False
    signals: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": str(self.action or "").strip().lower() or "allow",
            "reason": str(self.reason or "").strip(),
            "group": str(self.group or "").strip(),
            "allow_write": bool(self.allow_write),
            "allow_supersede": bool(self.allow_supersede),
            "signals": dict(self.signals or {}),
        }


class MemoryPolicy:
    """Policy for selecting durable user facts for long-term memory."""

    _CORE_LONG_TERM_KEYS = {
        "location_place",
        "interest",
        "role",
    }
    _WORK_CONTEXT_KEYS = {
        "work_context",
        "work_domain",
        "work_stack",
        "work_role",
    }
    _EXPLICIT_PREFERENCE_KEYS = {
        "preference",
        "preferred_tool",
        "preferred_language",
        "preferred_editor",
        "preferred_style",
    }
    _DENY_KEYS = {
        "today_status",
        "emotion",
        "mood",
        "smalltalk",
        "commentary",
        "event",
        "last_event",
        "temporary_state",
        "one_time_event",
    }
    _EPHEMERAL_MARKERS = {
        "today",
        "yesterday",
        "tomorrow",
        "сегодня",
        "вчера",
        "завтра",
        "утром",
        "вечером",
        "ночью",
        "сейчас",
        "разово",
        "одноразово",
        "временно",
        "temporary",
        "once",
    }
    _NOISE_VALUES = {
        "",
        "ok",
        "норм",
        "нормально",
        "такое",
        "не знаю",
        "unknown",
        "none",
    }
    _ASSISTANT_FACTUAL_WEB_INTENTS = {"fx_rate", "weather", "news_release"}
    _ASSISTANT_FACTUAL_WEB_CATEGORIES = {"finance", "price", "weather", "news", "version", "external"}
    _ASSISTANT_FACTUAL_MODES = {
        "fx_rate",
        "price",
        "historical_factual",
        "latest_factual",
        "weather",
    }
    _CURRENCY_MARKERS = {
        "курс",
        "валют",
        "usd",
        "eur",
        "uah",
        "грн",
        "доллар",
        "евро",
        "exchange rate",
        "forex",
        "межбанк",
        "cash",
        "buy",
        "sell",
    }
    _PRICE_MARKERS = {
        "price",
        "cost",
        "стоим",
        "цена",
        "стоит",
        "buy now",
        "auction",
    }
    _ATTRIBUTION_MARKERS = {
        "по данным",
        "согласно",
        "according to",
        "reported by",
        "source:",
        "sources:",
        "данные",
    }
    _HISTORICAL_MARKERS = {
        "в 20",
        "в 19",
        "год",
        "году",
        "историчес",
        "historical",
        "history",
        "last year",
        "earlier",
        "ранее",
    }
    _VOLATILE_MARKERS = {
        "сегодня",
        "today",
        "сейчас",
        "now",
        "завтра",
        "tomorrow",
        "актуаль",
        "latest",
        "последн",
        "current",
    }
    _ASSISTANT_MEMORY_MISS_MARKERS = {
        "не помню",
        "не могу вспомнить",
        "не вижу в памяти",
        "не вижу в истории",
        "не вижу точного факта",
        "не вижу точной модели",
        "не уверен",
        "don't remember",
        "do not remember",
        "can't remember",
        "cannot remember",
        "can't see it in memory",
        "cannot see it in memory",
        "i can't see it in memory",
        "i cannot see it in memory",
    }
    _ASSISTANT_SELF_CHECK_MARKERS = {
        "проверь сам",
        "можешь проверить",
        "можно проверить",
        "посмотри сам",
        "чтобы узнать",
        "как посмотреть",
        "проверь через",
        "you can check",
        "check it yourself",
        "to check",
        "run this command",
        "use this command",
    }
    _ASSISTANT_HELP_COMMAND_PATTERNS: tuple[re.Pattern[str], ...] = (
        re.compile(r"\bpython\s+--version\b", re.I),
        re.compile(r"\bwinver\b", re.I),
        re.compile(r"\bwmic\b", re.I),
        re.compile(r"\bdxdiag\b", re.I),
        re.compile(r"\blspci\b", re.I),
        re.compile(r"\blshw\b", re.I),
        re.compile(r"\bdevice manager\b", re.I),
        re.compile(r"\bдиспетчер устройств\b", re.I),
    )

    _SINGLETON_GROUPS: dict[str, set[str]] = {
        "identity_name": {"identity_name"},
        "identity_age": {"identity_age_years"},
        "project_name": {"project_name"},
        "environment_os": {"environment_os"},
        "environment_python": {"environment_runtime_python"},
        "environment_llm_model": {"environment_llm_model"},
        "environment_gpu_model": {"environment_gpu_model"},
        "environment_cpu_model": {"environment_cpu_model"},
        "environment_gpu_vram": {"environment_gpu_vram_size", "environment_gpu_vram_gb"},
        "environment_ram": {"environment_ram_size", "environment_ram_gb", "environment_memory_gb"},
        "issue_status": {"issue_status"},
    }
    _ASSISTANT_DENY_PREDICATES = {
        "identity_name",
        "identity_age_years",
        "environment_os",
        "environment_tool",
        "environment_runtime_python",
        "environment_llm_model",
        "environment_gpu_model",
        "environment_cpu_model",
        "environment_gpu_vram_size",
        "environment_gpu_vram_gb",
        "environment_ram_size",
        "environment_ram_gb",
        "environment_memory_gb",
        "project_name",
    }
    _ASSISTANT_ALLOW_PREDICATES = {
        "decision",
        "task",
        "task_goal",
        "agreed_plan",
    }

    def __init__(
        self,
        *,
        assistant_factual_min_quality: float = 0.62,
        assistant_factual_min_confidence: float = 0.72,
    ):
        self._assistant_factual_min_quality = max(0.0, min(1.0, float(assistant_factual_min_quality)))
        self._assistant_factual_min_confidence = max(0.0, min(1.0, float(assistant_factual_min_confidence)))

    def select_facts_for_long_term(
        self,
        facts: list[Fact],
        min_confidence: float = 0.75,
    ) -> list[Fact]:
        if not facts:
            return []

        deduped = self.dedupe_facts(facts)
        selected: list[Fact] = []

        for fact in deduped:
            if not self._passes_confidence(fact, min_confidence):
                continue
            if not self._is_allowed_key(fact.key):
                continue
            if not self._is_stable_enough(fact):
                continue
            if self._is_ephemeral_or_noise(fact):
                continue
            selected.append(fact)

        return selected

    def dedupe_facts(self, facts: list[Fact]) -> list[Fact]:
        if not facts:
            return []

        dedup: dict[tuple[str, str], Fact] = {}

        for fact in facts:
            normalized_key = self._norm_key(fact.key)
            normalized_value = self._norm_value(fact.value)
            if not normalized_key or not normalized_value:
                continue

            dedupe_key = (normalized_key, normalized_value)
            existing = dedup.get(dedupe_key)
            if existing is None:
                dedup[dedupe_key] = replace(
                    fact,
                    key=normalized_key,
                    value=self._canonical_value_preserve_case(fact.value),
                    confidence=round(float(fact.confidence), 3),
                )
                continue

            merged_confidence = max(existing.confidence, fact.confidence)
            merged_evidence = max(1, int(existing.evidence_count)) + max(1, int(fact.evidence_count))
            merged_stable = existing.stable or fact.stable or merged_evidence >= 3

            # Keep best-looking value formatting from the higher-confidence fact.
            if fact.confidence > existing.confidence:
                merged_value = self._canonical_value_preserve_case(fact.value)
                merged_source = fact.source
            else:
                merged_value = existing.value
                merged_source = existing.source

            dedup[dedupe_key] = replace(
                existing,
                value=merged_value,
                confidence=round(float(merged_confidence), 3),
                stable=merged_stable,
                evidence_count=merged_evidence,
                source=merged_source,
            )

        return list(dedup.values())

    def decide_assistant_message_write(
        self,
        *,
        text: str,
        metadata: dict[str, Any],
        memory_type: MemoryType,
        requested_scope: MemoryScope,
    ) -> AssistantWriteDecision:
        if memory_type != MemoryType.MESSAGE:
            return AssistantWriteDecision(reason="non_message_memory")
        if requested_scope in {MemoryScope.PRIVATE_RUNTIME, MemoryScope.TEMPORARY}:
            return AssistantWriteDecision(reason="runtime_scope_bypass")

        signals = self._assistant_factual_signals(text=text, metadata=metadata)
        signals = self._merge_signal_maps(
            signals,
            self._assistant_memory_help_signals(text=text, metadata=metadata),
        )
        if bool(signals.get("memory_help_noise")):
            return AssistantWriteDecision(
                action="temporary_only",
                reason="assistant_memory_miss_help_temporary_only",
                target_scope=MemoryScope.TEMPORARY,
                allow_fact_records=False,
                signals=signals,
            )
        if not bool(signals.get("risky_factual_claim")):
            return AssistantWriteDecision(reason="not_risky_factual_claim", signals=signals)

        if bool(signals.get("requires_web_verification")) and not bool(signals.get("web_used")):
            return AssistantWriteDecision(
                action="skip",
                reason="assistant_factual_unverified_no_web",
                allow_fact_records=False,
                signals=signals,
            )

        if bool(signals.get("has_conflict")):
            return AssistantWriteDecision(
                action="skip",
                reason="assistant_factual_conflicted_evidence",
                allow_fact_records=False,
                signals=signals,
            )

        if float(signals.get("conflict_severity") or 0.0) >= 0.55:
            return AssistantWriteDecision(
                action="skip",
                reason="assistant_factual_high_conflict_severity",
                allow_fact_records=False,
                signals=signals,
            )

        if bool(signals.get("unresolved_type_mismatch")) and bool(signals.get("strict_evidence_category")):
            return AssistantWriteDecision(
                action="skip",
                reason="assistant_factual_type_mismatch_unresolved",
                allow_fact_records=False,
                signals=signals,
            )

        if bool(signals.get("cautious_synthesis")) and bool(signals.get("strict_evidence_category")):
            return AssistantWriteDecision(
                action="skip",
                reason="assistant_factual_cautious_synthesis",
                allow_fact_records=False,
                signals=signals,
            )

        quality_score = float(signals.get("quality_score") or 0.0)
        if bool(signals.get("requires_web_verification")) and quality_score < self._assistant_factual_min_quality:
            return AssistantWriteDecision(
                action="skip",
                reason="assistant_factual_low_evidence_quality",
                allow_fact_records=False,
                signals=signals,
            )

        if bool(signals.get("strict_evidence_category")) and float(signals.get("final_factual_confidence") or 0.0) < self._assistant_factual_min_confidence:
            return AssistantWriteDecision(
                action="skip",
                reason="assistant_factual_low_final_confidence",
                allow_fact_records=False,
                signals=signals,
            )

        if bool(signals.get("volatile_fact")):
            return AssistantWriteDecision(
                action="temporary_only",
                reason="assistant_factual_temporary_only",
                target_scope=MemoryScope.TEMPORARY,
                allow_fact_records=False,
                signals=signals,
            )

        return AssistantWriteDecision(
            action="allow",
            reason="assistant_factual_verified",
            target_scope=requested_scope,
            allow_fact_records=False,
            signals=signals,
        )

    @classmethod
    def fact_group(cls, predicate: str) -> str:
        pred = str(predicate or "").strip().lower()
        for group_name, members in cls._SINGLETON_GROUPS.items():
            if pred in members:
                return group_name
        return pred

    @classmethod
    def is_singleton_group(cls, group: str) -> bool:
        return str(group or "").strip().lower() in cls._SINGLETON_GROUPS

    @classmethod
    def is_singleton_predicate(cls, predicate: str) -> bool:
        return cls.is_singleton_group(cls.fact_group(predicate))

    def decide_fact_write(
        self,
        *,
        fact: FactRecordV2,
        source_role: str,
        existing_record: MemoryRecord | None = None,
    ) -> FactWriteDecision:
        predicate = str(getattr(fact, "predicate", "") or "").strip().lower()
        subject = str(getattr(fact, "subject", "") or "").strip().lower()
        source_role_norm = str(source_role or "").strip().lower()
        group = self.fact_group(predicate)

        if source_role_norm == "assistant":
            if predicate not in self._ASSISTANT_ALLOW_PREDICATES:
                return FactWriteDecision(
                    action="skip",
                    reason="assistant_fact_predicate_not_allowlisted",
                    group=group,
                    allow_write=False,
                    allow_supersede=False,
                    signals={"predicate": predicate, "subject": subject},
                )
            if predicate in self._ASSISTANT_DENY_PREDICATES:
                return FactWriteDecision(
                    action="skip",
                    reason="assistant_fact_predicate_denied",
                    group=group,
                    allow_write=False,
                    allow_supersede=False,
                    signals={"predicate": predicate, "subject": subject},
                )

        if existing_record is None:
            return FactWriteDecision(
                action="allow",
                reason="new_fact",
                group=group,
                allow_write=True,
                allow_supersede=False,
            )

        old_fact = dict(dict(existing_record.metadata or {}).get("fact") or {})
        old_predicate = str(old_fact.get("predicate") or "").strip().lower()
        old_group = self.fact_group(old_predicate)

        if old_group != group:
            return FactWriteDecision(
                action="parallel",
                reason="different_groups",
                group=group,
                allow_write=True,
                allow_supersede=False,
            )

        old_conf = self._coerce_float(
            old_fact.get("confidence"),
            default=self._coerce_float(existing_record.confidence, default=0.0),
        )
        new_conf = self._coerce_float(getattr(fact, "confidence", 0.0), default=0.0)
        old_value = self._norm_value(old_fact.get("value"))
        new_value = self._norm_value(getattr(fact, "value", ""))

        if old_value and new_value and old_value == new_value:
            return FactWriteDecision(
                action="keep_existing",
                reason="same_value",
                group=group,
                allow_write=False,
                allow_supersede=False,
                signals={"old_conf": old_conf, "new_conf": new_conf},
            )

        if new_conf >= old_conf:
            return FactWriteDecision(
                action="supersede",
                reason="same_group_higher_or_equal_confidence",
                group=group,
                allow_write=True,
                allow_supersede=True,
                signals={"old_conf": old_conf, "new_conf": new_conf},
            )

        return FactWriteDecision(
            action="keep_existing",
            reason="same_group_lower_confidence",
            group=group,
            allow_write=False,
            allow_supersede=False,
            signals={"old_conf": old_conf, "new_conf": new_conf},
        )

    @staticmethod
    def resolve_source_kind(
        *,
        role: str,
        memory_type: MemoryType,
        metadata: dict[str, Any] | None = None,
        thinking: str = "",
    ) -> MemorySourceKind:
        meta = dict(metadata or {})
        explicit = str(meta.get("source_kind") or "").strip().lower()
        valid = {item.value: item for item in MemorySourceKind}
        if explicit in valid:
            return valid[explicit]

        role_norm = str(role or "").strip().lower()
        source_norm = str(meta.get("source") or "").strip().lower()
        thinking_norm = str(thinking or meta.get("thinking") or "").strip()

        if memory_type == MemoryType.TOOL_RESULT or role_norm == "tool":
            return MemorySourceKind.TOOL_RESULT
        if role_norm in {"user", "human"}:
            return MemorySourceKind.USER
        if role_norm in {"assistant", "ai", "bot"}:
            if source_norm in {"assistant_thought", "thinking", "reasoning"} and thinking_norm:
                return MemorySourceKind.ASSISTANT_THOUGHT
            if explicit == MemorySourceKind.ASSISTANT_THOUGHT.value:
                return MemorySourceKind.ASSISTANT_THOUGHT
            return MemorySourceKind.ASSISTANT_REPLY
        if role_norm == "system":
            if source_norm in {"web_v2", "tool", "tool_result", "search_tool", "web_tool"}:
                return MemorySourceKind.TOOL_RESULT
            return MemorySourceKind.SYSTEM_DECISION
        return MemorySourceKind.USER

    @staticmethod
    def sanitize_metadata_for_storage(
        *,
        metadata: dict[str, Any] | None,
        source_kind: MemorySourceKind,
        thinking: str = "",
        storage_profile: str = "compact",
        compact_for_storage: bool = True,
    ) -> dict[str, Any]:
        """Sanitize metadata for storage.

        Args:
            metadata: Raw metadata dict
            source_kind: Source kind for the record
            thinking: Optional thinking/reasoning text to strip
            storage_profile: "compact" (default) or "debug"

        compact profile:
            - Strip debug-only fields
            - Remove duplicate text projections from metadata
            - Remove memory_analysis entirely
            - Keep only compact emotion_profile
            - Keep only non-empty lists/dicts
            - Strip assistant_write_policy.signals

        debug profile:
            - Keep all fields for debugging
        """
        out = dict(metadata or {})
        hidden = str(thinking or out.get("thinking") or "").strip()
        legacy_entities = out.pop("entities", None)
        if legacy_entities and "runtime_entities" not in out:
            out["runtime_entities"] = legacy_entities
            out["legacy_runtime_entities_stripped"] = True
        out.pop("thinking", None)
        # Tags stored at top-level, not in metadata
        out.pop("tags", None)
        out["source_kind"] = str(source_kind.value)
        if source_kind in {MemorySourceKind.ASSISTANT_REPLY, MemorySourceKind.ASSISTANT_THOUGHT} and hidden:
            out["assistant_thinking_stripped"] = True

        if compact_for_storage:
            return sanitize_storage_metadata(metadata=out, storage_profile=storage_profile)
        return out

    @staticmethod
    def _compact_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
        """Compact metadata for storage by removing duplicates and debug-only fields."""
        return compact_metadata_payload(metadata)

    @staticmethod
    def allow_fact_records_for_source(*, source_kind: MemorySourceKind) -> bool:
        return source_kind != MemorySourceKind.ASSISTANT_THOUGHT

    @staticmethod
    def allow_claim_records_for_source(*, source_kind: MemorySourceKind) -> bool:
        return source_kind == MemorySourceKind.USER

    def is_fact_allowed_for_source(
        self,
        *,
        predicate: str,
        source_role: str,
        source_kind: MemorySourceKind,
    ) -> bool:
        role_norm = str(source_role or "").strip().lower()
        pred = str(predicate or "").strip().lower()
        if source_kind == MemorySourceKind.ASSISTANT_THOUGHT:
            return False
        if role_norm == "assistant":
            return pred in self._ASSISTANT_ALLOW_PREDICATES
        return True

    def _is_allowed_key(self, key: str) -> bool:
        key_norm = self._norm_key(key)
        if not key_norm:
            return False
        if key_norm in self._DENY_KEYS:
            return False
        if key_norm in self._CORE_LONG_TERM_KEYS:
            return True
        if key_norm in self._WORK_CONTEXT_KEYS:
            return True
        if key_norm in self._EXPLICIT_PREFERENCE_KEYS:
            return True
        if key_norm.startswith("preference_") or key_norm.startswith("preferred_"):
            return True
        if key_norm.startswith("work_"):
            return True
        return False

    @staticmethod
    def _passes_confidence(fact: Fact, min_confidence: float) -> bool:
        return float(fact.confidence) >= float(min_confidence)

    def _is_stable_enough(self, fact: Fact) -> bool:
        key_norm = self._norm_key(fact.key)

        # Core personal facts should be explicitly stable.
        if key_norm in self._CORE_LONG_TERM_KEYS:
            return bool(fact.stable)

        # Preferences/work context can become durable either by explicit stability
        # or repeated confirmations in evidence.
        if key_norm in self._WORK_CONTEXT_KEYS or key_norm in self._EXPLICIT_PREFERENCE_KEYS:
            return bool(fact.stable) or int(fact.evidence_count) >= 3

        if key_norm.startswith("preference_") or key_norm.startswith("preferred_"):
            return bool(fact.stable) or int(fact.evidence_count) >= 3

        if key_norm.startswith("work_"):
            return bool(fact.stable) or int(fact.evidence_count) >= 3

        return False

    def _is_ephemeral_or_noise(self, fact: Fact) -> bool:
        value_norm = self._norm_value(fact.value)
        if not value_norm:
            return True
        if value_norm in self._NOISE_VALUES:
            return True

        tokens = set(value_norm.split())
        if tokens.intersection(self._EPHEMERAL_MARKERS):
            return True

        # Very long values are usually sentence fragments, not compact facts.
        if len(value_norm) > 96:
            return True
        if len(tokens) > 10:
            return True

        return False

    @staticmethod
    def _norm_key(key: str) -> str:
        return normalize_text(str(key or ""))

    @staticmethod
    def _norm_value(value: str) -> str:
        return normalize_text(str(value or ""))

    @staticmethod
    def _canonical_value_preserve_case(value: str) -> str:
        clean = str(value or "").strip()
        return clean

    def _assistant_factual_signals(self, *, text: str, metadata: dict[str, Any]) -> dict[str, Any]:
        raw_text = str(text or "").strip()
        low_text = raw_text.lower()
        norm_text = normalize_text(raw_text)
        web_intent = self._norm_value(
            metadata.get("web_query_intent")
            or metadata.get("web_intent")
            or metadata.get("resolved_intent")
        )
        factual_mode = self._norm_value(
            metadata.get("web_factual_mode")
            or metadata.get("factual_response_mode")
            or self._nested(metadata, "web_evidence_quality", "factual_mode")
        )
        web_category = self._norm_value(
            metadata.get("web_primary_category")
            or metadata.get("primary_category")
            or metadata.get("category")
        )
        web_used = self._boolish(metadata.get("web_used"))
        quality_score = self._coerce_float(
            metadata.get("web_evidence_quality_score"),
            default=self._coerce_float(self._nested(metadata, "web_evidence_quality", "score"), default=0.0),
        )
        final_factual_confidence = self._coerce_float(
            metadata.get("web_final_factual_confidence"),
            default=self._coerce_float(
                self._nested(metadata, "web_evidence_quality", "final_factual_confidence"),
                default=quality_score,
            ),
        )
        conflict_severity = self._coerce_float(
            metadata.get("web_conflict_severity"),
            default=self._coerce_float(self._nested(metadata, "web_evidence_quality", "conflict_severity"), default=0.0),
        )
        selected_avg_quality = self._coerce_float(
            metadata.get("web_selected_avg_quality"),
            default=self._coerce_float(self._nested(metadata, "web_evidence_quality", "selected_avg_quality"), default=quality_score),
        )
        cautious_synthesis = self._boolish(
            metadata.get("web_cautious_synthesis")
            if metadata.get("web_cautious_synthesis") is not None
            else self._nested(metadata, "web_evidence_quality", "cautious_synthesis")
        )
        selected_page_type = self._norm_value(
            metadata.get("web_selected_result_factual_page_type")
            or self._nested(metadata, "web_evidence_quality", "selected_result_factual_page_type")
        )
        conflict_flags = self._collect_conflict_flags(metadata)
        has_source_conflict = "source_conflict" in conflict_flags
        has_numeric_conflict = "numeric_conflict" in conflict_flags
        has_quality_issue = "low_evidence_quality" in conflict_flags
        type_mismatch_notes = [
            str(x).strip()
            for x in list(
                metadata.get("web_type_mismatch_notes")
                or self._nested(metadata, "web_evidence_quality", "type_mismatch_notes")
                or []
            )
            if str(x).strip()
        ]
        true_conflict_notes = [
            str(x).strip()
            for x in list(
                metadata.get("web_true_conflict_notes")
                or self._nested(metadata, "web_evidence_quality", "true_conflict_notes")
                or []
            )
            if str(x).strip()
        ]
        unresolved_type_mismatch = bool(type_mismatch_notes and not selected_page_type)

        currency_hit = self._contains_any(low_text, self._CURRENCY_MARKERS)
        price_hit = self._contains_any(low_text, self._PRICE_MARKERS) or bool(re.search(r"[$€₴]\s*\d", raw_text))
        attribution_hit = self._contains_any(low_text, self._ATTRIBUTION_MARKERS)
        historical_hit = bool(re.search(r"\b(?:19|20)\d{2}\b", raw_text)) and (
            self._contains_any(low_text, self._HISTORICAL_MARKERS) or attribution_hit
        )
        volatile_hit = self._contains_any(low_text, self._VOLATILE_MARKERS)
        web_factual_hit = (
            factual_mode in self._ASSISTANT_FACTUAL_MODES
            or web_intent in self._ASSISTANT_FACTUAL_WEB_INTENTS
            or web_category in self._ASSISTANT_FACTUAL_WEB_CATEGORIES
        )
        numeric_hit = bool(re.search(r"\b\d+(?:[.,]\d+)?\b", raw_text))

        risky_factual_claim = bool(
            web_factual_hit
            or currency_hit
            or price_hit
            or attribution_hit
            or historical_hit
        )
        requires_web_verification = bool(
            web_factual_hit
            or factual_mode in self._ASSISTANT_FACTUAL_MODES
            or unresolved_type_mismatch
            or currency_hit
            or price_hit
            or attribution_hit
            or historical_hit
            or (numeric_hit and volatile_hit)
        )
        volatile_fact = bool(
            factual_mode in {"fx_rate", "price", "latest_factual", "weather"}
            or web_intent in {"fx_rate", "weather", "news_release"}
            or currency_hit
            or price_hit
            or (web_category in {"finance", "price", "weather", "news"} and (volatile_hit or numeric_hit))
        )
        strict_numeric_category = bool(
            web_category in {"finance", "price", "external"}
            and (numeric_hit or currency_hit or price_hit or historical_hit)
        )
        strict_evidence_category = bool(
            factual_mode in self._ASSISTANT_FACTUAL_MODES
            or strict_numeric_category
            or (
                web_category in {"finance", "price", "weather", "news", "external", "version"}
                and (numeric_hit or currency_hit or price_hit or historical_hit or volatile_hit)
            )
        )
        marker_hits: list[str] = []
        if web_factual_hit:
            marker_hits.append("web_factual")
        if factual_mode:
            marker_hits.append(f"factual_mode:{factual_mode}")
        if currency_hit:
            marker_hits.append("currency")
        if price_hit:
            marker_hits.append("price")
        if attribution_hit:
            marker_hits.append("attribution")
        if historical_hit:
            marker_hits.append("historical")
        if volatile_hit:
            marker_hits.append("volatile")
        if numeric_hit:
            marker_hits.append("numeric")
        if has_quality_issue:
            marker_hits.append("low_quality")
        if cautious_synthesis:
            marker_hits.append("cautious_synthesis")
        if strict_numeric_category:
            marker_hits.append("strict_numeric")
        if strict_evidence_category:
            marker_hits.append("strict_evidence")
        if unresolved_type_mismatch:
            marker_hits.append("type_mismatch_unresolved")

        return {
            "web_used": bool(web_used),
            "web_intent": str(web_intent or ""),
            "factual_mode": str(factual_mode or ""),
            "web_primary_category": str(web_category or ""),
            "quality_score": float(max(0.0, min(1.0, quality_score))),
            "quality_min_required": float(self._assistant_factual_min_quality),
            "final_factual_confidence": float(max(0.0, min(1.0, final_factual_confidence))),
            "confidence_min_required": float(self._assistant_factual_min_confidence),
            "conflict_severity": float(max(0.0, min(1.0, conflict_severity))),
            "selected_avg_quality": float(max(0.0, min(1.0, selected_avg_quality))),
            "cautious_synthesis": bool(cautious_synthesis),
            "conflict_flags": list(conflict_flags),
            "source_conflict": bool(has_source_conflict),
            "numeric_conflict": bool(has_numeric_conflict),
            "has_conflict": bool(has_source_conflict or has_numeric_conflict),
            "selected_result_factual_page_type": str(selected_page_type or ""),
            "type_mismatch_notes": type_mismatch_notes[:6],
            "true_conflict_notes": true_conflict_notes[:6],
            "unresolved_type_mismatch": bool(unresolved_type_mismatch),
            "risky_factual_claim": bool(risky_factual_claim),
            "requires_web_verification": bool(requires_web_verification),
            "volatile_fact": bool(volatile_fact),
            "strict_numeric_category": bool(strict_numeric_category),
            "strict_evidence_category": bool(strict_evidence_category),
            "marker_hits": marker_hits,
            "text_len": int(len(norm_text)),
        }

    def _assistant_memory_help_signals(self, *, text: str, metadata: dict[str, Any]) -> dict[str, Any]:
        raw_text = str(text or "").strip()
        low_text = normalize_text(raw_text).lower()
        factual_mode = self._norm_value(
            metadata.get("factual_response_mode")
            or metadata.get("web_factual_mode")
        )
        recall_mode = self._norm_value(metadata.get("memory_recall_mode"))
        miss_hit = self._contains_any(low_text, self._ASSISTANT_MEMORY_MISS_MARKERS)
        self_check_hit = self._contains_any(low_text, self._ASSISTANT_SELF_CHECK_MARKERS)
        command_markers = [
            pattern.pattern
            for pattern in self._ASSISTANT_HELP_COMMAND_PATTERNS
            if pattern.search(raw_text)
        ]
        helper_command_hit = bool(command_markers)
        memory_help_noise = bool(
            miss_hit
            or (self_check_hit and helper_command_hit)
            or (
                recall_mode in {"exact_fact_recall", "self_memory_exact"}
                and (self_check_hit or helper_command_hit)
            )
        )
        marker_hits: list[str] = []
        if miss_hit:
            marker_hits.append("memory_miss")
        if self_check_hit:
            marker_hits.append("manual_check")
        if helper_command_hit:
            marker_hits.append("helper_command")
        if factual_mode:
            marker_hits.append(f"factual_mode:{factual_mode}")
        if recall_mode:
            marker_hits.append(f"recall_mode:{recall_mode}")
        return {
            "assistant_reply_kind": "memory_miss_help" if memory_help_noise else "",
            "memory_help_noise": bool(memory_help_noise),
            "memory_miss_signal": bool(miss_hit),
            "manual_check_signal": bool(self_check_hit),
            "helper_command_signal": bool(helper_command_hit),
            "helper_commands": command_markers[:6],
            "marker_hits": marker_hits,
        }

    @staticmethod
    def _merge_signal_maps(*items: dict[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        marker_hits: list[str] = []
        for item in items:
            row = dict(item or {})
            for key, value in row.items():
                if key == "marker_hits":
                    continue
                out[key] = value
            for token in list(row.get("marker_hits") or []):
                value = str(token or "").strip()
                if value and value not in marker_hits:
                    marker_hits.append(value)
        if marker_hits:
            out["marker_hits"] = marker_hits
        return out

    @staticmethod
    def _contains_any(text: str, markers: set[str]) -> bool:
        src = str(text or "").strip().lower()
        if not src:
            return False
        return any(marker in src for marker in markers)

    @staticmethod
    def _coerce_float(value: Any, *, default: float) -> float:
        try:
            return float(value)
        except Exception:
            return float(default)

    @staticmethod
    def _nested(row: dict[str, Any], key: str, nested_key: str) -> Any:
        payload = row.get(key)
        if isinstance(payload, dict):
            return payload.get(nested_key)
        return None

    @classmethod
    def _collect_conflict_flags(cls, metadata: dict[str, Any]) -> list[str]:
        out: list[str] = []
        for key in ("web_conflict_flags", "conflict_flags"):
            value = metadata.get(key)
            if isinstance(value, (list, tuple, set)):
                for row in value:
                    item = str(row or "").strip().lower()
                    if item and item not in out:
                        out.append(item)
        if cls._boolish(metadata.get("web_conflicting_sources")):
            out.append("source_conflict")
        notes = metadata.get("web_conflict_notes")
        if isinstance(notes, (list, tuple, set)):
            for row in notes:
                note = str(row or "").strip().lower()
                if not note:
                    continue
                if "numeric_conflict" in note and "numeric_conflict" not in out:
                    out.append("numeric_conflict")
                if "source_conflict" in note and "source_conflict" not in out:
                    out.append("source_conflict")
        quality = metadata.get("web_evidence_quality")
        if isinstance(quality, dict) and cls._boolish(quality.get("has_conflict")) and "source_conflict" not in out:
            out.append("source_conflict")
        if cls._boolish(metadata.get("web_low_evidence_quality")) and "low_evidence_quality" not in out:
            out.append("low_evidence_quality")
        return out

    @staticmethod
    def _boolish(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        text = str(value or "").strip().lower()
        return text in {"1", "true", "yes", "on"}
