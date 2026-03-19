from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any


IDENTITY_CORE_KEYS = {
    "addressing.canonical_name",
    "addressing.allowed_forms",
    "addressing.forbidden_forms",
    "addressing.use_name_by_default",
    "interaction.prefers_directness",
    "interaction.prefers_short_answers",
    "interaction.prefers_examples_on_user_code",
    "interaction.allow_light_teasing",
    "boundaries.avoid_baby_tone",
    "boundaries.avoid_repeating_question",
    "boundaries.avoid_inventing_user_facts",
    "emotional.frustration_softening",
    "emotional.reduce_teasing_when_user_irritated",
    "assistant_traits.warmth_baseline",
    "assistant_traits.directness_baseline",
    "assistant_traits.empathy_floor",
    "assistant_traits.professionalism_floor",
    "assistant_traits.sarcasm_ceiling",
}

_KEY_ALIASES = {
    "canonical_name": "addressing.canonical_name",
    "allowed_forms": "addressing.allowed_forms",
    "forbidden_forms": "addressing.forbidden_forms",
    "use_name_by_default": "addressing.use_name_by_default",
    "interaction_style.prefers_directness": "interaction.prefers_directness",
    "interaction_style.prefers_short_answers": "interaction.prefers_short_answers",
    "interaction_style.prefers_examples_on_user_code": "interaction.prefers_examples_on_user_code",
    "interaction_style.allows_light_teasing": "interaction.allow_light_teasing",
    "interaction_style.allow_light_teasing": "interaction.allow_light_teasing",
    "boundaries.avoid_baby_talk": "boundaries.avoid_baby_tone",
    "boundaries.do_not_invent_user_facts": "boundaries.avoid_inventing_user_facts",
    "emotional_handling.deescalate_on_irritation": "emotional.frustration_softening",
    "emotional_rules.deescalate_on_irritation": "emotional.frustration_softening",
    "emotional_handling.playfulness_downshift_on_user_distress": "emotional.reduce_teasing_when_user_irritated",
    "emotional_rules.playfulness_downshift_on_user_distress": "emotional.reduce_teasing_when_user_irritated",
    "assistant_trait_baseline.warmth": "assistant_traits.warmth_baseline",
    "assistant_trait_baseline.directness": "assistant_traits.directness_baseline",
    "assistant_trait_baseline.empathy": "assistant_traits.empathy_floor",
    "assistant_trait_baseline.professionalism": "assistant_traits.professionalism_floor",
    "assistant_trait_baseline.sarcasm": "assistant_traits.sarcasm_ceiling",
}

_SCHEMA_MAP = {
    "addressing.canonical_name": ("addressing", "canonical_name"),
    "addressing.allowed_forms": ("addressing", "allowed_forms"),
    "addressing.forbidden_forms": ("addressing", "forbidden_forms"),
    "addressing.use_name_by_default": ("addressing", "use_name_by_default"),
    "interaction.prefers_directness": ("interaction_style", "prefers_directness"),
    "interaction.prefers_short_answers": ("interaction_style", "prefers_short_answers"),
    "interaction.prefers_examples_on_user_code": ("interaction_style", "prefers_examples_on_user_code"),
    "interaction.allow_light_teasing": ("interaction_style", "allow_light_teasing"),
    "boundaries.avoid_baby_tone": ("boundaries", "avoid_baby_tone"),
    "boundaries.avoid_repeating_question": ("boundaries", "avoid_repeating_question"),
    "boundaries.avoid_inventing_user_facts": ("boundaries", "avoid_inventing_user_facts"),
    "emotional.frustration_softening": ("emotional_rules", "frustration_softening"),
    "emotional.reduce_teasing_when_user_irritated": ("emotional_rules", "reduce_teasing_when_user_irritated"),
    "assistant_traits.warmth_baseline": ("assistant_trait_baseline", "warmth_baseline"),
    "assistant_traits.directness_baseline": ("assistant_trait_baseline", "directness_baseline"),
    "assistant_traits.empathy_floor": ("assistant_trait_baseline", "empathy_floor"),
    "assistant_traits.professionalism_floor": ("assistant_trait_baseline", "professionalism_floor"),
    "assistant_traits.sarcasm_ceiling": ("assistant_trait_baseline", "sarcasm_ceiling"),
}

_LIST_ITEM_KEYS = {"allowed_forms", "forbidden_forms"}
_BOOLEAN_ITEM_KEYS = {
    "use_name_by_default",
    "prefers_examples_on_user_code",
    "allow_light_teasing",
    "avoid_baby_tone",
    "avoid_repeating_question",
    "avoid_inventing_user_facts",
}
_FLOAT_ITEM_KEYS = {
    "prefers_directness",
    "prefers_short_answers",
    "frustration_softening",
    "reduce_teasing_when_user_irritated",
    "warmth_baseline",
    "directness_baseline",
    "empathy_floor",
    "professionalism_floor",
    "sarcasm_ceiling",
}

_EXPLICIT_ONLY_KEYS = {
    "addressing.allowed_forms",
    "addressing.use_name_by_default",
    "interaction.prefers_directness",
    "interaction.prefers_short_answers",
    "interaction.allow_light_teasing",
    "emotional.frustration_softening",
    "emotional.reduce_teasing_when_user_irritated",
    "assistant_traits.warmth_baseline",
    "assistant_traits.directness_baseline",
    "assistant_traits.empathy_floor",
    "assistant_traits.professionalism_floor",
    "assistant_traits.sarcasm_ceiling",
}
_DIRECT_PROHIBITION_KEYS = {
    "addressing.forbidden_forms",
    "boundaries.avoid_baby_tone",
    "boundaries.avoid_repeating_question",
    "boundaries.avoid_inventing_user_facts",
}
_STRONG_SIGNAL_KEYS = {
    "addressing.canonical_name",
    "interaction.prefers_examples_on_user_code",
}
_SYSTEM_SOURCE_KEYS = {
    "assistant_traits.warmth_baseline",
    "assistant_traits.directness_baseline",
    "assistant_traits.empathy_floor",
    "assistant_traits.professionalism_floor",
    "assistant_traits.sarcasm_ceiling",
}
_PROTECTED_OVERRIDE_KEYS = {
    "addressing.canonical_name",
    "addressing.allowed_forms",
    "addressing.forbidden_forms",
    "assistant_traits.warmth_baseline",
    "assistant_traits.directness_baseline",
    "assistant_traits.empathy_floor",
    "assistant_traits.professionalism_floor",
    "assistant_traits.sarcasm_ceiling",
}
PROTECTED_IDENTITY_CORE_KEYS = tuple(sorted(_PROTECTED_OVERRIDE_KEYS))

_DONOT_CALL_RE = re.compile(
    r"(?:не\s+называй\s+меня|не\s+зови\s+меня|don't\s+call\s+me|do\s+not\s+call\s+me)\s+"
    r"([A-Za-zА-Яа-яЁёІіЇїЄєҐґ][A-Za-zА-Яа-яЁёІіЇїЄєҐґ' -]{0,30}?)"
    r"(?=\s+(?:и|and|but)\b|[,.!?;]|$)",
    re.I,
)
_BOUNDARY_RULES: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(r"(?:не\s+сюсюкай|don't\s+baby(?:-|\s)?talk|do\s+not\s+baby(?:-|\s)?talk)", re.I),
        "boundaries.avoid_baby_tone",
    ),
    (
        re.compile(r"(?:не\s+повторяй\s+вопрос|don't\s+repeat\s+(?:the\s+)?question|do\s+not\s+repeat\s+(?:the\s+)?question)", re.I),
        "boundaries.avoid_repeating_question",
    ),
    (
        re.compile(
            r"(?:не\s+(?:придумывай|выдумывай)\s+факты\s+про\s+меня|don't\s+invent\s+(?:user\s+)?facts|do\s+not\s+invent\s+(?:user\s+)?facts|don't\s+make\s+up\s+facts\s+about\s+me)",
            re.I,
        ),
        "boundaries.avoid_inventing_user_facts",
    ),
)
_EXAMPLES_ON_USER_CODE_RE = re.compile(
    r"(?:пример|примеры|examples?).{0,32}(?:мо(?:е|ё)м?\s+код|my\s+code|user\s+code)|(?:на\s+мо(?:е|ё)м\s+коде|on\s+my\s+code).{0,24}(?:пример|examples?)",
    re.I,
)


@dataclass(frozen=True)
class IdentityCoreRecord:
    key: str
    value: Any
    confidence: float = 1.0
    source: str = ""
    requires_confirmation_to_override: bool = True
    updated_at: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "value": self.value,
            "confidence": float(self.confidence or 0.0),
            "source": self.source,
            "requires_confirmation_to_override": bool(self.requires_confirmation_to_override),
            "updated_at": float(self.updated_at or 0.0),
        }

    @classmethod
    def from_dict(cls, row: dict[str, Any] | None) -> "IdentityCoreRecord":
        item = dict(row or {})
        return cls(
            key=str(item.get("key") or "").strip(),
            value=item.get("value"),
            confidence=float(item.get("confidence") or 0.0),
            source=str(item.get("source") or "").strip(),
            requires_confirmation_to_override=bool(
                item.get("requires_confirmation_to_override", True)
            ),
            updated_at=float(item.get("updated_at") or 0.0),
        )


@dataclass(frozen=True)
class IdentityCoreCandidate:
    record: IdentityCoreRecord
    explicit_confirmation: bool = False
    signals: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "record": self.record.to_dict(),
            "explicit_confirmation": bool(self.explicit_confirmation),
            "signals": dict(self.signals or {}),
        }


@dataclass(frozen=True)
class IdentityCoreWriteDecision:
    action: str = "skip"  # allow | keep_existing | skip
    reason: str = ""
    normalized_key: str = ""
    allow_write: bool = False
    explicit_confirmation: bool = False
    requires_confirmation_to_override: bool = True
    signals: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": str(self.action or "").strip().lower() or "skip",
            "reason": str(self.reason or "").strip(),
            "normalized_key": str(self.normalized_key or "").strip(),
            "allow_write": bool(self.allow_write),
            "explicit_confirmation": bool(self.explicit_confirmation),
            "requires_confirmation_to_override": bool(self.requires_confirmation_to_override),
            "signals": dict(self.signals or {}),
        }


@dataclass(frozen=True)
class IdentityCoreSnapshot:
    addressing: dict[str, Any] = field(default_factory=dict)
    interaction_style: dict[str, Any] = field(default_factory=dict)
    boundaries: dict[str, Any] = field(default_factory=dict)
    emotional_rules: dict[str, Any] = field(default_factory=dict)
    assistant_trait_baseline: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "addressing": dict(self.addressing or {}),
            "interaction_style": dict(self.interaction_style or {}),
            "boundaries": dict(self.boundaries or {}),
            "emotional_rules": dict(self.emotional_rules or {}),
            "assistant_trait_baseline": dict(self.assistant_trait_baseline or {}),
        }


class IdentityCoreManager:
    _AUTO_NAME_CONFIDENCE = 0.88
    _STRONG_PREFERENCE_CONFIDENCE = 0.82

    def build_snapshot(
        self,
        *,
        rows: list[dict[str, Any]],
    ) -> IdentityCoreSnapshot:
        chosen: dict[str, IdentityCoreRecord] = {}
        for raw_row in list(rows or []):
            record = IdentityCoreRecord.from_dict(raw_row if isinstance(raw_row, dict) else {})
            key = self._normalize_key(record.key)
            if not key:
                continue
            normalized = IdentityCoreRecord(
                key=key,
                value=record.value,
                confidence=float(record.confidence or 0.0),
                source=str(record.source or "").strip(),
                requires_confirmation_to_override=bool(record.requires_confirmation_to_override),
                updated_at=float(record.updated_at or 0.0),
            )
            previous = chosen.get(key)
            if previous is None or self._is_better_candidate(new=normalized, old=previous):
                chosen[key] = normalized

        addressing: dict[str, Any] = {}
        interaction_style: dict[str, Any] = {}
        boundaries: dict[str, Any] = {}
        emotional_rules: dict[str, Any] = {}
        assistant_trait_baseline: dict[str, float] = {}

        for key, record in chosen.items():
            section, item_key = self._classify_key(key)
            if not section or not item_key:
                continue
            if section == "addressing":
                addressing[item_key] = self._normalize_value_for_key(item_key, record.value)
                continue
            if section == "interaction_style":
                interaction_style[item_key] = self._normalize_value_for_key(item_key, record.value)
                continue
            if section == "boundaries":
                boundaries[item_key] = bool(record.value)
                continue
            if section == "emotional_rules":
                emotional_rules[item_key] = self._normalize_value_for_key(item_key, record.value)
                continue
            if section == "assistant_trait_baseline":
                assistant_trait_baseline[item_key] = self._normalize_float(record.value, default=0.0)

        return IdentityCoreSnapshot(
            addressing=addressing,
            interaction_style=interaction_style,
            boundaries=boundaries,
            emotional_rules=emotional_rules,
            assistant_trait_baseline=assistant_trait_baseline,
        )

    def should_allow_override(
        self,
        *,
        key: str,
        old_value: Any,
        new_value: Any,
        explicit_confirmation: bool,
    ) -> bool:
        normalized_key = self._normalize_key(key)
        if not normalized_key:
            return True
        if self._values_equal(old_value, new_value):
            return True
        if bool(explicit_confirmation):
            return True
        if normalized_key in _PROTECTED_OVERRIDE_KEYS:
            return False
        return True

    def decide_write(
        self,
        *,
        candidate: IdentityCoreCandidate,
        existing_record: IdentityCoreRecord | None = None,
        source_role: str,
    ) -> IdentityCoreWriteDecision:
        record = candidate.record
        key = self._normalize_key(record.key)
        role = str(source_role or "").strip().lower()
        signals = dict(candidate.signals or {})
        signals["source_role"] = role
        if not key:
            return IdentityCoreWriteDecision(
                action="skip",
                reason="key_not_whitelisted",
                normalized_key="",
                allow_write=False,
                explicit_confirmation=bool(candidate.explicit_confirmation),
                signals=signals,
            )
        if role not in {"user", "system"}:
            return IdentityCoreWriteDecision(
                action="skip",
                reason="source_role_not_allowed",
                normalized_key=key,
                allow_write=False,
                explicit_confirmation=bool(candidate.explicit_confirmation),
                requires_confirmation_to_override=self._requires_confirmation_to_override(key),
                signals=signals,
            )

        explicit = bool(candidate.explicit_confirmation or signals.get("explicit_confirmation"))
        if existing_record is not None and self._values_equal(existing_record.value, record.value):
            return IdentityCoreWriteDecision(
                action="keep_existing",
                reason="same_value_existing",
                normalized_key=key,
                allow_write=False,
                explicit_confirmation=explicit,
                requires_confirmation_to_override=bool(
                    existing_record.requires_confirmation_to_override
                ),
                signals=signals,
            )

        if key in _SYSTEM_SOURCE_KEYS:
            if existing_record is not None and not self.should_allow_override(
                key=key,
                old_value=existing_record.value,
                new_value=record.value,
                explicit_confirmation=explicit,
            ):
                return IdentityCoreWriteDecision(
                    action="skip",
                    reason="override_requires_confirmation",
                    normalized_key=key,
                    allow_write=False,
                    explicit_confirmation=explicit,
                    requires_confirmation_to_override=self._requires_confirmation_to_override(key),
                    signals=signals,
                )
            if role == "system" or explicit or bool(signals.get("system_setting")):
                return self._allow(
                    key=key,
                    reason=("assistant_trait_from_system" if role == "system" or bool(signals.get("system_setting")) else "assistant_trait_explicit_confirmation"),
                    explicit_confirmation=explicit,
                    signals=signals,
                )
            return self._skip(
                key=key,
                reason="assistant_traits_require_system_or_confirmation",
                explicit_confirmation=explicit,
                signals=signals,
            )

        if key == "addressing.canonical_name":
            if existing_record is not None and not self.should_allow_override(
                key=key,
                old_value=existing_record.value,
                new_value=record.value,
                explicit_confirmation=explicit,
            ):
                return IdentityCoreWriteDecision(
                    action="skip",
                    reason="override_requires_confirmation",
                    normalized_key=key,
                    allow_write=False,
                    explicit_confirmation=explicit,
                    requires_confirmation_to_override=self._requires_confirmation_to_override(key),
                    signals=signals,
                )
            if explicit or bool(signals.get("stable_self_identification")) or float(record.confidence or 0.0) >= self._AUTO_NAME_CONFIDENCE:
                return self._allow(
                    key=key,
                    reason=("canonical_name_explicit_confirmation" if explicit else "canonical_name_strong_self_identification"),
                    explicit_confirmation=explicit,
                    signals=signals,
                )
            return self._skip(
                key=key,
                reason="canonical_name_requires_strong_self_identification",
                explicit_confirmation=explicit,
                signals=signals,
            )

        if key in _DIRECT_PROHIBITION_KEYS:
            if explicit or bool(signals.get("direct_user_prohibition")):
                return self._allow(
                    key=key,
                    reason=("direct_prohibition_explicit_confirmation" if explicit else "direct_user_prohibition"),
                    explicit_confirmation=explicit,
                    signals=signals,
                )
            return self._skip(
                key=key,
                reason="boundary_requires_direct_prohibition",
                explicit_confirmation=explicit,
                signals=signals,
            )

        if key == "interaction.prefers_examples_on_user_code":
            if existing_record is not None and not self.should_allow_override(
                key=key,
                old_value=existing_record.value,
                new_value=record.value,
                explicit_confirmation=explicit,
            ):
                return IdentityCoreWriteDecision(
                    action="skip",
                    reason="override_requires_confirmation",
                    normalized_key=key,
                    allow_write=False,
                    explicit_confirmation=explicit,
                    requires_confirmation_to_override=self._requires_confirmation_to_override(key),
                    signals=signals,
                )
            if explicit or bool(signals.get("stable_project_rule")) or float(record.confidence or 0.0) >= self._STRONG_PREFERENCE_CONFIDENCE:
                return self._allow(
                    key=key,
                    reason=("interaction_explicit_confirmation" if explicit else "stable_project_rule"),
                    explicit_confirmation=explicit,
                    signals=signals,
                )
            return self._skip(
                key=key,
                reason="interaction_requires_strong_preference_signal",
                explicit_confirmation=explicit,
                signals=signals,
            )

        if key in _EXPLICIT_ONLY_KEYS and not explicit:
            return self._skip(
                key=key,
                reason="identity_core_key_requires_explicit_confirmation",
                explicit_confirmation=explicit,
                signals=signals,
            )

        if explicit:
            if existing_record is not None and not self.should_allow_override(
                key=key,
                old_value=existing_record.value,
                new_value=record.value,
                explicit_confirmation=explicit,
            ):
                return IdentityCoreWriteDecision(
                    action="skip",
                    reason="override_requires_confirmation",
                    normalized_key=key,
                    allow_write=False,
                    explicit_confirmation=explicit,
                    requires_confirmation_to_override=self._requires_confirmation_to_override(key),
                    signals=signals,
                )
            return self._allow(
                key=key,
                reason="explicit_confirmation",
                explicit_confirmation=explicit,
                signals=signals,
            )

        return self._skip(
            key=key,
            reason="identity_core_signal_too_weak",
            explicit_confirmation=explicit,
            signals=signals,
        )

    def extract_candidates_from_fact(
        self,
        *,
        fact: Any,
        source_role: str,
        now_ts: float | None = None,
    ) -> list[IdentityCoreCandidate]:
        role = str(source_role or "").strip().lower()
        if role != "user":
            return []
        predicate = str(getattr(fact, "predicate", "") or "").strip().lower()
        value = getattr(fact, "value", None)
        confidence = self._coerce_float(getattr(fact, "confidence", 0.0), default=0.0)
        updated_at = float(getattr(fact, "updated_at", 0.0) or now_ts or time.time())
        if predicate == "identity_name":
            clean_value = self._clean_text_value(value)
            if not clean_value:
                return []
            return [
                IdentityCoreCandidate(
                    record=IdentityCoreRecord(
                        key="addressing.canonical_name",
                        value=clean_value,
                        confidence=confidence,
                        source="fact:self_identification",
                        requires_confirmation_to_override=True,
                        updated_at=updated_at,
                    ),
                    explicit_confirmation=bool(
                        dict(getattr(fact, "metadata", {}) or {}).get("identity_core_explicit_confirmation")
                    ),
                    signals={
                        "stable_self_identification": bool(confidence >= self._AUTO_NAME_CONFIDENCE),
                        "fact_predicate": predicate,
                    },
                )
            ]
        return []

    def extract_candidates_from_message(
        self,
        *,
        text: str,
        metadata: dict[str, Any] | None,
        source_role: str,
        now_ts: float | None = None,
    ) -> list[IdentityCoreCandidate]:
        role = str(source_role or "").strip().lower()
        if role != "user":
            return []
        src = str(text or "").strip()
        if not src:
            return []
        meta = dict(metadata or {})
        explicit = bool(meta.get("identity_core_explicit_confirmation") or meta.get("explicit_confirmation"))
        updated_at = float(now_ts or time.time())
        out: list[IdentityCoreCandidate] = []

        for match in _DONOT_CALL_RE.finditer(src):
            form = self._clean_text_value(match.group(1))
            if not form:
                continue
            out.append(
                IdentityCoreCandidate(
                    record=IdentityCoreRecord(
                        key="addressing.forbidden_forms",
                        value=[form],
                        confidence=0.97,
                        source="user_direct_prohibition",
                        requires_confirmation_to_override=True,
                        updated_at=updated_at,
                    ),
                    explicit_confirmation=explicit,
                    signals={
                        "direct_user_prohibition": True,
                        "message_rule": "forbidden_form",
                    },
                )
            )

        for pattern, key in _BOUNDARY_RULES:
            if not pattern.search(src):
                continue
            out.append(
                IdentityCoreCandidate(
                    record=IdentityCoreRecord(
                        key=key,
                        value=True,
                        confidence=0.96,
                        source="user_direct_prohibition",
                        requires_confirmation_to_override=True,
                        updated_at=updated_at,
                    ),
                    explicit_confirmation=explicit,
                    signals={
                        "direct_user_prohibition": True,
                        "message_rule": key,
                    },
                )
            )

        if _EXAMPLES_ON_USER_CODE_RE.search(src):
            out.append(
                IdentityCoreCandidate(
                    record=IdentityCoreRecord(
                        key="interaction.prefers_examples_on_user_code",
                        value=True,
                        confidence=0.90,
                        source="stable_project_rule",
                        requires_confirmation_to_override=True,
                        updated_at=updated_at,
                    ),
                    explicit_confirmation=explicit,
                    signals={
                        "stable_project_rule": True,
                        "message_rule": "interaction.prefers_examples_on_user_code",
                    },
                )
            )

        return out

    @staticmethod
    def merge_values_for_key(key: str, left: Any, right: Any) -> Any:
        normalized_key = IdentityCoreManager._normalize_key(key)
        if not normalized_key:
            return right
        _, item_key = IdentityCoreManager._classify_key(normalized_key)
        if item_key in _LIST_ITEM_KEYS:
            merged: list[str] = []
            seen: set[str] = set()
            for raw in [*list(left or []), *list(right or [])]:
                token = str(raw or "").strip()
                low = token.lower()
                if not token or low in seen:
                    continue
                seen.add(low)
                merged.append(token)
            return merged
        return right

    @staticmethod
    def _allow(
        *,
        key: str,
        reason: str,
        explicit_confirmation: bool,
        signals: dict[str, Any],
    ) -> IdentityCoreWriteDecision:
        return IdentityCoreWriteDecision(
            action="allow",
            reason=reason,
            normalized_key=key,
            allow_write=True,
            explicit_confirmation=bool(explicit_confirmation),
            requires_confirmation_to_override=IdentityCoreManager._requires_confirmation_to_override(key),
            signals=dict(signals or {}),
        )

    @staticmethod
    def _skip(
        *,
        key: str,
        reason: str,
        explicit_confirmation: bool,
        signals: dict[str, Any],
    ) -> IdentityCoreWriteDecision:
        return IdentityCoreWriteDecision(
            action="skip",
            reason=reason,
            normalized_key=key,
            allow_write=False,
            explicit_confirmation=bool(explicit_confirmation),
            requires_confirmation_to_override=IdentityCoreManager._requires_confirmation_to_override(key),
            signals=dict(signals or {}),
        )

    @staticmethod
    def _normalize_key(value: str) -> str:
        raw = str(value or "").strip().lower()
        if not raw:
            return ""
        aliased = str(_KEY_ALIASES.get(raw) or raw)
        return aliased if aliased in IDENTITY_CORE_KEYS else ""

    @staticmethod
    def _is_better_candidate(*, new: IdentityCoreRecord, old: IdentityCoreRecord) -> bool:
        if float(new.confidence or 0.0) != float(old.confidence or 0.0):
            return float(new.confidence or 0.0) > float(old.confidence or 0.0)
        if float(new.updated_at or 0.0) != float(old.updated_at or 0.0):
            return float(new.updated_at or 0.0) >= float(old.updated_at or 0.0)
        return bool(new.requires_confirmation_to_override) >= bool(old.requires_confirmation_to_override)

    @staticmethod
    def _classify_key(key: str) -> tuple[str, str]:
        raw = str(key or "").strip().lower()
        if not raw:
            return "", ""
        return tuple(_SCHEMA_MAP.get(raw) or ("", ""))

    @staticmethod
    def _requires_confirmation_to_override(key: str) -> bool:
        normalized = IdentityCoreManager._normalize_key(key)
        return bool(normalized)

    @staticmethod
    def _normalize_value_for_key(key: str, value: Any) -> Any:
        if key in _LIST_ITEM_KEYS:
            return [
                str(item).strip()
                for item in list(value or [])
                if str(item).strip()
            ]
        if key in _BOOLEAN_ITEM_KEYS:
            return bool(value)
        if key in _FLOAT_ITEM_KEYS:
            return IdentityCoreManager._normalize_float(value, default=0.0)
        return str(value).strip() if isinstance(value, str) else value

    @staticmethod
    def _normalize_float(value: Any, *, default: float) -> float:
        try:
            return max(0.0, min(1.0, float(value)))
        except Exception:
            return float(default)

    @staticmethod
    def _values_equal(left: Any, right: Any) -> bool:
        if isinstance(left, list) or isinstance(right, list):
            return [str(x).strip() for x in list(left or []) if str(x).strip()] == [
                str(x).strip() for x in list(right or []) if str(x).strip()
            ]
        return left == right

    @staticmethod
    def _coerce_float(value: Any, *, default: float) -> float:
        try:
            return float(value)
        except Exception:
            return float(default)

    @staticmethod
    def _clean_text_value(value: Any) -> str:
        token = str(value or "").strip().strip(".,:;!?")
        token = re.sub(r"\s+", " ", token)
        return token
