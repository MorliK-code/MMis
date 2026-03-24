from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from memory_core.processors.state_reducer import StateUpdate


def flatten_governor_profile_snapshot(
    snapshot: dict | None,
    layer_priority: tuple[str, ...] | None = None,
    include_active_facts: bool = True,
) -> dict:
    """Заглушка для обратной совместимости."""
    if not snapshot:
        return {}
    result = {}
    for key in ["traits", "mood", "relation", "active_task", "user_profile"]:
        if key in snapshot:
            result[key] = snapshot[key]
    if "flat_traits" in snapshot:
        result["flat_traits"] = snapshot["flat_traits"]
    elif "traits" in snapshot and isinstance(snapshot["traits"], dict):
        result["flat_traits"] = snapshot["traits"]
    return result


@dataclass(frozen=True)
class IdentityCore:
    character_id: str
    addressing: dict[str, Any] = field(default_factory=dict)
    interaction_style: dict[str, Any] = field(default_factory=dict)
    boundaries: dict[str, Any] = field(default_factory=dict)
    emotional_handling: dict[str, Any] = field(default_factory=dict)
    assistant_trait_baseline: dict[str, float] = field(default_factory=dict)
    updated_at: str = ""
    debug: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "character_id": str(self.character_id or ""),
            "addressing": dict(self.addressing or {}),
            "interaction_style": dict(self.interaction_style or {}),
            "boundaries": dict(self.boundaries or {}),
            "emotional_handling": dict(self.emotional_handling or {}),
            "assistant_trait_baseline": dict(self.assistant_trait_baseline or {}),
            "updated_at": str(self.updated_at or ""),
            "debug": dict(self.debug or {}),
        }


class IdentityCoreBuilder:
    """Build a stable identity core from stored preferences plus active profile fallback."""

    def build(
        self,
        *,
        character_id: str,
        active_profile_snapshot: dict[str, Any] | None,
        stored_identity_core: dict[str, Any] | None,
        memory_identity_core_snapshot: dict[str, Any] | None = None,
    ) -> IdentityCore:
        profile = self._flatten_profile_snapshot(active_profile_snapshot)
        runtime_stored = dict(stored_identity_core or {})
        memory_adapted = self._adapt_memory_identity_core_snapshot(memory_identity_core_snapshot)
        stored = self._merge_identity_core_layers(
            runtime_stored=runtime_stored,
            memory_stored=memory_adapted,
        )
        stored_addressing = self._normalize_addressing(stored.get("addressing"))
        profile_addressing = self._profile_addressing(profile)
        has_stored_addressing = self._has_meaningful_addressing(stored_addressing)
        stored_interaction_style = self._normalize_interaction_style(stored.get("interaction_style"))
        profile_interaction_style = self._profile_interaction_style(profile)
        stored_boundaries = self._normalize_boundaries(stored.get("boundaries"))
        stored_emotional_handling = self._normalize_emotional_handling(stored.get("emotional_handling"))
        stored_assistant_trait_baseline = self._normalize_assistant_trait_baseline(stored.get("assistant_trait_baseline"))
        profile_assistant_trait_baseline = self._profile_assistant_trait_baseline(profile)

        addressing = {
            "canonical_name": "",
            "allowed_forms": [],
            "forbidden_forms": [],
            "use_name_by_default": False,
            "allow_diminutives": False,
            "updated_at": "",
        }
        sources = {
            "canonical_name": "default",
            "allowed_forms": "default",
            "forbidden_forms": "default",
            "use_name_by_default": "default",
            "allow_diminutives": "default",
        }
        interaction_style: dict[str, Any] = {}
        interaction_sources = {
            "prefers_directness": "default",
            "prefers_short_answers": "default",
            "prefers_examples_on_user_code": "default",
            "allows_light_teasing": "default",
            "technical_collaboration_style": "default",
        }
        boundaries = dict(stored_boundaries or {})
        boundary_sources = {
            "avoid_overloaded_intros": ("identity_core" if "avoid_overloaded_intros" in boundaries else "default"),
            "avoid_baby_talk": ("identity_core" if "avoid_baby_talk" in boundaries else "default"),
            "avoid_overformal_tone": ("identity_core" if "avoid_overformal_tone" in boundaries else "default"),
            "do_not_invent_user_facts": ("identity_core" if "do_not_invent_user_facts" in boundaries else "default"),
        }
        emotional_handling = dict(stored_emotional_handling or {})
        emotional_sources = {
            "deescalate_on_irritation": (
                "identity_core" if "deescalate_on_irritation" in emotional_handling else "default"
            ),
            "treat_short_replies_as_low_bandwidth": (
                "identity_core" if "treat_short_replies_as_low_bandwidth" in emotional_handling else "default"
            ),
            "warmth_upshift_on_user_distress": (
                "identity_core" if "warmth_upshift_on_user_distress" in emotional_handling else "default"
            ),
            "playfulness_downshift_on_user_distress": (
                "identity_core" if "playfulness_downshift_on_user_distress" in emotional_handling else "default"
            ),
        }
        assistant_trait_baseline: dict[str, float] = {}
        assistant_trait_sources = {
            "warmth_baseline": "default",
            "directness_baseline": "default",
            "empathy_floor": "default",
            "professionalism_floor": "default",
            "sarcasm_ceiling": "default",
        }

        if str(stored_addressing.get("canonical_name") or "").strip():
            addressing["canonical_name"] = str(stored_addressing.get("canonical_name") or "").strip()
            sources["canonical_name"] = "identity_core"
        elif str(profile_addressing.get("canonical_name") or "").strip():
            addressing["canonical_name"] = str(profile_addressing.get("canonical_name") or "").strip()
            sources["canonical_name"] = "active_profile"

        stored_allowed = self._to_clean_list(stored_addressing.get("allowed_forms"))
        profile_allowed = self._to_clean_list(profile_addressing.get("allowed_forms"))
        if stored_allowed:
            addressing["allowed_forms"] = list(stored_allowed)
            sources["allowed_forms"] = "identity_core"
        elif profile_allowed:
            addressing["allowed_forms"] = list(profile_allowed)
            sources["allowed_forms"] = "active_profile"

        stored_forbidden = self._to_clean_list(stored_addressing.get("forbidden_forms"))
        profile_forbidden = self._to_clean_list(profile_addressing.get("forbidden_forms"))
        if stored_forbidden:
            addressing["forbidden_forms"] = list(stored_forbidden)
            sources["forbidden_forms"] = "identity_core"
        elif profile_forbidden:
            addressing["forbidden_forms"] = list(profile_forbidden)
            sources["forbidden_forms"] = "active_profile"

        if has_stored_addressing:
            addressing["use_name_by_default"] = bool(stored_addressing.get("use_name_by_default", False))
            addressing["allow_diminutives"] = bool(stored_addressing.get("allow_diminutives", False))
            sources["use_name_by_default"] = "identity_core"
            sources["allow_diminutives"] = "identity_core"
        else:
            if "use_name_by_default" in profile_addressing:
                addressing["use_name_by_default"] = bool(profile_addressing.get("use_name_by_default", False))
                sources["use_name_by_default"] = "active_profile"
            if "allow_diminutives" in profile_addressing:
                addressing["allow_diminutives"] = bool(profile_addressing.get("allow_diminutives", False))
                sources["allow_diminutives"] = "active_profile"

        canonical = str(addressing.get("canonical_name") or "").strip()
        if canonical:
            allowed = self._to_clean_list(addressing.get("allowed_forms"))
            if canonical.casefold() not in {str(x).casefold() for x in allowed}:
                addressing["allowed_forms"] = [canonical, *allowed]
            addressing["forbidden_forms"] = [
                x for x in self._to_clean_list(addressing.get("forbidden_forms"))
                if str(x).casefold() != canonical.casefold()
            ]

        addressing["updated_at"] = str(
            stored_addressing.get("updated_at")
            or stored.get("updated_at")
            or ""
        ).strip()

        for key in (
            "prefers_directness",
            "prefers_short_answers",
            "prefers_examples_on_user_code",
            "allows_light_teasing",
            "technical_collaboration_style",
        ):
            if key in stored_interaction_style:
                interaction_style[key] = stored_interaction_style.get(key)
                interaction_sources[key] = "identity_core"
            elif key in profile_interaction_style:
                interaction_style[key] = profile_interaction_style.get(key)
                interaction_sources[key] = "active_profile"

        for key in (
            "warmth_baseline",
            "directness_baseline",
            "empathy_floor",
            "professionalism_floor",
            "sarcasm_ceiling",
        ):
            if key in stored_assistant_trait_baseline:
                assistant_trait_baseline[key] = float(stored_assistant_trait_baseline.get(key))
                assistant_trait_sources[key] = "identity_core"
            elif key in profile_assistant_trait_baseline:
                assistant_trait_baseline[key] = float(profile_assistant_trait_baseline.get(key))
                assistant_trait_sources[key] = "active_profile"

        return IdentityCore(
            character_id=str(character_id or "assistant").strip() or "assistant",
            addressing=addressing,
            interaction_style=interaction_style,
            boundaries=boundaries,
            emotional_handling=emotional_handling,
            assistant_trait_baseline=assistant_trait_baseline,
            updated_at=str(stored.get("updated_at") or addressing.get("updated_at") or ""),
            debug={
                "sources": {
                    "addressing": sources,
                    "interaction_style": interaction_sources,
                    "boundaries": boundary_sources,
                    "emotional_handling": emotional_sources,
                    "assistant_trait_baseline": assistant_trait_sources,
                },
                "input_layers": {
                    "runtime_identity_core_keys": sorted(runtime_stored.keys()),
                    "memory_identity_core_keys": sorted(memory_adapted.keys()),
                    "resolved_identity_core_keys": sorted(stored.keys()),
                    "priority": [
                        "memory_identity_core",
                        "runtime_identity_core",
                        "active_profile",
                    ],
                },
                "has_stored_addressing": bool(has_stored_addressing),
                "stored_keys": sorted(stored.keys()),
                "profile_keys": sorted(profile.keys()),
            },
        )

    @staticmethod
    def _merge_identity_core_layers(
        *,
        runtime_stored: dict[str, Any] | None,
        memory_stored: dict[str, Any] | None,
    ) -> dict[str, Any]:
        runtime_row = dict(runtime_stored or {})
        memory_row = dict(memory_stored or {})
        out = dict(runtime_row)
        for key in (
            "addressing",
            "interaction_style",
            "boundaries",
            "emotional_handling",
            "assistant_trait_baseline",
        ):
            merged = dict(runtime_row.get(key) or {})
            merged.update(dict(memory_row.get(key) or {}))
            if merged:
                out[key] = merged
        updated_at = str(memory_row.get("updated_at") or runtime_row.get("updated_at") or "").strip()
        if updated_at:
            out["updated_at"] = updated_at
        return out

    @staticmethod
    def _adapt_memory_identity_core_snapshot(value: dict[str, Any] | None) -> dict[str, Any]:
        """
        Адаптирует memory identity snapshot в prompt-facing формат.

        Поддерживает два формата:
        1. Nested schema (addressing, interaction_style, boundaries, emotional_handling, assistant_trait_baseline)
        2. Flat schema от memory_core.identity.IdentityProfile (user_name, assistant_style, etc.)
        """
        row = dict(value or {})
        if not row:
            return {}

        # 1. Если это nested schema — используем текущую логику
        if any(key in row for key in (
            "addressing",
            "interaction_style",
            "boundaries",
            "emotional_handling",
            "emotional_rules",
            "assistant_trait_baseline",
        )):
            return IdentityCoreBuilder._adapt_nested_memory_identity_core_snapshot(row)

        # 2. Если это flat schema от memory_core.identity.IdentityProfile
        out: dict[str, Any] = {}

        # Адресация
        user_name = str(row.get("user_name") or "").strip()
        user_address_form = str(row.get("user_address_form") or "").strip().lower()
        
        if user_name or user_address_form:
            addressing: dict[str, Any] = {
                "canonical_name": user_name,
                "allowed_forms": [user_name] if user_name else [],
                "forbidden_forms": [],
                "use_name_by_default": bool(user_name),
                "allow_diminutives": bool(user_address_form == "ты"),
            }
            out["addressing"] = addressing

        # Interaction style из assistant_style и assistant_tone
        assistant_style = str(row.get("assistant_style") or "").strip().lower()
        assistant_tone = str(row.get("assistant_tone") or "").strip().lower()
        
        interaction_style: dict[str, Any] = {}
        
        if assistant_style == "concise":
            interaction_style["prefers_short_answers"] = 1.0
        elif assistant_style == "detailed":
            interaction_style["prefers_short_answers"] = 0.0
        
        if assistant_tone == "warm":
            interaction_style["prefers_directness"] = 0.72
        elif assistant_tone == "professional":
            interaction_style["prefers_directness"] = 0.82
        
        if interaction_style:
            out["interaction_style"] = interaction_style

        # Boundaries из never_do
        never_do = IdentityCoreBuilder._to_clean_list(row.get("never_do"))
        
        if never_do:
            boundaries: dict[str, Any] = {}
            joined = " ".join(never_do).lower()
            
            if "не выдум" in joined or "не придумы" in joined or "not invent" in joined:
                boundaries["do_not_invent_user_facts"] = True
            if "не сюсюк" in joined or "baby" in joined:
                boundaries["avoid_baby_talk"] = True
            if "не перегруж" in joined or "overload" in joined:
                boundaries["avoid_overloaded_intros"] = True
            
            if boundaries:
                out["boundaries"] = boundaries

        # Emotional handling из always_do
        always_do = IdentityCoreBuilder._to_clean_list(row.get("always_do"))
        
        if always_do:
            emotional_handling: dict[str, Any] = {}
            joined = " ".join(always_do).lower()
            
            if "спокой" in joined or "деэскал" in joined or "deescalat" in joined:
                emotional_handling["deescalate_on_irritation"] = True
            if "поддерж" in joined or "тепл" in joined or "warm" in joined:
                emotional_handling["warmth_upshift_on_user_distress"] = 0.2
            
            if emotional_handling:
                out["emotional_handling"] = emotional_handling

        # Assistant trait baseline из assistant_tone и assistant_style
        trait_baseline: dict[str, float] = {}
        
        if assistant_tone == "warm":
            trait_baseline["warmth_baseline"] = 0.78
        if assistant_tone == "professional":
            trait_baseline["professionalism_floor"] = 0.78
        if assistant_style == "friendly":
            trait_baseline["directness_baseline"] = 0.68
        elif assistant_style == "formal":
            trait_baseline["directness_baseline"] = 0.82
        
        if trait_baseline:
            out["assistant_trait_baseline"] = trait_baseline

        # Updated_at
        updated_at = str(row.get("updated_at") or "").strip()
        if updated_at:
            out["updated_at"] = updated_at

        return out

    @staticmethod
    def _adapt_nested_memory_identity_core_snapshot(row: dict[str, Any]) -> dict[str, Any]:
        """Адаптирует nested schema (старый формат)."""
        out: dict[str, Any] = {}

        addressing_in = dict(row.get("addressing") or {})
        if addressing_in:
            addressing: dict[str, Any] = {}
            if str(addressing_in.get("canonical_name") or "").strip():
                addressing["canonical_name"] = str(addressing_in.get("canonical_name") or "").strip()
            allowed_forms = IdentityCoreBuilder._to_clean_list(addressing_in.get("allowed_forms"))
            if allowed_forms:
                addressing["allowed_forms"] = allowed_forms
            forbidden_forms = IdentityCoreBuilder._to_clean_list(addressing_in.get("forbidden_forms"))
            if forbidden_forms:
                addressing["forbidden_forms"] = forbidden_forms
            if "use_name_by_default" in addressing_in:
                addressing["use_name_by_default"] = bool(addressing_in.get("use_name_by_default"))
            if "allow_diminutives" in addressing_in:
                addressing["allow_diminutives"] = bool(addressing_in.get("allow_diminutives"))
            if str(addressing_in.get("updated_at") or "").strip():
                addressing["updated_at"] = str(addressing_in.get("updated_at") or "").strip()
            if addressing:
                out["addressing"] = addressing

        interaction_in = dict(row.get("interaction_style") or {})
        if interaction_in:
            interaction: dict[str, Any] = {}
            if "prefers_directness" in interaction_in:
                interaction["prefers_directness"] = IdentityCoreBuilder._clamp01(
                    IdentityCoreBuilder._to_float(interaction_in.get("prefers_directness"), 0.0)
                )
            if "prefers_short_answers" in interaction_in:
                interaction["prefers_short_answers"] = IdentityCoreBuilder._clamp01(
                    IdentityCoreBuilder._to_float(interaction_in.get("prefers_short_answers"), 0.0)
                )
            if "allow_light_teasing" in interaction_in or "allows_light_teasing" in interaction_in:
                interaction["allows_light_teasing"] = bool(
                    interaction_in.get("allow_light_teasing", interaction_in.get("allows_light_teasing"))
                )
            if "prefers_examples_on_user_code" in interaction_in:
                interaction["prefers_examples_on_user_code"] = bool(
                    interaction_in.get("prefers_examples_on_user_code")
                )
            style = str(interaction_in.get("technical_collaboration_style") or "").strip().lower()
            if style in {"low", "medium", "high"}:
                interaction["technical_collaboration_style"] = style
            if interaction:
                out["interaction_style"] = interaction

        boundaries_in = dict(row.get("boundaries") or {})
        if boundaries_in:
            boundaries: dict[str, Any] = {}
            if "avoid_baby_tone" in boundaries_in or "avoid_baby_talk" in boundaries_in:
                boundaries["avoid_baby_talk"] = bool(
                    boundaries_in.get("avoid_baby_tone", boundaries_in.get("avoid_baby_talk"))
                )
            if "avoid_repeating_question" in boundaries_in:
                boundaries["avoid_repeating_question"] = bool(boundaries_in.get("avoid_repeating_question"))
            if "avoid_inventing_user_facts" in boundaries_in or "do_not_invent_user_facts" in boundaries_in:
                boundaries["do_not_invent_user_facts"] = bool(
                    boundaries_in.get(
                        "avoid_inventing_user_facts",
                        boundaries_in.get("do_not_invent_user_facts"),
                    )
                )
            if boundaries:
                out["boundaries"] = boundaries

        emotional_in = dict(row.get("emotional_handling") or row.get("emotional_rules") or {})
        if emotional_in:
            emotional: dict[str, Any] = {}
            if "deescalate_on_irritation" in emotional_in:
                emotional["deescalate_on_irritation"] = bool(emotional_in.get("deescalate_on_irritation"))
            if "treat_short_replies_as_low_bandwidth" in emotional_in:
                emotional["treat_short_replies_as_low_bandwidth"] = bool(
                    emotional_in.get("treat_short_replies_as_low_bandwidth")
                )
            if "frustration_softening" in emotional_in:
                frustration_softening = IdentityCoreBuilder._clamp01(
                    IdentityCoreBuilder._to_float(emotional_in.get("frustration_softening"), 0.0)
                )
                if frustration_softening > 0.0:
                    emotional["deescalate_on_irritation"] = True
                    emotional["warmth_upshift_on_user_distress"] = max(
                        frustration_softening,
                        IdentityCoreBuilder._to_float(
                            emotional.get("warmth_upshift_on_user_distress"),
                            0.0,
                        ),
                    )
            if "warmth_upshift_on_user_distress" in emotional_in:
                emotional["warmth_upshift_on_user_distress"] = IdentityCoreBuilder._clamp01(
                    IdentityCoreBuilder._to_float(emotional_in.get("warmth_upshift_on_user_distress"), 0.0)
                )
            if (
                "reduce_teasing_when_user_irritated" in emotional_in
                or "playfulness_downshift_on_user_distress" in emotional_in
            ):
                emotional["playfulness_downshift_on_user_distress"] = IdentityCoreBuilder._clamp01(
                    IdentityCoreBuilder._to_float(
                        emotional_in.get(
                            "reduce_teasing_when_user_irritated",
                            emotional_in.get("playfulness_downshift_on_user_distress"),
                        ),
                        0.0,
                    )
                )
            if emotional:
                out["emotional_handling"] = emotional

        traits_in = dict(row.get("assistant_trait_baseline") or {})
        if traits_in:
            baseline: dict[str, float] = {}
            for key in (
                "warmth_baseline",
                "directness_baseline",
                "empathy_floor",
                "professionalism_floor",
                "sarcasm_ceiling",
            ):
                if key in traits_in:
                    baseline[key] = IdentityCoreBuilder._clamp01(
                        IdentityCoreBuilder._to_float(traits_in.get(key), 0.0)
                    )
            if baseline:
                out["assistant_trait_baseline"] = baseline

        updated_at = str(row.get("updated_at") or "").strip()
        if updated_at:
            out["updated_at"] = updated_at

        return out

    @staticmethod
    def _flatten_profile_snapshot(snapshot: dict[str, Any] | None) -> dict[str, Any]:
        row = dict(snapshot or {})
        has_layered_snapshot = any(
            isinstance(row.get(layer_name), dict) and bool(dict(row.get(layer_name) or {}))
            for layer_name in ("persistent_traits", "volatile_preferences", "session_preferences")
        )
        return flatten_governor_profile_snapshot(
            row,
            layer_priority=("persistent_traits",),
            include_active_facts=not has_layered_snapshot,
        )

    @staticmethod
    def _profile_addressing(profile: dict[str, Any]) -> dict[str, Any]:
        return {
            "canonical_name": str(profile.get("identity_name") or "").strip(),
            "allowed_forms": IdentityCoreBuilder._to_clean_list(profile.get("name_allowed_forms")),
            "forbidden_forms": IdentityCoreBuilder._to_clean_list(profile.get("name_forbidden_forms")),
            "use_name_by_default": bool(profile.get("use_name_by_default", False)),
            "allow_diminutives": bool(profile.get("allow_diminutives", False)),
        }

    @staticmethod
    def _normalize_addressing(value: Any) -> dict[str, Any]:
        row = dict(value or {})
        out = {
            "canonical_name": str(row.get("canonical_name") or "").strip(),
            "allowed_forms": IdentityCoreBuilder._to_clean_list(row.get("allowed_forms")),
            "forbidden_forms": IdentityCoreBuilder._to_clean_list(row.get("forbidden_forms")),
            "use_name_by_default": bool(row.get("use_name_by_default", False)),
            "allow_diminutives": bool(row.get("allow_diminutives", False)),
            "updated_at": str(row.get("updated_at") or "").strip(),
        }
        return out

    @staticmethod
    def _profile_interaction_style(profile: dict[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        if "assistant_directness" in profile:
            out["prefers_directness"] = IdentityCoreBuilder._clamp01(
                IdentityCoreBuilder._to_float(profile.get("assistant_directness"), 0.74)
            )
        if "preferred_answer_brevity" in profile:
            out["prefers_short_answers"] = IdentityCoreBuilder._clamp01(
                IdentityCoreBuilder._to_float(profile.get("preferred_answer_brevity"), 0.0)
            )
        elif "assistant_verbosity" in profile:
            verbosity = IdentityCoreBuilder._clamp01(
                IdentityCoreBuilder._to_float(profile.get("assistant_verbosity"), 0.55)
            )
            out["prefers_short_answers"] = IdentityCoreBuilder._clamp01(1.0 - verbosity)
        if "prefers_examples_on_user_code" in profile:
            out["prefers_examples_on_user_code"] = bool(profile.get("prefers_examples_on_user_code"))
        if "assistant_teasing" in profile:
            out["allows_light_teasing"] = bool(
                IdentityCoreBuilder._to_float(profile.get("assistant_teasing"), 0.0) >= 0.22
            )
        elif "assistant_sarcasm" in profile:
            out["allows_light_teasing"] = bool(
                IdentityCoreBuilder._to_float(profile.get("assistant_sarcasm"), 0.0) >= 0.22
            )
        if "relation_technical_collaboration" in profile:
            technical = IdentityCoreBuilder._clamp01(
                IdentityCoreBuilder._to_float(profile.get("relation_technical_collaboration"), 0.0)
            )
            if technical >= 0.78:
                out["technical_collaboration_style"] = "high"
            elif technical >= 0.46:
                out["technical_collaboration_style"] = "medium"
            else:
                out["technical_collaboration_style"] = "low"
        return out

    @staticmethod
    def _normalize_interaction_style(value: Any) -> dict[str, Any]:
        row = dict(value or {})
        out: dict[str, Any] = {}
        if "prefers_directness" in row:
            out["prefers_directness"] = IdentityCoreBuilder._clamp01(
                IdentityCoreBuilder._to_float(row.get("prefers_directness"), 0.0)
            )
        if "prefers_short_answers" in row:
            out["prefers_short_answers"] = IdentityCoreBuilder._clamp01(
                IdentityCoreBuilder._to_float(row.get("prefers_short_answers"), 0.0)
            )
        if "prefers_examples_on_user_code" in row:
            out["prefers_examples_on_user_code"] = bool(row.get("prefers_examples_on_user_code"))
        if "allow_light_teasing" in row:
            out["allows_light_teasing"] = bool(row.get("allow_light_teasing"))
        if "allows_light_teasing" in row:
            out["allows_light_teasing"] = bool(row.get("allows_light_teasing"))
        style = str(row.get("technical_collaboration_style") or "").strip().lower()
        if style in {"low", "medium", "high"}:
            out["technical_collaboration_style"] = style
        return out

    @staticmethod
    def _normalize_boundaries(value: Any) -> dict[str, Any]:
        row = dict(value or {})
        out: dict[str, Any] = {}
        for key in (
            "avoid_overloaded_intros",
            "avoid_baby_talk",
            "avoid_repeating_question",
            "avoid_overformal_tone",
            "do_not_invent_user_facts",
        ):
            if key in row:
                out[key] = bool(row.get(key))
        return out

    @staticmethod
    def _normalize_emotional_handling(value: Any) -> dict[str, Any]:
        row = dict(value or {})
        out: dict[str, Any] = {}
        if "frustration_softening" in row:
            value = IdentityCoreBuilder._clamp01(
                IdentityCoreBuilder._to_float(row.get("frustration_softening"), 0.0)
            )
            if value > 0.0:
                out["deescalate_on_irritation"] = True
                out["warmth_upshift_on_user_distress"] = value
        if "reduce_teasing_when_user_irritated" in row:
            out["playfulness_downshift_on_user_distress"] = IdentityCoreBuilder._clamp01(
                IdentityCoreBuilder._to_float(row.get("reduce_teasing_when_user_irritated"), 0.0)
            )
        if "deescalate_on_irritation" in row:
            out["deescalate_on_irritation"] = bool(row.get("deescalate_on_irritation"))
        if "treat_short_replies_as_low_bandwidth" in row:
            out["treat_short_replies_as_low_bandwidth"] = bool(row.get("treat_short_replies_as_low_bandwidth"))
        if "warmth_upshift_on_user_distress" in row:
            out["warmth_upshift_on_user_distress"] = IdentityCoreBuilder._clamp01(
                IdentityCoreBuilder._to_float(row.get("warmth_upshift_on_user_distress"), 0.0)
            )
        if "playfulness_downshift_on_user_distress" in row:
            out["playfulness_downshift_on_user_distress"] = IdentityCoreBuilder._clamp01(
                IdentityCoreBuilder._to_float(row.get("playfulness_downshift_on_user_distress"), 0.0)
            )
        return out

    @staticmethod
    def _profile_assistant_trait_baseline(profile: dict[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        if "assistant_warmth" in profile:
            out["warmth_baseline"] = IdentityCoreBuilder._clamp01(
                IdentityCoreBuilder._to_float(profile.get("assistant_warmth"), 0.62)
            )
        if "assistant_directness" in profile:
            out["directness_baseline"] = IdentityCoreBuilder._clamp01(
                IdentityCoreBuilder._to_float(profile.get("assistant_directness"), 0.74)
            )
        if "assistant_empathy" in profile:
            out["empathy_floor"] = IdentityCoreBuilder._clamp01(
                IdentityCoreBuilder._to_float(profile.get("assistant_empathy"), 0.66)
            )
        if "assistant_professionalism" in profile:
            out["professionalism_floor"] = IdentityCoreBuilder._clamp01(
                IdentityCoreBuilder._to_float(profile.get("assistant_professionalism"), 0.78)
            )
        if "assistant_sarcasm" in profile:
            out["sarcasm_ceiling"] = IdentityCoreBuilder._clamp01(
                IdentityCoreBuilder._to_float(profile.get("assistant_sarcasm"), 0.22)
            )
        return out

    @staticmethod
    def _normalize_assistant_trait_baseline(value: Any) -> dict[str, float]:
        row = dict(value or {})
        out: dict[str, float] = {}
        for key in (
            "warmth_baseline",
            "directness_baseline",
            "empathy_floor",
            "professionalism_floor",
            "sarcasm_ceiling",
        ):
            if key in row:
                out[key] = IdentityCoreBuilder._clamp01(
                    IdentityCoreBuilder._to_float(row.get(key), 0.0)
                )
        return out

    @staticmethod
    def _has_meaningful_addressing(value: dict[str, Any] | None) -> bool:
        row = dict(value or {})
        return bool(
            str(row.get("canonical_name") or "").strip()
            or list(row.get("allowed_forms") or [])
            or list(row.get("forbidden_forms") or [])
            or str(row.get("updated_at") or "").strip()
        )

    @staticmethod
    def _to_clean_list(value: Any) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for row in list(value or []):
            item = str(row or "").strip()
            key = item.casefold()
            if not item or key in seen:
                continue
            seen.add(key)
            out.append(item)
        return out

    @staticmethod
    def _to_float(value: Any, default: float) -> float:
        try:
            return float(value)
        except Exception:
            return float(default)

    @staticmethod
    def _clamp01(value: float) -> float:
        return max(0.0, min(1.0, float(value)))
