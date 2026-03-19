from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class PersonaSnapshot:
    character_id: str
    mood: str
    relation_state: dict[str, Any] = field(default_factory=dict)
    user_addressing: dict[str, Any] = field(default_factory=dict)
    stable_traits: dict[str, float] = field(default_factory=dict)
    response_bias: dict[str, float] = field(default_factory=dict)
    boundaries: dict[str, Any] = field(default_factory=dict)
    emotional_handling: dict[str, Any] = field(default_factory=dict)
    user_profile_hints: dict[str, Any] = field(default_factory=dict)
    active_mode: str = "chatting"
    debug: dict[str, Any] = field(default_factory=dict)


class PersonaSnapshotBuilder:
    """Build persona from compact profile signals, not from raw memory snippets."""

    _FLAT_PROFILE_HINT_MAP = {
        "identity_name": "user_name",
        "preferred_editor": "preferred_editor",
        "preferred_language": "preferred_language",
        "project_name": "project_name",
        "environment_os": "os",
        "environment_runtime_python": "python",
        "environment_gpu_model": "gpu_model",
        "environment_ram_gb": "ram_gb",
        "environment_memory_gb": "ram_gb",
    }

    def build(
        self,
        *,
        character_id: str,
        active_profile_snapshot: dict[str, Any] | None,
        identity_core_snapshot: dict[str, Any] | None = None,
        memory_context: dict[str, Any] | None,
        state: dict[str, Any] | None,
        meta: dict[str, Any] | None,
    ) -> PersonaSnapshot:
        profile = self._flatten_profile_snapshot(active_profile_snapshot)
        memory = dict(memory_context or {})
        state_map = dict(state or {})
        meta_map = dict(meta or {})
        persistent_profile_keys = sorted(profile.keys())
        character_trait_defaults = self._normalize_trait_defaults(
            state_map.get("character_trait_defaults") or meta_map.get("character_trait_defaults") or {}
        )

        mood_source = "default"
        mood_raw = meta_map.get("emotion")
        if self._is_empty(mood_raw):
            mood_raw = meta_map.get("mood")
            if not self._is_empty(mood_raw):
                mood_source = "meta.mood"
        else:
            mood_source = "meta.emotion"
        if self._is_empty(mood_raw):
            mood_raw = state_map.get("mood")
            if not self._is_empty(mood_raw):
                mood_source = "state.mood"
        mood = str(mood_raw or "neutral").strip().lower() or "neutral"

        active_mode_source = "default"
        active_mode_raw = meta_map.get("active_mode")
        if not self._is_empty(active_mode_raw):
            active_mode_source = "meta.active_mode"
        else:
            active_mode_raw = state_map.get("active_mode")
            if not self._is_empty(active_mode_raw):
                active_mode_source = "state.active_mode"
        active_mode = str(active_mode_raw or "chatting").strip().lower() or "chatting"

        user_profile_hints = {
            "user_name": profile.get("identity_name"),
            "preferred_editor": profile.get("preferred_editor"),
            "preferred_language": profile.get("preferred_language"),
            "project_name": profile.get("project_name"),
            "os": profile.get("environment_os"),
            "python": profile.get("environment_runtime_python"),
            "gpu_model": profile.get("environment_gpu_model"),
            "ram_gb": profile.get("environment_ram_gb") or profile.get("environment_memory_gb"),
        }
        identity_core = dict(
            identity_core_snapshot
            or state_map.get("identity_core")
            or meta_map.get("identity_core")
            or {}
        )
        identity_core_addressing = dict(identity_core.get("addressing") or {})
        identity_core_interaction = dict(identity_core.get("interaction_style") or {})
        identity_core_boundaries = dict(identity_core.get("boundaries") or {})
        identity_core_emotional_handling = dict(identity_core.get("emotional_handling") or {})
        identity_core_assistant_trait_baseline = dict(identity_core.get("assistant_trait_baseline") or {})
        identity_core_name = str(identity_core_addressing.get("canonical_name") or "").strip()
        if identity_core_name:
            user_profile_hints["user_name"] = identity_core_name
        if "prefers_directness" in identity_core_interaction:
            user_profile_hints["preferred_directness"] = self._clamp01(
                self._to_float(identity_core_interaction.get("prefers_directness"), 0.74)
            )
        if "prefers_short_answers" in identity_core_interaction:
            user_profile_hints["preferred_answer_brevity"] = self._clamp01(
                self._to_float(identity_core_interaction.get("prefers_short_answers"), 0.0)
            )
        if "technical_collaboration_style" in identity_core_interaction:
            user_profile_hints["technical_collaboration_style"] = str(
                identity_core_interaction.get("technical_collaboration_style") or ""
            ).strip().lower()
        user_profile_hints = {key: value for key, value in user_profile_hints.items() if not self._is_empty(value)}
        profile_hint_fields = sorted(user_profile_hints.keys())

        relation_state = {
            "familiarity": self._to_float(profile.get("relation_familiarity"), 0.65),
            "trust": self._to_float(profile.get("relation_trust"), 0.72),
            "technical_collaboration": self._to_float(profile.get("relation_technical_collaboration"), 0.85),
        }
        relation_source = {
            "familiarity": ("persistent" if not self._is_empty(profile.get("relation_familiarity")) else "default"),
            "trust": ("persistent" if not self._is_empty(profile.get("relation_trust")) else "default"),
            "technical_collaboration": (
                "persistent" if not self._is_empty(profile.get("relation_technical_collaboration")) else "default"
            ),
        }
        relation_fields_from_persistent = sorted(
            [
                key for key in (
                    "relation_familiarity",
                    "relation_trust",
                    "relation_technical_collaboration",
                )
                if not self._is_empty(profile.get(key))
            ]
        )
        relation_fields_from_identity_core: list[str] = []
        if "technical_collaboration_style" in identity_core_interaction:
            style = str(identity_core_interaction.get("technical_collaboration_style") or "").strip().lower()
            style_to_value = {
                "low": 0.35,
                "medium": 0.66,
                "high": 0.88,
            }
            if style in style_to_value:
                relation_state["technical_collaboration"] = style_to_value[style]
                relation_source["technical_collaboration"] = "identity_core"
                relation_fields_from_identity_core.append("technical_collaboration")

        user_addressing = {
            "canonical_name": str(profile.get("identity_name") or ""),
            "allowed_forms": self._to_clean_list(profile.get("name_allowed_forms")),
            "forbidden_forms": self._to_clean_list(profile.get("name_forbidden_forms")),
            "use_name_by_default": bool(profile.get("use_name_by_default", False)),
            "allow_diminutives": bool(profile.get("allow_diminutives", False)),
        }
        addressing_source = {
            "canonical_name": ("persistent" if str(profile.get("identity_name") or "").strip() else "default"),
            "allowed_forms": ("persistent" if self._to_clean_list(profile.get("name_allowed_forms")) else "default"),
            "forbidden_forms": ("persistent" if self._to_clean_list(profile.get("name_forbidden_forms")) else "default"),
            "use_name_by_default": ("persistent" if "use_name_by_default" in profile else "default"),
            "allow_diminutives": ("persistent" if "allow_diminutives" in profile else "default"),
        }
        user_addressing_from_persistent = sorted(
            [
                key for key in (
                    "canonical_name",
                    "allowed_forms",
                    "forbidden_forms",
                    "use_name_by_default",
                    "allow_diminutives",
                )
                if not self._is_empty(user_addressing.get(key))
            ]
        )
        user_addressing_from_identity_core: list[str] = []
        if identity_core_addressing:
            if str(identity_core_addressing.get("canonical_name") or "").strip():
                user_addressing["canonical_name"] = str(identity_core_addressing.get("canonical_name") or "").strip()
                user_addressing_from_identity_core.append("canonical_name")
                addressing_source["canonical_name"] = "identity_core"
            if self._to_clean_list(identity_core_addressing.get("allowed_forms")):
                user_addressing["allowed_forms"] = self._to_clean_list(identity_core_addressing.get("allowed_forms"))
                user_addressing_from_identity_core.append("allowed_forms")
                addressing_source["allowed_forms"] = "identity_core"
            if self._to_clean_list(identity_core_addressing.get("forbidden_forms")):
                user_addressing["forbidden_forms"] = self._to_clean_list(identity_core_addressing.get("forbidden_forms"))
                user_addressing_from_identity_core.append("forbidden_forms")
                addressing_source["forbidden_forms"] = "identity_core"
            if "use_name_by_default" in identity_core_addressing:
                user_addressing["use_name_by_default"] = bool(identity_core_addressing.get("use_name_by_default"))
                user_addressing_from_identity_core.append("use_name_by_default")
                addressing_source["use_name_by_default"] = "identity_core"
            if "allow_diminutives" in identity_core_addressing:
                user_addressing["allow_diminutives"] = bool(identity_core_addressing.get("allow_diminutives"))
                user_addressing_from_identity_core.append("allow_diminutives")
                addressing_source["allow_diminutives"] = "identity_core"
        user_addressing_from_runtime: list[str] = []
        if state_map.get("user_addressing"):
            runtime_addressing = dict(state_map.get("user_addressing") or {})
            if str(runtime_addressing.get("canonical_name") or "").strip():
                user_addressing["canonical_name"] = str(runtime_addressing.get("canonical_name") or "").strip()
                user_addressing_from_runtime.append("canonical_name")
                addressing_source["canonical_name"] = "runtime_override"
            if self._to_clean_list(runtime_addressing.get("allowed_forms")):
                user_addressing["allowed_forms"] = self._to_clean_list(runtime_addressing.get("allowed_forms"))
                user_addressing_from_runtime.append("allowed_forms")
                addressing_source["allowed_forms"] = "runtime_override"
            if self._to_clean_list(runtime_addressing.get("forbidden_forms")):
                user_addressing["forbidden_forms"] = self._to_clean_list(runtime_addressing.get("forbidden_forms"))
                user_addressing_from_runtime.append("forbidden_forms")
                addressing_source["forbidden_forms"] = "runtime_override"
            if "use_name_by_default" in runtime_addressing:
                user_addressing["use_name_by_default"] = bool(runtime_addressing.get("use_name_by_default"))
                user_addressing_from_runtime.append("use_name_by_default")
                addressing_source["use_name_by_default"] = "runtime_override"
            if "allow_diminutives" in runtime_addressing:
                user_addressing["allow_diminutives"] = bool(runtime_addressing.get("allow_diminutives"))
                user_addressing_from_runtime.append("allow_diminutives")
                addressing_source["allow_diminutives"] = "runtime_override"

        baseline_traits = {
            "warmth": self._to_float(character_trait_defaults.get("warmth"), 0.68),
            "directness": self._to_float(character_trait_defaults.get("directness"), 0.74),
            "sarcasm": self._to_float(character_trait_defaults.get("sarcasm"), 0.32),
            "empathy": self._to_float(character_trait_defaults.get("empathy"), 0.72),
            "professionalism": self._to_float(character_trait_defaults.get("professionalism"), 0.78),
            "verbosity": self._to_float(character_trait_defaults.get("verbosity"), 0.55),
            "strictness": self._to_float(character_trait_defaults.get("strictness"), 0.52),
            "teasing": self._to_float(character_trait_defaults.get("teasing"), 0.46),
        }
        if not self._is_empty(profile.get("assistant_warmth")):
            baseline_traits["warmth"] = self._to_float(profile.get("assistant_warmth"), baseline_traits["warmth"])
        if not self._is_empty(profile.get("assistant_directness")):
            baseline_traits["directness"] = self._to_float(profile.get("assistant_directness"), baseline_traits["directness"])
        if not self._is_empty(profile.get("assistant_sarcasm")):
            baseline_traits["sarcasm"] = self._to_float(profile.get("assistant_sarcasm"), baseline_traits["sarcasm"])
        if not self._is_empty(profile.get("assistant_empathy")):
            baseline_traits["empathy"] = self._to_float(profile.get("assistant_empathy"), baseline_traits["empathy"])
        if not self._is_empty(profile.get("assistant_professionalism")):
            baseline_traits["professionalism"] = self._to_float(
                profile.get("assistant_professionalism"),
                baseline_traits["professionalism"],
            )
        trait_overrides = {
            "warmth": {
                "source": (
                    "active_profile"
                    if not self._is_empty(profile.get("assistant_warmth"))
                    else ("character_specs_default" if "warmth" in character_trait_defaults else "default")
                ),
                "value": baseline_traits["warmth"],
            },
            "directness": {
                "source": (
                    "active_profile"
                    if not self._is_empty(profile.get("assistant_directness"))
                    else ("character_specs_default" if "directness" in character_trait_defaults else "default")
                ),
                "value": baseline_traits["directness"],
            },
            "sarcasm": {
                "source": (
                    "active_profile"
                    if not self._is_empty(profile.get("assistant_sarcasm"))
                    else ("character_specs_default" if "sarcasm" in character_trait_defaults else "default")
                ),
                "value": baseline_traits["sarcasm"],
            },
            "empathy": {
                "source": (
                    "active_profile"
                    if not self._is_empty(profile.get("assistant_empathy"))
                    else ("character_specs_default" if "empathy" in character_trait_defaults else "default")
                ),
                "value": baseline_traits["empathy"],
            },
            "professionalism": {
                "source": (
                    "active_profile"
                    if not self._is_empty(profile.get("assistant_professionalism"))
                    else ("character_specs_default" if "professionalism" in character_trait_defaults else "default")
                ),
                "value": baseline_traits["professionalism"],
            },
            "verbosity": {
                "source": ("character_specs_default" if "verbosity" in character_trait_defaults else "default"),
                "value": baseline_traits["verbosity"],
            },
            "strictness": {
                "source": ("character_specs_default" if "strictness" in character_trait_defaults else "default"),
                "value": baseline_traits["strictness"],
            },
            "teasing": {
                "source": ("character_specs_default" if "teasing" in character_trait_defaults else "default"),
                "value": baseline_traits["teasing"],
            },
        }
        stable_traits_from_persistent = sorted(
            [
                key for key, profile_key in (
                    ("warmth", "assistant_warmth"),
                    ("directness", "assistant_directness"),
                    ("sarcasm", "assistant_sarcasm"),
                    ("empathy", "assistant_empathy"),
                    ("professionalism", "assistant_professionalism"),
                )
                if not self._is_empty(profile.get(profile_key))
            ]
        )
        stable_traits_from_character_defaults = sorted(
            [
                key for key in (
                    "warmth",
                    "directness",
                    "sarcasm",
                    "empathy",
                    "professionalism",
                    "verbosity",
                    "strictness",
                    "teasing",
                )
                if key in character_trait_defaults
            ]
        )
        stable_traits_from_identity_core: list[str] = []
        assistant_trait_baseline_source = {
            "warmth_baseline": ("identity_core" if "warmth_baseline" in identity_core_assistant_trait_baseline else trait_overrides["warmth"]["source"]),
            "directness_baseline": ("identity_core" if "directness_baseline" in identity_core_assistant_trait_baseline else trait_overrides["directness"]["source"]),
            "empathy_floor": ("identity_core" if "empathy_floor" in identity_core_assistant_trait_baseline else trait_overrides["empathy"]["source"]),
            "professionalism_floor": ("identity_core" if "professionalism_floor" in identity_core_assistant_trait_baseline else trait_overrides["professionalism"]["source"]),
            "sarcasm_ceiling": ("identity_core" if "sarcasm_ceiling" in identity_core_assistant_trait_baseline else trait_overrides["sarcasm"]["source"]),
        }
        if "warmth_baseline" in identity_core_assistant_trait_baseline:
            baseline_traits["warmth"] = self._clamp01(
                self._to_float(identity_core_assistant_trait_baseline.get("warmth_baseline"), baseline_traits["warmth"])
            )
            trait_overrides["warmth"] = {"source": "identity_core_baseline", "value": baseline_traits["warmth"]}
        if "directness_baseline" in identity_core_assistant_trait_baseline:
            baseline_traits["directness"] = self._clamp01(
                self._to_float(identity_core_assistant_trait_baseline.get("directness_baseline"), baseline_traits["directness"])
            )
            trait_overrides["directness"] = {"source": "identity_core_baseline", "value": baseline_traits["directness"]}
        if "empathy_floor" in identity_core_assistant_trait_baseline:
            baseline_traits["empathy"] = max(
                baseline_traits["empathy"],
                self._clamp01(self._to_float(identity_core_assistant_trait_baseline.get("empathy_floor"), baseline_traits["empathy"])),
            )
            trait_overrides["empathy"] = {"source": "identity_core_baseline", "value": baseline_traits["empathy"]}
        if "professionalism_floor" in identity_core_assistant_trait_baseline:
            baseline_traits["professionalism"] = max(
                baseline_traits["professionalism"],
                self._clamp01(self._to_float(identity_core_assistant_trait_baseline.get("professionalism_floor"), baseline_traits["professionalism"])),
            )
            trait_overrides["professionalism"] = {"source": "identity_core_baseline", "value": baseline_traits["professionalism"]}
        if "sarcasm_ceiling" in identity_core_assistant_trait_baseline:
            baseline_traits["sarcasm"] = min(
                baseline_traits["sarcasm"],
                self._clamp01(self._to_float(identity_core_assistant_trait_baseline.get("sarcasm_ceiling"), baseline_traits["sarcasm"])),
            )
            trait_overrides["sarcasm"] = {"source": "identity_core_baseline", "value": baseline_traits["sarcasm"]}

        if "prefers_directness" in identity_core_interaction:
            baseline_traits["directness"] = self._clamp01(
                self._to_float(identity_core_interaction.get("prefers_directness"), baseline_traits["directness"])
            )
            trait_overrides["directness"] = {
                "source": "identity_core",
                "value": baseline_traits["directness"],
            }
            stable_traits_from_identity_core.append("directness")
        if "prefers_short_answers" in identity_core_interaction:
            brevity = self._clamp01(
                self._to_float(identity_core_interaction.get("prefers_short_answers"), 0.0)
            )
            baseline_traits["verbosity"] = max(0.18, min(0.92, 0.9 - (0.72 * brevity)))
            trait_overrides["verbosity"] = {
                "source": "identity_core",
                "value": baseline_traits["verbosity"],
            }
            stable_traits_from_identity_core.append("verbosity")
        if "allow_light_teasing" in identity_core_interaction or "allows_light_teasing" in identity_core_interaction:
            baseline_traits["teasing"] = 0.38 if bool(
                identity_core_interaction.get("allow_light_teasing", identity_core_interaction.get("allows_light_teasing"))
            ) else 0.06
            trait_overrides["teasing"] = {
                "source": "identity_core",
                "value": baseline_traits["teasing"],
            }
            stable_traits_from_identity_core.append("teasing")
        stable_traits = dict(baseline_traits)
        ignored_runtime_traits = sorted(
            [
                key for key in ("warmth", "directness", "sarcasm", "empathy", "professionalism", "verbosity", "strictness", "teasing")
                if key in dict(state_map.get("stable_traits") or state_map.get("traits") or {})
            ]
        )

        metadata_tags = [
            str(x).strip().lower()
            for x in list(meta_map.get("metadata_tags") or [])
            if str(x).strip()
        ]
        response_bias = {
            "technical_mode": 1.0 if bool(meta_map.get("is_technical")) else 0.0,
            "needs_short_answer": 1.0 if "needs_short_answer" in metadata_tags else 0.0,
            "frustration_softening": 1.0 if mood in {"frustrated", "angry", "sad"} else 0.0,
        }
        dynamic_trait_modifiers = {
            "warmth_delta": 0.0,
            "directness_delta": 0.0,
            "sarcasm_delta": 0.0,
            "empathy_delta": 0.0,
            "professionalism_delta": 0.0,
            "verbosity_delta": 0.0,
            "teasing_delta": 0.0,
        }
        if response_bias["technical_mode"] > 0.0:
            dynamic_trait_modifiers["directness_delta"] += 0.12
            dynamic_trait_modifiers["professionalism_delta"] += 0.06
            dynamic_trait_modifiers["sarcasm_delta"] -= 0.08
        if response_bias["needs_short_answer"] > 0.0:
            dynamic_trait_modifiers["verbosity_delta"] -= 0.18
        emotional_handling = {
            key: identity_core_emotional_handling.get(key)
            for key in (
                "deescalate_on_irritation",
                "treat_short_replies_as_low_bandwidth",
                "warmth_upshift_on_user_distress",
                "playfulness_downshift_on_user_distress",
            )
            if key in identity_core_emotional_handling
        }
        if bool(emotional_handling.get("deescalate_on_irritation")) and mood in {"frustrated", "angry", "sad", "anxious", "tired"}:
            response_bias["frustration_softening"] = 1.0
        if mood in {"frustrated", "angry", "sad", "anxious", "tired"}:
            dynamic_trait_modifiers["warmth_delta"] += 0.10
            dynamic_trait_modifiers["empathy_delta"] += 0.08
            dynamic_trait_modifiers["sarcasm_delta"] -= 0.10
            if "warmth_upshift_on_user_distress" in emotional_handling:
                response_bias["warmth_upshift"] = self._clamp01(
                    self._to_float(emotional_handling.get("warmth_upshift_on_user_distress"), 0.0)
                )
                dynamic_trait_modifiers["warmth_delta"] += float(response_bias["warmth_upshift"] or 0.0)
            if "playfulness_downshift_on_user_distress" in emotional_handling:
                response_bias["playfulness_downshift"] = self._clamp01(
                    self._to_float(emotional_handling.get("playfulness_downshift_on_user_distress"), 0.0)
                )
                dynamic_trait_modifiers["teasing_delta"] -= float(response_bias["playfulness_downshift"] or 0.0)
                dynamic_trait_modifiers["sarcasm_delta"] -= 0.12 * float(response_bias["playfulness_downshift"] or 0.0)
        if active_mode in {"playful", "banter", "light", "casual"}:
            dynamic_trait_modifiers["teasing_delta"] += 0.10
        for trait_name, delta_key in (
            ("warmth", "warmth_delta"),
            ("directness", "directness_delta"),
            ("sarcasm", "sarcasm_delta"),
            ("empathy", "empathy_delta"),
            ("professionalism", "professionalism_delta"),
            ("verbosity", "verbosity_delta"),
            ("teasing", "teasing_delta"),
        ):
            if trait_name not in stable_traits and abs(float(dynamic_trait_modifiers.get(delta_key, 0.0) or 0.0)) <= 1e-9:
                continue
            stable_traits[trait_name] = self._clamp01(
                self._to_float(stable_traits.get(trait_name), baseline_traits.get(trait_name, 0.5))
                + float(dynamic_trait_modifiers.get(delta_key, 0.0) or 0.0)
            )
        response_bias_sources = sorted(
            [
                key for key, include in (
                    ("meta.is_technical", bool(meta_map.get("is_technical"))),
                    ("meta.metadata_tags.needs_short_answer", "needs_short_answer" in metadata_tags),
                    ("mood.frustration_softening", mood in {"frustrated", "angry", "sad"}),
                    ("identity_core.deescalate_on_irritation", bool(emotional_handling.get("deescalate_on_irritation")) and mood in {"frustrated", "angry", "sad", "anxious", "tired"}),
                    ("identity_core.warmth_upshift_on_user_distress", "warmth_upshift" in response_bias),
                    ("identity_core.playfulness_downshift_on_user_distress", "playfulness_downshift" in response_bias),
                )
                if include
            ]
        )
        turn_local_fields = sorted(
            [
                "mood",
                "active_mode",
                *(
                    ["response_bias.technical_mode"] if response_bias["technical_mode"] > 0.0 else []
                ),
                *(
                    ["response_bias.needs_short_answer"] if response_bias["needs_short_answer"] > 0.0 else []
                ),
                *(
                    ["response_bias.frustration_softening"] if response_bias["frustration_softening"] > 0.0 else []
                ),
                *(
                    ["response_bias.warmth_upshift"] if float(response_bias.get("warmth_upshift", 0.0) or 0.0) > 0.0 else []
                ),
                *(
                    ["response_bias.playfulness_downshift"] if float(response_bias.get("playfulness_downshift", 0.0) or 0.0) > 0.0 else []
                ),
            ]
        )
        boundaries = {
            key: bool(identity_core_boundaries.get(key))
            for key in (
                "avoid_overloaded_intros",
                "avoid_baby_talk",
                "avoid_overformal_tone",
                "do_not_invent_user_facts",
            )
            if key in identity_core_boundaries
        }

        return PersonaSnapshot(
            character_id=str(character_id or "assistant").strip() or "assistant",
            mood=mood,
            relation_state=relation_state,
            user_addressing=user_addressing,
            stable_traits=stable_traits,
            response_bias=response_bias,
            boundaries=boundaries,
            emotional_handling=emotional_handling,
            user_profile_hints=user_profile_hints,
            active_mode=active_mode,
            debug={
                "profile_keys": sorted(profile.keys()),
                "memory_blocks": sorted(dict(memory.get("blocks") or {}).keys()),
                "sources": {
                    "persistent_profile_keys": persistent_profile_keys,
                    "profile_hint_fields": profile_hint_fields,
                    "relation_fields_from_persistent": relation_fields_from_persistent,
                    "relation_fields_from_identity_core": sorted(set(relation_fields_from_identity_core)),
                    "relation_source": relation_source,
                    "user_addressing_from_persistent": sorted(user_addressing_from_persistent),
                    "user_addressing_from_identity_core": sorted(set(user_addressing_from_identity_core)),
                    "user_addressing_from_runtime": sorted(set(user_addressing_from_runtime)),
                    "addressing_source": addressing_source,
                    "stable_traits_from_persistent": stable_traits_from_persistent,
                    "stable_traits_from_character_defaults": stable_traits_from_character_defaults,
                    "stable_traits_from_identity_core": sorted(set(stable_traits_from_identity_core)),
                    "stable_traits_ignored_runtime": ignored_runtime_traits,
                    "trait_layer_priority": [
                        "character_specs_defaults",
                        "active_profile_snapshot",
                        "identity_core",
                        "turn_local_modifiers",
                    ],
                    "assistant_trait_baseline_source": assistant_trait_baseline_source,
                    "assistant_trait_baseline_fields_from_identity_core": sorted(
                        [
                            key for key in (
                                "warmth_baseline",
                                "directness_baseline",
                                "empathy_floor",
                                "professionalism_floor",
                                "sarcasm_ceiling",
                            )
                            if key in identity_core_assistant_trait_baseline
                        ]
                    ),
                    "baseline_traits": {key: float(self._clamp01(value)) for key, value in baseline_traits.items()},
                    "dynamic_trait_modifiers": {
                        key: float(value)
                        for key, value in dynamic_trait_modifiers.items()
                        if abs(float(value or 0.0)) > 1e-9
                    },
                    "interaction_style_source": {
                        "prefers_directness": ("identity_core" if "prefers_directness" in identity_core_interaction else "default"),
                        "prefers_short_answers": ("identity_core" if "prefers_short_answers" in identity_core_interaction else "default"),
                        "allow_light_teasing": (
                            "identity_core"
                            if ("allow_light_teasing" in identity_core_interaction or "allows_light_teasing" in identity_core_interaction)
                            else "default"
                        ),
                        "technical_collaboration_style": (
                            "identity_core" if "technical_collaboration_style" in identity_core_interaction else "default"
                        ),
                    },
                    "boundary_fields_from_identity_core": sorted(boundaries.keys()),
                    "boundary_source": {
                        "avoid_overloaded_intros": ("identity_core" if "avoid_overloaded_intros" in boundaries else "default"),
                        "avoid_baby_talk": ("identity_core" if "avoid_baby_talk" in boundaries else "default"),
                        "avoid_overformal_tone": ("identity_core" if "avoid_overformal_tone" in boundaries else "default"),
                        "do_not_invent_user_facts": ("identity_core" if "do_not_invent_user_facts" in boundaries else "default"),
                    },
                    "emotional_handling_fields_from_identity_core": sorted(emotional_handling.keys()),
                    "emotional_handling_source": {
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
                    },
                    "trait_overrides": trait_overrides,
                    "mood_source": mood_source,
                    "active_mode_source": active_mode_source,
                    "response_bias_sources": response_bias_sources,
                    "turn_local_fields": turn_local_fields,
                },
            },
        )

    def _flatten_profile_snapshot(self, snapshot: dict[str, Any] | None) -> dict[str, Any]:
        flat = dict(snapshot or {})
        active_facts = dict(flat.get("active_facts") or {})
        for item in active_facts.values():
            if not isinstance(item, dict):
                continue
            predicate = str(item.get("predicate") or "").strip().lower()
            value = item.get("value")
            if self._is_empty(value):
                continue
            if predicate in flat:
                existing = flat.get(predicate)
                if isinstance(existing, list):
                    if value not in existing:
                        flat[predicate] = [*existing, value]
                elif existing != value:
                    flat[predicate] = [existing, value]
            else:
                flat[predicate] = value
        return flat

    @staticmethod
    def _normalize_trait_defaults(value: Any) -> dict[str, float]:
        row = dict(value or {})
        out: dict[str, float] = {}
        for key in (
            "warmth",
            "directness",
            "sarcasm",
            "empathy",
            "professionalism",
            "verbosity",
            "strictness",
            "teasing",
        ):
            if key in row:
                out[key] = float(PersonaSnapshotBuilder._clamp01(PersonaSnapshotBuilder._to_float(row.get(key), 0.0)))
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

    @staticmethod
    def _to_clean_list(value: Any) -> list[str]:
        return [str(x).strip() for x in list(value or []) if str(x).strip()]

    @staticmethod
    def _is_empty(value: Any) -> bool:
        if value is None:
            return True
        if isinstance(value, str) and not value.strip():
            return True
        if isinstance(value, list) and not value:
            return True
        if isinstance(value, dict) and not value:
            return True
        return False
