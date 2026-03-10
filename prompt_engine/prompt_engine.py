from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from config.settings import load_config
from core.character_runtime import CharacterRuntime
from core.spec_registry import load_spec
from llm.provider_base import Message
from prompt_engine.prompt_registry import PromptRegistry
from prompt_engine.token_budget_manager import ContextBlock, TokenBudgetManager


@dataclass(frozen=True)
class PromptEngineResult:
    messages: list[Message]
    sections: dict[str, str]


class PromptEngine:
    """Compose final LLM messages from prompt templates, personality, state, and token budgets."""

    def __init__(
        self,
        registry: PromptRegistry | None = None,
        budget_manager: TokenBudgetManager | None = None,
        character_runtime: CharacterRuntime | None = None,
    ):
        self.registry = registry or PromptRegistry()
        self.budget_manager = budget_manager or TokenBudgetManager()
        self.character_engine = character_runtime or CharacterRuntime()
        self._settings = load_config()
        self._system_spec: dict[str, Any] = {}

    def compose(
        self,
        *,
        prompt_pack,
        state: dict[str, Any] | None = None,
        traits: dict[str, Any] | None = None,
        policies: dict[str, Any] | None = None,
    ) -> PromptEngineResult:
        state_map = dict(state or {})
        traits_map = dict(traits or {})
        policies_map = dict(policies or {})
        blocks = dict(getattr(prompt_pack, "blocks", {}) or {})
        memory_context = _as_dict(state_map.get("memory_context"))
        memory_blocks = _as_dict(memory_context.get("blocks"))
        self._system_spec = load_spec("system", required=False)
        prompt_toggles = _as_dict(self._system_spec.get("prompt_toggles"))
        state_toggles = _as_dict(state_map.get("prompt_toggles"))
        safety_default = _coerce_bool(
            prompt_toggles.get("response_safety_filter"),
            default=bool(self._settings.prompt_response_safety_filter_enabled),
        )
        formatting_default = _coerce_bool(
            prompt_toggles.get("response_formatting"),
            default=bool(self._settings.prompt_response_formatting_enabled),
        )
        safety_legacy_present = "prompt_response_safety_filter_enabled" in state_map
        safety_nested_present = "response_safety_filter" in state_toggles
        safety_legacy_value = state_map.get("prompt_response_safety_filter_enabled")
        safety_nested_value = state_toggles.get("response_safety_filter")
        if (safety_legacy_present and safety_legacy_value is None) or (
            safety_nested_present and safety_nested_value is None
        ):
            # Explicit null means "disable safety filter".
            safety_prompt_enabled = False
        else:
            safety_prompt_enabled = _coerce_bool(
                _pick_first(
                    safety_legacy_value,
                    safety_nested_value,
                ),
                default=safety_default,
            )
        formatting_prompt_enabled = _coerce_bool(
            _pick_first(
                state_map.get("prompt_response_formatting_enabled"),
                state_toggles.get("response_formatting"),
            ),
            default=formatting_default,
        )

        active_character = _pick(
            state_map.get("active_character_id"),
            traits_map.get("character"),
            policies_map.get("character"),
            state_map.get("active_personality_id"),
            "default",
        ).lower()

        safety_mode = str(state_map.get("safety_mode") or "").strip().lower()
        needs_safety_json = safety_mode in {"content_filter", "locked", "strict"}

        base_doc = self._safe_doc(key="system.base", fallback_text=blocks.get("system_role") or "")
        
        # When safety_mode is active, we just include the safety_doc rules, BUT we shouldn't force JSON format
        # unless JSON mode is explicitly enabled in the API parameters.
        safety_doc_text = ""
        if needs_safety_json and safety_prompt_enabled:
            s_doc = self._safe_doc(key="response.safety_filter", fallback_text="")
            # Strip out the explicit JSON requirement from the safety doc if it exists,
            # so the model can answer naturally, unless JSON mode is active.
            safety_doc_text = str(s_doc.get("text", "")).replace(
                'Возврат JSON:\n{"safe":true|false, "reason":"", "output":"..."}', 
                'Верни ответ в обычном текстовом формате.'
            )
            
        safety_doc = {"text": safety_doc_text}
        formatting_doc_text = ""
        if formatting_prompt_enabled:
            formatting_doc_text = str(self._safe_doc(key="response.formatting", fallback_text="").get("text", "")).strip()
        formatting_doc = {"text": formatting_doc_text}
        dynamic_rules = self._build_dynamic_rules_block(
            state_map=state_map,
            policies_map=policies_map,
        )
        
        character_prompt_block = str(state_map.get("character_prompt_block") or "").strip()
        if not character_prompt_block and active_character:
            try:
                character_prompt_block = str(self.character_engine.build_personality_block(active_character) or "").strip()
            except Exception:
                character_prompt_block = ""
        personality_core_text = character_prompt_block or f"[CHAR_META]\ncharacter={active_character}\nstyle_source=characters"
        active_personality = active_character or "default"

        user_profile_block = self._build_user_profile_block(state_map)
        metadata_block = self._build_metadata_block(state_map=state_map, blocks=blocks)
        tools_state_block = self._build_tools_state_block(state_map, memory_blocks=memory_blocks)
        memory_retrieval_block = self._build_memory_retrieval_block(blocks=blocks, memory_blocks=memory_blocks)
        web_evidence_block = self._build_web_evidence_block(
            blocks=blocks,
            memory_blocks=memory_blocks,
            state_map=state_map,
        )
        recent_chat_block = "" if memory_blocks else str(blocks.get("conversation_tail") or "")
        long_summary_block = str(
            _pick_first(
                memory_blocks.get("session_summary"),
                blocks.get("long_summary"),
                state_map.get("dialog_summary"),
                "",
            )
            or ""
        )
        user_block = str(_pick_first(memory_blocks.get("user_message"), blocks.get("user_message"), "") or "")
        verbosity_level = _resolve_verbosity_level(state_map=state_map, blocks=blocks)
        verbosity_limits = self.budget_manager.apply_verbosity(verbosity_level)
        memory_floor_tokens = max(48, int(self.budget_manager.budget.memory_retrieval * 0.28))
        web_v2_cfg = _as_dict(getattr(self._settings, "web_v2", {}))
        try:
            web_evidence_max_tokens = max(80, int(web_v2_cfg.get("evidence_budget_tokens") or 220))
        except Exception:
            web_evidence_max_tokens = 220

        context_blocks = [
            ContextBlock(
                id="system_core",
                content=base_doc["text"],
                bucket="system",
                priority=100,
                required=True,
                max_tokens=self.budget_manager.budget.system_core,
            ),
            ContextBlock(
                id="personality_core",
                content=personality_core_text,
                bucket="personality",
                priority=92,
                required=True,
                max_tokens=self.budget_manager.budget.personality,
            ),
            ContextBlock(
                id="rules",
                content=_join_non_empty([safety_doc["text"], formatting_doc["text"], dynamic_rules]),
                bucket="rules",
                priority=96,
                required=True,
                max_tokens=self.budget_manager.budget.rules,
            ),
            ContextBlock(
                id="user_profile",
                content=user_profile_block,
                bucket="user_profile",
                priority=74,
                shrink_strategy="summarize",
                max_tokens=int(verbosity_limits.get("user_profile", self.budget_manager.budget.user_profile)),
            ),
            ContextBlock(
                id="metadata",
                content=metadata_block,
                bucket="metadata",
                priority=80,
                max_tokens=self.budget_manager.budget.metadata,
            ),
            ContextBlock(
                id="tools_state",
                content=tools_state_block,
                bucket="tools",
                priority=66,
                shrink_strategy="truncate_head",
                max_tokens=self.budget_manager.budget.tools_state,
            ),
            ContextBlock(
                id="memory_retrieval",
                content=memory_retrieval_block,
                bucket="memory",
                priority=78,
                shrink_strategy="summarize",
                max_tokens=self.budget_manager.budget.memory_retrieval,
                min_tokens=memory_floor_tokens,
            ),
            ContextBlock(
                id="web_evidence",
                content=web_evidence_block,
                bucket="memory",
                priority=79,
                shrink_strategy="summarize",
                max_tokens=web_evidence_max_tokens,
            ),
            ContextBlock(
                id="recent_chat",
                content=recent_chat_block,
                bucket="history",
                priority=66,
                shrink_strategy="summarize",
                max_tokens=int(verbosity_limits.get("history", self.budget_manager.budget.recent_chat)),
            ),
            ContextBlock(
                id="long_summary",
                content=long_summary_block,
                bucket="long_summary",
                priority=70,
                shrink_strategy="summarize",
                max_tokens=int(verbosity_limits.get("long_summary", self.budget_manager.budget.long_summary)),
            ),
            ContextBlock(
                id="output_schema",
                content=str(blocks.get("output_schema") or ""),
                bucket="output",
                priority=88,
                max_tokens=int(verbosity_limits.get("output", self.budget_manager.budget.output)),
            ),
            ContextBlock(
                id="user",
                content=user_block,
                bucket="user",
                priority=100,
                required=True,
                max_tokens=self.budget_manager.budget.user,
                min_tokens=32,
            ),
        ]

        fitted, budget_stats = self.budget_manager.fit_context_blocks(context_blocks)
        system_content = _join_sections(
            [
                ("SYSTEM_CORE", fitted.get("system_core", "")),
                ("PERSONALITY", fitted.get("personality_core", "")),
                ("RULES", fitted.get("rules", "")),
                ("USER_PROFILE", fitted.get("user_profile", "")),
                ("METADATA", fitted.get("metadata", "")),
                ("TOOLS_STATE", fitted.get("tools_state", "")),
                ("MEMORY", fitted.get("memory_retrieval", "")),
                ("WEB_EVIDENCE", fitted.get("web_evidence", "")),
                ("RECENT_CHAT", fitted.get("recent_chat", "")),
                ("LONG_SUMMARY", fitted.get("long_summary", "")),
                ("OUTPUT_SCHEMA", fitted.get("output_schema", "")),
            ]
        )
        user_content = str(fitted.get("user") or "").strip()

        messages = [
            Message(role="system", content=system_content),
            Message(role="user", content=user_content),
        ]

        active_rule_prompts: list[str] = []
        if safety_doc_text:
            active_rule_prompts.append("response.safety_filter")
        if formatting_doc_text:
            active_rule_prompts.append("response.formatting")
        sections = {
            "active_personality_id": active_personality,
            "active_personality_name": active_personality,
            "active_personality_version": "character-driven",
            "active_character_id": active_character,
            "base_prompt_id": base_doc["id"],
            "base_prompt_version": base_doc["version"],
            "persona_prompt_id": "character_prompt_block",
            "persona_prompt_version": "character-driven",
            "style_prompt_id": "character_prompt_block",
            "style_prompt_version": "character-driven",
            "rules_prompt_id": "+".join(active_rule_prompts) if active_rule_prompts else "none",
            "rules_prompt_version": "registry",
            "system": system_content,
            "user": user_content,
        }
        sections["budget_total_tokens"] = str(budget_stats.get("total_tokens") or 0)
        sections["budget_dropped"] = ",".join([str(x) for x in list(budget_stats.get("dropped") or [])])
        sections["budget_trimmed"] = ",".join([str(x) for x in list(budget_stats.get("trimmed") or [])])
        sections["dialog_verbosity_level"] = f"{verbosity_level:.3f}"
        sections["web_evidence_included"] = "1" if str(fitted.get("web_evidence") or "").strip() else "0"

        return PromptEngineResult(messages=messages, sections=sections)

    def _safe_doc(self, *, key: str | None = None, path: str | None = None, fallback_text: str = "") -> dict[str, str]:
        if key:
            try:
                doc = self.registry.get_prompt(key)
                return {
                    "id": str(doc.id or key),
                    "version": str(doc.version or "0.0.0"),
                    "text": str(doc.text or "").strip(),
                }
            except Exception:
                pass
        if path:
            rel = str(path or "").replace("\\", "/").strip().strip("/")
            if rel:
                try:
                    doc = self.registry.get_by_path(rel)
                    return {
                        "id": str(doc.id or rel),
                        "version": str(doc.version or "0.0.0"),
                        "text": str(doc.text or "").strip(),
                    }
                except Exception:
                    pass
        return {
            "id": str(key or path or "fallback"),
            "version": "0.0.0",
            "text": str(fallback_text or "").strip(),
        }

    @staticmethod
    def _build_user_profile_block(state_map: dict[str, Any]) -> str:
        profile_summary = _as_dict(state_map.get("profile_summary"))
        user_profile = profile_summary.get("user")
        if isinstance(user_profile, str) and user_profile.strip():
            return user_profile.strip()
        if isinstance(user_profile, dict):
            lines = []
            for key, value in user_profile.items():
                if value is None:
                    continue
                lines.append(f"- {key}: {value}")
            if lines:
                return "\n".join(lines)
        return ""

    @staticmethod
    def _build_metadata_block(*, state_map: dict[str, Any], blocks: dict[str, str]) -> str:
        tags_block = str(blocks.get("context_tags") or "").strip()
        context = _as_dict(state_map.get("context_tags"))
        lines = []
        if tags_block:
            lines.append(tags_block)
        for key in (
            "lang",
            "intent",
            "mood",
            "active_mode",
            "topic",
            "user_greeting",
            "allow_greeting",
            "new_session",
            "greeted_today",
            "conversation_state",
            "smalltalk_allowed",
            "should_ask_back",
            "local_date",
            "local_region",
            "greeting_allowed",
            "dialog_sarcasm_level",
            "dialog_warmth_level",
            "dialog_strictness_level",
            "dialog_verbosity_level",
            "is_technical",
            "allowed_term",
            "use_term_now",
            "address_terms_policy",
        ):
            value = str(context.get(key) or "").strip()
            if value:
                lines.append(f"- {key}: {value}")
        dialog_mode = {
            "greeting_allowed": str(context.get("greeting_allowed") or context.get("allow_greeting") or "").strip().lower(),
            "smalltalk_allowed": str(context.get("smalltalk_allowed") or "").strip().lower(),
            "sarcasm_level": str(context.get("dialog_sarcasm_level") or "").strip(),
            "warmth_level": str(context.get("dialog_warmth_level") or "").strip(),
            "strictness_level": str(context.get("dialog_strictness_level") or "").strip(),
            "verbosity_level": str(context.get("dialog_verbosity_level") or "").strip(),
        }
        if any(dialog_mode.values()):
            pretty = ", ".join([f"{k}: {v}" for k, v in dialog_mode.items() if v])
            lines.append(f"- dialog_mode: {{{pretty}}}")
            lines.append("- runtime_constraints: hard constraints are enforced in response post-filter.")
        if str(context.get("use_term_now") or "").strip().lower() in {"false", "0", "no"}:
            lines.append("- terms_rule: do not use endearment terms in this reply.")
        elif str(context.get("use_term_now") or "").strip().lower() in {"true", "1", "yes"}:
            lines.append("- terms_rule: allowed at most one short endearment term.")
        return "\n".join(lines).strip()

    @staticmethod
    def _build_tools_state_block(state_map: dict[str, Any], *, memory_blocks: dict[str, Any] | None = None) -> str:
        context_tool = str(_as_dict(memory_blocks).get("active_tool_state") or "").strip()
        if context_tool:
            return context_tool
        value = state_map.get("last_tool_result")
        if value is None:
            return ""
        if isinstance(value, str):
            return value.strip()
        try:
            import json

            return json.dumps(value, ensure_ascii=False)
        except Exception:
            return str(value)

    @staticmethod
    def _build_memory_retrieval_block(*, blocks: dict[str, str], memory_blocks: dict[str, Any]) -> str:
        if not memory_blocks:
            return str(blocks.get("retrieved_memories") or "")
        order = [
            ("WORKING_MEMORY", "working_memory"),
            ("SESSION_SUMMARY", "session_summary"),
            ("SEMANTIC_FACTS", "retrieved_semantic"),
            ("EPISODIC_MEMORIES", "retrieved_episodic"),
            ("DOCUMENT_EVIDENCE", "retrieved_docs"),
            ("TASK_TOOL_STATE", "active_tool_state"),
            ("UNRESOLVED_ITEMS", "unresolved_items"),
        ]
        parts: list[str] = []
        for title, key in order:
            text = str(memory_blocks.get(key) or "").strip()
            if not text:
                continue
            parts.append(f"[{title}]\n{text}")
        return "\n\n".join(parts).strip()

    @staticmethod
    def _build_web_evidence_block(
        *,
        blocks: dict[str, str],
        memory_blocks: dict[str, Any],
        state_map: dict[str, Any],
    ) -> str:
        direct_candidates = [
            str(memory_blocks.get("web_evidence") or "").strip(),
            str(blocks.get("web_evidence") or "").strip(),
            str(_as_dict(state_map.get("web_evidence_context")).get("prompt_block") or "").strip(),
        ]
        for item in direct_candidates:
            if item:
                return item

        context = _as_dict(state_map.get("web_evidence_context"))
        if not context:
            return ""
        summary = str(context.get("summary") or "").strip()
        freshness = str(context.get("freshness_summary") or "").strip()
        key_facts = [str(x).strip() for x in list(context.get("key_facts") or []) if str(x).strip()]
        citations = [str(x).strip() for x in list(context.get("compact_citations") or []) if str(x).strip()]
        conflict_notes = [str(x).strip() for x in list(context.get("conflict_notes") or []) if str(x).strip()]
        conflicts = bool(context.get("conflicting_sources"))

        lines: list[str] = []
        if summary:
            lines.append(f"- summary: {summary}")
        if freshness:
            lines.append(f"- freshness: {freshness}")
        if key_facts:
            lines.append("- key_facts:")
            for fact in list(key_facts)[:8]:
                lines.append(f"  - {fact}")
        if citations:
            lines.append("- citations:")
            for citation in list(citations)[:4]:
                lines.append(f"  - {citation}")
        lines.append(f"- source_conflicts: {'yes' if conflicts else 'no'}")
        if conflicts and conflict_notes:
            lines.append("- conflict_notes:")
            for note in list(conflict_notes)[:4]:
                lines.append(f"  - {note}")
        if not lines:
            return ""
        return "[WEB_EVIDENCE]\n" + "\n".join(lines).strip()

    @staticmethod
    def _build_dynamic_rules_block(*, state_map: dict[str, Any], policies_map: dict[str, Any]) -> str:
        rows: list[str] = []
        for key in ("rules", "policy_rules", "constraints"):
            value = policies_map.get(key)
            if isinstance(value, list):
                candidates = list(value)
            elif value is None:
                candidates = []
            else:
                candidates = [value]
            for item in candidates:
                text = str(item or "").strip()
                if not text:
                    continue
                rows.append(text if text.startswith("- ") else f"- {text}")

        context = _as_dict(state_map.get("context_tags"))
        style = str(context.get("web_response_style") or "").strip().lower()
        if style == "factual_direct":
            rows.append("- Time-sensitive web answers must be direct and factual without rhetorical/flirty openers.")

        deduped: list[str] = []
        seen: set[str] = set()
        for row in rows:
            key = str(row or "").strip().lower()
            if not key or key in seen:
                continue
            seen.add(key)
            deduped.append(row)
        return "\n".join(deduped).strip()


def _resolve_persona_name(
    *,
    traits_map: dict[str, Any],
    state_map: dict[str, Any],
    policies_map: dict[str, Any],
) -> str:
    values = [
        traits_map.get("persona"),
        traits_map.get("personality"),
        state_map.get("active_personality_id"),
        state_map.get("persona"),
        state_map.get("personality"),
        policies_map.get("persona"),
        policies_map.get("personality"),
        traits_map.get("profile"),
        state_map.get("profile"),
        policies_map.get("profile"),
    ]
    raw = ""
    for value in values:
        text = str(value or "").strip().lower()
        if text:
            raw = text
            break

    if any(x in raw for x in ("flirty", "playful", "romantic")):
        return "flirty"
    if any(x in raw for x in ("strict", "formal")):
        return "strict"
    if "support" in raw:
        return "supportive"
    return "default"


def _resolve_verbosity_level(*, state_map: dict[str, Any], blocks: dict[str, str]) -> float:
    context = _as_dict(state_map.get("context_tags"))
    dialog_mode = _as_dict(state_map.get("dialog_mode"))
    candidates = [
        dialog_mode.get("verbosity_level"),
        context.get("dialog_verbosity_level"),
        context.get("verbosity_level"),
    ]
    for value in candidates:
        try:
            return max(0.0, min(1.0, float(value)))
        except Exception:
            continue
    tags_block = str(blocks.get("context_tags") or "")
    match = re.search(r"dialog_verbosity_level:\s*([0-9]*\.?[0-9]+)", tags_block)
    if match:
        try:
            return max(0.0, min(1.0, float(match.group(1))))
        except Exception:
            pass
    return 0.46


def _join_sections(parts: list[tuple[str, str]]) -> str:
    out: list[str] = []
    for name, text in parts:
        block = str(text or "").strip()
        if not block or block == "- none":
            continue
        out.append(f"<<<{name}>>>\n{block}\n<<<END_{name}>>>")
    return "\n\n".join(out).strip()


def _as_dict(value) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if value is None:
        return {}
    try:
        return dict(vars(value))
    except Exception:
        return {}


def _pick_first(*values):
    for value in values:
        if value is None:
            continue
        return value
    return None


def _coerce_bool(value, *, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return bool(default)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return bool(default)


def _pick(*values) -> str:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return ""


def _join_non_empty(items: list[str]) -> str:
    rows = [str(x or "").strip() for x in list(items or []) if str(x or "").strip()]
    return "\n\n".join(rows).strip()


def _first_sentences(text: str, *, max_lines: int = 4) -> str:
    lines = [x.strip() for x in str(text or "").splitlines() if x.strip()]
    if not lines:
        return ""
    return "\n".join(lines[:max_lines]).strip()
