from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any

from llm.tokenizer import estimate_tokens


_BOM_ARTIFACTS = ("\ufeff", "\ufffe", "ï»¿")
_ZERO_WIDTH_RE = re.compile(r"[\u200B-\u200F\u2060\uFEFF]")
_CONTROL_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")
_HSPACE_RE = re.compile(r"[ \t\f\v]+")


@dataclass(frozen=True)
class PromptBudgets:
    total_tokens: int = 2200
    system_tokens: int = 240
    persona_tokens: int = 200
    state_tokens: int = 180
    tags_tokens: int = 100
    memory_tokens: int = 600
    long_summary_tokens: int = 180
    tail_tokens: int = 500
    user_tokens: int = 240
    output_schema_tokens: int = 220
    tail_turns: int = 8
    tail_turn_tokens: int = 120
    memory_item_tokens: int = 110

    @classmethod
    def from_policies(cls, policies: dict[str, Any]) -> PromptBudgets:
        p = _as_dict(policies)
        b = _as_dict(p.get("budgets"))
        return cls(
            total_tokens=_to_int(b.get("total_tokens", p.get("total_tokens")), 2200, 256),
            system_tokens=_to_int(b.get("system_tokens", p.get("system_tokens")), 240, 32),
            persona_tokens=_to_int(b.get("persona_tokens", p.get("persona_tokens")), 200, 32),
            state_tokens=_to_int(b.get("state_tokens", p.get("state_tokens")), 180, 32),
            tags_tokens=_to_int(b.get("tags_tokens", p.get("tags_tokens")), 100, 16),
            memory_tokens=_to_int(b.get("memory_tokens", p.get("memory_tokens")), 600, 64),
            long_summary_tokens=_to_int(b.get("long_summary_tokens", p.get("long_summary_tokens")), 180, 32),
            tail_tokens=_to_int(b.get("tail_tokens", p.get("tail_tokens")), 500, 64),
            user_tokens=_to_int(b.get("user_tokens", p.get("user_tokens")), 240, 32),
            output_schema_tokens=_to_int(
                b.get("output_schema_tokens", p.get("output_schema_tokens")),
                220,
                32,
            ),
            tail_turns=_to_int(
                b.get("tail_turns", p.get("tail_turns", p.get("conversation_tail_n"))),
                8,
                1,
            ),
            tail_turn_tokens=_to_int(
                b.get("tail_turn_tokens", p.get("tail_turn_tokens")),
                120,
                24,
            ),
            memory_item_tokens=_to_int(
                b.get("memory_item_tokens", p.get("memory_item_tokens")),
                110,
                24,
            ),
        )

    def as_dict(self) -> dict[str, int]:
        return {
            "total_tokens": self.total_tokens,
            "system_tokens": self.system_tokens,
            "persona_tokens": self.persona_tokens,
            "state_tokens": self.state_tokens,
            "tags_tokens": self.tags_tokens,
            "memory_tokens": self.memory_tokens,
            "long_summary_tokens": self.long_summary_tokens,
            "tail_tokens": self.tail_tokens,
            "user_tokens": self.user_tokens,
            "output_schema_tokens": self.output_schema_tokens,
            "tail_turns": self.tail_turns,
            "tail_turn_tokens": self.tail_turn_tokens,
            "memory_item_tokens": self.memory_item_tokens,
        }


@dataclass(frozen=True)
class PromptPack:
    system_prompt: str
    user_message: str
    full_prompt: str
    messages: list[dict[str, str]]
    blocks: dict[str, str]
    token_usage: dict[str, int]
    budgets: dict[str, int]
    cut_info: dict[str, Any] = field(default_factory=dict)
    context_tags: dict[str, str] = field(default_factory=dict)
    selected_memories: list[dict[str, Any]] = field(default_factory=list)
    conversation_tail: list[dict[str, str]] = field(default_factory=list)


class PromptBuilder:
    DEFAULT_SYSTEM_RULES = [
        "You are MMis assistant.",
        "Follow system rules and active policies strictly.",
        "Prefer concise, clear, and actionable replies.",
        "If context is insufficient, ask a clarifying question.",
        "If use_term_now is false, do not use endearment address terms.",
        "If use_term_now is true, you may use at most one short address term.",
    ]
    DEFAULT_OUTPUT_SCHEMA = (
        "Return plain text by default. "
        "For tool invocation return strict JSON object: "
        '{"tool":"<name>","args":{...}}.'
    )
    _FULL_ORDER = (
        "system_role",
        "persona",
        "state_summary",
        "context_tags",
        "retrieved_memories",
        "long_summary",
        "conversation_tail",
        "user_message",
        "output_schema",
    )
    _SYSTEM_ORDER = (
        "system_role",
        "persona",
        "state_summary",
        "context_tags",
        "retrieved_memories",
        "long_summary",
        "conversation_tail",
        "output_schema",
    )
    _SHRINK_ORDER = (
        "retrieved_memories",
        "conversation_tail",
        "long_summary",
        "state_summary",
        "persona",
        "context_tags",
    )

    @classmethod
    def build(
        cls,
        state,
        user_msg: str,
        retrieved_memories,
        traits,
        policies,
    ) -> PromptPack:
        return cls()._build(
            state=state,
            user_msg=user_msg,
            retrieved_memories=retrieved_memories,
            traits=traits,
            policies=policies,
        )

    def _build(
        self,
        state,
        user_msg: str,
        retrieved_memories,
        traits,
        policies,
    ) -> PromptPack:
        state_map = _as_dict(state)
        traits_map = _as_dict(traits)
        policies_map = _coerce_policies(policies)
        budgets = PromptBudgets.from_policies(policies_map)

        context_tags = self._extract_context_tags(state_map, traits_map, policies_map)
        memories_block, selected_memories, dropped_memories = self._build_memories_block(
            retrieved_memories,
            budgets,
        )
        tail_block, conversation_tail, dropped_tail = self._build_conversation_tail_block(
            state_map,
            budgets,
        )
        long_summary_block = self._build_long_summary_block(state_map, budgets, dropped_tail=dropped_tail)

        blocks = {
            "system_role": self._build_system_role_block(policies_map),
            "persona": self._build_persona_block(state_map, traits_map, policies_map),
            "state_summary": self._build_state_summary_block(state_map),
            "context_tags": self._build_context_tags_block(context_tags),
            "retrieved_memories": memories_block,
            "long_summary": long_summary_block,
            "conversation_tail": tail_block,
            "user_message": _normalize_text(user_msg),
            "output_schema": self._build_output_schema_block(policies_map),
        }
        blocks, cut_info = self._apply_block_budgets(blocks, budgets)
        blocks = self._enforce_total_budget(blocks, budgets, cut_info)

        full_prompt = self._render_blocks(blocks, self._FULL_ORDER)
        system_prompt = self._render_blocks(blocks, self._SYSTEM_ORDER)
        user_message = blocks.get("user_message", "")
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]

        token_usage = {name: estimate_tokens(text) for name, text in blocks.items()}
        token_usage["system_prompt"] = estimate_tokens(system_prompt)
        token_usage["full_prompt"] = estimate_tokens(full_prompt)

        cut_info["dropped_memories"] = dropped_memories
        cut_info["dropped_tail_messages"] = dropped_tail

        return PromptPack(
            system_prompt=system_prompt,
            user_message=user_message,
            full_prompt=full_prompt,
            messages=messages,
            blocks=blocks,
            token_usage=token_usage,
            budgets=budgets.as_dict(),
            cut_info=cut_info,
            context_tags=context_tags,
            selected_memories=selected_memories,
            conversation_tail=conversation_tail,
        )

    def _build_system_role_block(self, policies: dict[str, Any]) -> str:
        lines: list[str] = []
        lines.extend(self.DEFAULT_SYSTEM_RULES)
        for key in ("system_role", "role", "global_rule"):
            value = _normalize_text(policies.get(key))
            if value:
                lines.append(value)

        for key in ("rules", "policy_rules", "constraints"):
            for item in _as_list(policies.get(key)):
                value = _normalize_text(item)
                if value:
                    lines.append(value)

        unique: list[str] = []
        seen: set[str] = set()
        for line in lines:
            low = line.lower()
            if low in seen:
                continue
            seen.add(low)
            unique.append(line)
        return "\n".join(f"- {line}" for line in unique)

    def _build_persona_block(
        self,
        state: dict[str, Any],
        traits: dict[str, Any],
        policies: dict[str, Any],
    ) -> str:
        character_prompt = _normalize_text(state.get("character_prompt_block"))
        if character_prompt:
            return character_prompt

        character = _normalize_text(
            state.get("active_character_id")
            or state.get("character")
            or traits.get("character")
            or policies.get("character")
            or "default"
        )
        lines = [
            f"- character: {character}",
            "- style_source: data/characters/<character_id>",
            "- note: no legacy personality profile overlays.",
        ]
        return _normalize_text("\n".join(lines))

    def _build_state_summary_block(self, state: dict[str, Any]) -> str:
        mode = _normalize_text(state.get("mode")) or "default"
        task = (
            _normalize_text(state.get("current_task"))
            or _normalize_text(state.get("task"))
            or "none"
        )
        summary = (
            _normalize_text(state.get("dialog_summary"))
            or _normalize_text(state.get("summary"))
            or _normalize_text(state.get("short_summary"))
            or "none"
        )
        focus = _normalize_text(state.get("focus") or state.get("active_goal") or "")

        lines = [
            f"- mode: {mode}",
            f"- current_task: {task}",
            f"- dialog_summary: {summary}",
        ]
        personality = _normalize_text(state.get("active_personality_id") or state.get("personality") or "default")
        lines.append(f"- active_personality: {personality}")
        character = _normalize_text(state.get("active_character_id") or state.get("character") or personality)
        lines.append(f"- active_character: {character}")
        blend = _as_dict(state.get("personality_blend"))
        if blend and bool(blend.get("active")):
            lines.append(
                f"- personality_blend: from={blend.get('from')} to={blend.get('to')} "
                f"step={blend.get('step')}/{blend.get('steps')}"
            )
        if focus:
            lines.append(f"- focus: {focus}")
        return "\n".join(lines)

    def _extract_context_tags(
        self,
        state: dict[str, Any],
        traits: dict[str, Any],
        policies: dict[str, Any],
    ) -> dict[str, str]:
        state_tags = _as_dict(state.get("context_tags"))
        policy_tags = _as_dict(policies.get("context_tags"))
        trait_tags = _as_dict(traits.get("context_tags"))

        def _pick(*values) -> str:
            for value in values:
                text = _normalize_text(value)
                if text:
                    return text
            return ""

        tags = {
            "lang": _pick(
                state_tags.get("lang"),
                state.get("lang"),
                state.get("language"),
                policy_tags.get("lang"),
                trait_tags.get("lang"),
            ),
            "mood": _pick(
                state_tags.get("mood"),
                state.get("mood"),
                policy_tags.get("mood"),
                trait_tags.get("mood"),
            ),
            "intent": _pick(
                state_tags.get("intent"),
                state.get("intent"),
                policy_tags.get("intent"),
                trait_tags.get("intent"),
            ),
            "topic": _pick(
                state_tags.get("topic"),
                state.get("topic"),
                state.get("current_topic"),
                policy_tags.get("topic"),
                trait_tags.get("topic"),
            ),
            "user_greeting": _pick(
                state_tags.get("user_greeting"),
                state.get("user_greeting"),
                policy_tags.get("user_greeting"),
                trait_tags.get("user_greeting"),
            ),
            "allow_greeting": _pick(
                state_tags.get("allow_greeting"),
                state.get("allow_greeting"),
                policy_tags.get("allow_greeting"),
                trait_tags.get("allow_greeting"),
            ),
            "new_session": _pick(
                state_tags.get("new_session"),
                state.get("new_session"),
                policy_tags.get("new_session"),
                trait_tags.get("new_session"),
            ),
            "greeted_today": _pick(
                state_tags.get("greeted_today"),
                state.get("greeted_today"),
                policy_tags.get("greeted_today"),
                trait_tags.get("greeted_today"),
            ),
            "conversation_state": _pick(
                state_tags.get("conversation_state"),
                state.get("conversation_state"),
                policy_tags.get("conversation_state"),
                trait_tags.get("conversation_state"),
            ),
            "smalltalk_allowed": _pick(
                state_tags.get("smalltalk_allowed"),
                state.get("smalltalk_allowed"),
                policy_tags.get("smalltalk_allowed"),
                trait_tags.get("smalltalk_allowed"),
            ),
            "greeting_allowed": _pick(
                state_tags.get("greeting_allowed"),
                state.get("greeting_allowed"),
                policy_tags.get("greeting_allowed"),
                trait_tags.get("greeting_allowed"),
            ),
            "dialog_sarcasm_level": _pick(
                state_tags.get("dialog_sarcasm_level"),
                state.get("dialog_sarcasm_level"),
                policy_tags.get("dialog_sarcasm_level"),
                trait_tags.get("dialog_sarcasm_level"),
            ),
            "dialog_warmth_level": _pick(
                state_tags.get("dialog_warmth_level"),
                state.get("dialog_warmth_level"),
                policy_tags.get("dialog_warmth_level"),
                trait_tags.get("dialog_warmth_level"),
            ),
            "dialog_strictness_level": _pick(
                state_tags.get("dialog_strictness_level"),
                state.get("dialog_strictness_level"),
                policy_tags.get("dialog_strictness_level"),
                trait_tags.get("dialog_strictness_level"),
            ),
            "dialog_verbosity_level": _pick(
                state_tags.get("dialog_verbosity_level"),
                state.get("dialog_verbosity_level"),
                policy_tags.get("dialog_verbosity_level"),
                trait_tags.get("dialog_verbosity_level"),
            ),
            "is_technical": _pick(
                state_tags.get("is_technical"),
                state.get("is_technical"),
                policy_tags.get("is_technical"),
                trait_tags.get("is_technical"),
            ),
            "should_ask_back": _pick(
                state_tags.get("should_ask_back"),
                state.get("should_ask_back"),
                policy_tags.get("should_ask_back"),
                trait_tags.get("should_ask_back"),
            ),
            "local_date": _pick(
                state_tags.get("local_date"),
                state.get("local_date"),
                policy_tags.get("local_date"),
                trait_tags.get("local_date"),
            ),
            "local_region": _pick(
                state_tags.get("local_region"),
                state.get("local_region"),
                policy_tags.get("local_region"),
                trait_tags.get("local_region"),
            ),
            "allowed_term": _pick(
                state_tags.get("allowed_term"),
                state.get("allowed_term"),
                policy_tags.get("allowed_term"),
                trait_tags.get("allowed_term"),
            ),
            "use_term_now": _pick(
                state_tags.get("use_term_now"),
                state.get("use_term_now"),
                policy_tags.get("use_term_now"),
                trait_tags.get("use_term_now"),
            ),
            "address_terms_policy": _pick(
                state_tags.get("address_terms_policy"),
                state.get("address_terms_policy"),
                policy_tags.get("address_terms_policy"),
                trait_tags.get("address_terms_policy"),
            ),
        }
        return {k: v for k, v in tags.items() if v}

    def _build_context_tags_block(self, tags: dict[str, str]) -> str:
        if not tags:
            return "- none"
        return "\n".join(f"- {k}: {v}" for k, v in tags.items())

    def _build_memories_block(
        self,
        retrieved_memories,
        budgets: PromptBudgets,
    ) -> tuple[str, list[dict[str, Any]], int]:
        raw_items = []
        for item in _as_list(retrieved_memories):
            row = _coerce_memory(item)
            if not row["text"] or not row["relevant"]:
                continue
            raw_items.append(row)

        raw_items.sort(
            key=lambda x: (x["priority"], x["confidence"], len(x["text"])),
            reverse=True,
        )

        selected: list[dict[str, Any]] = []
        lines: list[str] = []
        used_tokens = 0
        for row in raw_items:
            memory_text, _ = _clip_to_tokens(row["text"], budgets.memory_item_tokens)
            parts = [f"conf={row['confidence']:.2f}"]
            if row["source"]:
                parts.append(f"src={row['source']}")
            if row["topic"]:
                parts.append(f"topic={row['topic']}")
            line = f"- ({', '.join(parts)}) {memory_text}"
            line_tokens = estimate_tokens(line)
            if used_tokens + line_tokens > budgets.memory_tokens:
                continue
            used_tokens += line_tokens
            lines.append(line)
            selected.append(row)

        if not lines:
            return "- none", [], len(raw_items)
        return "\n".join(lines), selected, max(0, len(raw_items) - len(selected))

    def _build_conversation_tail_block(
        self,
        state: dict[str, Any],
        budgets: PromptBudgets,
    ) -> tuple[str, list[dict[str, str]], int]:
        source = (
            state.get("conversation_tail")
            or state.get("history")
            or state.get("messages")
            or []
        )
        parsed = [_coerce_turn(item) for item in _as_list(source)]
        parsed = [x for x in parsed if x["content"]]
        if not parsed:
            return "- none", [], 0

        tail = parsed[-budgets.tail_turns :]
        selected_rev: list[dict[str, str]] = []
        used_tokens = 0
        for row in reversed(tail):
            clipped, _ = _clip_to_tokens(row["content"], budgets.tail_turn_tokens)
            line = f"- {row['role']}: {clipped}"
            line_tokens = estimate_tokens(line)
            if used_tokens + line_tokens > budgets.tail_tokens:
                continue
            used_tokens += line_tokens
            selected_rev.append({"role": row["role"], "content": clipped})

        selected = list(reversed(selected_rev))
        if not selected:
            return "- none", [], len(tail)
        block = "\n".join(f"- {x['role']}: {x['content']}" for x in selected)
        return block, selected, max(0, len(tail) - len(selected))

    def _build_long_summary_block(
        self,
        state: dict[str, Any],
        budgets: PromptBudgets,
        *,
        dropped_tail: int,
    ) -> str:
        explicit = _normalize_text(
            state.get("long_summary")
            or state.get("rolling_summary")
            or state.get("dialog_summary")
            or state.get("short_summary")
            or "",
        )
        if explicit:
            clipped, _ = _clip_to_tokens(explicit, budgets.long_summary_tokens)
            return clipped

        history = [_coerce_turn(item) for item in _as_list(state.get("history"))]
        history = [x for x in history if x["content"]]
        if dropped_tail <= 0 or len(history) <= budgets.tail_turns:
            return "- none"

        older = history[: max(0, len(history) - budgets.tail_turns)]
        if not older:
            return "- none"
        points = []
        for row in older[:4]:
            points.append(f"- {row['role']}: {row['content']}")
        if len(older) > 4:
            points.append(f"- ... {len(older) - 4} older turns compressed")
        text = "\n".join(points)
        clipped, _ = _clip_to_tokens(text, budgets.long_summary_tokens)
        return clipped if clipped else "- none"

    def _build_output_schema_block(self, policies: dict[str, Any]) -> str:
        for key in ("output_schema", "response_schema", "tool_schema"):
            value = policies.get(key)
            if isinstance(value, (dict, list)):
                return _normalize_text(json.dumps(value, ensure_ascii=False, indent=2))
            text = _normalize_text(value)
            if text:
                return text
        return self.DEFAULT_OUTPUT_SCHEMA

    def _apply_block_budgets(
        self,
        blocks: dict[str, str],
        budgets: PromptBudgets,
    ) -> tuple[dict[str, str], dict[str, Any]]:
        limits = {
            "system_role": budgets.system_tokens,
            "persona": budgets.persona_tokens,
            "state_summary": budgets.state_tokens,
            "context_tags": budgets.tags_tokens,
            "retrieved_memories": budgets.memory_tokens,
            "long_summary": budgets.long_summary_tokens,
            "conversation_tail": budgets.tail_tokens,
            "user_message": budgets.user_tokens,
            "output_schema": budgets.output_schema_tokens,
        }
        out: dict[str, str] = {}
        cut_info: dict[str, Any] = {}
        for key, value in blocks.items():
            clipped, was_cut = _clip_to_tokens(_normalize_text(value), limits.get(key, 240))
            out[key] = clipped
            if was_cut:
                cut_info[f"{key}_trimmed"] = True
        return out, cut_info

    def _enforce_total_budget(
        self,
        blocks: dict[str, str],
        budgets: PromptBudgets,
        cut_info: dict[str, Any],
    ) -> dict[str, str]:
        out = dict(blocks)
        while self._full_token_count(out) > budgets.total_tokens:
            changed = False
            for key in self._SHRINK_ORDER:
                current = out.get(key, "")
                if not current or current == "- none":
                    continue
                shrunk = _shrink_block(current, key)
                if shrunk == current:
                    continue
                out[key] = shrunk
                cut_info[f"{key}_trimmed_total"] = True
                changed = True
                break
            if not changed:
                break
        return out

    def _render_blocks(self, blocks: dict[str, str], order: tuple[str, ...]) -> str:
        parts: list[str] = []
        for key in order:
            value = _normalize_text(blocks.get(key))
            if not value:
                continue
            parts.append(f"<<<{key.upper()}>>>\n{value}\n<<<END_{key.upper()}>>>")
        return "\n\n".join(parts).strip()

    def _full_token_count(self, blocks: dict[str, str]) -> int:
        return estimate_tokens(self._render_blocks(blocks, self._FULL_ORDER))


def _as_dict(value) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if value is None:
        return {}
    if isinstance(value, (list, tuple, set)):
        return {"items": list(value)}
    try:
        return dict(vars(value))
    except Exception:
        return {}


def _as_list(value) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _coerce_policies(policies) -> dict[str, Any]:
    if isinstance(policies, dict):
        return dict(policies)
    if isinstance(policies, (list, tuple)):
        return {"rules": list(policies)}
    text = _normalize_text(policies)
    return {"rules": [text]} if text else {}


def _to_int(value, default: int, minimum: int) -> int:
    try:
        out = int(value)
    except Exception:
        out = int(default)
    return max(minimum, out)


def _to_float(value, default: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    try:
        out = float(value)
    except Exception:
        out = float(default)
    if out < minimum:
        return float(minimum)
    if out > maximum:
        return float(maximum)
    return float(out)


def _normalize_text(value) -> str:
    text = str(value or "")
    for marker in _BOM_ARTIFACTS:
        text = text.replace(marker, "")
    text = _ZERO_WIDTH_RE.sub("", text)
    text = _CONTROL_RE.sub(" ", text)
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\xa0", " ")
    lines = []
    for raw_line in text.split("\n"):
        line = _HSPACE_RE.sub(" ", raw_line).strip()
        lines.append(line)
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _clip_to_tokens(text: str, max_tokens: int) -> tuple[str, bool]:
    normalized = _normalize_text(text)
    if not normalized:
        return "", False
    if estimate_tokens(normalized) <= max_tokens:
        return normalized, False

    max_chars = max(16, int(max_tokens) * 4)
    clipped = normalized[:max_chars]
    if len(clipped) < len(normalized):
        cut = clipped.rsplit(" ", 1)[0].strip()
        clipped = (cut if cut else clipped.strip()) + " …"
    return clipped.strip(), True


def _coerce_memory(item) -> dict[str, Any]:
    if isinstance(item, dict):
        text = _normalize_text(
            item.get("text")
            or item.get("fact")
            or item.get("content")
            or item.get("memory")
            or ""
        )
        confidence = _to_float(
            item.get("confidence", item.get("score", item.get("similarity", 0.5))),
            0.5,
            0.0,
            1.0,
        )
        priority = _to_float(item.get("priority", item.get("relevance", confidence)), 0.5, 0.0, 1.0)
        return {
            "text": text,
            "confidence": confidence,
            "priority": priority,
            "source": _normalize_text(item.get("source") or item.get("from") or ""),
            "topic": _normalize_text(item.get("topic") or ""),
            "relevant": bool(item.get("relevant", True)),
        }
    text = _normalize_text(item)
    return {
        "text": text,
        "confidence": 0.5,
        "priority": 0.5,
        "source": "",
        "topic": "",
        "relevant": bool(text),
    }


def _coerce_turn(item) -> dict[str, str]:
    role = ""
    content = ""
    if isinstance(item, dict):
        role = str(item.get("role") or item.get("speaker") or "").strip().lower()
        content = _normalize_text(item.get("content") or item.get("text") or "")
    elif isinstance(item, (list, tuple)) and len(item) >= 2:
        role = str(item[0] or "").strip().lower()
        content = _normalize_text(item[1] or "")
    else:
        content = _normalize_text(item)

    if role in {"ai", "bot"}:
        role = "assistant"
    if role not in {"user", "assistant", "system"}:
        role = "user"
    return {"role": role, "content": content}


def _shrink_block(text: str, block_name: str) -> str:
    raw = _normalize_text(text)
    if not raw:
        return raw

    lines = [line for line in raw.splitlines() if line.strip()]
    if len(lines) > 1:
        if block_name == "conversation_tail":
            lines = lines[1:]
        else:
            lines = lines[:-1]
        return "\n".join(lines).strip() if lines else "- none"

    if len(raw) <= 24:
        return ""
    cut = raw[: int(len(raw) * 0.8)].rsplit(" ", 1)[0].strip()
    return (cut + " …").strip() if cut else ""


def build_prompt(user_text: str, system_hint: str = "") -> str:
    """Backward-compatible helper kept for existing call sites."""
    hint = _normalize_text(system_hint)
    text = _normalize_text(user_text)
    return f"{hint}\n\n{text}".strip() if hint else text
