from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from modules.character.composer import CharacterComposeResult, CharacterComposer
from modules.character.evaluator import RuleEvaluator
from modules.character.storage import CharacterStorage
from utils.datetime_local import now_local_iso, now_local_ts, parse_time_to_epoch, to_local_iso


@dataclass(frozen=True)
class CharacterUpdateResult:
    character_id: str
    mood: str
    llm_profile: str
    traits: dict[str, Any]
    trait_values: dict[str, Any]
    state: dict[str, Any]
    prompt_block: str
    used_prompt_files: list[str] = field(default_factory=list)
    changes: list[dict[str, Any]] = field(default_factory=list)
    effective_traits: dict[str, float] = field(default_factory=dict)
    style_coefficients: dict[str, float] = field(default_factory=dict)


class CharacterEngine:
    def __init__(
        self,
        *,
        storage: CharacterStorage | None = None,
        evaluator: RuleEvaluator | None = None,
        composer: CharacterComposer | None = None,
    ):
        self.storage = storage or CharacterStorage()
        self.evaluator = evaluator or RuleEvaluator()
        self.composer = composer or CharacterComposer()
        self.storage.ensure_defaults()

    def list_ids(self) -> list[str]:
        return self.storage.list_character_ids(include_disabled=False)

    def get_manifest(self) -> dict[str, Any]:
        return self.storage.sync_manifest()

    def get_active_character_id(self, state: dict[str, Any] | None = None) -> str:
        state_map = dict(state or {})
        from_state = str(state_map.get("active_character_id") or "").strip().lower()
        if from_state:
            return from_state
        manifest = self.storage.load_manifest()
        from_manifest = str(manifest.get("active_character_id") or "").strip().lower()
        if from_manifest:
            return from_manifest
        ids = self.list_ids()
        return ids[0] if ids else "asya"

    def set_active_character(self, character_id: str) -> str:
        target = str(character_id or "").strip().lower()
        known = set(self.list_ids())
        if target not in known:
            raise ValueError(f"unknown character: {target}")
        manifest = self.storage.load_manifest()
        previous = str(manifest.get("active_character_id") or "").strip().lower()
        manifest["active_character_id"] = target
        self.storage.save_manifest(manifest)
        self.storage.append_manifest_event(
            {
                "type": "active_character_changed",
                "from": previous,
                "to": target,
                "source": "manual",
            }
        )
        if previous and previous != target:
            self.storage.append_event(previous, {"type": "switch_out", "source": "manual", "to": target})
        self.storage.append_event(target, {"type": "switch", "source": "manual", "target": target})
        return target

    def list_traits(self, character_id: str) -> dict[str, Any]:
        _, _, merged = self._load_traits(character_id)
        return {k: _clean_trait(v) for k, v in merged.items()}

    def set_trait(
        self,
        character_id: str,
        trait_name: str,
        *,
        value: Any,
        confidence: float = 0.8,
        trait_type: str | None = None,
    ) -> dict[str, Any]:
        cid = self._validate_character(character_id)
        builtin, _, merged = self._load_traits(cid)
        name = self._normalize_trait_name(trait_name)
        if not name:
            raise ValueError("trait name is empty")
        now = time.time()

        row = dict(merged.get(name) or {})
        ttype = str(trait_type or row.get("type") or "scalar").strip().lower()
        row["type"] = "flag" if ttype == "flag" else "scalar"
        row.setdefault("min", 0.0)
        row.setdefault("max", 1.0)
        row["confidence"] = _clamp01(confidence)
        row["updated_at"] = now_local_iso()
        row["last_used_ts"] = now_local_ts()
        row["disabled"] = False
        if row["type"] == "flag":
            row["value"] = bool(value)
        else:
            row["value"] = _clamp(_to_float(value, 0.0), _to_float(row.get("min"), 0.0), _to_float(row.get("max"), 1.0))
        merged[name] = row

        learned_delta = self._extract_learned_delta(builtin, merged)
        self.storage.save_learned_traits(cid, learned_delta)
        state = self.storage.load_state(cid)
        state["last_update_ts"] = now_local_ts()
        self._refresh_active_lists(state=state, traits=merged)
        self.storage.save_state(cid, state)
        self.storage.append_event(
            cid,
            {
                "type": "trait_set",
                "trait": name,
                "value": row.get("value"),
                "confidence": row.get("confidence"),
                "source": "manual",
            },
        )
        return _clean_trait(row)

    def remove_trait(self, character_id: str, trait_name: str) -> bool:
        cid = self._validate_character(character_id)
        builtin, _, merged = self._load_traits(cid)
        name = self._normalize_trait_name(trait_name)
        if not name or name not in merged:
            return False
        now = time.time()

        if name in builtin:
            row = dict(merged.get(name) or {})
            row["disabled"] = True
            if str(row.get("type") or "scalar").lower() == "flag":
                row["value"] = False
            else:
                row["value"] = _to_float(row.get("min"), 0.0)
            row["updated_at"] = now_local_iso()
            merged[name] = row
        else:
            merged.pop(name, None)

        learned_delta = self._extract_learned_delta(builtin, merged)
        self.storage.save_learned_traits(cid, learned_delta)
        state = self.storage.load_state(cid)
        state["last_update_ts"] = now_local_ts()
        self._refresh_active_lists(state=state, traits=merged)
        self.storage.save_state(cid, state)
        self.storage.append_event(cid, {"type": "trait_remove", "trait": name, "source": "manual"})
        return True

    def build_prompt(self, character_id: str) -> str:
        cid = self._validate_character(character_id)
        character = self.storage.load_character(cid)
        state = self.storage.load_state(cid)
        _, _, merged = self._load_traits(cid)
        context_tags = dict(state.get("context_tags") or {})
        is_technical = str(context_tags.get("is_technical") or "").strip().lower() in {"1", "true", "yes", "on"}
        composed = self.composer.compose(
            storage=self.storage,
            character_id=cid,
            character=character,
            state=state,
            traits=merged,
            dialog_mode=dict(state.get("dialog_mode") or {}),
            context_meta={
                "intent": str(context_tags.get("intent") or ""),
                "is_technical": is_technical,
                "active_mode": str(context_tags.get("active_mode") or state.get("mode") or "chatting"),
            },
        )
        return composed.prompt

    def update(
        self,
        *,
        text: str,
        meta: dict[str, Any] | None,
        active_character_id: str | None = None,
    ) -> CharacterUpdateResult:
        meta_map = dict(meta or {})
        cid = self._validate_character(active_character_id or self.get_active_character_id(meta_map))
        character = self.storage.load_character(cid)
        state = self.storage.load_state(cid)
        builtin, _, merged = self._load_traits(cid)
        rules_payload = self.storage.load_rules(cid)

        now = time.time()
        changes: list[dict[str, Any]] = []
        self._apply_decay(merged, now=now, changes=changes)

        ctx = self._build_context(text=text, meta=meta_map, state=state)
        for rule in list(rules_payload.get("rules") or []):
            if not isinstance(rule, dict):
                continue
            when = dict(rule.get("when") or {})
            if not self.evaluator.matches(when, ctx=ctx, traits=merged):
                continue
            self._apply_actions(
                merged=merged,
                state=state,
                actions=list(rule.get("apply") or []),
                now=now,
                changes=changes,
            )

        self._resolve_conflicts(merged=merged, state=state, now=now, changes=changes)
        self._refresh_active_lists(state=state, traits=merged)
        self._apply_cleanup(
            builtin=builtin,
            merged=merged,
            state=state,
            rules=rules_payload,
            now=now,
            changes=changes,
        )
        self._refresh_active_lists(state=state, traits=merged)
        state["last_update_ts"] = now_local_ts()
        if not str(state.get("mood") or "").strip():
            state["mood"] = str(character.get("default_mood") or "thoughtful")

        learned_delta = self._extract_learned_delta(builtin, merged)
        self.storage.save_learned_traits(cid, learned_delta)
        self.storage.save_state(cid, state)

        composed = self.composer.compose(
            storage=self.storage,
            character_id=cid,
            character=character,
            state=state,
            traits=merged,
            context_meta={"active_mode": str(meta_map.get("active_mode") or meta_map.get("mode") or "chatting")},
        )
        self.storage.append_event(
            cid,
            {
                "type": "character_update",
                "intent": ctx.get("intent"),
                "emotion": ctx.get("emotion"),
                "mode": ctx.get("mode"),
                "mood": state.get("mood"),
                "changes": changes[:16],
            },
        )

        trait_values = self._flat_trait_values(merged)
        llm_profile = str(character.get("llm_profile") or "BALANCED").strip().upper() or "BALANCED"
        return CharacterUpdateResult(
            character_id=cid,
            mood=str(composed.mood or state.get("mood") or "thoughtful"),
            llm_profile=llm_profile,
            traits={k: _clean_trait(v) for k, v in merged.items()},
            trait_values=trait_values,
            state=dict(state),
            prompt_block=composed.prompt,
            used_prompt_files=list(composed.used_files),
            changes=changes,
            effective_traits=dict(composed.effective_traits or {}),
            style_coefficients=dict(composed.style_coefficients or {}),
        )

    def _validate_character(self, character_id: str) -> str:
        cid = str(character_id or "").strip().lower()
        ids = set(self.list_ids())
        if cid in ids:
            return cid
        if ids:
            return sorted(ids)[0]
        return "asya"

    def _load_traits(self, character_id: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        builtin = dict(self.storage.load_builtin_traits(character_id))
        learned = dict(self.storage.load_learned_traits(character_id))
        merged: dict[str, Any] = {}

        for name, row in builtin.items():
            key = self._normalize_trait_name(name)
            if not key:
                continue
            payload = _normalize_trait(row)
            payload["_source"] = "builtin"
            merged[key] = payload
        for name, row in learned.items():
            key = self._normalize_trait_name(name)
            if not key:
                continue
            payload = _normalize_trait(row)
            existing = dict(merged.get(key) or {})
            existing.update(payload)
            existing["_source"] = "learned"
            merged[key] = existing

        return builtin, learned, merged

    @staticmethod
    def _normalize_trait_name(value: str) -> str:
        raw = str(value or "").strip().lower()
        if not raw:
            return ""
        return "".join(ch for ch in raw if ch.isalnum() or ch in {"_", "-"})

    def _build_context(self, *, text: str, meta: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
        tags = list(meta.get("metadata_tags") or meta.get("tags") or [])
        return {
            "text": str(text or ""),
            "intent": str(meta.get("intent") or "").strip().lower(),
            "emotion": str(meta.get("mood") or meta.get("emotion") or "").strip().lower(),
            "mode": str(meta.get("mode") or state.get("mode") or "chat").strip().lower(),
            "topic": str(meta.get("topic") or "").strip().lower(),
            "tags": [str(x).strip().lower() for x in tags if str(x).strip()],
        }

    def _apply_actions(
        self,
        *,
        merged: dict[str, Any],
        state: dict[str, Any],
        actions: list[Any],
        now: float,
        changes: list[dict[str, Any]],
    ) -> None:
        for row in list(actions or []):
            if not isinstance(row, dict):
                continue
            if "set_mood" in row:
                mood = str(row.get("set_mood") or "").strip().lower()
                if mood:
                    prev = str(state.get("mood") or "")
                    state["mood"] = mood
                    if prev != mood:
                        changes.append({"kind": "mood", "from": prev, "to": mood, "source": "rule"})
                continue

            trait_name = self._normalize_trait_name(row.get("trait"))
            if not trait_name:
                continue
            op = str(row.get("op") or "add").strip().lower()
            value = row.get("value", 0.0)
            ttype = str(row.get("type") or "").strip().lower()

            trait = dict(merged.get(trait_name) or {})
            if not trait:
                trait = _normalize_trait(
                    {
                        "type": ("flag" if ttype == "flag" else "scalar"),
                        "value": (False if ttype == "flag" else 0.0),
                        "min": 0.0,
                        "max": 1.0,
                        "confidence": 0.45,
                    }
                )
                trait["_source"] = "learned"

            trait_type = str(trait.get("type") or "scalar").strip().lower()
            before = trait.get("value")
            if op == "remove":
                trait["disabled"] = True
                trait["value"] = False if trait_type == "flag" else _to_float(trait.get("min"), 0.0)
            elif trait_type == "flag":
                if op == "toggle":
                    trait["value"] = not bool(trait.get("value"))
                elif op in {"set", "add", "update"}:
                    trait["value"] = bool(value)
                elif op == "mul":
                    trait["value"] = bool(trait.get("value")) and bool(value)
                trait["disabled"] = False
            else:
                cur = _to_float(trait.get("value"), 0.0)
                min_v = _to_float(trait.get("min"), 0.0)
                max_v = _to_float(trait.get("max"), 1.0)
                val = _to_float(value, 0.0)
                if op in {"set", "update"}:
                    nxt = val
                elif op == "mul":
                    nxt = cur * val
                else:
                    nxt = cur + val
                trait["value"] = _clamp(nxt, min_v, max_v)
                trait["disabled"] = False

            trait["confidence"] = _clamp01(_to_float(trait.get("confidence"), 0.5) + 0.03)
            trait["updated_at"] = now_local_iso()
            trait["last_used_ts"] = now_local_ts()
            merged[trait_name] = trait
            if before != trait.get("value"):
                change_kind = _trait_change_kind(trait_name)
                changes.append(
                    {
                        "kind": change_kind,
                        "trait": trait_name,
                        "op": op,
                        "from": before,
                        "to": trait.get("value"),
                        "source": "rule",
                    }
                )

    def _apply_decay(self, merged: dict[str, Any], *, now: float, changes: list[dict[str, Any]]) -> None:
        for name, row in list(merged.items()):
            trait = dict(row or {})
            if str(trait.get("type") or "scalar").strip().lower() != "scalar":
                continue
            decay = _to_float(trait.get("decay_per_day"), 0.0)
            if decay <= 0:
                continue
            last_ts = _to_float(trait.get("decay_ts"), _to_float(trait.get("updated_at"), now))
            if last_ts <= 0:
                last_ts = now
            days = max(0.0, (now - last_ts) / 86400.0)
            if days < 0.02:
                continue
            cur = _to_float(trait.get("value"), 0.0)
            min_v = _to_float(trait.get("min"), 0.0)
            max_v = _to_float(trait.get("max"), 1.0)
            nxt = _clamp(cur - (decay * days), min_v, max_v)
            if abs(nxt - cur) < 1e-6:
                trait["decay_ts"] = now_local_ts()
                merged[name] = trait
                continue
            trait["value"] = nxt
            trait["decay_ts"] = now_local_ts()
            merged[name] = trait
            changes.append({"kind": "decay", "trait": name, "from": cur, "to": nxt})

    def _resolve_conflicts(self, *, merged: dict[str, Any], state: dict[str, Any], now: float, changes: list[dict[str, Any]]) -> None:
        sarcasm = _trait_scalar(merged, "sarcasm")
        romance = _trait_scalar(merged, "romance")
        if sarcasm > 0.72 and romance > 0.68:
            row = dict(merged.get("sarcasm") or {})
            before = _to_float(row.get("value"), sarcasm)
            row["value"] = _clamp(before - 0.12, _to_float(row.get("min"), 0.0), _to_float(row.get("max"), 1.0))
            row["updated_at"] = now_local_iso()
            row["last_used_ts"] = now_local_ts()
            row["_source"] = "learned"
            merged["sarcasm"] = row
            changes.append({"kind": "conflict", "trait": "sarcasm", "from": before, "to": row.get("value"), "reason": "romance_vs_sarcasm"})

        mood = str(state.get("mood") or "").strip().lower()
        if mood == "focused":
            play = _trait_scalar(merged, "playfulness")
            if play > 0.55:
                row = dict(merged.get("playfulness") or {})
                before = _to_float(row.get("value"), play)
                row["value"] = _clamp(before - 0.1, _to_float(row.get("min"), 0.0), _to_float(row.get("max"), 1.0))
                row["updated_at"] = now_local_iso()
                row["last_used_ts"] = now_local_ts()
                row["_source"] = "learned"
                merged["playfulness"] = row
                changes.append({"kind": "conflict", "trait": "playfulness", "from": before, "to": row.get("value"), "reason": "focused_mode"})

    def _apply_cleanup(
        self,
        *,
        builtin: dict[str, Any],
        merged: dict[str, Any],
        state: dict[str, Any],
        rules: dict[str, Any],
        now: float,
        changes: list[dict[str, Any]],
    ) -> None:
        cleanup = dict(rules.get("cleanup") or {})
        conf_threshold = _to_float(cleanup.get("remove_if_confidence_below"), 0.25)
        unused_days = max(1.0, _to_float(cleanup.get("remove_if_unused_days"), 45.0))
        remove_after_sec = unused_days * 86400.0

        for name in list(merged.keys()):
            row = dict(merged.get(name) or {})
            is_builtin = name in builtin
            conf = _to_float(row.get("confidence"), 0.0)
            last_ts = max(_to_float(row.get("last_used_ts"), 0.0), _to_float(row.get("updated_at"), 0.0))
            stale = (now - last_ts) > remove_after_sec if last_ts > 0 else False
            ttl_days = _to_float(row.get("ttl_days"), 0.0)
            ttl_expired = False
            if ttl_days > 0 and last_ts > 0:
                ttl_expired = (now - last_ts) > (ttl_days * 86400.0)

            remove = False
            if conf < conf_threshold and not is_builtin:
                remove = True
            if stale and not is_builtin:
                remove = True
            if ttl_expired:
                remove = True

            if remove:
                merged.pop(name, None)
                changes.append({"kind": "cleanup", "trait": name, "reason": "confidence_or_ttl"})

        counters = dict(state.get("counters") or {})
        counters.setdefault("banter_hits", 0)
        counters.setdefault("comfort_hits", 0)
        state["counters"] = counters

    def _refresh_active_lists(self, *, state: dict[str, Any], traits: dict[str, Any]) -> None:
        active: list[str] = []
        disabled: list[str] = []
        for name, row in list(traits.items()):
            trait = dict(row or {})
            if bool(trait.get("disabled", False)):
                disabled.append(name)
                continue
            ttype = str(trait.get("type") or "scalar").strip().lower()
            value = trait.get("value")
            if ttype == "flag":
                if bool(value):
                    active.append(name)
            else:
                if _to_float(value, 0.0) > 0.05:
                    active.append(name)
        state["active_traits"] = sorted(set(active))
        state["disabled_traits"] = sorted(set(disabled))

    def _extract_learned_delta(self, builtin: dict[str, Any], merged: dict[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for name, row in list(merged.items()):
            clean = _clean_trait(row)
            base = dict(builtin.get(name) or {})
            if name not in builtin:
                out[name] = clean
                continue
            if _has_override(base, clean):
                out[name] = clean
        return out

    @staticmethod
    def _flat_trait_values(traits: dict[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for name, row in dict(traits or {}).items():
            if str(name).startswith("_"):
                continue
            payload = dict(row or {})
            value = payload.get("value")
            out[str(name)] = bool(value) if str(payload.get("type") or "").lower() == "flag" else _to_float(value, 0.0)
        return out


def _normalize_trait(value: dict[str, Any] | None) -> dict[str, Any]:
    row = dict(value or {})
    ttype = str(row.get("type") or "scalar").strip().lower()
    out = {
        "type": "flag" if ttype == "flag" else "scalar",
        "value": row.get("value", False if ttype == "flag" else 0.0),
        "min": _to_float(row.get("min"), 0.0),
        "max": _to_float(row.get("max"), 1.0),
        "decay_per_day": max(0.0, _to_float(row.get("decay_per_day"), 0.0)),
        "confidence": _clamp01(_to_float(row.get("confidence"), 0.6)),
        "tags": [str(x).strip().lower() for x in list(row.get("tags") or []) if str(x).strip()],
        "prompt_file": str(row.get("prompt_file") or "").strip(),
        "ttl_days": max(0.0, _to_float(row.get("ttl_days"), 0.0)),
        "disabled": bool(row.get("disabled", False)),
        "updated_at": str(row.get("updated_at") or ""),
        "last_used_ts": to_local_iso(row.get("last_used_ts"), default=""),
        "decay_ts": to_local_iso(row.get("decay_ts"), default=""),
    }
    if out["type"] == "flag":
        out["value"] = bool(out["value"])
    else:
        out["value"] = _clamp(_to_float(out["value"], 0.0), out["min"], out["max"])
    return out


def _clean_trait(value: dict[str, Any]) -> dict[str, Any]:
    row = dict(value or {})
    row.pop("_source", None)
    return row


def _has_override(base: dict[str, Any], current: dict[str, Any]) -> bool:
    base_n = _normalize_trait(base)
    cur_n = _normalize_trait(current)
    for key in ("type", "value", "disabled", "prompt_file"):
        if base_n.get(key) != cur_n.get(key):
            return True
    if abs(_to_float(base_n.get("confidence"), 0.0) - _to_float(cur_n.get("confidence"), 0.0)) >= 0.08:
        return True
    return False


def _trait_scalar(traits: dict[str, Any], name: str) -> float:
    row = dict(traits.get(str(name).strip().lower()) or {})
    value = row.get("value")
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    return _to_float(value, 0.0)


def _to_float(value, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return parse_time_to_epoch(value, float(default))


def _clamp(value: float, minimum: float, maximum: float) -> float:
    lo = float(min(minimum, maximum))
    hi = float(max(minimum, maximum))
    return max(lo, min(hi, float(value)))


def _clamp01(value: float) -> float:
    return _clamp(float(value), 0.0, 1.0)


def _trait_change_kind(trait_name: str) -> str:
    key = str(trait_name or "").strip().lower()
    if key in {"playfulness", "thoughtfulness"}:
        return "derived_axis"
    return "trait"

