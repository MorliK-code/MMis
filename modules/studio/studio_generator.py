from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from config.settings import DATA_DIR, get_model_profiles
from llm.provider_base import LLMProviderBase, LLMRequest, Message
from llm.task_router import run_task_model_json
from modules.character.mode_profile import resolve_mode_profile
from modules.character.storage import CharacterStorage
from utils.datetime_local import now_local_iso


@dataclass(frozen=True)
class StudioGeneratorReply:
    text: str
    state: dict[str, Any] = field(default_factory=dict)
    done: bool = False
    error: str = ""
    payload: dict[str, Any] = field(default_factory=dict)
    memory_ops: list[dict[str, Any]] = field(default_factory=list)


class StudioGenerator:
    KEY = "studio_generator"

    PHASE_SEED = "seed_capture"
    PHASE_CLARIFY = "clarify"
    PHASE_REVIEW = "review"
    PHASE_DONE = "done"

    OP_CREATE_CHARACTER = "create_character"
    OP_UPDATE_CHARACTER = "update_character"
    OP_UPDATE_MODES = "update_modes"
    OP_MIXED = "mixed"
    OP_BUILD_PACK = "build_character_pack"

    VALID_OPERATIONS = {
        OP_CREATE_CHARACTER,
        OP_UPDATE_CHARACTER,
        OP_UPDATE_MODES,
        OP_MIXED,
        OP_BUILD_PACK,
    }
    VALID_SCOPE = {"global", "character", "mixed"}
    VALID_VIBES = {"balanced", "playful", "strict", "soft"}
    DEFAULT_LLM_PROFILE = "BALANCED"
    VALID_MODE_ACTIONS = {"add", "update", "remove"}
    CONFIDENCE_THRESHOLD = 0.75
    MAX_OPTIONS_RETRIES = 3

    def __init__(
        self,
        *,
        storage: CharacterStorage | None = None,
        specs_root: str | Path | None = None,
        character_specs_root: str | Path | None = None,
    ) -> None:
        self.storage = storage or CharacterStorage()
        self.specs_root = (
            Path(specs_root).expanduser().resolve()
            if specs_root is not None
            else (DATA_DIR / "specs" / "rules_for_all").resolve()
        )
        self.specs_root.mkdir(parents=True, exist_ok=True)
        self.default_character_specs_root = (DATA_DIR / "specs" / "characters").resolve()
        self.character_specs_root = (
            Path(character_specs_root).expanduser().resolve()
            if character_specs_root is not None
            else self.default_character_specs_root
        )
        self.character_specs_root.mkdir(parents=True, exist_ok=True)
        self._llm_profiles_cache: list[str] = []

    def is_active(self, state: dict[str, Any]) -> bool:
        row = self._state_from(state)
        return bool(row.get("active", False))

    def status(
        self,
        state: dict[str, Any],
        *,
        provider: LLMProviderBase | None = None,
        model: str = "",
        command_alias: str = "studio",
    ) -> StudioGeneratorReply:
        row = self._state_from(state)
        row["command_alias"] = self._normalize_command_alias(command_alias, fallback=row.get("command_alias"))
        cmd = self._cmd(row)
        if not bool(row.get("active", False)):
            return StudioGeneratorReply(
                text=(
                    "Studio выключен.\n"
                    "Команды:\n"
                    f"- /{cmd} start [seed]\n"
                    f"- /{cmd} status\n"
                    f"- /{cmd} apply\n"
                    f"- /{cmd} cancel"
                ),
                state=row,
            )

        phase = str(row.get("phase") or self.PHASE_SEED)
        op = str(row.get("operation_type") or "(не определён)")
        targets = dict(row.get("targets") or {})
        unresolved = list(row.get("unresolved_fields") or [])
        if phase == self.PHASE_CLARIFY:
            cur = self._current_field(row)
            q = self._question_for_field(field_id=cur, row=row, provider=provider, model=model) if cur else {}
            current = str(q.get("prompt") or cur or "(ожидание)")
        elif phase == self.PHASE_REVIEW:
            current = "review"
        else:
            current = "seed_capture"
        return StudioGeneratorReply(
            text=(
                "Studio активен.\n"
                f"- фаза: {phase}\n"
                f"- operation_type: {op}\n"
                f"- targets.characters: {', '.join(targets.get('character_ids') or []) or '(none)'}\n"
                f"- targets.modes: {', '.join(targets.get('mode_ids') or []) or '(none)'}\n"
                f"- targets.scope: {str(targets.get('scope') or 'mixed')}\n"
                f"- текущий вопрос: {current}\n"
                f"- осталось уточнить: {len(unresolved)}\n"
                f"Если всё ок: /{cmd} apply"
            ),
            state=row,
        )

    def start(
        self,
        state: dict[str, Any],
        seed: str = "",
        *,
        provider: LLMProviderBase | None = None,
        model: str = "",
        command_alias: str = "studio",
    ) -> StudioGeneratorReply:
        _ = state
        row = self._empty_state()
        row["active"] = True
        row["phase"] = self.PHASE_SEED
        row["command_alias"] = self._normalize_command_alias(command_alias)

        seed_text = str(seed or "").strip()
        if seed_text:
            row["seed_prompt"] = seed_text
            self._bootstrap_from_seed(row=row, provider=provider, model=model)
            if str(row.get("phase") or "") == self.PHASE_REVIEW:
                return self._render_review(row)
            return self._ask_current(row=row, provider=provider, model=model)

        return StudioGeneratorReply(
            text=(
                "Шаг 1/4. Опиши свободно, что нужно сделать в studio.\n"
                "Поддерживаются сценарии: создать персонажа, обновить персонажа, добавить/обновить mode, смешанный запрос."
            ),
            state=row,
        )

    def cancel(self, state: dict[str, Any], *, command_alias: str = "studio") -> StudioGeneratorReply:
        _ = state
        row = self._empty_state()
        row["active"] = False
        row["phase"] = self.PHASE_DONE
        row["command_alias"] = self._normalize_command_alias(command_alias)
        return StudioGeneratorReply(
            text="Studio остановлен.",
            state=row,
            done=True,
            payload={"branch": "studio", "action": "cancel", "done": True},
        )

    def ingest(
        self,
        state: dict[str, Any],
        user_text: str,
        *,
        provider: LLMProviderBase | None = None,
        model: str = "",
        command_alias: str = "",
    ) -> StudioGeneratorReply:
        row = self._state_from(state)
        if command_alias:
            row["command_alias"] = self._normalize_command_alias(command_alias, fallback=row.get("command_alias"))
        if not bool(row.get("active", False)):
            return StudioGeneratorReply(text=f"Wizard не запущен. Используй /{self._cmd(row)} start", state=row, error="wizard_inactive")

        phase = str(row.get("phase") or self.PHASE_SEED)
        src = str(user_text or "").strip()

        if phase == self.PHASE_SEED:
            if not src:
                return StudioGeneratorReply(
                    text="Нужен seed-запрос. Напиши свободно: создать/обновить персонажа и/или modes.",
                    state=row,
                )
            row["seed_prompt"] = src
            self._bootstrap_from_seed(row=row, provider=provider, model=model)
            if str(row.get("phase") or "") == self.PHASE_REVIEW:
                return self._render_review(row)
            return self._ask_current(row=row, provider=provider, model=model)

        if phase == self.PHASE_CLARIFY:
            return self._handle_clarify_turn(row=row, src=src, provider=provider, model=model)

        if phase == self.PHASE_REVIEW:
            edit = self._parse_review_edit(src, row=row)
            if edit is None:
                cmd = self._cmd(row)
                return StudioGeneratorReply(
                    text=(
                        "Review уже готов.\n"
                        f"Если всё ок: /{cmd} apply\n"
                        f"Отмена: /{cmd} cancel\n"
                        "Или поправь поле: <field>: <value>"
                    ),
                    state=row,
                )
            field_id, parsed = edit
            if not self._apply_answer_to_state(row=row, field_id=field_id, parsed=parsed):
                return StudioGeneratorReply(text="Не удалось применить изменение поля.", state=row)
            conf = dict(row.get("confidence_map") or {})
            conf[field_id] = 1.0
            row["confidence_map"] = conf
            row["updated_at"] = now_local_iso()
            self._refresh_unresolved(row)
            if str(row.get("phase") or "") == self.PHASE_REVIEW:
                return self._render_review(row)
            return self._ask_current(row=row, provider=provider, model=model)

        row["phase"] = self.PHASE_SEED
        return StudioGeneratorReply(text="Состояние студии было неконсистентным. Начнём снова: опиши задачу свободно.", state=row)

    def apply(
        self,
        state: dict[str, Any],
        *,
        provider: LLMProviderBase | None = None,
        model: str = "",
        command_alias: str = "",
    ) -> StudioGeneratorReply:
        row = self._state_from(state)
        if command_alias:
            row["command_alias"] = self._normalize_command_alias(command_alias, fallback=row.get("command_alias"))
        if not bool(row.get("active", False)):
            return StudioGeneratorReply(text=f"Wizard не запущен. Используй /{self._cmd(row)} start", state=row, error="wizard_inactive")

        self._refresh_unresolved(row)
        unresolved = list(row.get("unresolved_fields") or [])
        if unresolved:
            row["phase"] = self.PHASE_CLARIFY
            row["current_field"] = str(unresolved[0] if unresolved else "")
            row["clarify_index"] = 0
            return StudioGeneratorReply(
                text=(
                    "Нужно завершить уточнения перед применением: "
                    + ", ".join(unresolved)
                    + f". Продолжи диалог, затем /{self._cmd(row)} apply"
                ),
                state=row,
                error="missing_fields",
            )

        try:
            result = self._apply_draft_changes(row=row, provider=provider, model=model)
        except Exception as exc:
            return StudioGeneratorReply(
                text=f"Ошибка применения studio: {type(exc).__name__}",
                state=row,
                error=f"{type(exc).__name__}:{exc}",
            )

        done_state = self._empty_state()
        done_state["active"] = False
        done_state["phase"] = self.PHASE_DONE
        done_state["command_alias"] = self._cmd(row)
        done_state["last_result"] = result

        ops: list[dict[str, Any]] = []
        set_active_cid = str(result.get("set_active_character") or "").strip().lower()
        if set_active_cid:
            ops.append({"op": "state_character", "value": set_active_cid, "locked": True, "reason": "studio_apply"})
            ops.append({"op": "state_personality", "value": set_active_cid, "locked": True, "reason": "studio_apply"})

        return StudioGeneratorReply(
            text=(
                "Готово. Studio обновлен.\n"
                f"- operation_type: {result.get('operation_type')}\n"
                f"- targets.characters: {', '.join(result.get('targets', {}).get('character_ids') or []) or '(none)'}\n"
                f"- targets.modes: {', '.join(result.get('targets', {}).get('mode_ids') or []) or '(none)'}\n"
                f"- files: {', '.join(result.get('written_files') or []) or '(none)'}\n"
                f"- blueprints: {len(result.get('blueprint_files') or [])}\n"
                f"- artifacts: {len(result.get('artifact_files') or [])}\n"
                f"- planned_changes: {len(result.get('planned_changes') or [])}"
            ),
            state=done_state,
            done=True,
            payload={
                "branch": "studio",
                "action": "apply",
                "done": True,
                "operation_type": str(result.get("operation_type") or ""),
                "targets": dict(result.get("targets") or {}),
                "result": dict(result or {}),
            },
            memory_ops=ops,
        )

    def _handle_clarify_turn(
        self,
        *,
        row: dict[str, Any],
        src: str,
        provider: LLMProviderBase | None,
        model: str,
    ) -> StudioGeneratorReply:
        field_id = self._current_field(row)
        if not field_id:
            self._refresh_unresolved(row)
            field_id = self._current_field(row)
            if not field_id:
                row["phase"] = self.PHASE_REVIEW
                return self._render_review(row)

        question = self._question_for_field(field_id=field_id, row=row, provider=provider, model=model)
        if str(row.get("awaiting_custom_for") or "") == field_id:
            parsed = self._parse_answer(question_id=field_id, text=src, allow_numbered=False)
            if parsed is None:
                return StudioGeneratorReply(text="Введи свой вариант обычным текстом.", state=row)
            row["awaiting_custom_for"] = ""
        else:
            parsed = self._parse_answer(question_id=field_id, text=src, allow_numbered=True, row=row)
            if parsed == "__pick_custom__":
                row["awaiting_custom_for"] = field_id
                return StudioGeneratorReply(text="Ок, введи свой вариант текстом (не номером).", state=row)

        if parsed is None:
            return StudioGeneratorReply(
                text=(
                    "Не смогла распознать ответ.\n"
                    f"{str(question.get('prompt') or '')}\n"
                    f"{self._render_options(question)}\n"
                    f"{self._render_impact(question)}"
                ),
                state=row,
            )

        if not self._apply_answer_to_state(row=row, field_id=field_id, parsed=parsed):
            return StudioGeneratorReply(text="Значение не удалось применить. Попробуй другой вариант.", state=row)

        conf = dict(row.get("confidence_map") or {})
        conf[field_id] = 1.0
        row["confidence_map"] = conf
        row["updated_at"] = now_local_iso()
        self._refresh_unresolved(row)
        if str(row.get("phase") or "") == self.PHASE_REVIEW:
            return self._render_review(row)
        return self._ask_current(row=row, provider=provider, model=model)

    def _bootstrap_from_seed(
        self,
        *,
        row: dict[str, Any],
        provider: LLMProviderBase | None,
        model: str,
    ) -> None:
        seed = str(row.get("seed_prompt") or "").strip()
        existing_chars = self._known_character_ids()
        known_modes = self._known_modes()

        merged, llm_conf, llm_meta, character_action_intent = self._bootstrap_extract_intent(
            seed=seed,
            provider=provider,
            model=model,
            existing_chars=existing_chars,
            known_modes=known_modes,
        )
        row["last_llm"] = dict(llm_meta or {})

        op = self._coerce_field_value(field_id="operation_type", value=merged.get("operation_type"))
        if not op:
            op = self._infer_operation_type(seed=seed)
        if op in {self.OP_CREATE_CHARACTER, self.OP_UPDATE_CHARACTER}:
            if character_action_intent == "create":
                op = self.OP_CREATE_CHARACTER
            elif character_action_intent == "update":
                op = self.OP_UPDATE_CHARACTER
        row["operation_type"] = str(op or self.OP_CREATE_CHARACTER)

        targets_map, target_candidates = self._bootstrap_resolve_targets(
            seed=seed,
            operation_type=str(row.get("operation_type") or ""),
            raw_targets=dict(merged.get("targets") or {}),
            existing_chars=existing_chars,
            known_modes=known_modes,
        )
        row["targets"] = dict(targets_map)
        row["target_candidates"] = dict(target_candidates)

        draft = self._bootstrap_hydrate_draft(
            seed=seed,
            operation_type=str(row.get("operation_type") or ""),
            targets=targets_map,
            merged=merged,
            existing_chars=existing_chars,
            known_modes=known_modes,
        )
        row["targets"] = dict(targets_map)
        row["draft_changes"] = dict(draft)
        row["data"] = self._snapshot_data_from_draft(row)

        self._bootstrap_refresh_unresolved(
            row=row,
            seed=seed,
            llm_conf=llm_conf,
            character_action_intent=character_action_intent,
        )

    def _bootstrap_extract_intent(
        self,
        *,
        seed: str,
        provider: LLMProviderBase | None,
        model: str,
        existing_chars: list[str],
        known_modes: list[str],
    ) -> tuple[dict[str, Any], dict[str, float], dict[str, Any], str]:
        llm_payload: dict[str, Any] = {}
        llm_conf: dict[str, float] = {}
        llm_meta: dict[str, Any] = {}
        if seed and provider is not None:
            llm_payload, llm_conf, llm_meta = self._extract_seed_with_llm(seed=seed, provider=provider, model=model)
        heuristic = self._extract_seed_heuristic(seed=seed, existing_chars=existing_chars, known_modes=known_modes)
        merged = self._merge_seed_payloads(primary=heuristic, secondary=llm_payload)
        character_action_intent = self._infer_character_action_intent(seed=seed)
        return merged, llm_conf, llm_meta, character_action_intent

    def _bootstrap_resolve_targets(
        self,
        *,
        seed: str,
        operation_type: str,
        raw_targets: dict[str, Any],
        existing_chars: list[str],
        known_modes: list[str],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        target_chars = [self._safe_id(x) for x in list(raw_targets.get("character_ids") or []) if self._safe_id(x)]
        target_modes = [self._mode_id(x) for x in list(raw_targets.get("mode_ids") or []) if self._mode_id(x)]
        scope = str(raw_targets.get("scope") or "").strip().lower()
        if scope not in self.VALID_SCOPE:
            if operation_type == self.OP_UPDATE_MODES:
                scope = "global"
            elif operation_type == self.OP_MIXED:
                scope = "mixed"
            else:
                scope = "character"

        resolved_chars, char_candidates, missing_update_chars = self._resolve_character_targets(
            requested=target_chars,
            operation_type=operation_type,
            existing=existing_chars,
        )
        resolved_modes = self._resolve_mode_targets(requested=target_modes, known_modes=known_modes)
        scope = self._normalize_scope_from_seed(
            seed=seed,
            operation_type=operation_type,
            requested_chars=target_chars,
            resolved_chars=resolved_chars,
            scope=scope,
        )
        targets = {
            "character_ids": list(resolved_chars),
            "mode_ids": list(resolved_modes),
            "scope": scope,
        }
        candidates = {
            "character_ids": list(char_candidates),
            "mode_ids": [],
            "missing_update_characters": list(missing_update_chars),
        }
        return targets, candidates

    def _bootstrap_hydrate_draft(
        self,
        *,
        seed: str,
        operation_type: str,
        targets: dict[str, Any],
        merged: dict[str, Any],
        existing_chars: list[str],
        known_modes: list[str],
    ) -> dict[str, Any]:
        draft = self._empty_draft_changes()
        character_block = dict(merged.get("character") or {})
        mode_changes = [dict(x) for x in list(merged.get("mode_changes") or []) if isinstance(x, dict)]

        self._hydrate_character_draft(
            draft=draft,
            operation_type=operation_type,
            targets=targets,
            character_payload=character_block,
            seed=seed,
            existing_chars=existing_chars,
        )
        self._hydrate_modes_draft(
            draft=draft,
            operation_type=operation_type,
            targets=targets,
            mode_payloads=mode_changes,
            seed=seed,
            known_modes=known_modes,
        )
        self._ensure_mode_scope_character_targets(
            draft=draft,
            operation_type=operation_type,
            targets=targets,
        )
        return draft

    def _bootstrap_refresh_unresolved(
        self,
        *,
        row: dict[str, Any],
        seed: str,
        llm_conf: dict[str, float],
        character_action_intent: str,
    ) -> None:
        conf = self._sanitize_confidence_map(llm_conf)
        defaults = {
            "operation_type": 0.78,
            "target_character_id": 0.72,
            "target_mode_id": 0.72,
            "display_name": 0.76,
            "default_mode": 0.76,
            "llm_profile": 0.74,
            "mode_scope": 0.74,
        }
        for key, value in defaults.items():
            conf.setdefault(key, value)
        if row["operation_type"] in {self.OP_CREATE_CHARACTER, self.OP_UPDATE_CHARACTER}:
            if character_action_intent in {"create", "update"}:
                conf["operation_type"] = 1.0
            elif character_action_intent == "unknown" and self._needs_operation_type_clarify(
                seed=seed,
                operation_type=str(row.get("operation_type") or ""),
            ):
                conf["operation_type"] = min(self._clamp_float(conf.get("operation_type"), default=0.78), 0.45)
        row["confidence_map"] = conf
        row["updated_at"] = now_local_iso()
        self._refresh_unresolved(row)

    def _extract_seed_with_llm(
        self,
        *,
        seed: str,
        provider: LLMProviderBase,
        model: str,
    ) -> tuple[dict[str, Any], dict[str, float], dict[str, Any]]:
        known_modes = self._known_modes()
        known_chars = self._known_character_ids()
        llm_profiles_enum = self._llm_profile_enum_for_prompt()
        system_prompt = (
            "Ты анализируешь запрос для Specs Studio. "
            "Верни только JSON. Без комментариев. "
            "Формат: "
            "{"
            "\"operation_type\":\"create_character|update_character|update_modes|mixed|build_character_pack\","
            "\"targets\":{\"character_ids\":[],\"mode_ids\":[],\"scope\":\"global|character|mixed\"},"
            "\"character\":{"
            "\"character_id\":\"\","
            "\"display_name\":\"\","
            "\"vibe\":\"balanced|playful|strict|soft\","
            "\"technicality\":0.0,"
            "\"energy\":0.0,"
            "\"default_mode\":\"\","
            "\"extra_modes\":[],"
            f"\"llm_profile\":\"{llm_profiles_enum}\","
            "\"set_active\":true"
            "},"
            "\"mode_changes\":[{"
            "\"mode_id\":\"\","
            "\"action\":\"add|update|remove\","
            "\"description\":\"\","
            "\"legacy_mode\":\"\","
            "\"show_parameters\":false,"
            "\"show_summary\":false"
            "}],"
            "\"confidence\":{\"field\":0..1}"
            "}"
        )
        user_prompt = (
            f"known_character_ids={json.dumps(known_chars, ensure_ascii=False)}\n"
            f"known_modes={json.dumps(known_modes, ensure_ascii=False)}\n"
            f"seed={seed}"
        )
        try:
            result = run_task_model_json(
                "studio_seed_extract",
                user_prompt,
                system_prompt=system_prompt,
                metadata={"think": True, "studio_specs_task": "seed_extract"},
                required_fields=("operation_type",),
                max_output_chars=8000,
                max_retries=1,
            )
            payload = self._coerce_json_object(result.json_payload)
            raw_data = dict(payload.get("data") or {})
            character = dict(payload.get("character") or {})
            if raw_data and not character:
                character = dict(raw_data)
            out: dict[str, Any] = {
                "operation_type": payload.get("operation_type"),
                "targets": dict(payload.get("targets") or {}),
                "character": character,
                "mode_changes": [dict(x) for x in list(payload.get("mode_changes") or payload.get("modes") or []) if isinstance(x, dict)],
            }
            if not out["targets"]:
                out["targets"] = {
                    "character_ids": [character.get("character_id")] if character.get("character_id") else [],
                    "mode_ids": list(character.get("extra_modes") or []),
                    "scope": "character",
                }
            conf = self._sanitize_confidence_map(payload.get("confidence"))
            return out, conf, self._task_result_telemetry(result)
        except Exception:
            pass
        req = LLMRequest(
            model=str(model or "").strip(),
            messages=[
                Message(
                    role="system",
                    content=(
                        "Ты анализируешь запрос для Specs Studio. "
                        "Верни только JSON. Без комментариев. "
                        "Формат: "
                        "{"
                        "\"operation_type\":\"create_character|update_character|update_modes|mixed|build_character_pack\","
                        "\"targets\":{\"character_ids\":[],\"mode_ids\":[],\"scope\":\"global|character|mixed\"},"
                        "\"character\":{"
                        "\"character_id\":\"\","
                        "\"display_name\":\"\","
                        "\"vibe\":\"balanced|playful|strict|soft\","
                        "\"technicality\":0.0,"
                        "\"energy\":0.0,"
                        "\"default_mode\":\"\","
                        "\"extra_modes\":[],"
                        f"\"llm_profile\":\"{llm_profiles_enum}\","
                        "\"set_active\":true"
                        "},"
                        "\"mode_changes\":[{"
                        "\"mode_id\":\"\","
                        "\"action\":\"add|update|remove\","
                        "\"description\":\"\","
                        "\"legacy_mode\":\"\","
                        "\"show_parameters\":false,"
                        "\"show_summary\":false"
                        "}],"
                        "\"confidence\":{\"field\":0..1}"
                        "}"
                    ),
                ),
                Message(
                    role="user",
                    content=(
                        f"known_character_ids={json.dumps(known_chars, ensure_ascii=False)}\n"
                        f"known_modes={json.dumps(known_modes, ensure_ascii=False)}\n"
                        f"seed={seed}"
                    ),
                ),
            ],
            json_mode=True,
            max_tokens=540,
            temperature=0.35,
            top_p=0.9,
            metadata={"think": True, "studio_specs_task": "seed_extract"},
        )
        try:
            resp = provider.generate(req)
        except Exception:
            return {}, {}, {}

        payload = self._extract_json_object(str(resp.text or ""))
        # Backward compatibility with previous extractor format.
        raw_data = dict(payload.get("data") or {})
        character = dict(payload.get("character") or {})
        if raw_data and not character:
            character = dict(raw_data)

        out: dict[str, Any] = {
            "operation_type": payload.get("operation_type"),
            "targets": dict(payload.get("targets") or {}),
            "character": character,
            "mode_changes": [dict(x) for x in list(payload.get("mode_changes") or payload.get("modes") or []) if isinstance(x, dict)],
        }
        if not out["targets"]:
            out["targets"] = {
                "character_ids": [character.get("character_id")] if character.get("character_id") else [],
                "mode_ids": list(character.get("extra_modes") or []),
                "scope": "character",
            }
        conf = self._sanitize_confidence_map(payload.get("confidence"))
        usage = getattr(resp, "usage", None)
        telemetry = {
            "thinking": str(getattr(resp, "thinking", "") or ""),
            "prompt_eval_count": int(getattr(usage, "prompt_tokens", 0) or 0) if usage is not None else 0,
            "eval_count": int(getattr(usage, "completion_tokens", 0) or 0) if usage is not None else 0,
            "total_tokens": int(getattr(usage, "total_tokens", 0) or 0) if usage is not None else 0,
            "model": str(getattr(resp, "model", "") or ""),
        }
        return out, conf, telemetry

    def _extract_seed_heuristic(
        self,
        *,
        seed: str,
        existing_chars: list[str],
        known_modes: list[str],
    ) -> dict[str, Any]:
        text = str(seed or "").strip()
        low = text.lower()
        op = self._infer_operation_type(seed=text)
        chars = self._detect_character_mentions(text=low, existing_chars=existing_chars)
        modes = self._detect_mode_mentions(text=low, known_modes=known_modes)

        vibe = "balanced"
        if any(x in low for x in ["playful", "игрив", "весел"]):
            vibe = "playful"
        elif any(x in low for x in ["strict", "строг", "жестк"]):
            vibe = "strict"
        elif any(x in low for x in ["soft", "мягк", "нежн"]):
            vibe = "soft"

        technicality = 0.8 if any(x in low for x in ["код", "debug", "engineer", "тех", "программ"]) else 0.55
        energy = 0.8 if any(x in low for x in ["энерг", "быстр", "динам"]) else 0.55

        llm_profile = self._default_llm_profile()
        for profile in self._available_llm_profiles():
            pattern = r"\b" + re.escape(profile.lower()) + r"\b"
            if re.search(pattern, low):
                llm_profile = profile
                break

        set_active = not any(x in low for x in ["не актив", "not active", "без активации"])
        default_mode = modes[0] if modes else ("helper" if op in {self.OP_CREATE_CHARACTER, self.OP_UPDATE_CHARACTER, self.OP_MIXED, self.OP_BUILD_PACK} else "chatting")
        extra_modes = [x for x in modes if x != default_mode]
        mode_changes: list[dict[str, Any]] = []
        for mid in modes:
            action = self._infer_mode_action_from_seed(seed=low, mode_id=mid, known_modes=known_modes)
            mode_changes.append(
                {
                    "mode_id": mid,
                    "action": action,
                    "description": "",
                    "legacy_mode": "chat",
                    "show_parameters": False,
                    "show_summary": False,
                }
            )

        quoted = re.findall(r"['\"]([^'\"]{2,64})['\"]", text)
        display_name = quoted[0].strip() if quoted else ""
        hinted_name = self._extract_seed_name_hint(seed=text)
        if not display_name and hinted_name:
            display_name = hinted_name
        character_id = chars[0] if chars else ""
        if not character_id and op in {self.OP_CREATE_CHARACTER, self.OP_BUILD_PACK}:
            guessed = self._guess_character_id_from_seed(seed=text, existing=existing_chars)
            character_id = guessed
        if not display_name and character_id:
            display_name = self._display_name(character_id)

        scope = "character"
        if op == self.OP_UPDATE_MODES:
            has_global_hint = any(x in low for x in ["global", "глобаль", "везде"])
            has_character_hint = bool(chars) or ("для " in low) or ("for " in low)
            if has_character_hint and has_global_hint:
                scope = "mixed"
            elif has_character_hint:
                scope = "character"
            else:
                scope = "global"
        elif op == self.OP_MIXED:
            scope = "mixed"

        return {
            "operation_type": op,
            "targets": {
                "character_ids": chars[:2],
                "mode_ids": modes[:4],
                "scope": scope,
            },
            "character": {
                "character_id": character_id,
                "display_name": display_name,
                "vibe": vibe,
                "technicality": technicality,
                "energy": energy,
                "default_mode": default_mode,
                "extra_modes": extra_modes,
                "llm_profile": llm_profile,
                "set_active": set_active,
            },
            "mode_changes": mode_changes,
        }

    def _merge_seed_payloads(self, *, primary: dict[str, Any], secondary: dict[str, Any]) -> dict[str, Any]:
        out = dict(primary or {})
        sec = dict(secondary or {})
        for key in ("operation_type", "targets", "character", "mode_changes"):
            pval = out.get(key)
            sval = sec.get(key)
            if key == "targets":
                merged_targets = dict(pval or {})
                merged_targets.update(dict(sval or {}))
                merged_targets["character_ids"] = list(
                    dict.fromkeys([self._safe_id(x) for x in list((pval or {}).get("character_ids") or []) + list((sval or {}).get("character_ids") or []) if self._safe_id(x)])
                )
                merged_targets["mode_ids"] = list(
                    dict.fromkeys([self._mode_id(x) for x in list((pval or {}).get("mode_ids") or []) + list((sval or {}).get("mode_ids") or []) if self._mode_id(x)])
                )
                out[key] = merged_targets
            elif key == "character":
                merged_character = dict(pval or {})
                merged_character.update({k: v for k, v in dict(sval or {}).items() if v not in (None, "", [])})
                out[key] = merged_character
            elif key == "mode_changes":
                seen: set[str] = set()
                merged_modes: list[dict[str, Any]] = []
                for one in list(pval or []) + list(sval or []):
                    row = dict(one or {})
                    mode_id = self._mode_id(row.get("mode_id"))
                    if not mode_id:
                        continue
                    if mode_id in seen:
                        idx = next((i for i, item in enumerate(merged_modes) if self._mode_id(item.get("mode_id")) == mode_id), -1)
                        if idx >= 0:
                            merged = dict(merged_modes[idx])
                            merged.update({k: v for k, v in row.items() if v not in (None, "")})
                            merged_modes[idx] = merged
                        continue
                    seen.add(mode_id)
                    row["mode_id"] = mode_id
                    merged_modes.append(row)
                out[key] = merged_modes
            else:
                out[key] = sval if sval not in (None, "", [], {}) else pval
        return out

    def _resolve_character_targets(
        self,
        *,
        requested: list[str],
        operation_type: str,
        existing: list[str],
    ) -> tuple[list[str], list[str], list[str]]:
        known = sorted({self._safe_id(x) for x in list(existing or []) if self._safe_id(x)})
        resolved: list[str] = []
        candidates: list[str] = []
        missing_for_update: list[str] = []
        for raw in list(requested or []):
            tid = self._safe_id(raw)
            if not tid:
                continue
            if tid in known:
                if tid not in resolved:
                    resolved.append(tid)
                continue
            fuzzy = [cid for cid in known if cid.startswith(tid) or tid in cid]
            if len(fuzzy) == 1:
                if fuzzy[0] not in resolved:
                    resolved.append(fuzzy[0])
                continue
            if len(fuzzy) > 1:
                for one in fuzzy:
                    if one not in candidates:
                        candidates.append(one)
                continue
            if operation_type in {self.OP_CREATE_CHARACTER, self.OP_BUILD_PACK}:
                if tid not in resolved:
                    resolved.append(tid)
            else:
                if tid not in missing_for_update:
                    missing_for_update.append(tid)
        return resolved, candidates, missing_for_update

    def _resolve_mode_targets(self, *, requested: list[str], known_modes: list[str]) -> list[str]:
        known = sorted({self._mode_id(x) for x in list(known_modes or []) if self._mode_id(x)})
        out: list[str] = []
        for raw in list(requested or []):
            mid = self._mode_id(raw)
            if not mid:
                continue
            if mid in known and mid not in out:
                out.append(mid)
                continue
            fuzzy = [x for x in known if x.startswith(mid) or mid in x]
            if len(fuzzy) == 1 and fuzzy[0] not in out:
                out.append(fuzzy[0])
                continue
            if mid not in out:
                out.append(mid)
        return out

    def _hydrate_character_draft(
        self,
        *,
        draft: dict[str, Any],
        operation_type: str,
        targets: dict[str, Any],
        character_payload: dict[str, Any],
        seed: str,
        existing_chars: list[str],
    ) -> None:
        if operation_type not in {self.OP_CREATE_CHARACTER, self.OP_UPDATE_CHARACTER, self.OP_MIXED, self.OP_BUILD_PACK}:
            return

        character_targets = [self._safe_id(x) for x in list(targets.get("character_ids") or []) if self._safe_id(x)]
        cid = self._safe_id(character_payload.get("character_id"))
        if cid and cid not in character_targets:
            character_targets.insert(0, cid)
        if not character_targets and operation_type in {self.OP_CREATE_CHARACTER, self.OP_BUILD_PACK}:
            guess = self._guess_character_id_from_seed(seed=seed, existing=existing_chars)
            if guess:
                character_targets = [guess]
                targets["character_ids"] = list(character_targets)

        if not character_targets:
            return

        targets["character_ids"] = list(character_targets)
        first = character_targets[0]
        name = str(character_payload.get("display_name") or "").strip() or self._display_name(first)
        vibe = self._coerce_field_value(field_id="vibe", value=character_payload.get("vibe")) or "balanced"
        technicality = self._coerce_field_value(field_id="technicality", value=character_payload.get("technicality"))
        if technicality is None:
            technicality = 0.55
        energy = self._coerce_field_value(field_id="energy", value=character_payload.get("energy"))
        if energy is None:
            energy = 0.55
        default_mode = self._coerce_field_value(field_id="default_mode", value=character_payload.get("default_mode")) or "helper"
        extra_modes = self._coerce_field_value(field_id="extra_modes", value=character_payload.get("extra_modes"))
        if extra_modes is None:
            extra_modes = []
        llm_profile = self._coerce_field_value(field_id="llm_profile", value=character_payload.get("llm_profile")) or self._default_llm_profile()
        set_active = self._coerce_field_value(field_id="set_active", value=character_payload.get("set_active"))
        if set_active is None:
            set_active = operation_type in {self.OP_CREATE_CHARACTER, self.OP_MIXED}

        for char_id in character_targets:
            create_flag = char_id not in set(existing_chars or [])
            if operation_type == self.OP_UPDATE_CHARACTER:
                create_flag = False
            patch = {
                "name": name,
                "default_mode": default_mode,
                "default_mood": "neutral",
                "llm_profile": llm_profile,
            }
            draft["characters"][char_id] = {
                "action": "create" if create_flag else "update",
                "patch": patch,
                "persona_state_patch": {},
                "persona_spec_patch": {"modes_add": [default_mode] + [x for x in list(extra_modes or []) if x and x != default_mode]},
                "style": {
                    "vibe": vibe,
                    "technicality": float(technicality),
                    "energy": float(energy),
                },
                "extra_modes": [x for x in list(extra_modes or []) if x],
                "set_active": bool(set_active),
            }

        draft["set_active_character"] = first if bool(set_active) else str(draft.get("set_active_character") or "")

    def _hydrate_modes_draft(
        self,
        *,
        draft: dict[str, Any],
        operation_type: str,
        targets: dict[str, Any],
        mode_payloads: list[dict[str, Any]],
        seed: str,
        known_modes: list[str],
    ) -> None:
        if operation_type not in {self.OP_UPDATE_MODES, self.OP_MIXED}:
            return

        mode_targets = [self._mode_id(x) for x in list(targets.get("mode_ids") or []) if self._mode_id(x)]
        if not mode_targets:
            mode_targets = self._detect_mode_mentions(text=str(seed or "").lower(), known_modes=known_modes)
            targets["mode_ids"] = list(mode_targets)

        by_mode: dict[str, dict[str, Any]] = {}
        for raw in list(mode_payloads or []):
            mode_id = self._mode_id(raw.get("mode_id"))
            if not mode_id:
                continue
            row = by_mode.setdefault(
                mode_id,
                {
                    "mode_id": mode_id,
                    "action": "update",
                    "description": "",
                    "legacy_mode": "chat",
                    "show_parameters": False,
                    "show_summary": False,
                },
            )
            row.update({k: v for k, v in dict(raw).items() if v not in (None, "")})
            row["mode_id"] = mode_id

        for mode_id in list(mode_targets or []):
            if mode_id not in by_mode:
                by_mode[mode_id] = {
                    "mode_id": mode_id,
                    "action": "add" if mode_id not in set(known_modes or []) else "update",
                    "description": f"Custom mode: {mode_id}",
                    "legacy_mode": "chat",
                    "show_parameters": False,
                    "show_summary": False,
                }

        for mode_id, change in by_mode.items():
            action = self._coerce_field_value(field_id="mode_action", value=change.get("action")) or ("add" if mode_id not in set(known_modes or []) else "update")
            description = str(change.get("description") or f"Custom mode: {mode_id}").strip()
            legacy = self._coerce_field_value(field_id="mode_legacy_mode", value=change.get("legacy_mode")) or "chat"
            show_parameters = self._coerce_field_value(field_id="mode_show_parameters", value=change.get("show_parameters"))
            show_summary = self._coerce_field_value(field_id="mode_show_summary", value=change.get("show_summary"))
            draft["modes"][mode_id] = {
                "action": action,
                "patch": {
                    "description": description,
                    "legacy_mode": legacy,
                    "output_format_default": {
                        "show_parameters": bool(show_parameters) if show_parameters is not None else False,
                        "show_summary": bool(show_summary) if show_summary is not None else False,
                    },
                    "persona_effects": dict(self._default_mode_spec_entry(mode_id).get("persona_effects") or {}),
                },
            }

    def _ensure_mode_scope_character_targets(
        self,
        *,
        draft: dict[str, Any],
        operation_type: str,
        targets: dict[str, Any],
    ) -> None:
        if str(operation_type or "").strip().lower() != self.OP_UPDATE_MODES:
            return
        scope = str(targets.get("scope") or "").strip().lower()
        chars = [self._safe_id(x) for x in list(targets.get("character_ids") or []) if self._safe_id(x)]
        draft.setdefault("characters", {})
        if scope not in {"character", "mixed"}:
            for cid in list(chars):
                change = dict(draft["characters"].get(cid) or {})
                if not change:
                    continue
                has_payload = bool(change.get("patch")) or bool(change.get("persona_state_patch")) or bool(change.get("style")) or bool(
                    change.get("extra_modes")
                )
                if not has_payload:
                    draft["characters"].pop(cid, None)
            return
        for cid in list(chars):
            if cid not in dict(draft.get("characters") or {}):
                draft["characters"][cid] = self._default_mode_scope_character_change()

    def _ask_current(self, *, row: dict[str, Any], provider: LLMProviderBase | None = None, model: str = "") -> StudioGeneratorReply:
        phase = str(row.get("phase") or self.PHASE_SEED)
        if phase == self.PHASE_SEED:
            return StudioGeneratorReply(
                text=(
                    "Шаг 1/4. Опиши свободно, что нужно сделать в studio.\n"
                    "Примеры: 'создай персонажа...', 'обнови asya...', 'добавь mode analyst...', 'обнови asya и mode engineer...'"
                ),
                state=row,
            )
        if phase == self.PHASE_REVIEW:
            return self._render_review(row)
        if phase != self.PHASE_CLARIFY:
            row["phase"] = self.PHASE_CLARIFY

        field_id = self._current_field(row)
        if not field_id:
            self._refresh_unresolved(row)
            field_id = self._current_field(row)
            if not field_id:
                row["phase"] = self.PHASE_REVIEW
                return self._render_review(row)

        question = self._question_for_field(field_id=field_id, row=row, provider=provider, model=model)
        unresolved = list(row.get("unresolved_fields") or [])
        idx = unresolved.index(field_id) + 1 if field_id in unresolved else 1
        row["clarify_index"] = max(0, idx - 1)
        total = max(1, len(unresolved))
        body = (
            f"Шаг уточнения {idx}/{total}. {str(question.get('prompt') or '')}\n"
            f"{self._render_options(question)}\n"
            f"{self._render_impact(question)}"
        )
        return StudioGeneratorReply(text=body, state=row)

    def _render_review(self, row: dict[str, Any]) -> StudioGeneratorReply:
        cmd = self._cmd(row)
        row["phase"] = self.PHASE_REVIEW
        row["updated_at"] = now_local_iso()
        return StudioGeneratorReply(
            text=(
                "Review готов.\n"
                f"{self._preview(row)}\n"
                f"Если всё ок: /{cmd} apply\n"
                f"Если нужно отменить: /{cmd} cancel\n"
                "Чтобы изменить поле, напиши: <field>: <value>"
            ),
            state=row,
        )

    def _preview(self, row: dict[str, Any]) -> str:
        op = str(row.get("operation_type") or "")
        targets = dict(row.get("targets") or {})
        planned = self._planned_changes(row)
        artifacts_plan = self._preview_artifacts_plan(row)
        lines = [
            f"operation_type: {op or '(не определён)'}",
            f"targets.character_ids: {', '.join(targets.get('character_ids') or []) or '(none)'}",
            f"targets.mode_ids: {', '.join(targets.get('mode_ids') or []) or '(none)'}",
            f"targets.scope: {str(targets.get('scope') or 'mixed')}",
            "planned_changes:",
        ]
        if planned:
            lines.extend([f"- {x}" for x in planned])
        else:
            lines.append("- (none)")
        if artifacts_plan:
            lines.append("artifacts_plan:")
            lines.extend([f"- {x}" for x in artifacts_plan])
        return "\n".join(lines)

    @staticmethod
    def _render_options(question: dict[str, Any]) -> str:
        opts = [str(x).strip() for x in list(question.get("options") or []) if str(x).strip()]
        allow_custom = bool(question.get("allow_custom", True))
        if not opts:
            return "Введи свой вариант текстом."
        rows = [f"{idx}. {item}" for idx, item in enumerate(opts, start=1)]
        if allow_custom:
            rows.append("0. Ввести свой вариант")
        return "Варианты:\n" + "\n".join(rows)

    @staticmethod
    def _render_impact(question: dict[str, Any]) -> str:
        impact = question.get("impact")
        if isinstance(impact, dict):
            changed = str(impact.get("changed") or "").strip() or "точность patch-изменений в studio."
            files = str(impact.get("files") or "").strip() or "data/specs/rules_for_all/*"
            character = str(impact.get("character") or "").strip() or "выбранная цель apply."
            result = str(impact.get("result") or "").strip() or "предсказуемое применение изменений без потери существующих данных."
            return (
                "Что влияет:\n"
                f"- Изменит: {changed}\n"
                f"- Файлы: {files}\n"
                f"- Для персонажа: {character}\n"
                f"- Результат: {result}"
            )
        text = str(impact or "").strip()
        if not text:
            return (
                "Что влияет:\n"
                "- Изменит: точность patch-изменений в studio.\n"
                "- Файлы: data/specs/rules_for_all/*.\n"
                "- Для персонажа: выбранная цель apply.\n"
                "- Результат: предсказуемое применение изменений без потери существующих данных."
            )
        return (
            "Что влияет:\n"
            f"- Изменит: {text}\n"
            "- Файлы: data/specs/rules_for_all/*.\n"
            "- Для персонажа: выбранная цель apply.\n"
            "- Результат: предсказуемое применение изменений без потери существующих данных."
        )

    def _question_for_field(
        self,
        *,
        field_id: str,
        row: dict[str, Any],
        provider: LLMProviderBase | None,
        model: str,
    ) -> dict[str, Any]:
        qid = str(field_id or "").strip().lower()
        question = dict(self._question_templates().get(qid) or {"id": qid, "prompt": qid, "impact": ""})
        question["options"] = self._ensure_step_options(
            row=row,
            question_id=qid,
            provider=provider,
            model=model,
        )
        question["impact"] = self._impact_payload_for_field(field_id=qid, row=row)
        return question

    def _question_templates(self) -> dict[str, dict[str, Any]]:
        known_modes = self._known_modes()
        return {
            "operation_type": {
                "id": "operation_type",
                "prompt": "Выбери тип операции.",
                "impact": "Определяет, какие файлы и блоки будут изменены в apply.",
                "options": [self.OP_CREATE_CHARACTER, self.OP_UPDATE_CHARACTER, self.OP_UPDATE_MODES, self.OP_MIXED, self.OP_BUILD_PACK],
            },
            "target_character_id": {
                "id": "target_character_id",
                "prompt": "Укажи character_id для операции с персонажем.",
                "impact": "В update будет изменён существующий персонаж; в create будет создан новый.",
            },
            "display_name": {
                "id": "display_name",
                "prompt": "Введи отображаемое имя персонажа.",
                "impact": "Это имя увидит пользователь в metadata и specs.",
            },
            "vibe": {
                "id": "vibe",
                "prompt": "Выбери vibe.",
                "impact": "Влияет на baseline-тон и распределение traits в persona_state.",
                "options": sorted(self.VALID_VIBES),
            },
            "technicality": {
                "id": "technicality",
                "prompt": "Укажи technicality (low/mid/high или 0..1).",
                "impact": "Чем выше значение, тем более технический и структурный стиль ответов.",
                "options": ["low", "mid", "high"],
            },
            "energy": {
                "id": "energy",
                "prompt": "Укажи energy (low/mid/high или 0..1).",
                "impact": "Регулирует динамичность и насыщенность тона.",
                "options": ["low", "mid", "high"],
            },
            "default_mode": {
                "id": "default_mode",
                "prompt": "Выбери default_mode.",
                "impact": "Этот режим будет активироваться по умолчанию для персонажа.",
                "options": list(known_modes),
            },
            "extra_modes": {
                "id": "extra_modes",
                "prompt": "Укажи extra_modes (через запятую) или none.",
                "impact": "Эти modes синхронизируются в taxonomy/modes_spec и persona_spec персонажа.",
            },
            "llm_profile": {
                "id": "llm_profile",
                "prompt": "Выбери llm_profile.",
                "impact": "Профиль управляет балансом скорости и качества генерации.",
                "options": self._available_llm_profiles(),
                "allow_custom": False,
            },
            "set_active": {
                "id": "set_active",
                "prompt": "Сделать персонажа активным после apply?",
                "impact": "Если yes, active_character_id обновится в manifest.",
                "options": ["yes", "no"],
            },
            "character_changes": {
                "id": "character_changes",
                "prompt": "Нужно хотя бы одно изменение персонажа. Выбери вариант или введи вручную key=value.",
                "impact": "Без этого update персонажа не будет иметь эффекта.",
            },
            "target_mode_id": {
                "id": "target_mode_id",
                "prompt": "Укажи mode_id для изменения.",
                "impact": "Mode будет добавлен/обновлён в taxonomy и modes_spec.",
            },
            "mode_scope": {
                "id": "mode_scope",
                "prompt": "Выбери scope для modes.",
                "impact": "global: только глобальные specs; character: также persona_spec выбранного персонажа; mixed: оба варианта.",
                "options": ["global", "character", "mixed"],
            },
            "mode_action": {
                "id": "mode_action",
                "prompt": "Выбери действие для mode.",
                "impact": "add создаёт mode, update правит существующий, remove удаляет только при явном интенте.",
                "options": ["add", "update", "remove"],
            },
            "mode_description": {
                "id": "mode_description",
                "prompt": "Введи описание mode.",
                "impact": "Описание попадёт в modes_spec и влияет на назначение режима.",
            },
            "mode_legacy_mode": {
                "id": "mode_legacy_mode",
                "prompt": "Выбери legacy_mode для mode.",
                "impact": "Нужно для совместимости старого роутинга режимов.",
                "options": ["chat", "task", "coding", "debug", "chatting", "helper", "engineer", "debugger", "planner"],
            },
            "mode_changes": {
                "id": "mode_changes",
                "prompt": "Нужно хотя бы одно изменение mode.",
                "impact": "Без этого update_modes не даст изменений в файлах.",
            },
        }

    def _allow_custom_input(self, *, question_id: str) -> bool:
        qid = str(question_id or "").strip().lower()
        template = dict(self._question_templates().get(qid) or {})
        return bool(template.get("allow_custom", True))

    def _impact_payload_for_field(self, *, field_id: str, row: dict[str, Any]) -> dict[str, str]:
        qid = str(field_id or "").strip().lower()
        op = str(row.get("operation_type") or "").strip().lower()
        targets = dict(row.get("targets") or {})
        scope = str(targets.get("scope") or "mixed").strip().lower() or "mixed"
        char_id = str((list(targets.get("character_ids") or [""]) or [""])[0] or "").strip()
        if not char_id:
            draft_chars = [self._safe_id(x) for x in list(dict(dict(row.get("draft_changes") or {}).get("characters") or {}).keys()) if self._safe_id(x)]
            char_id = str(draft_chars[0] if draft_chars else "<character_id>")
        character_file = f"data/specs/characters/{char_id}/character.json"
        persona_state_file = f"data/specs/characters/{char_id}/persona_state.json"
        persona_spec_file = f"data/specs/characters/{char_id}/persona_spec.json"
        taxonomy_file = "data/specs/rules_for_all/taxonomy.json"
        modes_spec_file = "data/specs/rules_for_all/modes_spec.json"
        manifest_file = "runtime manifest.json"

        if qid == "operation_type":
            return {
                "changed": "маршрут apply: создание, обновление персонажа, modes или смешанный сценарий.",
                "files": f"{taxonomy_file}, {modes_spec_file}, {character_file}, {persona_state_file}, {persona_spec_file}",
                "character": "все target-персонажи текущей студии.",
                "result": "studio соберет корректный change-set и не пропустит обязательные уточнения.",
            }
        if qid == "target_character_id":
            return {
                "changed": "цель patch для character-блоков и привязка scope=character/mixed.",
                "files": f"{character_file}, {persona_state_file}, {persona_spec_file}",
                "character": char_id,
                "result": "изменения применятся к нужному персонажу без создания дубликатов id.",
            }
        if qid == "display_name":
            return {
                "changed": "поле имени персонажа в спецификациях и отображении.",
                "files": f"{character_file}, {persona_state_file}, {persona_spec_file}",
                "character": char_id,
                "result": "имя в интерфейсе и persona-слоях будет синхронизировано.",
            }
        if qid in {"vibe", "technicality", "energy"}:
            return {
                "changed": "style knobs и распределение traits для ответов персонажа.",
                "files": persona_state_file,
                "character": char_id,
                "result": "тон, строгость и техническая глубина ответов станут предсказуемыми.",
            }
        if qid == "default_mode":
            return {
                "changed": "режим по умолчанию и привязка mode в persona.",
                "files": f"{character_file}, {persona_spec_file}, {taxonomy_file}, {modes_spec_file}",
                "character": char_id,
                "result": "новые чаты персонажа стартуют в выбранном режиме без ручного переключения.",
            }
        if qid == "llm_profile":
            return {
                "changed": "строгое поле llm_profile из доступных profile keys.",
                "files": character_file,
                "character": char_id,
                "result": "персонаж использует валидный профиль генерации без невалидных значений.",
            }
        if qid == "set_active":
            return {
                "changed": "active character after apply.",
                "files": manifest_file,
                "character": char_id,
                "result": "следующий обычный чат может пойти сразу от выбранного персонажа.",
            }
        if qid in {"target_mode_id", "mode_action", "mode_description", "mode_legacy_mode", "mode_changes"}:
            files = [taxonomy_file, modes_spec_file]
            if scope in {"character", "mixed"}:
                files.append(persona_spec_file)
            return {
                "changed": "добавление/обновление/удаление mode и его metadata.",
                "files": ", ".join(files),
                "character": char_id if scope in {"character", "mixed"} else "глобальные modes",
                "result": "mode синхронизируется между taxonomy, modes_spec и персонажем по выбранному scope.",
            }
        if qid == "mode_scope":
            return {
                "changed": "область применения mode-изменений: global, character или mixed.",
                "files": f"{taxonomy_file}, {modes_spec_file}, {persona_spec_file}",
                "character": char_id if scope != "global" else "персонаж не затрагивается при global",
                "result": "apply обновит только нужные файлы и не тронет лишние.",
            }
        if qid == "character_changes":
            return {
                "changed": "patch поля character/style для update без полного перезаписывания.",
                "files": f"{character_file}, {persona_state_file}, {persona_spec_file}",
                "character": char_id,
                "result": "обновятся только запрошенные поля, остальные данные сохранятся.",
            }
        template = dict(self._question_templates().get(qid) or {})
        impact_text = str(template.get("impact") or "").strip() or "точность patch-изменений в studio."
        return {
            "changed": impact_text,
            "files": "data/specs/rules_for_all/*",
            "character": char_id if char_id else "выбранная цель apply.",
            "result": "предсказуемое применение изменений без потери существующих данных.",
        }

    def _character_target_options(
        self,
        *,
        row: dict[str, Any],
        provider: LLMProviderBase | None,
        model: str,
    ) -> list[str]:
        op = str(row.get("operation_type") or "").strip().lower()
        seed = str(row.get("seed_prompt") or "")
        existing = [self._safe_id(x) for x in list(self._known_character_ids()) if self._safe_id(x)]
        candidate_hints = [self._safe_id(x) for x in list(dict(row.get("target_candidates") or {}).get("character_ids") or []) if self._safe_id(x)]

        if op in {self.OP_UPDATE_CHARACTER, self.OP_UPDATE_MODES, self.OP_MIXED}:
            ranked = self._rank_existing_character_ids(seed=seed, existing=existing, preferred=candidate_hints)
            if ranked:
                return ranked
            return list(dict.fromkeys(candidate_hints + existing))

        llm_opts: list[str] = []
        if provider is not None:
            llm_opts = self._generate_options_with_retries(
                question_id="target_character_id",
                prompt="Предложи 3-5 character_id для нового персонажа (snake_case/kebab, lowercase).",
                row=row,
                provider=provider,
                model=model,
            )

        deterministic = self._seed_character_id_candidates(seed=seed, existing=existing)
        merged: list[str] = []
        existing_set = set(existing)
        for item in list(llm_opts) + list(deterministic) + list(candidate_hints):
            cid = self._safe_id(item)
            if not cid:
                continue
            if cid in existing_set:
                cid = f"{cid}_new"
            if cid not in merged:
                merged.append(cid)
            if len(merged) >= 5:
                break
        if not merged:
            merged = deterministic[:5]
        return merged

    def _rank_existing_character_ids(self, *, seed: str, existing: list[str], preferred: list[str]) -> list[str]:
        low = str(seed or "").lower()
        hint_set = {self._safe_id(x) for x in list(preferred or []) if self._safe_id(x)}
        tokens = [self._safe_id(x) for x in re.findall(r"[a-zA-Z0-9_-]{2,40}", low)]
        scored: list[tuple[int, str]] = []
        for cid in list(existing or []):
            score = 0
            if cid in hint_set:
                score += 100
            if cid and re.search(rf"\b{re.escape(cid)}\b", low):
                score += 60
            for tok in tokens:
                if not tok:
                    continue
                if tok == cid:
                    score += 40
                elif cid.startswith(tok):
                    score += 22
                elif tok in cid:
                    score += 14
            scored.append((score, cid))
        scored.sort(key=lambda item: (-int(item[0]), str(item[1])))
        ranked = [cid for score, cid in scored if score > 0]
        for cid in list(existing or []):
            if cid not in ranked:
                ranked.append(cid)
        return ranked

    def _seed_character_id_candidates(self, *, seed: str, existing: list[str]) -> list[str]:
        low = str(seed or "").lower()
        existing_set = {self._safe_id(x) for x in list(existing or []) if self._safe_id(x)}
        stop = {
            "studio",
            "character",
            "персонажа",
            "персонаж",
            "создай",
            "создать",
            "обнови",
            "обновить",
            "update",
            "create",
            "new",
            "mode",
            "мод",
            "добавь",
            "add",
            "build",
            "pack",
        }
        out: list[str] = []
        hint = self._extract_seed_name_hint(seed=seed)
        hint_id = self._seed_id_from_name(hint)
        if hint_id:
            candidate = hint_id if hint_id not in existing_set else f"{hint_id}_new"
            if candidate not in out:
                out.append(candidate)

        chunks = re.findall(r"[a-zA-Z][a-zA-Z0-9_-]{2,40}", low)
        for raw in chunks:
            token = self._safe_id(raw)
            if not token or token in stop:
                continue
            candidate = token if token not in existing_set else f"{token}_new"
            if candidate not in out:
                out.append(candidate)
            if len(out) >= 5:
                break
        defaults = ["luna", "nova", "assistant", "mentor", "analyst"]
        for base in defaults:
            candidate = base if base not in existing_set else f"{base}_new"
            if candidate not in out:
                out.append(candidate)
            if len(out) >= 5:
                break
        return out

    def _ensure_step_options(
        self,
        *,
        row: dict[str, Any],
        question_id: str,
        provider: LLMProviderBase | None,
        model: str,
    ) -> list[str]:
        qid = str(question_id or "").strip().lower()
        state_opts = dict(row.get("step_options") or {})

        if qid == "llm_profile":
            profiles = self._available_llm_profiles()
            state_opts[qid] = list(profiles)
            row["step_options"] = state_opts
            return list(profiles)

        if qid == "target_character_id":
            options = self._character_target_options(row=row, provider=provider, model=model)
            state_opts[qid] = list(options)
            row["step_options"] = state_opts
            return list(options)

        existing = [str(x).strip() for x in list(state_opts.get(qid) or []) if str(x).strip()]
        if existing:
            return existing

        question = dict(self._question_templates().get(qid) or {})
        fixed = [str(x).strip() for x in list(question.get("options") or []) if str(x).strip()]
        if fixed:
            state_opts[qid] = fixed
            row["step_options"] = state_opts
            return fixed

        dynamic = self._generate_options_with_retries(
            question_id=qid,
            prompt=str(question.get("prompt") or qid),
            row=row,
            provider=provider,
            model=model,
        )
        if len(dynamic) < 2:
            dynamic = self._fallback_options(question_id=qid, row=row)
        state_opts[qid] = [str(x).strip() for x in dynamic if str(x).strip()]
        row["step_options"] = state_opts
        return list(state_opts[qid])

    def _generate_options_with_retries(
        self,
        *,
        question_id: str,
        prompt: str,
        row: dict[str, Any],
        provider: LLMProviderBase | None,
        model: str,
    ) -> list[str]:
        if provider is None:
            return []
        out: list[str] = []
        telemetry: dict[str, Any] = {}
        for attempt in range(1, self.MAX_OPTIONS_RETRIES + 1):
            opts, llm = self._generate_options_with_llm(
                question_id=question_id,
                prompt=prompt,
                row=row,
                provider=provider,
                model=model,
                attempt=attempt,
            )
            telemetry = dict(llm or {})
            cleaned = self._clean_options(question_id=question_id, options=opts)
            if len(cleaned) >= 2:
                out = cleaned
                break
        if telemetry:
            total = self._clamp_int(telemetry.get("total_tokens"), default=0)
            if total <= 0:
                telemetry["total_tokens"] = self._clamp_int(telemetry.get("prompt_eval_count"), default=0) + self._clamp_int(
                    telemetry.get("eval_count"), default=0
                )
            row["last_llm"] = telemetry
        return out

    def _generate_options_with_llm(
        self,
        *,
        question_id: str,
        prompt: str,
        row: dict[str, Any],
        provider: LLMProviderBase,
        model: str,
        attempt: int,
    ) -> tuple[list[str], dict[str, Any]]:
        system_prompt = (
            "Сгенерируй 3-5 вариантов для шага студии. "
            "Верни только JSON: {\"options\":[\"...\",\"...\",\"...\"]}. "
            "Не добавляй пункт 'свой вариант'."
        )
        user_prompt = (
            f"question_id={question_id}\n"
            f"prompt={prompt}\n"
            f"attempt={attempt}\n"
            f"operation_type={row.get('operation_type')}\n"
            f"targets={json.dumps(dict(row.get('targets') or {}), ensure_ascii=False)}\n"
            f"draft={json.dumps(dict(row.get('draft_changes') or {}), ensure_ascii=False)}"
        )
        try:
            result = run_task_model_json(
                "studio_options",
                user_prompt,
                system_prompt=system_prompt,
                metadata={"think": True, "studio_specs_task": "options", "attempt": int(attempt)},
                temperature=(0.7 if attempt == 1 else 0.45),
                required_fields=("options",),
                max_output_chars=1600,
                max_retries=1,
            )
            payload = self._coerce_json_object(result.json_payload)
            out: list[str] = []
            seen: set[str] = set()
            for item in list(payload.get("options") or []):
                text = str(item or "").strip()
                if not text:
                    continue
                key = text.lower()
                if key in seen:
                    continue
                seen.add(key)
                out.append(text)
                if len(out) >= 5:
                    break
            return out, self._task_result_telemetry(result)
        except Exception:
            pass
        req = LLMRequest(
            model=str(model or "").strip(),
            messages=[
                Message(
                    role="system",
                    content=(
                        "Сгенерируй 3-5 вариантов для шага студии. "
                        "Верни только JSON: {\"options\":[\"...\",\"...\",\"...\"]}. "
                        "Не добавляй пункт 'свой вариант'."
                    ),
                ),
                Message(
                    role="user",
                    content=(
                        f"question_id={question_id}\n"
                        f"prompt={prompt}\n"
                        f"attempt={attempt}\n"
                        f"operation_type={row.get('operation_type')}\n"
                        f"targets={json.dumps(dict(row.get('targets') or {}), ensure_ascii=False)}\n"
                        f"draft={json.dumps(dict(row.get('draft_changes') or {}), ensure_ascii=False)}"
                    ),
                ),
            ],
            json_mode=True,
            max_tokens=220,
            temperature=0.7 if attempt == 1 else 0.45,
            top_p=0.9,
            metadata={"think": True, "studio_specs_task": "options", "attempt": int(attempt)},
        )
        try:
            resp = provider.generate(req)
        except Exception:
            return [], {}

        payload = self._extract_json_object(str(resp.text or ""))
        out: list[str] = []
        seen: set[str] = set()
        for item in list(payload.get("options") or []):
            text = str(item or "").strip()
            if not text:
                continue
            key = text.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(text)
            if len(out) >= 5:
                break

        usage = getattr(resp, "usage", None)
        telemetry = {
            "thinking": str(getattr(resp, "thinking", "") or ""),
            "prompt_eval_count": int(getattr(usage, "prompt_tokens", 0) or 0) if usage is not None else 0,
            "eval_count": int(getattr(usage, "completion_tokens", 0) or 0) if usage is not None else 0,
            "total_tokens": int(getattr(usage, "total_tokens", 0) or 0) if usage is not None else 0,
            "model": str(getattr(resp, "model", "") or ""),
        }
        return out, telemetry

    def _clean_options(self, *, question_id: str, options: list[str]) -> list[str]:
        qid = str(question_id or "").strip().lower()
        out: list[str] = []
        seen: set[str] = set()
        fixed_sets: dict[str, set[str]] = {
            "operation_type": set(self.VALID_OPERATIONS),
            "vibe": set(self.VALID_VIBES),
            "technicality": {"low", "mid", "high"},
            "energy": {"low", "mid", "high"},
            "llm_profile": self._llm_profile_set(),
            "set_active": {"yes", "no"},
            "mode_scope": set(self.VALID_SCOPE),
            "mode_action": set(self.VALID_MODE_ACTIONS),
        }
        allowed = fixed_sets.get(qid)
        max_items = len(allowed) if qid == "llm_profile" and allowed else 5
        for item in list(options or []):
            text = str(item or "").strip()
            if not text:
                continue
            if qid == "target_character_id":
                text = self._safe_id(text)
                if not text:
                    continue
            elif qid == "target_mode_id":
                text = self._mode_id(text)
                if not text:
                    continue
            if allowed is not None:
                norm = text.upper() if qid == "llm_profile" else text.lower()
                if norm not in allowed:
                    continue
                text = norm
            key = text.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(text)
            if len(out) >= max_items:
                break
        return out

    def _fallback_options(self, *, question_id: str, row: dict[str, Any]) -> list[str]:
        qid = str(question_id or "").strip().lower()
        chars = [str(x).strip().lower() for x in list(self._known_character_ids()) if str(x).strip()]
        candidate_chars = [str(x).strip().lower() for x in list(dict(row.get("target_candidates") or {}).get("character_ids") or []) if str(x).strip()]
        merged_chars = list(dict.fromkeys(candidate_chars + chars))
        if not merged_chars:
            merged_chars = ["asya", "luna", "nova"]
        if qid == "target_character_id":
            op = str(row.get("operation_type") or "").strip().lower()
            if op in {self.OP_UPDATE_CHARACTER, self.OP_UPDATE_MODES, self.OP_MIXED}:
                ranked = self._rank_existing_character_ids(seed=str(row.get("seed_prompt") or ""), existing=chars, preferred=candidate_chars)
                return ranked if ranked else merged_chars[:5]
            return self._seed_character_id_candidates(seed=str(row.get("seed_prompt") or ""), existing=chars)[:5]
        modes = self._known_modes()
        mapping: dict[str, list[str]] = {
            "display_name": [self._display_name(merged_chars[0]), "Luna", "Nova"],
            "character_changes": ["default_mode=helper", "llm_profile=QUALITY", "vibe=playful"],
            "llm_profile": self._available_llm_profiles(),
            "target_mode_id": modes[:5] if modes else ["helper", "engineer", "debugger"],
            "mode_changes": ["action=update, description=Technical mode", "action=add, description=Analytics mode", "action=remove"],
            "mode_description": ["Technical implementation mode", "Debug-first mode", "Planning with checkpoints"],
            "mode_legacy_mode": ["chat", "task", "coding", "debug"],
            "extra_modes": ["helper, debugger", "planner, engineer", "none"],
        }
        if qid in mapping:
            return mapping[qid]
        return []

    def _parse_answer(
        self,
        *,
        question_id: str,
        text: str,
        allow_numbered: bool = False,
        row: dict[str, Any] | None = None,
    ):
        src = str(text or "").strip()
        qid = str(question_id or "").strip().lower()
        allow_custom = self._allow_custom_input(question_id=qid)
        if not src:
            return None
        if allow_numbered:
            picked = self._parse_index(src)
            if picked is not None:
                if picked == 0:
                    return "__pick_custom__" if allow_custom else None
                options = [str(x).strip() for x in list(dict(row or {}).get("step_options", {}).get(qid) or []) if str(x).strip()]
                if 1 <= picked <= len(options):
                    src = options[picked - 1]
                else:
                    return None
        return self._coerce_field_value(field_id=qid, value=src)

    def _coerce_field_value(self, *, field_id: str, value: Any):
        qid = str(field_id or "").strip().lower()
        if qid == "operation_type":
            token = str(value or "").strip().lower()
            return token if token in self.VALID_OPERATIONS else None
        if qid == "target_character_id":
            cid = self._safe_id(value)
            return cid if cid else None
        if qid == "display_name":
            text = str(value or "").strip()
            return text[:64] if text else None
        if qid == "vibe":
            token = str(value or "").strip().lower()
            return token if token in self.VALID_VIBES else None
        if qid in {"technicality", "energy"}:
            return self._parse_level(value)
        if qid == "default_mode":
            token = self._mode_id(value)
            return token if token else None
        if qid == "extra_modes":
            if isinstance(value, list):
                cleaned: list[str] = []
                seen: set[str] = set()
                for one in list(value or []):
                    mode = self._mode_id(one)
                    if not mode or mode in seen:
                        continue
                    seen.add(mode)
                    cleaned.append(mode)
                return cleaned
            token = str(value or "").strip().lower()
            if token in {"none", "no", "-", "нет", "пусто"}:
                return []
            items: list[str] = []
            seen: set[str] = set()
            for one in str(value or "").replace(";", ",").split(","):
                mode = self._mode_id(one)
                if not mode or mode in seen:
                    continue
                seen.add(mode)
                items.append(mode)
            return items
        if qid == "llm_profile":
            token = str(value or "").strip().upper()
            return token if token in self._llm_profile_set() else None
        if qid == "set_active":
            token = str(value or "").strip().lower()
            if token in {"yes", "y", "true", "1", "да"}:
                return True
            if token in {"no", "n", "false", "0", "нет"}:
                return False
            return None
        if qid == "character_changes":
            text = str(value or "").strip()
            return text if text else None
        if qid == "target_mode_id":
            mid = self._mode_id(value)
            return mid if mid else None
        if qid == "mode_scope":
            token = str(value or "").strip().lower()
            return token if token in self.VALID_SCOPE else None
        if qid == "mode_action":
            token = str(value or "").strip().lower()
            return token if token in self.VALID_MODE_ACTIONS else None
        if qid == "mode_description":
            text = str(value or "").strip()
            return text[:180] if text else None
        if qid == "mode_legacy_mode":
            token = self._mode_id(value)
            return token if token else None
        if qid in {"mode_show_parameters", "mode_show_summary"}:
            token = str(value or "").strip().lower()
            if token in {"yes", "y", "true", "1", "on", "да"}:
                return True
            if token in {"no", "n", "false", "0", "off", "нет"}:
                return False
            return None
        if qid == "mode_changes":
            text = str(value or "").strip()
            return text if text else None
        return str(value or "").strip() or None

    def _apply_answer_to_state(self, *, row: dict[str, Any], field_id: str, parsed: Any) -> bool:
        fid = str(field_id or "").strip().lower()
        draft = dict(row.get("draft_changes") or self._empty_draft_changes())
        targets = dict(row.get("targets") or {})
        op = str(row.get("operation_type") or "")
        changed = False

        if fid == "operation_type":
            row["operation_type"] = str(parsed or "")
            op = str(parsed or "")
            changed = True
        elif fid == "target_character_id":
            cid = self._safe_id(parsed)
            if cid:
                targets["character_ids"] = [cid]
                draft.setdefault("characters", {})
                if cid not in dict(draft.get("characters") or {}):
                    if str(op or "").strip().lower() == self.OP_UPDATE_MODES:
                        draft["characters"][cid] = self._default_mode_scope_character_change()
                    else:
                        draft["characters"][cid] = self._default_character_change(cid=cid, create=(op == self.OP_CREATE_CHARACTER))
                changed = True
        elif fid == "display_name":
            cid = self._first_character_target(row=row, create_if_missing=True)
            if cid:
                draft["characters"][cid]["patch"]["name"] = str(parsed)
                changed = True
        elif fid in {"vibe", "technicality", "energy"}:
            cid = self._first_character_target(row=row, create_if_missing=True)
            if cid:
                draft["characters"][cid].setdefault("style", {})
                draft["characters"][cid]["style"][fid] = parsed
                changed = True
        elif fid == "default_mode":
            cid = self._first_character_target(row=row, create_if_missing=True)
            if cid:
                mode_id = self._mode_id(parsed)
                if mode_id:
                    draft["characters"][cid]["patch"]["default_mode"] = mode_id
                    persona_patch = dict(draft["characters"][cid].get("persona_spec_patch") or {})
                    modes_add = [self._mode_id(x) for x in list(persona_patch.get("modes_add") or []) if self._mode_id(x)]
                    if mode_id not in modes_add:
                        modes_add.insert(0, mode_id)
                    persona_patch["modes_add"] = list(dict.fromkeys(modes_add))
                    draft["characters"][cid]["persona_spec_patch"] = persona_patch
                    changed = True
        elif fid == "extra_modes":
            cid = self._first_character_target(row=row, create_if_missing=True)
            if cid:
                values = [self._mode_id(x) for x in list(parsed or []) if self._mode_id(x)]
                draft["characters"][cid]["extra_modes"] = values
                persona_patch = dict(draft["characters"][cid].get("persona_spec_patch") or {})
                base = [self._mode_id(x) for x in list(persona_patch.get("modes_add") or []) if self._mode_id(x)]
                for one in values:
                    if one not in base:
                        base.append(one)
                persona_patch["modes_add"] = base
                draft["characters"][cid]["persona_spec_patch"] = persona_patch
                changed = True
        elif fid == "llm_profile":
            cid = self._first_character_target(row=row, create_if_missing=True)
            if cid:
                draft["characters"][cid]["patch"]["llm_profile"] = str(parsed)
                changed = True
        elif fid == "set_active":
            cid = self._first_character_target(row=row, create_if_missing=False)
            if bool(parsed) and cid:
                draft["set_active_character"] = cid
            elif not bool(parsed):
                draft["set_active_character"] = ""
            if cid:
                draft["characters"][cid]["set_active"] = bool(parsed)
            changed = True
        elif fid == "character_changes":
            changed = self._apply_character_change_directive(row=row, text=str(parsed or ""))
        elif fid == "target_mode_id":
            mid = self._mode_id(parsed)
            if mid:
                targets["mode_ids"] = [mid]
                draft.setdefault("modes", {})
                if mid not in dict(draft.get("modes") or {}):
                    draft["modes"][mid] = self._default_mode_change(mode_id=mid)
                changed = True
        elif fid == "mode_scope":
            scope = str(parsed or "").strip().lower()
            if scope in self.VALID_SCOPE:
                targets["scope"] = scope
                self._ensure_mode_scope_character_targets(
                    draft=draft,
                    operation_type=str(op or "").strip().lower(),
                    targets=targets,
                )
                changed = True
        elif fid == "mode_action":
            for mid in list(targets.get("mode_ids") or []):
                draft.setdefault("modes", {})
                draft["modes"].setdefault(mid, self._default_mode_change(mode_id=mid))
                draft["modes"][mid]["action"] = str(parsed)
                changed = True
        elif fid == "mode_description":
            for mid in list(targets.get("mode_ids") or []):
                draft.setdefault("modes", {})
                draft["modes"].setdefault(mid, self._default_mode_change(mode_id=mid))
                patch = dict(draft["modes"][mid].get("patch") or {})
                patch["description"] = str(parsed)
                draft["modes"][mid]["patch"] = patch
                changed = True
        elif fid == "mode_legacy_mode":
            for mid in list(targets.get("mode_ids") or []):
                draft.setdefault("modes", {})
                draft["modes"].setdefault(mid, self._default_mode_change(mode_id=mid))
                patch = dict(draft["modes"][mid].get("patch") or {})
                patch["legacy_mode"] = str(parsed)
                draft["modes"][mid]["patch"] = patch
                changed = True
        elif fid in {"mode_show_parameters", "mode_show_summary"}:
            flag_name = "show_parameters" if fid == "mode_show_parameters" else "show_summary"
            for mid in list(targets.get("mode_ids") or []):
                draft.setdefault("modes", {})
                draft["modes"].setdefault(mid, self._default_mode_change(mode_id=mid))
                patch = dict(draft["modes"][mid].get("patch") or {})
                fmt = dict(patch.get("output_format_default") or {})
                fmt[flag_name] = bool(parsed)
                patch["output_format_default"] = fmt
                if not isinstance(patch.get("persona_effects"), dict):
                    patch["persona_effects"] = dict(self._default_mode_spec_entry(mid).get("persona_effects") or {})
                draft["modes"][mid]["patch"] = patch
                changed = True
        elif fid == "mode_changes":
            changed = self._apply_mode_change_directive(row=row, text=str(parsed or ""))
        else:
            return False

        if changed:
            if str(op or "").strip().lower() == self.OP_UPDATE_MODES:
                self._ensure_mode_scope_character_targets(
                    draft=draft,
                    operation_type=str(op or "").strip().lower(),
                    targets=targets,
                )
            row["targets"] = targets
            row["draft_changes"] = draft
            row["data"] = self._snapshot_data_from_draft(row)
        return changed

    def _apply_character_change_directive(self, *, row: dict[str, Any], text: str) -> bool:
        src = str(text or "").strip()
        if not src:
            return False
        cid = self._first_character_target(row=row, create_if_missing=True)
        if not cid:
            return False
        draft = dict(row.get("draft_changes") or self._empty_draft_changes())
        draft["characters"].setdefault(cid, self._default_character_change(cid=cid, create=False))

        changed = False
        chunks = [x.strip() for x in re.split(r"[;,]", src) if x.strip()]
        for chunk in chunks:
            if "=" not in chunk:
                if chunk.lower() in self.VALID_VIBES:
                    draft["characters"][cid].setdefault("style", {})["vibe"] = chunk.lower()
                    changed = True
                continue
            key, raw = chunk.split("=", 1)
            k = str(key or "").strip().lower()
            v = str(raw or "").strip()
            if k in {"name", "display_name"}:
                draft["characters"][cid]["patch"]["name"] = v
                changed = True
            elif k in {"default_mode", "mode"}:
                mode = self._mode_id(v)
                if mode:
                    draft["characters"][cid]["patch"]["default_mode"] = mode
                    changed = True
            elif k in {"llm_profile", "profile"}:
                prof = self._coerce_field_value(field_id="llm_profile", value=v)
                if prof:
                    draft["characters"][cid]["patch"]["llm_profile"] = prof
                    changed = True
            elif k == "vibe":
                vb = self._coerce_field_value(field_id="vibe", value=v)
                if vb:
                    draft["characters"][cid].setdefault("style", {})["vibe"] = vb
                    changed = True
            elif k == "technicality":
                num = self._parse_level(v)
                if num is not None:
                    draft["characters"][cid].setdefault("style", {})["technicality"] = num
                    changed = True
            elif k == "energy":
                num = self._parse_level(v)
                if num is not None:
                    draft["characters"][cid].setdefault("style", {})["energy"] = num
                    changed = True
            elif k == "set_active":
                flag = self._coerce_field_value(field_id="set_active", value=v)
                if flag is not None:
                    draft["characters"][cid]["set_active"] = bool(flag)
                    draft["set_active_character"] = cid if bool(flag) else ""
                    changed = True
            elif k in {"extra_modes", "modes"}:
                values = self._coerce_field_value(field_id="extra_modes", value=v)
                if values is not None:
                    draft["characters"][cid]["extra_modes"] = list(values)
                    changed = True
        if changed:
            row["draft_changes"] = draft
            row["data"] = self._snapshot_data_from_draft(row)
        return changed

    def _apply_mode_change_directive(self, *, row: dict[str, Any], text: str) -> bool:
        src = str(text or "").strip()
        if not src:
            return False
        draft = dict(row.get("draft_changes") or self._empty_draft_changes())
        targets = dict(row.get("targets") or {})
        modes = dict(draft.get("modes") or {})

        pairs: dict[str, str] = {}
        for chunk in [x.strip() for x in re.split(r"[;,]", src) if x.strip()]:
            if "=" not in chunk:
                continue
            key, raw = chunk.split("=", 1)
            pairs[str(key or "").strip().lower()] = str(raw or "").strip()

        mode_id = self._mode_id(pairs.get("mode_id") or (list(targets.get("mode_ids") or [""])[0] if list(targets.get("mode_ids") or []) else ""))
        if not mode_id:
            return False
        targets["mode_ids"] = [mode_id]
        mode_row = dict(modes.get(mode_id) or self._default_mode_change(mode_id=mode_id))
        action = self._coerce_field_value(field_id="mode_action", value=pairs.get("action"))
        if action:
            mode_row["action"] = action
        patch = dict(mode_row.get("patch") or {})
        if pairs.get("description"):
            patch["description"] = pairs["description"]
        legacy = self._coerce_field_value(field_id="mode_legacy_mode", value=pairs.get("legacy_mode"))
        if legacy:
            patch["legacy_mode"] = legacy
        if "show_parameters" in pairs:
            flag = self._coerce_field_value(field_id="mode_show_parameters", value=pairs.get("show_parameters"))
            if flag is not None:
                fmt = dict(patch.get("output_format_default") or {})
                fmt["show_parameters"] = bool(flag)
                patch["output_format_default"] = fmt
        if "show_summary" in pairs:
            flag = self._coerce_field_value(field_id="mode_show_summary", value=pairs.get("show_summary"))
            if flag is not None:
                fmt = dict(patch.get("output_format_default") or {})
                fmt["show_summary"] = bool(flag)
                patch["output_format_default"] = fmt
        if not isinstance(patch.get("persona_effects"), dict):
            patch["persona_effects"] = dict(self._default_mode_spec_entry(mode_id).get("persona_effects") or {})
        mode_row["patch"] = patch
        modes[mode_id] = mode_row
        draft["modes"] = modes
        row["targets"] = targets
        row["draft_changes"] = draft
        row["data"] = self._snapshot_data_from_draft(row)
        return True

    def _refresh_unresolved(self, row: dict[str, Any]) -> None:
        op = str(row.get("operation_type") or "")
        targets = dict(row.get("targets") or {})
        draft = dict(row.get("draft_changes") or {})
        conf = self._sanitize_confidence_map(row.get("confidence_map"))
        unresolved: list[str] = []

        if op not in self.VALID_OPERATIONS:
            unresolved.append("operation_type")

        low_conf_fields = [
            "operation_type",
            "target_character_id",
            "display_name",
            "default_mode",
            "llm_profile",
            "target_mode_id",
            "mode_scope",
        ]

        requires_character = op in {self.OP_CREATE_CHARACTER, self.OP_UPDATE_CHARACTER, self.OP_MIXED, self.OP_BUILD_PACK}
        if requires_character:
            char_targets = [self._safe_id(x) for x in list(targets.get("character_ids") or []) if self._safe_id(x)]
            missing_update_chars = [self._safe_id(x) for x in list(dict(row.get("target_candidates") or {}).get("missing_update_characters") or []) if self._safe_id(x)]
            if not char_targets or bool(list(dict(row.get("target_candidates") or {}).get("character_ids") or [])) or bool(missing_update_chars):
                unresolved.append("target_character_id")
            else:
                cid = char_targets[0]
                char_change = dict(dict(draft.get("characters") or {}).get(cid) or {})
                patch = dict(char_change.get("patch") or {})
                style = dict(char_change.get("style") or {})
                if op in {self.OP_CREATE_CHARACTER, self.OP_BUILD_PACK}:
                    if not str(patch.get("name") or "").strip():
                        unresolved.append("display_name")
                    if not self._mode_id(patch.get("default_mode")):
                        unresolved.append("default_mode")
                    if str(patch.get("llm_profile") or "").strip().upper() not in self._llm_profile_set():
                        unresolved.append("llm_profile")
                    if "set_active" not in char_change:
                        unresolved.append("set_active")
                if op in {self.OP_UPDATE_CHARACTER, self.OP_MIXED}:
                    if "llm_profile" in patch and str(patch.get("llm_profile") or "").strip().upper() not in self._llm_profile_set():
                        unresolved.append("llm_profile")
                    has_change = bool(patch) or bool(style) or bool(char_change.get("extra_modes"))
                    if not has_change:
                        unresolved.append("character_changes")

        requires_modes = op in {self.OP_UPDATE_MODES, self.OP_MIXED}
        if requires_modes:
            mode_targets = [self._mode_id(x) for x in list(targets.get("mode_ids") or []) if self._mode_id(x)]
            if not mode_targets:
                unresolved.append("target_mode_id")
            if str(targets.get("scope") or "").strip().lower() not in self.VALID_SCOPE:
                unresolved.append("mode_scope")
            if not bool(dict(draft.get("modes") or {})):
                unresolved.append("mode_changes")
            scope = str(targets.get("scope") or "").strip().lower()
            char_targets = [self._safe_id(x) for x in list(targets.get("character_ids") or []) if self._safe_id(x)]
            if op == self.OP_UPDATE_MODES and scope in {"character", "mixed"} and not char_targets:
                unresolved.append("target_character_id")

        for field_id in list(low_conf_fields):
            if field_id in unresolved:
                continue
            if field_id in {"target_mode_id", "mode_scope"} and not requires_modes:
                continue
            needs_mode_character = bool(
                op == self.OP_UPDATE_MODES and str(targets.get("scope") or "").strip().lower() in {"character", "mixed"}
            )
            if field_id in {"target_character_id", "display_name", "default_mode", "llm_profile"} and not requires_character and not (
                needs_mode_character and field_id == "target_character_id"
            ):
                continue
            if field_id in {"display_name", "default_mode", "llm_profile"} and op not in {self.OP_CREATE_CHARACTER, self.OP_BUILD_PACK}:
                continue
            if self._clamp_float(conf.get(field_id), default=1.0) < self.CONFIDENCE_THRESHOLD:
                unresolved.append(field_id)

        row["unresolved_fields"] = self._order_unresolved_fields(op=op, fields=unresolved)
        row["updated_at"] = now_local_iso()
        if not unresolved:
            row["phase"] = self.PHASE_REVIEW
            row["clarify_index"] = 0
            row["current_field"] = ""
            return
        row["phase"] = self.PHASE_CLARIFY
        row["clarify_index"] = 0
        row["current_field"] = str((list(row.get("unresolved_fields") or [""])[0] or ""))

    def _order_unresolved_fields(self, *, op: str, fields: list[str]) -> list[str]:
        operation = str(op or "").strip().lower()
        if operation in {self.OP_CREATE_CHARACTER, self.OP_BUILD_PACK}:
            priority = [
                "operation_type",
                "target_character_id",
                "display_name",
                "default_mode",
                "llm_profile",
                "vibe",
                "technicality",
                "energy",
                "set_active",
                "target_mode_id",
                "mode_scope",
                "mode_changes",
            ]
        elif operation == self.OP_UPDATE_CHARACTER:
            priority = [
                "operation_type",
                "target_character_id",
                "character_changes",
                "llm_profile",
                "set_active",
                "display_name",
                "default_mode",
                "target_mode_id",
                "mode_scope",
            ]
        elif operation == self.OP_UPDATE_MODES:
            priority = [
                "operation_type",
                "target_mode_id",
                "mode_scope",
                "mode_action",
                "mode_changes",
                "target_character_id",
            ]
        else:
            priority = [
                "operation_type",
                "target_mode_id",
                "mode_scope",
                "target_character_id",
                "character_changes",
                "display_name",
                "default_mode",
                "llm_profile",
                "set_active",
                "mode_changes",
            ]

        source = [str(x or "").strip().lower() for x in list(fields or []) if str(x or "").strip()]
        source_set = set(source)
        ordered: list[str] = []
        seen: set[str] = set()
        for key in list(priority) + source:
            token = str(key or "").strip().lower()
            if not token or token in seen or token not in source_set:
                continue
            seen.add(token)
            ordered.append(token)
        return ordered

    def _current_field(self, row: dict[str, Any]) -> str:
        cur = str(row.get("current_field") or "").strip().lower()
        unresolved = [str(x).strip().lower() for x in list(row.get("unresolved_fields") or []) if str(x).strip()]
        if cur and cur in unresolved:
            return cur
        if unresolved:
            row["current_field"] = unresolved[0]
            row["clarify_index"] = 0
            return unresolved[0]
        row["current_field"] = ""
        row["clarify_index"] = 0
        return ""

    def _planned_changes(self, row: dict[str, Any]) -> list[str]:
        draft = dict(row.get("draft_changes") or {})
        out: list[str] = []
        for cid, raw in dict(draft.get("characters") or {}).items():
            change = dict(raw or {})
            action = str(change.get("action") or "update")
            patch = dict(change.get("patch") or {})
            style = dict(change.get("style") or {})
            parts: list[str] = []
            if patch:
                parts.extend([f"{k}={v}" for k, v in patch.items()])
            if style:
                parts.extend([f"{k}={v}" for k, v in style.items()])
            extra = [self._mode_id(x) for x in list(change.get("extra_modes") or []) if self._mode_id(x)]
            if extra:
                parts.append(f"extra_modes={','.join(extra)}")
            if change.get("set_active") is True:
                parts.append("set_active=true")
            out.append(f"{'create' if action == 'create' else 'update'} character {cid}" + (f" ({'; '.join(parts)})" if parts else ""))
        for mid, raw in dict(draft.get("modes") or {}).items():
            change = dict(raw or {})
            action = str(change.get("action") or "update")
            patch = dict(change.get("patch") or {})
            out.append(f"{action if action in {'add', 'update', 'remove'} else 'update_mode'} mode {mid}" + (f" ({', '.join([f'{k}={v}' for k, v in patch.items()])})" if patch else ""))
        set_active_cid = str(draft.get("set_active_character") or "").strip().lower()
        if set_active_cid:
            out.append(f"set_active {set_active_cid}")
        return out

    def _preview_artifacts_plan(self, row: dict[str, Any]) -> list[str]:
        op = str(row.get("operation_type") or "").strip().lower()
        if op not in {self.OP_CREATE_CHARACTER, self.OP_UPDATE_CHARACTER, self.OP_MIXED, self.OP_BUILD_PACK}:
            return []
        draft = dict(row.get("draft_changes") or {})
        targets = dict(row.get("targets") or {})
        chars = [self._safe_id(x) for x in list(dict(draft.get("characters") or {}).keys()) if self._safe_id(x)]
        if not chars:
            chars = [self._safe_id(x) for x in list(targets.get("character_ids") or []) if self._safe_id(x)]
        out: list[str] = []
        for cid in list(dict.fromkeys([x for x in chars if x])):
            out.append(f"{cid}: studio_blueprint.json + artifacts(character.json, persona_state.json, persona_spec.json, evolution_spec.json)")
        return out

    def _apply_draft_changes(
        self,
        *,
        row: dict[str, Any],
        provider: LLMProviderBase | None = None,
        model: str = "",
    ) -> dict[str, Any]:
        op = str(row.get("operation_type") or "").strip().lower()
        if op == self.OP_BUILD_PACK:
            return self._apply_build_pack_changes(row=row, provider=provider, model=model)
        targets = dict(row.get("targets") or {})
        draft = dict(row.get("draft_changes") or self._empty_draft_changes())
        planned = self._planned_changes(row)
        written_files: list[str] = []

        taxonomy_path = (self.specs_root / "taxonomy.json").resolve()
        modes_spec_path = (self.specs_root / "modes_spec.json").resolve()
        taxonomy = self._read_json(taxonomy_path)
        modes_spec = self._read_json(modes_spec_path)
        taxonomy_changed = False
        modes_spec_changed = False

        taxonomy.setdefault("modes", [])
        taxonomy.setdefault("aliases", {})
        aliases = dict(taxonomy.get("aliases") or {})
        aliases_modes = dict(aliases.get("modes") or {})
        aliases_modes.setdefault("chat", "chatting")
        aliases_modes.setdefault("task", "helper")
        aliases_modes.setdefault("coding", "engineer")
        aliases_modes.setdefault("debug", "debugger")
        aliases["modes"] = aliases_modes
        taxonomy["aliases"] = aliases
        taxonomy_modes = [self._mode_id(x) for x in list(taxonomy.get("modes") or []) if self._mode_id(x)]

        modes_spec.setdefault("schema_version", 1)
        modes_map = dict(modes_spec.get("modes") or {})

        added_modes: list[str] = []
        updated_modes: list[str] = []
        removed_modes: list[str] = []
        for mode_id, raw in dict(draft.get("modes") or {}).items():
            mid = self._mode_id(mode_id)
            if not mid:
                continue
            change = dict(raw or {})
            action = str(change.get("action") or "update").strip().lower()
            if action not in self.VALID_MODE_ACTIONS:
                action = "update"
            patch = dict(change.get("patch") or {})
            exists = mid in modes_map

            if action == "remove":
                # remove only when explicit intent was provided
                if mid in taxonomy_modes:
                    taxonomy_modes = [x for x in taxonomy_modes if x != mid]
                    taxonomy_changed = True
                if mid in modes_map:
                    modes_map.pop(mid, None)
                    modes_spec_changed = True
                removed_modes.append(mid)
                continue

            if mid not in taxonomy_modes:
                taxonomy_modes.append(mid)
                taxonomy_changed = True
                added_modes.append(mid)
            elif mid not in updated_modes:
                updated_modes.append(mid)

            entry = dict(modes_map.get(mid) or {})
            base_entry = self._default_mode_spec_entry(mid)
            merged_entry = self._patch_merge(base_entry, entry)
            merged_entry = self._patch_merge(merged_entry, patch)
            if not isinstance(merged_entry.get("persona_effects"), dict):
                merged_entry["persona_effects"] = dict(base_entry.get("persona_effects") or {})
            if merged_entry != entry:
                modes_spec_changed = True
            modes_map[mid] = merged_entry
            if not exists and mid not in added_modes:
                added_modes.append(mid)

        taxonomy["modes"] = list(dict.fromkeys([x for x in taxonomy_modes if x]))
        modes_spec["modes"] = modes_map

        if taxonomy_changed or bool(draft.get("modes")):
            self._write_json(taxonomy_path, taxonomy)
            written_files.append(str(taxonomy_path))
        if modes_spec_changed or bool(draft.get("modes")):
            self._write_json(modes_spec_path, modes_spec)
            written_files.append(str(modes_spec_path))

        created_characters: list[str] = []
        updated_characters: list[str] = []
        set_active_character = str(draft.get("set_active_character") or "").strip().lower()
        mode_scope = str(targets.get("scope") or "mixed").strip().lower()
        mode_ids_for_character_sync = [self._mode_id(x) for x in list(dict(draft.get("modes") or {}).keys()) if self._mode_id(x)]

        for cid, raw in dict(draft.get("characters") or {}).items():
            char_id = self._safe_id(cid)
            if not char_id:
                continue
            change = dict(raw or {})
            action = str(change.get("action") or "update").strip().lower()

            spec_dir = (self.character_specs_root / char_id).resolve()
            spec_dir.mkdir(parents=True, exist_ok=True)
            character_path = spec_dir / "character.json"
            persona_state_path = spec_dir / "persona_state.json"
            persona_spec_path = spec_dir / "persona_spec.json"
            evolution_path = spec_dir / "evolution_spec.json"

            character = self._read_json(character_path)
            persona_state = self._read_json(persona_state_path)
            persona_spec = self._read_json(persona_spec_path)
            evolution = self._read_json(evolution_path)
            if not character:
                character = self._default_character_json(char_id)
            if not persona_state:
                persona_state = self._default_persona_state_json(char_id, character_name=str(character.get("name") or self._display_name(char_id)))
            if not persona_spec:
                persona_spec = self._default_persona_spec_json(character_name=str(character.get("name") or self._display_name(char_id)))
            if not evolution:
                evolution = self._default_evolution_json()

            before_character = dict(character)
            before_persona_state = dict(persona_state)
            before_persona_spec = dict(persona_spec)

            patch = dict(change.get("patch") or {})
            character = self._patch_merge(character, patch)
            character["schema_version"] = int(character.get("schema_version") or 1)
            character["character_id"] = char_id
            character["id"] = char_id
            if "model_profile" in character and str(character.get("llm_profile") or "").strip():
                character["model_profile"] = str(character.get("llm_profile") or "").strip()

            display_name = str(character.get("name") or self._display_name(char_id)).strip()
            persona_state.setdefault("schema_version", 1)
            persona_state["character_id"] = char_id
            persona_state["name"] = display_name
            persona_state.setdefault("mood", str(character.get("default_mood") or "neutral"))
            traits = dict(persona_state.get("traits") or {})
            style = dict(change.get("style") or {})
            if style or action == "create":
                vibe = str(style.get("vibe") or "balanced")
                technicality = self._parse_level(style.get("technicality"))
                energy = self._parse_level(style.get("energy"))
                if technicality is None:
                    technicality = 0.55
                if energy is None:
                    energy = 0.55
                self._apply_style_knobs(traits=traits, vibe=vibe, technicality=float(technicality), energy=float(energy))
            persona_state["traits"] = traits

            persona_state_patch = dict(change.get("persona_state_patch") or {})
            if persona_state_patch:
                persona_state = self._patch_merge(persona_state, persona_state_patch)

            persona_spec.setdefault("schema_version", 1)
            persona_spec.setdefault("identity", [f"You are {display_name}."])
            persona_spec.setdefault(
                "locks_map",
                {
                    "feminine": "Always use feminine grammatical gender for self-reference.",
                    "informal_you": "Use informal address form ('ты').",
                },
            )
            locks_map = dict(persona_spec.get("locks_map") or {})
            locks_map["informal_you"] = "Use informal address form ('ты')."
            persona_spec["locks_map"] = locks_map
            persona_spec.setdefault("bans_template", "Do not use banned word: {term}.")
            persona_spec.setdefault("moods", {"neutral": ["Tone: neutral. Keep balanced and practical tone."]})
            modes_map_for_character = dict(persona_spec.get("modes") or {})

            char_modes: list[str] = []
            default_mode = self._mode_id(character.get("default_mode"))
            if default_mode:
                char_modes.append(default_mode)
            for one in list(change.get("extra_modes") or []):
                mid = self._mode_id(one)
                if mid:
                    char_modes.append(mid)
            for one in list(dict(change.get("persona_spec_patch") or {}).get("modes_add") or []):
                mid = self._mode_id(one)
                if mid:
                    char_modes.append(mid)

            # Sync mode changes into character persona_spec when scope requires it.
            if mode_ids_for_character_sync and mode_scope in {"character", "mixed"}:
                for mid in mode_ids_for_character_sync:
                    mode_change = dict(dict(draft.get("modes") or {}).get(mid) or {})
                    action_mode = str(mode_change.get("action") or "update").strip().lower()
                    if action_mode == "remove":
                        modes_map_for_character.pop(mid, None)
                        continue
                    mode_patch = dict(mode_change.get("patch") or {})
                    description = str(mode_patch.get("description") or "").strip()
                    if description:
                        modes_map_for_character[mid] = [description]
                    else:
                        modes_map_for_character[mid] = list(modes_map_for_character.get(mid) or self._default_mode_lines(mid))

            for mode in list(dict.fromkeys([x for x in char_modes if x])):
                modes_map_for_character[mode] = list(modes_map_for_character.get(mode) or self._default_mode_lines(mode))
                if mode not in taxonomy["modes"]:
                    taxonomy["modes"].append(mode)
                    if mode not in modes_map:
                        modes_map[mode] = self._default_mode_spec_entry(mode)
                    if mode not in added_modes:
                        added_modes.append(mode)
            persona_spec["modes"] = modes_map_for_character

            if character != before_character:
                self._write_json(character_path, character)
                written_files.append(str(character_path))
            if persona_state != before_persona_state:
                self._write_json(persona_state_path, persona_state)
                written_files.append(str(persona_state_path))
            if persona_spec != before_persona_spec:
                self._write_json(persona_spec_path, persona_spec)
                written_files.append(str(persona_spec_path))
            if not evolution_path.exists():
                self._write_json(evolution_path, evolution)
                written_files.append(str(evolution_path))

            runtime_path = self._sync_runtime_character_payload(char_id=char_id, character=character)
            if runtime_path:
                written_files.append(runtime_path)

            if action == "create":
                created_characters.append(char_id)
            else:
                updated_characters.append(char_id)
            if bool(change.get("set_active")):
                set_active_character = char_id

        # Modes could be appended while writing characters.
        taxonomy["modes"] = list(dict.fromkeys([self._mode_id(x) for x in list(taxonomy.get("modes") or []) if self._mode_id(x)]))
        modes_spec["modes"] = modes_map
        if str(taxonomy_path) not in written_files:
            self._write_json(taxonomy_path, taxonomy)
            written_files.append(str(taxonomy_path))
        if str(modes_spec_path) not in written_files:
            self._write_json(modes_spec_path, modes_spec)
            written_files.append(str(modes_spec_path))

        if set_active_character and self.character_specs_root == self.default_character_specs_root:
            self.storage.sync_manifest()
            manifest = self.storage.load_manifest()
            manifest["active_character_id"] = set_active_character
            self.storage.save_manifest(manifest)

        blueprint_files: list[str] = []
        artifact_files: list[str] = []
        if op in {self.OP_CREATE_CHARACTER, self.OP_UPDATE_CHARACTER, self.OP_MIXED}:
            artifacts_result = self._sync_blueprints_for_characters(
                op=op,
                draft=draft,
                targets=targets,
                set_active_character=set_active_character,
                provider=provider,
                model=model,
            )
            written_files.extend(list(artifacts_result.get("written_files") or []))
            blueprint_files.extend(list(artifacts_result.get("blueprint_files") or []))
            artifact_files.extend(list(artifacts_result.get("artifact_files") or []))

        written_files = list(dict.fromkeys(written_files))
        return {
            "operation_type": op,
            "targets": {
                "character_ids": [self._safe_id(x) for x in list(targets.get("character_ids") or []) if self._safe_id(x)],
                "mode_ids": [self._mode_id(x) for x in list(targets.get("mode_ids") or []) if self._mode_id(x)],
                "scope": str(targets.get("scope") or "mixed"),
            },
            "planned_changes": planned,
            "written_files": written_files,
            "added_modes": sorted(list(dict.fromkeys([x for x in added_modes if x]))),
            "updated_modes": sorted(list(dict.fromkeys([x for x in updated_modes if x]))),
            "removed_modes": sorted(list(dict.fromkeys([x for x in removed_modes if x]))),
            "created_characters": sorted(list(dict.fromkeys([x for x in created_characters if x]))),
            "updated_characters": sorted(list(dict.fromkeys([x for x in updated_characters if x]))),
            "set_active_character": set_active_character,
            "blueprint_files": list(dict.fromkeys([str(x) for x in blueprint_files if str(x).strip()])),
            "artifact_files": list(dict.fromkeys([str(x) for x in artifact_files if str(x).strip()])),
        }

    def _sync_blueprints_for_characters(
        self,
        *,
        op: str,
        draft: dict[str, Any],
        targets: dict[str, Any],
        set_active_character: str,
        provider: LLMProviderBase | None = None,
        model: str = "",
    ) -> dict[str, list[str]]:
        char_ids = [self._safe_id(x) for x in list(dict(draft.get("characters") or {}).keys()) if self._safe_id(x)]
        if not char_ids:
            char_ids = [self._safe_id(x) for x in list(targets.get("character_ids") or []) if self._safe_id(x)]
        if not char_ids:
            return {"written_files": [], "blueprint_files": [], "artifact_files": []}

        written_files: list[str] = []
        blueprint_files: list[str] = []
        artifact_files: list[str] = []
        for cid in list(dict.fromkeys([x for x in char_ids if x])):
            source_data = self._build_character_source_data_for_blueprint(
                cid=cid,
                op=op,
                draft=draft,
                targets=targets,
                set_active_character=set_active_character,
            )
            blueprint: dict[str, Any] = {}
            if provider is not None:
                blueprint = self._generate_pack_blueprint_with_llm(data=source_data, provider=provider, model=model)
            if not blueprint:
                blueprint = self._fallback_pack_blueprint(source_data=source_data)

            spec_dir = (self.character_specs_root / cid).resolve()
            spec_dir.mkdir(parents=True, exist_ok=True)
            blueprint_path = spec_dir / "studio_blueprint.json"
            blueprint_payload = {
                "schema_version": 1,
                "program": "studio_generator",
                "created_at": now_local_iso(),
                "operation_type": str(op or self.OP_CREATE_CHARACTER),
                "character_id": cid,
                "source": source_data,
                "blueprint": blueprint,
            }
            self._write_json(blueprint_path, blueprint_payload)
            blueprint_files.append(str(blueprint_path))
            written_files.append(str(blueprint_path))
            generated = self._materialize_pack_artifacts(spec_dir=spec_dir, payload=blueprint, source_data=source_data)
            artifact_files.extend(list(generated))
            written_files.extend(list(generated))
        return {
            "written_files": list(dict.fromkeys([str(x) for x in written_files if str(x).strip()])),
            "blueprint_files": list(dict.fromkeys([str(x) for x in blueprint_files if str(x).strip()])),
            "artifact_files": list(dict.fromkeys([str(x) for x in artifact_files if str(x).strip()])),
        }

    def _build_character_source_data_for_blueprint(
        self,
        *,
        cid: str,
        op: str,
        draft: dict[str, Any],
        targets: dict[str, Any],
        set_active_character: str,
    ) -> dict[str, Any]:
        char_id = self._safe_id(cid)
        spec_dir = (self.character_specs_root / char_id).resolve()
        character = self._read_json(spec_dir / "character.json")
        persona_spec = self._read_json(spec_dir / "persona_spec.json")
        change = dict(dict(draft.get("characters") or {}).get(char_id) or {})
        patch = dict(change.get("patch") or {})
        style = dict(change.get("style") or {})
        display_name = str(character.get("name") or patch.get("name") or self._display_name(char_id)).strip() or self._display_name(char_id)
        default_mode = self._mode_id(character.get("default_mode") or patch.get("default_mode")) or "helper"
        llm_profile = self._safe_llm_profile(character.get("llm_profile") or patch.get("llm_profile"))

        modes_from_spec = [self._mode_id(x) for x in list(dict(persona_spec.get("modes") or {}).keys()) if self._mode_id(x)]
        extra_modes = [self._mode_id(x) for x in list(change.get("extra_modes") or []) if self._mode_id(x)]
        extra_modes.extend([x for x in modes_from_spec if x and x != default_mode])
        extra_modes = [x for x in list(dict.fromkeys(extra_modes)) if x and x != default_mode]

        vibe = str(style.get("vibe") or "balanced").strip().lower()
        if vibe not in self.VALID_VIBES:
            vibe = "balanced"
        technicality = self._parse_level(style.get("technicality"))
        energy = self._parse_level(style.get("energy"))
        if technicality is None:
            technicality = 0.55
        if energy is None:
            energy = 0.55

        scope = str(targets.get("scope") or "character").strip().lower() or "character"
        mode_ids = [self._mode_id(x) for x in list(targets.get("mode_ids") or []) if self._mode_id(x)]
        return {
            "character_id": char_id,
            "display_name": display_name,
            "vibe": vibe,
            "technicality": float(technicality),
            "energy": float(energy),
            "default_mode": default_mode,
            "extra_modes": extra_modes,
            "llm_profile": llm_profile,
            "set_active": bool(str(set_active_character or "").strip().lower() == char_id or bool(change.get("set_active"))),
            "operation_type": str(op or self.OP_CREATE_CHARACTER),
            "targets": {
                "character_ids": [char_id],
                "mode_ids": mode_ids,
                "scope": scope,
            },
        }

    def _apply_build_pack_changes(
        self,
        *,
        row: dict[str, Any],
        provider: LLMProviderBase | None = None,
        model: str = "",
    ) -> dict[str, Any]:
        targets = dict(row.get("targets") or {})
        draft = dict(row.get("draft_changes") or self._empty_draft_changes())
        planned = self._planned_changes(row)
        char_targets = [self._safe_id(x) for x in list(targets.get("character_ids") or []) if self._safe_id(x)]
        if not char_targets:
            char_targets = [self._safe_id(x) for x in list(dict(draft.get("characters") or {}).keys()) if self._safe_id(x)]
        guessed_char_id = self._guess_character_id_from_seed(
            seed=str(row.get("seed_prompt") or ""),
            existing=self._known_character_ids(),
        )
        char_id = char_targets[0] if char_targets else (guessed_char_id or "studio_character")
        targets["character_ids"] = [char_id]
        targets["scope"] = "character"

        char_change = dict(dict(draft.get("characters") or {}).get(char_id) or self._default_character_change(cid=char_id, create=True))
        patch = dict(char_change.get("patch") or {})
        style = dict(char_change.get("style") or {})
        display_name = str(patch.get("name") or self._display_name(char_id)).strip() or self._display_name(char_id)
        default_mode = self._mode_id(patch.get("default_mode")) or "helper"
        extra_modes = [self._mode_id(x) for x in list(char_change.get("extra_modes") or []) if self._mode_id(x)]
        llm_profile = str(patch.get("llm_profile") or self._default_llm_profile()).strip().upper()
        if llm_profile not in self._llm_profile_set():
            llm_profile = self._default_llm_profile()
        set_active = bool(char_change.get("set_active")) or str(draft.get("set_active_character") or "").strip().lower() == char_id
        vibe = str(style.get("vibe") or "balanced").strip().lower()
        if vibe not in self.VALID_VIBES:
            vibe = "balanced"
        technicality = self._parse_level(style.get("technicality"))
        energy = self._parse_level(style.get("energy"))
        if technicality is None:
            technicality = 0.55
        if energy is None:
            energy = 0.55

        source_data = {
            "character_id": char_id,
            "display_name": display_name,
            "vibe": vibe,
            "technicality": float(technicality),
            "energy": float(energy),
            "default_mode": default_mode,
            "extra_modes": list(dict.fromkeys([x for x in extra_modes if x and x != default_mode])),
            "llm_profile": llm_profile,
            "set_active": bool(set_active),
            "operation_type": self.OP_BUILD_PACK,
            "targets": {"character_ids": [char_id], "mode_ids": [], "scope": "character"},
        }

        blueprint: dict[str, Any] = {}
        if provider is not None:
            blueprint = self._generate_pack_blueprint_with_llm(data=source_data, provider=provider, model=model)
        if not blueprint:
            blueprint = self._fallback_pack_blueprint(source_data=source_data)

        spec_dir = (self.character_specs_root / char_id).resolve()
        spec_dir.mkdir(parents=True, exist_ok=True)
        character_path = spec_dir / "character.json"
        existed_before = character_path.exists()
        blueprint_path = spec_dir / "studio_blueprint.json"
        blueprint_payload = {
            "schema_version": 1,
            "program": "studio_generator",
            "created_at": now_local_iso(),
            "operation_type": self.OP_BUILD_PACK,
            "character_id": char_id,
            "source": source_data,
            "blueprint": blueprint,
        }
        self._write_json(blueprint_path, blueprint_payload)
        written_files = [str(blueprint_path)]
        written_files.extend(self._materialize_pack_artifacts(spec_dir=spec_dir, payload=blueprint, source_data=source_data))

        taxonomy_path = (self.specs_root / "taxonomy.json").resolve()
        modes_spec_path = (self.specs_root / "modes_spec.json").resolve()
        taxonomy = self._read_json(taxonomy_path)
        modes_spec = self._read_json(modes_spec_path)
        taxonomy.setdefault("modes", [])
        taxonomy.setdefault("aliases", {})
        aliases = dict(taxonomy.get("aliases") or {})
        aliases_modes = dict(aliases.get("modes") or {})
        aliases_modes.setdefault("chat", "chatting")
        aliases_modes.setdefault("task", "helper")
        aliases_modes.setdefault("coding", "engineer")
        aliases_modes.setdefault("debug", "debugger")
        aliases["modes"] = aliases_modes
        taxonomy["aliases"] = aliases
        modes_spec.setdefault("schema_version", 1)
        modes_map = dict(modes_spec.get("modes") or {})
        known_modes = [self._mode_id(x) for x in list(taxonomy.get("modes") or []) if self._mode_id(x)]
        added_modes: list[str] = []
        all_modes = [default_mode] + [x for x in list(source_data.get("extra_modes") or []) if x]
        for mode_id in list(dict.fromkeys([self._mode_id(x) for x in all_modes if self._mode_id(x)])):
            if mode_id not in known_modes:
                known_modes.append(mode_id)
                added_modes.append(mode_id)
            if mode_id not in modes_map:
                modes_map[mode_id] = self._default_mode_spec_entry(mode_id)
        taxonomy["modes"] = list(dict.fromkeys([x for x in known_modes if x]))
        modes_spec["modes"] = modes_map
        self._write_json(taxonomy_path, taxonomy)
        self._write_json(modes_spec_path, modes_spec)
        written_files.extend([str(taxonomy_path), str(modes_spec_path)])

        set_active_character = char_id if bool(set_active) else ""
        if set_active_character and self.character_specs_root == self.default_character_specs_root:
            self.storage.sync_manifest()
            manifest = self.storage.load_manifest()
            manifest["active_character_id"] = set_active_character
            self.storage.save_manifest(manifest)

        spec_character = self._read_json(spec_dir / "character.json")
        runtime_path = self._sync_runtime_character_payload(
            char_id=char_id,
            character=spec_character or {"id": char_id, "name": display_name, "llm_profile": llm_profile, "default_mode": default_mode, "default_mood": "neutral"},
        )
        if runtime_path:
            written_files.append(runtime_path)

        written_files = list(dict.fromkeys(written_files))
        return {
            "operation_type": self.OP_BUILD_PACK,
            "targets": {"character_ids": [char_id], "mode_ids": [], "scope": "character"},
            "planned_changes": planned or [f"build_character_pack {char_id}"],
            "written_files": written_files,
            "added_modes": sorted(list(dict.fromkeys([x for x in added_modes if x]))),
            "updated_modes": [],
            "removed_modes": [],
            "created_characters": [char_id] if not existed_before else [],
            "updated_characters": [char_id] if existed_before else [],
            "set_active_character": set_active_character,
            "build_pack": {"character_id": char_id, "blueprint_path": str(blueprint_path)},
        }

    def _sync_runtime_character_payload(self, *, char_id: str, character: dict[str, Any]) -> str:
        cid = self._safe_id(char_id)
        if not cid:
            return ""
        try:
            self.storage.ensure_character_structure(cid)
            runtime_path = (self.storage.character_dir(cid) / "character.json").resolve()
        except Exception:
            return ""

        current = self._read_json(runtime_path)
        if not current:
            current = {"id": cid}
        before = dict(current)

        current["id"] = cid
        current["name"] = str(character.get("name") or self._display_name(cid)).strip() or self._display_name(cid)
        current["default_mood"] = str(character.get("default_mood") or current.get("default_mood") or "neutral").strip() or "neutral"
        current["llm_profile"] = self._safe_llm_profile(character.get("llm_profile"))
        default_mode = self._mode_id(character.get("default_mode"))
        if default_mode:
            current["default_mode"] = default_mode

        if current != before:
            self._write_json(runtime_path, current)
            return str(runtime_path)
        return ""

    def _generate_pack_blueprint_with_llm(
        self,
        *,
        data: dict[str, Any],
        provider: LLMProviderBase,
        model: str,
    ) -> dict[str, Any]:
        system_prompt = (
            "You are Studio character-pack generator. "
            "Return only strict JSON object with keys: "
            "character, modes, dialog_policy, prompts, samples, artifacts. "
            "artifacts must be relative paths under character directory, kind=json|text, action=create_or_update."
        )
        user_prompt = "Generate a complete character pack for this data:\n" + json.dumps(data, ensure_ascii=False)
        try:
            result = run_task_model_json(
                "studio_pack_blueprint",
                user_prompt,
                system_prompt=system_prompt,
                metadata={"think": True, "studio_specs_task": "build_pack"},
                required_fields=("character", "modes", "dialog_policy", "prompts", "samples", "artifacts"),
                max_output_chars=24000,
                max_retries=1,
            )
            return self._coerce_json_object(result.json_payload)
        except Exception:
            pass
        req = LLMRequest(
            model=str(model or "").strip(),
            messages=[
                Message(
                    role="system",
                    content=(
                        "You are Studio character-pack generator. "
                        "Return only strict JSON object with keys: "
                        "character, modes, dialog_policy, prompts, samples, artifacts. "
                        "artifacts must be relative paths under character directory, kind=json|text, action=create_or_update."
                    ),
                ),
                Message(
                    role="user",
                    content=(
                        "Generate a complete character pack for this data:\n"
                        + json.dumps(data, ensure_ascii=False)
                    ),
                ),
            ],
            json_mode=True,
            max_tokens=-1,
            temperature=0.55,
            top_p=0.9,
            metadata={"think": True, "studio_specs_task": "build_pack"},
        )
        try:
            resp = provider.generate(req)
        except Exception:
            return {}
        payload = self._extract_json_object(str(resp.text or ""))
        return dict(payload) if isinstance(payload, dict) else {}

    def _fallback_pack_blueprint(self, *, source_data: dict[str, Any]) -> dict[str, Any]:
        modes = [self._mode_id(source_data.get("default_mode"))] + [self._mode_id(x) for x in list(source_data.get("extra_modes") or [])]
        modes = [x for x in list(dict.fromkeys(modes)) if x]
        return {
            "character": {
                "id": str(source_data.get("character_id") or ""),
                "name": str(source_data.get("display_name") or ""),
                "vibe": str(source_data.get("vibe") or "balanced"),
                "llm_profile": str(source_data.get("llm_profile") or self._default_llm_profile()),
            },
            "modes": [{"id": mid, "description": f"Mode {mid}"} for mid in modes],
            "dialog_policy": {
                "do": ["Keep role consistency.", "Ask clarifying questions when requirements are ambiguous."],
                "avoid": ["Do not silently drop requested constraints."],
                "escalation": ["If data is missing, ask one focused follow-up question."],
            },
            "prompts": {
                "system": f"You are {source_data.get('display_name')}.",
                "style": "Keep concise, actionable responses.",
                "boundaries": "Follow safety policy and avoid unsafe explicit content.",
            },
            "samples": {
                "opener": "Готова помочь. Что именно настроим?",
                "clarification_question": "Уточни, какой режим нужен по умолчанию?",
                "refusal_safe": "Не могу помочь с этим, но предложу безопасную альтернативу.",
            },
            "artifacts": self._default_pack_artifacts(source_data=source_data),
        }

    def _materialize_pack_artifacts(
        self,
        *,
        spec_dir: Path,
        payload: dict[str, Any],
        source_data: dict[str, Any],
    ) -> list[str]:
        written: list[str] = []
        artifacts = self._coerce_artifacts(payload.get("artifacts"))
        if not artifacts:
            artifacts = self._default_pack_artifacts(source_data=source_data)
        for item in list(artifacts):
            rel = self._safe_rel_path(item.get("path"))
            if not rel:
                continue
            target = (spec_dir / rel).resolve()
            if target != spec_dir and spec_dir not in target.parents:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            kind = str(item.get("kind") or "json").strip().lower()
            action = str(item.get("action") or "create_or_update").strip().lower()
            content = item.get("content")
            if kind == "json":
                obj = dict(content) if isinstance(content, dict) else {}
                if action == "create_or_update" and target.exists():
                    existing = self._read_json(target)
                    if existing:
                        obj = self._patch_merge(existing, obj)
                self._write_json(target, obj)
            else:
                text = str(content or "").strip()
                if action == "create_or_update" and target.exists():
                    old = target.read_text(encoding="utf-8") if target.exists() else ""
                    if old and text and text not in old:
                        text = old.rstrip() + "\n" + text
                target.write_text(text, encoding="utf-8")
            written.append(str(target))
        return written

    def _default_pack_artifacts(self, *, source_data: dict[str, Any]) -> list[dict[str, Any]]:
        cid = self._safe_id(source_data.get("character_id"))
        name = str(source_data.get("display_name") or self._display_name(cid)).strip() or self._display_name(cid)
        default_mode = self._mode_id(source_data.get("default_mode")) or "helper"
        extra_modes = [self._mode_id(x) for x in list(source_data.get("extra_modes") or []) if self._mode_id(x)]
        llm_profile = str(source_data.get("llm_profile") or self._default_llm_profile()).strip().upper()
        if llm_profile not in self._llm_profile_set():
            llm_profile = self._default_llm_profile()
        vibe = str(source_data.get("vibe") or "balanced").strip().lower()
        technicality = self._parse_level(source_data.get("technicality"))
        energy = self._parse_level(source_data.get("energy"))
        if technicality is None:
            technicality = 0.55
        if energy is None:
            energy = 0.55

        character = self._default_character_json(cid)
        character = self._patch_merge(
            character,
            {
                "name": name,
                "default_mode": default_mode,
                "llm_profile": llm_profile,
            },
        )
        persona_state = self._default_persona_state_json(cid, character_name=name)
        traits = dict(persona_state.get("traits") or {})
        self._apply_style_knobs(traits=traits, vibe=vibe, technicality=float(technicality), energy=float(energy))
        persona_state["traits"] = traits
        persona_spec = self._default_persona_spec_json(character_name=name)
        modes_map: dict[str, list[str]] = {}
        all_modes = [default_mode] + [x for x in extra_modes if x and x != default_mode]
        for mode_id in list(dict.fromkeys(all_modes)):
            modes_map[mode_id] = self._default_mode_lines(mode_id)
        persona_spec["modes"] = modes_map
        evolution = self._default_evolution_json()

        return [
            {"path": "character.json", "kind": "json", "action": "create_or_update", "content": character},
            {"path": "persona_state.json", "kind": "json", "action": "create_or_update", "content": persona_state},
            {"path": "persona_spec.json", "kind": "json", "action": "create_or_update", "content": persona_spec},
            {"path": "evolution_spec.json", "kind": "json", "action": "create_or_update", "content": evolution},
        ]

    @staticmethod
    def _coerce_artifacts(value: Any) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for raw in list(value or []):
            if not isinstance(raw, dict):
                continue
            out.append(
                {
                    "path": str(raw.get("path") or "").strip(),
                    "kind": str(raw.get("kind") or "json").strip().lower(),
                    "action": str(raw.get("action") or "create_or_update").strip().lower(),
                    "content": raw.get("content"),
                }
            )
        return out

    @staticmethod
    def _safe_rel_path(value: Any) -> str:
        raw = str(value or "").strip().replace("\\", "/")
        if not raw or raw.startswith("/") or raw.startswith("../") or "/../" in raw or "..\\" in raw:
            return ""
        return raw

    def _state_from(self, state: dict[str, Any]) -> dict[str, Any]:
        row = dict(dict(state or {}).get(self.KEY) or {})
        if not row:
            return self._empty_state()
        out = self._empty_state()
        out.update(row)
        out["active"] = bool(out.get("active", False))
        out["phase"] = str(out.get("phase") or self.PHASE_SEED)
        out["seed_prompt"] = str(out.get("seed_prompt") or "")
        out["command_alias"] = self._normalize_command_alias(out.get("command_alias"), fallback="studio")
        out["operation_type"] = str(out.get("operation_type") or "")
        out["targets"] = self._sanitize_targets(out.get("targets"))
        out["draft_changes"] = self._sanitize_draft_changes(out.get("draft_changes"))
        out["data"] = dict(out.get("data") or {})
        out["confidence_map"] = self._sanitize_confidence_map(out.get("confidence_map"))
        out["unresolved_fields"] = [str(x).strip() for x in list(out.get("unresolved_fields") or []) if str(x).strip()]
        out["clarify_index"] = max(0, int(out.get("clarify_index", 0)))
        out["current_field"] = str(out.get("current_field") or "")
        out["step_options"] = dict(out.get("step_options") or {})
        out["awaiting_custom_for"] = str(out.get("awaiting_custom_for") or "")
        out["last_llm"] = dict(out.get("last_llm") or {})
        out["last_result"] = dict(out.get("last_result") or {})
        out["target_candidates"] = {
            "character_ids": [self._safe_id(x) for x in list(dict(out.get("target_candidates") or {}).get("character_ids") or []) if self._safe_id(x)],
            "mode_ids": [self._mode_id(x) for x in list(dict(out.get("target_candidates") or {}).get("mode_ids") or []) if self._mode_id(x)],
            "missing_update_characters": [self._safe_id(x) for x in list(dict(out.get("target_candidates") or {}).get("missing_update_characters") or []) if self._safe_id(x)],
        }
        return out

    @staticmethod
    def _empty_state() -> dict[str, Any]:
        return {
            "active": False,
            "phase": StudioGenerator.PHASE_SEED,
            "created_at": now_local_iso(),
            "updated_at": now_local_iso(),
            "seed_prompt": "",
            "command_alias": "studio",
            "operation_type": "",
            "targets": {"character_ids": [], "mode_ids": [], "scope": "mixed"},
            "draft_changes": StudioGenerator._empty_draft_changes(),
            "data": {},
            "confidence_map": {},
            "unresolved_fields": [],
            "clarify_index": 0,
            "current_field": "",
            "step_options": {},
            "awaiting_custom_for": "",
            "last_llm": {},
            "last_result": {},
            "target_candidates": {"character_ids": [], "mode_ids": [], "missing_update_characters": []},
        }

    @staticmethod
    def _empty_draft_changes() -> dict[str, Any]:
        return {
            "characters": {},
            "modes": {},
            "set_active_character": "",
        }

    def _sanitize_targets(self, value: Any) -> dict[str, Any]:
        src = dict(value or {}) if isinstance(value, dict) else {}
        scope = str(src.get("scope") or "mixed").strip().lower()
        if scope not in self.VALID_SCOPE:
            scope = "mixed"
        return {
            "character_ids": [self._safe_id(x) for x in list(src.get("character_ids") or []) if self._safe_id(x)],
            "mode_ids": [self._mode_id(x) for x in list(src.get("mode_ids") or []) if self._mode_id(x)],
            "scope": scope,
        }

    def _sanitize_draft_changes(self, value: Any) -> dict[str, Any]:
        src = dict(value or {}) if isinstance(value, dict) else {}
        out = self._empty_draft_changes()
        chars = dict(src.get("characters") or {})
        for cid, raw in chars.items():
            char_id = self._safe_id(cid)
            if not char_id:
                continue
            row = dict(raw or {})
            out["characters"][char_id] = {
                "action": "create" if str(row.get("action") or "").strip().lower() == "create" else "update",
                "patch": dict(row.get("patch") or {}),
                "persona_state_patch": dict(row.get("persona_state_patch") or {}),
                "persona_spec_patch": dict(row.get("persona_spec_patch") or {}),
                "style": dict(row.get("style") or {}),
                "extra_modes": [self._mode_id(x) for x in list(row.get("extra_modes") or []) if self._mode_id(x)],
                "set_active": bool(row.get("set_active", False)),
            }
        modes = dict(src.get("modes") or {})
        for mid, raw in modes.items():
            mode_id = self._mode_id(mid)
            if not mode_id:
                continue
            row = dict(raw or {})
            patch = self._patch_merge(self._default_mode_spec_entry(mode_id), dict(row.get("patch") or {}))
            out["modes"][mode_id] = {
                "action": self._coerce_field_value(field_id="mode_action", value=row.get("action")) or "update",
                "patch": dict(patch or {}),
            }
        out["set_active_character"] = self._safe_id(src.get("set_active_character"))
        return out

    def _snapshot_data_from_draft(self, row: dict[str, Any]) -> dict[str, Any]:
        data = dict(row.get("data") or {})
        data["operation_type"] = str(row.get("operation_type") or "")
        targets = dict(row.get("targets") or {})
        data["targets"] = {
            "character_ids": list(targets.get("character_ids") or []),
            "mode_ids": list(targets.get("mode_ids") or []),
            "scope": str(targets.get("scope") or "mixed"),
        }
        chars = dict(dict(row.get("draft_changes") or {}).get("characters") or {})
        if chars:
            first_cid = list(chars.keys())[0]
            first = dict(chars.get(first_cid) or {})
            patch = dict(first.get("patch") or {})
            style = dict(first.get("style") or {})
            data["character_id"] = first_cid
            data["display_name"] = str(patch.get("name") or self._display_name(first_cid))
            data["default_mode"] = str(patch.get("default_mode") or "")
            data["llm_profile"] = str(patch.get("llm_profile") or "")
            data["vibe"] = str(style.get("vibe") or "")
            data["technicality"] = style.get("technicality")
            data["energy"] = style.get("energy")
            data["extra_modes"] = list(first.get("extra_modes") or [])
            data["set_active"] = bool(first.get("set_active", False))
        modes = dict(dict(row.get("draft_changes") or {}).get("modes") or {})
        if modes:
            first_mid = list(modes.keys())[0]
            first_mode = dict(modes.get(first_mid) or {})
            mode_patch = dict(first_mode.get("patch") or {})
            data["mode_id"] = first_mid
            data["mode_action"] = str(first_mode.get("action") or "")
            data["mode_description"] = str(mode_patch.get("description") or "")
        return data

    @staticmethod
    def _extract_json_object(text: str) -> dict[str, Any]:
        raw = str(text or "").strip()
        if not raw:
            return {}
        try:
            obj = json.loads(raw)
            return dict(obj) if isinstance(obj, dict) else {}
        except Exception:
            pass
        start = raw.find("{")
        end = raw.rfind("}")
        if start < 0 or end <= start:
            return {}
        try:
            obj = json.loads(raw[start : end + 1])
        except Exception:
            return {}
        return dict(obj) if isinstance(obj, dict) else {}

    @staticmethod
    def _response_telemetry(resp: Any) -> dict[str, Any]:
        usage = getattr(resp, "usage", None)
        return {
            "thinking": str(getattr(resp, "thinking", "") or ""),
            "prompt_eval_count": int(getattr(usage, "prompt_tokens", 0) or 0) if usage is not None else 0,
            "eval_count": int(getattr(usage, "completion_tokens", 0) or 0) if usage is not None else 0,
            "total_tokens": int(getattr(usage, "total_tokens", 0) or 0) if usage is not None else 0,
            "model": str(getattr(resp, "model", "") or ""),
        }

    @staticmethod
    def _task_result_telemetry(result: Any) -> dict[str, Any]:
        response = getattr(result, "response", None)
        out = StudioGenerator._response_telemetry(response)
        profile_name = str(getattr(result, "profile_name", "") or "").strip()
        if profile_name:
            out["task_profile"] = profile_name
        out["used_fallback"] = bool(getattr(result, "used_fallback", False))
        return out

    @staticmethod
    def _coerce_json_object(payload: Any) -> dict[str, Any]:
        return dict(payload) if isinstance(payload, dict) else {}

    def _parse_review_edit(self, text: str, *, row: dict[str, Any]) -> tuple[str, Any] | None:
        src = str(text or "").strip()
        if not src:
            return None
        match = re.match(r"^\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:\s*(.+)$", src)
        if not match:
            return None
        field_id = str(match.group(1) or "").strip().lower()
        allowed = set(self._question_templates().keys())
        aliases = {
            "character_id": "target_character_id",
            "mode_id": "target_mode_id",
            "scope": "mode_scope",
            "action": "mode_action",
            "name": "display_name",
        }
        field_id = aliases.get(field_id, field_id)
        if field_id not in allowed:
            return None
        parsed = self._parse_answer(question_id=field_id, text=str(match.group(2) or ""), allow_numbered=False, row=row)
        if parsed is None:
            return None
        return field_id, parsed

    @staticmethod
    def _parse_index(value: str) -> int | None:
        src = str(value or "").strip()
        match = re.match(r"^(\d+)\s*[.)]?\s*$", src)
        if not match:
            return None
        try:
            return int(str(match.group(1)))
        except Exception:
            return None

    @staticmethod
    def _display_name(character_id: str) -> str:
        cleaned = str(character_id or "").strip().replace("_", " ").replace("-", " ")
        parts = [x for x in cleaned.split(" ") if x]
        if not parts:
            return "Character"
        return " ".join(p[:1].upper() + p[1:] for p in parts)

    @staticmethod
    def _safe_id(value: Any) -> str:
        raw = str(value or "").strip().lower()
        out = "".join(ch for ch in raw if ch.isalnum() or ch in {"_", "-"})
        return out or ""

    @staticmethod
    def _seed_id_from_name(value: Any) -> str:
        raw = str(value or "").strip().lower()
        if not raw:
            return ""
        translit_map = {
            "а": "a",
            "б": "b",
            "в": "v",
            "г": "g",
            "ґ": "g",
            "д": "d",
            "е": "e",
            "ё": "e",
            "є": "ye",
            "ж": "zh",
            "з": "z",
            "и": "i",
            "і": "i",
            "ї": "yi",
            "й": "y",
            "к": "k",
            "л": "l",
            "м": "m",
            "н": "n",
            "о": "o",
            "п": "p",
            "р": "r",
            "с": "s",
            "т": "t",
            "у": "u",
            "ф": "f",
            "х": "h",
            "ц": "ts",
            "ч": "ch",
            "ш": "sh",
            "щ": "sch",
            "ь": "",
            "ы": "y",
            "ъ": "",
            "э": "e",
            "ю": "yu",
            "я": "ya",
        }
        chunks: list[str] = []
        for ch in raw:
            if ch in translit_map:
                chunks.append(str(translit_map[ch]))
                continue
            if ch.isascii():
                chunks.append(ch)
                continue
            if ch.isalnum():
                continue
            chunks.append("_")
        compact = "".join(chunks)
        compact = re.sub(r"[^a-z0-9_-]+", "_", compact)
        compact = re.sub(r"_+", "_", compact).strip("_-")
        if not compact:
            return ""
        if len(compact) > 40:
            compact = compact[:40].rstrip("_-")
        return compact

    def _extract_seed_name_hint(self, *, seed: str) -> str:
        text = str(seed or "").strip()
        if not text:
            return ""
        quoted = re.findall(r"[\"'«»“”]([^\"'«»“”]{2,64})[\"'«»“”]", text)
        for item in quoted:
            token = str(item or "").strip()
            if token:
                words = re.findall(r"[A-Za-zА-Яа-яЁёІіЇїЄєҐґ][A-Za-zА-Яа-яЁёІіЇїЄєҐґ0-9_-]{1,40}", token)
                if words:
                    return str(words[0] or "").strip()
        patterns = [
            r"(?:с\s+именем|имя|зовут|назови|назвать|named|name)\s+([A-Za-zА-Яа-яЁёІіЇїЄєҐґ][A-Za-zА-Яа-яЁёІіЇїЄєҐґ0-9_-]{1,40})",
            r"(?:character|персонаж)\s+([A-Za-z][A-Za-z0-9_-]{1,40})",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if not match:
                continue
            token = str(match.group(1) or "").strip()
            if token:
                return token
        return ""

    def _guess_character_id_from_seed(self, *, seed: str, existing: list[str]) -> str:
        existing_set = {self._safe_id(x) for x in list(existing or []) if self._safe_id(x)}
        hint = self._extract_seed_name_hint(seed=seed)
        if hint:
            hint_id = self._seed_id_from_name(hint)
            if hint_id:
                return hint_id if hint_id not in existing_set else f"{hint_id}_new"
        options = self._seed_character_id_candidates(seed=seed, existing=list(existing_set))
        if options:
            return str(options[0] or "")
        return ""

    @staticmethod
    def _mode_id(value: Any) -> str:
        raw = str(value or "").strip().lower().replace(" ", "_")
        out = "".join(ch for ch in raw if ch.isalnum() or ch in {"_", "-"})
        return out or ""

    def _available_llm_profiles(self) -> list[str]:
        names: list[str] = []
        seen: set[str] = set()
        for key in list(get_model_profiles().keys()):
            token = str(key or "").strip().upper()
            if not token or token in seen:
                continue
            seen.add(token)
            names.append(token)
        if not names:
            names = [self.DEFAULT_LLM_PROFILE]
        self._llm_profiles_cache = list(names)
        return list(names)

    def _llm_profile_set(self) -> set[str]:
        return set(self._available_llm_profiles())

    def _default_llm_profile(self) -> str:
        names = self._available_llm_profiles()
        if self.DEFAULT_LLM_PROFILE in names:
            return self.DEFAULT_LLM_PROFILE
        return str(names[0])

    def _safe_llm_profile(self, value: Any) -> str:
        token = str(value or "").strip().upper()
        return token if token in self._llm_profile_set() else self._default_llm_profile()

    def _llm_profile_enum_for_prompt(self) -> str:
        values = self._available_llm_profiles()
        return "|".join(values)

    def _known_modes(self) -> list[str]:
        taxonomy = self._read_json(self.specs_root / "taxonomy.json")
        raw = [self._mode_id(x) for x in list(taxonomy.get("modes") or []) if self._mode_id(x)]
        if raw:
            return list(dict.fromkeys(raw))
        return ["chatting", "helper", "engineer", "debugger", "planner", "spicy_chat"]

    def _known_character_ids(self) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()
        root = self.character_specs_root
        if root.exists():
            for row in sorted(root.iterdir(), key=lambda p: p.name.lower()):
                if not row.is_dir() or row.name.startswith("_"):
                    continue
                cid = self._safe_id(row.name)
                if not cid or cid in seen:
                    continue
                seen.add(cid)
                out.append(cid)
        return out

    def _first_character_target(self, *, row: dict[str, Any], create_if_missing: bool) -> str:
        targets = dict(row.get("targets") or {})
        chars = [self._safe_id(x) for x in list(targets.get("character_ids") or []) if self._safe_id(x)]
        draft = dict(row.get("draft_changes") or self._empty_draft_changes())
        if chars:
            cid = chars[0]
            draft.setdefault("characters", {})
            if cid not in dict(draft.get("characters") or {}) and create_if_missing:
                draft["characters"][cid] = self._default_character_change(cid=cid, create=(str(row.get("operation_type") or "") == self.OP_CREATE_CHARACTER))
                row["draft_changes"] = draft
            return cid
        if create_if_missing:
            seed = str(row.get("seed_prompt") or "").strip()
            guess = self._guess_character_id_from_seed(seed=seed, existing=self._known_character_ids())
            if not guess:
                guess = "character"
            targets["character_ids"] = [guess]
            row["targets"] = targets
            draft.setdefault("characters", {})
            draft["characters"][guess] = self._default_character_change(cid=guess, create=True)
            row["draft_changes"] = draft
            return guess
        return ""

    def _default_character_change(self, *, cid: str, create: bool) -> dict[str, Any]:
        return {
            "action": "create" if create else "update",
            "patch": {
                "name": self._display_name(cid),
                "default_mode": "helper",
                "default_mood": "neutral",
                "llm_profile": self._default_llm_profile(),
            },
            "persona_state_patch": {},
            "persona_spec_patch": {"modes_add": ["helper"]},
            "style": {"vibe": "balanced", "technicality": 0.55, "energy": 0.55},
            "extra_modes": [],
            "set_active": bool(create),
        }

    @staticmethod
    def _default_mode_scope_character_change() -> dict[str, Any]:
        return {
            "action": "update",
            "patch": {},
            "persona_state_patch": {},
            "persona_spec_patch": {"modes_add": []},
            "style": {},
            "extra_modes": [],
            "set_active": False,
        }

    def _default_mode_change(self, *, mode_id: str) -> dict[str, Any]:
        return {
            "action": "add" if mode_id not in set(self._known_modes()) else "update",
            "patch": dict(self._default_mode_spec_entry(mode_id)),
        }

    @staticmethod
    def _default_mode_lines(mode: str) -> list[str]:
        key = str(mode or "").strip().lower()
        presets = {
            "chatting": ["Keep friendly conversational tone.", "Light humor is allowed when relevant."],
            "helper": ["Prioritize support and clarity.", "Keep warm tone and reduce sarcasm."],
            "engineer": ["Respond in structured technical format.", "Prefer concrete steps and concise explanations."],
            "debugger": ["Use hypothesis-driven debugging flow.", "Ask for diagnostics and reproduction steps."],
            "planner": ["Build plans with phases and checkpoints.", "Provide acceptance criteria."],
            "spicy_chat": ["Use a bold playful tone.", "Keep safety boundaries.", "Do not switch to explicit unsafe content."],
        }
        return list(presets.get(key) or [f"Mode {key}: keep consistent style and clear boundaries."])

    def _default_mode_spec_entry(self, mode_id: str) -> dict[str, Any]:
        mid = self._mode_id(mode_id) or "chatting"
        profile = resolve_mode_profile(mode=mid)
        return {
            "description": f"Custom mode: {mid}",
            "legacy_mode": "chat",
            "output_format_default": {"show_parameters": False, "show_summary": False},
            "persona_effects": {
                "trait_targets": dict(profile.trait_targets or {}),
                "dialog_targets": dict(profile.dialog_targets or {}),
                "prompt_lines": list(profile.prompt_lines or []),
            },
        }

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8-sig"))
        except Exception:
            return {}
        return dict(payload) if isinstance(payload, dict) else {}

    @staticmethod
    def _write_json(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(dict(payload or {}), ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def _patch_merge(base: Any, patch: Any) -> Any:
        if not isinstance(patch, dict):
            return patch
        src = dict(base or {}) if isinstance(base, dict) else {}
        out = dict(src)
        for key, value in dict(patch).items():
            if isinstance(value, dict) and isinstance(out.get(key), dict):
                out[key] = StudioGenerator._patch_merge(out.get(key), value)
            else:
                out[key] = value
        return out

    def _default_character_json(self, cid: str) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "character_id": cid,
            "id": cid,
            "name": self._display_name(cid),
            "default_mood": "neutral",
            "default_mode": "chatting",
            "llm_profile": self._default_llm_profile(),
            "locks": {"feminine": True, "informal_you": True},
        }

    def _default_persona_state_json(self, cid: str, *, character_name: str) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "character_id": cid,
            "name": character_name,
            "mood": "neutral",
            "traits": {
                "warmth": 0.62,
                "sarcasm": 0.2,
                "teasing": 0.26,
                "verbosity": 0.5,
                "strictness": 0.54,
                "empathy": 0.64,
            },
            "locks": {"feminine": True, "informal_you": True},
            "bans": [],
            "learned": {"preferences_confirmed": [], "preferences_pending": [], "style_bias": {}},
        }

    @staticmethod
    def _default_persona_spec_json(*, character_name: str) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "identity": [f"You are {character_name}."],
            "locks_map": {
                "feminine": "Always use feminine grammatical gender for self-reference.",
                "informal_you": "Use informal address form ('ты').",
            },
            "bans_template": "Do not use banned word: {term}.",
            "moods": {"neutral": ["Tone: neutral. Keep balanced and practical tone."]},
            "modes": {},
            "traits_rules": {},
        }

    @staticmethod
    def _default_evolution_json() -> dict[str, Any]:
        return {
            "schema_version": 1,
            "rules": [],
            "cleanup": {"remove_if_confidence_below": 0.25, "remove_if_unused_days": 45},
        }

    @staticmethod
    def _apply_style_knobs(*, traits: dict[str, Any], vibe: str, technicality: float, energy: float) -> None:
        out = dict(traits or {})
        for key in (
            "warmth",
            "sarcasm",
            "teasing",
            "verbosity",
            "strictness",
            "empathy",
            "professionalism",
            "directness",
            "energy",
        ):
            out.setdefault(key, 0.55 if key != "sarcasm" else 0.25)
        vibe_key = str(vibe or "balanced").strip().lower()
        if vibe_key == "playful":
            out["playfulness"] = min(1.0, float(out.get("playfulness", 0.4)) + 0.15)
            out["teasing"] = min(1.0, float(out.get("teasing", 0.3)) + 0.1)
        elif vibe_key == "strict":
            out["strictness"] = min(1.0, float(out.get("strictness", 0.5)) + 0.14)
            out["warmth"] = max(0.0, float(out.get("warmth", 0.6)) - 0.08)
        elif vibe_key == "soft":
            out["warmth"] = min(1.0, float(out.get("warmth", 0.6)) + 0.12)
            out["strictness"] = max(0.0, float(out.get("strictness", 0.5)) - 0.08)
        tech = max(0.0, min(1.0, float(technicality)))
        out["professionalism"] = max(0.0, min(1.0, 0.35 + tech * 0.6))
        out["directness"] = max(0.0, min(1.0, 0.3 + tech * 0.55))
        out["energy"] = max(0.0, min(1.0, float(energy)))
        for key, value in out.items():
            try:
                traits[key] = max(0.0, min(1.0, float(value)))
            except Exception:
                continue

    def _detect_character_mentions(self, *, text: str, existing_chars: list[str]) -> list[str]:
        out: list[str] = []
        low = str(text or "").lower()
        for cid in list(existing_chars or []):
            key = self._safe_id(cid)
            if key and key in low and key not in out:
                out.append(key)
        explicit = re.findall(r"(?:character|персонаж|перса|у|для|for)\s+([a-zA-Z0-9_-]{2,40})", low)
        for raw in explicit:
            key = self._safe_id(raw)
            if key and key not in out:
                out.append(key)
        return out

    def _detect_mode_mentions(self, *, text: str, known_modes: list[str]) -> list[str]:
        out: list[str] = []
        low = str(text or "").lower()
        for mid in list(known_modes or []):
            key = self._mode_id(mid)
            if key and key in low and key not in out:
                out.append(key)
        explicit = re.findall(r"(?:mode|мод|режим)\s+([a-zA-Z0-9_-]{2,40})", low)
        for raw in explicit:
            key = self._mode_id(raw)
            if key and key not in out:
                out.append(key)
        return out

    def _normalize_scope_from_seed(
        self,
        *,
        seed: str,
        operation_type: str,
        requested_chars: list[str],
        resolved_chars: list[str],
        scope: str,
    ) -> str:
        op = str(operation_type or "").strip().lower()
        scope_token = str(scope or "").strip().lower()
        if scope_token not in self.VALID_SCOPE:
            if op == self.OP_UPDATE_MODES:
                scope_token = "global"
            elif op == self.OP_MIXED:
                scope_token = "mixed"
            else:
                scope_token = "character"
        if op != self.OP_UPDATE_MODES:
            return scope_token
        low = str(seed or "").strip().lower()
        has_character_hint = bool(list(requested_chars or [])) or bool(list(resolved_chars or [])) or ("для " in low) or ("for " in low)
        if not has_character_hint:
            return scope_token
        has_global_hint = any(x in low for x in ["global", "глобаль", "везде"])
        if has_global_hint:
            return "mixed"
        if scope_token == "global":
            return "character"
        return scope_token

    def _infer_mode_action_from_seed(self, *, seed: str, mode_id: str, known_modes: list[str]) -> str:
        low = str(seed or "").strip().lower()
        mid = self._mode_id(mode_id)
        if any(x in low for x in ["remove", "delete", "удали", "удалить", "убери"]):
            return "remove"
        if any(x in low for x in ["add ", "добав", "создай режим", "создать режим"]):
            return "add"
        if any(x in low for x in ["update", "edit", "обнов", "измени", "поправ"]):
            return "update"
        return "update" if mid in set(self._mode_id(x) for x in list(known_modes or [])) else "add"

    def _infer_operation_type(self, *, seed: str) -> str:
        low = str(seed or "").strip().lower()
        has_pack = any(x in low for x in ["pack", "blueprint", "artifact", "артефакт", "блюпринт", "пак"])
        has_update = any(x in low for x in ["update", "change", "измени", "обнов", "поправ", "редакт"])
        has_mode = any(x in low for x in ["mode", "мод", "режим"])
        has_create = any(x in low for x in ["create", "созда", "сделай", "новый"])
        if has_pack:
            return self.OP_BUILD_PACK
        if has_update and has_mode:
            return self.OP_MIXED
        if has_mode and not any(x in low for x in ["персонаж", "character"]):
            return self.OP_UPDATE_MODES
        if has_update and not has_create:
            return self.OP_UPDATE_CHARACTER
        if has_mode and has_create:
            return self.OP_MIXED
        return self.OP_CREATE_CHARACTER

    @staticmethod
    def _infer_character_action_intent(*, seed: str) -> str:
        low = str(seed or "").strip().lower()
        has_create = any(x in low for x in ["create", "new", "создай", "создать", "новый"])
        has_update = any(x in low for x in ["update", "change", "edit", "обнови", "обновить", "измени", "изменить", "поправ"])
        if has_create and not has_update:
            return "create"
        if has_update and not has_create:
            return "update"
        return "unknown"

    @staticmethod
    def _needs_operation_type_clarify(*, seed: str, operation_type: str) -> bool:
        op = str(operation_type or "").strip().lower()
        if op in {StudioGenerator.OP_UPDATE_MODES, StudioGenerator.OP_MIXED}:
            return False
        low = str(seed or "").strip().lower()
        has_create = any(x in low for x in ["create", "new", "создай", "создать", "новый"])
        has_update = any(x in low for x in ["update", "change", "edit", "обнови", "обновить", "измени", "изменить", "поправ"])
        if has_create and has_update:
            return True
        has_character_context = any(x in low for x in ["character", "персонаж", "перса", "для ", "for "])
        has_mode_context = any(x in low for x in ["mode", "мод", "режим"])
        return bool(has_character_context and not has_mode_context and not has_create and not has_update)

    @staticmethod
    def _parse_level(value: Any) -> float | None:
        token = str(value or "").strip().lower()
        mapping = {"low": 0.35, "mid": 0.55, "medium": 0.55, "high": 0.8}
        if token in mapping:
            return float(mapping[token])
        try:
            num = float(token.replace(",", "."))
        except Exception:
            return None
        return max(0.0, min(1.0, num))

    @staticmethod
    def _normalize_command_alias(value: Any, fallback: Any = "studio") -> str:
        token = str(value or fallback or "studio").strip().lower()
        return "studio"

    @staticmethod
    def _clamp_float(value: Any, *, default: float) -> float:
        try:
            out = float(value)
        except Exception:
            out = float(default)
        return max(0.0, min(1.0, out))

    @staticmethod
    def _clamp_int(value: Any, *, default: int) -> int:
        try:
            return int(value)
        except Exception:
            return int(default)

    def _sanitize_confidence_map(self, value: Any) -> dict[str, float]:
        src = dict(value or {}) if isinstance(value, dict) else {}
        out: dict[str, float] = {}
        for key, raw in src.items():
            name = str(key or "").strip().lower()
            if not name:
                continue
            out[name] = self._clamp_float(raw, default=0.0)
        return out

    def _cmd(self, row: dict[str, Any]) -> str:
        return self._normalize_command_alias(row.get("command_alias"), fallback="studio")


__all__ = ["StudioGenerator", "StudioGeneratorReply"]


