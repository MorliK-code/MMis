from __future__ import annotations

from _output_utils import enable_unittest_json_output

enable_unittest_json_output()

import json
import tempfile
import unittest
from pathlib import Path

from config.settings import get_model_profiles
from llm.provider_base import (
    LLMProviderBase,
    LLMRequest,
    LLMResponse,
    ModelInfo,
    ProviderHealth,
    Timings,
    Usage,
)
from modules.character.storage import CharacterStorage
from modules.studio.studio_generator import StudioGenerator

PROFILES = get_model_profiles()


class _SpecsProvider(LLMProviderBase):
    def generate(self, req: LLMRequest) -> LLMResponse:
        task = str((req.metadata or {}).get("studio_specs_task") or "").strip()
        if task == "seed_extract":
            user = str((req.messages[-1].content if req.messages else "") or "")
            seed = ""
            for line in user.splitlines():
                if line.startswith("seed="):
                    seed = line[len("seed=") :].strip()
                    break
            payload = self._payload_for_seed(seed)
            text = json.dumps(payload, ensure_ascii=False)
        elif task == "options":
            text = json.dumps({"options": ["variant 1", "variant 2", "variant 3"]}, ensure_ascii=False)
        elif task == "build_pack":
            text = json.dumps({}, ensure_ascii=False)
        else:
            text = "ok"
        return LLMResponse(
            text=text,
            usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            timings=Timings(latency_ms=1.0),
            model="stub",
        )

    @staticmethod
    def _payload_for_seed(seed: str) -> dict:
        low = str(seed or "").lower()
        if "create-case" in low:
            return {
                "operation_type": "create_character",
                "targets": {"character_ids": ["luna_new"], "mode_ids": ["helper", "debugger"], "scope": "character"},
                "character": {
                    "character_id": "luna_new",
                    "display_name": "Luna",
                    "vibe": "playful",
                    "technicality": 0.8,
                    "energy": 0.7,
                    "default_mode": "helper",
                    "extra_modes": ["debugger"],
                    "llm_profile": "QUALITY",
                    "set_active": True,
                },
                "mode_changes": [],
                "confidence": {
                    "operation_type": 0.95,
                    "target_character_id": 0.95,
                    "display_name": 0.95,
                    "default_mode": 0.95,
                    "llm_profile": 0.95,
                    "set_active": 0.95,
                },
            }
        if "update asya" in low:
            return {
                "operation_type": "update_character",
                "targets": {"character_ids": ["asya"], "mode_ids": [], "scope": "character"},
                "character": {
                    "character_id": "asya",
                    "display_name": "Asya",
                    "vibe": "strict",
                    "technicality": 0.9,
                    "energy": 0.5,
                    "default_mode": "engineer",
                    "extra_modes": ["debugger"],
                    "llm_profile": "QUALITY",
                    "set_active": False,
                },
                "mode_changes": [],
                "confidence": {"operation_type": 0.95, "target_character_id": 0.95},
            }
        if "add mode analyst" in low:
            return {
                "operation_type": "update_modes",
                "targets": {"character_ids": [], "mode_ids": ["analyst"], "scope": "global"},
                "character": {},
                "mode_changes": [
                    {
                        "mode_id": "analyst",
                        "action": "add",
                        "description": "Data analysis mode",
                        "legacy_mode": "task",
                        "show_parameters": True,
                        "show_summary": True,
                    }
                ],
                "confidence": {"operation_type": 0.95, "target_mode_id": 0.95, "mode_scope": 0.95},
            }
        if "update mode engineer" in low:
            return {
                "operation_type": "update_modes",
                "targets": {"character_ids": [], "mode_ids": ["engineer"], "scope": "global"},
                "character": {},
                "mode_changes": [
                    {
                        "mode_id": "engineer",
                        "action": "update",
                        "description": "Updated engineer mode",
                        "legacy_mode": "coding",
                        "show_parameters": True,
                        "show_summary": True,
                    }
                ],
                "confidence": {"operation_type": 0.95, "target_mode_id": 0.95, "mode_scope": 0.95},
            }
        if "mixed asya analyst" in low:
            return {
                "operation_type": "mixed",
                "targets": {"character_ids": ["asya"], "mode_ids": ["analyst"], "scope": "mixed"},
                "character": {
                    "character_id": "asya",
                    "display_name": "Asya",
                    "vibe": "balanced",
                    "technicality": 0.7,
                    "energy": 0.6,
                    "default_mode": "helper",
                    "extra_modes": ["analyst"],
                    "llm_profile": "BALANCED",
                    "set_active": True,
                },
                "mode_changes": [
                    {
                        "mode_id": "analyst",
                        "action": "add",
                        "description": "Analyst mixed mode",
                        "legacy_mode": "task",
                        "show_parameters": False,
                        "show_summary": True,
                    }
                ],
                "confidence": {"operation_type": 0.95, "target_character_id": 0.95, "target_mode_id": 0.95, "mode_scope": 0.95},
            }
        if "ambiguous as" in low:
            return {
                "operation_type": "update_character",
                "targets": {"character_ids": ["as"], "mode_ids": [], "scope": "character"},
                "character": {"character_id": "as"},
                "mode_changes": [],
                "confidence": {"operation_type": 0.95, "target_character_id": 0.95},
            }
        return {
            "operation_type": "create_character",
            "targets": {"character_ids": ["demo"], "mode_ids": [], "scope": "character"},
            "character": {"character_id": "demo", "display_name": "Demo", "default_mode": "helper", "llm_profile": "BALANCED", "set_active": False},
            "mode_changes": [],
            "confidence": {"operation_type": 0.95, "target_character_id": 0.95},
        }

    def healthcheck(self) -> ProviderHealth:
        return ProviderHealth(ok=True, provider="stub", detail="ok", model="stub")

    def model_info(self, model: str = "") -> ModelInfo:
        return ModelInfo(provider="stub", model=model or "stub")

    def list_models(self) -> list[str]:
        return ["stub"]


class StudioGeneratorUnifiedWorkflowTests(unittest.TestCase):
    def _prepare_specs_root(self, root: Path) -> None:
        specs = root / "specs"
        chars = specs / "characters"
        chars.mkdir(parents=True, exist_ok=True)
        (specs / "taxonomy.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "modes": ["chatting", "helper", "engineer", "debugger", "planner"],
                    "aliases": {"modes": {"chat": "chatting", "task": "helper", "coding": "engineer", "debug": "debugger"}},
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        (specs / "modes_spec.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "modes": {
                        "engineer": {
                            "description": "Old engineer mode",
                            "legacy_mode": "coding",
                            "output_format_default": {"show_parameters": True, "show_summary": True},
                        }
                    },
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        self._write_character(chars, "asya", "Asya", custom_field="keep_me")
        self._write_character(chars, "aslan", "Aslan")

    @staticmethod
    def _write_character(chars_root: Path, cid: str, name: str, custom_field: str = "") -> None:
        d = chars_root / cid
        d.mkdir(parents=True, exist_ok=True)
        character = {
            "schema_version": 1,
            "character_id": cid,
            "id": cid,
            "name": name,
            "default_mode": "chatting",
            "default_mood": "neutral",
            "llm_profile": "BALANCED",
        }
        if custom_field:
            character["custom_field"] = custom_field
        (d / "character.json").write_text(json.dumps(character, ensure_ascii=False, indent=2), encoding="utf-8")
        (d / "persona_state.json").write_text(
            json.dumps({"schema_version": 1, "character_id": cid, "name": name, "mood": "neutral", "traits": {"warmth": 0.6}}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (d / "persona_spec.json").write_text(
            json.dumps({"schema_version": 1, "identity": [f"You are {name}."], "modes": {"chatting": ["Friendly"]}}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (d / "evolution_spec.json").write_text(json.dumps({"schema_version": 1, "rules": []}, ensure_ascii=False, indent=2), encoding="utf-8")

    def test_create_flow_apply_creates_files(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mmis_studio_unified_create_") as tmp:
            root = Path(tmp)
            self._prepare_specs_root(root)
            storage = CharacterStorage(root=root / "runtime", logs_root=root / "logs")
            studio = StudioGenerator(storage=storage, specs_root=root / "specs", character_specs_root=root / "specs" / "characters")
            provider = _SpecsProvider()

            started = studio.start({}, seed="create-case", provider=provider)
            self.assertIn("Review", started.text)
            applied = studio.apply({StudioGenerator.KEY: dict(started.state or {})}, provider=provider)
            self.assertTrue(applied.done)
            self.assertIn("Studio", applied.text)
            self.assertTrue((root / "specs" / "characters" / "luna_new" / "character.json").exists())
            self.assertTrue((root / "specs" / "characters" / "luna_new" / "studio_blueprint.json").exists())
            persona_spec = json.loads((root / "specs" / "characters" / "luna_new" / "persona_spec.json").read_text(encoding="utf-8"))
            locks_map = dict(persona_spec.get("locks_map") or {})
            self.assertEqual(str(locks_map.get("informal_you") or ""), "Use informal address form ('ты').")
            runtime_character = json.loads((root / "runtime" / "luna_new" / "character.json").read_text(encoding="utf-8"))
            self.assertEqual(str(runtime_character.get("llm_profile")), "QUALITY")

    def test_update_character_uses_patch_merge(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mmis_studio_unified_update_char_") as tmp:
            root = Path(tmp)
            self._prepare_specs_root(root)
            studio = StudioGenerator(specs_root=root / "specs", character_specs_root=root / "specs" / "characters")
            provider = _SpecsProvider()

            started = studio.start({}, seed="update asya strict", provider=provider)
            applied = studio.apply({StudioGenerator.KEY: dict(started.state or {})}, provider=provider)
            self.assertTrue(applied.done)

            updated = json.loads((root / "specs" / "characters" / "asya" / "character.json").read_text(encoding="utf-8"))
            self.assertEqual(str(updated.get("default_mode")), "engineer")
            self.assertEqual(str(updated.get("llm_profile")), "QUALITY")
            self.assertEqual(str(updated.get("custom_field")), "keep_me")
            self.assertTrue((root / "specs" / "characters" / "asya" / "studio_blueprint.json").exists())

    def test_add_mode_global_updates_taxonomy_and_modes_spec(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mmis_studio_unified_add_mode_") as tmp:
            root = Path(tmp)
            self._prepare_specs_root(root)
            studio = StudioGenerator(specs_root=root / "specs", character_specs_root=root / "specs" / "characters")
            provider = _SpecsProvider()

            started = studio.start({}, seed="add mode analyst global", provider=provider)
            applied = studio.apply({StudioGenerator.KEY: dict(started.state or {})}, provider=provider)
            self.assertTrue(applied.done)
            taxonomy = json.loads((root / "specs" / "taxonomy.json").read_text(encoding="utf-8"))
            modes_spec = json.loads((root / "specs" / "modes_spec.json").read_text(encoding="utf-8"))
            self.assertIn("analyst", list(taxonomy.get("modes") or []))
            self.assertIn("analyst", dict(modes_spec.get("modes") or {}))

    def test_update_existing_mode_no_duplicates(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mmis_studio_unified_update_mode_") as tmp:
            root = Path(tmp)
            self._prepare_specs_root(root)
            studio = StudioGenerator(specs_root=root / "specs", character_specs_root=root / "specs" / "characters")
            provider = _SpecsProvider()

            started = studio.start({}, seed="update mode engineer quality", provider=provider)
            applied = studio.apply({StudioGenerator.KEY: dict(started.state or {})}, provider=provider)
            self.assertTrue(applied.done)

            taxonomy = json.loads((root / "specs" / "taxonomy.json").read_text(encoding="utf-8"))
            modes = [x for x in list(taxonomy.get("modes") or []) if str(x) == "engineer"]
            self.assertEqual(len(modes), 1)
            modes_spec = json.loads((root / "specs" / "modes_spec.json").read_text(encoding="utf-8"))
            engineer = dict(dict(modes_spec.get("modes") or {}).get("engineer") or {})
            self.assertEqual(str(engineer.get("description")), "Updated engineer mode")

    def test_mixed_updates_character_and_mode(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mmis_studio_unified_mixed_") as tmp:
            root = Path(tmp)
            self._prepare_specs_root(root)
            studio = StudioGenerator(specs_root=root / "specs", character_specs_root=root / "specs" / "characters")
            provider = _SpecsProvider()

            started = studio.start({}, seed="mixed asya analyst", provider=provider)
            applied = studio.apply({StudioGenerator.KEY: dict(started.state or {})}, provider=provider)
            self.assertTrue(applied.done)

            asya = json.loads((root / "specs" / "characters" / "asya" / "character.json").read_text(encoding="utf-8"))
            self.assertEqual(str(asya.get("default_mode")), "helper")
            taxonomy = json.loads((root / "specs" / "taxonomy.json").read_text(encoding="utf-8"))
            self.assertIn("analyst", list(taxonomy.get("modes") or []))
            self.assertTrue((root / "specs" / "characters" / "asya" / "studio_blueprint.json").exists())

    def test_ambiguous_target_forces_clarify(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mmis_studio_unified_ambiguous_") as tmp:
            root = Path(tmp)
            self._prepare_specs_root(root)
            studio = StudioGenerator(specs_root=root / "specs", character_specs_root=root / "specs" / "characters")
            provider = _SpecsProvider()

            started = studio.start({}, seed="ambiguous as", provider=provider)
            state = dict(started.state or {})
            self.assertEqual(str(state.get("phase")), "clarify")
            self.assertIn("target_character_id", list(state.get("unresolved_fields") or []))

    def test_negative_apply_missing_keeps_studio_active(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mmis_studio_unified_missing_") as tmp:
            root = Path(tmp)
            self._prepare_specs_root(root)
            studio = StudioGenerator(specs_root=root / "specs", character_specs_root=root / "specs" / "characters")

            started = studio.start({}, seed="")
            state = dict(started.state or {})
            applied = studio.apply({StudioGenerator.KEY: state})
            self.assertEqual(str(applied.error or ""), "missing_fields")
            self.assertTrue(bool(dict(applied.state or {}).get("active", False)))

    def test_create_character_explicit_intent_no_redundant_action_question(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mmis_studio_intent_create_") as tmp:
            root = Path(tmp)
            self._prepare_specs_root(root)
            studio = StudioGenerator(specs_root=root / "specs", character_specs_root=root / "specs" / "characters")
            provider = _SpecsProvider()

            started = studio.start({}, seed="создай нового персонажа luna", provider=provider)
            state = dict(started.state or {})
            self.assertEqual(str(state.get("operation_type") or ""), "create_character")
            self.assertNotIn("operation_type", list(state.get("unresolved_fields") or []))

    def test_create_character_named_seed_does_not_use_whole_prompt_as_id(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mmis_studio_named_seed_") as tmp:
            root = Path(tmp)
            self._prepare_specs_root(root)
            studio = StudioGenerator(specs_root=root / "specs", character_specs_root=root / "specs" / "characters")

            started = studio.start({}, seed="создай персонажа с именем Анита, она должна быть более эмоциональной")
            state = dict(started.state or {})
            targets = [str(x).strip().lower() for x in list(dict(state.get("targets") or {}).get("character_ids") or []) if str(x).strip()]
            self.assertTrue(targets)
            self.assertEqual(targets[0], "anita")
            self.assertNotIn("создай", targets[0])
            self.assertNotIn("персонажа", targets[0])

            options = [str(x).strip().lower() for x in list(dict(state.get("step_options") or {}).get("target_character_id") or []) if str(x).strip()]
            self.assertIn("anita", options)

    def test_update_character_explicit_intent_existing_candidates(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mmis_studio_intent_update_") as tmp:
            root = Path(tmp)
            self._prepare_specs_root(root)
            studio = StudioGenerator(specs_root=root / "specs", character_specs_root=root / "specs" / "characters")
            provider = _SpecsProvider()

            started = studio.start({}, seed="ambiguous as", provider=provider)
            state = dict(started.state or {})
            self.assertEqual(str(state.get("phase") or ""), "clarify")
            options = [str(x) for x in list(dict(state.get("step_options") or {}).get("target_character_id") or [])]
            self.assertTrue(options)
            self.assertIn("asya", options)
            self.assertIn("aslan", options)

    def test_target_character_id_create_smart_candidates(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mmis_studio_target_create_") as tmp:
            root = Path(tmp)
            self._prepare_specs_root(root)
            studio = StudioGenerator(specs_root=root / "specs", character_specs_root=root / "specs" / "characters")

            row = studio._empty_state()  # type: ignore[attr-defined]
            row["operation_type"] = studio.OP_CREATE_CHARACTER
            row["seed_prompt"] = "create an assistant for analytics and planning"
            row["target_candidates"] = {"character_ids": [], "mode_ids": [], "missing_update_characters": []}
            options = studio._character_target_options(row=row, provider=None, model="")  # type: ignore[attr-defined]

            self.assertGreaterEqual(len(options), 3)
            self.assertLessEqual(len(options), 5)
            for item in options:
                self.assertEqual(item, item.lower())
                self.assertNotIn(" ", item)

    def test_target_character_id_update_ranked_existing(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mmis_studio_target_update_ranked_") as tmp:
            root = Path(tmp)
            self._prepare_specs_root(root)
            studio = StudioGenerator(specs_root=root / "specs", character_specs_root=root / "specs" / "characters")

            row = studio._empty_state()  # type: ignore[attr-defined]
            row["operation_type"] = studio.OP_UPDATE_CHARACTER
            row["seed_prompt"] = "update asya with stricter style"
            row["target_candidates"] = {"character_ids": [], "mode_ids": [], "missing_update_characters": []}
            options = studio._character_target_options(row=row, provider=None, model="")  # type: ignore[attr-defined]

            self.assertTrue(options)
            self.assertEqual(options[0], "asya")
            self.assertIn("aslan", options)

    def test_mode_add_intent_does_not_ask_add_or_update_again_when_explicit(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mmis_studio_mode_add_intent_") as tmp:
            root = Path(tmp)
            self._prepare_specs_root(root)
            studio = StudioGenerator(specs_root=root / "specs", character_specs_root=root / "specs" / "characters")

            started = studio.start({}, seed="добавь mode analyst для asya")
            state = dict(started.state or {})
            self.assertIn(str(state.get("phase") or ""), {"clarify", "review"})
            self.assertNotIn("mode_action", list(state.get("unresolved_fields") or []))
            mode_targets = list(dict(state.get("targets") or {}).get("mode_ids") or [])
            self.assertIn("analyst", mode_targets)
            self.assertEqual(str(dict(state.get("targets") or {}).get("scope") or ""), "character")
            draft_mode = dict(dict(dict(state.get("draft_changes") or {}).get("modes") or {}).get("analyst") or {})
            self.assertEqual(str(draft_mode.get("action") or ""), "add")

    def test_impact_block_verbose_for_character_steps(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mmis_studio_impact_verbose_") as tmp:
            root = Path(tmp)
            self._prepare_specs_root(root)
            studio = StudioGenerator(specs_root=root / "specs", character_specs_root=root / "specs" / "characters")
            provider = _SpecsProvider()

            started = studio.start({}, seed="ambiguous as", provider=provider)
            text = str(started.text or "")
            self.assertIn("Что влияет", text)
            self.assertIn("Изменит:", text)
            self.assertIn("Файлы:", text)
            self.assertIn("Для персонажа:", text)
            self.assertIn("Результат:", text)

    def test_llm_profile_full_dynamic_enum_no_custom(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mmis_studio_llm_profiles_") as tmp:
            root = Path(tmp)
            self._prepare_specs_root(root)
            studio = StudioGenerator(specs_root=root / "specs", character_specs_root=root / "specs" / "characters")

            row = studio._empty_state()  # type: ignore[attr-defined]
            row["active"] = True
            row["phase"] = studio.PHASE_CLARIFY
            row["operation_type"] = studio.OP_CREATE_CHARACTER
            row["targets"] = {"character_ids": ["asya"], "mode_ids": [], "scope": "character"}
            row["draft_changes"] = studio._empty_draft_changes()  # type: ignore[attr-defined]
            row["draft_changes"]["characters"] = {"asya": studio._default_character_change(cid="asya", create=True)}  # type: ignore[attr-defined]

            question = studio._question_for_field(field_id="llm_profile", row=row, provider=None, model="")  # type: ignore[attr-defined]
            expected_profiles = [str(x).strip().upper() for x in list(PROFILES.keys())]
            self.assertEqual(list(question.get("options") or []), expected_profiles)
            self.assertFalse(bool(question.get("allow_custom", True)))

            row["step_options"] = {"llm_profile": list(expected_profiles)}
            parsed = studio._parse_answer(question_id="llm_profile", text="0", allow_numbered=True, row=row)  # type: ignore[attr-defined]
            self.assertIsNone(parsed)


if __name__ == "__main__":
    unittest.main()
