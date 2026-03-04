from __future__ import annotations

from _output_utils import enable_unittest_json_output

enable_unittest_json_output()

import json
import tempfile
import unittest
from pathlib import Path

from config.model_config import PROFILES
from core.response_pipeline import PROFILE_AUTONOMOUS, PipelineContext, ResponsePipeline
from llm.provider_base import (
    LLMProviderBase,
    LLMRequest,
    LLMResponse,
    Message,
    ModelInfo,
    ProviderHealth,
    Timings,
    Usage,
)
from modules.studio.studio_generator import StudioGenerator


class _StudioStubProvider(LLMProviderBase):
    def __init__(self) -> None:
        self.calls = 0
        self.last_req: LLMRequest | None = None

    def generate(self, req: LLMRequest) -> LLMResponse:
        self.last_req = req
        self.calls += 1
        task = str((req.metadata or {}).get("studio_specs_task") or "").strip()
        if task == "seed_extract":
            payload = {
                "operation_type": "build_character_pack",
                "targets": {"character_ids": ["luna_program"], "mode_ids": ["friend_chat", "helper"], "scope": "character"},
                "character": {
                    "character_id": "luna_program",
                    "display_name": "Luna",
                    "vibe": "balanced",
                    "technicality": 0.8,
                    "energy": 0.55,
                    "default_mode": "friend_chat",
                    "extra_modes": ["helper"],
                    "llm_profile": "BALANCED",
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
        elif task == "build_pack":
            payload = {
                "character": {"id": "luna_program", "name": "Luna", "vibe": "balanced", "llm_profile": "BALANCED"},
                "modes": [{"id": "friend_chat", "description": "Friendly mode"}, {"id": "helper", "description": "Helper mode"}],
                "dialog_policy": {"do": ["Be helpful"], "avoid": ["Be vague"], "escalation": ["Ask follow-up"]},
                "prompts": {"system": "You are Luna.", "style": "Pragmatic", "boundaries": "Keep safe"},
                "samples": {"opener": "Привет", "clarification_question": "Уточни цель", "refusal_safe": "Не могу"},
                "artifacts": [
                    {
                        "path": "character.json",
                        "kind": "json",
                        "action": "create_or_update",
                        "content": {
                            "schema_version": 1,
                            "character_id": "luna_program",
                            "id": "luna_program",
                            "name": "Luna",
                            "default_mode": "friend_chat",
                            "default_mood": "neutral",
                            "llm_profile": "BALANCED"
                        }
                    }
                ],
            }
        else:
            payload = {"options": ["variant one", "variant two", "variant three"]}
        return LLMResponse(
            text=json.dumps(payload, ensure_ascii=False),
            usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            timings=Timings(latency_ms=1.0),
            model="stub",
        )

    def healthcheck(self) -> ProviderHealth:
        return ProviderHealth(ok=True, provider="stub", detail="ok", model="stub")

    def model_info(self, model: str = "") -> ModelInfo:
        return ModelInfo(provider="stub", model=model or "stub")

    def list_models(self) -> list[str]:
        return ["stub"]


class OllamaCaptureProvider(LLMProviderBase):
    def __init__(self) -> None:
        self.last_req: LLMRequest | None = None

    def generate(self, req: LLMRequest) -> LLMResponse:
        self.last_req = req
        return LLMResponse(
            text="ok",
            usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            timings=Timings(latency_ms=1.0),
            model="stub",
        )

    def healthcheck(self) -> ProviderHealth:
        return ProviderHealth(ok=True, provider="ollama", detail="ok", model="stub")

    def model_info(self, model: str = "") -> ModelInfo:
        return ModelInfo(provider="ollama", model=model or "stub")

    def list_models(self) -> list[str]:
        return ["stub"]


class CharacterStudioTests(unittest.TestCase):
    def test_studio_build_pack_and_apply_writes_specs_files(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mmis_studio_build_pack_") as tmp:
            root = Path(tmp)
            studio = StudioGenerator(specs_root=root / "specs", character_specs_root=root / "specs" / "characters")
            provider = _StudioStubProvider()
            pipeline = ResponsePipeline(provider=provider, studio_generator=studio)

            state = {"history": [], "quality_profile": "BALANCED", "active_mode": "friend_chat"}

            r1 = pipeline.run(
                route="command",
                user_msg="/studio start build-pack luna_program",
                state=state,
                meta={"source": "test", "store_turn": False},
                retrieved_memories=[],
                traits={},
                policies={},
            )
            start_state = dict(r1.structured_output.get("studio_generator") or {})
            self.assertTrue(bool(start_state.get("active", False)))
            self.assertIn(str(start_state.get("phase") or ""), {"clarify", "review"})
            state["studio_generator"] = dict(r1.structured_output.get("studio_generator") or {})

            apply = pipeline.run(
                route="command",
                user_msg="/studio apply",
                state=state,
                meta={"source": "test", "store_turn": False},
                retrieved_memories=[],
                traits={},
                policies={},
            )
            self.assertIn("Studio", str(apply.text or ""))
            self.assertGreaterEqual(int(provider.calls), 1)
            self.assertIsNotNone(provider.last_req)
            self.assertTrue(bool((provider.last_req.metadata or {}).get("think", False)))
            spec_dir = root / "specs" / "characters" / "luna_program"
            self.assertTrue((spec_dir / "studio_blueprint.json").exists())
            self.assertTrue((spec_dir / "character.json").exists())

    def test_autonomous_profile_sets_unlimited_tokens_for_ollama(self) -> None:
        provider = OllamaCaptureProvider()
        pipeline = ResponsePipeline(provider=provider)
        stage = pipeline._stages["generate"]
        assert hasattr(stage, "_build_request")

        ctx = PipelineContext(
            route="chat",
            user_msg="test",
            clean_user_msg="test",
            state={"quality_profile": PROFILE_AUTONOMOUS},
            meta={},
            retrieved_memories=[],
            traits={},
            policies={},
            profile=PROFILE_AUTONOMOUS,
            prompt_messages=[Message(role="system", content="s"), Message(role="user", content="u")],
        )
        req = stage._build_request(ctx)  # type: ignore[attr-defined]
        self.assertEqual(int(req.max_tokens or 0), -1)

    def test_build_pack_requires_valid_llm_profile_from_dynamic_enum(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mmis_studio_llm_profile_strict_") as tmp:
            root = Path(tmp)
            studio = StudioGenerator(specs_root=root / "specs", character_specs_root=root / "specs" / "characters")

            row = studio._empty_state()  # type: ignore[attr-defined]
            row["active"] = True
            row["phase"] = studio.PHASE_REVIEW
            row["operation_type"] = studio.OP_BUILD_PACK
            row["targets"] = {"character_ids": ["luna_program"], "mode_ids": [], "scope": "character"}
            draft = studio._empty_draft_changes()  # type: ignore[attr-defined]
            draft["characters"] = {"luna_program": studio._default_character_change(cid="luna_program", create=True)}  # type: ignore[attr-defined]
            draft["characters"]["luna_program"]["patch"]["llm_profile"] = "NOT_VALID_PROFILE"
            row["draft_changes"] = draft
            row["unresolved_fields"] = []

            result = studio.apply({StudioGenerator.KEY: row}, provider=_StudioStubProvider())
            self.assertEqual(str(result.error or ""), "missing_fields")
            self.assertFalse(result.done)
            self.assertIn("llm_profile", list(dict(result.state or {}).get("unresolved_fields") or []))

            q = studio._question_for_field(field_id="llm_profile", row=dict(result.state or {}), provider=None, model="")  # type: ignore[attr-defined]
            self.assertEqual(list(q.get("options") or []), [str(x).upper() for x in list(PROFILES.keys())])


if __name__ == "__main__":
    unittest.main()
