from __future__ import annotations

from _output_utils import enable_unittest_json_output

enable_unittest_json_output()

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from core.brain import Brain
from core.response_pipeline import ResponsePipeline
from llm.provider_base import LLMProviderBase, LLMRequest, LLMResponse, ModelInfo, ProviderHealth, Timings, Usage
from modules.studio.studio_generator import StudioGenerator


class _DispatchProbeProvider(LLMProviderBase):
    def __init__(self) -> None:
        self.main_calls = 0
        self.studio_calls = 0

    def generate(self, req: LLMRequest) -> LLMResponse:
        task = str((req.metadata or {}).get("studio_specs_task") or "").strip()
        if task:
            self.studio_calls += 1
            if task == "seed_extract":
                payload = {
                    "operation_type": "create_character",
                    "targets": {"character_ids": ["demo_spec"], "mode_ids": ["helper", "debugger"], "scope": "character"},
                    "character": {
                        "character_id": "demo_spec",
                        "display_name": "Demo",
                        "vibe": "balanced",
                        "technicality": 0.8,
                        "energy": 0.6,
                        "default_mode": "helper",
                        "extra_modes": ["debugger"],
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
            else:
                payload = {"options": ["variant 1", "variant 2", "variant 3"]}
            text = json.dumps(payload, ensure_ascii=False)
        else:
            self.main_calls += 1
            text = "main-model-answer"
        return LLMResponse(
            text=text,
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


class StudioLocalCommandsTests(unittest.TestCase):
    def _new_pipeline(self) -> tuple[ResponsePipeline, _DispatchProbeProvider]:
        provider = _DispatchProbeProvider()
        tmp = tempfile.TemporaryDirectory(prefix="mmis_studio_local_")
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        studio = StudioGenerator(specs_root=root / "specs", character_specs_root=root / "specs" / "characters")
        pipeline = ResponsePipeline(provider=provider, studio_generator=studio)
        return pipeline, provider

    @staticmethod
    def _run(pipeline: ResponsePipeline, *, route: str, msg: str, state: dict):
        return pipeline.run(
            route=route,
            user_msg=msg,
            state=state,
            meta={"source": "test", "store_turn": False, "conversation_id": "studio-local-test"},
            retrieved_memories=[],
            traits={},
            policies={},
        )

    def test_studio_apply_is_local_and_next_chat_goes_main(self) -> None:
        pipeline, provider = self._new_pipeline()
        state = {"history": [], "quality_profile": "BALANCED", "active_mode": "chatting"}

        start = self._run(pipeline, route="command", msg="/studio start demo_spec", state=state)
        state["studio_generator"] = dict(start.structured_output.get("studio_generator") or {})
        self.assertGreaterEqual(provider.studio_calls, 1)

        before_main = provider.main_calls
        apply = self._run(pipeline, route="command", msg="/studio apply", state=state)
        state["studio_generator"] = dict(apply.structured_output.get("studio_generator") or {})

        self.assertIn("Studio", str(apply.text or ""))
        self.assertEqual(before_main, provider.main_calls)
        self.assertTrue(any("command=studio:apply_local" in str(x) for x in list(apply.logs or [])))
        turn_ops = [x for x in list(apply.memory_ops or []) if str(x.get("op")) in {"turn_user", "turn_assistant"}]
        self.assertFalse(turn_ops)

        chat = self._run(pipeline, route="chat", msg="привет", state=state)
        self.assertEqual("main-model-answer", str(chat.text or ""))
        self.assertGreater(provider.main_calls, before_main)

    def test_studio_cancel_is_local(self) -> None:
        pipeline, provider = self._new_pipeline()
        state = {"history": [], "quality_profile": "BALANCED", "active_mode": "chatting"}

        start = self._run(pipeline, route="command", msg="/studio start demo_spec", state=state)
        state["studio_generator"] = dict(start.structured_output.get("studio_generator") or {})

        before_main = provider.main_calls
        cancel = self._run(pipeline, route="command", msg="/studio cancel", state=state)
        self.assertEqual(before_main, provider.main_calls)
        self.assertTrue(any("command=studio:cancel_local" in str(x) for x in list(cancel.logs or [])))

    def test_removed_specs_commands_are_handled_locally(self) -> None:
        pipeline, provider = self._new_pipeline()
        state = {"history": [], "quality_profile": "BALANCED", "active_mode": "chatting"}

        before_main = provider.main_calls
        result = self._run(pipeline, route="command", msg="/specs start demo", state=state)
        self.assertEqual(before_main, provider.main_calls)
        self.assertIn("Команда удалена", str(result.text or ""))

    def test_removed_studio_mode_command_is_handled_locally(self) -> None:
        pipeline, provider = self._new_pipeline()
        state = {"history": [], "quality_profile": "BALANCED", "active_mode": "chatting"}

        before_main = provider.main_calls
        result = self._run(pipeline, route="command", msg="/studio mode specs", state=state)
        self.assertEqual(before_main, provider.main_calls)
        self.assertIn("Команда удалена", str(result.text or ""))

    def test_apply_with_missing_fields_keeps_studio_active_and_stays_local(self) -> None:
        pipeline, provider = self._new_pipeline()
        state = {"history": [], "quality_profile": "BALANCED", "active_mode": "chatting"}

        start = self._run(pipeline, route="command", msg="/studio start", state=state)
        state["studio_generator"] = dict(start.structured_output.get("studio_generator") or {})
        before_main = provider.main_calls
        apply = self._run(pipeline, route="command", msg="/studio apply", state=state)
        state["studio_generator"] = dict(apply.structured_output.get("studio_generator") or {})

        self.assertEqual(before_main, provider.main_calls)
        studio_row = dict(state.get("studio_generator") or {})
        self.assertTrue(bool(studio_row.get("active", False)))
        self.assertEqual(str(studio_row.get("phase") or ""), "clarify")

    def test_short_apply_cancel_commands_are_local(self) -> None:
        pipeline, provider = self._new_pipeline()
        state = {"history": [], "quality_profile": "BALANCED", "active_mode": "chatting"}

        start = self._run(pipeline, route="command", msg="/studio start demo_spec", state=state)
        state["studio_generator"] = dict(start.structured_output.get("studio_generator") or {})
        before_main = provider.main_calls

        short_apply = self._run(pipeline, route="command", msg="/apply", state=state)
        self.assertEqual(before_main, provider.main_calls)
        self.assertTrue(any("apply_local" in str(x) for x in list(short_apply.logs or [])))

        start2 = self._run(pipeline, route="command", msg="/studio start demo_spec", state=state)
        state["studio_generator"] = dict(start2.structured_output.get("studio_generator") or {})
        short_cancel = self._run(pipeline, route="command", msg="/cancel", state=state)
        self.assertTrue(any("cancel_local" in str(x) for x in list(short_cancel.logs or [])))

    def test_successful_apply_invalidates_spec_cache(self) -> None:
        pipeline, _provider = self._new_pipeline()
        state = {"history": [], "quality_profile": "BALANCED", "active_mode": "chatting"}
        start = self._run(pipeline, route="command", msg="/studio start demo_spec", state=state)
        state["studio_generator"] = dict(start.structured_output.get("studio_generator") or {})
        with patch("core.response_pipeline.invalidate_spec_cache") as invalidate:
            _ = self._run(pipeline, route="command", msg="/studio apply", state=state)
            invalidate.assert_called_once()

    def test_brain_apply_without_ts_in_memory_ops_does_not_fallback(self) -> None:
        provider = _DispatchProbeProvider()
        with tempfile.TemporaryDirectory(prefix="mmis_studio_brain_apply_") as tmp:
            root = Path(tmp)
            studio = StudioGenerator(specs_root=root / "specs", character_specs_root=root / "specs" / "characters")
            pipeline = ResponsePipeline(provider=provider, studio_generator=studio)
            brain = Brain(provider=provider, response_pipeline=pipeline)

            started = brain.handle_message("/studio start demo_spec", meta={"source": "test", "store_turn": False})
            self.assertEqual(started.status, "ok")
            applied = brain.handle_message("/apply", meta={"source": "test", "store_turn": False})

            self.assertEqual(applied.status, "ok")
            self.assertNotIn("Я затупила", str(applied.text or ""))
            self.assertIn("Studio", str(applied.text or ""))


if __name__ == "__main__":
    unittest.main()
