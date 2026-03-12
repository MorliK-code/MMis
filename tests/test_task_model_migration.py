from __future__ import annotations

import unittest
from unittest.mock import patch

from core.response_pipeline import OutputFormatStage, PipelineContext
from llm.provider_base import LLMResponse
from llm.task_router import TaskModelExecutionResult
from modules.studio.studio_generator import StudioGenerator


class _FailProvider:
    def generate(self, req):
        raise AssertionError("legacy provider path should not be used in this test")


class TaskModelMigrationTests(unittest.TestCase):
    def _task_result(self, *, task_name: str, text: str = "", json_payload=None) -> TaskModelExecutionResult:
        return TaskModelExecutionResult(
            task_name=task_name,
            profile_name=task_name,
            provider="ollama",
            model="fake-task-model",
            text=text,
            response=LLMResponse(text=text, model="fake-task-model"),
            used_fallback=False,
            attempted_profiles=(task_name,),
            json_payload=json_payload,
        )

    def test_studio_seed_extract_uses_task_router(self) -> None:
        generator = StudioGenerator()
        with patch(
            "modules.studio.studio_generator.run_task_model_json",
            return_value=self._task_result(
                task_name="studio_seed_extract",
                json_payload={"operation_type": "create_character", "character": {"character_id": "asya"}},
            ),
        ) as mocked:
            out, conf, telemetry = generator._extract_seed_with_llm(
                seed="create character asya",
                provider=_FailProvider(),
                model="legacy-model",
            )

        self.assertEqual(out.get("operation_type"), "create_character")
        self.assertEqual(out.get("character", {}).get("character_id"), "asya")
        self.assertEqual(conf, {})
        self.assertEqual(telemetry.get("task_profile"), "studio_seed_extract")
        mocked.assert_called_once()
        self.assertEqual(mocked.call_args.args[0], "studio_seed_extract")

    def test_studio_options_uses_task_router(self) -> None:
        generator = StudioGenerator()
        with patch(
            "modules.studio.studio_generator.run_task_model_json",
            return_value=self._task_result(
                task_name="studio_options",
                json_payload={"options": ["one", "two", "three"]},
            ),
        ) as mocked:
            options, telemetry = generator._generate_options_with_llm(
                question_id="vibe",
                prompt="Pick vibe",
                row={"operation_type": "create_character", "targets": {}, "draft_changes": {}},
                provider=_FailProvider(),
                model="legacy-model",
                attempt=1,
            )

        self.assertEqual(options[:2], ["one", "two"])
        self.assertEqual(telemetry.get("task_profile"), "studio_options")
        mocked.assert_called_once()
        self.assertEqual(mocked.call_args.args[0], "studio_options")

    def test_studio_pack_blueprint_uses_task_router(self) -> None:
        generator = StudioGenerator()
        with patch(
            "modules.studio.studio_generator.run_task_model_json",
            return_value=self._task_result(
                task_name="studio_pack_blueprint",
                json_payload={"character": {"id": "asya"}, "modes": []},
            ),
        ) as mocked:
            payload = generator._generate_pack_blueprint_with_llm(
                data={"character_id": "asya"},
                provider=_FailProvider(),
                model="legacy-model",
            )

        self.assertEqual(payload.get("character", {}).get("id"), "asya")
        mocked.assert_called_once()
        self.assertEqual(mocked.call_args.args[0], "studio_pack_blueprint")

    def test_summary_mini_pass_uses_task_router(self) -> None:
        stage = OutputFormatStage(provider=_FailProvider())
        ctx = PipelineContext(
            route="chat",
            user_msg="hi",
            state={},
            meta={"trace_id": "trace_test"},
            retrieved_memories=[],
            traits={},
            policies={},
            profile="BALANCED",
            stats={},
        )
        with patch(
            "core.response_pipeline.run_task_model",
            return_value=self._task_result(task_name="summary_mini_pass", text="Short summary."),
        ) as mocked:
            summary = stage._summary_mini_pass(ctx, "Longer assistant response.")

        self.assertEqual(summary, "Short summary.")
        self.assertIn("stage=output_format summary_mini_pass=task_model", ctx.logs)
        mocked.assert_called_once()
        self.assertEqual(mocked.call_args.args[0], "summary_mini_pass")


if __name__ == "__main__":
    unittest.main()
