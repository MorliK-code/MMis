from __future__ import annotations

import copy
import unittest
from dataclasses import replace

from config.settings import load_config
from llm.provider_base import LLMRequest, LLMResponse, ModelInfo, ProviderHealth
from llm.task_models import TaskModelRegistry
from llm.task_router import TaskModelRouter, TaskModelValidationError


class _FakeProvider:
    def __init__(self, *, profile_name: str, scripted, requests_store: dict[str, list[LLMRequest]]):
        self._profile_name = profile_name
        self._scripted = scripted
        self._requests_store = requests_store

    def generate(self, req: LLMRequest) -> LLMResponse:
        self._requests_store.setdefault(self._profile_name, []).append(req)
        if not self._scripted:
            raise RuntimeError(f"No scripted response for {self._profile_name}")
        action = self._scripted.pop(0)
        if isinstance(action, Exception):
            raise action
        if isinstance(action, LLMResponse):
            return action
        return LLMResponse(text=str(action), model=str(req.model or "fake-model"))

    def healthcheck(self) -> ProviderHealth:
        return ProviderHealth(ok=True, provider="fake", detail="ok", model="fake-model")

    def model_info(self, model: str = "") -> ModelInfo:
        return ModelInfo(provider="fake", model=model or "fake-model")

    def list_models(self) -> list[str]:
        return ["fake-model"]


class TaskModelRouterTests(unittest.TestCase):
    def _router(self, *, rows=None, scripted=None):
        settings = load_config(force_reload=True)
        task_rows = copy.deepcopy(rows if rows is not None else settings.task_model_profiles)
        registry = TaskModelRegistry.from_settings(replace(settings, task_model_profiles=task_rows))
        plan = {str(k): list(v) for k, v in dict(scripted or {}).items()}
        requests_store: dict[str, list[LLMRequest]] = {}

        def provider_factory(profile):
            return _FakeProvider(
                profile_name=profile.name,
                scripted=plan.setdefault(profile.name, []),
                requests_store=requests_store,
            )

        router = TaskModelRouter(registry=registry, provider_factory=provider_factory)
        return router, requests_store

    def test_run_task_model_uses_profile_settings(self) -> None:
        router, requests = self._router(scripted={"emotion": ["calm"]})

        result = router.run_task_model("emotion", "Classify emotion.")

        self.assertEqual(result.profile_name, "emotion")
        self.assertFalse(result.used_fallback)
        req = requests["emotion"][0]
        self.assertEqual(req.model, result.model)
        self.assertEqual(req.temperature, 0.15)
        self.assertEqual(req.max_tokens, 128)
        self.assertEqual(req.metadata.get("task_name"), "emotion")
        self.assertFalse(req.json_mode)

    def test_disabled_primary_profile_uses_fallback_profile(self) -> None:
        settings = load_config(force_reload=True)
        rows = copy.deepcopy(settings.task_model_profiles)
        rows["emotion"]["enabled"] = False
        rows["emotion"]["fallback_profile"] = "tagging"
        router, _ = self._router(rows=rows, scripted={"tagging": ["tag-result"]})

        result = router.run_task_model("emotion", "Fallback test.")

        self.assertEqual(result.profile_name, "tagging")
        self.assertTrue(result.used_fallback)
        self.assertEqual(result.attempted_profiles, ("emotion", "tagging"))

    def test_failed_primary_profile_falls_back(self) -> None:
        settings = load_config(force_reload=True)
        rows = copy.deepcopy(settings.task_model_profiles)
        rows["emotion"]["fallback_profile"] = "tagging"
        router, _ = self._router(
            rows=rows,
            scripted={
                "emotion": [RuntimeError("boom")],
                "tagging": ["recovered"],
            },
        )

        result = router.run_task_model("emotion", "Fallback on failure.")

        self.assertEqual(result.profile_name, "tagging")
        self.assertTrue(result.used_fallback)
        self.assertEqual(result.text, "recovered")

    def test_run_task_model_json_parses_payload(self) -> None:
        router, requests = self._router(
            scripted={"query_rewrite": ['```json\n{"query":"usd rate kyiv"}\n```']}
        )

        result = router.run_task_model_json("query_rewrite", "Rewrite the query.")

        self.assertEqual(result.json_payload, {"query": "usd rate kyiv"})
        req = requests["query_rewrite"][0]
        self.assertTrue(req.json_mode)
        self.assertIn("Return valid JSON only.", req.messages[0].content)

    def test_empty_output_raises_validation_error(self) -> None:
        router, _ = self._router(scripted={"emotion": ["   "]})

        with self.assertRaises(TaskModelValidationError):
            router.run_task_model("emotion", "Should fail.")

    def test_missing_required_fields_uses_profile_fallback(self) -> None:
        settings = load_config(force_reload=True)
        rows = copy.deepcopy(settings.task_model_profiles)
        rows["query_rewrite"]["fallback_profile"] = "fact_filter"
        router, _ = self._router(
            rows=rows,
            scripted={
                "query_rewrite": ['{"note":"missing query"}'],
                "fact_filter": ['{"query":"usd rate kyiv"}'],
            },
        )

        result = router.run_task_model_json(
            "query_rewrite",
            "Rewrite the query.",
            required_fields=("query",),
        )

        self.assertEqual(result.profile_name, "fact_filter")
        self.assertTrue(result.used_fallback)
        self.assertEqual(result.json_payload, {"query": "usd rate kyiv"})
        self.assertEqual(result.attempts[0].status, "validation_failed")

    def test_retry_on_same_profile_before_succeeding(self) -> None:
        router, _ = self._router(
            scripted={
                "emotion": [RuntimeError("temporary timeout"), "calm"],
            }
        )

        result = router.run_task_model("emotion", "Classify emotion.", max_retries=1)

        self.assertEqual(result.profile_name, "emotion")
        self.assertEqual(result.text, "calm")
        self.assertEqual(result.retries_used, 1)
        self.assertEqual(len(result.attempts), 2)
        self.assertEqual(result.attempts[0].retry_index, 0)
        self.assertEqual(result.attempts[1].retry_index, 1)

    def test_deterministic_fallback_is_used_after_invalid_output(self) -> None:
        router, _ = self._router(scripted={"emotion": ["   "]})

        result = router.run_task_model(
            "emotion",
            "Classify emotion.",
            deterministic_fallback=lambda failure: "neutral",
        )

        self.assertTrue(result.used_fallback)
        self.assertTrue(result.deterministic_fallback_used)
        self.assertEqual(result.fallback_reason, "deterministic_fallback")
        self.assertEqual(result.text, "neutral")

    def test_run_task_model_choice_parses_short_label(self) -> None:
        router, _ = self._router(
            scripted={"intent_judge": ['{"label":"weather"}']}
        )

        result = router.run_task_model_choice(
            "intent_judge",
            "Classify intent.",
            allowed_labels=("weather", "fx_rate", "chat"),
        )

        self.assertEqual(result.parsed_output, "weather")
        self.assertEqual(result.text, '{"label":"weather"}')


if __name__ == "__main__":
    unittest.main()
