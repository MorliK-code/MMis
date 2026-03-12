from __future__ import annotations

import copy
import unittest
from dataclasses import replace

from config.settings import load_config
from llm.task_models import TaskModelRegistry, get_task_profile, is_task_enabled


class TaskModelRegistryTests(unittest.TestCase):
    def test_default_task_profiles_are_available(self) -> None:
        registry = TaskModelRegistry.from_settings(load_config(force_reload=True))

        expected = {
            "emotion",
            "tagging",
            "intent_judge",
            "query_rewrite",
            "source_relevance",
            "fact_filter",
            "final_response",
        }
        self.assertTrue(expected.issubset(set(registry.names())))

        profile = registry.get_task_profile("emotion")
        self.assertIsNotNone(profile)
        assert profile is not None
        self.assertEqual(profile.name, "emotion")
        self.assertEqual(profile.provider, "ollama")
        self.assertTrue(registry.is_task_enabled("emotion"))

    def test_module_level_api_returns_profiles(self) -> None:
        profile = get_task_profile("query_rewrite", force_reload=True)
        self.assertIsNotNone(profile)
        assert profile is not None
        self.assertEqual(profile.name, "query_rewrite")
        self.assertTrue(is_task_enabled("query_rewrite"))
        self.assertIsNone(get_task_profile("missing_task"))
        self.assertFalse(is_task_enabled("missing_task"))

    def test_disabled_profile_is_reported(self) -> None:
        settings = load_config(force_reload=True)
        rows = copy.deepcopy(settings.task_model_profiles)
        rows["tagging"]["enabled"] = False
        registry = TaskModelRegistry.from_settings(replace(settings, task_model_profiles=rows))

        self.assertFalse(registry.is_task_enabled("tagging"))
        result = registry.resolve_task("tagging")
        self.assertEqual(result.status, "disabled")
        self.assertEqual(result.reason, "profile_disabled")

    def test_invalid_fallback_profile_fails_validation(self) -> None:
        settings = load_config(force_reload=True)
        rows = copy.deepcopy(settings.task_model_profiles)
        rows["emotion"]["fallback_profile"] = "missing_profile"

        with self.assertRaises(ValueError):
            TaskModelRegistry.from_settings(replace(settings, task_model_profiles=rows))


if __name__ == "__main__":
    unittest.main()
