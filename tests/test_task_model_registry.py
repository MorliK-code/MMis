from __future__ import annotations

import unittest
from dataclasses import replace

from config.settings import load_config
from llm.task_models import TaskModelRegistry, get_task_profile, is_task_enabled


class TaskModelRegistryTests(unittest.TestCase):
    def test_default_task_profiles_are_empty(self) -> None:
        registry = TaskModelRegistry.from_settings(load_config(force_reload=True))

        self.assertEqual(registry.names(), ())
        self.assertIsNone(registry.get_task_profile("emotion"))
        self.assertFalse(registry.is_task_enabled("emotion"))

    def test_module_level_api_reports_missing_profile(self) -> None:
        profile = get_task_profile("query_rewrite", force_reload=True)
        self.assertIsNone(profile)
        self.assertFalse(is_task_enabled("query_rewrite"))
        self.assertIsNone(get_task_profile("missing_task"))
        self.assertFalse(is_task_enabled("missing_task"))

    def test_disabled_profile_is_reported(self) -> None:
        settings = load_config(force_reload=True)
        rows = {
            "tagging": {
                "name": "tagging",
                "provider": "ollama",
                "model": "fake-model",
                "temperature": 0.1,
                "max_tokens": 128,
                "timeout": 20.0,
                "enabled": True,
                "fallback_profile": "",
            }
        }
        rows["tagging"]["enabled"] = False
        registry = TaskModelRegistry.from_settings(replace(settings, task_model_profiles=rows))

        self.assertFalse(registry.is_task_enabled("tagging"))
        result = registry.resolve_task("tagging")
        self.assertEqual(result.status, "disabled")
        self.assertEqual(result.reason, "profile_disabled")

    def test_invalid_fallback_profile_fails_validation(self) -> None:
        settings = load_config(force_reload=True)
        rows = {
            "emotion": {
                "name": "emotion",
                "provider": "ollama",
                "model": "fake-model",
                "temperature": 0.1,
                "max_tokens": 128,
                "timeout": 20.0,
                "enabled": True,
                "fallback_profile": "missing_profile",
            }
        }

        with self.assertRaises(ValueError):
            TaskModelRegistry.from_settings(replace(settings, task_model_profiles=rows))


if __name__ == "__main__":
    unittest.main()
