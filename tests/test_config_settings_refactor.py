from __future__ import annotations

from _output_utils import enable_unittest_json_output

enable_unittest_json_output()

import json
import os
import tempfile
import unittest
from pathlib import Path

from config.settings import BASE_DIR, DIR_PATH_TOKEN, get_profile, load_config, update_config_values


class ConfigSettingsRefactorTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="mmis_cfg_refactor_")
        self._cfg_path = Path(self._tmp.name).resolve() / "config.json"
        self._old_env = {
            "MMIS_CONFIG_FILE": os.environ.get("MMIS_CONFIG_FILE"),
            "MMIS_API_PORT": os.environ.get("MMIS_API_PORT"),
        }
        os.environ["MMIS_CONFIG_FILE"] = str(self._cfg_path)
        os.environ.pop("MMIS_API_PORT", None)

    def tearDown(self) -> None:
        for key, value in self._old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        self._tmp.cleanup()
        load_config(force_reload=True)

    def _read_cfg(self) -> dict:
        if not self._cfg_path.exists():
            return {}
        data = json.loads(self._cfg_path.read_text(encoding="utf-8-sig") or "{}")
        return data if isinstance(data, dict) else {}

    def test_load_config_creates_structured_config_when_missing(self) -> None:
        settings = load_config(force_reload=True)
        self.assertTrue(self._cfg_path.exists())
        self.assertEqual(settings.port, 8027)
        cfg = self._read_cfg()
        self.assertIn("app", cfg)
        self.assertIn("api", cfg)
        self.assertIn("llm", cfg)
        self.assertIn("internet", cfg)
        self.assertEqual(int(cfg.get("api", {}).get("port") or 0), 8027)
        self.assertTrue(str(cfg.get("memory", {}).get("memory_dir") or "").lower().startswith(DIR_PATH_TOKEN))
        self.assertTrue(str(cfg.get("paths", {}).get("data_dir") or "").lower().startswith(DIR_PATH_TOKEN))

    def test_legacy_flat_config_is_migrated_with_backup(self) -> None:
        legacy = {
            "port": 8035,
            "model_name": "stub-model",
            "console_runtime_state": {
                "api_base_url": "http://127.0.0.1:9999",
                "web_mode": "off",
            },
            "custom_legacy": 123,
        }
        self._cfg_path.parent.mkdir(parents=True, exist_ok=True)
        self._cfg_path.write_text(json.dumps(legacy, ensure_ascii=False, indent=2), encoding="utf-8")

        settings = load_config(force_reload=True)
        self.assertEqual(int(settings.port), 8035)
        self.assertEqual(str(settings.model_name), "stub-model")

        cfg = self._read_cfg()
        self.assertEqual(int(cfg.get("api", {}).get("port") or 0), 8035)
        self.assertEqual(str(cfg.get("llm", {}).get("model_name") or ""), "stub-model")
        self.assertEqual(str(cfg.get("internet", {}).get("web_mode") or ""), "off")
        self.assertNotIn("last_api_base_url", dict(cfg.get("ui", {}).get("console") or {}))
        extra = dict(cfg.get("features", {}).get("extra_legacy") or {})
        self.assertEqual(int(extra.get("custom_legacy") or 0), 123)
        self.assertNotIn("console_runtime_state", extra)

        backups = list(self._cfg_path.parent.glob("config.pre_migration.*.json"))
        self.assertTrue(backups)

    def test_existing_config_has_priority_over_env(self) -> None:
        cfg = load_config(force_reload=True)
        self.assertTrue(self._cfg_path.exists())
        os.environ["MMIS_API_PORT"] = "9200"
        update_config_values({"api.port": 9100})
        settings = load_config(force_reload=True)
        self.assertEqual(int(settings.port), 9100)

    def test_update_config_values_persists_immediately(self) -> None:
        load_config(force_reload=True)
        update_config_values({"api.port": 8041, "internet.web_mode": "off"})
        cfg = self._read_cfg()
        self.assertEqual(int(cfg.get("api", {}).get("port") or 0), 8041)
        self.assertEqual(str(cfg.get("internet", {}).get("web_mode") or ""), "off")

    def test_project_absolute_paths_are_migrated_to_dir_path_token(self) -> None:
        memory_abs = str((BASE_DIR / "data" / "memory_storage").resolve())
        cache_abs = str((BASE_DIR / "data" / "cache").resolve())
        row = {
            "memory": {
                "memory_dir": memory_abs,
                "cache_dir": cache_abs,
            }
        }
        self._cfg_path.parent.mkdir(parents=True, exist_ok=True)
        self._cfg_path.write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")

        settings = load_config(force_reload=True)
        cfg = self._read_cfg()
        self.assertTrue(str(cfg.get("memory", {}).get("memory_dir") or "").lower().startswith(DIR_PATH_TOKEN))
        self.assertTrue(str(cfg.get("memory", {}).get("cache_dir") or "").lower().startswith(DIR_PATH_TOKEN))
        self.assertEqual(settings.memory_dir.resolve(), (BASE_DIR / "data" / "memory_storage").resolve())

    def test_mmis_config_file_path_override_is_respected(self) -> None:
        settings = load_config(force_reload=True)
        self.assertEqual(Path(settings.config_file or "").resolve(), self._cfg_path.resolve())
        self.assertTrue(self._cfg_path.exists())

    def test_model_profiles_are_stored_in_config_and_used_by_get_profile(self) -> None:
        load_config(force_reload=True)
        cfg = self._read_cfg()
        profiles = dict(cfg.get("llm", {}).get("profiles") or {})
        self.assertIn("BALANCED", profiles)
        update_config_values({"llm.profiles.BALANCED.generation.temperature": 0.23})
        profile = get_profile("BALANCED")
        self.assertEqual(float(profile.generation.temperature), 0.23)

    def test_memory_backend_accepts_chroma_and_chromadb(self) -> None:
        load_config(force_reload=True)
        update_config_values({"memory.backend": "chroma"})
        self.assertEqual(str(load_config(force_reload=True).memory_backend), "chroma")
        update_config_values({"memory.backend": "chromadb"})
        self.assertEqual(str(load_config(force_reload=True).memory_backend), "chromadb")

    def test_memory_backend_rejects_local_with_explicit_instruction(self) -> None:
        load_config(force_reload=True)
        cfg = self._read_cfg()
        memory = dict(cfg.get("memory") or {})
        memory["backend"] = "local"
        cfg["memory"] = memory
        self._cfg_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")

        with self.assertRaises(ValueError) as ctx:
            load_config(force_reload=True)
        message = str(ctx.exception)
        self.assertIn("memory.backend must be one of ['chroma', 'chromadb']", message)
        self.assertIn("Switch memory.backend to 'chroma' or 'chromadb'", message)

    def test_memory_v2_thresholds_and_importance_weights_are_loaded(self) -> None:
        load_config(force_reload=True)
        update_config_values(
            {
                "memory.lifecycle.promotion_thresholds.message_importance": 0.93,
                "memory.lifecycle.promotion_thresholds.message_confidence": 0.61,
                "memory.lifecycle.promotion_signal_boosts.project": 0.27,
                "memory.lifecycle.promotion_signal_boosts.fact": 0.31,
                "memory.lifecycle.promotion_signal_boosts.decision": 0.29,
                "memory.lifecycle.promotion_signal_boosts.smalltalk_penalty": 0.35,
                "memory.scoring.importance_weights.base": 0.11,
                "memory.scoring.importance_weights.decision": 0.22,
                "memory.scoring.importance_weights.remember": 0.33,
                "memory.scoring.importance_weights.project": 0.44,
            }
        )
        settings = load_config(force_reload=True)
        self.assertAlmostEqual(float(settings.memory_promotion_message_importance_threshold), 0.93, places=6)
        self.assertAlmostEqual(float(settings.memory_promotion_message_confidence_threshold), 0.61, places=6)
        self.assertAlmostEqual(float(settings.memory_promotion_project_signal_boost), 0.27, places=6)
        self.assertAlmostEqual(float(settings.memory_promotion_fact_signal_boost), 0.31, places=6)
        self.assertAlmostEqual(float(settings.memory_promotion_decision_signal_boost), 0.29, places=6)
        self.assertAlmostEqual(float(settings.memory_promotion_smalltalk_penalty), 0.35, places=6)
        self.assertAlmostEqual(float(settings.memory_importance_weight_base), 0.11, places=6)
        self.assertAlmostEqual(float(settings.memory_importance_weight_decision), 0.22, places=6)
        self.assertAlmostEqual(float(settings.memory_importance_weight_remember), 0.33, places=6)
        self.assertAlmostEqual(float(settings.memory_importance_weight_project), 0.44, places=6)

    def test_memory_v2_retrieval_fusion_weights_are_loaded(self) -> None:
        load_config(force_reload=True)
        update_config_values(
            {
                "memory.retrieval.fusion_weights.semantic_similarity": 0.51,
                "memory.retrieval.fusion_weights.lexical_score": 0.41,
                "memory.retrieval.fusion_weights.scope_match_score": 0.19,
            }
        )
        settings = load_config(force_reload=True)
        self.assertAlmostEqual(float(settings.memory_retrieval_weight_semantic_similarity), 0.51, places=6)
        self.assertAlmostEqual(float(settings.memory_retrieval_weight_lexical_score), 0.41, places=6)
        self.assertAlmostEqual(float(settings.memory_retrieval_weight_scope_match_score), 0.19, places=6)

    def test_memory_v2_salience_weights_are_loaded(self) -> None:
        load_config(force_reload=True)
        update_config_values(
            {
                "memory.scoring.salience_weights.novelty": 0.31,
                "memory.scoring.salience_weights.permanence": 0.27,
                "memory.scoring.salience_weights.explicit_save_signal": 0.91,
            }
        )
        settings = load_config(force_reload=True)
        self.assertAlmostEqual(float(settings.memory_salience_weight_novelty), 0.31, places=6)
        self.assertAlmostEqual(float(settings.memory_salience_weight_permanence), 0.27, places=6)
        self.assertAlmostEqual(float(settings.memory_salience_weight_explicit_save_signal), 0.91, places=6)

    def test_logging_and_paths_sections_are_present(self) -> None:
        load_config(force_reload=True)
        cfg = self._read_cfg()
        logging_row = dict(cfg.get("logging") or {})
        paths_row = dict(cfg.get("paths") or {})
        self.assertIn("channels", logging_row)
        self.assertEqual(str(logging_row.get("web_trace_logger") or ""), "web.trace")
        self.assertIn("data_dir", paths_row)
        self.assertIn("models_dir", paths_row)
        self.assertNotIn("base_dir", paths_row)
        self.assertNotIn("config_dir", paths_row)
        self.assertNotIn("default_memory_dir", paths_row)
        self.assertNotIn("legacy_memory_dir", paths_row)

    def test_deduplicates_ui_global_runtime_keys_into_canonical(self) -> None:
        row = {
            "llm": {
                "model_name": "canonical-model",
                "json_mode_enabled": False,
                "thinking_enabled": True,
            },
            "internet": {
                "web_mode": "off",
                "web_auto_profile": "balanced",
            },
            "ui": {
                "console": {
                    "model": "legacy-ui-model",
                    "json_mode_enabled": True,
                    "last_api_base_url": "http://127.0.0.1:8000",
                    "runtime": {
                        "think_enabled": False,
                        "web_mode": "auto",
                        "web_auto_profile": "aggressive",
                        "json_mode_enabled": True,
                        "mode_lock": True,
                    },
                },
            },
            "startup": {
                "read_only_tools": False,
            },
            "paths": {
                "base_dir": "X:/tmp/base",
                "config_dir": "X:/tmp/config",
                "default_memory_dir": "X:/tmp/memory_default",
                "legacy_memory_dir": "X:/tmp/memory_legacy",
            },
        }
        self._cfg_path.parent.mkdir(parents=True, exist_ok=True)
        self._cfg_path.write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")

        settings = load_config(force_reload=True)
        cfg = self._read_cfg()

        self.assertEqual(str(settings.model_name), "canonical-model")
        self.assertEqual(str(cfg.get("llm", {}).get("model_name") or ""), "canonical-model")
        self.assertFalse(bool(cfg.get("llm", {}).get("json_mode_enabled")))
        self.assertTrue(bool(cfg.get("llm", {}).get("thinking_enabled")))
        self.assertEqual(str(cfg.get("internet", {}).get("web_mode") or ""), "off")
        self.assertEqual(str(cfg.get("internet", {}).get("web_auto_profile") or ""), "balanced")

        console_row = dict(cfg.get("ui", {}).get("console") or {})
        self.assertNotIn("model", console_row)
        self.assertNotIn("json_mode_enabled", console_row)
        self.assertNotIn("last_api_base_url", console_row)
        runtime = dict(console_row.get("runtime") or {})
        self.assertNotIn("think_enabled", runtime)
        self.assertNotIn("web_mode", runtime)
        self.assertNotIn("web_auto_profile", runtime)
        self.assertNotIn("json_mode_enabled", runtime)
        self.assertTrue(bool(runtime.get("mode_lock")))

        self.assertEqual(str(cfg.get("startup", {}).get("safety_mode") or ""), "allow_os_actions")
        self.assertNotIn("read_only_tools", dict(cfg.get("startup") or {}))
        paths_row = dict(cfg.get("paths") or {})
        self.assertNotIn("base_dir", paths_row)
        self.assertNotIn("config_dir", paths_row)
        self.assertNotIn("default_memory_dir", paths_row)
        self.assertNotIn("legacy_memory_dir", paths_row)

    def test_removes_console_runtime_state_from_extra_legacy(self) -> None:
        row = {
            "llm": {"model_name": "canonical-model"},
            "features": {
                "extra_legacy": {
                    "console_runtime_state": {
                        "web_mode": "off",
                        "think_enabled": False,
                    },
                    "keep_me": 42,
                }
            },
        }
        self._cfg_path.parent.mkdir(parents=True, exist_ok=True)
        self._cfg_path.write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")

        load_config(force_reload=True)
        cfg = self._read_cfg()
        extra = dict(cfg.get("features", {}).get("extra_legacy") or {})
        self.assertEqual(int(extra.get("keep_me") or 0), 42)
        self.assertNotIn("console_runtime_state", extra)


if __name__ == "__main__":
    unittest.main()
