from __future__ import annotations

try:
    from _output_utils import enable_unittest_json_output
except ModuleNotFoundError:
    from tests._output_utils import enable_unittest_json_output

enable_unittest_json_output()

import unittest
import subprocess
from contextlib import redirect_stdout
from io import StringIO
from types import SimpleNamespace
from unittest.mock import patch

from ui_console import (
    ConsoleState,
    _handle_command,
    _print_help,
    _stop_runtime_ollama_model,
    _sync_runtime_pref_to_settings_file,
    _sync_ui_state_to_settings_file,
)


class _ApiStub:
    base_url = "http://127.0.0.1:8000"


class _ApiRuntimeStub:
    def __init__(self, base_url: str, model: str) -> None:
        self.base_url = base_url
        self._model = model

    def get_runtime_model(self) -> str:
        return str(self._model)

    def health(self) -> dict:
        return {"model": self._model}


class UiConsoleCommandRoutingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.state = ConsoleState(api=_ApiStub())

    def _assert_passthrough(self, command: str) -> None:
        with patch("ui_console._ensure_connected_or_start", return_value=True), patch(
            "ui_console._send_chat", return_value=0
        ) as send_chat:
            keep_running = _handle_command(self.state, command)
            self.assertTrue(keep_running)
            send_chat.assert_called_once()
            args, kwargs = send_chat.call_args
            self.assertGreaterEqual(len(args), 2)
            self.assertIs(args[0], self.state)
            self.assertEqual(args[1], command)

    def test_output_status_passthrough(self) -> None:
        self._assert_passthrough("/output status")

    def test_brain_debug_passthrough(self) -> None:
        self._assert_passthrough("/brain_debug")

    def test_mode_lock_passthrough(self) -> None:
        self._assert_passthrough("/mode_lock on")

    def test_output_parameters_persisted_after_successful_passthrough(self) -> None:
        with patch("ui_console._ensure_connected_or_start", return_value=True), patch(
            "ui_console._send_chat", return_value=0
        ), patch("ui_console._set_runtime_pref") as set_pref:
            keep_running = _handle_command(self.state, "/output parameters on")
            self.assertTrue(keep_running)
            set_pref.assert_called_once_with(self.state, "output_parameters", True)

    def test_help_stays_local(self) -> None:
        with patch("ui_console._send_chat", return_value=0) as send_chat:
            keep_running = _handle_command(self.state, "/help")
            self.assertTrue(keep_running)
            send_chat.assert_not_called()

    def test_cleanapi_stays_local(self) -> None:
        with patch("ui_console.clean_mmis_api_processes", return_value=2) as cleaner, patch(
            "ui_console._send_chat", return_value=0
        ) as send_chat:
            keep_running = _handle_command(self.state, "/cleanapi")
            self.assertTrue(keep_running)
            cleaner.assert_called_once()
            send_chat.assert_not_called()

    def test_restartall_stays_local_and_keeps_loop(self) -> None:
        with patch("ui_console._restart_full_app", return_value=True) as restart_all, patch(
            "ui_console._print_health_short"
        ) as health_short, patch("ui_console._ensure_model_backend") as ensure_backend, patch(
            "ui_console._send_chat", return_value=0
        ) as send_chat:
            keep_running = _handle_command(self.state, "/restartall")
            self.assertTrue(keep_running)
            restart_all.assert_called_once_with(self.state)
            health_short.assert_called_once_with(self.state)
            ensure_backend.assert_called_once_with(self.state)
            send_chat.assert_not_called()

    def test_hide_thinking_saves_ui_state(self) -> None:
        with patch("ui_console._save_ui_state") as save_state:
            keep_running = _handle_command(self.state, "/hide-thinking")
            self.assertTrue(keep_running)
            self.assertFalse(self.state.show_thinking)
            save_state.assert_called_once_with(self.state)

    def test_mode_command_persists_mode_lock_on(self) -> None:
        with patch("ui_console._ensure_connected_or_start", return_value=True), patch(
            "ui_console._send_chat", return_value=0
        ), patch("ui_console._set_runtime_pref") as set_pref:
            keep_running = _handle_command(self.state, "/mode helper")
            self.assertTrue(keep_running)
            set_pref.assert_any_call(self.state, "active_mode", "helper")
            set_pref.assert_any_call(self.state, "mode_lock", True)

    def test_mode_auto_command_persists_mode_lock_off(self) -> None:
        with patch("ui_console._ensure_connected_or_start", return_value=True), patch(
            "ui_console._send_chat", return_value=0
        ), patch("ui_console._set_runtime_pref") as set_pref:
            keep_running = _handle_command(self.state, "/mode auto")
            self.assertTrue(keep_running)
            set_pref.assert_called_once_with(self.state, "mode_lock", False)

    def test_web_auto_status_passthrough(self) -> None:
        with patch("ui_console._ensure_connected_or_start", return_value=True), patch(
            "ui_console._send_chat", return_value=0
        ) as send_chat:
            keep_running = _handle_command(self.state, "/web-auto status")
            self.assertTrue(keep_running)
            send_chat.assert_called_once()
            args, kwargs = send_chat.call_args
            self.assertEqual(args[1], "/web-auto status")
            self.assertTrue(bool(kwargs.get("command_output")))

    def test_web_auto_profile_persisted_after_passthrough(self) -> None:
        with patch("ui_console._ensure_connected_or_start", return_value=True), patch(
            "ui_console._send_chat", return_value=0
        ), patch("ui_console._set_runtime_pref") as set_pref:
            keep_running = _handle_command(self.state, "/web-auto aggressive")
            self.assertTrue(keep_running)
            set_pref.assert_called_once_with(self.state, "web_auto_profile", "aggressive")

    def test_help_text_has_no_mojibake_placeholders(self) -> None:
        buf = StringIO()
        with redirect_stdout(buf):
            _print_help()
        text = buf.getvalue()
        self.assertNotIn("????????", text)
        self.assertIn("/studio apply|cancel", text)
        self.assertIn("локальные команды студии", text)


    def test_stop_runtime_ollama_model_for_local_api_calls_ollama_stop(self) -> None:
        state = ConsoleState(api=_ApiRuntimeStub("http://127.0.0.1:8000", "qwen3:8b"))
        with patch("ui_console.subprocess.run") as run_mock:
            run_mock.side_effect = [
                SimpleNamespace(returncode=0, stdout="NAME ID SIZE PROCESSOR CONTEXT UNTIL\nqwen3:8b 500a1f067a9f 6.0GB cpu 4096 soon\n"),
                SimpleNamespace(returncode=0, stdout="NAME ID SIZE MODIFIED\nqwen3:8b 500a1f067a9f 6.0GB now\n"),
                SimpleNamespace(returncode=0, stdout=""),
                SimpleNamespace(returncode=0, stdout="NAME ID SIZE PROCESSOR CONTEXT UNTIL\n"),
            ]
            ok = _stop_runtime_ollama_model(state)
            self.assertTrue(ok)
            self.assertGreaterEqual(len(run_mock.call_args_list), 4)
            self.assertEqual(
                run_mock.call_args_list[0].kwargs,
                {
                    "stdout": subprocess.PIPE,
                    "stderr": subprocess.DEVNULL,
                    "check": False,
                    "timeout": 8.0,
                    "text": True,
                    "encoding": "utf-8",
                    "errors": "replace",
                },
            )
            self.assertEqual(run_mock.call_args_list[0].args[0], ["ollama", "ps"])
            self.assertEqual(
                run_mock.call_args_list[1].kwargs,
                {
                    "stdout": subprocess.PIPE,
                    "stderr": subprocess.DEVNULL,
                    "check": False,
                    "timeout": 8.0,
                    "text": True,
                    "encoding": "utf-8",
                    "errors": "replace",
                },
            )
            self.assertEqual(run_mock.call_args_list[1].args[0], ["ollama", "list"])
            self.assertEqual(run_mock.call_args_list[2].args[0], ["ollama", "stop", "qwen3:8b"])
            self.assertEqual(run_mock.call_args_list[3].args[0], ["ollama", "ps"])

    def test_stop_runtime_ollama_model_skips_remote_api(self) -> None:
        state = ConsoleState(api=_ApiRuntimeStub("http://10.0.0.9:8000", "qwen3:8b"))
        with patch("ui_console.subprocess.run") as run_mock:
            ok = _stop_runtime_ollama_model(state)
            self.assertFalse(ok)
            run_mock.assert_not_called()

    def test_sync_runtime_pref_model_writes_only_canonical_key(self) -> None:
        with patch("ui_console.update_config_values") as update_cfg:
            _sync_runtime_pref_to_settings_file("model", "qwen3:8b")
            update_cfg.assert_called_once()
            updates = dict(update_cfg.call_args.args[0] or {})
            self.assertEqual(str(updates.get("llm.model_name") or ""), "qwen3:8b")
            self.assertNotIn("ui.console.model", updates)

    def test_sync_runtime_pref_json_writes_only_canonical_key(self) -> None:
        with patch("ui_console.update_config_values") as update_cfg:
            _sync_runtime_pref_to_settings_file("json_mode_enabled", True)
            update_cfg.assert_called_once()
            updates = dict(update_cfg.call_args.args[0] or {})
            self.assertTrue(bool(updates.get("llm.json_mode_enabled")))
            self.assertNotIn("ui.console.json_mode_enabled", updates)
            self.assertNotIn("ui.console.runtime.json_mode_enabled", updates)

    def test_sync_ui_state_filters_global_runtime_keys_from_config(self) -> None:
        state = ConsoleState(api=_ApiStub())
        state.store_turn = True
        state.show_thinking = True
        state.thinking_first = True
        state.prefs = {
            "runtime": {
                "think_enabled": False,
                "web_mode": "off",
                "web_auto_profile": "aggressive",
                "json_mode_enabled": True,
                "mode_lock": True,
                "active_mode": "helper",
                "output_parameters": True,
                "output_summary": False,
            }
        }
        with patch("ui_console.update_config_values") as update_cfg:
            _sync_ui_state_to_settings_file(state)
            update_cfg.assert_called_once()
            updates = dict(update_cfg.call_args.args[0] or {})
            runtime = dict(updates.get("ui.console.runtime") or {})
            self.assertIn("mode_lock", runtime)
            self.assertIn("active_mode", runtime)
            self.assertIn("output_parameters", runtime)
            self.assertIn("output_summary", runtime)
            self.assertNotIn("think_enabled", runtime)
            self.assertNotIn("web_mode", runtime)
            self.assertNotIn("web_auto_profile", runtime)
            self.assertNotIn("json_mode_enabled", runtime)


if __name__ == "__main__":
    unittest.main()
