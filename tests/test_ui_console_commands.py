from __future__ import annotations

try:
    from _output_utils import enable_unittest_json_output
except ModuleNotFoundError:
    from tests._output_utils import enable_unittest_json_output

enable_unittest_json_output()

import unittest
from contextlib import redirect_stdout
from io import StringIO
from unittest.mock import patch

from ui_console import ConsoleState, _handle_command, _print_help


class _ApiStub:
    base_url = "http://127.0.0.1:8000"


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

    def test_help_text_has_no_mojibake_placeholders(self) -> None:
        buf = StringIO()
        with redirect_stdout(buf):
            _print_help()
        text = buf.getvalue()
        self.assertNotIn("????????", text)
        self.assertIn("/studio apply|cancel", text)
        self.assertIn("локальные команды студии", text)


if __name__ == "__main__":
    unittest.main()
