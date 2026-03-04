from __future__ import annotations

try:
    from _output_utils import enable_unittest_json_output
except ModuleNotFoundError:
    from tests._output_utils import enable_unittest_json_output

enable_unittest_json_output()

import unittest
from unittest.mock import patch

from ui_console import ConsoleState, _handle_command


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
            send_chat.assert_called_once_with(self.state, command)

    def test_output_status_passthrough(self) -> None:
        self._assert_passthrough("/output status")

    def test_brain_debug_passthrough(self) -> None:
        self._assert_passthrough("/brain_debug")

    def test_mode_lock_passthrough(self) -> None:
        self._assert_passthrough("/mode_lock on")

    def test_help_stays_local(self) -> None:
        with patch("ui_console._send_chat", return_value=0) as send_chat:
            keep_running = _handle_command(self.state, "/help")
            self.assertTrue(keep_running)
            send_chat.assert_not_called()


if __name__ == "__main__":
    unittest.main()
