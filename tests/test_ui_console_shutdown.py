from __future__ import annotations

import unittest
from unittest import mock

import ui_console


class UiConsoleShutdownTests(unittest.TestCase):
    def test_stop_memory_ollama_model_uses_configured_memory_profile(self) -> None:
        with mock.patch.object(ui_console, "_stop_ollama_models", return_value=True) as stop_models:
            self.assertTrue(ui_console._stop_memory_ollama_model())

        stop_models.assert_called_once_with(["qwen3:4b"])

    def test_stop_api_process_unloads_memory_model_even_without_live_api_process(self) -> None:
        state = ui_console.ConsoleState(api=mock.Mock(), api_process=None)

        with mock.patch.object(ui_console, "_stop_runtime_ollama_model", return_value=False) as stop_runtime:
            with mock.patch.object(ui_console, "_stop_memory_ollama_model", return_value=True) as stop_memory:
                with mock.patch.object(ui_console, "_stop_owned_ollama_process", return_value=False) as stop_owned:
                    ui_console._stop_api_process(state)

        stop_runtime.assert_called_once_with(state)
        stop_memory.assert_called_once_with()
        stop_owned.assert_called_once_with(state)


if __name__ == "__main__":
    unittest.main()
