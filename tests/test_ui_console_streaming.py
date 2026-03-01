<<<<<<< HEAD
﻿from __future__ import annotations

from _output_utils import enable_unittest_json_output
enable_unittest_json_output()
=======
from __future__ import annotations
>>>>>>> 51b8456b510b4061cb471a6e7b7574d205e99e04

import io
import unittest
from contextlib import redirect_stdout

from ui_console import _ConsoleChunkRenderer, _StreamRealtimePrinter, _sanitize_stream_text


class ConsoleStreamingTests(unittest.TestCase):
    def test_thinking_first_buffers_answer_until_thinking(self):
        renderer = _ConsoleChunkRenderer()
        printer = _StreamRealtimePrinter(renderer, prefer_thinking_first=True, show_thinking=True)
        sink = io.StringIO()

        with redirect_stdout(sink):
            printer.on_answer("A1")
            printer.on_answer("A2")
            printer.on_thinking("T1")
            printer.on_answer("A3")
            printer.finalize()

        out = sink.getvalue()
        self.assertTrue(out.startswith("thinking> "))
        self.assertIn("\nassistant> ", out)
        self.assertEqual(printer.rendered_thinking(), "T1")
        self.assertEqual(printer.rendered_answer(), "A1A2A3")

    def test_no_thinking_preference_streams_answer_immediately(self):
        renderer = _ConsoleChunkRenderer()
        printer = _StreamRealtimePrinter(renderer, prefer_thinking_first=False, show_thinking=False)
        sink = io.StringIO()

        with redirect_stdout(sink):
            printer.on_answer("B1")
            printer.on_answer("B2")
            printer.finalize()

        self.assertEqual(sink.getvalue(), "assistant> B1B2")
        self.assertEqual(printer.rendered_answer(), "B1B2")
        self.assertEqual(printer.rendered_thinking(), "")

    def test_renderer_extracts_output_from_safety_json_stream(self):
        renderer = _ConsoleChunkRenderer()
        self.assertEqual(renderer.feed('{"safe":true,"output":"Hel'), "")
        self.assertEqual(renderer.feed('lo!"}'), "Hello!")

    def test_sanitize_stream_text_removes_carriage_returns(self):
        self.assertEqual(_sanitize_stream_text("ab\rcd"), "ab\ncd")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
<<<<<<< HEAD

=======
>>>>>>> 51b8456b510b4061cb471a6e7b7574d205e99e04
