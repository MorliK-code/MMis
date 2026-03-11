from __future__ import annotations

from _output_utils import enable_unittest_json_output

enable_unittest_json_output()

import unittest

from ui import api_client as api_client_module
from ui.api_client import ApiClient


class _FakeStreamResponse:
    def __init__(self, lines: list[str]):
        self._lines = [str(x).encode("utf-8") + b"\n" for x in lines]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False

    def __iter__(self):
        return iter(self._lines)


class UiApiClientOutputFieldsTests(unittest.TestCase):
    def setUp(self) -> None:
        self._orig_urlopen = api_client_module.urllib_request.urlopen

    def tearDown(self) -> None:
        api_client_module.urllib_request.urlopen = self._orig_urlopen

    def test_stream_reply_parses_parameters_and_summary(self) -> None:
        lines = [
            '{"event":"chunk","data":"hello"}',
            (
                '{"event":"final","data":{"answer":"hello","thinking":"","stats":{"served_model":"stub"},'
                '"model":"stub","parameters":{"mode":"debugger"},"summary":"short"}}'
            ),
        ]
        api_client_module.urllib_request.urlopen = lambda req, timeout=None: _FakeStreamResponse(lines)
        client = ApiClient(base_url="http://example.test")
        reply = client.stream_chat("hi", store_turn=False)
        self.assertEqual(reply.answer, "hello")
        self.assertEqual(dict(reply.parameters or {}).get("mode"), "debugger")
        self.assertEqual(reply.summary, "short")

    def test_stream_reply_defaults_to_none_when_fields_absent(self) -> None:
        lines = [
            '{"event":"chunk","data":"hello"}',
            '{"event":"final","data":{"answer":"hello","thinking":"","stats":{},"model":"stub"}}',
        ]
        api_client_module.urllib_request.urlopen = lambda req, timeout=None: _FakeStreamResponse(lines)
        client = ApiClient(base_url="http://example.test")
        reply = client.stream_chat("hi", store_turn=False)
        self.assertEqual(reply.answer, "hello")
        self.assertIsNone(reply.parameters)
        self.assertIsNone(reply.summary)


if __name__ == "__main__":
    unittest.main()
