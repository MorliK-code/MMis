from __future__ import annotations

from _output_utils import enable_unittest_json_output

enable_unittest_json_output()

import json
import logging
import tempfile
import unittest
from pathlib import Path

from config.settings import AppSettings, setup_logging
from utils.logger import get_logger, log_json


class LoggingConfigTests(unittest.TestCase):
    def test_web_trace_logger_writes_ndjson_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir) / "logs"
            settings = AppSettings(log_dir=log_dir, log_level="INFO")
            try:
                setup_logging(settings=settings, force=True)

                logger = get_logger("web.trace")
                log_json(
                    logger,
                    "web_trace_test",
                    trace_id="trace-unit-1",
                    route="chat",
                    payload={"ok": True},
                )
                for handler in list(logger.handlers):
                    try:
                        handler.flush()
                    except Exception:
                        pass
                for handler in list(logging.getLogger().handlers):
                    try:
                        handler.flush()
                    except Exception:
                        pass

                path = log_dir / "web_trace.jsonl"
                self.assertTrue(path.exists())
                rows = [x for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
                self.assertTrue(rows)
                payload = json.loads(rows[-1])
                self.assertEqual(str(payload.get("event") or ""), "web_trace_test")
                self.assertEqual(str(payload.get("trace_id") or ""), "trace-unit-1")
            finally:
                logging.shutdown()


if __name__ == "__main__":
    unittest.main()
