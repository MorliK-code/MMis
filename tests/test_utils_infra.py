from __future__ import annotations

try:
    from _output_utils import enable_unittest_json_output
except ModuleNotFoundError:
    from tests._output_utils import enable_unittest_json_output
enable_unittest_json_output()

import sys
import tempfile
import time
import unittest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from modules.character.dialog_policies import deterministic_term_gate_score
from utils.logger import configure_logging, get_logger, log_json
from utils.metrics import inc, observe, reset, set_gauge, snapshot
from utils.timers import Timer, measure_time
from utils.validators import clamp_params, safe_path, validate_tool_call_schema


class UtilsInfraTests(unittest.TestCase):
    def setUp(self) -> None:
        reset()

    def test_metrics_and_timer(self) -> None:
        with Timer("unit.test.timer"):
            time.sleep(0.01)
        inc("counter_a")
        set_gauge("queue_size", 3)
        observe("latency_ms", 12.0)
        shot = snapshot()

        self.assertGreaterEqual(shot["counters"].get("counter_a", 0.0), 1.0)
        self.assertEqual(shot["gauges"].get("queue_size"), 3.0)
        self.assertIn("unit.test.timer", shot["histograms"])

    def test_measure_time_backward_compat(self) -> None:
        with measure_time() as elapsed:
            time.sleep(0.005)
        self.assertGreater(elapsed(), 0.0)

    def test_validators(self) -> None:
        ok, err = validate_tool_call_schema({"tool": "search", "args": {"q": "x"}})
        self.assertTrue(ok)
        self.assertEqual(err, "")

        params = clamp_params(temperature=7.0, top_p=-1.0, max_tokens=999999)
        self.assertEqual(params["temperature"], 2.0)
        self.assertEqual(params["top_p"], 0.0)
        self.assertEqual(params["max_tokens"], 200000)

        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            inside = safe_path(base / "a.txt", base)
            self.assertTrue(str(inside).startswith(str(base)))

    def test_logger_helpers(self) -> None:
        configure_logging()
        logger = get_logger("unit.logger")
        log_json(logger, "evt", x=1)
        logger.info("plain")

    def test_deterministic_gate_helper_stable(self) -> None:
        score_a = deterministic_term_gate_score(
            conversation_id="conv-1",
            turn_id=7,
            allowed_term="милашка",
            user_text="тест",
        )
        score_b = deterministic_term_gate_score(
            conversation_id="conv-1",
            turn_id=7,
            allowed_term="милашка",
            user_text="тест",
        )
        self.assertEqual(score_a, score_b)


if __name__ == "__main__":
    unittest.main()


