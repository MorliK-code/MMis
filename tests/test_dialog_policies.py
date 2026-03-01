from __future__ import annotations

try:
    from _output_utils import enable_unittest_json_output
except ModuleNotFoundError:
    from tests._output_utils import enable_unittest_json_output

enable_unittest_json_output()

import json
import sys
import unittest
from datetime import datetime
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from modules.character.dialog_policies import (
    compute_dialog_flags,
    compute_dialog_mode,
    deterministic_term_gate_score,
    is_technical,
    is_user_greeting,
)

RU_HELLO = "\u043f\u0440\u0438\u0432\u0435\u0442"
RU_FORWARD_HELLO = "\u043f\u0435\u0440\u0435\u0434\u0430\u0439 \u043f\u0440\u0438\u0432\u0435\u0442"
RU_TERM = "\u043c\u0438\u043b\u0430\u0448\u043a\u0430"
RU_DISABLE_TERM = "\u043d\u0435 \u043d\u0430\u0437\u044b\u0432\u0430\u0439 \u043c\u0435\u043d\u044f \u043c\u0438\u043b\u0430\u0448\u043a\u0430"
RU_ENABLE_TERM = "\u043c\u043e\u0436\u043d\u043e \u0441\u043d\u043e\u0432\u0430 \u043d\u0430\u0437\u044b\u0432\u0430\u0442\u044c \u043c\u0438\u043b\u0430\u0448\u043a\u0430"


class DialogPoliciesTests(unittest.TestCase):
    def test_is_user_greeting_simple(self) -> None:
        self.assertTrue(is_user_greeting(RU_HELLO))

    def test_is_user_greeting_exclusion(self) -> None:
        self.assertFalse(is_user_greeting(RU_FORWARD_HELLO))

    def test_technical_with_greeting_disables_smalltalk(self) -> None:
        flags = compute_dialog_flags(
            f"{RU_HELLO}, \u0443 \u043c\u0435\u043d\u044f \u043e\u0448\u0438\u0431\u043a\u0430 Traceback: ValueError",
            now="2026-03-01T10:00:00+02:00",
            state={"last_turn_ts": "2026-03-01T09:59:00+02:00", "conversation_state": "continuing_smalltalk"},
            config={"new_session_after_min": 360, "greeting_max_words": 6, "greeting_max_chars": 35},
        )
        self.assertFalse(flags["smalltalk_allowed"])
        self.assertFalse(flags["allow_greeting"])

    def test_long_technical_message(self) -> None:
        src = "Traceback (most recent call last): File \"C:\\repo\\main.py\", line 1, in <module> Exception: fail"
        self.assertTrue(is_technical(src))
        flags = compute_dialog_flags(
            src,
            now="2026-03-01T10:00:00+02:00",
            state={"last_turn_ts": "2026-03-01T09:58:00+02:00", "conversation_state": "continuing_smalltalk"},
            config={"new_session_after_min": 360, "greeting_max_words": 6, "greeting_max_chars": 35},
        )
        self.assertFalse(flags["smalltalk_allowed"])

    def test_dialog_mode_shape(self) -> None:
        flags = compute_dialog_flags(
            RU_HELLO,
            now="2026-03-01T10:00:00+02:00",
            state={"last_turn_ts": "2026-03-01T06:00:00+02:00", "conversation_state": "new_session"},
        )
        mode = dict(flags.get("dialog_mode") or {})
        self.assertIn("greeting_allowed", mode)
        self.assertIn("smalltalk_allowed", mode)
        self.assertIn("sarcasm_level", mode)
        self.assertIn("warmth_level", mode)
        self.assertIn("strictness_level", mode)
        self.assertIn("verbosity_level", mode)

    def test_technical_mode_forces_low_sarcasm(self) -> None:
        mode = compute_dialog_mode(
            text="Traceback: ValueError in C:\\repo\\main.py",
            now="2026-03-01T10:00:00+02:00",
            state={"last_turn_ts": "2026-03-01T09:58:00+02:00", "conversation_state": "continuing_smalltalk"},
            metadata={"intent": "coding_help", "emotion": "neutral"},
        )["dialog_mode"]
        self.assertFalse(mode["smalltalk_allowed"])
        self.assertLessEqual(float(mode["sarcasm_level"]), 0.12)
        self.assertGreaterEqual(float(mode["strictness_level"]), 0.75)

    def test_disable_directive_bans_term_immediately(self) -> None:
        state = {
            "conversation_id": "conv-a",
            "turn_id": 4,
            "address_terms": {
                "banned_terms": [],
                "banned_terms_until": {},
                "terms_used_count_session": 0,
                "term_used_turn_index": 0,
                "last_term_used_at": 0.0,
            },
        }
        flags = compute_dialog_flags(RU_DISABLE_TERM, now="2026-03-01T10:00:00+02:00", state=state)
        policy = dict(flags.get("address_terms_policy") or {})
        self.assertIn(RU_TERM, list(policy.get("ban_updates", {}).get("add_terms") or []))
        self.assertFalse(bool(policy.get("use_term_now", True)))

    def test_enable_directive_unbans_term(self) -> None:
        state = {
            "conversation_id": "conv-a",
            "turn_id": 5,
            "address_terms": {
                "banned_terms": [RU_TERM],
                "banned_terms_until": {RU_TERM: "session:conv-a"},
                "terms_used_count_session": 0,
                "term_used_turn_index": 0,
                "last_term_used_at": 0.0,
            },
        }
        flags = compute_dialog_flags(RU_ENABLE_TERM, now="2026-03-01T10:01:00+02:00", state=state)
        policy = dict(flags.get("address_terms_policy") or {})
        self.assertIn(RU_TERM, list(policy.get("unban_updates", {}).get("remove_terms") or []))

    def test_terms_cooldown_requires_turns_and_seconds(self) -> None:
        state = {
            "conversation_id": "conv-a",
            "turn_id": 8,
            "address_terms": {
                "last_term_used_at": "2026-03-01T09:58:30+02:00",
                "term_used_turn_index": 2,
                "terms_used_count_session": 1,
                "session_id_snapshot": "conv-a",
                "banned_terms": [],
                "banned_terms_until": {},
            },
        }
        flags = compute_dialog_flags(
            "ok",
            now="2026-03-01T10:00:00+02:00",
            state=state,
            config={"terms_cooldown_turns": 6, "terms_cooldown_seconds": 900},
        )
        policy = dict(flags.get("address_terms_policy") or {})
        self.assertTrue(bool(policy.get("cooldown_passed_turns")))
        self.assertFalse(bool(policy.get("cooldown_passed_seconds")))
        self.assertFalse(bool(policy.get("cooldown_passed")))

    def test_terms_max_per_session_blocks_after_limit(self) -> None:
        state = {
            "conversation_id": "conv-a",
            "turn_id": 10,
            "address_terms": {
                "last_term_used_at": 0.0,
                "term_used_turn_index": 0,
                "terms_used_count_session": 3,
                "session_id_snapshot": "conv-a",
                "banned_terms": [],
                "banned_terms_until": {},
            },
        }
        flags = compute_dialog_flags(RU_HELLO, now="2026-03-01T10:00:00+02:00", state=state, config={"terms_max_per_session": 3})
        policy = dict(flags.get("address_terms_policy") or {})
        self.assertFalse(bool(policy.get("session_limit_ok")))
        self.assertFalse(bool(policy.get("use_term_now")))

    def test_deterministic_gate_is_stable(self) -> None:
        a = deterministic_term_gate_score(
            conversation_id="conv-a",
            turn_id=12,
            allowed_term=RU_TERM,
            user_text="\u043f\u0440\u043e\u0441\u0442\u043e \u0442\u0435\u0441\u0442",
        )
        b = deterministic_term_gate_score(
            conversation_id="conv-a",
            turn_id=12,
            allowed_term=RU_TERM,
            user_text="\u043f\u0440\u043e\u0441\u0442\u043e \u0442\u0435\u0441\u0442",
        )
        self.assertEqual(a, b)

    def test_json_output_cases(self) -> None:
        out_dir = ROOT_DIR / "tests" / "output" / "dialog_policies_nametest"
        self.assertTrue(out_dir.exists(), f"Missing output dir: {out_dir}")
        path = out_dir / "cases_20260301.json"
        self.assertTrue(path.exists(), f"Missing output file: {path}")
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        cases = list(payload.get("cases") or [])
        self.assertGreaterEqual(len(cases), 4)

        for idx, row in enumerate(cases, start=1):
            test_num = int(row.get("test_num") or 0)
            self.assertEqual(test_num, idx, f"test_num sequence mismatch in {path.name}")
            case = str(row.get("case") or f"case_{idx}")
            text = str(row.get("text") or "")
            now = row.get("now")
            state = dict(row.get("state") or {})
            config = dict(row.get("config") or {})
            expected = dict(row.get("expected") or {})
            flags = compute_dialog_flags(text, now=now, state=state, config=config)
            for key, value in expected.items():
                self.assertEqual(flags.get(key), value, f"Case={case}, key={key}")

    def test_auto_creates_data_dialog_police_config(self) -> None:
        path = ROOT_DIR / "data" / "dialog_police.py"
        _ = compute_dialog_flags(RU_HELLO, now=None, state={"conversation_state": "new_session"})
        self.assertTrue(path.exists(), f"Expected auto-generated config file: {path}")

    def test_local_date_and_region_use_system_timezone(self) -> None:
        flags = compute_dialog_flags(RU_HELLO, now=None, state={"conversation_state": "new_session"})
        expected_local_date = datetime.now().astimezone().date().isoformat()
        self.assertEqual(flags.get("local_date"), expected_local_date)
        self.assertTrue(str(flags.get("local_region") or "").strip())


if __name__ == "__main__":
    unittest.main()

