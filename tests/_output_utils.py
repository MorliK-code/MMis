from __future__ import annotations

import json
import os
import threading
import time
import unittest
from datetime import datetime
from pathlib import Path
from typing import Any

_LOCK = threading.Lock()
_HOOK_ENABLED = False
_PROCESS_ID = os.getpid()

_OUTPUT_DIR = Path(__file__).resolve().parent / "output"
_OUTPUT_FILE = _OUTPUT_DIR / "tests_output.json"


def enable_unittest_json_output() -> None:
    """Enable automatic unittest result logging to tests/output/tests_output.json."""
    global _HOOK_ENABLED
    if _HOOK_ENABLED:
        return
    _HOOK_ENABLED = True

    original_run = unittest.TestCase.run

    def _wrapped_run(self: unittest.TestCase, result=None):  # type: ignore[override]
        started = time.perf_counter()
        run_result = original_run(self, result)
        duration_ms = round((time.perf_counter() - started) * 1000.0, 3)

        try:
            active_result = result or run_result or getattr(getattr(self, "_outcome", None), "result", None)
            status = _resolve_status(self, active_result)
            detail = _resolve_detail(self, active_result, status)
            write_case_output(
                test_id=self.id(),
                status=status,
                duration_ms=duration_ms,
                detail=detail,
                module=self.__class__.__module__,
                class_name=self.__class__.__name__,
                test_name=getattr(self, "_testMethodName", ""),
            )
        except Exception:
            # Never break tests because of output logger issues.
            pass

        return run_result

    unittest.TestCase.run = _wrapped_run  # type: ignore[assignment]


def write_case_output(
    *,
    test_id: str,
    status: str,
    duration_ms: float,
    detail: str = "",
    module: str = "",
    class_name: str = "",
    test_name: str = "",
) -> None:
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with _LOCK:
        payload = _read_output()
        cases = list(payload.get("cases") or [])
        next_num = len(cases) + 1

        row: dict[str, Any] = {
            "test_num": next_num,
            "timestamp": _now_iso(),
            "pid": _PROCESS_ID,
            "test_id": str(test_id or ""),
            "module": str(module or ""),
            "class": str(class_name or ""),
            "test_name": str(test_name or ""),
            "status": str(status or "unknown"),
            "duration_ms": float(duration_ms or 0.0),
        }
        if detail:
            row["detail"] = str(detail)

        cases.append(row)
        payload["cases"] = cases
        payload["updated_at"] = _now_iso()
        _write_output(payload)


def _resolve_status(test: unittest.TestCase, result: Any) -> str:
    if result is None:
        return "passed"
    if _contains_test(getattr(result, "failures", []), test):
        return "failed"
    if _contains_test(getattr(result, "errors", []), test):
        return "error"
    if _contains_test(getattr(result, "skipped", []), test):
        return "skipped"
    if _contains_test(getattr(result, "expectedFailures", []), test):
        return "expected_failure"
    if _contains_test(getattr(result, "unexpectedSuccesses", []), test):
        return "unexpected_success"
    return "passed"


def _resolve_detail(test: unittest.TestCase, result: Any, status: str) -> str:
    if result is None:
        return ""

    if status == "failed":
        return _lookup_detail(getattr(result, "failures", []), test)
    if status == "error":
        return _lookup_detail(getattr(result, "errors", []), test)
    if status == "skipped":
        return _lookup_detail(getattr(result, "skipped", []), test)
    if status == "expected_failure":
        return _lookup_detail(getattr(result, "expectedFailures", []), test)
    return ""


def _lookup_detail(rows: Any, test: unittest.TestCase) -> str:
    test_id = _safe_test_id(test)
    for row in list(rows or []):
        if isinstance(row, tuple) and row:
            item_test = row[0]
            item_detail = row[1] if len(row) > 1 else ""
        else:
            item_test = row
            item_detail = ""
        if _safe_test_id(item_test) == test_id:
            detail = str(item_detail or "").strip()
            if len(detail) > 2000:
                detail = detail[:2000] + " ...<trimmed>"
            return detail
    return ""


def _contains_test(rows: Any, test: unittest.TestCase) -> bool:
    test_id = _safe_test_id(test)
    for row in list(rows or []):
        item_test = row[0] if isinstance(row, tuple) and row else row
        if _safe_test_id(item_test) == test_id:
            return True
    return False


def _safe_test_id(test: Any) -> str:
    try:
        return str(test.id())
    except Exception:
        return str(test)


def _read_output() -> dict[str, Any]:
    if not _OUTPUT_FILE.exists():
        return {"version": 1, "created_at": _now_iso(), "updated_at": _now_iso(), "cases": []}
    try:
        payload = json.loads(_OUTPUT_FILE.read_text(encoding="utf-8-sig"))
    except Exception:
        payload = {}
    if not isinstance(payload, dict):
        payload = {}
    payload.setdefault("version", 1)
    payload.setdefault("created_at", _now_iso())
    payload.setdefault("updated_at", _now_iso())
    payload.setdefault("cases", [])
    if not isinstance(payload.get("cases"), list):
        payload["cases"] = []
    return payload


def _write_output(payload: dict[str, Any]) -> None:
    tmp = _OUTPUT_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(_OUTPUT_FILE)


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")

