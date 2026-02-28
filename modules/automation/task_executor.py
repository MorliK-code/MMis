from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

from modules.automation.browser_controller import BrowserController
from modules.automation.os_actions import ActionResult, OSActions
from utils.logger import get_logger


LOGGER = get_logger(__name__)


@dataclass(frozen=True)
class TaskStep:
    id: str
    action: str
    params: dict[str, Any] = field(default_factory=dict)
    retries: int = 0
    backoff_s: float = 0.4
    requires_confirmation: bool = False


@dataclass(frozen=True)
class Task:
    id: str
    goal: str
    steps: list[TaskStep]
    constraints: dict[str, Any] = field(default_factory=dict)
    requires_confirmation: bool = False


@dataclass(frozen=True)
class StepResult:
    step_id: str
    action: str
    ok: bool
    attempts: int
    result: dict[str, Any] = field(default_factory=dict)
    error: str = ""
    duration_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TaskResult:
    id: str
    goal: str
    ok: bool
    steps: list[StepResult] = field(default_factory=list)
    started_at: float = 0.0
    finished_at: float = 0.0
    stopped_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "goal": self.goal,
            "ok": self.ok,
            "steps": [x.to_dict() for x in self.steps],
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "stopped_reason": self.stopped_reason,
        }


class TaskExecutor:
    """Executes step plans; planning and policy stay in core."""

    def __init__(
        self,
        *,
        browser: BrowserController | None = None,
        os_actions: OSActions | None = None,
        event_store: Any | None = None,
        confirm_callback: Callable[[str], bool] | None = None,
        safety_trigger: Callable[[TaskStep], bool] | None = None,
    ):
        self.browser = browser or BrowserController()
        self.os_actions = os_actions or OSActions()
        self.event_store = event_store
        self.confirm_callback = confirm_callback
        self.safety_trigger = safety_trigger

    def execute(self, task: Task | dict[str, Any]) -> TaskResult:
        item = _coerce_task(task)
        started = time.time()
        LOGGER.info("task_execute start id=%s steps=%s goal=%s", item.id, len(item.steps), item.goal)

        if item.requires_confirmation and not self._confirm(item.goal):
            LOGGER.warning("task_execute blocked confirmation id=%s", item.id)
            return TaskResult(
                id=item.id,
                goal=item.goal,
                ok=False,
                steps=[],
                started_at=started,
                finished_at=time.time(),
                stopped_reason="task_confirmation_required",
            )

        step_results: list[StepResult] = []
        stop_on_error = bool(item.constraints.get("stop_on_error", True))
        stopped_reason = ""

        for step in item.steps:
            if callable(self.safety_trigger) and self.safety_trigger(step):
                stopped_reason = "safety_trigger"
                LOGGER.warning("task_execute safety_trigger id=%s step=%s", item.id, step.id)
                break

            if step.requires_confirmation and not self._confirm(f"step:{step.id}:{step.action}"):
                stopped_reason = f"step_confirmation_required:{step.id}"
                LOGGER.warning("task_execute step confirmation blocked id=%s step=%s", item.id, step.id)
                break

            self._log_event("tool_call", {"task_id": item.id, "step_id": step.id, "action": step.action, "params": step.params})
            result = self._execute_with_retries(step)
            step_results.append(result)
            self._log_event(
                "tool_result",
                {
                    "task_id": item.id,
                    "step_id": step.id,
                    "action": step.action,
                    "ok": result.ok,
                    "error": result.error,
                    "result": result.result,
                },
            )

            if (not result.ok) and stop_on_error:
                stopped_reason = f"step_failed:{step.id}"
                LOGGER.warning("task_execute step failed id=%s step=%s error=%s", item.id, step.id, result.error)
                break

        ok = (not stopped_reason) and all(x.ok for x in step_results)
        LOGGER.info(
            "task_execute done id=%s ok=%s steps_done=%s stopped_reason=%s",
            item.id,
            ok,
            len(step_results),
            stopped_reason,
        )
        return TaskResult(
            id=item.id,
            goal=item.goal,
            ok=ok,
            steps=step_results,
            started_at=started,
            finished_at=time.time(),
            stopped_reason=stopped_reason,
        )

    def _execute_with_retries(self, step: TaskStep) -> StepResult:
        attempts = max(1, int(step.retries) + 1)
        started = time.perf_counter()
        last_error = ""
        last_result: dict[str, Any] = {}

        for attempt in range(1, attempts + 1):
            result = self._execute_step(step)
            last_result = result.to_dict() if isinstance(result, ActionResult) else dict(result or {})
            if result.ok:
                return StepResult(
                    step_id=step.id,
                    action=step.action,
                    ok=True,
                    attempts=attempt,
                    result=last_result,
                    error="",
                    duration_ms=(time.perf_counter() - started) * 1000.0,
                )

            last_error = str(result.error or "unknown error")
            if attempt < attempts:
                time.sleep(max(0.0, float(step.backoff_s)) * attempt)

        return StepResult(
            step_id=step.id,
            action=step.action,
            ok=False,
            attempts=attempts,
            result=last_result,
            error=last_error,
            duration_ms=(time.perf_counter() - started) * 1000.0,
        )

    def _execute_step(self, step: TaskStep) -> ActionResult:
        action = str(step.action or "").strip().lower()
        p = dict(step.params or {})

        try:
            if action == "open_url":
                return self.browser.open_url(str(p.get("url", "")))
            if action == "search_in_page":
                return self.browser.search_in_page(str(p.get("text", "")))
            if action == "click":
                return self.browser.click(p.get("selector_or_coords") or p.get("coords"))
            if action == "type":
                return self.browser.type(str(p.get("text", "")))
            if action == "press":
                return self.browser.press(p.get("keys") or p.get("key") or "")
            if action == "scroll":
                return self.browser.scroll(int(p.get("amount", 0)))

            if action == "open_app":
                return self.os_actions.open_app(str(p.get("path") or p.get("name") or ""))
            if action == "focus_window":
                return self.os_actions.focus_window(title=str(p.get("title", "")), process=str(p.get("process", "")))
            if action == "run_command":
                return self.os_actions.run_command(
                    str(p.get("cmd", "")),
                    capture=bool(p.get("capture", True)),
                    cwd=p.get("cwd"),
                    timeout_s=p.get("timeout_s"),
                )
            if action == "read_file":
                return self.os_actions.read_file(p.get("path", ""), max_kb=p.get("max_kb"))
            if action == "write_file":
                return self.os_actions.write_file(p.get("path", ""), str(p.get("content", "")))
            if action == "hotkey":
                keys = p.get("keys") or []
                return self.os_actions.hotkey(*list(keys))
            if action == "type_text":
                return self.os_actions.type_text(str(p.get("text", "")))

            return ActionResult(ok=False, action=action, error=f"unknown action: {action}")
        except Exception as exc:
            return ActionResult(ok=False, action=action, error=str(exc), data={"params": p})

    def _confirm(self, label: str) -> bool:
        if not callable(self.confirm_callback):
            return False
        try:
            return bool(self.confirm_callback(label))
        except Exception:
            return False

    def _log_event(self, event_type: str, payload: dict[str, Any]) -> None:
        store = self.event_store
        if store is None:
            return
        append = getattr(store, "append", None)
        if not callable(append):
            return
        try:
            append({"type": event_type, "payload": payload, "tags": ["task_executor"]})
        except Exception as exc:
            LOGGER.warning("task executor event log failed: %s", exc)


def execute_task(task: dict[str, Any]) -> dict:
    result = _DEFAULT_EXECUTOR.execute(task)
    return result.to_dict()


def _coerce_task(value: Task | dict[str, Any]) -> Task:
    if isinstance(value, Task):
        return value

    raw = dict(value or {})
    steps_raw = list(raw.get("steps") or [])
    steps: list[TaskStep] = []
    for idx, row in enumerate(steps_raw):
        item = dict(row or {})
        steps.append(
            TaskStep(
                id=str(item.get("id") or f"step-{idx+1}"),
                action=str(item.get("action") or ""),
                params=dict(item.get("params") or {}),
                retries=max(0, int(item.get("retries", 0) or 0)),
                backoff_s=max(0.0, float(item.get("backoff_s", 0.4) or 0.0)),
                requires_confirmation=bool(item.get("requires_confirmation", False)),
            )
        )

    return Task(
        id=str(raw.get("id") or f"task-{int(time.time() * 1000)}"),
        goal=str(raw.get("goal") or ""),
        steps=steps,
        constraints=dict(raw.get("constraints") or {}),
        requires_confirmation=bool(raw.get("requires_confirmation", False)),
    )


_DEFAULT_EXECUTOR = TaskExecutor()
