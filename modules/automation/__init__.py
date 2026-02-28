from __future__ import annotations

from modules.automation.browser_controller import BrowserConfig, BrowserController
from modules.automation.os_actions import ActionResult, OSActions, OSActionConfig, run_os_action
from modules.automation.task_executor import Task, TaskExecutor, TaskResult, TaskStep, execute_task

__all__ = [
    "ActionResult",
    "OSActionConfig",
    "OSActions",
    "run_os_action",
    "BrowserConfig",
    "BrowserController",
    "Task",
    "TaskStep",
    "TaskResult",
    "TaskExecutor",
    "execute_task",
]
