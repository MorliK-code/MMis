from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from dataclasses import dataclass
from typing import Any, Optional

import ollama

from config import MODEL_NAME, build_ollama_options

logger = logging.getLogger(__name__)


@dataclass
class PromptTimings:
    answer_ms: float
    total_duration_ms: Optional[float] = None
    eval_duration_ms: Optional[float] = None
    prompt_eval_duration_ms: Optional[float] = None


@dataclass
class PromptUsage:
    eval_count: Optional[int] = None
    prompt_eval_count: Optional[int] = None


@dataclass
class PromptResponse:
    content: str
    raw: dict[str, Any]
    timings: PromptTimings
    usage: PromptUsage


DEFAULT_TIMEOUT_SEC = 20.0


def _duration_ms(value: Any) -> Optional[float]:
    if not isinstance(value, (int, float)):
        return None
    return round(float(value) / 1_000_000, 1)


def run_chat_prompt(
    *,
    task_type: str,
    messages: list[dict[str, str]],
    options_override: Optional[dict[str, Any]] = None,
    timeout_sec: Optional[float] = None,
    default_content: str = "",
) -> PromptResponse:
    options = build_ollama_options(task_type)
    if options_override:
        options.update(options_override)

    t0 = time.perf_counter()

    def _call() -> dict[str, Any]:
        return ollama.chat(model=MODEL_NAME, messages=messages, options=options)

    try:
        if timeout_sec is None:
            raw = _call()
        else:
            with ThreadPoolExecutor(max_workers=1) as executor:
                raw = executor.submit(_call).result(timeout=timeout_sec)
    except TimeoutError:
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
        logger.warning("llm timeout task_type=%s timeout_sec=%s", task_type, timeout_sec)
        return PromptResponse(
            content=default_content,
            raw={},
            timings=PromptTimings(answer_ms=elapsed_ms),
            usage=PromptUsage(),
        )
    except Exception:
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
        logger.exception("llm failure task_type=%s", task_type)
        return PromptResponse(
            content=default_content,
            raw={},
            timings=PromptTimings(answer_ms=elapsed_ms),
            usage=PromptUsage(),
        )

    elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
    content = ((raw.get("message", {}) or {}).get("content", "") or "").strip()

    response = PromptResponse(
        content=content,
        raw=raw,
        timings=PromptTimings(
            answer_ms=elapsed_ms,
            total_duration_ms=_duration_ms(raw.get("total_duration")),
            eval_duration_ms=_duration_ms(raw.get("eval_duration")),
            prompt_eval_duration_ms=_duration_ms(raw.get("prompt_eval_duration")),
        ),
        usage=PromptUsage(
            eval_count=raw.get("eval_count"),
            prompt_eval_count=raw.get("prompt_eval_count"),
        ),
    )
    logger.info("latency.%s_ms=%.2f", task_type, elapsed_ms)
    return response
