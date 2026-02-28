from __future__ import annotations

import functools
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Iterator


@dataclass
class Timer:
    """Context-manager timer with optional metric emission."""

    name: str = ""
    observer: Callable[[str, float], None] | None = None
    auto_record: bool = True
    started_at: float | None = None
    stopped_at: float | None = None

    def __enter__(self) -> "Timer":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.stop()
        return False

    def start(self) -> None:
        self.started_at = time.perf_counter()
        self.stopped_at = None

    def stop(self) -> float:
        if self.started_at is None:
            self.start()
        self.stopped_at = time.perf_counter()
        elapsed = self.elapsed_ms
        if self.auto_record and self.name:
            self._record(elapsed)
        return elapsed

    @property
    def elapsed_s(self) -> float:
        if self.started_at is None:
            return 0.0
        end = self.stopped_at if self.stopped_at is not None else time.perf_counter()
        return float(end - self.started_at)

    @property
    def elapsed_ms(self) -> float:
        return self.elapsed_s * 1000.0

    def _record(self, value_ms: float) -> None:
        if callable(self.observer):
            try:
                self.observer(self.name, value_ms)
                return
            except Exception:
                pass
        try:
            from utils.metrics import observe

            observe(self.name, value_ms)
        except Exception:
            return


@contextmanager
def measure_time() -> Iterator[Callable[[], float]]:
    """Backward-compatible context manager used in existing code.

    Usage:
        with measure_time() as elapsed:
            ...
        print(elapsed())
    """
    start = time.perf_counter()
    yield lambda: time.perf_counter() - start


@contextmanager
def timed_block(name: str = "") -> Iterator[Timer]:
    timer = Timer(name=name)
    timer.start()
    try:
        yield timer
    finally:
        timer.stop()


def timeit(name: str | None = None):
    """Decorator that measures function latency in ms and writes to metrics."""

    def decorator(fn):
        metric_name = str(name or f"{fn.__module__}.{fn.__name__}")

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            with Timer(metric_name):
                return fn(*args, **kwargs)

        return wrapper

    return decorator


def now_ms() -> int:
    return int(time.time() * 1000)
