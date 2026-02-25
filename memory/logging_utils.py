import logging
from collections import Counter
from typing import Dict


LOGGER_NAME = "memory"


def get_memory_logger() -> logging.Logger:
    logger = logging.getLogger(LOGGER_NAME)
    if logger.handlers:
        return logger

    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger


class ErrorMetrics:
    def __init__(self) -> None:
        self._counters: Counter = Counter()

    def increment(self, category: str) -> None:
        self._counters[category] += 1

    def snapshot(self) -> Dict[str, int]:
        return dict(self._counters)

