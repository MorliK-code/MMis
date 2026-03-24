"""
LLM Priority Manager — управление приоритетами LLM.

Главная LLM всегда в приоритете.
Memory LLM приостанавливается, если главная LLM нуждается в ресурсах.

Конфигурация в memory_core/config.json:
{
  "llm": {
    "priority": {
      "enabled": true,
      "level": 1,
      "wait_timeout": 300.0
    }
  }
}
"""

from __future__ import annotations

import threading
import time
from typing import Any

from config.settings import load_config
from utils.logger import get_logger


LOGGER = get_logger(__name__)

# Загружаем конфиг для получения настроек приоритета
_cfg = load_config()
_memory_cfg = getattr(_cfg, 'memory_core', None) if hasattr(_cfg, 'memory_core') else None
_priority_cfg = getattr(_memory_cfg, 'llm', {}).get('priority', {}) if _memory_cfg else {}

# Приоритеты по умолчанию
DEFAULT_PRIORITY_ENABLED = bool(_priority_cfg.get('enabled', True))
DEFAULT_MEMORY_PRIORITY = int(_priority_cfg.get('level', 1))
DEFAULT_WAIT_TIMEOUT = float(_priority_cfg.get('wait_timeout', 300.0))


class LLMPriorityManager:
    """
    Менеджер приоритетов LLM.

    Приоритеты:
    - 0: Главная LLM (highest priority)
    - 1: Memory LLM (lower priority)
    """

    PRIORITY_MAIN = 0
    PRIORITY_MEMORY = DEFAULT_MEMORY_PRIORITY

    def __init__(self, enabled: bool = DEFAULT_PRIORITY_ENABLED, wait_timeout: float = DEFAULT_WAIT_TIMEOUT):
        """
        Инициализирует менеджер приоритетов.

        Args:
            enabled: Включить управление приоритетами.
            wait_timeout: Максимальное время ожидания для Memory LLM (сек).
        """
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        
        # Счётчики активных запросов
        self._main_llm_active = 0
        self._memory_llm_active = 0
        
        # Флаг ожидания
        self._memory_llm_waiting = False
        
        # Настройки
        self.enabled = enabled
        self.wait_timeout = wait_timeout
        
        LOGGER.info(f"LLMPriorityManager initialized (enabled={self.enabled}, timeout={self.wait_timeout}s)")

    def acquire(self, priority: int) -> bool:
        """
        Запрашивает доступ к LLM.

        Args:
            priority: Приоритет запроса (0=main, 1=memory).

        Returns:
            True если доступ получен, False если нужно ждать.
        """
        with self._lock:
            if priority == self.PRIORITY_MAIN:
                # Главная LLM всегда получает доступ
                self._main_llm_active += 1
                
                # Если Memory LLM активен, помечаем что главная ждёт
                if self._memory_llm_active > 0:
                    LOGGER.warning(f"Main LLM waiting for {self._memory_llm_active} memory LLM task(s) to complete")
                
                LOGGER.debug(f"Main LLM acquired (active: {self._main_llm_active})")
                return True
            
            elif priority == self.PRIORITY_MEMORY:
                # Memory LLM получает доступ только если главная не активна
                if self._main_llm_active > 0:
                    self._memory_llm_waiting = True
                    LOGGER.debug(f"Memory LLM waiting (main LLM active: {self._main_llm_active})")
                    return False
                
                self._memory_llm_active += 1
                LOGGER.debug(f"Memory LLM acquired (active: {self._memory_llm_active})")
                return True
            
            return False

    def release(self, priority: int) -> None:
        """
        Освобождает доступ к LLM.

        Args:
            priority: Приоритет запроса (0=main, 1=memory).
        """
        with self._lock:
            if priority == self.PRIORITY_MAIN:
                self._main_llm_active = max(0, self._main_llm_active - 1)
                
                # Если есть ждущие Memory LLM, уведомляем
                if self._memory_llm_waiting and self._main_llm_active == 0:
                    self._condition.notify_all()
                
                LOGGER.debug(f"Main LLM released (active: {self._main_llm_active})")
            
            elif priority == self.PRIORITY_MEMORY:
                self._memory_llm_active = max(0, self._memory_llm_active - 1)
                
                # Если Memory LLM больше не активны, сбрасываем флаг
                if self._memory_llm_active == 0:
                    self._memory_llm_waiting = False
                
                LOGGER.debug(f"Memory LLM released (active: {self._memory_llm_active})")

    def wait_for_turn(self, priority: int, timeout: float | None = None) -> bool:
        """
        Ждёт своей очереди для выполнения.

        Args:
            priority: Приоритет запроса.
            timeout: Максимальное время ожидания (None = из конфига).

        Returns:
            True если доступ получен, False если таймаут.
        """
        # Если приоритеты отключены — всегда разрешаем
        if not self.enabled:
            return self.acquire(priority)
        
        if priority == self.PRIORITY_MAIN:
            # Главная LLM не ждёт
            return self.acquire(priority)
        
        # Memory LLM ждёт
        actual_timeout = timeout if timeout is not None else self.wait_timeout
        deadline = time.time() + actual_timeout
        
        with self._lock:
            while self._main_llm_active > 0:
                remaining = deadline - time.time()
                if remaining <= 0:
                    LOGGER.warning(f"Memory LLM timeout waiting for main LLM ({self.wait_timeout}s)")
                    return False
                
                self._memory_llm_waiting = True
                self._condition.wait(timeout=min(remaining, 1.0))
            
            return self.acquire(priority)

    def get_status(self) -> dict[str, Any]:
        """
        Получает статус менеджера.

        Returns:
            Статус в виде словаря.
        """
        with self._lock:
            return {
                "main_llm_active": self._main_llm_active,
                "memory_llm_active": self._memory_llm_active,
                "memory_llm_waiting": self._memory_llm_waiting,
            }


# Глобальный экземпляр
_priority_manager: LLMPriorityManager | None = None


def get_priority_manager() -> LLMPriorityManager:
    """Получает глобальный менеджер приоритетов."""
    global _priority_manager
    if _priority_manager is None:
        _priority_manager = LLMPriorityManager(
            enabled=DEFAULT_PRIORITY_ENABLED,
            wait_timeout=DEFAULT_WAIT_TIMEOUT,
        )
    return _priority_manager


def reset_priority_manager() -> None:
    """Сбрасывает глобальный менеджер приоритетов."""
    global _priority_manager
    _priority_manager = None


# Декораторы для удобного использования

def main_llm_call(func):
    """
    Декоратор для вызовов главной LLM.

    Автоматически управляет приоритетом.
    """
    def wrapper(*args, **kwargs):
        manager = get_priority_manager()
        manager.acquire(LLMPriorityManager.PRIORITY_MAIN)
        try:
            return func(*args, **kwargs)
        finally:
            manager.release(LLMPriorityManager.PRIORITY_MAIN)
    return wrapper


def memory_llm_call(func):
    """
    Декоратор для вызовов Memory LLM.

    Автоматически управляет приоритетом и ждёт если нужно.
    """
    def wrapper(*args, **kwargs):
        manager = get_priority_manager()
        if not manager.wait_for_turn(LLMPriorityManager.PRIORITY_MEMORY, timeout=300.0):
            raise TimeoutError("Memory LLM timed out waiting for main LLM")
        try:
            return func(*args, **kwargs)
        finally:
            manager.release(LLMPriorityManager.PRIORITY_MEMORY)
    return wrapper
