# LLM Priority Manager — Приоритеты LLM

## Конфигурация

Параметры в `memory_core/config.json`:

```json
{
  "llm": {
    "priority": {
      "enabled": true,
      "level": 1,
      "wait_timeout": 300.0,
      "description": "Priority 0=main (highest), 1=memory (lower). Memory LLM waits if main LLM is active."
    }
  }
}
```

**Параметры:**
- `enabled` (bool): Включить управление приоритетами
- `level` (int): Приоритет Memory LLM (0=main, 1=memory)
- `wait_timeout` (float): Максимальное время ожидания (сек)

## Описание

Главная LLM всегда в приоритете.
Memory LLM приостанавливается, если главная LLM нуждается в ресурсах.

## Архитектура

```
┌─────────────────────────────────────┐
│     LLMPriorityManager              │
│                                     │
│  PRIORITY_MAIN = 0 (highest)       │
│  PRIORITY_MEMORY = 1 (lower)       │
│                                     │
│  - Счётчики активных запросов      │
│  - Очередь ожидания для Memory LLM │
│  - Уведомления при освобождении    │
└─────────────────────────────────────┘
           ▲              ▲
           │              │
    ┌──────┴──────┐ ┌────┴────────┐
    │ Main LLM    │ │ Memory LLM  │
    │ (priority 0)│ │ (priority 1)│
    └─────────────┘ └─────────────┘
```

## Как работает

### Главная LLM (priority 0)
- Всегда получает доступ немедленно
- Если Memory LLM активен — главная LLM начинает выполнение, Memory LLM ждёт

### Memory LLM (priority 1)
- Получает доступ только если главная LLM не активна
- Если главная LLM активна — ждёт в очереди (timeout 300 сек)
- Получает уведомление когда главная LLM завершается

## API

### Базовое использование

```python
from llm.priority_manager import get_priority_manager, LLMPriorityManager

manager = get_priority_manager()

# Главная LLM
manager.acquire(LLMPriorityManager.PRIORITY_MAIN)
try:
    # Выполнение запроса главной LLM
    response = llm.generate(request)
finally:
    manager.release(LLMPriorityManager.PRIORITY_MAIN)

# Memory LLM (с ожиданием)
if manager.wait_for_turn(LLMPriorityManager.PRIORITY_MEMORY, timeout=300.0):
    try:
        # Выполнение запроса Memory LLM
        response = memory_llm.generate(request)
    finally:
        manager.release(LLMPriorityManager.PRIORITY_MEMORY)
else:
    raise TimeoutError("Memory LLM timed out")
```

### Декораторы

```python
from llm.priority_manager import main_llm_call, memory_llm_call

@main_llm_call
def generate_main(request):
    return llm.generate(request)

@memory_llm_call
def generate_memory(request):
    return memory_llm.generate(request)
```

### Статус

```python
status = manager.get_status()
# {
#     "main_llm_active": 0,
#     "memory_llm_active": 0,
#     "memory_llm_waiting": False
# }
```

## Интеграция с OllamaProvider

OllamaProvider автоматически определяет приоритет по metadata.source:

```python
# Memory LLM запрос
req = LLMRequest(
    model="qwen3:4b",
    messages=[...],
    metadata={"source": "memory_llm_process"}  # ← priority 1
)

# Главная LLM запрос
req = LLMRequest(
    model="qwen3:8b",
    messages=[...],
    metadata={"source": "main_chat"}  # ← priority 0 (default)
)
```

## Логирование

```
2026-03-24 | INFO | llm.ollama_provider | llm_generate_start ... priority="main"
2026-03-24 | DEBUG | llm.priority_manager | Main LLM acquired (active: 1)
2026-03-24 | DEBUG | llm.priority_manager | Memory LLM waiting (main LLM active: 1)
2026-03-24 | DEBUG | llm.priority_manager | Main LLM released (active: 0)
2026-03-24 | DEBUG | llm.priority_manager | Memory LLM acquired (active: 1)
```

## Конфигурация

### Таймаут ожидания

Memory LLM ждёт главную LLM максимум 300 секунд (5 минут):

```python
manager.wait_for_turn(
    LLMPriorityManager.PRIORITY_MEMORY,
    timeout=300.0  # 5 минут
)
```

### Изменение приоритетов

Приоритеты заданы константами:

```python
LLMPriorityManager.PRIORITY_MAIN = 0      # Highest
LLMPriorityManager.PRIORITY_MEMORY = 1    # Lower
```

## Сценарии использования

### Сценарий 1: Главная LLM отвечает пользователю

```
User: "Привет!"
    ↓
Main LLM acquire (priority 0) → ✅ Granted
    ↓
Main LLM generates response
    ↓
Main LLM release
    ↓
Memory LLM can now process
```

### Сценарий 2: Memory LLM обрабатывает, пользователь прерывает

```
Memory LLM processing (priority 1)
    ↓
User: "Срочный вопрос!"
    ↓
Main LLM acquire (priority 0) → ✅ Granted (preempts memory)
    ↓
Main LLM responds immediately
    ↓
Main LLM release
    ↓
Memory LLM resumes (was waiting)
```

### Сценарий 3: Multiple Memory LLM tasks

```
Memory LLM task 1 (priority 1) → ✅ Active
Memory LLM task 2 (priority 1) → ⏳ Waiting (main_llm_active=0, but memory_llm_active=1)
    ↓
Task 1 completes
    ↓
Memory LLM task 2 → ✅ Active
```

## Преимущества

1. **Главная LLM всегда приоритетна** — пользователь не ждёт
2. **Memory LLM не блокирует** — приостанавливается автоматически
3. **Прозрачная интеграция** — декораторы или явные вызовы
4. **Логирование** — видно кто и когда ждёт
5. **Timeout защита** — Memory LLM не ждёт бесконечно

## Файлы

- `llm/priority_manager.py` — менеджер приоритетов
- `llm/ollama_provider.py` — интеграция с Ollama

## Тесты

```bash
python -c "from llm.priority_manager import *; print('OK')"
```
