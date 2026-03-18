#!/usr/bin/env python
"""Test compact memory storage profile."""

import json
import tempfile
from pathlib import Path

from memory.memory_manager import MemoryManager
from memory.memory_models import MemoryEvent, MemoryScope, MemoryType, DebugRequest


def main():
    # Создаём временный менеджер для теста
    temp_dir = Path(tempfile.mkdtemp(prefix="mmis-test-"))
    print(f"Using temp dir: {temp_dir}")
    
    manager = MemoryManager(root_dir=temp_dir / "memory")
    
    # Записываем несколько тестовых сообщений
    test_messages = [
        ("user", "Привет! Меня зовут Александр."),
        ("assistant", "Привет, Александр! Чем могу помочь?"),
        ("user", "Я работаю на Python 3.11 с RTX 3050 Ti и 32 GB RAM."),
        ("assistant", "Отлично! Вижу у тебя мощная система."),
        ("user", "Какая у меня ОС?"),
        ("assistant", "Не помню точную модель. Проверь сам через winver."),
    ]
    
    print("\nЗаписываем тестовые сообщения...")
    for role, text in test_messages:
        result = manager.ingest_event(
            MemoryEvent(
                role=role,
                text=text,
                namespace="test-conv",
                scope=MemoryScope.CONVERSATION,
                memory_type=MemoryType.MESSAGE,
                metadata={"conversation_id": "test-conv"},
            )
        )
        print(f"  {role}: stored={len(result.stored_ids)}, facts={len(result.extracted_facts)}")
    
    # Экспортируем
    print("\nЭкспортируем память...")
    
    # Проверяем напрямую хранилище
    all_records = list(manager._store.iter_records(namespace="test-conv"))
    print(f"Найдено записей в хранилище: {len(all_records)}")
    
    # Формируем экспорт вручную
    export_data = {
        "exported_at": __import__("time").time(),
        "storage_profile": manager._storage_profile,
        "records": [r.to_dict() for r in all_records],
    }
    
    # Сохраняем экспорт
    export_path = temp_dir / "memory_export.json"
    with open(export_path, "w", encoding="utf-8") as f:
        json.dump(export_data, f, indent=2, ensure_ascii=False)
    
    # Считаем размер
    export_size = export_path.stat().st_size
    records_count = len(export_data.get("records", []))
    
    print(f"\n=== Результаты ===")
    print(f"Записей: {records_count}")
    print(f"Размер экспорта: {export_size:,} байт ({export_size/1024:.1f} KB)")
    
    # Показываем структуру одной записи
    if export_data.get("records"):
        record = export_data["records"][0]
        metadata = record.get("metadata", {})
        
        print(f"\n=== Структура metadata первой записи ===")
        print(f"Ключей в metadata: {len(metadata)}")
        
        # Проверяем наличие компактных полей
        has_search_text = "search_text" in metadata
        has_memory_analysis = "memory_analysis" in metadata
        has_promotion_debug = "promotion_debug" in metadata
        has_lifecycle_decision = "lifecycle_decision" in metadata
        has_assistant_write_policy = "assistant_write_policy" in metadata
        
        print(f"  search_text: {has_search_text}")
        print(f"  memory_analysis: {has_memory_analysis} (должно быть False)")
        print(f"  promotion_debug: {has_promotion_debug} (должно быть False)")
        print(f"  lifecycle_decision: {has_lifecycle_decision}")
        print(f"  assistant_write_policy: {has_assistant_write_policy}")
        
        # Показываем размер metadata
        metadata_json = json.dumps(metadata, ensure_ascii=False)
        print(f"\nРазмер metadata: {len(metadata_json):,} байт")
        
        # Показываем пример metadata
        print(f"\n=== Пример metadata (первые 500 символов) ===")
        print(metadata_json[:500] + "..." if len(metadata_json) > 500 else metadata_json)
    
    print(f"\nЭкспорт сохранён: {export_path}")
    print(f"Temp dir: {temp_dir}")


if __name__ == "__main__":
    main()
