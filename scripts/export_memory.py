#!/usr/bin/env python
"""Export memory from main storage with compact profile."""

import json
import time
from pathlib import Path

from memory.memory_manager import MemoryManager
from memory.memory_models import DebugRequest


def main():
    print("Инициализация MemoryManager...")
    manager = MemoryManager()
    
    print(f"Storage profile: {manager._storage_profile}")
    
    # Получаем все записи
    print("\nЧтение всех записей...")
    all_records = list(manager._store.iter_records())
    print(f"Всего записей: {len(all_records)}")
    
    # Формируем экспорт
    export_data = {
        "exported_at": time.time(),
        "storage_profile": manager._storage_profile,
        "filters": {
            "namespace": "all",
            "memory_type": "",
            "status": "",
            "contains": "",
            "include_embedding": False
        },
        "store": {
            "memory_backend": manager._store.__class__.__name__,
            "exported_count": len(all_records),
        },
        "records": [r.to_dict() for r in all_records],
    }
    
    # Сохраняем
    export_path = Path("data/exports/memory_export.json")
    export_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(export_path, "w", encoding="utf-8") as f:
        json.dump(export_data, f, indent=2, ensure_ascii=False)
    
    export_size = export_path.stat().st_size
    
    print(f"\n=== Результаты ===")
    print(f"Записей: {len(all_records)}")
    print(f"Размер экспорта: {export_size:,} байт ({export_size/1024:.1f} KB)")
    print(f"Экспорт сохранён: {export_path}")
    
    # Анализируем размер metadata
    if all_records:
        total_metadata_size = 0
        total_keys = 0
        for r in all_records:
            meta_json = json.dumps(r.metadata, ensure_ascii=False)
            total_metadata_size += len(meta_json)
            total_keys += len(r.metadata)
        
        avg_metadata_size = total_metadata_size / len(all_records)
        avg_keys = total_keys / len(all_records)
        
        print(f"\n=== Статистика metadata ===")
        print(f"Средний размер metadata: {avg_metadata_size:.0f} байт")
        print(f"Среднее кол-во ключей: {avg_keys:.1f}")
        print(f"Общий размер metadata: {total_metadata_size:,} байт ({total_metadata_size/1024:.1f} KB)")


if __name__ == "__main__":
    main()
