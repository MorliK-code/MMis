#!/usr/bin/env python
"""Restore memory from export with compact profile."""

import json
import time
from pathlib import Path

from memory.memory_manager import MemoryManager
from memory.memory_policy import MemoryPolicy
from memory.memory_models import (
    MemoryRecord, MemorySourceKind, MemoryType, MemoryLevel, 
    MemoryScope, MemoryStatus
)


def compact_record_metadata(metadata: dict, source_kind: str = "user") -> dict:
    """Apply compact profile to existing metadata."""
    return MemoryPolicy.sanitize_metadata_for_storage(
        metadata=metadata,
        source_kind=MemorySourceKind(source_kind) if source_kind else MemorySourceKind.USER,
        thinking="",
        storage_profile="compact",
    )


def main():
    print("=== Восстановление памяти из экспорта с компактизацией ===\n")
    
    # Читаем старый экспорт
    export_path = Path("data/exports/memory_export.json")
    if not export_path.exists():
        print(f"Файл экспорта не найден: {export_path}")
        return
    
    print(f"Чтение экспорта: {export_path}")
    with open(export_path, "r", encoding="utf-8") as f:
        export_data = json.load(f)
    
    old_records = export_data.get("records", [])
    print(f"Записей в экспорте: {len(old_records)}")
    
    if not old_records:
        print("Экспорт пустой.")
        return
    
    # Считаем размер до
    total_size_before = sum(
        len(json.dumps(r.get("metadata", {}), ensure_ascii=False)) 
        for r in old_records
    )
    total_keys_before = sum(len(r.get("metadata", {})) for r in old_records)
    
    print(f"\nДо компактизации:")
    print(f"  Общий размер metadata: {total_size_before:,} байт ({total_size_before/1024:.1f} KB)")
    print(f"  Среднее кол-во ключей: {total_keys_before/len(old_records):.1f}")
    
    # Инициализируем менеджер
    print("\nИнициализация MemoryManager...")
    manager = MemoryManager()
    
    # Восстанавливаем записи
    print("\nВосстановление записей с компактизацией...")
    migrated_count = 0
    total_size_after = 0
    total_keys_after = 0
    
    for i, old_record in enumerate(old_records):
        # Извлекаем данные
        metadata = dict(old_record.get("metadata") or {})
        source_kind = metadata.get("source_kind", "user")
        
        # Применяем компактизацию
        compact_meta = compact_record_metadata(metadata, source_kind)
        
        # Создаём запись
        try:
            record = MemoryRecord(
                id=str(old_record.get("id", "")),
                text=str(old_record.get("text", "")),
                memory_type=MemoryType(str(old_record.get("memory_type", "message"))),
                level=MemoryLevel(str(old_record.get("level", "l0_working"))),
                scope=MemoryScope(str(old_record.get("scope", "conversation"))),
                namespace=str(old_record.get("namespace", "default")),
                metadata=compact_meta,
                embedding=(old_record.get("embedding") or []),
                importance=float(old_record.get("importance", 0.5)),
                confidence=float(old_record.get("confidence", 0.5)),
                created_at=float(old_record.get("created_at", time.time())),
                updated_at=float(old_record.get("updated_at", time.time())),
                expires_at=(float(old_record["expires_at"]) if old_record.get("expires_at") else None),
                status=MemoryStatus(str(old_record.get("status", "active"))),
                version=int(old_record.get("version", 1)),
                parent_id=(str(old_record.get("parent_id")) if old_record.get("parent_id") else None),
                chunk_index=(int(old_record["chunk_index"]) if old_record.get("chunk_index") is not None else None),
                source_event_id=str(old_record.get("source_event_id", "")),
                embedding_model=str(old_record.get("embedding_model", "")),
                embedding_fingerprint=str(old_record.get("embedding_fingerprint", "")),
                embedding_version=str(old_record.get("embedding_version", "")),
            )
            
            # Сохраняем
            manager._store.upsert(record)
            
            # Считаем размер после
            meta_json = json.dumps(compact_meta, ensure_ascii=False)
            total_size_after += len(meta_json)
            total_keys_after += len(compact_meta)
            
            migrated_count += 1
            
            if (i + 1) % 100 == 0:
                print(f"  Обработано: {i + 1}/{len(old_records)}")
                
        except Exception as e:
            print(f"  Ошибка при записи {old_record.get('id', 'unknown')}: {e}")
    
    print(f"\nВосстановлено: {migrated_count}/{len(old_records)} записей")
    
    # Статистика после
    print(f"\nПосле компактизации:")
    print(f"  Общий размер metadata: {total_size_after:,} байт ({total_size_after/1024:.1f} KB)")
    print(f"  Среднее кол-во ключей: {total_keys_after/len(old_records):.1f}")
    
    # Экономия
    saved_bytes = total_size_before - total_size_after
    saved_percent = (saved_bytes / total_size_before * 100) if total_size_before > 0 else 0
    
    print(f"\n=== Экономия ===")
    print(f"  Сокращение: {saved_bytes:,} байт ({saved_bytes/1024:.1f} KB)")
    print(f"  Процент: {saved_percent:.1f}%")
    
    # Создаём новый экспорт
    print("\n=== Создание нового экспорта ===")
    new_export_path = Path("data/exports/memory_export_compact.json")
    
    new_export = {
        "exported_at": time.time(),
        "storage_profile": "compact",
        "migrated_from": str(export_path),
        "records_count": migrated_count,
        "records": [r.to_dict() for r in manager._store.iter_records()],
    }
    
    with open(new_export_path, "w", encoding="utf-8") as f:
        json.dump(new_export, f, indent=2, ensure_ascii=False)
    
    new_export_size = new_export_path.stat().st_size
    old_export_size = export_path.stat().st_size
    
    print(f"Старый экспорт: {export_path} ({old_export_size:,} байт, {old_export_size/1024:.1f} KB)")
    print(f"Новый экспорт: {new_export_path} ({new_export_size:,} байт, {new_export_size/1024:.1f} KB)")
    
    file_saved = old_export_size - new_export_size
    file_saved_percent = (file_saved / old_export_size * 100) if old_export_size > 0 else 0
    print(f"\n  Сокращение файла: {file_saved:,} байт ({file_saved/1024:.1f} KB, {file_saved_percent:.1f}%)")
    
    print("\n=== Готово ===")


if __name__ == "__main__":
    main()
