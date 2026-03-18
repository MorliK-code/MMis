#!/usr/bin/env python
"""Migrate existing memory storage to compact profile."""

import json
import time
from pathlib import Path

from memory.memory_manager import MemoryManager
from memory.memory_policy import MemoryPolicy
from memory.memory_models import MemoryRecord, MemorySourceKind


def compact_record_metadata(metadata: dict, source_kind: str = "user") -> dict:
    """Apply compact profile to existing metadata."""
    return MemoryPolicy.sanitize_metadata_for_storage(
        metadata=metadata,
        source_kind=MemorySourceKind(source_kind) if source_kind else MemorySourceKind.USER,
        thinking="",
        storage_profile="compact",
    )


def main():
    print("=== Миграция хранилища памяти к компактному профилю ===\n")
    
    print("Инициализация MemoryManager...")
    manager = MemoryManager()
    
    print(f"Текущий storage profile: {manager._storage_profile}")
    
    # Читаем все записи
    print("\nЧтение всех записей из хранилища...")
    all_records = list(manager._store.iter_records())
    print(f"Всего записей: {len(all_records)}")
    
    if not all_records:
        print("Хранилище пустое. Нечего мигрировать.")
        return
    
    # Считаем размер до миграции
    total_size_before = sum(len(json.dumps(r.metadata, ensure_ascii=False)) for r in all_records)
    total_keys_before = sum(len(r.metadata) for r in all_records)
    
    print(f"\nДо миграции:")
    print(f"  Общий размер metadata: {total_size_before:,} байт ({total_size_before/1024:.1f} KB)")
    print(f"  Среднее кол-во ключей: {total_keys_before/len(all_records):.1f}")
    
    # Мигрируем записи
    print("\nМиграция записей...")
    migrated_count = 0
    total_size_after = 0
    total_keys_after = 0
    
    for i, record in enumerate(all_records):
        # Получаем source_kind из metadata
        source_kind = record.metadata.get("source_kind", "user")
        
        # Применяем компактизацию
        compact_meta = compact_record_metadata(record.metadata, source_kind)
        
        # Создаём новую запись с компактным metadata
        new_record = MemoryRecord(
            id=record.id,
            text=record.text,
            memory_type=record.memory_type,
            level=record.level,
            scope=record.scope,
            namespace=record.namespace,
            metadata=compact_meta,
            embedding=record.embedding,
            importance=record.importance,
            confidence=record.confidence,
            created_at=record.created_at,
            updated_at=record.updated_at,
            expires_at=record.expires_at,
            status=record.status,
            version=record.version + 1,  # Инкрементируем версию
            parent_id=record.parent_id,
            chunk_index=record.chunk_index,
            source_event_id=record.source_event_id,
            embedding_model=record.embedding_model,
            embedding_fingerprint=record.embedding_fingerprint,
            embedding_version=record.embedding_version,
        )
        
        # Сохраняем обратно
        manager._store.upsert(new_record)
        
        # Считаем размер после
        meta_json = json.dumps(compact_meta, ensure_ascii=False)
        total_size_after += len(meta_json)
        total_keys_after += len(compact_meta)
        
        migrated_count += 1
        
        if (i + 1) % 50 == 0:
            print(f"  Обработано: {i + 1}/{len(all_records)}")
    
    print(f"\nМиграция завершена: {migrated_count} записей")
    
    # Статистика после миграции
    print(f"\nПосле миграции:")
    print(f"  Общий размер metadata: {total_size_after:,} байт ({total_size_after/1024:.1f} KB)")
    print(f"  Среднее кол-во ключей: {total_keys_after/len(all_records):.1f}")
    
    # Экономия
    saved_bytes = total_size_before - total_size_after
    saved_percent = (saved_bytes / total_size_before * 100) if total_size_before > 0 else 0
    
    print(f"\n=== Экономия ===")
    print(f"  Сокращение: {saved_bytes:,} байт ({saved_bytes/1024:.1f} KB)")
    print(f"  Процент: {saved_percent:.1f}%")
    
    # Создаём бэкап старого экспорта
    print("\n=== Создание бэкапа ===")
    backup_path = Path("data/exports/memory_export_pre_migration.json")
    backup_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Экспортируем "как было" из старых записей
    old_export = {
        "exported_at": time.time(),
        "note": "Pre-migration backup",
        "records": [r.to_dict() for r in all_records],
    }
    
    # Восстанавливаем старые metadata для бэкапа
    # (они уже изменены в хранилище, но у нас есть original в all_records)
    old_export["records"] = []
    for r in all_records:
        record_dict = r.to_dict()
        old_export["records"].append(record_dict)
    
    with open(backup_path, "w", encoding="utf-8") as f:
        json.dump(old_export, f, indent=2, ensure_ascii=False)
    
    print(f"Бэкап сохранён: {backup_path}")
    
    # Новый экспорт
    print("\n=== Создание нового экспорта ===")
    new_export_path = Path("data/exports/memory_export.json")
    
    new_export = {
        "exported_at": time.time(),
        "storage_profile": "compact",
        "migrated": True,
        "records": [r.to_dict() for r in manager._store.iter_records()],
    }
    
    with open(new_export_path, "w", encoding="utf-8") as f:
        json.dump(new_export, f, indent=2, ensure_ascii=False)
    
    new_export_size = new_export_path.stat().st_size
    print(f"Новый экспорт: {new_export_path} ({new_export_size:,} байт, {new_export_size/1024:.1f} KB)")
    
    print("\n=== Готово ===")
    print("Старые данные мигрированы к компактному профилю.")
    print(f"Бэкап старых данных: {backup_path}")


if __name__ == "__main__":
    main()
