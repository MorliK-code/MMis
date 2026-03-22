from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from core.spec_registry import SpecRegistry


class SpecRegistryMigrationTests(unittest.TestCase):
    def test_legacy_migration_skips_performance_profiles_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_root = Path(tmp)
            src_root = tmp_root / "legacy_specs"
            dst_root = tmp_root / "rules_for_all"
            src_root.mkdir(parents=True, exist_ok=True)
            dst_root.mkdir(parents=True, exist_ok=True)

            (src_root / "taxonomy.json").write_text('{"schema_version": 1}', encoding="utf-8")
            (src_root / "performance_profiles.json").write_text('{"profiles": {}}', encoding="utf-8")

            registry = SpecRegistry(root=dst_root)
            registry._legacy_root = src_root
            registry._migrate_legacy_specs()

            self.assertTrue((dst_root / "taxonomy.json").exists())
            self.assertFalse((dst_root / "performance_profiles.json").exists())


if __name__ == "__main__":
    unittest.main()
