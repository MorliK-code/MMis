from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ui.settings_schema import SETTINGS_CATEGORIES
from config.settings import get_config_payload


def dotted_get(payload: dict, path: str, default=None):
    current = payload
    for part in [p for p in path.split(".") if p]:
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def iter_specs():
    for category in SETTINGS_CATEGORIES:
        for card in category.cards:
            for spec in card.settings:
                yield category.key, card.title, spec


def main() -> int:
    payload = get_config_payload(force_reload=True)

    missing_in_config: list[str] = []
    stale_aliases: list[str] = []

    for _category, _card, spec in iter_specs():
        value = dotted_get(payload, spec.path, None)

        if value is None and not spec.path.startswith("ui."):
            missing_in_config.append(spec.path)

        if spec.path.endswith(".worker_enabled"):
            stale_aliases.append(spec.path)

    if missing_in_config:
        print("Настройки есть в UI, но отсутствуют в config payload:")
        for path in missing_in_config:
            print("  -", path)

    if stale_aliases:
        print("Найдены старые/нежелательные alias paths:")
        for path in stale_aliases:
            print("  -", path)

    if missing_in_config or stale_aliases:
        return 1

    print("OK: settings schema matches runtime config payload.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
