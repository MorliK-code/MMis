from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ui.settings_schema import SETTINGS_CATEGORIES
from ui.settings_window import _DESCRIPTION_BY_PATH, _TITLE_BY_PATH


def main() -> int:
    missing_title: list[str] = []
    missing_description: list[str] = []

    for category in SETTINGS_CATEGORIES:
        for card in category.cards:
            for spec in card.settings:
                title = str(_TITLE_BY_PATH.get(spec.path) or spec.title or "").strip()
                description = str(_DESCRIPTION_BY_PATH.get(spec.path) or spec.description or "").strip()

                if not title:
                    missing_title.append(spec.path)

                if not description:
                    missing_description.append(spec.path)

    if missing_title:
        print("Нет title:")
        for path in missing_title:
            print("  -", path)

    if missing_description:
        print("Нет description:")
        for path in missing_description:
            print("  -", path)

    if missing_title or missing_description:
        return 1

    print("OK: у всех настроек есть title и description для подсказок.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
