from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TARGET_NAME = "studio_generator.py"

def is_mojibake_line(s: str) -> bool:
    # сигнатуры как в твоём примере: "С„Р°Р·Р°", "С‚РµРє..."
    markers = ("ф", "т", "я", "ь", "ш", "щ", "ч", "ю", "э", "а", "о", "е", "и")
    hits = sum(s.count(m) for m in markers)
    return hits >= 2  # достаточно мягко, но не слишком

matches = list(ROOT.rglob(TARGET_NAME))
if not matches:
    raise FileNotFoundError(f"Не нашёл {TARGET_NAME} внутри {ROOT}")
if len(matches) > 1:
    print("Найдено несколько файлов:")
    for p in matches:
        print(" -", p)
    raise SystemExit("Уточни TARGET_NAME.")

p = matches[0]

text = p.read_text(encoding="utf-8-sig", errors="strict")
lines = text.splitlines(keepends=True)

changed = 0
out = []

for line in lines:
    if is_mojibake_line(line):
        # Пытаемся восстановить. Если в строке есть неподдерживаемые символы — просто пропускаем её.
        try:
            fixed_line = line.encode("cp1251").decode("utf-8")
            out.append(fixed_line)
            changed += 1
            continue
        except UnicodeEncodeError:
            # fallback: игнорируем мусорные символы, но возвращаем русский
            fixed_line = line.encode("cp1251", errors="ignore").decode("utf-8", errors="replace")
            out.append(fixed_line)
            changed += 1
            continue
        except UnicodeDecodeError:
            out.append(line)
            continue
    out.append(line)

new_text = "".join(out)

if new_text == text:
    print("Ничего не изменилось:", p)
else:
    p.write_text(new_text, encoding="utf-8")
    print(f"Исправлено строк: {changed}")
    print("Файл:", p)