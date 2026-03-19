"""
convert_covid.py
================
Конвертация датасета COVID-19 Radiography → ImageFolder для классификации.

Исходная структура (Kaggle):
    COVID-19_Radiography_Database/
    ├── COVID/
    │   ├── images/
    │   │   ├── COVID-1.png
    │   │   └── ...
    │   └── masks/   (маски лёгких — нам не нужны)
    ├── Normal/
    │   ├── images/
    │   └── masks/
    ├── Lung_Opacity/
    │   ├── images/
    │   └── masks/
    └── Viral Pneumonia/
        ├── images/
        └── masks/

Примечание: в некоторых версиях датасета изображения лежат прямо в папке класса
без подпапки images/. Скрипт обрабатывает оба варианта.

Целевая структура (ImageFolder):
    COVID_clf/
    ├── train/
    │   ├── COVID/
    │   ├── Normal/
    │   ├── Lung_Opacity/
    │   └── Viral_Pneumonia/
    ├── valid/
    │   └── ...
    └── test/
        └── ...

Разбивка: 70% train / 15% valid / 15% test (seed=42, стратифицированная)
"""

import shutil
import random
from pathlib import Path

# ─── НАСТРОЙКИ ────────────────────────────────────────────────────────────────
SOURCE_PATH = Path(r"C:\datasets\COVID-19_Radiography_Database")  # ← исходный датасет
OUTPUT_PATH = Path(r"C:\datasets\COVID_clf")                      # ← результат

# Маппинг оригинальных имён папок → имена классов в выходном датасете
# (пробелы в именах папок заменяем на _)
CLASS_MAP = {
    "COVID":            "COVID",
    "Normal":           "Normal",
    "Lung_Opacity":     "Lung_Opacity",
    "Viral Pneumonia":  "Viral_Pneumonia",
}

TRAIN_RATIO = 0.70
VALID_RATIO = 0.15
TEST_RATIO  = 0.15
RANDOM_SEED = 42
IMAGE_EXTS  = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}
# ──────────────────────────────────────────────────────────────────────────────


def find_images_in_class_dir(class_dir: Path) -> list[Path]:
    """
    Ищет изображения в папке класса.
    Поддерживает два варианта структуры:
      1. class_dir/images/*.png  (новая версия Kaggle)
      2. class_dir/*.png         (старая версия)
    """
    images_subdir = class_dir / "images"
    if images_subdir.exists() and images_subdir.is_dir():
        search_dir = images_subdir
    else:
        search_dir = class_dir

    result = []
    for p in search_dir.iterdir():
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            result.append(p)
    return sorted(result)


def split_files(files: list[Path], seed: int) -> tuple[list, list, list]:
    """Воспроизводимая разбивка на train/valid/test."""
    rng = random.Random(seed)
    files = files.copy()
    rng.shuffle(files)

    n = len(files)
    n_train = int(n * TRAIN_RATIO)
    n_valid = int(n * VALID_RATIO)

    return files[:n_train], files[n_train:n_train + n_valid], files[n_train + n_valid:]


def main():
    if not SOURCE_PATH.exists():
        raise FileNotFoundError(f"Не найдена папка датасета: {SOURCE_PATH}")

    # Находим папки классов (ищем по CLASS_MAP)
    found_classes = {}
    for orig_name, out_name in CLASS_MAP.items():
        class_dir = SOURCE_PATH / orig_name
        if class_dir.exists():
            found_classes[orig_name] = (class_dir, out_name)
        else:
            print(f"[WARNING] Папка класса не найдена: {class_dir}")

    if not found_classes:
        raise RuntimeError(
            f"Не найдено ни одной папки классов в {SOURCE_PATH}.\n"
            f"Ожидались: {list(CLASS_MAP.keys())}"
        )

    print(f"Найдено классов: {len(found_classes)}")

    # Создаём выходные папки
    out_class_names = [out_name for _, out_name in found_classes.values()]
    for split in ("train", "valid", "test"):
        for cls_name in out_class_names:
            (OUTPUT_PATH / split / cls_name).mkdir(parents=True, exist_ok=True)

    total_stats = {}

    for orig_name, (class_dir, out_name) in found_classes.items():
        images = find_images_in_class_dir(class_dir)
        print(f"\nКласс '{orig_name}' → '{out_name}': {len(images)} изображений")

        if not images:
            print(f"  [WARNING] Изображения не найдены, пропускаем")
            continue

        train_files, valid_files, test_files = split_files(images, RANDOM_SEED)
        split_map = {"train": train_files, "valid": valid_files, "test": test_files}
        total_stats[out_name] = {}

        for split_name, files in split_map.items():
            print(f"  {split_name}: {len(files)}")
            total_stats[out_name][split_name] = len(files)

            for src in files:
                dst = OUTPUT_PATH / split_name / out_name / src.name
                shutil.copy2(src, dst)

    # Итоговая статистика
    print("\n" + "="*50)
    print("ИТОГОВАЯ СТРУКТУРА:")
    print("="*50)
    grand_total = 0
    for split in ("train", "valid", "test"):
        split_total = 0
        print(f"\n{split}/")
        for cls_name in out_class_names:
            if cls_name in total_stats:
                count = total_stats[cls_name].get(split, 0)
                print(f"  {cls_name}: {count}")
                split_total += count
        print(f"  Итого {split}: {split_total}")
        grand_total += split_total

    print(f"\n✓ Всего скопировано: {grand_total} изображений")
    print(f"✓ Результат: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
