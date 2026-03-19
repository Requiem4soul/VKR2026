"""
convert_breakhis.py
===================
Конвертация датасета BreakHis → ImageFolder для классификации.

Исходная структура:
    BreaKHis_v1/
    └── breast/
        ├── benign/
        │   └── SOB/
        │       ├── adenosis/
        │       │   └── SOB_B_A_.../
        │       │       ├── 40X/  *.png
        │       │       ├── 100X/ *.png
        │       │       ├── 200X/ *.png
        │       │       └── 400X/ *.png
        │       ├── fibroadenoma/...
        │       ├── phyllodes_tumor/...
        │       └── tubular_adenoma/...
        └── malignant/
            └── SOB/
                ├── ductal_carcinoma/...
                ├── lobular_carcinoma/...
                ├── mucinous_carcinoma/...
                └── papillary_carcinoma/...

Целевая структура (ImageFolder):
    BreakHis_YOLO/
    ├── train/
    │   ├── benign/
    │   └── malignant/
    ├── valid/
    │   ├── benign/
    │   └── malignant/
    └── test/
        ├── benign/
        └── malignant/

Разбивка: 70% train / 15% valid / 15% test (seed=42)
"""

import shutil
import random
from pathlib import Path

# ─── НАСТРОЙКИ ────────────────────────────────────────────────────────────────
SOURCE_PATH = Path(r"N:\ORIG_Datasets\BreaKHis_v1\histology_slides")   # ← путь к исходному датасету
OUTPUT_PATH = Path(r"N:\ORIG_Datasets\CONVERTED_DATASETS\BreakHis_clf")  # ← куда сохранить результат

TRAIN_RATIO = 0.70
VALID_RATIO = 0.15
TEST_RATIO  = 0.15
RANDOM_SEED = 42
# ──────────────────────────────────────────────────────────────────────────────

assert abs(TRAIN_RATIO + VALID_RATIO + TEST_RATIO - 1.0) < 1e-9, "Сумма долей должна быть 1.0"

CLASSES = ["benign", "malignant"]
SPLITS  = ["train", "valid", "test"]


def collect_images(class_dir: Path) -> list[Path]:
    """Рекурсивно собирает все .png/.jpg/.jpeg из папки класса."""
    images = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tiff"):
        images.extend(class_dir.rglob(ext))
    return sorted(images)


def split_files(files: list[Path], seed: int) -> tuple[list, list, list]:
    """Делит список файлов на train/valid/test воспроизводимо."""
    rng = random.Random(seed)
    files = files.copy()
    rng.shuffle(files)

    n = len(files)
    n_train = int(n * TRAIN_RATIO)
    n_valid = int(n * VALID_RATIO)

    train = files[:n_train]
    valid = files[n_train : n_train + n_valid]
    test  = files[n_train + n_valid:]
    return train, valid, test


def main():
    breast_dir = SOURCE_PATH / "breast"
    if not breast_dir.exists():
        raise FileNotFoundError(f"Не найдена папка: {breast_dir}")

    # Создаём выходную структуру
    for split in SPLITS:
        for cls in CLASSES:
            (OUTPUT_PATH / split / cls).mkdir(parents=True, exist_ok=True)

    total_copied = 0

    for cls in CLASSES:
        cls_dir = breast_dir / cls / "SOB"
        if not cls_dir.exists():
            print(f"[WARNING] Не найдена папка: {cls_dir}, пропускаем")
            continue

        all_images = collect_images(cls_dir)
        print(f"\nКласс '{cls}': найдено {len(all_images)} изображений")

        train_files, valid_files, test_files = split_files(all_images, RANDOM_SEED)
        split_map = {"train": train_files, "valid": valid_files, "test": test_files}

        for split, files in split_map.items():
            print(f"  {split}: {len(files)} файлов")
            for src in files:
                dst = OUTPUT_PATH / split / cls / src.name
                # Если имена совпадают (маловероятно, но возможно) — добавляем суффикс
                if dst.exists():
                    dst = OUTPUT_PATH / split / cls / f"{src.stem}_{src.parent.parent.name}{src.suffix}"
                shutil.copy2(src, dst)
                total_copied += 1

    print(f"\n✓ Готово! Скопировано файлов: {total_copied}")
    print(f"  Результат: {OUTPUT_PATH}")

    # Итоговая статистика
    print("\nСтруктура датасета:")
    for split in SPLITS:
        for cls in CLASSES:
            count = len(list((OUTPUT_PATH / split / cls).glob("*")))
            print(f"  {split}/{cls}: {count}")


if __name__ == "__main__":
    main()
