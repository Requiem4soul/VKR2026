"""
convert_minneapple.py
=====================
Конвертация датасета MinneApple → YOLO detection format.

Исходная структура (после распаковки detection.tar.gz):
    detection/
    ├── train/
    │   ├── images/
    │   └── masks/
    └── test/
        ├── images/
        └── masks/  ← пустая / отсутствует (авторы не публиковали тестовые маски)

ВАЖНО: оригинальный test/ не содержит масок, поэтому он НЕ используется.
Все 670 изображений из оригинального train/ делятся 70/15/15 → train/valid/test,
что обеспечивает наличие аннотаций во всех трёх сплитах.

Разбивка: 70% train / 15% valid / 15% test  (seed=42)
  train: 469 изображений
  valid: 100 изображений
  test:  101 изображений

Целевая структура:
    MinneApple_yolo/
    ├── train/ ├── images/ └── labels/
    ├── valid/ ├── images/ └── labels/
    ├── test/  ├── images/ └── labels/
    └── data.yaml

Один класс: apple (0)
"""

import shutil
import random
import yaml
import numpy as np
from pathlib import Path

# ─── НАСТРОЙКИ ────────────────────────────────────────────────────────────────
SOURCE_PATH = Path(r"N:\ORIG_Datasets\detection")
OUTPUT_PATH = Path(r"N:\ORIG_Datasets\CONVERTED_DATASETS\MinneApple_yolo")

TRAIN_RATIO = 0.70
VALID_RATIO = 0.15
# TEST_RATIO  = 1 - TRAIN_RATIO - VALID_RATIO = 0.15 (остаток)
RANDOM_SEED = 42
MIN_BOX_AREA = 100
# ──────────────────────────────────────────────────────────────────────────────

try:
    import cv2
except ImportError:
    raise ImportError("Установите OpenCV: pip install opencv-python")


def mask_to_yolo_bboxes(mask_path: Path, img_width: int, img_height: int) -> list[str]:
    """
    Конвертирует маску → список строк YOLO-формата.
    Каждая связная компонента → отдельный bounding box.
    """
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return []

    _, binary = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)
    num_labels, labels_img = cv2.connectedComponents(binary)

    lines = []
    for label_id in range(1, num_labels):
        component_mask = (labels_img == label_id).astype(np.uint8)
        ys, xs = np.where(component_mask > 0)
        if len(xs) == 0:
            continue

        x_min, x_max = int(xs.min()), int(xs.max())
        y_min, y_max = int(ys.min()), int(ys.max())
        box_w = x_max - x_min + 1
        box_h = y_max - y_min + 1

        if box_w * box_h < MIN_BOX_AREA:
            continue

        cx = max(0.0, min(1.0, (x_min + x_max) / 2.0 / img_width))
        cy = max(0.0, min(1.0, (y_min + y_max) / 2.0 / img_height))
        nw = max(0.0, min(1.0, box_w / img_width))
        nh = max(0.0, min(1.0, box_h / img_height))

        lines.append(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

    return lines


def process_split(src_split_dir: Path, dst_split_dir: Path,
                  img_list: list[Path]) -> dict:
    """Обрабатывает один сплит: конвертирует маски и копирует изображения."""
    stats = {"images": 0, "labels": 0, "boxes": 0, "no_mask": 0}

    for img_path in img_list:
        mask_path = src_split_dir / "masks" / img_path.name

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  [WARNING] Не удалось прочитать: {img_path.name}")
            continue

        h, w = img.shape[:2]

        shutil.copy2(img_path, dst_split_dir / "images" / img_path.name)
        stats["images"] += 1

        label_path = dst_split_dir / "labels" / (img_path.stem + ".txt")
        if mask_path.exists():
            yolo_lines = mask_to_yolo_bboxes(mask_path, w, h)
            if yolo_lines:
                label_path.write_text("\n".join(yolo_lines) + "\n")
                stats["labels"] += 1
                stats["boxes"]  += len(yolo_lines)
            else:
                label_path.write_text("")
        else:
            label_path.write_text("")
            stats["no_mask"] += 1

    return stats


def main():
    src_train_dir = SOURCE_PATH / "train"

    if not src_train_dir.exists():
        raise FileNotFoundError(f"Не найдена папка: {src_train_dir}")

    # Собираем все изображения из оригинального train (единственный источник с масками)
    all_imgs = sorted(
        list((src_train_dir / "images").glob("*.png")) +
        list((src_train_dir / "images").glob("*.jpg"))
    )
    print(f"Оригинальный train (с масками): {len(all_imgs)} изображений")
    print(f"Оригинальный test/ игнорируется — маски не опубликованы авторами датасета.")

    # Перемешиваем и делим 70/15/15
    rng = random.Random(RANDOM_SEED)
    imgs = all_imgs.copy()
    rng.shuffle(imgs)

    n_train = int(len(imgs) * TRAIN_RATIO)
    n_valid = int(len(imgs) * VALID_RATIO)
    # test — остаток, чтобы сумма была точно равна len(imgs)

    train_imgs = imgs[:n_train]
    valid_imgs = imgs[n_train:n_train + n_valid]
    test_imgs  = imgs[n_train + n_valid:]

    print(f"\nРазбивка 70/15/15 (seed={RANDOM_SEED}):")
    print(f"  train: {len(train_imgs)}")
    print(f"  valid: {len(valid_imgs)}")
    print(f"  test:  {len(test_imgs)}")

    # Создаём структуру папок (если OUTPUT_PATH существует — пересоздаём)
    if OUTPUT_PATH.exists():
        print(f"\n[INFO] Удаляю существующую папку: {OUTPUT_PATH}")
        shutil.rmtree(OUTPUT_PATH)

    for split in ("train", "valid", "test"):
        (OUTPUT_PATH / split / "images").mkdir(parents=True, exist_ok=True)
        (OUTPUT_PATH / split / "labels").mkdir(parents=True, exist_ok=True)

    # Все три сплита берут маски из src_train_dir
    print("\nОбрабатываю train...")
    s = process_split(src_train_dir, OUTPUT_PATH / "train", train_imgs)
    print(f"  ✓ {s['images']} изображений, {s['labels']} аннотаций, {s['boxes']} боксов")

    print("Обрабатываю valid...")
    s = process_split(src_train_dir, OUTPUT_PATH / "valid", valid_imgs)
    print(f"  ✓ {s['images']} изображений, {s['labels']} аннотаций, {s['boxes']} боксов")

    print("Обрабатываю test...")
    s = process_split(src_train_dir, OUTPUT_PATH / "test", test_imgs)
    print(f"  ✓ {s['images']} изображений, {s['labels']} аннотаций, {s['boxes']} боксов")

    # data.yaml с абсолютными путями
    yaml_content = {
        "train": "../train/images",
        "val":   "../valid/images",
        "test":  "../test/images",
        "nc":    1,
        "names": ["apple"],
    }
    yaml_path = OUTPUT_PATH / "data.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(yaml_content, f, default_flow_style=False, allow_unicode=True)

    print(f"\n✓ data.yaml: {yaml_path}")
    print(f"✓ Готово! Результат: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
