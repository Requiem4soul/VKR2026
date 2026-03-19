"""
convert_minneapple.py
=====================
Конвертация датасета MinneApple → YOLO detection format.

Исходная структура (после распаковки detection.tar.gz):
    detection/
    ├── train/
    │   ├── images/
    │   │   ├── 20150921_132032_966_image.png
    │   │   └── ...
    │   └── masks/
    │       ├── 20150921_132032_966_image.png  ← бинарные маски (один файл = один объект?)
    │       └── ...
    └── test/
        ├── images/
        └── masks/

Формат масок MinneApple:
    Маски хранятся как PNG-изображения с полигональными аннотациями.
    Каждая уникальная ненулевая интенсивность (или связная область) = отдельный объект.
    Мы конвертируем каждую связную область в bounding box.

Целевая структура:
    MinneApple_yolo/
    ├── train/
    │   ├── images/  *.png
    │   └── labels/  *.txt
    ├── valid/
    │   ├── images/
    │   └── labels/
    ├── test/
    │   ├── images/
    │   └── labels/
    └── data.yaml

Разбивка: оригинальный train → train (85%) + valid (15%), test → test
Один класс: apple (0)
Seed=42
"""

import shutil
import random
import yaml
import numpy as np
from pathlib import Path

# ─── НАСТРОЙКИ ────────────────────────────────────────────────────────────────
SOURCE_PATH = Path(r"C:\datasets\detection")     # ← папка detection/ после распаковки
OUTPUT_PATH = Path(r"C:\datasets\MinneApple_yolo")  # ← результат

VALID_FROM_TRAIN_RATIO = 0.15
RANDOM_SEED = 42
MIN_BOX_AREA = 100   # минимальная площадь bbox в пикселях (фильтр шума)
# ──────────────────────────────────────────────────────────────────────────────

try:
    import cv2
except ImportError:
    raise ImportError("Установите OpenCV: pip install opencv-python")


def mask_to_yolo_bboxes(mask_path: Path, img_width: int, img_height: int) -> list[str]:
    """
    Конвертирует маску → список строк YOLO-формата.

    Алгоритм:
    1. Читаем маску как grayscale
    2. Ищем все связные компоненты (contours или connectedComponents)
    3. Для каждой компоненты вычисляем bounding box
    4. Нормализуем в диапазон [0, 1]
    5. Возвращаем строки "0 cx cy w h"
    """
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return []

    lines = []

    # Метод 1: через connectedComponents (лучше для наложенных масок)
    # Бинаризуем
    _, binary = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)

    # Находим связные компоненты
    num_labels, labels_img = cv2.connectedComponents(binary)

    for label_id in range(1, num_labels):  # 0 — фон
        # Маска для конкретного объекта
        component_mask = (labels_img == label_id).astype(np.uint8)
        ys, xs = np.where(component_mask > 0)

        if len(xs) == 0:
            continue

        x_min, x_max = int(xs.min()), int(xs.max())
        y_min, y_max = int(ys.min()), int(ys.max())

        box_w = x_max - x_min + 1
        box_h = y_max - y_min + 1

        # Фильтр слишком маленьких боксов (шум)
        if box_w * box_h < MIN_BOX_AREA:
            continue

        # YOLO нормализованные координаты
        cx = (x_min + x_max) / 2.0 / img_width
        cy = (y_min + y_max) / 2.0 / img_height
        nw = box_w / img_width
        nh = box_h / img_height

        # Клиппинг на случай выхода за границы
        cx = max(0.0, min(1.0, cx))
        cy = max(0.0, min(1.0, cy))
        nw = max(0.0, min(1.0, nw))
        nh = max(0.0, min(1.0, nh))

        lines.append(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

    return lines


def process_split(src_split_dir: Path, dst_split_dir: Path, img_list: list[Path]) -> dict:
    """Обрабатывает один сплит: конвертирует маски и копирует изображения."""
    stats = {"images": 0, "labels": 0, "boxes": 0, "no_mask": 0}

    for img_path in img_list:
        # Ищем соответствующую маску
        mask_path = src_split_dir / "masks" / img_path.name

        # Определяем размер изображения
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  [WARNING] Не удалось прочитать изображение: {img_path.name}")
            continue

        h, w = img.shape[:2]

        # Копируем изображение
        dst_img = dst_split_dir / "images" / img_path.name
        shutil.copy2(img_path, dst_img)
        stats["images"] += 1

        # Конвертируем маску
        if mask_path.exists():
            yolo_lines = mask_to_yolo_bboxes(mask_path, w, h)
            if yolo_lines:
                label_path = dst_split_dir / "labels" / (img_path.stem + ".txt")
                label_path.write_text("\n".join(yolo_lines) + "\n")
                stats["labels"] += 1
                stats["boxes"]  += len(yolo_lines)
            else:
                # Пустой label файл (изображение без аннотаций)
                label_path = dst_split_dir / "labels" / (img_path.stem + ".txt")
                label_path.write_text("")
        else:
            # Нет маски — создаём пустой label
            label_path = dst_split_dir / "labels" / (img_path.stem + ".txt")
            label_path.write_text("")
            stats["no_mask"] += 1

    return stats


def main():
    src_train_dir = SOURCE_PATH / "train"
    src_test_dir  = SOURCE_PATH / "test"

    for d in [src_train_dir, src_test_dir]:
        if not d.exists():
            raise FileNotFoundError(f"Не найдена папка: {d}")

    # Собираем все изображения из оригинального train
    train_all = sorted(list((src_train_dir / "images").glob("*.png")) +
                       list((src_train_dir / "images").glob("*.jpg")))
    test_imgs  = sorted(list((src_test_dir  / "images").glob("*.png")) +
                        list((src_test_dir  / "images").glob("*.jpg")))

    print(f"Оригинальный train: {len(train_all)} изображений")
    print(f"Оригинальный test:  {len(test_imgs)} изображений")

    # Делим train → train + valid
    rng = random.Random(RANDOM_SEED)
    train_copy = train_all.copy()
    rng.shuffle(train_copy)
    n_valid = int(len(train_copy) * VALID_FROM_TRAIN_RATIO)
    valid_imgs = train_copy[:n_valid]
    train_imgs = train_copy[n_valid:]

    print(f"\nПосле разбивки:")
    print(f"  train: {len(train_imgs)}")
    print(f"  valid: {len(valid_imgs)}")
    print(f"  test:  {len(test_imgs)}")

    # Создаём структуру папок
    for split in ("train", "valid", "test"):
        (OUTPUT_PATH / split / "images").mkdir(parents=True, exist_ok=True)
        (OUTPUT_PATH / split / "labels").mkdir(parents=True, exist_ok=True)

    # Обрабатываем каждый сплит
    # valid берёт маски из src_train_dir (т.к. исходно это train-изображения)
    print("\nОбрабатываю train...")
    s = process_split(src_train_dir, OUTPUT_PATH / "train", train_imgs)
    print(f"  ✓ {s['images']} изображений, {s['labels']} аннотаций, {s['boxes']} боксов")

    print("Обрабатываю valid...")
    s = process_split(src_train_dir, OUTPUT_PATH / "valid", valid_imgs)
    print(f"  ✓ {s['images']} изображений, {s['labels']} аннотаций, {s['boxes']} боксов")

    print("Обрабатываю test...")
    s = process_split(src_test_dir,  OUTPUT_PATH / "test",  test_imgs)
    print(f"  ✓ {s['images']} изображений, {s['labels']} аннотаций, {s['boxes']} боксов")

    # Создаём data.yaml
    yaml_content = {
        "path": str(OUTPUT_PATH.resolve()),
        "train": "train/images",
        "val":   "valid/images",
        "test":  "test/images",
        "nc":    1,
        "names": ["apple"],
    }

    yaml_path = OUTPUT_PATH / "data.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(yaml_content, f, default_flow_style=False, allow_unicode=True)

    print(f"\n✓ data.yaml: {yaml_path}")
    print(f"✓ Готово! Результат: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
