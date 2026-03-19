"""
convert_wgisd.py
================
Конвертация датасета Embrapa WGISD → YOLO detection format.

Исходная структура:
    thsant-wgisd-ab223e5/
    ├── data/
    │   ├── CDY_2015.jpg       ← изображение
    │   ├── CDY_2015.txt       ← аннотация (уже YOLO-формат, class=0 для всех)
    │   ├── CFR_2015.jpg
    │   ├── CFR_2015.txt
    │   └── ...
    ├── classes.txt            ← 5 сортов: CDY CFR CSV SVB VVS
    ├── train.txt              ← список файлов для train (пути или стемы)
    ├── test.txt               ← список файлов для test
    └── ...

Формат аннотации (уже YOLO):
    0 0.512 0.433 0.234 0.187
    0 0.721 0.612 0.198 0.145

Важно: в оригинале class=0 для всего. Мы сохраняем как есть (один класс grape).

Целевая структура:
    WGISD_yolo/
    ├── train/
    │   ├── images/  *.jpg
    │   └── labels/  *.txt
    ├── valid/
    │   ├── images/
    │   └── labels/
    ├── test/
    │   ├── images/
    │   └── labels/
    └── data.yaml

Разбивка: train.txt → train (80%) + valid (20%), test.txt → test
Seed=42 для воспроизводимости разбивки train→train/valid.
"""

import shutil
import random
import yaml
from pathlib import Path

# ─── НАСТРОЙКИ ────────────────────────────────────────────────────────────────
SOURCE_PATH = Path(r"N:\ORIG_Datasets\thsant-wgisd-ab223e5")  # ← исходный датасет
OUTPUT_PATH = Path(r"N:\ORIG_Datasets\CONVERTED_DATASETS\WGISD_yolo")            # ← результат

# Доля валидации из train-сплита оригинала
VALID_FROM_TRAIN_RATIO = 0.20
RANDOM_SEED = 42
# ──────────────────────────────────────────────────────────────────────────────


def read_split_file(txt_path: Path, data_dir: Path) -> list[Path]:
    """
    Читает train.txt / test.txt и возвращает список Path к изображениям.
    Формат строк может быть:
        data/CDY_2015.jpg
        CDY_2015
        CDY_2015.jpg
    """
    images = []
    with open(txt_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            p = Path(line)
            # Убираем расширение если есть, потом ищем jpg/png
            stem = p.stem

            # Ищем в data/
            for ext in (".jpg", ".jpeg", ".png"):
                candidate = data_dir / (stem + ext)
                if candidate.exists():
                    images.append(candidate)
                    break
            else:
                # Попробуем как абсолютный или относительный путь
                candidate = SOURCE_PATH / p
                if candidate.exists():
                    images.append(candidate)
                else:
                    print(f"  [WARNING] Файл не найден для строки: '{line}'")

    return images


def copy_image_and_label(img_path: Path, split_dir: Path):
    """Копирует изображение и соответствующую аннотацию в нужную папку."""
    dst_img = split_dir / "images" / img_path.name
    shutil.copy2(img_path, dst_img)

    label_src = img_path.with_suffix(".txt")
    if label_src.exists():
        dst_label = split_dir / "labels" / (img_path.stem + ".txt")
        shutil.copy2(label_src, dst_label)
    else:
        print(f"  [WARNING] Аннотация не найдена: {label_src.name}")


def main():
    data_dir = SOURCE_PATH / "data"
    train_txt = SOURCE_PATH / "train.txt"
    test_txt  = SOURCE_PATH / "test.txt"

    for p in [data_dir, train_txt, test_txt]:
        if not p.exists():
            raise FileNotFoundError(f"Не найден: {p}")

    # Читаем классы
    classes_file = SOURCE_PATH / "classes.txt"
    if classes_file.exists():
        class_names = [l.strip() for l in classes_file.read_text().splitlines() if l.strip()]
    else:
        class_names = ["CDY", "CFR", "CSV", "SVB", "VVS"]
    print(f"Классы: {class_names}")

    # Читаем списки файлов
    train_all = read_split_file(train_txt, data_dir)
    test_imgs  = read_split_file(test_txt,  data_dir)
    print(f"\nОригинальный train: {len(train_all)} изображений")
    print(f"Оригинальный test:  {len(test_imgs)} изображений")

    # Делим train → train + valid
    rng = random.Random(RANDOM_SEED)
    train_all_copy = train_all.copy()
    rng.shuffle(train_all_copy)
    n_valid = int(len(train_all_copy) * VALID_FROM_TRAIN_RATIO)
    valid_imgs = train_all_copy[:n_valid]
    train_imgs = train_all_copy[n_valid:]

    print(f"\nПосле разбивки:")
    print(f"  train: {len(train_imgs)}")
    print(f"  valid: {len(valid_imgs)}")
    print(f"  test:  {len(test_imgs)}")

    # Создаём структуру папок
    for split in ("train", "valid", "test"):
        (OUTPUT_PATH / split / "images").mkdir(parents=True, exist_ok=True)
        (OUTPUT_PATH / split / "labels").mkdir(parents=True, exist_ok=True)

    # Копируем файлы
    for split_name, imgs in [("train", train_imgs), ("valid", valid_imgs), ("test", test_imgs)]:
        print(f"\nКопирую {split_name}...")
        for img in imgs:
            copy_image_and_label(img, OUTPUT_PATH / split_name)
        print(f"  ✓ {len(imgs)} изображений")

    # Создаём data.yaml
    # Пути абсолютные — YOLO их понимает
    yaml_content = {
        "path": str(OUTPUT_PATH.resolve()),
        "train": "train/images",
        "val":   "valid/images",
        "test":  "test/images",
        "nc":    len(class_names),
        "names": class_names,
    }

    yaml_path = OUTPUT_PATH / "data.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(yaml_content, f, default_flow_style=False, allow_unicode=True)

    print(f"\n✓ data.yaml сохранён: {yaml_path}")
    print(f"\n✓ Готово! Результат: {OUTPUT_PATH}")

    # Статистика аннотаций
    print("\nПроверка аннотаций:")
    for split in ("train", "valid", "test"):
        n_imgs   = len(list((OUTPUT_PATH / split / "images").glob("*")))
        n_labels = len(list((OUTPUT_PATH / split / "labels").glob("*.txt")))
        print(f"  {split}: {n_imgs} изображений, {n_labels} аннотаций")


if __name__ == "__main__":
    main()
