"""
Разбивка YOLO-датасета на train/valid/test (70% / 15% / 15%)
с фиксированным seed=42.

Ожидаемая структура входной папки:
    dataset_root/
        images/          <- все изображения (.jpg, .jpeg, .png)
        labels/          <- все аннотации (.txt, совпадают по имени с изображениями)
        data.yaml        <- конфиг датасета

Результат:
    dataset_root/
        train/
            images/
            labels/
        valid/
            images/
            labels/
        test/
            images/
            labels/
        data.yaml        <- обновлённый конфиг

Использование:
    python split_yolo_dataset.py --dataset_path /path/to/dataset
    python split_yolo_dataset.py --dataset_path /path/to/dataset --train 0.7 --valid 0.15 --test 0.15 --seed 42
"""

import argparse
import os
import random
import shutil
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def split_dataset(
    dataset_path: str,
    train_ratio: float = 0.70,
    valid_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> None:
    assert abs(train_ratio + valid_ratio + test_ratio - 1.0) < 1e-6, \
        "Сумма долей должна быть равна 1.0"

    root = Path(dataset_path)
    images_dir = root / "images"
    labels_dir = root / "labels"

    assert images_dir.exists(), f"Папка images не найдена: {images_dir}"
    assert labels_dir.exists(), f"Папка labels не найдена: {labels_dir}"

    # Собираем все изображения
    image_files = sorted([
        f for f in images_dir.iterdir()
        if f.suffix.lower() in IMAGE_EXTENSIONS
    ])
    print(f"Найдено изображений: {len(image_files)}")

    # Проверяем наличие аннотаций
    missing_labels = []
    for img in image_files:
        label = labels_dir / (img.stem + ".txt")
        if not label.exists():
            missing_labels.append(img.name)
    if missing_labels:
        print(f"⚠ Нет аннотаций для {len(missing_labels)} изображений: {missing_labels[:5]}...")

    # Перемешиваем с фиксированным seed
    random.seed(seed)
    shuffled = image_files.copy()
    random.shuffle(shuffled)

    # Вычисляем границы разбиения
    n = len(shuffled)
    n_train = int(n * train_ratio)
    n_valid = int(n * valid_ratio)

    splits = {
        "train": shuffled[:n_train],
        "valid": shuffled[n_train:n_train + n_valid],
        "test":  shuffled[n_train + n_valid:],
    }

    for split_name, files in splits.items():
        print(f"  {split_name}: {len(files)} изображений")

    # Создаём папки и копируем файлы
    for split_name, files in splits.items():
        split_img_dir = root / split_name / "images"
        split_lbl_dir = root / split_name / "labels"
        split_img_dir.mkdir(parents=True, exist_ok=True)
        split_lbl_dir.mkdir(parents=True, exist_ok=True)

        for img_path in files:
            # Копируем изображение
            shutil.copy2(img_path, split_img_dir / img_path.name)

            # Копируем аннотацию если есть
            label_path = labels_dir / (img_path.stem + ".txt")
            if label_path.exists():
                shutil.copy2(label_path, split_lbl_dir / label_path.name)

    # Обновляем data.yaml
    yaml_path = root / "data.yaml"
    if yaml_path.exists():
        with open(yaml_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Обновляем пути train/val/test
        lines = content.splitlines()
        new_lines = []
        updated = {"train": False, "val": False, "test": False}

        for line in lines:
            stripped = line.strip()
            if stripped.startswith("train:") and not updated["train"]:
                new_lines.append(f"train: {root / 'train' / 'images'}")
                updated["train"] = True
            elif stripped.startswith("val:") and not updated["val"]:
                new_lines.append(f"val: {root / 'valid' / 'images'}")
                updated["val"] = True
            elif stripped.startswith("test:") and not updated["test"]:
                new_lines.append(f"test: {root / 'test' / 'images'}")
                updated["test"] = True
            else:
                new_lines.append(line)

        # Если строк не было — добавляем в начало
        header = []
        if not updated["train"]:
            header.append(f"train: {root / 'train' / 'images'}")
        if not updated["val"]:
            header.append(f"val: {root / 'valid' / 'images'}")
        if not updated["test"]:
            header.append(f"test: {root / 'test' / 'images'}")

        final_content = "\n".join(header + new_lines) if header else "\n".join(new_lines)

        with open(yaml_path, "w", encoding="utf-8") as f:
            f.write(final_content)

        print(f"\ndata.yaml обновлён: {yaml_path}")
    else:
        print("\n⚠ data.yaml не найден — создайте его вручную")

    print("\n✅ Разбиение завершено:")
    for split_name in ("train", "valid", "test"):
        n_imgs = len(list((root / split_name / "images").iterdir()))
        print(f"   {split_name}/images: {n_imgs} файлов")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Разбивка YOLO-датасета на train/valid/test")
    parser.add_argument("--dataset_path", type=str, required=True,
                        help="Путь к корневой папке датасета")
    parser.add_argument("--train", type=float, default=0.70, help="Доля train (по умолч. 0.70)")
    parser.add_argument("--valid", type=float, default=0.15, help="Доля valid (по умолч. 0.15)")
    parser.add_argument("--test",  type=float, default=0.15, help="Доля test  (по умолч. 0.15)")
    parser.add_argument("--seed",  type=int,   default=42,   help="Seed (по умолч. 42)")
    args = parser.parse_args()

    split_dataset(
        dataset_path=args.dataset_path,
        train_ratio=args.train,
        valid_ratio=args.valid,
        test_ratio=args.test,
        seed=args.seed,
    )
