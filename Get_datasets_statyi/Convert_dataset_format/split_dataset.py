"""
split_dataset.py — Разбивка датасета классификации на train/valid/test

Использование:
    python split_dataset.py --src "N:/VKR_Datasets/0_COVID19_Pneumonia_clf" --dst "N:/VKR_Datasets/COVID19_Pneumonia_clf_split" --train 0.70 --valid 0.15 --test 0.15 --seed 42

Входная структура (папки по классам, без сплитов):
    src/
        covid/      *.jpg, *.png
        normal/     *.jpg, *.png
        pneumonia/  *.jpg, *.png

Выходная структура (ImageFolder, совместимая с ClassificationDataset):
    dst/
        train/
            covid/
            normal/
            pneumonia/
        valid/
            covid/
            normal/
            pneumonia/
        test/
            covid/
            normal/
            pneumonia/
        dataset_info.json   ← нужен чтобы препроцессор знал тип задачи
"""

import argparse
import json
import random
import shutil
from pathlib import Path


SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def collect_images(class_dir: Path) -> list[Path]:
    images = []
    for p in class_dir.iterdir():
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS:
            images.append(p)
    return images


def split_list(items: list, train: float, valid: float, seed: int):
    rng = random.Random(seed)
    shuffled = items[:]
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * train)
    n_valid = int(n * valid)
    return (
        shuffled[:n_train],
        shuffled[n_train:n_train + n_valid],
        shuffled[n_train + n_valid:],
    )


def main():
    parser = argparse.ArgumentParser(description="Разбивка датасета классификации")
    parser.add_argument("--src",   required=True, help="Путь к исходному датасету")
    parser.add_argument("--dst",   required=True, help="Путь для нового датасета")
    parser.add_argument("--train", type=float, default=0.70, help="Доля train (по умолч. 0.70)")
    parser.add_argument("--valid", type=float, default=0.15, help="Доля valid (по умолч. 0.15)")
    parser.add_argument("--test",  type=float, default=0.15, help="Доля test  (по умолч. 0.15)")
    parser.add_argument("--seed",  type=int,   default=42,   help="Seed для воспроизводимости")
    args = parser.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)

    # Проверки
    if not src.exists():
        print(f"[ОШИБКА] Папка не найдена: {src}")
        return

    total = round(args.train + args.valid + args.test, 6)
    if abs(total - 1.0) > 0.001:
        print(f"[ОШИБКА] train + valid + test должно быть = 1.0, получилось {total}")
        return

    # Находим классы — подпапки исходного датасета
    class_dirs = sorted([d for d in src.iterdir() if d.is_dir()])
    if not class_dirs:
        print(f"[ОШИБКА] В {src} не найдено подпапок с классами")
        return

    print(f"Источник:    {src}")
    print(f"Назначение:  {dst}")
    print(f"Классы:      {[d.name for d in class_dirs]}")
    print(f"Разбивка:    train={args.train:.0%}  valid={args.valid:.0%}  test={args.test:.0%}")
    print(f"Seed:        {args.seed}")
    print()

    if dst.exists():
        answer = input(f"Папка {dst} уже существует. Перезаписать? (y/n): ")
        if answer.lower() != "y":
            print("Отменено.")
            return
        shutil.rmtree(dst)

    # Создаём структуру папок
    for split in ("train", "valid", "test"):
        for cls_dir in class_dirs:
            (dst / split / cls_dir.name).mkdir(parents=True, exist_ok=True)

    # Копируем файлы
    total_stats = {"train": 0, "valid": 0, "test": 0}
    class_stats = {}

    for cls_dir in class_dirs:
        images = collect_images(cls_dir)
        if not images:
            print(f"  [ПРОПУСК] {cls_dir.name}: нет изображений")
            continue

        train_imgs, valid_imgs, test_imgs = split_list(
            images, args.train, args.valid, args.seed
        )

        class_stats[cls_dir.name] = {
            "total": len(images),
            "train": len(train_imgs),
            "valid": len(valid_imgs),
            "test":  len(test_imgs),
        }

        for split, imgs in [("train", train_imgs), ("valid", valid_imgs), ("test", test_imgs)]:
            target_dir = dst / split / cls_dir.name
            for img in imgs:
                shutil.copy2(img, target_dir / img.name)
            total_stats[split] += len(imgs)

        print(f"  {cls_dir.name:15s}  всего={len(images):5d}  "
              f"train={len(train_imgs):5d}  valid={len(valid_imgs):4d}  test={len(test_imgs):4d}")

    # Записываем dataset_info.json
    # Это критично — без него DatasetPreprocessor определяет тип как "yolo"
    dataset_info = {
        "task":         "multi-class",
        "num_classes":  len(class_dirs),
        "num_channels": 3,
        "image_size":   224,
        "classes":      [d.name for d in class_dirs],
        "split_seed":   args.seed,
        "split_ratio":  {
            "train": args.train,
            "valid": args.valid,
            "test":  args.test,
        },
    }
    with open(dst / "dataset_info.json", "w", encoding="utf-8") as f:
        json.dump(dataset_info, f, indent=2, ensure_ascii=False)

    print()
    print(f"Итого:  train={total_stats['train']}  valid={total_stats['valid']}  test={total_stats['test']}")
    print(f"dataset_info.json записан: {dst / 'dataset_info.json'}")
    print(f"\nГотово! Датасет сохранён в: {dst}")


if __name__ == "__main__":
    main()
