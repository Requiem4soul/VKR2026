"""
convert_figshare_brain_tumor.py — Конвертация Figshare Brain Tumor датасета

Источник датасета:
    Cheng J. et al. (2017). "Enhanced Performance of Brain Tumor Classification
    via Tumor Region Augmentation and Partition."
    figshare. DOI: 10.6084/m9.figshare.1512427

Входная структура (оригинальный Kaggle-датасет):
    <src>/
        Testing/
            glioma/       *.jpg
            meningioma/   *.jpg
            notumor/      *.jpg
            pituitary/    *.jpg
        Training/
            glioma/       *.jpg
            meningioma/   *.jpg
            notumor/      *.jpg
            pituitary/    *.jpg

Выходная структура:
    <dst>/
        train/
            glioma/
            meningioma/
            notumor/
            pituitary/
        valid/
            glioma/
            meningioma/
            notumor/
            pituitary/
        test/
            glioma/
            meningioma/
            notumor/
            pituitary/
        dataset_info.json

Примечание о разбивке:
    Оригинальный датасет имеет нестандартный сплит (Training ~400, Testing ~1400),
    что противоречит стандартной практике обучения. Поэтому все изображения
    из обеих папок объединяются и повторно делятся стратифицированно 70/15/15.

    Стратифицированная разбивка необходима из-за дисбаланса классов:
    глиома (~1426), гипофиз (~930), менингиома (~708), нет опухоли (~500).
    King & Zeng (2001) "Logistic Regression in Rare Events Data",
    Political Analysis, 9(2), 137-163.

Использование:
    python convert_figshare_brain_tumor.py --src "D:/Datasets/figshare_raw" --dst "D:/Datasets/0_FigshareBrainTumor"
    python convert_figshare_brain_tumor.py --src "D:/Datasets/figshare_raw" --dst "D:/Datasets/0_FigshareBrainTumor" --train 0.70 --valid 0.15 --seed 42
"""

import argparse
import json
import random
import shutil
from pathlib import Path

# ── Константы ──────────────────────────────────────────────────────────────

CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
SPLITS  = ["train", "valid", "test"]

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


# ── Вспомогательные функции ────────────────────────────────────────────────

def collect_images(class_dir: Path) -> list[Path]:
    """Собирает все изображения из папки (без рекурсии)."""
    images = []
    for p in class_dir.iterdir():
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS:
            images.append(p)
    return sorted(images)


def split_list(
    items: list[Path],
    train_ratio: float,
    valid_ratio: float,
    seed: int,
) -> tuple[list[Path], list[Path], list[Path]]:
    """
    Стратифицированная разбивка одного класса на train/valid/test.
    Seed фиксирует воспроизводимость (Dodge & Karam, 2017).
    """
    rng = random.Random(seed)
    shuffled = items[:]
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * train_ratio)
    n_valid = int(n * valid_ratio)

    train = shuffled[:n_train]
    valid = shuffled[n_train : n_train + n_valid]
    test  = shuffled[n_train + n_valid:]
    return train, valid, test


# ── Основная логика ────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Конвертация Figshare Brain Tumor датасета в train/valid/test структуру"
    )
    parser.add_argument("--src",   required=True,       help="Путь к исходному датасету (папка с Testing/ и Training/)")
    parser.add_argument("--dst",   required=True,       help="Путь для нового датасета")
    parser.add_argument("--train", type=float, default=0.70, help="Доля train (по умолч. 0.70)")
    parser.add_argument("--valid", type=float, default=0.15, help="Доля valid (по умолч. 0.15)")
    parser.add_argument("--seed",  type=int,   default=42,   help="Seed для воспроизводимости")
    args = parser.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)

    train_ratio = args.train
    valid_ratio = args.valid
    test_ratio  = round(1.0 - train_ratio - valid_ratio, 6)

    # ── Проверки ───────────────────────────────────────────────────────────
    if not src.exists():
        print(f"[ОШИБКА] Папка не найдена: {src}")
        return

    for folder in ("Testing", "Training"):
        if not (src / folder).exists():
            print(f"[ОШИБКА] Не найдена подпапка '{folder}' в {src}")
            print("  Убедитесь что --src указывает на папку, содержащую Testing/ и Training/")
            return

    if abs(train_ratio + valid_ratio + test_ratio - 1.0) > 0.001:
        print(f"[ОШИБКА] train + valid + test должно быть = 1.0")
        return

    print("=" * 60)
    print("Figshare Brain Tumor — конвертация датасета")
    print("=" * 60)
    print(f"Источник:   {src}")
    print(f"Назначение: {dst}")
    print(f"Разбивка:   train={train_ratio:.0%}  valid={valid_ratio:.0%}  test={test_ratio:.0%}")
    print(f"Seed:       {args.seed}")
    print()

    # ── Перезапись выходной папки ──────────────────────────────────────────
    if dst.exists():
        answer = input(f"Папка {dst} уже существует. Перезаписать? (y/n): ")
        if answer.lower() != "y":
            print("Отменено.")
            return
        shutil.rmtree(dst)

    # ── Создаём структуру ──────────────────────────────────────────────────
    for split in SPLITS:
        for cls in CLASSES:
            (dst / split / cls).mkdir(parents=True, exist_ok=True)

    # ── Обрабатываем каждый класс стратифицированно ────────────────────────
    total_stats = {s: 0 for s in SPLITS}
    class_stats = {}

    print("Разбивка по классам:")
    for cls in CLASSES:
        # Объединяем изображения из Training/ и Testing/
        all_images: list[Path] = []
        for folder in ("Training", "Testing"):
            cls_dir = src / folder / cls
            if cls_dir.exists():
                found = collect_images(cls_dir)
                all_images.extend(found)
            else:
                print(f"  [ПРЕДУПРЕЖДЕНИЕ] Не найдена папка: {cls_dir}")

        if not all_images:
            print(f"  [ПРОПУСК] {cls}: нет изображений")
            continue

        # Дедупликация по имени файла на случай пересечений между Training и Testing
        seen_names: set[str] = set()
        unique_images: list[Path] = []
        for img in all_images:
            if img.name not in seen_names:
                seen_names.add(img.name)
                unique_images.append(img)
            else:
                # Имя совпадает — добавляем с суффиксом папки
                new_name = f"{img.stem}_{img.parent.parent.name}{img.suffix}"
                if new_name not in seen_names:
                    seen_names.add(new_name)
                    unique_images.append(img)

        train_imgs, valid_imgs, test_imgs = split_list(
            unique_images, train_ratio, valid_ratio, args.seed
        )

        class_stats[cls] = {
            "total": len(unique_images),
            "train": len(train_imgs),
            "valid": len(valid_imgs),
            "test":  len(test_imgs),
        }

        # Копируем файлы
        for split, imgs in [("train", train_imgs), ("valid", valid_imgs), ("test", test_imgs)]:
            target_dir = dst / split / cls
            for img_path in imgs:
                target_name = img_path.name
                target_file = target_dir / target_name
                # Обрабатываем редкое совпадение имён из разных папок
                if target_file.exists():
                    target_name = f"{img_path.stem}_{img_path.parent.parent.name}{img_path.suffix}"
                    target_file = target_dir / target_name
                shutil.copy2(img_path, target_file)
            total_stats[split] += len(imgs)

        print(
            f"  {cls:12s}  всего={len(unique_images):5d}  "
            f"train={len(train_imgs):5d}  valid={len(valid_imgs):4d}  test={len(test_imgs):4d}"
        )

    # ── dataset_info.json ──────────────────────────────────────────────────
    # Нужен чтобы ClassificationTrainer и препроцессор корректно
    # определяли тип задачи и модальность (Litjens et al., 2017).
    dataset_info = {
        "task":        "multi-class",
        "num_classes": len(CLASSES),
        "num_channels": 3,
        "image_size":  224,
        "classes":     CLASSES,
        "source":      "Cheng et al. (2017), figshare, DOI: 10.6084/m9.figshare.1512427",
        "split_seed":  args.seed,
        "split_ratio": {
            "train": train_ratio,
            "valid": valid_ratio,
            "test":  test_ratio,
        },
    }
    with open(dst / "dataset_info.json", "w", encoding="utf-8") as f:
        json.dump(dataset_info, f, indent=2, ensure_ascii=False)

    # ── Итог ───────────────────────────────────────────────────────────────
    print()
    print(f"Итого:  train={total_stats['train']}  valid={total_stats['valid']}  test={total_stats['test']}")
    print(f"dataset_info.json записан.")
    print(f"\nГотово! Датасет сохранён в: {dst}")


if __name__ == "__main__":
    main()
