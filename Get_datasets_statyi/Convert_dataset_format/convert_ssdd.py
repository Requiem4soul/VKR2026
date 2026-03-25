"""
Конвертация SSDD (COCO format) → YOLO формат для использования с Ultralytics.

Структура входных данных:
    <ssdd_root>/
        coco_style/
            annotations/
                train.json
                test.json
            images/
                train/   *.jpg
                test/    *.jpg

Структура выходных данных (YOLO):
    <output_root>/
        train/
            images/  *.jpg
            labels/  *.txt
        valid/
            images/  *.jpg
            labels/  *.txt
        test/
            images/  *.jpg
            labels/  *.txt
        data.yaml

Train разбивается на train/valid (85%/15%, seed=42).
Test берётся целиком из test.json.

Использование:
    python convert_ssdd_to_yolo.py --ssdd_root "D:/SSDD" --output "N:/VKR_Datasets/0_SSDD_yolo"

Источник датасета:
    Zhang et al. (2021) "SAR Ship Detection Dataset (SSDD): Official Release
    and Comprehensive Data Analysis", Remote Sensing, 13(18), 3690.
    https://doi.org/10.3390/rs13183690
"""

import json
import shutil
import random
import argparse
from pathlib import Path


def coco_to_yolo_bbox(bbox, img_w, img_h):
    """
    Конвертирует COCO bbox [x, y, w, h] (абсолютные пиксели)
    в YOLO формат [cx, cy, w, h] (нормализованные 0-1).
    """
    x, y, w, h = bbox
    cx = (x + w / 2) / img_w
    cy = (y + h / 2) / img_h
    nw = w / img_w
    nh = h / img_h
    # Clamp на случай выхода за границы из-за погрешностей аннотаций
    cx = max(0.0, min(1.0, cx))
    cy = max(0.0, min(1.0, cy))
    nw = max(0.0, min(1.0, nw))
    nh = max(0.0, min(1.0, nh))
    return cx, cy, nw, nh


def convert_split(json_path, images_src_dir, out_images_dir, out_labels_dir):
    """
    Конвертирует один сплит (train или test).
    Возвращает количество изображений и аннотаций.
    """
    out_images_dir.mkdir(parents=True, exist_ok=True)
    out_labels_dir.mkdir(parents=True, exist_ok=True)

    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    # Строим индексы
    img_info = {img["id"]: img for img in data["images"]}

    # Группируем аннотации по image_id
    anns_by_image = {}
    for ann in data["annotations"]:
        iid = ann["image_id"]
        anns_by_image.setdefault(iid, []).append(ann)

    n_imgs = 0
    n_anns = 0

    for img_id, info in img_info.items():
        fname = info["file_name"]
        img_w = info["width"]
        img_h = info["height"]

        src = images_src_dir / fname
        if not src.exists():
            # Иногда file_name содержит подпапку, пробуем только имя файла
            src = images_src_dir / Path(fname).name
        if not src.exists():
            print(f"  [ПРОПУСК] изображение не найдено: {fname}")
            continue

        # Копируем изображение
        dst_img = out_images_dir / Path(fname).name
        shutil.copy2(src, dst_img)

        # Создаём label файл
        label_name = Path(fname).stem + ".txt"
        dst_label = out_labels_dir / label_name

        anns = anns_by_image.get(img_id, [])
        lines = []
        for ann in anns:
            if ann.get("ignore", 0):
                continue
            cx, cy, nw, nh = coco_to_yolo_bbox(ann["bbox"], img_w, img_h)
            # SSDD имеет один класс: ship → class_id = 0
            lines.append(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

        dst_label.write_text("\n".join(lines), encoding="utf-8")
        n_imgs += 1
        n_anns += len(lines)

    return n_imgs, n_anns


def main():
    parser = argparse.ArgumentParser(description="Convert SSDD COCO → YOLO")
    parser.add_argument("--ssdd_root", required=True,
                        help="Путь к папке SSDD (содержит coco_style/)")
    parser.add_argument("--output", required=True,
                        help="Путь к выходному датасету в YOLO формате")
    parser.add_argument("--val_ratio", type=float, default=0.15,
                        help="Доля валидации от train (default: 0.15)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    args = parser.parse_args()

    ssdd_root = Path(args.ssdd_root)
    out_root = Path(args.output)
    coco_dir = ssdd_root / "coco_style"

    ann_dir = coco_dir / "annotations"
    img_dir = coco_dir / "images"

    print(f"Источник: {coco_dir}")
    print(f"Назначение: {out_root}")
    print()

    # ── Конвертируем test целиком ──────────────────────────────────────────
    print("Конвертирую test...")
    n_test_imgs, n_test_anns = convert_split(
        json_path=ann_dir / "test.json",
        images_src_dir=img_dir / "test",
        out_images_dir=out_root / "test" / "images",
        out_labels_dir=out_root / "test" / "labels",
    )
    print(f"  test: {n_test_imgs} изображений, {n_test_anns} аннотаций")

    # ── Конвертируем train во временную папку, потом делим ─────────────────
    print("Конвертирую train...")
    tmp_dir = out_root / "_tmp_train"
    n_all_imgs, n_all_anns = convert_split(
        json_path=ann_dir / "train.json",
        images_src_dir=img_dir / "train",
        out_images_dir=tmp_dir / "images",
        out_labels_dir=tmp_dir / "labels",
    )
    print(f"  train (до разбивки): {n_all_imgs} изображений, {n_all_anns} аннотаций")

    # ── Разбиваем train → train/valid ──────────────────────────────────────
    all_stems = sorted([p.stem for p in (tmp_dir / "images").iterdir()])
    random.seed(args.seed)
    random.shuffle(all_stems)

    n_val = int(len(all_stems) * args.val_ratio)
    val_stems = set(all_stems[:n_val])
    train_stems = set(all_stems[n_val:])

    for split_name, stems in [("train", train_stems), ("valid", val_stems)]:
        (out_root / split_name / "images").mkdir(parents=True, exist_ok=True)
        (out_root / split_name / "labels").mkdir(parents=True, exist_ok=True)
        for stem in stems:
            # Ищем изображение (может быть jpg или jpeg)
            for ext in [".jpg", ".jpeg", ".png"]:
                src_img = tmp_dir / "images" / (stem + ext)
                if src_img.exists():
                    shutil.copy2(src_img, out_root / split_name / "images" / src_img.name)
                    break
            src_lbl = tmp_dir / "labels" / (stem + ".txt")
            if src_lbl.exists():
                shutil.copy2(src_lbl, out_root / split_name / "labels" / src_lbl.name)

    n_train = len(train_stems)
    n_val_actual = len(val_stems)
    print(f"  train: {n_train} изображений")
    print(f"  valid: {n_val_actual} изображений")

    # Удаляем временную папку
    shutil.rmtree(tmp_dir)

    # ── Пишем data.yaml ────────────────────────────────────────────────────
    yaml_content = f"""# SSDD — SAR Ship Detection Dataset (YOLO format)
# Zhang et al. (2021) Remote Sensing 13(18):3690
# https://doi.org/10.3390/rs13183690
#
# Конвертировано из COCO формата скриптом convert_ssdd_to_yolo.py
# train/valid split: {int((1 - args.val_ratio) * 100)}%/{int(args.val_ratio * 100)}%, seed={args.seed}

train: {(out_root / 'train' / 'images').as_posix()}
val:   {(out_root / 'valid' / 'images').as_posix()}
test:  {(out_root / 'test'  / 'images').as_posix()}

nc: 1
names: ['ship']
"""
    (out_root / "data.yaml").write_text(yaml_content, encoding="utf-8")

    print()
    print("=" * 60)
    print("Готово!")
    print(f"  train:  {n_train} изображений")
    print(f"  valid:  {n_val_actual} изображений")
    print(f"  test:   {n_test_imgs} изображений")
    print(f"  data.yaml: {out_root / 'data.yaml'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
