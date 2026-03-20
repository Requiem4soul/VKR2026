"""
apply_minneapple_baseline.py
============================
Применяет предобработку авторов статьи к датасету MinneApple.

  MinneApple_yolo  →  MinneApple_yolo_CLAHE_AB
  Метод: CLAHE на L-канале LAB + Alpha Blending 0.5 с оригиналом
  Источник: Wang et al. (2023), DOI: 10.3390/app131910760
            Section 3.2 «Optimal Image Data Generation»

Использование:
    python apply_minneapple_baseline.py

Зависимости: opencv-python, pyyaml, tqdm
"""

import shutil
import yaml
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

# ─── НАСТРОЙКИ ────────────────────────────────────────────────────────────────
DATASETS_ROOT  = Path(r"N:\ORIG_Datasets\CONVERTED_DATASETS")
MINNEAPPLE_SRC = DATASETS_ROOT / "0_MinneApple_yolo"
MINNEAPPLE_DST = DATASETS_ROOT / "1_MinneApple_yolo_CLAHE_AB"
IMAGE_EXTS     = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}
# ──────────────────────────────────────────────────────────────────────────────


def apply_clahe_alpha_blend(image: np.ndarray) -> np.ndarray:
    """
    CLAHE на L-канале LAB + Alpha Blending (ratio=0.5).
    Wang et al. (2023), Section 3.2 «Optimal Image Data Generation».

    Параметры:
    - Gaussian blur (3,3) — нормализация пикселей
    - CLAHE clipLimit=3.0 на L-канале LAB
    - Повышение яркости +5
    - Alpha blending 0.5/0.5 оригинала и обработанного
    """
    is_gray = (len(image.shape) == 2)
    if is_gray:
        bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        bgr = image.copy()

    original_bgr = bgr.copy()

    blurred = cv2.GaussianBlur(bgr, (3, 3), 0)

    lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l_ch)
    bgr_enhanced = cv2.cvtColor(
        cv2.merge([l_enhanced, a_ch, b_ch]), cv2.COLOR_LAB2BGR
    )

    bgr_brightened = np.clip(
        bgr_enhanced.astype(np.int32) + 5, 0, 255
    ).astype(np.uint8)

    result_bgr = cv2.addWeighted(original_bgr, 0.5, bgr_brightened, 0.5, 0)

    if is_gray:
        return cv2.cvtColor(result_bgr, cv2.COLOR_BGR2GRAY)
    return result_bgr


def process_yolo_dataset(src: Path, dst: Path, preprocess_fn):
    """YOLO формат: split/images + split/labels + data.yaml."""
    print(f"  Источник:   {src}")
    print(f"  Назначение: {dst}")

    if dst.exists():
        print(f"  [INFO] Удаляю существующую папку: {dst.name}")
        shutil.rmtree(dst)

    total = 0
    errors = 0

    for split_name in ("train", "valid", "test"):
        src_images = src / split_name / "images"
        src_labels = src / split_name / "labels"
        if not src_images.exists():
            continue

        dst_images = dst / split_name / "images"
        dst_labels = dst / split_name / "labels"
        dst_images.mkdir(parents=True, exist_ok=True)
        dst_labels.mkdir(parents=True, exist_ok=True)

        images = sorted([f for f in src_images.iterdir()
                         if f.suffix.lower() in IMAGE_EXTS])

        for img_path in tqdm(images, desc=f"  {split_name}", leave=False):
            img = cv2.imread(str(img_path))
            if img is None:
                errors += 1
                continue

            cv2.imwrite(str(dst_images / img_path.name), preprocess_fn(img))

            label_src = src_labels / (img_path.stem + ".txt")
            if label_src.exists():
                shutil.copy2(label_src, dst_labels / label_src.name)

            total += 1

    yaml_src = src / "data.yaml"
    if yaml_src.exists():
        with open(yaml_src, "r", encoding="utf-8") as f:
            yaml_data = yaml.safe_load(f)
        yaml_data["path"]  = str(dst.resolve())
        yaml_data["train"] = "train/images"
        yaml_data["val"]   = "valid/images"
        yaml_data["test"]  = "test/images"
        with open(dst / "data.yaml", "w", encoding="utf-8") as f:
            yaml.dump(yaml_data, f, default_flow_style=False, allow_unicode=True)
        print(f"  data.yaml обновлён")

    print(f"  ✓ Обработано: {total} | Ошибок: {errors}")
    return total


def main():
    print("=" * 60)
    print("MinneApple — CLAHE (LAB) + Alpha Blending")
    print("Wang et al. (2023), DOI: 10.3390/app131910760")
    print("=" * 60)

    if not MINNEAPPLE_SRC.exists():
        print(f"\n[ОШИБКА] Не найден исходный датасет: {MINNEAPPLE_SRC}")
        print("Сначала запусти convert_minneapple.py")
        return

    process_yolo_dataset(MINNEAPPLE_SRC, MINNEAPPLE_DST, apply_clahe_alpha_blend)

    print("\n✓ Готово!")
    print(f"  Оригинал:       {MINNEAPPLE_SRC.name}")
    print(f"  С предобработкой: {MINNEAPPLE_DST.name}")


if __name__ == "__main__":
    main()
