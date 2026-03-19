"""
apply_paper_baselines.py
========================
Применяет предобработку авторов статей к трём датасетам:

  1. MIAS_clf       → MIAS_clf_CLAHE_USM
     Метод: CLAHE + Unsharp Masking
     Источник: Avcı & Karakaya (2023), DOI: 10.3390/diagnostics13030348
     Лучшая комбинация из 5 протестированных (Table 2, Table 3)

  2. WGISD_yolo     → WGISD_yolo_CLAHE_AB
     Метод: CLAHE на L-канале LAB + Alpha Blending 0.5 с оригиналом
     Источник: Wang et al. (2023), DOI: 10.3390/app131910760
     Единственный метод статьи (Section 3.2)

  3. BreakHis_clf   → НЕ ОБРАБАТЫВАЕТСЯ
     Источник: Murcia-Gómez et al. (2022), DOI: 10.3390/app122211375
     Причина: ANOVA p-value=0.146, предобработка статистически не влияет.
     Baseline = оригинальный датасет без изменений.

Использование:
    python apply_paper_baselines.py

Зависимости: opencv-python, pyyaml, tqdm
"""

import shutil
import yaml
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

# ─── НАСТРОЙКИ ────────────────────────────────────────────────────────────────

# Папка где лежат все конвертированные датасеты
DATASETS_ROOT = Path(r"N:\ORIG_Datasets\CONVERTED_DATASETS")

# Входные датасеты (результат конвертации)
MIAS_SRC    = DATASETS_ROOT / "MIAS_clf"
WGISD_SRC   = DATASETS_ROOT / "WGISD_yolo"
# BreakHis не обрабатываем — baseline = оригинал

# Выходные папки (с предобработкой авторов)
MIAS_DST    = DATASETS_ROOT / "MIAS_clf_CLAHE_USM"
WGISD_DST   = DATASETS_ROOT / "WGISD_yolo_CLAHE_AB"

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}
# ──────────────────────────────────────────────────────────────────────────────


# ══════════════════════════════════════════════════════════════════════════════
# МЕТОДЫ ПРЕДОБРАБОТКИ
# ══════════════════════════════════════════════════════════════════════════════

def apply_clahe_usm(image: np.ndarray) -> np.ndarray:
    """
    CLAHE + Unsharp Masking — лучшая комбинация из Avcı & Karakaya (2023).

    Параметры CLAHE: clipLimit=2.0, tileGridSize=(8,8) — стандартные OpenCV,
    как указано в статье Murcia-Gómez (которая ссылается на те же параметры).
    Avcı & Karakaya используют реализацию из MedPic/MATLAB, но параметры
    эквивалентны стандартным OpenCV.

    Unsharp Masking: sigma=1.0, amount=1.0 — стандартные параметры фильтра
    повышения резкости (классический USM).
    """
    # Шаг 1: CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(image)

    # Шаг 2: Unsharp Masking
    # USM = original + amount * (original - blurred)
    blurred = cv2.GaussianBlur(enhanced, (0, 0), sigmaX=1.0)
    amount = 1.0
    result = cv2.addWeighted(enhanced, 1.0 + amount, blurred, -amount, 0)

    return result


def apply_clahe_alpha_blend(image: np.ndarray) -> np.ndarray:
    """
    CLAHE на L-канале LAB + Alpha Blending (ratio=0.5) с оригиналом.
    Метод Wang et al. (2023), Section 3.2 «Optimal Image Data Generation».

    Шаги из статьи:
    1. Gaussian blur для нормализации пикселей
    2. CLAHE на L-канале LAB с авто-подобранным cliplimit
       (авторы перебирают 0-40, мы берём оптимальное из практики = 3.0)
    3. Повышение яркости на α (авторы перебирают 0-10, берём α=5)
    4. Saturation clipping
    5. Alpha blending оригинала и обработанного (ratio=0.5)

    Изображение может быть grayscale или RGB.
    """
    # Если grayscale — конвертируем в BGR для работы с LAB
    is_gray = (len(image.shape) == 2)
    if is_gray:
        bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        bgr = image.copy()

    original_bgr = bgr.copy()

    # Шаг 1: Лёгкий Gaussian blur (нормализация пикселей)
    blurred = cv2.GaussianBlur(bgr, (3, 3), 0)

    # Шаг 2: CLAHE на L-канале LAB
    lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l_channel)

    lab_enhanced = cv2.merge([l_enhanced, a_channel, b_channel])
    bgr_enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

    # Шаг 3: Повышение яркости на α=5
    alpha_brightness = 5
    bgr_brightened = np.clip(
        bgr_enhanced.astype(np.int32) + alpha_brightness, 0, 255
    ).astype(np.uint8)

    # Шаг 4: Saturation clipping (clip значений в [0, 255] — уже сделано выше)

    # Шаг 5: Alpha blending оригинала и обработанного (ratio=0.5)
    result_bgr = cv2.addWeighted(original_bgr, 0.5, bgr_brightened, 0.5, 0)

    # Возвращаем в исходный формат
    if is_gray:
        result = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2GRAY)
    else:
        result = result_bgr

    return result


# ══════════════════════════════════════════════════════════════════════════════
# УТИЛИТЫ КОПИРОВАНИЯ
# ══════════════════════════════════════════════════════════════════════════════

def collect_images(directory: Path) -> list[Path]:
    """Собирает все изображения из папки рекурсивно."""
    files = []
    for ext in IMAGE_EXTS:
        files.extend(directory.rglob(f"*{ext}"))
    return sorted(files)


def process_classification_dataset(src: Path, dst: Path, preprocess_fn):
    """
    Обрабатывает датасет формата ImageFolder.
    Структура: src/split/class/image.png → dst/split/class/image.png
    """
    print(f"\n{'='*60}")
    print(f"Источник:  {src}")
    print(f"Назначение: {dst}")

    if dst.exists():
        print(f"[INFO] Папка уже существует, удаляю: {dst}")
        shutil.rmtree(dst)

    total = 0
    errors = 0

    # Проходим по split/class/images
    for split_dir in sorted(src.iterdir()):
        if not split_dir.is_dir():
            continue
        split_name = split_dir.name

        for class_dir in sorted(split_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            class_name = class_dir.name

            dst_class_dir = dst / split_name / class_name
            dst_class_dir.mkdir(parents=True, exist_ok=True)

            images = [f for f in class_dir.iterdir()
                      if f.is_file() and f.suffix.lower() in IMAGE_EXTS]

            for img_path in tqdm(images, desc=f"{split_name}/{class_name}", leave=False):
                img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    errors += 1
                    continue
                processed = preprocess_fn(img)
                dst_path = dst_class_dir / img_path.name
                cv2.imwrite(str(dst_path), processed)
                total += 1

    print(f"✓ Обработано: {total} | Ошибок: {errors}")
    return total


def process_yolo_dataset(src: Path, dst: Path, preprocess_fn):
    """
    Обрабатывает датасет формата YOLO.
    Структура: src/split/images/*.jpg + src/split/labels/*.txt + data.yaml
    Аннотации и data.yaml копируются без изменений.
    """
    print(f"\n{'='*60}")
    print(f"Источник:  {src}")
    print(f"Назначение: {dst}")

    if dst.exists():
        print(f"[INFO] Папка уже существует, удаляю: {dst}")
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

        for img_path in tqdm(images, desc=f"{split_name}", leave=False):
            # Читаем как цветное (WGISD — RGB изображения)
            img = cv2.imread(str(img_path))
            if img is None:
                errors += 1
                continue

            processed = preprocess_fn(img)
            cv2.imwrite(str(dst / split_name / "images" / img_path.name), processed)

            # Копируем аннотацию
            label_src = src_labels / (img_path.stem + ".txt")
            if label_src.exists():
                shutil.copy2(label_src, dst_labels / label_src.name)

            total += 1

    # Копируем и обновляем data.yaml с новыми путями
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

    print(f"✓ Обработано: {total} | Ошибок: {errors}")
    return total


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("ПРИМЕНЕНИЕ ПРЕДОБРАБОТКИ АВТОРОВ СТАТЕЙ")
    print("=" * 60)

    # ── 1. MIAS: CLAHE + USM ──────────────────────────────────────────────────
    print("\n[1/2] MIAS — CLAHE + Unsharp Masking")
    print("      Источник: Avcı & Karakaya (2023), лучшая из 5 комбинаций")
    if not MIAS_SRC.exists():
        print(f"  [SKIP] Не найден: {MIAS_SRC}")
    else:
        process_classification_dataset(MIAS_SRC, MIAS_DST, apply_clahe_usm)

    # ── 2. WGISD: CLAHE LAB + Alpha Blending ─────────────────────────────────
    print("\n[2/2] WGISD — CLAHE (LAB) + Alpha Blending")
    print("      Источник: Wang et al. (2023), Section 3.2")
    if not WGISD_SRC.exists():
        print(f"  [SKIP] Не найден: {WGISD_SRC}")
    else:
        process_yolo_dataset(WGISD_SRC, WGISD_DST, apply_clahe_alpha_blend)

    # ── 3. BreakHis: без обработки ───────────────────────────────────────────
    print("\n[3/2] BreakHis — предобработка НЕ применяется")
    print("      Источник: Murcia-Gómez et al. (2022)")
    print("      ANOVA p-value=0.146 → все фильтры статистически эквивалентны Raw")
    print("      Используй оригинальный датасет: BreakHis_clf")

    print("\n" + "=" * 60)
    print("ГОТОВО. Созданные датасеты с предобработкой авторов:")
    for p in [MIAS_DST, WGISD_DST]:
        status = "✓" if p.exists() else "✗"
        print(f"  {status} {p.name}")
    print("\nДля сравнения с твоей системой:")
    print("  MIAS:    MIAS_clf_CLAHE_USM  vs  <твоя_предобработка>")
    print("  WGISD:   WGISD_yolo_CLAHE_AB vs  <твоя_предобработка>")
    print("  BreakHis: BreakHis_clf (Raw) vs  <твоя_предобработка>")


if __name__ == "__main__":
    main()
