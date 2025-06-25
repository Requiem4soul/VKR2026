import shutil
from pathlib import Path
from tqdm import tqdm
import cv2
import numpy as np
from config import DATASETS_GLOBAL_PATH
from Data.Datasets.dataset_work import get_dataset_path

SOURCE_NAME = "SAR_low"
TARGET_NAME = "SAR_low_preprocessing_med5_CLACHE2_8"

src_path = get_dataset_path(SOURCE_NAME)
dst_path = get_dataset_path(TARGET_NAME)

splits = ["train", "valid", "test"]
subfolders = ["images", "labels"]

# Удаляем старую версию (если есть)
if dst_path.exists():
    shutil.rmtree(dst_path)
    print(f"[!] Старый '{TARGET_NAME}' удалён.")

# Создаём папку назначения, если её ещё нет
dst_path.mkdir(parents=True, exist_ok=True)

# Копируем YAML
shutil.copy(src_path / "data.yaml", dst_path / "data.yaml")

# Создаём структуру папок
for split in splits:
    for sub in subfolders:
        (dst_path / split / sub).mkdir(parents=True, exist_ok=True)

# Функция предобработки
def preprocess_image(img_path, clahe_clip=2.0, clahe_tile=8, median_ksize=5):
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Не удалось загрузить {img_path}")

    # Медианный фильтр
    img_med = cv2.medianBlur(img, median_ksize)

    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(clahe_tile, clahe_tile))
    img_clahe = clahe.apply(img_med)

    return img_clahe

# Обрабатываем и копируем файлы
for split in splits:
    for img_path in tqdm((src_path / split / "images").glob("*.jpg"), desc=f"Обработка {split}"):
        img_processed = preprocess_image(img_path)

        # Сохраняем обработанное изображение
        target_img_path = dst_path / split / "images" / img_path.name
        cv2.imwrite(str(target_img_path), img_processed)

        # Копируем разметку (txt) без изменений
        label_src_path = src_path / split / "labels" / (img_path.stem + ".txt")
        label_dst_path = dst_path / split / "labels" / (img_path.stem + ".txt")
        if label_src_path.exists():
            shutil.copy(label_src_path, label_dst_path)
        else:
            print(f"[!] Нет разметки для {img_path.name}")

print(f"\nДатасет с предобработкой создан в '{TARGET_NAME}'.")
