import shutil
from pathlib import Path
from random import sample, shuffle
from tqdm import tqdm
from config import DATASETS_GLOBAL_PATH
from Data.Datasets.dataset_work import get_dataset_path

SOURCE_NAME = "SAR"
TARGET_NAME = "SAR_low"

# Пути
src_path = get_dataset_path(SOURCE_NAME)
dst_path = get_dataset_path(TARGET_NAME)

# Целевые размеры
n_train = 1500
n_val = int(n_train * 0.2)    # 300
n_test = int(n_train * 0.1)   # 150
total_needed = n_train + n_val + n_test  # 1950

# Подкаталоги
splits = ["train", "valid", "test"]
subfolders = ["images", "labels"]

# Копируем data.yaml
shutil.copy(src_path / "data.yaml", dst_path / "data.yaml")

# Получаем все изображения
train_imgs = list((src_path / "train" / "images").glob("*.jpg"))
if len(train_imgs) < total_needed:
    raise ValueError(f"Недостаточно изображений в исходном train (нужно минимум {total_needed}, а есть {len(train_imgs)})")

# Случайный выбор нужного количества
selected = sample(train_imgs, total_needed)
shuffle(selected)

# Разделение по сплитам
split_data = {
    "train": selected[:n_train],
    "valid": selected[n_train:n_train + n_val],
    "test": selected[n_train + n_val:]
}

# Копирование файлов
for split in splits:
    for sub in subfolders:
        (dst_path / split / sub).mkdir(parents=True, exist_ok=True)

    for img_path in tqdm(split_data[split], desc=f"Копируем {split}"):
        label_path = Path(str(img_path).replace("images", "labels").replace(".jpg", ".txt"))
        target_img = dst_path / split / "images" / img_path.name
        target_lbl = dst_path / split / "labels" / label_path.name

        shutil.copy(img_path, target_img)
        if label_path.exists():
            shutil.copy(label_path, target_lbl)
        else:
            print(f"[!] Нет разметки для {img_path.name}")

# Итог
print(f"\n✅ Создан урезанный датасет '{TARGET_NAME}':")
print(f"- Train: {n_train}")
print(f"- Valid: {n_val}")
print(f"- Test:  {n_test}")
