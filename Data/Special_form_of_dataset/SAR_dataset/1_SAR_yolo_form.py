import shutil
from pathlib import Path
from random import shuffle
from config import DATASETS_GLOBAL_PATH
from Data.Datasets.dataset_work import get_dataset_path
from tqdm import tqdm

dataset_name = "SAR"

# Установили путь до датасета SAR
SAR_path = get_dataset_path(dataset_name)

# 1. Собираем все .jpg файлы
all_images = list(SAR_path.glob("*.jpg"))
shuffle(all_images)  # Перемешаем для случайного деления

# 70% train, 20% valid, 10% test
total = len(all_images)
train_count = int(total * 0.7)
val_count = int(total * 0.2)
test_count = total - train_count - val_count

splits = {
    "train": all_images[:train_count],
    "valid": all_images[train_count:train_count + val_count],
    "test": all_images[train_count + val_count:]
}

# 2. Создаём директории
for split_name in splits:
    for subfolder in ["images", "labels"]:
        target_dir = SAR_path / split_name / subfolder
        target_dir.mkdir(parents=True, exist_ok=True)

# 3. Копируем изображения и соответствующие .txt-файлы
for split_name, images in tqdm(splits.items(), desc='Приводим датасет к единному формату'):
    for img_path in images:
        label_path = img_path.with_suffix(".txt")

        # Назначение
        img_target = SAR_path / split_name / "images" / img_path.name
        label_target = SAR_path / split_name / "labels" / label_path.name

        # Копирование
        shutil.copy(img_path, img_target)

        if label_path.exists():
            shutil.copy(label_path, label_target)
        else:
            print(f"[!] Внимание: не найдена разметка для {img_path.name}, пропущено.")

print(f"Итоги для датасета {dataset_name}:")
print(f"- Train: {train_count} изображений")
print(f"- Val:   {val_count} изображений")
print(f"- Test:  {test_count} изображений")
