from ultralytics import YOLO
import torch
import os

def train_models():
    # Проверка устройства
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Используем устройство: {device}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Пути к YAML-файлам
    data_original = r"N:\VKR_Datasets\SAR\data.yaml"

    # Параметры обучения
    epochs = 40
    batch_size = 8
    if torch.cuda.is_available():
        total_vram = torch.cuda.get_device_properties(0).total_memory
        print(f"VRAM: {total_vram / 1024 ** 3:.2f} ГБ")
        if total_vram >= 8 * 1024 ** 3:
            batch_size = 16

        # 1. Оригинальные изображения
    print("Обучение модели на оригинальных изображениях...")
    model_original = YOLO('yolov8m.pt')
    results_original = model_original.train(
        data=data_original,
        epochs=epochs,
        batch=batch_size,
        device=device,
        name='yolov8m_SAR_original',
        verbose=True
    )

if __name__ == '__main__':
    train_models()