import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy
from skimage.filters import gaussian
from skimage.util import img_as_ubyte


def analyze_image(path):
    # Шаг 1: Загрузка RGB-изображения
    img_rgb = cv2.imread(path)
    if img_rgb is None:
        raise FileNotFoundError(f"Не удалось загрузить изображение по пути: {path}")

    print(f"Исходное изображение (RGB): dtype={img_rgb.dtype}, min={img_rgb.min()}, max={img_rgb.max()}")

    # Шаг 2: Перевод в оттенки серого
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

    print(f"Grayscale изображение: dtype={gray.dtype}, min={gray.min()}, max={gray.max()}")
    print(f"Среднее значение яркости: {gray.mean():.2f}, медиана: {np.median(gray):.2f}\n")

    # Шаг 3: Построение гистограммы
    hist, bins = np.histogram(gray.flatten(), bins=256, range=[0, 256])

    plt.figure(figsize=(8, 5))
    plt.title("Гистограмма яркости (оттенков серого)")
    plt.xlabel("Яркость (0–255)")
    plt.ylabel("Частота")
    plt.bar(bins[:-1], hist, width=1, color='gray', edgecolor='black')
    plt.xlim([0, 255])
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

    # Шаг 4: Печать только оттенков с ненулевой частотой
    print("Оттенки с ненулевым количеством пикселей:")
    for intensity, count in enumerate(hist):
        if count > 0:
            print(f"  Оттенок {intensity:3d}: {int(count)} пикселей")

    # Шаг 5: Метрики
    hist_norm = hist / hist.sum()
    img_entropy = entropy(hist_norm[hist_norm > 0])  # избегаем log(0)
    std_dev = np.std(gray)
    gray_smoothed = gaussian(gray, sigma=1)
    noise_metric = np.mean((gray - img_as_ubyte(gray_smoothed)) ** 2)

    print(f"\nМетрики изображения:")
    print(f"  Энтропия: {img_entropy:.4f}")
    print(f"  Стандартное отклонение яркости: {std_dev:.2f}")
    print(f"  Среднеквадратичное отклонение от сглаженного (эвристика шума): {noise_metric:.2f}")


# Пример использования:
analyze_image("N:/VKR_Datasets/SAR_low/test/images/Gao_ship_hh_020160825440209047.jpg")
