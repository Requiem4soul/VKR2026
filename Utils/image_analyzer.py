import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy.signal import find_peaks
from scipy import stats


@dataclass
class ImageMetrics:
    """Метрики одного изображения"""
    # Шум
    snr_db: float
    noise_variance: float
    noise_level: str  # 'low', 'medium', 'high'
    noise_type: str  # 'gaussian', 'salt_pepper', 'speckle', 'unknown'

    # Контраст
    global_contrast: float
    local_contrast: float
    michelson_contrast: float
    contrast_level: str

    # Яркость
    mean_brightness: float
    median_brightness: float
    brightness_level: str  # 'dark', 'normal', 'bright'

    # Резкость
    sharpness_score: float
    blur_detected: bool

    # Гистограмма (для диагностики)
    histogram_span: float
    histogram_peaks: int
    is_bimodal: bool

    # Динамический диапазон
    dynamic_range: float
    clipping_detected: bool

    # Общая оценка качества
    overall_quality: str  # 'poor', 'fair', 'good', 'excellent'
    needs_preprocessing: bool
    recommended_methods: List[str]


@dataclass
class DatasetMetrics:
    """Агрегированные метрики датасета"""
    num_images: int

    # Статистика по шуму
    avg_snr: float
    std_snr: float
    noise_distribution: Dict[str, int]

    # Статистика по контрасту
    avg_contrast: float
    std_contrast: float
    contrast_distribution: Dict[str, int]

    # Статистика по яркости
    avg_brightness: float
    std_brightness: float
    brightness_distribution: Dict[str, int]

    # Статистика по резкости
    avg_sharpness: float
    blur_count: int

    # Общая характеристика
    dataset_homogeneity: float
    dominant_issues: List[str]

    # Рекомендации
    recommended_global_preprocessing: List[str]
    needs_adaptive_preprocessing: bool
    suggested_clusters: int


class UniversalImageAnalyzer:
    """
    Универсальный анализатор изображений
    Работает с любым датасетом независимо от диапазона значений и размера
    """

    def __init__(self, verbose: bool = False):
        self.verbose = verbose

        self.thresholds = {
            'snr': {
                'low': 15,
                'medium': 25,
                'high': 35
            },
            'contrast': {
                'low': 0.2,
                'medium': 0.4,
                'high': 0.7
            },
            'brightness': {
                'dark': 0.3,
                'bright': 0.7
            },
            'sharpness': {
                'blur_threshold': 100
            }
        }

    def normalize_image(self, image: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Нормализует изображение к диапазону [0, 1]"""
        original_dtype = image.dtype
        original_range = (image.min(), image.max())

        if image.dtype == np.uint8:
            max_val = 255
        elif image.dtype == np.uint16:
            max_val = 65535
        else:
            max_val = image.max() if image.max() > 1 else 1

        normalized = image.astype(np.float32) / max_val

        metadata = {
            'original_dtype': original_dtype,
            'original_range': original_range,
            'max_val': max_val,
            'is_color': len(image.shape) == 3
        }

        return normalized, metadata

    def estimate_noise_snr(self, image: np.ndarray) -> Tuple[float, float]:
        """Оценка уровня шума через SNR"""
        if len(image.shape) == 3:
            if image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
            elif image.shape[2] == 1:
                image = image[:, :, 0]
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        signal = cv2.GaussianBlur(image, (5, 5), 1.0)
        noise = image - signal

        signal_power = np.mean(signal ** 2)
        noise_power = np.mean(noise ** 2)

        if noise_power < 1e-10:
            snr_db = 60
        else:
            snr = signal_power / noise_power
            snr_db = 10 * np.log10(snr)

        noise_variance = np.var(noise)

        return snr_db, noise_variance

    def identify_noise_type(self, image: np.ndarray) -> str:
        """
        Определяет тип шума в изображении

        Основано на статистическом анализе шумовой компоненты.

        Научное обоснование:
        - Gaussian noise: симметричное распределение (skewness ~ 0, kurtosis ~ 3)
        - Salt & Pepper: тяжёлые хвосты (high kurtosis > 5)
        - Speckle: мультипликативный (sigma растёт с интенсивностью)

        References:
        - Gonzalez & Woods, "Digital Image Processing" (2018)
        - Buades et al., "A Review of Image Denoising Algorithms" (2005)

        Returns:
            'gaussian', 'salt_pepper', 'speckle', или 'unknown'
        """
        if len(image.shape) == 3:
            if image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
            elif image.shape[2] == 1:
                image = image[:, :, 0]
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        signal = cv2.GaussianBlur(image, (5, 5), 1.0)
        noise = image - signal
        noise_flat = noise.flatten()

        skewness = float(stats.skew(noise_flat))
        kurtosis_val = float(stats.kurtosis(noise_flat))

        is_multiplicative = self._check_multiplicative_noise(image, noise)

        if is_multiplicative:
            return 'speckle'
        elif kurtosis_val > 5:
            return 'salt_pepper'
        elif abs(skewness) < 0.5:
            return 'gaussian'
        else:
            return 'unknown'

    def _check_multiplicative_noise(self, image: np.ndarray, noise: np.ndarray) -> bool:
        """
        Проверяет является ли шум мультипликативным (speckle).

        Основано на методе Lee (1980):
        Lee, J.S. (1980). "Digital image enhancement and noise filtering
        by use of local statistics" IEEE TPAMI, 2(2), 165-168.

        Returns:
            True если шум мультипликативный (speckle)
        """
        percentile_75 = np.percentile(image, 75)
        percentile_25 = np.percentile(image, 25)

        bright_mask = image > percentile_75
        dark_mask = image < percentile_25

        if np.sum(bright_mask) < 100 or np.sum(dark_mask) < 100:
            return False

        noise_std_bright = np.std(noise[bright_mask])
        noise_std_dark = np.std(noise[dark_mask])

        if noise_std_dark < 1e-10:
            return False

        ratio = noise_std_bright / noise_std_dark

        return ratio > 1.5

    def measure_contrast(self, image: np.ndarray) -> Dict[str, float]:
        """Измерение контраста"""
        if len(image.shape) == 3:
            if image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
            elif image.shape[2] == 1:
                image = image[:, :, 0]
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        i_max = image.max()
        i_min = image.min()

        if (i_max + i_min) < 1e-10:
            michelson = 0
        else:
            michelson = (i_max - i_min) / (i_max + i_min)

        rms_contrast = np.std(image)

        patch_size = 32
        h, w = image.shape
        local_contrasts = []

        for i in range(0, h - patch_size, patch_size):
            for j in range(0, w - patch_size, patch_size):
                patch = image[i:i + patch_size, j:j + patch_size]
                if patch.size > 0:
                    local_std = np.std(patch)
                    local_mean = np.mean(patch)
                    if local_mean > 1e-10:
                        local_contrasts.append(local_std / local_mean)

        local_contrast = np.mean(local_contrasts) if local_contrasts else 0

        return {
            'michelson': float(michelson),
            'rms': float(rms_contrast),
            'local': float(local_contrast),
            'global': float(michelson)
        }

    def measure_sharpness(self, image: np.ndarray) -> Tuple[float, bool]:
        """Измерение резкости через Laplacian variance"""
        if len(image.shape) == 3:
            if image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
            elif image.shape[2] == 1:
                image = image[:, :, 0]
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        image_uint8 = (image * 255).astype(np.uint8)
        laplacian = cv2.Laplacian(image_uint8, cv2.CV_64F)
        sharpness_score = laplacian.var()

        adaptive_threshold = self.thresholds['sharpness']['blur_threshold']
        blur_detected = sharpness_score < adaptive_threshold

        return float(sharpness_score), blur_detected

    def analyze_histogram(self, image: np.ndarray) -> Dict:
        """Анализ гистограммы для диагностики проблем"""
        if len(image.shape) == 3:
            if image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
            elif image.shape[2] == 1:
                image = image[:, :, 0]
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        hist, bins = np.histogram(image.flatten(), bins=256, range=[0, 1])

        non_zero_bins = np.where(hist > 0)[0]
        if len(non_zero_bins) > 0:
            histogram_span = (non_zero_bins[-1] - non_zero_bins[0]) / 255
        else:
            histogram_span = 0

        peaks, _ = find_peaks(hist, height=hist.max() * 0.1)
        num_peaks = len(peaks)
        is_bimodal = num_peaks >= 2

        edge_threshold = hist.max() * 0.05
        clipping_detected = (hist[0] > edge_threshold) or (hist[-1] > edge_threshold)

        return {
            'span': float(histogram_span),
            'num_peaks': int(num_peaks),
            'is_bimodal': bool(is_bimodal),
            'clipping_detected': bool(clipping_detected),
            'histogram': hist
        }

    def analyze_image(self, image_path: Path) -> ImageMetrics:
        """Полный анализ одного изображения"""
        image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError(f"Не удалось загрузить {image_path}")

        if len(image.shape) == 3:
            if image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
            elif image.shape[2] == 1:
                image = image[:, :, 0]
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        image_norm, metadata = self.normalize_image(image)

        # Анализ шума
        snr_db, noise_var = self.estimate_noise_snr(image_norm)

        if snr_db < self.thresholds['snr']['low']:
            noise_level = 'high'
        elif snr_db < self.thresholds['snr']['medium']:
            noise_level = 'medium'
        else:
            noise_level = 'low'

        # Анализ контраста
        contrast_metrics = self.measure_contrast(image_norm)
        global_contrast = contrast_metrics['global']

        if global_contrast < self.thresholds['contrast']['low']:
            contrast_level = 'low'
        elif global_contrast < self.thresholds['contrast']['medium']:
            contrast_level = 'medium'
        else:
            contrast_level = 'high'

        # Анализ яркости
        mean_brightness = float(np.mean(image_norm))
        median_brightness = float(np.median(image_norm))

        if mean_brightness < self.thresholds['brightness']['dark']:
            brightness_level = 'dark'
        elif mean_brightness > self.thresholds['brightness']['bright']:
            brightness_level = 'bright'
        else:
            brightness_level = 'normal'

        # Анализ резкости
        sharpness, blur_detected = self.measure_sharpness(image_norm)

        # Анализ гистограммы
        hist_analysis = self.analyze_histogram(image_norm)

        # Динамический диапазон
        dynamic_range = float(image_norm.max() - image_norm.min())

        # Рекомендации
        recommended_methods = []
        needs_preprocessing = False

        if noise_level in ['medium', 'high']:
            recommended_methods.append('denoise')
            needs_preprocessing = True

        if contrast_level == 'low':
            recommended_methods.append('contrast_enhancement')
            needs_preprocessing = True

        if brightness_level in ['dark', 'bright']:
            recommended_methods.append('brightness_correction')
            needs_preprocessing = True

        if blur_detected:
            recommended_methods.append('sharpening')
            needs_preprocessing = True

        if hist_analysis['clipping_detected']:
            recommended_methods.append('dynamic_range_expansion')
            needs_preprocessing = True

        # Общая оценка
        quality_score = 0
        if noise_level == 'low': quality_score += 1
        if contrast_level in ['medium', 'high']: quality_score += 1
        if brightness_level == 'normal': quality_score += 1
        if not blur_detected: quality_score += 1

        if quality_score >= 4:
            overall_quality = 'excellent'
        elif quality_score >= 3:
            overall_quality = 'good'
        elif quality_score >= 2:
            overall_quality = 'fair'
        else:
            overall_quality = 'poor'

        return ImageMetrics(
            snr_db=snr_db,
            noise_variance=noise_var,
            noise_level=noise_level,
            noise_type=self.identify_noise_type(image_norm),
            global_contrast=global_contrast,
            local_contrast=contrast_metrics['local'],
            michelson_contrast=contrast_metrics['michelson'],
            contrast_level=contrast_level,
            mean_brightness=mean_brightness,
            median_brightness=median_brightness,
            brightness_level=brightness_level,
            sharpness_score=sharpness,
            blur_detected=blur_detected,
            histogram_span=hist_analysis['span'],
            histogram_peaks=hist_analysis['num_peaks'],
            is_bimodal=hist_analysis['is_bimodal'],
            dynamic_range=dynamic_range,
            clipping_detected=hist_analysis['clipping_detected'],
            overall_quality=overall_quality,
            needs_preprocessing=needs_preprocessing,
            recommended_methods=recommended_methods
        )

    def analyze_dataset(
            self,
            dataset_path: Path,
            sample_size: Optional[int] = None,
            split: str = 'train'
    ) -> Tuple[DatasetMetrics, List[ImageMetrics]]:
        """Анализ всего датасета.

        Поддерживает структуры:
        1. YOLO: <split>/images/*.jpg
        2. Классификация ImageFolder: <split>/<class_name>/*.jpg
        3. Классификация flat: <split>/*.jpg
        """
        split_dir = dataset_path / split

        if not split_dir.exists():
            raise ValueError(f"Директория {split_dir} не найдена")

        # Формат 1: YOLO (split/images/)
        images_dir = split_dir / "images"
        if images_dir.exists() and images_dir.is_dir():
            image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
        else:
            # Форматы 2 & 3: Классификация — ImageFolder или flat.
            # rglob собирает файлы из подпапок (class_name/) и из самой split/
            image_files = []
            for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff"):
                image_files.extend(split_dir.rglob(ext))

        if not image_files:
            raise ValueError(
                f"Изображения не найдены в {split_dir}. "
                "Ожидается YOLO (split/images/) или классификация "
                "(split/class/ или split/*.jpg)."
            )

        if sample_size and sample_size < len(image_files):
            import random
            image_files = random.sample(image_files, sample_size)

        print(f"Анализируем {len(image_files)} изображений из {split}...")

        all_metrics = []
        for img_path in image_files:
            try:
                metrics = self.analyze_image(img_path)
                all_metrics.append(metrics)

                if self.verbose:
                    print(f"  {img_path.name}: {metrics.overall_quality}, "
                          f"noise={metrics.noise_level}, contrast={metrics.contrast_level}")

            except Exception as e:
                print(f"Ошибка при анализе {img_path}: {e}")
                continue

        dataset_metrics = self._aggregate_metrics(all_metrics)

        return dataset_metrics, all_metrics

    def _aggregate_metrics(self, all_metrics: List[ImageMetrics]) -> DatasetMetrics:
        """Агрегация метрик по датасету"""
        if not all_metrics:
            raise ValueError("Нет метрик для агрегации")

        snr_values = [m.snr_db for m in all_metrics]
        contrast_values = [m.global_contrast for m in all_metrics]
        brightness_values = [m.mean_brightness for m in all_metrics]
        sharpness_values = [m.sharpness_score for m in all_metrics]

        noise_dist = {
            'low': sum(1 for m in all_metrics if m.noise_level == 'low'),
            'medium': sum(1 for m in all_metrics if m.noise_level == 'medium'),
            'high': sum(1 for m in all_metrics if m.noise_level == 'high')
        }

        contrast_dist = {
            'low': sum(1 for m in all_metrics if m.contrast_level == 'low'),
            'medium': sum(1 for m in all_metrics if m.contrast_level == 'medium'),
            'high': sum(1 for m in all_metrics if m.contrast_level == 'high')
        }

        brightness_dist = {
            'dark': sum(1 for m in all_metrics if m.brightness_level == 'dark'),
            'normal': sum(1 for m in all_metrics if m.brightness_level == 'normal'),
            'bright': sum(1 for m in all_metrics if m.brightness_level == 'bright')
        }

        blur_count = sum(1 for m in all_metrics if m.blur_detected)

        cv_snr = np.std(snr_values) / (np.mean(snr_values) + 1e-10)
        cv_contrast = np.std(contrast_values) / (np.mean(contrast_values) + 1e-10)
        cv_brightness = np.std(brightness_values) / (np.mean(brightness_values) + 1e-10)

        avg_cv = (cv_snr + cv_contrast + cv_brightness) / 3
        homogeneity = max(0, 1 - avg_cv)

        dominant_issues = []
        if noise_dist['high'] > len(all_metrics) * 0.3:
            dominant_issues.append('high_noise')
        if contrast_dist['low'] > len(all_metrics) * 0.3:
            dominant_issues.append('low_contrast')
        if brightness_dist['dark'] > len(all_metrics) * 0.3:
            dominant_issues.append('dark_images')
        if brightness_dist['bright'] > len(all_metrics) * 0.3:
            dominant_issues.append('bright_images')
        if blur_count > len(all_metrics) * 0.3:
            dominant_issues.append('blur')

        recommended_global = []
        if 'high_noise' in dominant_issues:
            recommended_global.append('denoise')
        if 'low_contrast' in dominant_issues:
            recommended_global.append('contrast_enhancement')
        if 'dark_images' in dominant_issues or 'bright_images' in dominant_issues:
            recommended_global.append('brightness_correction')

        needs_adaptive = homogeneity < 0.7

        if needs_adaptive:
            if homogeneity < 0.4:
                suggested_clusters = 4
            elif homogeneity < 0.6:
                suggested_clusters = 3
            else:
                suggested_clusters = 2
        else:
            suggested_clusters = 1

        return DatasetMetrics(
            num_images=len(all_metrics),
            avg_snr=float(np.mean(snr_values)),
            std_snr=float(np.std(snr_values)),
            noise_distribution=noise_dist,
            avg_contrast=float(np.mean(contrast_values)),
            std_contrast=float(np.std(contrast_values)),
            contrast_distribution=contrast_dist,
            avg_brightness=float(np.mean(brightness_values)),
            std_brightness=float(np.std(brightness_values)),
            brightness_distribution=brightness_dist,
            avg_sharpness=float(np.mean(sharpness_values)),
            blur_count=blur_count,
            dataset_homogeneity=float(homogeneity),
            dominant_issues=dominant_issues,
            recommended_global_preprocessing=recommended_global,
            needs_adaptive_preprocessing=needs_adaptive,
            suggested_clusters=suggested_clusters
        )

    def print_dataset_report(self, dataset_metrics: DatasetMetrics):
        """Вывод отчёта по датасету"""
        print("\n" + "=" * 70)
        print("ОТЧЁТ ПО АНАЛИЗУ ДАТАСЕТА")
        print("=" * 70)

        print(f"\nОбщая информация:")
        print(f"  Проанализировано изображений: {dataset_metrics.num_images}")
        print(f"  Однородность датасета: {dataset_metrics.dataset_homogeneity:.2%}")

        print(f"\nШум:")
        print(f"  Средний SNR: {dataset_metrics.avg_snr:.1f} dB (+-{dataset_metrics.std_snr:.1f})")
        print(f"  Распределение:")
        for level, count in dataset_metrics.noise_distribution.items():
            pct = count / dataset_metrics.num_images * 100
            print(f"    {level}: {count} ({pct:.1f}%)")

        print(f"\nКонтраст:")
        print(f"  Средний контраст: {dataset_metrics.avg_contrast:.3f} (+-{dataset_metrics.std_contrast:.3f})")
        print(f"  Распределение:")
        for level, count in dataset_metrics.contrast_distribution.items():
            pct = count / dataset_metrics.num_images * 100
            print(f"    {level}: {count} ({pct:.1f}%)")

        print(f"\nЯркость:")
        print(f"  Средняя яркость: {dataset_metrics.avg_brightness:.3f} (+-{dataset_metrics.std_brightness:.3f})")
        print(f"  Распределение:")
        for level, count in dataset_metrics.brightness_distribution.items():
            pct = count / dataset_metrics.num_images * 100
            print(f"    {level}: {count} ({pct:.1f}%)")

        print(f"\nРезкость:")
        print(f"  Средний score: {dataset_metrics.avg_sharpness:.1f}")
        blur_pct = dataset_metrics.blur_count / dataset_metrics.num_images * 100
        print(f"  Размытых изображений: {dataset_metrics.blur_count} ({blur_pct:.1f}%)")

        print(f"\nДоминирующие проблемы:")
        if dataset_metrics.dominant_issues:
            for issue in dataset_metrics.dominant_issues:
                print(f"  - {issue}")
        else:
            print("  Проблем не обнаружено")

        print(f"\nРекомендации:")
        print(
            f"  Глобальная предобработка: {', '.join(dataset_metrics.recommended_global_preprocessing) or 'не требуется'}")
        print(f"  Адаптивная предобработка: {'ДА' if dataset_metrics.needs_adaptive_preprocessing else 'НЕТ'}")
        if dataset_metrics.needs_adaptive_preprocessing:
            print(f"  Рекомендуемое количество кластеров: {dataset_metrics.suggested_clusters}")

        print("\n" + "=" * 70 + "\n")
