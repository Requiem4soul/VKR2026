"""
Методы предобработки изображений

Содержит все методы для улучшения качества изображений:
- Шумоподавление (denoise)
- Улучшение контраста (contrast_enhancement)
- Коррекция яркости (brightness_correction)
- Увеличение резкости (sharpening)
"""

import cv2
import numpy as np
from typing import Dict, List, Any, Optional


class PreprocessingMethods:
    """Реализация всех методов предобработки"""

    @staticmethod
    def denoise(
            image: np.ndarray,
            method: str = 'auto',
            noise_type: str = 'unknown',
            noise_level: str = 'medium',
            **kwargs
    ) -> np.ndarray:
        """
        Шумоподавление с автоматическим выбором фильтра

        Научное обоснование выбора четырёх классических пространственных фильтров.

        Набор фильтров сформирован на основе:
          - Gonzalez & Woods (2018) "Digital Image Processing", 4th ed., Pearson —
            глава 5 выделяет Gaussian, Median, Wiener и Adaptive (Bilateral) как
            основные пространственные фильтры шумоподавления.
          - Fan et al. (2019) "Brief review of image denoising techniques",
            Visual Computing for Industry, Biomedicine, and Art, 2(1) —
            сравнительный обзор подтверждает тот же набор как базовый.
          - Tomasi & Manduchi (1998) "Bilateral filtering for gray and color images",
            ICCV — оригинальная статья по Bilateral filter.
          - Wiener N. (1949) "Extrapolation, Interpolation and Smoothing of
            Stationary Time Series", MIT Press — оригинальный Wiener filter.

        Покрытие типов шума:
          1. Gaussian noise → Bilateral filter (сохраняет края)
             Tomasi & Manduchi (1998), ibid.
          2. Salt & Pepper → Median filter (золотой стандарт для импульсного шума)
             Gonzalez & Woods (2018), ibid.
          3. Смешанный / неизвестный → Gaussian blur (быстрый, общего назначения)
             Gonzalez & Woods (2018), ibid.
          4. Адаптивный (любой тип) → Wiener filter (локально адаптируется к
             дисперсии шума; заменяет NLM ради вычислительной эффективности)
             Wiener (1949), ibid.; Fan et al. (2019), ibid.

        Args:
            image: Входное изображение
            method: 'auto' (автовыбор), 'median', 'gaussian', 'bilateral', 'wiener'
            noise_type: 'gaussian', 'salt_pepper', 'speckle', 'unknown'
            noise_level: 'low', 'medium', 'high'
            **kwargs: Дополнительные параметры для фильтров

        Returns:
            Обработанное изображение
        """

        # Автоматический выбор фильтра на основе типа шума.
        # Gonzalez & Woods (2018) гл. 5; Fan et al. (2019).
        if method == 'auto':
            if noise_type == 'gaussian':
                # Bilateral filter идеален для Gaussian noise — сохраняет края
                # Tomasi & Manduchi (1998)
                method = 'bilateral'

            elif noise_type in ('salt_pepper', 'speckle'):
                # Median filter — золотой стандарт для импульсного шума
                # Gonzalez & Woods (2018)
                method = 'median'

            else:  # unknown
                # Wiener filter — адаптируется к локальной дисперсии шума,
                # подходит при неизвестном типе шума. Wiener (1949).
                method = 'wiener'
                if 'size' not in kwargs:
                    kwargs['size'] = 5

        # Применяем выбранный фильтр
        if method == 'median':
            # Adaptive kernel size based on noise level
            if noise_level == 'high':
                ksize = kwargs.get('ksize', 7)
            elif noise_level == 'medium':
                ksize = kwargs.get('ksize', 5)
            else:  # low
                ksize = kwargs.get('ksize', 3)

            return cv2.medianBlur(image, ksize)

        elif method == 'bilateral':
            # Адаптивные параметры для bilateral filter
            if noise_level == 'high':
                d = kwargs.get('d', 9)
                sigma_color = kwargs.get('sigma_color', 90)
                sigma_space = kwargs.get('sigma_space', 90)
            elif noise_level == 'medium':
                d = kwargs.get('d', 9)
                sigma_color = kwargs.get('sigma_color', 75)
                sigma_space = kwargs.get('sigma_space', 75)
            else:  # low
                d = kwargs.get('d', 5)
                sigma_color = kwargs.get('sigma_color', 50)
                sigma_space = kwargs.get('sigma_space', 50)

            return cv2.bilateralFilter(image, d, sigma_color, sigma_space)

        elif method == 'gaussian':
            # Gaussian blur — быстрый линейный фильтр общего назначения.
            # Эффективен для равномерного фонового шума.
            # Gonzalez & Woods (2018), гл. 3.
            ksize = kwargs.get('ksize', 3)
            # ksize должен быть нечётным
            if ksize % 2 == 0:
                ksize += 1
            return cv2.GaussianBlur(image, (ksize, ksize), 0)

        elif method == 'wiener':
            # Wiener filter — адаптивный фильтр, минимизирующий среднеквадратичную
            # ошибку. Локально адаптируется к дисперсии шума: в однородных областях
            # сглаживает сильнее, вблизи краёв — слабее.
            # Wiener (1949); реализация через scipy.signal.wiener.
            # Заменяет NLM: сопоставимое качество при на порядок меньших вычислениях.
            # Fan et al. (2019) "Brief review of image denoising techniques".
            from scipy.signal import wiener as scipy_wiener
            size = kwargs.get('size', 5)
            # scipy.signal.wiener работает с float, возвращает float
            img_float = image.astype(np.float64)
            if img_float.ndim == 3:
                # Применяем поканально для RGB
                channels = [scipy_wiener(img_float[:, :, c], mysize=size)
                            for c in range(img_float.shape[2])]
                filtered = np.stack(channels, axis=2)
            else:
                filtered = scipy_wiener(img_float, mysize=size)
            return np.clip(filtered, 0, 255).astype(np.uint8)

        # Если метод не распознан — возвращаем оригинал
        return image

    @staticmethod
    def contrast_enhancement(
            image: np.ndarray,
            method: str = 'clahe',
            **kwargs
    ) -> np.ndarray:
        """
        Улучшение контраста

        Args:
            method: 'clahe', 'histogram_eq'
            **kwargs: Дополнительные параметры
        """
        if method == 'clahe':
            clip_limit = kwargs.get('clip_limit', 2.0)
            tile_grid_size = kwargs.get('tile_grid_size', (8, 8))
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

            if image.ndim == 3:
                # RGB: применяем CLAHE только к L-каналу в LAB пространстве.
                # Это стандартный подход для цветных изображений —
                # улучшаем яркость не затрагивая цвет.
                # Pisano et al. (1998); Gonzalez & Woods (2018).
                lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
                lab[:, :, 0] = clahe.apply(lab[:, :, 0])
                return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            else:
                return clahe.apply(image)

        elif method == 'histogram_eq':
            if image.ndim == 3:
                # RGB: equalizeHist только по Y-каналу в YCrCb
                ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
                ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
                return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
            else:
                return cv2.equalizeHist(image)

        return image

    @staticmethod
    def brightness_correction(
            image: np.ndarray,
            target_brightness: float = 0.5,
            **kwargs
    ) -> np.ndarray:
        """
        Коррекция яркости

        Args:
            target_brightness: Целевая яркость (0-1)
            **kwargs: Дополнительные параметры
        """
        # Для RGB вычисляем яркость по luminance-формуле (ITU-R BT.601),
        # а не как среднее по всем каналам — это физически корректнее.
        # Для grayscale среднее пикселей.
        img_f = image.astype(np.float32) / 255.0
        if image.ndim == 3:
            # BGR порядок в OpenCV: B=0, G=1, R=2
            luminance = 0.114 * img_f[:, :, 0] + 0.587 * img_f[:, :, 1] + 0.299 * img_f[:, :, 2]
            current_brightness = float(np.mean(luminance))
        else:
            current_brightness = float(np.mean(img_f))

        if current_brightness < 1e-10:
            return image

        factor = target_brightness / current_brightness
        factor = np.clip(factor, 0.5, 2.0)

        corrected = (img_f * factor * 255.0).clip(0, 255).astype(np.uint8)
        return corrected

    @staticmethod
    def sharpening(
            image: np.ndarray,
            method: str = 'unsharp_mask',
            **kwargs
    ) -> np.ndarray:
        """
        Увеличение резкости

        Args:
            method: 'unsharp_mask', 'laplacian'
            **kwargs: Дополнительные параметры
        """
        if method == 'unsharp_mask':
            # Unsharp masking
            blurred = cv2.GaussianBlur(image, (0, 0), 3)
            alpha = kwargs.get('alpha', 1.5)
            sharpened = cv2.addWeighted(image, 1 + alpha, blurred, -alpha, 0)
            return sharpened

        elif method == 'laplacian':
            laplacian = cv2.Laplacian(image, cv2.CV_64F)
            alpha = kwargs.get('alpha', 0.5)

            sharpened = image.astype(np.float64) - alpha * laplacian
            sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)

            return sharpened

        return image

    @staticmethod
    def apply_pipeline(
            image: np.ndarray,
            methods: List[str],
            params: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> np.ndarray:
        """
        Применяет последовательность методов

        Args:
            image: Входное изображение
            methods: Список методов ['denoise', 'contrast_enhancement', ...]
            params: Параметры для каждого метода
                    Например: {
                        'denoise': {'method': 'median', 'ksize': 5},
                        'contrast_enhancement': {'clip_limit': 2.0}
                    }
        """
        if params is None:
            params = {}

        result = image.copy()

        for method in methods:
            method_params = params.get(method, {}).copy()

            # Поддержка ключей вида "denoise__1", "contrast_enhancement__0" и т.д.
            # merge_methods_params добавляет суффикс __N когда один тип метода
            # встречается несколько раз в пайплайне (например Gaussian + Wiener).
            base_method = method.split("__")[0]

            if base_method == 'denoise':
                denoise_method = method_params.pop('method', 'auto')
                result = PreprocessingMethods.denoise(
                    result,
                    method=denoise_method,
                    **method_params
                )
            elif base_method == 'contrast_enhancement':
                result = PreprocessingMethods.contrast_enhancement(result, **method_params)
            elif base_method == 'brightness_correction':
                result = PreprocessingMethods.brightness_correction(result, **method_params)
            elif base_method == 'sharpening':
                result = PreprocessingMethods.sharpening(result, **method_params)

        return result
