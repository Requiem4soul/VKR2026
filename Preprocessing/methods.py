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

        Научное обоснование выбора фильтров:

        1. Gaussian noise → Bilateral filter
           - Сохраняет края при удалении Gaussian шума
           - Tomasi & Manduchi (1998) "Bilateral filtering for gray and color images"

        2. Salt & Pepper → Median filter
           - Оптимален для импульсного шума
           - Gonzalez & Woods (2018) "Digital Image Processing"

        3. Speckle (SAR) → Median filter
           - Эффективен для мультипликативного шума
           - Lee (1980) "Digital image enhancement and noise filtering"

        Args:
            image: Входное изображение
            method: 'auto' (автовыбор), 'median', 'bilateral', 'nlm'
            noise_type: 'gaussian', 'salt_pepper', 'speckle', 'unknown'
            noise_level: 'low', 'medium', 'high'
            **kwargs: Дополнительные параметры для фильтров

        Returns:
            Обработанное изображение
        """

        # Автоматический выбор фильтра на основе типа шума
        if method == 'auto':
            if noise_type == 'gaussian':
                # Bilateral filter идеален для Gaussian noise
                method = 'bilateral'

            elif noise_type == 'salt_pepper':
                # Median filter - золотой стандарт для импульсного шума
                method = 'median'

            elif noise_type == 'speckle':
                # Median также хорош для speckle (SAR)
                method = 'median'


            else:  # unknown
                # Консервативный подход: median с минимальным kernel
                # Научное обоснование: Yin et al. (1996) - median robust к разным типам шума
                # Используем ksize=3 чтобы минимизировать риск повреждения деталей
                method = 'median'
                if 'ksize' not in kwargs:
                    kwargs['ksize'] = 3  # Минимальный размер для безопасности

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

        elif method == 'nlm':
            # Non-Local Means (для очень сильного Gaussian шума)
            if noise_level == 'high':
                h = kwargs.get('h', 15)
            elif noise_level == 'medium':
                h = kwargs.get('h', 10)
            else:
                h = kwargs.get('h', 7)

            template_window_size = kwargs.get('template_window_size', 7)
            search_window_size = kwargs.get('search_window_size', 21)

            return cv2.fastNlMeansDenoising(
                image, None, h,
                template_window_size,
                search_window_size
            )

        # Если метод не распознан - возвращаем оригинал
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
        """
        if method == 'clahe':
            clip_limit = kwargs.get('clip_limit', 2.0)
            tile_grid_size = kwargs.get('tile_grid_size', (8, 8))

            clahe = cv2.createCLAHE(
                clipLimit=clip_limit,
                tileGridSize=tile_grid_size
            )
            return clahe.apply(image)

        elif method == 'histogram_eq':
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
        """
        current_brightness = np.mean(image.astype(np.float32) / 255.0)

        if current_brightness < 1e-10:
            return image

        factor = target_brightness / current_brightness
        factor = np.clip(factor, 0.5, 2.0)

        corrected = (image.astype(np.float32) * factor).clip(0, 255).astype(np.uint8)
        return corrected

    @staticmethod
    def sharpening(image: np.ndarray, method: str = 'unsharp_mask', **kwargs) -> np.ndarray:
        """
        Увеличение резкости

        Args:
            method: 'unsharp_mask', 'laplacian'
        """
        if method == 'unsharp_mask':
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
        """
        if params is None:
            params = {}

        result = image.copy()

        for method in methods:
            method_params = params.get(method, {})

            if method == 'denoise':
                # Передаём информацию о шуме для автовыбора фильтра
                result = PreprocessingMethods.denoise(
                    result,
                    method='auto',  # Автоматический выбор
                    **method_params
                )
            elif method == 'contrast_enhancement':
                result = PreprocessingMethods.contrast_enhancement(result, **method_params)
            elif method == 'brightness_correction':
                result = PreprocessingMethods.brightness_correction(result, **method_params)
            elif method == 'sharpening':
                result = PreprocessingMethods.sharpening(result, **method_params)

        return result