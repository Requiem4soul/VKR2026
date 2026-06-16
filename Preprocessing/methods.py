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
        Устаревший код, который не используется. Кстати ещё с первой реализации самой
        """
        squeezed = False
        if image.ndim == 3 and image.shape[2] == 1:
            image = image[:, :, 0]
            squeezed = True

        if method == 'auto':
            if noise_type == 'gaussian':
                method = 'bilateral'

            elif noise_type in ('salt_pepper', 'speckle'):
                method = 'median'

            else:
                method = 'wiener'
                if 'size' not in kwargs:
                    kwargs['size'] = 5

        result = image  # fallback если метод не распознан
        if method == 'median':
            if noise_level == 'high':
                ksize = kwargs.get('ksize', 7)
            elif noise_level == 'medium':
                ksize = kwargs.get('ksize', 5)
            else:  # low
                ksize = kwargs.get('ksize', 3)

            result = cv2.medianBlur(image, ksize)

        elif method == 'bilateral':
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

            result = cv2.bilateralFilter(image, d, sigma_color, sigma_space)

        elif method == 'gaussian':
            ksize = kwargs.get('ksize', 3)
            if ksize % 2 == 0:
                ksize += 1
            result = cv2.GaussianBlur(image, (ksize, ksize), 0)

        elif method == 'wiener':
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
            result = np.clip(filtered, 0, 255).astype(np.uint8)

        elif method == 'lee':
            ksize = kwargs.get('ksize', 3)
            enl = kwargs.get('enl', 1.0)

            def _lee_single_channel(img_ch: np.ndarray) -> np.ndarray:
                img_f = img_ch.astype(np.float64)
                # Локальное среднее и дисперсия через box filter
                mean = cv2.boxFilter(img_f, ddepth=-1, ksize=(ksize, ksize))
                mean_sq = cv2.boxFilter(img_f ** 2, ddepth=-1, ksize=(ksize, ksize))
                var_y = mean_sq - mean ** 2
                var_y = np.maximum(var_y, 0.0)
                var_x = var_y - (mean ** 2) / enl
                var_x = np.maximum(var_x, 0.0)
                k = np.where(var_y > 0, var_x / var_y, 0.0)
                filtered = mean + k * (img_f - mean)
                return np.clip(filtered, 0, 255).astype(np.uint8)

            if image.ndim == 3:
                channels = [_lee_single_channel(image[:, :, c])
                            for c in range(image.shape[2])]
                result = np.stack(channels, axis=2)
            else:
                result = _lee_single_channel(image)

        # Восстанавливаем (H,W,1) если исходное изображение было таким
        if squeezed:
            result = result[:, :, np.newaxis]
        return result

    @staticmethod
    def contrast_enhancement(
            image: np.ndarray,
            method: str = 'clahe',
            **kwargs
    ) -> np.ndarray:
        """
        Улучшение контраста.
        """
        squeezed = False
        if image.ndim == 3 and image.shape[2] == 1:
            image = image[:, :, 0]
            squeezed = True

        if method == 'clahe':
            clip_limit = kwargs.get('clip_limit', 2.0)
            tile_grid_size = kwargs.get('tile_grid_size', (8, 8))
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

            if image.ndim == 3:
                # RGB: применяем CLAHE только к L-каналу в LAB пространстве..
                lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
                lab[:, :, 0] = clahe.apply(lab[:, :, 0])
                result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            else:
                # Grayscale (2D): применяем CLAHE напрямую.
                result = clahe.apply(image)

        elif method == 'histogram_eq':
            if image.ndim == 3:
                # RGB: equalizeHist только по Y-каналу в YCrCb
                ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
                ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
                result = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
            else:
                result = cv2.equalizeHist(image)
        else:
            result = image

        # Восстанавливаем (H,W,1) если исходное изображение было таким
        if squeezed:
            result = result[:, :, np.newaxis]
        return result

    @staticmethod
    def gamma_correction(
            image: np.ndarray,
            gamma: float = 1.0,
            **kwargs
    ) -> np.ndarray:
        """
        Гамма-коррекция яркости изображения.
        """
        if gamma <= 0:
            return image

        lut = np.array(
            [((i / 255.0) ** gamma) * 255.0 for i in range(256)],
            dtype=np.uint8,
        )
        return cv2.LUT(image, lut)

    @staticmethod
    def sharpening(
            image: np.ndarray,
            method: str = 'unsharp_mask',
            **kwargs
    ) -> np.ndarray:
        """
        Увеличение резкости
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
        """
        if params is None:
            params = {}

        result = image.copy()

        for method in methods:
            method_params = params.get(method, {}).copy()

            # Поддержка ключей вида "denoise__1", "contrast_enhancement__0" и т.д.
            # merge_methods_params добавляет суффикс __N когда один тип метода
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
                result = PreprocessingMethods.gamma_correction(result, **method_params)
            elif base_method == 'sharpening':
                result = PreprocessingMethods.sharpening(result, **method_params)

        return result
