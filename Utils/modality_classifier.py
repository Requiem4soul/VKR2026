"""
Классификация модальности изображений по метрикам.

Научные источники:
- Oliver & Quegan (2004) - SAR
- Pham et al. (2000) - Medical imaging
- Gonzalez & Woods (2018) - Natural images
- Vollmer & Möllmann (2017) - Infrared/Thermal
- Sternberg (1983) - Microscopy

ЦВЕТОВОЙ ФИЛЬТР (добавлен):
    SAR и рентген физически не могут давать цветные изображения.
    Если датасет определён как цветной (is_color_dataset=True),
    типы 'sar' и 'medical_xray' исключаются до скоринга.
    Это жёсткое ограничение, основанное на физической природе модальностей:
    - Oliver & Quegan (2004) — SAR всегда grayscale
    - Pham et al. (2000) — рентген всегда grayscale
    - Gonzalez & Woods (2018) — цветность как первичный признак классификации
"""

from typing import Dict, Tuple
import numpy as np


class ImageModalityClassifier:
    """
    Классификатор типа/модальности изображений на основе их метрик
    
    Определяет к какой категории относится датасет:
    - SAR (Synthetic Aperture Radar)
    - Medical X-ray
    - Natural Photo
    - Infrared/Thermal
    - Microscopy
    """
    
    # Пороги из научной литературы
    THRESHOLDS = {
        'sar': {
            'contrast': (0.85, 1.0),
            'brightness': (0.0, 0.2),
            'snr': (0, 15),
            'sharpness': (3000, 10000),
            'source': 'Oliver & Quegan (2004)',
            'description': 'Радарные изображения с характерным speckle шумом'
        },
        'medical_xray': {
            'contrast': (0.4, 0.8),
            'brightness': (0.1, 0.4),
            'snr': (20, 40),
            'dynamic_range': (0.0, 0.7),
            'source': 'Pham et al. (2000)',
            'description': 'Медицинские рентгеновские снимки'
        },
        'natural_photo': {
            'contrast': (0.3, 0.7),
            'brightness': (0.3, 0.7),
            'snr': (15, 35),
            'source': 'Gonzalez & Woods (2018)',
            'description': 'Естественные фотографии с нормальным освещением'
        },
        'infrared': {
            'contrast': (0.0, 0.4),
            'brightness': (0.2, 0.6),
            'snr': (10, 25),
            'blur_ratio': (0.4, 1.0),
            'source': 'Vollmer & Möllmann (2017)',
            'description': 'Тепловизионные/инфракрасные изображения'
        },
        'microscopy': {
            'contrast': (0.5, 0.9),
            'brightness': (0.15, 0.5),
            'snr': (8, 20),
            'bimodal': True,
            'source': 'Sternberg (1983)',
            'description': 'Микроскопические изображения'
        }
    }
    
    # Типы, которые физически не могут давать цветные изображения.
    # Источники: Oliver & Quegan (2004) для SAR,
    #            Pham et al. (2000) для рентгена.
    _GRAYSCALE_ONLY_MODALITIES = {'sar', 'medical_xray'}

    def classify(self, dataset_metrics) -> Dict:
        """
        Классифицирует тип датасета на основе его метрик.

        Если датасет цветной (is_color_dataset=True), типы 'sar' и 'medical_xray'
        исключаются из рассмотрения как физически невозможные.
        Источники: Oliver & Quegan (2004), Pham et al. (2000),
                   Gonzalez & Woods (2018).

        Args:
            dataset_metrics: Объект DatasetMetrics с агрегированными метриками

        Returns:
            dict: {
                'modality': str,      # Определённый тип
                'confidence': float,  # Уверенность (0-1)
                'source': str,        # Научная ссылка
                'description': str,   # Описание типа
                'all_scores': dict,   # Оценки для всех типов
                'is_color': bool,     # Цветной ли датасет
                'color_diversity': float  # Среднее MICD
            }
        """
        # Определяем цветной ли датасет (если атрибут доступен)
        is_color = getattr(dataset_metrics, 'is_color_dataset', False)
        color_diversity = getattr(dataset_metrics, 'avg_color_diversity', 0.0)

        scores = {}
        for modality, thresholds in self.THRESHOLDS.items():
            # Цветной датасет не может быть SAR или рентгеном —
            # жёсткое исключение на основе физической природы модальностей.
            # Oliver & Quegan (2004); Pham et al. (2000)
            if is_color and modality in self._GRAYSCALE_ONLY_MODALITIES:
                scores[modality] = 0.0
                continue
            score = self._calculate_match_score(dataset_metrics, thresholds)
            scores[modality] = score

        best_modality = max(scores, key=scores.get)
        confidence = scores[best_modality]

        return {
            'modality': best_modality,
            'confidence': confidence,
            'source': self.THRESHOLDS[best_modality]['source'],
            'description': self.THRESHOLDS[best_modality]['description'],
            'all_scores': scores,
            'is_color': is_color,
            'color_diversity': color_diversity,
        }
    
    def _calculate_match_score(
        self, 
        metrics, 
        thresholds: Dict
    ) -> float:
        """
        Считает насколько метрики датасета соответствуют данному типу
        
        Returns:
            float: Оценка соответствия от 0.0 до 1.0
        """
        score = 0.0
        count = 0
        
        # 1. Контраст
        if 'contrast' in thresholds:
            low, high = thresholds['contrast']
            if low <= metrics.avg_contrast <= high:
                score += 1.0
            else:
                # Частичная оценка если близко к границе
                distance = min(
                    abs(metrics.avg_contrast - low),
                    abs(metrics.avg_contrast - high)
                )
                if distance < 0.15:  # Толерантность 15%
                    score += 0.5
            count += 1
        
        # 2. Яркость
        if 'brightness' in thresholds:
            low, high = thresholds['brightness']
            if low <= metrics.avg_brightness <= high:
                score += 1.0
            else:
                distance = min(
                    abs(metrics.avg_brightness - low),
                    abs(metrics.avg_brightness - high)
                )
                if distance < 0.15:
                    score += 0.5
            count += 1
        
        # 3. SNR (Signal-to-Noise Ratio)
        if 'snr' in thresholds:
            low, high = thresholds['snr']
            if low <= metrics.avg_snr <= high:
                score += 1.0
            else:
                # Для SNR толерантность 5 dB
                distance = min(
                    abs(metrics.avg_snr - low),
                    abs(metrics.avg_snr - high)
                )
                if distance < 5:
                    score += 0.5
            count += 1
        
        # 4. Резкость (для SAR - характерный признак speckle)
        if 'sharpness' in thresholds:
            low, high = thresholds['sharpness']
            if low <= metrics.avg_sharpness <= high:
                score += 1.0
            count += 1
        
        # 5. Динамический диапазон (для медицинских снимков)
        if 'dynamic_range' in thresholds:
            # Рассчитываем как разницу между 95-м и 5-м перцентилями
            # (если такие данные есть в метриках)
            if hasattr(metrics, 'dynamic_range'):
                low, high = thresholds['dynamic_range']
                if low <= metrics.dynamic_range <= high:
                    score += 1.0
                count += 1
        
        # 6. Размытие (для infrared)
        if 'blur_ratio' in thresholds:
            blur_ratio = metrics.blur_count / metrics.num_images if metrics.num_images > 0 else 0
            low, high = thresholds['blur_ratio']
            if low <= blur_ratio <= high:
                score += 1.0
            count += 1
        
        # 7. Биmodal гистограмма (для microscopy)
        if 'bimodal' in thresholds and thresholds['bimodal']:
            # Проверяем наличие двух пиков в распределении яркости
            # Это можно определить по низкой однородности
            if hasattr(metrics, 'homogeneity'):
                if metrics.homogeneity < 0.7:  # Низкая однородность = возможная биmodальность
                    score += 0.7
            count += 1
        
        return score / count if count > 0 else 0.0
    
    def print_classification_report(self, classification: Dict):
        """
        Красиво печатает результаты классификации
        
        Args:
            classification: Результат classify()
        """
        print("\n" + "="*70)
        print("КЛАССИФИКАЦИЯ ТИПА ДАТАСЕТА")
        print("="*70)

        print(f"\n  Тип: {classification['modality'].upper()}")
        print(f"  Уверенность: {classification['confidence'] * 100:.1f}%")
        print(f"  Описание: {classification['description']}")
        print(f"  Источник: {classification['source']}")

        is_color = classification.get('is_color', None)
        if is_color is not None:
            color_label = "цветной" if is_color else "grayscale"
            micd = classification.get('color_diversity', 0.0)
            print(f"  Цветность: {color_label} (MICD={micd:.1f})")
            if is_color:
                print(f"  (SAR и рентген исключены — физически grayscale-модальности)")

        print(f"\n  Оценки по всем типам:")
        sorted_scores = sorted(
            classification['all_scores'].items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        for modality, score in sorted_scores:
            bar_length = int(score * 30)
            bar = "█" * bar_length + "░" * (30 - bar_length)
            print(f"   {modality:15s} {bar} {score*100:5.1f}%")
        
        print("="*70)
    
    @staticmethod
    def get_modality_characteristics(modality: str) -> Dict:
        """
        Возвращает характеристики конкретного типа изображений
        
        Args:
            modality: Название типа ('sar', 'medical_xray', и т.д.)
            
        Returns:
            dict: Характеристики из THRESHOLDS
        """
        return ImageModalityClassifier.THRESHOLDS.get(modality, {})


def demonstrate_classifier():
    """Демонстрация работы классификатора"""
    
    # Создаём mock объект для примера
    class MockMetrics:
        def __init__(self, contrast, brightness, snr, sharpness=1000, 
                     blur_count=0, num_images=100):
            self.avg_contrast = contrast
            self.avg_brightness = brightness
            self.avg_snr = snr
            self.avg_sharpness = sharpness
            self.blur_count = blur_count
            self.num_images = num_images
    
    classifier = ImageModalityClassifier()
    
    print("\n" + "="*70)
    print("ДЕМОНСТРАЦИЯ КЛАССИФИКАТОРА")
    print("="*70)
    
    # Пример 1: SAR изображение
    print("\n1. SAR датасет:")
    sar_metrics = MockMetrics(
        contrast=0.92,
        brightness=0.15,
        snr=12,
        sharpness=4500
    )
    sar_result = classifier.classify(sar_metrics)
    classifier.print_classification_report(sar_result)
    
    # Пример 2: Natural Photo
    print("\n2. Обычные фотографии:")
    natural_metrics = MockMetrics(
        contrast=0.55,
        brightness=0.50,
        snr=25
    )
    natural_result = classifier.classify(natural_metrics)
    classifier.print_classification_report(natural_result)
    
    # Пример 3: Infrared
    print("\n3. Тепловизионные изображения:")
    infrared_metrics = MockMetrics(
        contrast=0.25,
        brightness=0.40,
        snr=18,
        blur_count=60,
        num_images=100
    )
    infrared_result = classifier.classify(infrared_metrics)
    classifier.print_classification_report(infrared_result)


if __name__ == '__main__':
    # Запускаем демонстрацию
    demonstrate_classifier()
