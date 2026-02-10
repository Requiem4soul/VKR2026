"""
Адаптивный выбор стратегии предобработки с учётом типа датасета

КРИТИЧЕСКИЕ ИЗМЕНЕНИЯ (v2.0):
- Добавлена классификация типа датасета (SAR/Medical/Natural/Infrared/Microscopy)
- Учитываются правила предобработки для каждого типа
- Блокируются опасные методы (например, brightness_correction для SAR)
- Параметры методов адаптируются под тип датасета

Автор: Система адаптивной предобработки
Дата: 2025
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from Utils.image_analyzer import UniversalImageAnalyzer, ImageMetrics


class AdaptivePreprocessingSelector:
    """
    Выбирает стратегию предобработки на основе анализа датасета
    
    НОВОЕ в v2.0:
    - Принимает информацию о типе датасета (modality_info)
    - Применяет правила из PreprocessingRules
    - Блокирует неподходящие методы
    """

    def __init__(
        self, 
        analyzer: UniversalImageAnalyzer,
        modality_info: Optional[Dict] = None
    ):
        """
        Args:
            analyzer: Анализатор изображений
            modality_info: Информация о типе датасета из ImageModalityClassifier
                          {'modality': 'sar', 'confidence': 0.95, ...}
                          Если None - используется 'natural_photo' по умолчанию
        """
        self.analyzer = analyzer
        self.modality_info = modality_info or {
            'modality': 'natural_photo',
            'confidence': 1.0,
            'source': 'default'
        }

    def select_strategy(
            self,
            dataset_path: Path,
            split: str = 'train'
    ) -> Dict:
        """
        Определяет оптимальную стратегию предобработки
        
        Returns:
            dict: Стратегия с учётом типа датасета
        """
        dataset_metrics, image_metrics = self.analyzer.analyze_dataset(
            dataset_path,
            split=split
        )

        # Определяем стратегию (глобальная vs адаптивная)
        if not dataset_metrics.needs_adaptive_preprocessing:
            # ГЛОБАЛЬНАЯ стратегия
            methods = self._filter_methods_by_modality(
                dataset_metrics.recommended_global_preprocessing
            )
            
            return {
                'strategy': 'global',
                'methods': methods,
                'dataset_metrics': dataset_metrics,
                'modality': self.modality_info['modality'],
                'modality_confidence': self.modality_info['confidence']
            }

        else:
            # АДАПТИВНАЯ стратегия
            clusters = self._cluster_images(
                image_metrics,
                n_clusters=dataset_metrics.suggested_clusters
            )

            return {
                'strategy': 'adaptive',
                'n_clusters': dataset_metrics.suggested_clusters,
                'clusters': clusters,
                'dataset_metrics': dataset_metrics,
                'image_metrics': image_metrics,
                'modality': self.modality_info['modality'],
                'modality_confidence': self.modality_info['confidence']
            }

    def _cluster_images(
            self,
            image_metrics: List[ImageMetrics],
            n_clusters: int
    ) -> Dict:
        """
        Кластеризует изображения по характеристикам
        
        ИЗМЕНЕНИЕ: Рекомендации для кластеров фильтруются по типу датасета
        """
        features = np.array([
            [
                m.snr_db,
                m.global_contrast,
                m.mean_brightness,
                m.sharpness_score
            ]
            for m in image_metrics
        ])

        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(features_scaled)

        clusters = {}
        for cluster_id in range(n_clusters):
            cluster_indices = np.where(labels == cluster_id)[0]
            cluster_metrics = [image_metrics[i] for i in cluster_indices]

            avg_snr = np.mean([m.snr_db for m in cluster_metrics])
            avg_contrast = np.mean([m.global_contrast for m in cluster_metrics])
            avg_brightness = np.mean([m.mean_brightness for m in cluster_metrics])

            # Рекомендуем методы ДЛЯ КЛАСТЕРА
            cluster_preprocessing = self._recommend_for_cluster(cluster_metrics)
            
            # ⚠️ НОВОЕ: Фильтруем по типу датасета
            cluster_preprocessing = self._filter_methods_by_modality(cluster_preprocessing)

            clusters[cluster_id] = {
                'size': len(cluster_metrics),
                'characteristics': {
                    'avg_snr': avg_snr,
                    'avg_contrast': avg_contrast,
                    'avg_brightness': avg_brightness
                },
                'preprocessing': cluster_preprocessing,
                'image_indices': cluster_indices.tolist()
            }

        return clusters

    def _recommend_for_cluster(
            self,
            cluster_metrics: List[ImageMetrics]
    ) -> List[str]:
        """
        Рекомендует предобработку для кластера БЕЗ учёта типа датасета
        (фильтрация происходит позже в _filter_methods_by_modality)
        """
        high_noise_ratio = sum(1 for m in cluster_metrics if m.noise_level == 'high') / len(cluster_metrics)
        low_contrast_ratio = sum(1 for m in cluster_metrics if m.contrast_level == 'low') / len(cluster_metrics)
        dark_ratio = sum(1 for m in cluster_metrics if m.brightness_level == 'dark') / len(cluster_metrics)
        blur_ratio = sum(1 for m in cluster_metrics if m.blur_detected) / len(cluster_metrics)

        methods = []
        
        if high_noise_ratio > 0.5:
            methods.append('denoise')
        
        if low_contrast_ratio > 0.5:
            methods.append('contrast_enhancement')
        
        if dark_ratio > 0.5:
            methods.append('brightness_correction')
        
        if blur_ratio > 0.5:
            methods.append('sharpening')

        return methods
    
    def _filter_methods_by_modality(self, methods: List[str]) -> List[str]:
        """
        🔥 НОВАЯ ФУНКЦИЯ: Фильтрует методы согласно правилам типа датасета
        
        Блокирует опасные методы:
        - brightness_correction для SAR (искажает физическую информацию)
        - brightness_correction для medical_xray (диагностически важна)
        - brightness_correction для microscopy (разрушает bimodal распределение)
        
        Args:
            methods: Список рекомендованных методов
            
        Returns:
            list: Отфильтрованный список методов
        """
        # Импортируем правила
        from Utils.preprocessing_rules import PreprocessingRules
        
        modality = self.modality_info['modality']
        filtered_methods = []
        blocked_methods = []
        
        for method in methods:
            if PreprocessingRules.is_method_allowed(modality, method):
                filtered_methods.append(method)
            else:
                blocked_methods.append(method)
        
        # Логируем заблокированные методы
        if blocked_methods:
            print(f"\n⚠️  ВНИМАНИЕ: Заблокированы методы для типа '{modality}':")
            for method in blocked_methods:
                rationale = PreprocessingRules.get_rationale(modality, method)
                print(f"   ❌ {method}")
                print(f"      Причина: {rationale}")
        
        return filtered_methods
    
    def get_method_params(self, method: str) -> Dict:
        """
        🔥 НОВАЯ ФУНКЦИЯ: Возвращает параметры метода для типа датасета
        
        Args:
            method: Название метода ('denoise', 'contrast_enhancement', и т.д.)
            
        Returns:
            dict: Параметры для данного метода и типа датасета
        """
        from Utils.preprocessing_rules import PreprocessingRules
        
        modality = self.modality_info['modality']
        return PreprocessingRules.get_method_params(modality, method)
    
    def print_strategy_info(self, strategy: Dict):
        """
        Красиво печатает информацию о выбранной стратегии
        
        Args:
            strategy: Результат select_strategy()
        """
        print(f"\n{'='*70}")
        print(f"СТРАТЕГИЯ ПРЕДОБРАБОТКИ")
        print(f"{'='*70}")
        
        print(f"\n📌 Тип датасета: {strategy['modality'].upper()}")
        print(f"   Уверенность: {strategy['modality_confidence']*100:.1f}%")
        
        if strategy['strategy'] == 'global':
            print(f"\n🎯 Подход: ГЛОБАЛЬНАЯ предобработка")
            print(f"   (все изображения обрабатываются одинаково)")
            
            if strategy['methods']:
                print(f"\n   Методы:")
                for method in strategy['methods']:
                    params = self.get_method_params(method)
                    print(f"      ✓ {method}")
                    if params:
                        print(f"        Параметры: {params}")
            else:
                print(f"\n   Методы: не требуются")
        
        else:
            print(f"\n🎯 Подход: АДАПТИВНАЯ предобработка")
            print(f"   (разные кластеры обрабатываются по-разному)")
            print(f"   Кластеров: {strategy['n_clusters']}")
            
            for cluster_id, cluster_info in strategy['clusters'].items():
                print(f"\n   📦 Кластер {cluster_id}:")
                print(f"      Размер: {cluster_info['size']} изображений")
                print(f"      SNR: {cluster_info['characteristics']['avg_snr']:.1f} dB")
                print(f"      Контраст: {cluster_info['characteristics']['avg_contrast']:.3f}")
                print(f"      Яркость: {cluster_info['characteristics']['avg_brightness']:.3f}")
                
                if cluster_info['preprocessing']:
                    print(f"      Методы:")
                    for method in cluster_info['preprocessing']:
                        params = self.get_method_params(method)
                        print(f"         ✓ {method}")
                        if params:
                            print(f"           Параметры: {params}")
                else:
                    print(f"      Методы: не требуются")
        
        print(f"\n{'='*70}")


# Пример использования
def demonstrate_selector():
    """Демонстрация работы селектора с учётом типа датасета"""
    
    # Создаём mock объект для примера
    class MockAnalyzer:
        pass
    
    # Пример 1: SAR датасет
    print("\n" + "="*70)
    print("ПРИМЕР 1: SAR ДАТАСЕТ")
    print("="*70)
    
    sar_modality_info = {
        'modality': 'sar',
        'confidence': 0.95,
        'source': 'Oliver & Quegan (2004)'
    }
    
    selector_sar = AdaptivePreprocessingSelector(
        analyzer=MockAnalyzer(),
        modality_info=sar_modality_info
    )
    
    # Симулируем рекомендации
    recommended_methods = ['denoise', 'brightness_correction', 'contrast_enhancement']
    filtered_methods = selector_sar._filter_methods_by_modality(recommended_methods)
    
    print(f"\n✅ Рекомендовано: {recommended_methods}")
    print(f"✅ После фильтрации: {filtered_methods}")
    
    # Пример 2: Natural Photo
    print("\n" + "="*70)
    print("ПРИМЕР 2: ОБЫЧНЫЕ ФОТОГРАФИИ")
    print("="*70)
    
    natural_modality_info = {
        'modality': 'natural_photo',
        'confidence': 0.98,
        'source': 'Gonzalez & Woods (2018)'
    }
    
    selector_natural = AdaptivePreprocessingSelector(
        analyzer=MockAnalyzer(),
        modality_info=natural_modality_info
    )
    
    recommended_methods = ['denoise', 'brightness_correction', 'contrast_enhancement']
    filtered_methods = selector_natural._filter_methods_by_modality(recommended_methods)
    
    print(f"\n✅ Рекомендовано: {recommended_methods}")
    print(f"✅ После фильтрации: {filtered_methods}")


if __name__ == '__main__':
    # Запускаем демонстрацию
    demonstrate_selector()
