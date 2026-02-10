"""
Применение предобработки с автоматическим определением типа датасета

КРИТИЧЕСКИЕ ИЗМЕНЕНИЯ (v2.0):
- Автоматическая классификация типа датасета
- Вывод правил предобработки для типа
- Блокировка опасных методов

Автор: Система адаптивной предобработки
Дата: 2025
"""

from pathlib import Path
from typing import Dict

from Data.Datasets.dataset_work import get_dataset_path, list_available_datasets
from Utils.image_analyzer import UniversalImageAnalyzer
from Utils.modality_classifier import ImageModalityClassifier
from Utils.preprocessing_rules import PreprocessingRules
from Utils.preprocessing_selector import AdaptivePreprocessingSelector
from Preprocessing.applicator import DatasetPreprocessor


def apply_preprocessing_with_modality_detection():
    """
    Полный цикл применения предобработки с учётом типа датасета
    
    НОВОЕ в v2.0:
    1. Определяет тип датасета (SAR/Medical/Natural/Infrared/Microscopy)
    2. Показывает правила для этого типа
    3. Блокирует неподходящие методы
    4. Применяет предобработку с адаптированными параметрами
    """
    
    print("=" * 80)
    print("ПРИМЕНЕНИЕ ПРЕДОБРАБОТКИ С ОПРЕДЕЛЕНИЕМ ТИПА ДАТАСЕТА")
    print("=" * 80)

    # ========================================================================
    # ШАГ 1: ВЫБОР ДАТАСЕТА
    # ========================================================================
    datasets = list_available_datasets(verbose=False)

    print("\nДоступные датасеты:")
    for i, name in enumerate(datasets):
        print(f"  [{i}] {name}")

    while True:
        try:
            idx = int(input("\nВведите номер датасета: "))
            if 0 <= idx < len(datasets):
                dataset_name = datasets[idx]
                break
            else:
                print("Неверный номер!")
        except ValueError:
            print("Введите число!")

    dataset_path = get_dataset_path(dataset_name)
    print(f"\n✅ Выбран датасет: {dataset_name}")

    # ========================================================================
    # ШАГ 2: АНАЛИЗ ДАТАСЕТА
    # ========================================================================
    print(f"\n{'='*80}")
    print(f"ШАГ 2: АНАЛИЗ ДАТАСЕТА")
    print(f"{'='*80}")

    analyzer = UniversalImageAnalyzer(verbose=False)
    
    print("\n⏳ Анализирую изображения...")
    dataset_metrics, image_metrics = analyzer.analyze_dataset(
        dataset_path, 
        split='train'
    )

    # Краткий отчёт
    print(f"\n📊 Базовые метрики:")
    print(f"   Изображений: {dataset_metrics.num_images}")
    print(f"   SNR: {dataset_metrics.avg_snr:.2f} dB")
    print(f"   Контраст: {dataset_metrics.avg_contrast:.4f}")
    print(f"   Яркость: {dataset_metrics.avg_brightness:.4f}")
    print(f"   Резкость: {dataset_metrics.avg_sharpness:.0f}")

    # ========================================================================
    # ШАГ 3: КЛАССИФИКАЦИЯ ТИПА ДАТАСЕТА 🔥 НОВОЕ!
    # ========================================================================
    print(f"\n{'='*80}")
    print(f"ШАГ 3: КЛАССИФИКАЦИЯ ТИПА ДАТАСЕТА")
    print(f"{'='*80}")

    classifier = ImageModalityClassifier()
    modality_info = classifier.classify(dataset_metrics)
    
    # Красиво показываем результат
    classifier.print_classification_report(modality_info)

    # ========================================================================
    # ШАГ 4: ПРАВИЛА ПРЕДОБРАБОТКИ ДЛЯ ТИПА 🔥 НОВОЕ!
    # ========================================================================
    print(f"\n{'='*80}")
    print(f"ШАГ 4: ПРАВИЛА ПРЕДОБРАБОТКИ ДЛЯ ТИПА '{modality_info['modality'].upper()}'")
    print(f"{'='*80}")
    
    PreprocessingRules.print_rules_summary(modality_info['modality'])

    # ========================================================================
    # ШАГ 5: ВЫБОР СТРАТЕГИИ С УЧЁТОМ ТИПА 🔥 ИЗМЕНЕНО!
    # ========================================================================
    print(f"\n{'='*80}")
    print(f"ШАГ 5: ВЫБОР СТРАТЕГИИ ПРЕДОБРАБОТКИ")
    print(f"{'='*80}")

    # Создаём селектор С ИНФОРМАЦИЕЙ О ТИПЕ
    selector = AdaptivePreprocessingSelector(
        analyzer=analyzer,
        modality_info=modality_info  # 🔥 Передаём тип!
    )
    
    strategy = selector.select_strategy(dataset_path, split='train')
    
    # Красиво показываем стратегию
    selector.print_strategy_info(strategy)

    # ========================================================================
    # ШАГ 6: ПОДТВЕРЖДЕНИЕ
    # ========================================================================
    print(f"\n{'='*80}")
    confirm = input("\n❓ Применить предобработку с учётом типа датасета? (y/n): ")

    if confirm.lower() != 'y':
        print("Отменено.")
        return

    # ========================================================================
    # ШАГ 7: НАЗВАНИЕ НОВОГО ДАТАСЕТА
    # ========================================================================
    default_name = f"{dataset_name}_preprocessed_{modality_info['modality']}"
    new_name = input(f"\nВведите название нового датасета [{default_name}]: ").strip()
    if not new_name:
        new_name = default_name

    # ========================================================================
    # ШАГ 8: ПРИМЕНЕНИЕ ПРЕДОБРАБОТКИ
    # ========================================================================
    print(f"\n{'='*80}")
    print(f"ШАГ 8: ПРИМЕНЕНИЕ ПРЕДОБРАБОТКИ")
    print(f"{'='*80}")

    preprocessor = DatasetPreprocessor()

    if strategy['strategy'] == 'adaptive':
        print("\n⏳ Применяю адаптивную предобработку...")
        
        # Получаем параметры методов для типа датасета
        params = _build_params_for_modality(modality_info, selector)
        
        preprocessor.apply_adaptive_preprocessing(
            source_dataset=dataset_name,
            target_dataset=new_name,
            clusters=strategy['clusters'],
            image_metrics=image_metrics,
            params=params  # 🔥 Передаём адаптированные параметры
        )
    else:
        print("\n⏳ Применяю глобальную предобработку...")
        
        # Получаем параметры методов для типа датасета
        params = _build_params_for_modality(modality_info, selector)
        
        preprocessor.apply_global_preprocessing(
            source_dataset=dataset_name,
            target_dataset=new_name,
            methods=strategy['methods'],
            params=params  # 🔥 Передаём адаптированные параметры
        )

    print(f"\n✅ Предобработка завершена!")
    print(f"   Новый датасет: {new_name}")
    print(f"   Тип датасета: {modality_info['modality']}")

    # ========================================================================
    # ШАГ 9: АНАЛИЗ РЕЗУЛЬТАТА
    # ========================================================================
    print(f"\n{'='*80}")
    print(f"ШАГ 9: АНАЛИЗ РЕЗУЛЬТАТА")
    print(f"{'='*80}")

    new_dataset_path = get_dataset_path(new_name)
    
    print("\n⏳ Анализирую предобработанный датасет...")
    preprocessed_metrics, _ = analyzer.analyze_dataset(new_dataset_path, split='train')

    # ========================================================================
    # ШАГ 10: СРАВНЕНИЕ
    # ========================================================================
    print(f"\n{'='*80}")
    print(f"ШАГ 10: СРАВНЕНИЕ ДО И ПОСЛЕ")
    print(f"{'='*80}")

    _print_comparison(dataset_metrics, preprocessed_metrics, modality_info)


def _build_params_for_modality(modality_info: Dict, selector: AdaptivePreprocessingSelector) -> Dict:
    """
    Строит словарь параметров для методов на основе типа датасета
    
    Returns:
        dict: {'denoise': {...}, 'contrast_enhancement': {...}, ...}
    """
    params = {}
    
    methods = ['denoise', 'contrast_enhancement', 'brightness_correction', 'sharpening']
    
    for method in methods:
        method_params = selector.get_method_params(method)
        if method_params:
            params[method] = method_params
    
    return params


def _print_comparison(
    original_metrics, 
    preprocessed_metrics, 
    modality_info: Dict
):
    """
    Печатает сравнение метрик до и после предобработки
    
    Args:
        original_metrics: Метрики оригинального датасета
        preprocessed_metrics: Метрики предобработанного датасета
        modality_info: Информация о типе датасета
    """
    
    print(f"\n📊 Сравнение метрик:")
    print(f"   Тип датасета: {modality_info['modality'].upper()}")
    print()
    
    # SNR
    snr_before = original_metrics.avg_snr
    snr_after = preprocessed_metrics.avg_snr
    snr_change = snr_after - snr_before
    
    print(f"   SNR:")
    print(f"      До:      {snr_before:6.2f} dB")
    print(f"      После:   {snr_after:6.2f} dB")
    print(f"      Изменение: {snr_change:+6.2f} dB {'✓' if snr_change > 0 else ''}")
    
    # Контраст
    contrast_before = original_metrics.avg_contrast
    contrast_after = preprocessed_metrics.avg_contrast
    contrast_change = contrast_after - contrast_before
    
    print(f"\n   Контраст:")
    print(f"      До:      {contrast_before:.4f}")
    print(f"      После:   {contrast_after:.4f}")
    print(f"      Изменение: {contrast_change:+.4f} {'✓' if contrast_change > 0 else ''}")
    
    # Яркость
    brightness_before = original_metrics.avg_brightness
    brightness_after = preprocessed_metrics.avg_brightness
    brightness_change = brightness_after - brightness_before
    
    print(f"\n   Яркость:")
    print(f"      До:      {brightness_before:.4f}")
    print(f"      После:   {brightness_after:.4f}")
    print(f"      Изменение: {brightness_change:+.4f}")
    
    # Однородность
    if hasattr(original_metrics, 'homogeneity'):
        homog_before = original_metrics.homogeneity
        homog_after = preprocessed_metrics.homogeneity
        homog_change = homog_after - homog_before
        
        print(f"\n   Однородность:")
        print(f"      До:      {homog_before:.4f}")
        print(f"      После:   {homog_after:.4f}")
        print(f"      Изменение: {homog_change:+.4f} {'✓' if homog_change > 0 else ''}")
    
    # Проблемные изображения
    if hasattr(original_metrics, 'noise_distribution'):
        high_noise_before = original_metrics.noise_distribution.get('high', 0)
        high_noise_after = preprocessed_metrics.noise_distribution.get('high', 0)
        
        print(f"\n   Изображения с высоким шумом:")
        print(f"      До:      {high_noise_before}")
        print(f"      После:   {high_noise_after}")
        print(f"      Исправлено: {high_noise_before - high_noise_after} {'✓' if high_noise_after < high_noise_before else ''}")
    
    # Важное предупреждение для некоторых типов
    if modality_info['modality'] in ['sar', 'medical_xray', 'microscopy']:
        print(f"\n⚠️  ВАЖНО для типа '{modality_info['modality']}':")
        
        if modality_info['modality'] == 'sar':
            print(f"   Яркость НЕ ДОЛЖНА сильно меняться (физическая характеристика)")
            if abs(brightness_change) > 0.1:
                print(f"   ❗ ВНИМАНИЕ: Яркость изменилась на {brightness_change:+.2f} - это может быть проблемой!")
        
        elif modality_info['modality'] == 'medical_xray':
            print(f"   Диагностическая информация должна сохраниться")
            
        elif modality_info['modality'] == 'microscopy':
            print(f"   Bimodal распределение (фон/объекты) должно сохраниться")


if __name__ == '__main__':
    apply_preprocessing_with_modality_detection()
