"""
Применение предобработки к датасету с автоматическим сравнением метрик

Workflow:
1. Анализ оригинального датасета
2. Рекомендация стратегии предобработки
3. Применение предобработки
4. Анализ результата
5. Сравнение метрик ДО/ПОСЛЕ
6. Научное обоснование эффективности
"""

from pathlib import Path
from Utils.image_analyzer import UniversalImageAnalyzer
from Utils.preprocessing_selector import AdaptivePreprocessingSelector
from Preprocessing.applicator import DatasetPreprocessor
from Data.Datasets.dataset_work import get_dataset_path, list_available_datasets
import json
from datetime import datetime


def print_comparison_report(original_metrics, preprocessed_metrics, comparison_data):
    """
    Красивый вывод сравнения метрик

    Args:
        original_metrics: DatasetMetrics до обработки
        preprocessed_metrics: DatasetMetrics после обработки
        comparison_data: dict с результатами сравнения
    """
    print("\n" + "=" * 80)
    print("📊 СРАВНЕНИЕ МЕТРИК ДО И ПОСЛЕ ПРЕДОБРАБОТКИ")
    print("=" * 80)

    # 1. Шум (SNR)
    snr_data = comparison_data['snr']
    print(f"\n🔊 ШУМОПОДАВЛЕНИЕ (Signal-to-Noise Ratio):")
    print(f"   До обработки:  {snr_data['before']:.2f} dB")
    print(f"   После:         {snr_data['after']:.2f} dB")

    if snr_data['improvement_db'] > 0:
        print(f"   ✅ Улучшение:  +{snr_data['improvement_db']:.2f} dB ({snr_data['improvement_pct']:+.1f}%)")
    elif snr_data['improvement_db'] < 0:
        print(f"   ⚠️  Ухудшение:  {snr_data['improvement_db']:.2f} dB ({snr_data['improvement_pct']:+.1f}%)")
    else:
        print(f"   ➖ Без изменений")

    # 2. Контраст
    contrast_data = comparison_data['contrast']
    print(f"\n📈 КОНТРАСТ:")
    print(f"   До обработки:  {contrast_data['before']:.4f}")
    print(f"   После:         {contrast_data['after']:.4f}")

    if contrast_data['improvement'] > 0.01:
        print(f"   ✅ Улучшение:  +{contrast_data['improvement']:.4f} ({contrast_data['improvement_pct']:+.1f}%)")
    elif contrast_data['improvement'] < -0.01:
        print(f"   ⚠️  Ухудшение:  {contrast_data['improvement']:.4f} ({contrast_data['improvement_pct']:+.1f}%)")
    else:
        print(f"   ➖ Без значительных изменений")

    # 3. Яркость
    brightness_data = comparison_data['brightness']
    print(f"\n💡 ЯРКОСТЬ (оптимум = 0.5):")
    print(f"   До обработки:  {brightness_data['before']:.4f} (ошибка: {brightness_data['error_before']:.4f})")
    print(f"   После:         {brightness_data['after']:.4f} (ошибка: {brightness_data['error_after']:.4f})")

    if brightness_data['improvement'] > 0.01:
        print(f"   ✅ Стала ближе к оптимуму на {brightness_data['improvement']:.4f}")
    elif brightness_data['improvement'] < -0.01:
        print(f"   ⚠️  Стала дальше от оптимума на {abs(brightness_data['improvement']):.4f}")
    else:
        print(f"   ➖ Без значительных изменений")

    # 4. Резкость
    sharpness_data = comparison_data['sharpness']
    print(f"\n🔪 РЕЗКОСТЬ:")
    print(f"   До обработки:  {sharpness_data['before']:.2f}")
    print(f"   После:         {sharpness_data['after']:.2f}")

    if sharpness_data['improvement'] > 5:
        print(f"   ✅ Улучшение:  +{sharpness_data['improvement']:.2f} ({sharpness_data['improvement_pct']:+.1f}%)")
    elif sharpness_data['improvement'] < -5:
        print(f"   ⚠️  Ухудшение:  {sharpness_data['improvement']:.2f} ({sharpness_data['improvement_pct']:+.1f}%)")
    else:
        print(f"   ➖ Без значительных изменений")

    # 5. Однородность
    homogeneity_data = comparison_data['homogeneity']
    print(f"\n🎯 ОДНОРОДНОСТЬ ДАТАСЕТА:")
    print(f"   До обработки:  {homogeneity_data['before']:.4f}")
    print(f"   После:         {homogeneity_data['after']:.4f}")

    if homogeneity_data['improvement'] > 0.05:
        print(f"   ✅ Улучшение:  +{homogeneity_data['improvement']:.4f}")
        print(f"      → Датасет стал более однородным")
    elif homogeneity_data['improvement'] < -0.05:
        print(f"   ⚠️  Ухудшение:  {homogeneity_data['improvement']:.4f}")
    else:
        print(f"   ➖ Без значительных изменений")

    # 6. Проблемные изображения
    problems = comparison_data['problematic_images']
    print(f"\n📊 ИСПРАВЛЕНИЕ ПРОБЛЕМНЫХ ИЗОБРАЖЕНИЙ:")

    high_noise_fixed = problems['high_noise']['fixed']
    print(f"\n   Высокий шум (SNR < 15 dB):")
    print(f"      До:  {problems['high_noise']['before']} изображений")
    print(f"      После: {problems['high_noise']['after']} изображений")
    if high_noise_fixed > 0:
        print(f"      ✅ Исправлено: {high_noise_fixed} изображений")
    elif high_noise_fixed < 0:
        print(f"      ⚠️  Появилось ещё: {abs(high_noise_fixed)} изображений")

    low_contrast_fixed = problems['low_contrast']['fixed']
    print(f"\n   Низкий контраст:")
    print(f"      До:  {problems['low_contrast']['before']} изображений")
    print(f"      После: {problems['low_contrast']['after']} изображений")
    if low_contrast_fixed > 0:
        print(f"      ✅ Исправлено: {low_contrast_fixed} изображений")
    elif low_contrast_fixed < 0:
        print(f"      ⚠️  Появилось ещё: {abs(low_contrast_fixed)} изображений")

    # 7. Общая оценка
    print("\n" + "=" * 80)
    print("📈 ОБЩАЯ ОЦЕНКА ЭФФЕКТИВНОСТИ")
    print("=" * 80)

    positive_effects = []
    negative_effects = []

    if snr_data['improvement_db'] > 1.0:
        positive_effects.append(f"SNR улучшен на {snr_data['improvement_db']:.1f} dB")
    elif snr_data['improvement_db'] < -1.0:
        negative_effects.append(f"SNR ухудшился на {abs(snr_data['improvement_db']):.1f} dB")

    if contrast_data['improvement_pct'] > 10:
        positive_effects.append(f"Контраст улучшен на {contrast_data['improvement_pct']:.1f}%")
    elif contrast_data['improvement_pct'] < -10:
        negative_effects.append(f"Контраст ухудшился на {abs(contrast_data['improvement_pct']):.1f}%")

    if brightness_data['improvement'] > 0.05:
        positive_effects.append(f"Яркость стала ближе к оптимуму")

    if homogeneity_data['improvement'] > 0.05:
        positive_effects.append(f"Однородность датасета повышена на {homogeneity_data['improvement']:.2f}")

    if high_noise_fixed > 0:
        positive_effects.append(f"Исправлено {high_noise_fixed} шумных изображений")

    if low_contrast_fixed > 0:
        positive_effects.append(f"Исправлено {low_contrast_fixed} изображений с низким контрастом")

    if positive_effects:
        print("\n✅ ПОЛОЖИТЕЛЬНЫЕ ЭФФЕКТЫ:")
        for effect in positive_effects:
            print(f"   • {effect}")

    if negative_effects:
        print("\n⚠️  НЕГАТИВНЫЕ ЭФФЕКТЫ:")
        for effect in negative_effects:
            print(f"   • {effect}")

    if not positive_effects and not negative_effects:
        print("\n➖ Значительных изменений не обнаружено")

    # 8. Рекомендация
    print("\n" + "=" * 80)

    if len(positive_effects) >= 2 and len(negative_effects) == 0:
        print("🎉 РЕКОМЕНДАЦИЯ: Предобработка эффективна! Используйте этот датасет для обучения.")
    elif len(positive_effects) > len(negative_effects):
        print("✅ РЕКОМЕНДАЦИЯ: Предобработка улучшила качество датасета.")
    elif len(negative_effects) > len(positive_effects):
        print("⚠️  РЕКОМЕНДАЦИЯ: Предобработка ухудшила некоторые метрики. Рассмотрите другие методы.")
    else:
        print("➖ РЕКОМЕНДАЦИЯ: Эффект предобработки незначителен.")

    print("=" * 80)


def calculate_comparison_metrics(original_metrics, preprocessed_metrics):
    """
    Рассчитывает метрики сравнения

    Returns:
        dict с результатами сравнения
    """
    comparison = {}

    # 1. SNR
    orig_snr = original_metrics.avg_snr
    prep_snr = preprocessed_metrics.avg_snr
    snr_improvement = prep_snr - orig_snr
    snr_improvement_pct = (snr_improvement / abs(orig_snr)) * 100 if orig_snr != 0 else 0

    comparison['snr'] = {
        'before': orig_snr,
        'after': prep_snr,
        'improvement_db': snr_improvement,
        'improvement_pct': snr_improvement_pct
    }

    # 2. Контраст
    orig_contrast = original_metrics.avg_contrast
    prep_contrast = preprocessed_metrics.avg_contrast
    contrast_improvement = prep_contrast - orig_contrast
    contrast_improvement_pct = (contrast_improvement / orig_contrast) * 100 if orig_contrast != 0 else 0

    comparison['contrast'] = {
        'before': orig_contrast,
        'after': prep_contrast,
        'improvement': contrast_improvement,
        'improvement_pct': contrast_improvement_pct
    }

    # 3. Яркость
    orig_brightness = original_metrics.avg_brightness
    prep_brightness = preprocessed_metrics.avg_brightness

    orig_brightness_error = abs(orig_brightness - 0.5)
    prep_brightness_error = abs(prep_brightness - 0.5)
    brightness_improvement = orig_brightness_error - prep_brightness_error

    comparison['brightness'] = {
        'before': orig_brightness,
        'after': prep_brightness,
        'error_before': orig_brightness_error,
        'error_after': prep_brightness_error,
        'improvement': brightness_improvement
    }

    # 4. Резкость
    orig_sharpness = original_metrics.avg_sharpness
    prep_sharpness = preprocessed_metrics.avg_sharpness
    sharpness_improvement = prep_sharpness - orig_sharpness
    sharpness_improvement_pct = (sharpness_improvement / orig_sharpness) * 100 if orig_sharpness != 0 else 0

    comparison['sharpness'] = {
        'before': orig_sharpness,
        'after': prep_sharpness,
        'improvement': sharpness_improvement,
        'improvement_pct': sharpness_improvement_pct
    }

    # 5. Однородность
    orig_homogeneity = original_metrics.homogeneity
    prep_homogeneity = preprocessed_metrics.homogeneity
    homogeneity_improvement = prep_homogeneity - orig_homogeneity

    comparison['homogeneity'] = {
        'before': orig_homogeneity,
        'after': prep_homogeneity,
        'improvement': homogeneity_improvement
    }

    # 6. Проблемные изображения
    orig_high_noise = original_metrics.noise_distribution.get('high', 0)
    prep_high_noise = preprocessed_metrics.noise_distribution.get('high', 0)

    orig_low_contrast = original_metrics.contrast_distribution.get('low', 0)
    prep_low_contrast = preprocessed_metrics.contrast_distribution.get('low', 0)

    comparison['problematic_images'] = {
        'high_noise': {
            'before': orig_high_noise,
            'after': prep_high_noise,
            'fixed': orig_high_noise - prep_high_noise
        },
        'low_contrast': {
            'before': orig_low_contrast,
            'after': prep_low_contrast,
            'fixed': orig_low_contrast - prep_low_contrast
        }
    }

    return comparison


def apply_preprocessing_with_comparison():
    """
    Полный цикл применения предобработки с автоматическим сравнением
    """
    print("=" * 80)
    print("ПРИМЕНЕНИЕ ПРЕДОБРАБОТКИ С ОЦЕНКОЙ ЭФФЕКТИВНОСТИ")
    print("=" * 80)

    # 1. Выбор датасета
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

    # 2. Анализ ОРИГИНАЛЬНОГО датасета
    print(f"\n{'='*80}")
    print(f"ШАГ 1: АНАЛИЗ ОРИГИНАЛЬНОГО ДАТАСЕТА '{dataset_name}'")
    print(f"{'='*80}")

    analyzer = UniversalImageAnalyzer()
    print("\n⏳ Анализирую train split...")
    original_metrics, original_image_metrics = analyzer.analyze_dataset(dataset_path, split='train')

    # Показываем краткий отчёт
    print(f"\n📊 Результаты анализа:")
    print(f"   SNR: {original_metrics.avg_snr:.2f} dB")
    print(f"   Контраст: {original_metrics.avg_contrast:.4f}")
    print(f"   Яркость: {original_metrics.avg_brightness:.4f}")
    print(f"   Однородность: {original_metrics.homogeneity:.4f}")

    # 3. Рекомендации
    print(f"\n{'='*80}")
    print(f"ШАГ 2: РЕКОМЕНДАЦИИ ПО ПРЕДОБРАБОТКЕ")
    print(f"{'='*80}")

    if original_metrics.needs_adaptive_preprocessing:
        print(f"\n🎯 Стратегия: АДАПТИВНАЯ (датасет неоднороден)")
        print(f"   Рекомендуемое количество кластеров: {original_metrics.suggested_clusters}")

        selector = AdaptivePreprocessingSelector()
        image_metrics = original_image_metrics

        clustering_result = selector.cluster_and_recommend(image_metrics, original_metrics)

        for cluster_info in clustering_result:
            print(f"\n   Кластер {cluster_info['cluster_id']}:")
            print(f"      Размер: {len(cluster_info['image_indices'])} изображений")
            print(f"      Методы: {', '.join(cluster_info['preprocessing']) if cluster_info['preprocessing'] else 'нет'}")
    else:
        print(f"\n🎯 Стратегия: ГЛОБАЛЬНАЯ (датасет однороден)")
        print(f"   Рекомендуемые методы: {', '.join(original_metrics.recommended_preprocessing)}")

    # 4. Подтверждение
    print(f"\n{'='*80}")
    confirm = input("\n❓ Применить предобработку? (y/n): ")

    if confirm.lower() != 'y':
        print("Отменено.")
        return

    # 5. Название нового датасета
    default_name = f"{dataset_name}_preprocessed"
    new_name = input(f"\nВведите название нового датасета [{default_name}]: ").strip()
    if not new_name:
        new_name = default_name

    # 6. Применение предобработки
    print(f"\n{'='*80}")
    print(f"ШАГ 3: ПРИМЕНЕНИЕ ПРЕДОБРАБОТКИ")
    print(f"{'='*80}")

    preprocessor = DatasetPreprocessor()

    if original_metrics.needs_adaptive_preprocessing:
        print("\n⏳ Применяю адаптивную предобработку...")
        preprocessor.apply_adaptive_preprocessing(
            source_dataset_name=dataset_name,
            target_dataset_name=new_name,
            dataset_metrics=original_metrics,
            image_metrics=image_metrics
        )
    else:
        print("\n⏳ Применяю глобальную предобработку...")
        preprocessor.apply_global_preprocessing(
            source_dataset_name=dataset_name,
            target_dataset_name=new_name,
            methods=original_metrics.recommended_preprocessing
        )

    print(f"\n✅ Предобработка завершена! Новый датасет: {new_name}")

    # 7. Анализ РЕЗУЛЬТАТА
    print(f"\n{'='*80}")
    print(f"ШАГ 4: АНАЛИЗ РЕЗУЛЬТАТА")
    print(f"{'='*80}")

    new_dataset_path = get_dataset_path(new_name)

    print("\n⏳ Анализирую предобработанный датасет...")
    preprocessed_metrics, preprocessed_image_metrics = analyzer.analyze_dataset(new_dataset_path, split='train')

    # 8. СРАВНЕНИЕ
    print(f"\n{'='*80}")
    print(f"ШАГ 5: СРАВНЕНИЕ И ОЦЕНКА ЭФФЕКТИВНОСТИ")
    print(f"{'='*80}")

    comparison_data = calculate_comparison_metrics(original_metrics, preprocessed_metrics)
    print_comparison_report(original_metrics, preprocessed_metrics, comparison_data)

    # 9. Сохранение результатов
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = f"preprocessing_results_{dataset_name}_to_{new_name}_{timestamp}.json"

    results = {
        'timestamp': timestamp,
        'original_dataset': dataset_name,
        'preprocessed_dataset': new_name,
        'strategy': 'adaptive' if original_metrics.needs_adaptive_preprocessing else 'global',
        'original_metrics': {
            'avg_snr': original_metrics.avg_snr,
            'avg_contrast': original_metrics.avg_contrast,
            'avg_brightness': original_metrics.avg_brightness,
            'homogeneity': original_metrics.homogeneity
        },
        'preprocessed_metrics': {
            'avg_snr': preprocessed_metrics.avg_snr,
            'avg_contrast': preprocessed_metrics.avg_contrast,
            'avg_brightness': preprocessed_metrics.avg_brightness,
            'homogeneity': preprocessed_metrics.homogeneity
        },
        'comparison': comparison_data
    }

    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n💾 Результаты сохранены в: {results_file}")
    print("\n" + "=" * 80)
    print("🎉 ГОТОВО!")
    print("=" * 80)


if __name__ == "__main__":
    apply_preprocessing_with_comparison()