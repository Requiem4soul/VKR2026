"""
ДЕМОНСТРАЦИЯ: Система определения типа датасета

Этот скрипт показывает как работает новая система на примерах.
Запустите его чтобы убедиться что всё работает правильно.

Место: Положите в корень проекта (рядом с apply_preprocessing.py)

Usage:
    python demo_modality_system.py
"""

import numpy as np
from typing import Dict


def create_mock_metrics(contrast, brightness, snr, sharpness=1000, blur_count=0, num_images=100):
    """Создаёт mock объект с метриками датасета"""
    class MockMetrics:
        def __init__(self):
            self.avg_contrast = contrast
            self.avg_brightness = brightness
            self.avg_snr = snr
            self.avg_sharpness = sharpness
            self.blur_count = blur_count
            self.num_images = num_images

    return MockMetrics()


def demo_1_classification():
    """Демонстрация классификации разных типов датасетов"""
    print("\n" + "="*80)
    print("ДЕМОНСТРАЦИЯ 1: КЛАССИФИКАЦИЯ ТИПОВ ДАТАСЕТОВ")
    print("="*80)

    from Utils.modality_classifier import ImageModalityClassifier

    classifier = ImageModalityClassifier()

    # Пример 1: SAR
    print("\n📡 Пример 1: SAR датасет")
    print("-" * 40)
    sar_metrics = create_mock_metrics(
        contrast=0.92,
        brightness=0.15,
        snr=12,
        sharpness=4500
    )
    sar_result = classifier.classify(sar_metrics)
    classifier.print_classification_report(sar_result)

    # Пример 2: Medical X-ray
    print("\n🏥 Пример 2: Medical X-ray датасет")
    print("-" * 40)
    medical_metrics = create_mock_metrics(
        contrast=0.65,
        brightness=0.25,
        snr=30
    )
    medical_result = classifier.classify(medical_metrics)
    classifier.print_classification_report(medical_result)

    # Пример 3: Natural Photo
    print("\n📷 Пример 3: Natural Photo датасет")
    print("-" * 40)
    natural_metrics = create_mock_metrics(
        contrast=0.55,
        brightness=0.50,
        snr=25
    )
    natural_result = classifier.classify(natural_metrics)
    classifier.print_classification_report(natural_result)

    # Пример 4: Infrared
    print("\n🌡️  Пример 4: Infrared датасет")
    print("-" * 40)
    infrared_metrics = create_mock_metrics(
        contrast=0.25,
        brightness=0.40,
        snr=18,
        blur_count=60,
        num_images=100
    )
    infrared_result = classifier.classify(infrared_metrics)
    classifier.print_classification_report(infrared_result)

    # Пример 5: Microscopy
    print("\n🔬 Пример 5: Microscopy датасет")
    print("-" * 40)
    microscopy_metrics = create_mock_metrics(
        contrast=0.75,
        brightness=0.30,
        snr=15
    )
    microscopy_result = classifier.classify(microscopy_metrics)
    classifier.print_classification_report(microscopy_result)


def demo_2_rules():
    """Демонстрация правил предобработки"""
    print("\n" + "="*80)
    print("ДЕМОНСТРАЦИЯ 2: ПРАВИЛА ПРЕДОБРАБОТКИ")
    print("="*80)

    from Utils.preprocessing_rules import PreprocessingRules

    types = ['sar', 'medical_xray', 'natural_photo', 'infrared', 'microscopy']

    for modality in types:
        PreprocessingRules.print_rules_summary(modality)


def demo_3_filtering():
    """Демонстрация фильтрации методов"""
    print("\n" + "="*80)
    print("ДЕМОНСТРАЦИЯ 3: ФИЛЬТРАЦИЯ МЕТОДОВ ПО ТИПУ")
    print("="*80)

    from Utils.preprocessing_rules import PreprocessingRules

    # Набор методов которые могли бы быть рекомендованы
    all_methods = ['denoise', 'brightness_correction', 'contrast_enhancement', 'sharpening']

    types = ['sar', 'medical_xray', 'natural_photo']

    for modality in types:
        print(f"\n{'='*70}")
        print(f"Тип: {modality.upper()}")
        print(f"{'='*70}")

        print(f"\nРекомендованные методы: {all_methods}")
        print(f"\nПроверка разрешений:")

        allowed = []
        blocked = []

        for method in all_methods:
            is_allowed = PreprocessingRules.is_method_allowed(modality, method)
            if is_allowed:
                allowed.append(method)
                print(f"   ✅ {method:25s} - РАЗРЕШЁН")
            else:
                blocked.append(method)
                rationale = PreprocessingRules.get_rationale(modality, method)
                print(f"   ❌ {method:25s} - ЗАПРЕЩЁН")
                print(f"      Причина: {rationale[:60]}...")

        print(f"\nИтого:")
        print(f"   Разрешено:  {allowed}")
        print(f"   Заблокировано: {blocked}")


def demo_4_parameters():
    """Демонстрация параметров для методов"""
    print("\n" + "="*80)
    print("ДЕМОНСТРАЦИЯ 4: ПАРАМЕТРЫ МЕТОДОВ ДЛЯ РАЗНЫХ ТИПОВ")
    print("="*80)

    from Utils.preprocessing_rules import PreprocessingRules

    method = 'contrast_enhancement'
    types = ['sar', 'medical_xray', 'natural_photo', 'infrared', 'microscopy']

    print(f"\nМетод: {method.upper()}")
    print(f"{'='*70}")

    for modality in types:
        params = PreprocessingRules.get_method_params(modality, method)

        if params:
            print(f"\n{modality:15s}: {params}")
        else:
            print(f"\n{modality:15s}: (метод запрещён или параметры не определены)")

    # Ещё один пример - denoise
    print(f"\n\nМетод: DENOISE")
    print(f"{'='*70}")

    for modality in types:
        params = PreprocessingRules.get_method_params(modality, 'denoise')

        if params:
            print(f"\n{modality:15s}: {params}")


def demo_5_comparison():
    """Сравнение обработки одного датасета разными способами"""
    print("\n" + "="*80)
    print("ДЕМОНСТРАЦИЯ 5: СРАВНЕНИЕ ПОДХОДОВ")
    print("="*80)

    print("\nСценарий: У нас есть SAR датасет")
    print("Сравним что произойдёт при разных подходах:")

    print("\n" + "-"*70)
    print("ПОДХОД 1: БЕЗ УЧЁТА ТИПА (старая система)")
    print("-"*70)
    print("\nРекомендованные методы:")
    print("   1. denoise")
    print("   2. brightness_correction  ← ОПАСНО для SAR!")
    print("   3. contrast_enhancement   ← Не нужно для SAR")

    print("\nПоследствия:")
    print("   ❌ Яркость увеличится → Потеряем информацию о материалах")
    print("   ❌ Контраст усилится → Артефакты от speckle шума")
    print("   ❌ Качество детекции УХУДШИТСЯ")

    print("\n" + "-"*70)
    print("ПОДХОД 2: С УЧЁТОМ ТИПА (новая система)")
    print("-"*70)
    print("\nОпределён тип: SAR")
    print("\nРекомендованные методы:")
    print("   1. denoise (median, ksize=5)")

    print("\nЗаблокированные методы:")
    print("   ❌ brightness_correction")
    print("      Причина: Низкая яркость — физическое свойство")
    print("   ❌ contrast_enhancement")
    print("      Причина: Высокий контраст уже есть")

    print("\nПоследствия:")
    print("   ✅ Яркость сохранена → Информация о материалах не потеряна")
    print("   ✅ Только подавление speckle → Без артефактов")
    print("   ✅ Качество детекции УЛУЧШИТСЯ")


def demo_6_code_examples():
    """Примеры кода для использования"""
    print("\n" + "="*80)
    print("ДЕМОНСТРАЦИЯ 6: ПРИМЕРЫ КОДА")
    print("="*80)

    print("\n" + "-"*70)
    print("Пример 1: Определить тип датасета")
    print("-"*70)
    print("""
from Utils.image_analyzer import UniversalImageAnalyzer
from Utils.modality_classifier import ImageModalityClassifier

# Анализируем датасет
analyzer = UniversalImageAnalyzer()
dataset_metrics, _ = analyzer.analyze_dataset('SAR_low', split='train')

# Определяем тип
classifier = ImageModalityClassifier()
modality_info = classifier.classify(dataset_metrics)

print(f"Тип: {modality_info['modality']}")
print(f"Уверенность: {modality_info['confidence']*100:.1f}%")
print(f"Источник: {modality_info['source']}")
    """)

    print("\n" + "-"*70)
    print("Пример 2: Проверить разрешён ли метод")
    print("-"*70)
    print("""
from Utils.preprocessing_rules import PreprocessingRules

modality = 'sar'
method = 'brightness_correction'

allowed = PreprocessingRules.is_method_allowed(modality, method)
print(f"{method} для {modality}: {'✅ Разрешён' if allowed else '❌ Запрещён'}")

if not allowed:
    rationale = PreprocessingRules.get_rationale(modality, method)
    print(f"Причина: {rationale}")
    """)

    print("\n" + "-"*70)
    print("Пример 3: Получить параметры метода")
    print("-"*70)
    print("""
from Utils.preprocessing_rules import PreprocessingRules

modality = 'natural_photo'
method = 'contrast_enhancement'

params = PreprocessingRules.get_method_params(modality, method)
print(f"Параметры для {method} в {modality}:")
print(params)
# → {'method': 'clahe', 'clip_limit': 2.0, 'tile_grid_size': (8, 8)}
    """)


def main():
    """Запускает все демонстрации"""
    print("\n" + "="*80)
    print("ДЕМОНСТРАЦИЯ СИСТЕМЫ ОПРЕДЕЛЕНИЯ ТИПА ДАТАСЕТА v2.0")
    print("="*80)

    print("\nЭтот скрипт покажет:")
    print("   1. Как классифицируются разные типы датасетов")
    print("   2. Какие правила применяются для каждого типа")
    print("   3. Как блокируются опасные методы")
    print("   4. Как адаптируются параметры")
    print("   5. Сравнение старого и нового подхода")
    print("   6. Примеры кода для использования")

    input("\nНажмите Enter для начала...")

    # Запускаем демонстрации
    try:
        demo_1_classification()
        input("\n\nНажмите Enter для продолжения...")

        demo_2_rules()
        input("\n\nНажмите Enter для продолжения...")

        demo_3_filtering()
        input("\n\nНажмите Enter для продолжения...")

        demo_4_parameters()
        input("\n\nНажмите Enter для продолжения...")

        demo_5_comparison()
        input("\n\nНажмите Enter для продолжения...")

        demo_6_code_examples()

    except ImportError as e:
        print(f"\n❌ Ошибка импорта: {e}")
        print("\nУбедитесь что файлы находятся в правильных директориях:")
        print("   - modality_classifier.py → Utils/")
        print("   - preprocessing_rules.py → Utils/")
        print("\nИ что вы запускаете скрипт из корня проекта!")

    print("\n" + "="*80)
    print("ДЕМОНСТРАЦИЯ ЗАВЕРШЕНА")
    print("="*80)
    print("\nТеперь вы можете:")
    print("   1. Запустить apply_preprocessing.py для применения к реальному датасету")
    print("   2. Прочитать подробную документацию (INTEGRATION_GUIDE.md)")


if __name__ == '__main__':
    main()