"""
Применение предобработки с автоматическим определением типа датасета

ИЗМЕНЕНИЯ (v3.0 — Intensity Variants):
    После подбора идеальной стратегии (методы + параметры под тип датасета)
    добавлен интерактивный выбор вариантов интенсивности: слабый / базовый / сильный.

    Научное обоснование:
    Montaha et al. (2022, Front. Med., doi: 10.3389/fmed.2022.924979)
    "MNet-10: A robust shallow CNN performing ablation study on medical images"
    Авторы показали на 8 датасетах, что оптимальный уровень интенсивности
    предобработки нельзя определить теоретически — он выявляется эмпирически
    через сравнение метрик обученной модели. На части датасетов оригинальные
    данные превосходили обработанные, что прямо мотивирует проверку нескольких
    вариантов интенсивности.

    Старое поведение (v2.0) полностью сохранено: если пользователь выбирает
    только базовый вариант — создаётся ровно один датасет, как и раньше.

Автор: Система адаптивной предобработки
Дата: 2025
"""

from pathlib import Path
from typing import Dict, List

from Data.Datasets.dataset_work import get_dataset_path, list_available_datasets
from Utils.image_analyzer import UniversalImageAnalyzer
from Utils.modality_classifier import ImageModalityClassifier
from Utils.preprocessing_rules import PreprocessingRules
from Utils.preprocessing_selector import AdaptivePreprocessingSelector
from Utils.intensity_variants import generate_intensity_variants, print_variants_comparison
from Preprocessing.applicator import DatasetPreprocessor


def apply_preprocessing_with_modality_detection():
    """
    Полный цикл применения предобработки с учётом типа датасета.

    НОВОЕ в v3.0:
    После Шага 7 (название датасета) добавлен Шаг 7.5 — интерактивный
    выбор вариантов интенсивности. Позволяет создать 1–3 датасета
    (слабый / базовый / сильный) за один запуск.
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
    print(f"\nВыбран датасет: {dataset_name}")

    # ========================================================================
    # ШАГ 2: АНАЛИЗ ДАТАСЕТА
    # ========================================================================
    print(f"\n{'='*80}")
    print(f"ШАГ 2: АНАЛИЗ ДАТАСЕТА")
    print(f"{'='*80}")

    analyzer = UniversalImageAnalyzer(verbose=False)

    print("\nАнализ изображений...")
    dataset_metrics, image_metrics = analyzer.analyze_dataset(
        dataset_path,
        split='train'
    )

    print(f"\n📊 Базовые метрики:")
    print(f"   Изображений: {dataset_metrics.num_images}")
    print(f"   SNR: {dataset_metrics.avg_snr:.2f} dB")
    print(f"   Контраст: {dataset_metrics.avg_contrast:.4f}")
    print(f"   Яркость: {dataset_metrics.avg_brightness:.4f}")
    print(f"   Резкость: {dataset_metrics.avg_sharpness:.0f}")

    # ========================================================================
    # ШАГ 3: КЛАССИФИКАЦИЯ ТИПА ДАТАСЕТА
    # ========================================================================
    print(f"\n{'='*80}")
    print(f"ШАГ 3: КЛАССИФИКАЦИЯ ТИПА ДАТАСЕТА")
    print(f"{'='*80}")

    classifier = ImageModalityClassifier()
    modality_info = classifier.classify(dataset_metrics)
    classifier.print_classification_report(modality_info)

    # ========================================================================
    # ШАГ 4: ПРАВИЛА ПРЕДОБРАБОТКИ ДЛЯ ТИПА
    # ========================================================================
    print(f"\n{'='*80}")
    print(f"ШАГ 4: ПРАВИЛА ПРЕДОБРАБОТКИ ДЛЯ ТИПА '{modality_info['modality'].upper()}'")
    print(f"{'='*80}")

    PreprocessingRules.print_rules_summary(modality_info['modality'])

    # ========================================================================
    # ШАГ 5: ВЫБОР СТРАТЕГИИ С УЧЁТОМ ТИПА
    # ========================================================================
    print(f"\n{'='*80}")
    print(f"ШАГ 5: ВЫБОР СТРАТЕГИИ ПРЕДОБРАБОТКИ")
    print(f"{'='*80}")

    selector = AdaptivePreprocessingSelector(
        analyzer=analyzer,
        modality_info=modality_info
    )

    strategy = selector.select_strategy(dataset_path, split='train')
    selector.print_strategy_info(strategy)

    # ========================================================================
    # ШАГ 6: ПОДТВЕРЖДЕНИЕ
    # ========================================================================
    print(f"\n{'='*80}")
    confirm = input("\nПрименить предобработку с учётом типа датасета? (y/n): ")

    if confirm.lower() != 'y':
        print("Отменено.")
        return

    # ========================================================================
    # ШАГ 7: БАЗОВОЕ НАЗВАНИЕ НОВОГО ДАТАСЕТА
    # ========================================================================
    default_name = f"{dataset_name}_preprocessed_{modality_info['modality']}"
    base_name = input(f"\nВведите базовое название нового датасета [{default_name}]: ").strip()
    if not base_name:
        base_name = default_name

    # ========================================================================
    # ШАГ 7.5: ВЫБОР ВАРИАНТОВ ИНТЕНСИВНОСТИ
    # ========================================================================
    selected_variants = _ask_intensity_variants()

    base_params = _build_params_for_modality(modality_info, selector)
    methods = strategy.get('methods', [])

    variants_params = generate_intensity_variants(
        base_params=base_params,
        methods=methods,
        variants=selected_variants
    )

    if len(selected_variants) > 1:
        print_variants_comparison(variants_params, methods)

    dataset_names = _ask_dataset_names(base_name, selected_variants)

    # ========================================================================
    # ШАГ 8: ПРИМЕНЕНИЕ ПРЕДОБРАБОТКИ ДЛЯ КАЖДОГО ВАРИАНТА
    # ========================================================================
    print(f"\n{'='*80}")
    print(f"ШАГ 8: ПРИМЕНЕНИЕ ПРЕДОБРАБОТКИ")
    print(f"{'='*80}")

    preprocessor = DatasetPreprocessor()

    for level in selected_variants:
        new_name     = dataset_names[level]
        level_params = variants_params[level]

        print(f"\n{'─'*60}")
        if len(selected_variants) > 1:
            level_labels = {'weak': 'слабый', 'base': 'базовый', 'strong': 'сильный'}
            print(f"  Вариант: {level_labels.get(level, level.upper())}  ->  {new_name}")
        print(f"{'─'*60}")

        if strategy['strategy'] == 'adaptive':
            print("\nПрименяю адаптивную предобработку...")
            preprocessor.apply_adaptive_preprocessing(
                source_dataset=dataset_name,
                target_dataset=new_name,
                clusters=strategy['clusters'],
                image_metrics=image_metrics,
                params=level_params
            )
        else:
            print("\nПрименяю глобальную предобработку...")
            preprocessor.apply_global_preprocessing(
                source_dataset=dataset_name,
                target_dataset=new_name,
                methods=methods,
                params=level_params
            )

    print(f"\nПредобработка завершена.")
    print(f"   Тип датасета: {modality_info['modality']}")
    print(f"   Создано датасетов: {len(selected_variants)}")
    for level in selected_variants:
        print(f"      {level:6s} → {dataset_names[level]}")

    # ========================================================================
    # ШАГ 9: АНАЛИЗ РЕЗУЛЬТАТОВ (все созданные варианты)
    # ========================================================================
    print(f"\n{'='*80}")
    print(f"ШАГ 9: АНАЛИЗ РЕЗУЛЬТАТОВ")
    print(f"{'='*80}")

    all_preprocessed_metrics = {}

    for level in selected_variants:
        ds_name  = dataset_names[level]
        ds_path  = get_dataset_path(ds_name)
        lbl      = {'weak': '🟡 СЛАБЫЙ', 'base': '🟢 БАЗОВЫЙ', 'strong': '🔴 СИЛЬНЫЙ'}.get(level, level)
        print(f"\n⏳ Анализирую {lbl} → '{ds_name}'...")
        metrics, _ = analyzer.analyze_dataset(ds_path, split='train')
        all_preprocessed_metrics[level] = metrics

    # ========================================================================
    # ШАГ 10: СРАВНЕНИЕ ДО И ПОСЛЕ
    # ========================================================================
    print(f"\n{'='*80}")
    print(f"ШАГ 10: СРАВНЕНИЕ ДО И ПОСЛЕ")
    print(f"{'='*80}")

    _print_comparison_all(
        original_metrics=dataset_metrics,
        variants_metrics=all_preprocessed_metrics,
        selected_variants=selected_variants,
        modality_info=modality_info
    )


# =============================================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# =============================================================================

def _ask_intensity_variants() -> List[str]:
    """
    Спрашивает пользователя какие варианты интенсивности нужны.

    Сначала спрашивает — нужны ли вообще дополнительные варианты (y/n).
    Если нет — возвращает только ['base'].
    Если да — спрашивает отдельно про слабый и сильный (y/n каждый).
    Базовый включён всегда.
    """
    print(f"\n{'='*80}")
    print(f"ШАГ 7.5: ВАРИАНТЫ ИНТЕНСИВНОСТИ ПРЕДОБРАБОТКИ")
    print(f"{'='*80}")
    print("""
  Помимо базового варианта можно создать дополнительные с другой интенсивностью:
    🟡 СЛАБЫЙ  (weak)   — параметры ×0.5 от базового
    🟢 БАЗОВЫЙ (base)   — параметры как подобраны для типа датасета (всегда)
    🔴 СИЛЬНЫЙ (strong) — параметры ×2.0 от базового

  Это позволит эмпирически выбрать оптимальный вариант после обучения модели.
  (Montaha et al., 2022: оптимум определяется только через сравнение на модели)
    """)

    ans = input("  Создать дополнительные варианты интенсивности? [y/n]: ").strip().lower()
    if ans != 'y':
        print("  → Будет создан только базовый вариант.")
        return ['base']

    selected = ['base']

    ans_weak = input("\n  Создать слабый вариант (weak, ×0.5)? [y/n]: ").strip().lower()
    if ans_weak == 'y':
        selected.append('weak')

    ans_strong = input("  Создать сильный вариант (strong, ×2.0)? [y/n]: ").strip().lower()
    if ans_strong == 'y':
        selected.append('strong')

    ordered = [v for v in ['weak', 'base', 'strong'] if v in selected]
    return ordered


def _ask_dataset_names(base_name: str, variants: List[str]) -> Dict[str, str]:
    """
    Спрашивает имена датасетов для каждого варианта.
    Если вариант один — имя без суффикса.
    Если несколько — предлагает имена с суффиксами _weak / _base / _strong.
    """
    names = {}

    if len(variants) == 1:
        names[variants[0]] = base_name
        return names

    print(f"\n  Названия датасетов для каждого варианта.")
    print(f"  (нажмите Enter, чтобы принять предложенное имя)\n")

    for level in variants:
        default = f"{base_name}_{level}"
        ans = input(f"  [{level}] [{default}]: ").strip()
        names[level] = ans if ans else default

    return names


def _build_params_for_modality(modality_info: Dict, selector: AdaptivePreprocessingSelector) -> Dict:
    """
    Строит словарь параметров для методов на основе типа датасета.
    """
    params = {}
    methods = ['denoise', 'contrast_enhancement', 'brightness_correction', 'sharpening']
    for method in methods:
        method_params = selector.get_method_params(method)
        if method_params:
            params[method] = method_params
    return params


def _print_comparison_all(
    original_metrics,
    variants_metrics: Dict,
    selected_variants: List[str],
    modality_info: Dict
):
    """
    Сравнение стандартных метрик оригинала с каждым вариантом предобработки.
    Один блок на вариант. Только метрики из DatasetMetrics — без самописных.

    Метрики:
      - SNR (avg_snr, std_snr)          — стандартная метрика шума
      - Контраст (avg_contrast,          — Michelson contrast
                  std_contrast)          — вариативность контраста по изображениям
      - Резкость (avg_sharpness)         — Laplacian variance
      - blur_count                       — изображения ниже порога резкости
      - Яркость (avg_brightness,
                 std_brightness)
      - noise_distribution['high']       — изображения с высоким шумом
    """
    level_labels = {
        'weak':   '🟡 СЛАБЫЙ   (weak)',
        'base':   '🟢 БАЗОВЫЙ  (base)',
        'strong': '🔴 СИЛЬНЫЙ  (strong)',
    }
    variants_order = [v for v in ['weak', 'base', 'strong'] if v in selected_variants]

    print(f"\n  Тип датасета: {modality_info['modality'].upper()}")
    print(f"  Изображений:  {original_metrics.num_images}")

    for lv in variants_order:
        var   = variants_metrics[lv]
        label = level_labels.get(lv, lv.upper())

        print(f"\n{'='*55}")
        print(f"  {label}")
        print(f"{'='*55}")

        print(f"  Шум:")
        _row("SNR среднее",       original_metrics.avg_snr,    var.avg_snr,    ".2f", " dB", '+')
        _row("SNR разброс (std)", original_metrics.std_snr,    var.std_snr,    ".2f", " dB", '-')
        _row_int("С высоким шумом",
                 original_metrics.noise_distribution.get('high', 0),
                 var.noise_distribution.get('high', 0), '-')

        print(f"\n  Контраст и резкость:")
        _row("avg_contrast",      original_metrics.avg_contrast,  var.avg_contrast,  ".4f")
        _row("std_contrast",      original_metrics.std_contrast,  var.std_contrast,  ".4f")
        _row("Резкость",          original_metrics.avg_sharpness, var.avg_sharpness, ".1f")
        _row_int("Размытых",      original_metrics.blur_count,    var.blur_count,    '-')

        print(f"\n  Яркость:")
        _row("Среднее",           original_metrics.avg_brightness,     var.avg_brightness,     ".4f", good='~')
        _row("Разброс (std)",     original_metrics.std_brightness,     var.std_brightness,     ".4f", '-')


def _row(label: str, before: float, after: float,
         fmt: str = ".4f", unit: str = "", good: str = '+'):
    """Строка сравнения: название  до → после  (изменение ▲/▼ ✓)"""
    change = after - before
    if change > 1e-6:
        arrow, mark = '▲', ('✓' if good == '+' else '')
    elif change < -1e-6:
        arrow, mark = '▼', ('✓' if good == '-' else '')
    else:
        arrow, mark = '—', ''
    b = f"{before:{fmt}}{unit}"
    a = f"{after:{fmt}}{unit}"
    c = f"{change:+{fmt}}{unit}"
    print(f"    {label:<22}  {b}  →  {a}  ({c} {arrow} {mark})")


def _row_int(label: str, before, after, good: str = '-'):
    """Строка сравнения для целочисленной метрики."""
    before, after = int(before), int(after)
    change = after - before
    if change > 0:
        arrow, mark = '▲', ('✓' if good == '+' else '')
    elif change < 0:
        arrow, mark = '▼', ('✓' if good == '-' else '')
    else:
        arrow, mark = '—', ''
    print(f"    {label:<22}  {before}  →  {after}  ({change:+d} {arrow} {mark})")


if __name__ == '__main__':
    apply_preprocessing_with_modality_detection()
