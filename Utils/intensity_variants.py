"""
Генерация вариантов предобработки с разной интенсивностью параметров.

Научное обоснование диапазона интенсивностей:
    Montaha et al. (2022, Front. Med., doi: 10.3389/fmed.2022.924979)
    "MNet-10: A robust shallow CNN performing ablation study on medical images"

    Авторы исследовали 8 датасетов разных модальностей и установили:
    1. Оптимальный метод предобработки нельзя определить теоретически —
       он определяется эмпирически через сравнение метрик на конкретном датасете.
    2. На ряде датасетов необработанные данные давали лучший результат,
       что мотивирует генерацию вариантов разной интенсивности для выбора оптимума.

    Параметрический диапазон x0.5–x2.0 выбран эвристически как практически
    обоснованный диапазон умеренного изменения параметров.
"""

import copy
from typing import Dict, List


# =============================================================================
# МНОЖИТЕЛИ ИНТЕНСИВНОСТИ
# =============================================================================

_MULTIPLIERS = {
    'weak': {
        'denoise': {
            'ksize':        0.6,
            'h':            0.5,
            'sigma_color':  0.6,
            'sigma_space':  0.6,
            'd':            0.7,
        },
        'contrast_enhancement': {
            'clip_limit':   0.5,
        },
        'brightness_correction': {
            'target_brightness': 0.7,
        },
        'sharpening': {
            'alpha':  0.5,
            'amount': 0.5,
        },
    },
    'strong': {
        'denoise': {
            'ksize':        1.7,
            'h':            2.0,
            'sigma_color':  1.5,
            'sigma_space':  1.5,
            'd':            1.4,
        },
        'contrast_enhancement': {
            'clip_limit':   2.0,
        },
        'brightness_correction': {
            'target_brightness': 1.3,
        },
        'sharpening': {
            'alpha':  2.0,
            'amount': 2.0,
        },
    },
}

_ODD_INT_PARAMS = {'ksize'}
_INT_PARAMS     = {'ksize', 'd', 'template_window_size', 'search_window_size'}

_PARAM_LIMITS = {
    'ksize':             {'min': 3,   'max': 11},
    'h':                 {'min': 3,   'max': 30},
    'sigma_color':       {'min': 15,  'max': 150},
    'sigma_space':       {'min': 15,  'max': 150},
    'd':                 {'min': 3,   'max': 15},
    'clip_limit':        {'min': 0.3, 'max': 4.0},
    'alpha':             {'min': 0.3, 'max': 3.0},
    'amount':            {'min': 0.3, 'max': 3.0},
    'target_brightness': {'min': 0.1, 'max': 0.9},
}


def _scale_param(name: str, value, mult: float):
    """Масштабирует числовой параметр с проверкой типа и пределов."""
    if not isinstance(value, (int, float)):
        return value  # tuple, str и пр. — не трогаем

    v = value * mult

    if name in _PARAM_LIMITS:
        lim = _PARAM_LIMITS[name]
        v = max(lim['min'], min(lim['max'], v))

    if name in _ODD_INT_PARAMS:
        v = int(v)
        if v % 2 == 0:
            v += 1
        v = max(3, v)
    elif name in _INT_PARAMS:
        v = max(1, int(v))

    return v


def _apply_intensity_to_params(base: Dict, method: str, level: str) -> Dict:
    """Возвращает копию параметров метода с применённым уровнем интенсивности."""
    if level == 'base':
        return copy.deepcopy(base)

    mults = _MULTIPLIERS.get(level, {}).get(method, {})
    result = copy.deepcopy(base)

    # Если base пустой, но для метода есть дефолтные значения параметров —
    # инициализируем их, чтобы было что масштабировать
    _DEFAULTS = {
        'brightness_correction': {'target_brightness': 0.5},
    }
    if not result and method in _DEFAULTS:
        result = copy.deepcopy(_DEFAULTS[method])

    for param, mult in mults.items():
        if param in result:
            result[param] = _scale_param(param, result[param], mult)
    return result


def generate_intensity_variants(
    base_params: Dict,
    methods: List[str],
    variants: List[str]
) -> Dict[str, Dict]:
    """
    Генерирует варианты параметров предобработки с разной интенсивностью.

    Args:
        base_params: Базовые параметры {'denoise': {...}, 'contrast_enhancement': {...}}
        methods:     Активные методы из strategy['methods']
        variants:    Список уровней: любой набор из ['weak', 'base', 'strong']

    Returns:
        {'base': {...}, 'weak': {...}, 'strong': {...}}
    """
    invalid = set(variants) - {'weak', 'base', 'strong'}
    if invalid:
        raise ValueError(f"Недопустимые уровни: {invalid}. Допустимые: weak, base, strong")

    result = {}
    for level in variants:
        level_params = {}
        for method in methods:
            level_params[method] = _apply_intensity_to_params(
                base_params.get(method, {}), method, level
            )
        result[level] = level_params
    return result


def print_variants_comparison(variants: Dict[str, Dict], methods: List[str]):
    """Красиво выводит сравнение параметров по уровням интенсивности."""
    labels = {
        'weak':   'слабый   (weak)  ',
        'base':   'базовый  (base)  ',
        'strong': 'сильный  (strong)',
    }

    print(f"\n{'='*70}")
    print("ВАРИАНТЫ ИНТЕНСИВНОСТИ ПРЕДОБРАБОТКИ")
    print(f"{'='*70}")
    print(
        "\n  Montaha et al. (2022, Front. Med.): оптимальная интенсивность\n"
        "  предобработки определяется эмпирически — через метрики обученной\n"
        "  модели на каждом из созданных вариантов датасета.\n"
    )

    for method in methods:
        print(f"  {method.upper()}")
        print(f"  {'─'*60}")
        for level in ['weak', 'base', 'strong']:
            if level not in variants:
                continue
            params = variants[level].get(method, {})
            label = labels.get(level, level)
            nums = ', '.join(
                f"{k}={v}" for k, v in params.items()
                if isinstance(v, (int, float))
            ) or 'без числовых параметров'
            print(f"  {label} : {nums}")
        print()

    print(f"  Итого вариантов: {len(variants)}  →  создаётся датасетов: {len(variants)}")
    print(f"{'='*70}\n")
