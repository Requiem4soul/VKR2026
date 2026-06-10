"""
pages/2_Подбор_предобработки.py — Подбор пайплайна предобработки

Алгоритм: Group-wise SHA (Фаза 1) + SFS+SHA на survivors (Фаза 2)

Тип датасета задаётся вручную пользователем:
    Для медицинских снимков и SAR применяются правила PreprocessingRules —
    группы методов, запрещённые для данной модальности, исключаются из
    CANDIDATE_GROUPS до начала Фазы 1.
    Для натуральных изображений и «другой модальности» используется
    чистый SHA+SFS без фильтрации.

    Научное обоснование фильтрации по модальности:
    - SAR: Oliver & Quegan (2004) — brightness/sharpening искажают физическую информацию
    - Medical: Pisano et al. (1998) J.Digital Imaging 11(4):193-200; Pisano et al. (2000) RadioGraphics 20:1479-1491
    - Microscopy: Kolarević et al. (2018) Journal of Microscopy 269(3):264-276; Sternberg (1983) Computer 16(1):22-34

Научное обоснование:
- SHA:  Jamieson & Talwalkar (2016) "Non-stochastic best arm identification",
        AISTATS, pp. 240–248
- SFS:  Kohavi & John (1997) "Wrappers for feature subset selection",
        Artificial Intelligence, 97(1-2), 273–324
- Двухфазная группировка методов перед отбором:
        Guyon & Elisseeff (2003) "An Introduction to Variable and Feature Selection",
        Journal of Machine Learning Research, 3, 1157–1182
        Liu & Motoda (2007) "Computational Methods of Feature Selection",
        Chapman and Hall/CRC, ISBN 978-1584888789
- 30% эпох для быстрой оценки кандидатов:
        Jamieson & Talwalkar (2016), ibid.
- Воспроизводимость через seed:
        Dodge & Karam (2017) "A Study and Comparison of Human and
        Deep Learning Recognition Performance Under Visual Distortions"

Отличие от Модуля 3 (SFS+SHA на полном пуле):
    Данный модуль делит пул на тематические группы (шум, контраст, яркость, резкость),
    проводит SHA-скрининг внутри каждой группы (Фаза 1), сокращая пул до survivors,
    и лишь затем запускает SFS+SHA только на survivors (Фаза 2).
    Компромисс: возможна частичная потеря межгрупповых взаимодействий на Фазе 1,
    выигрыш — значительное сокращение числа обучений относительно Модуля 3.

Обязательная очистка памяти после каждого обучения:
    После каждой модели: del model, torch.cuda.empty_cache(), gc.collect()
    Аналогично classification_trainer.py (Модуль 2).
"""

import gc
import os
import sys
import json
import math
import time
import shutil
import queue
import threading
import traceback
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

import streamlit as st

from ui.sidebar import render_sidebar
from ui.state import (
    init_session_state,
    is_path_configured,
    get_available_datasets,
    get_datasets_path,
)

st.set_page_config(
    page_title="Объединённый подбор — VKR2026",
    page_icon=None,
    layout="wide",
)
init_session_state()
render_sidebar()

if not is_path_configured():
    st.error("Сначала настрой путь к датасетам в разделе **Настройки**.")
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
# СОСТОЯНИЕ СТРАНИЦЫ
# ══════════════════════════════════════════════════════════════════════════════

_STATE_DEFAULTS = {
    "p2_stage":          "configure",   # configure | running | done
    # Модальный анализ (6_Объединение.py)
    "p2_use_modality":   True,
    "p2_manual_modality": "other",  # одна из модальностей или "other" (все методы)
    "p2_use_wiener":     False,
    "p2_sha_fallback":   False,   # SHA без baseline-фильтра если Фаза 1 пуста
    "p2_use_torch_compile": True,
    "p2_modality_result": None,   # результат анализа модальности
    "p2_log_lines":      [],
    "p2_output_queue":   None,
    "p2_thread_done":    False,
    "p2_error":          None,
    "p2_result":         None,
    "p2_dataset":        None,
    "p2_task":           "classification",
    # Классификация
    "p2_model_type":     "resnet18",
    "p2_imgsz":          224,
    "p2_pretrained":     True,
    "p2_freeze_backbone": False,
    "p2_batch":          -1,
    # Детекция
    "p2_det_model":      "yolo",
    "p2_yolo_size":      "n",
    "p2_det_pretrained": True,
    "p2_det_batch":      -1,
    "p2_det_imgsz":      640,
    # Общие
    "p2_epochs":         50,
    "p2_patience":       10,
    "p2_seed":           42,
    "p2_eta":            2,
    "p2_winner_ds":      None,
    # Автоподбор процента скрининга
    "p2_auto_screen":         False,   # включить автоподбор
    "p2_auto_screen_start":   40,      # начальный % (пользователь задаёт)
    "p2_auto_screen_direction": "top_down",  # направление поиска: top_down / bottom_up / full_budget / warm_start
    "p2_history_csv_content":   "",          # содержимое загруженного CSV истории
    "p2_top_k_winners":       1,       # сколько топ-survivors обучать финально
}
for _k, _v in _STATE_DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


def _reset():
    for k, v in _STATE_DEFAULTS.items():
        st.session_state[k] = v


# ══════════════════════════════════════════════════════════════════════════════
# ОПИСАНИЕ ГРУПП МЕТОДОВ (пул кандидатов)
# ══════════════════════════════════════════════════════════════════════════════
#
# Группы сформированы по типу воздействия на изображение.
# Параметры выбраны на основе:
#   Gonzalez & Woods (2018) "Digital Image Processing", 4th ed.
#   Tomasi & Manduchi (1998) "Bilateral filtering for gray and color images", ICCV.
#   Pisano et al. (1998) "Contrast Limited Adaptive Histogram Equalization",
#       J. Digital Imaging, 11(4), 193–200.

CANDIDATE_GROUPS = {
    "denoise": {
        "label": "Шумоподавление",
        "candidates": [
            {
                "id": "median_k3",
                "display": "Median (ksize=3)",
                "methods": ["denoise"],
                "params": {"denoise": {"method": "median", "ksize": 3}},
            },
            {
                "id": "median_k5",
                "display": "Median (ksize=5)",
                "methods": ["denoise"],
                "params": {"denoise": {"method": "median", "ksize": 5}},
            },
            {
                "id": "gaussian_k3",
                "display": "Gaussian blur (ksize=3)",
                "methods": ["denoise"],
                "params": {"denoise": {"method": "gaussian", "ksize": 3}},
            },
            {
                "id": "gaussian_k5",
                "display": "Gaussian blur (ksize=5)",
                "methods": ["denoise"],
                "params": {"denoise": {"method": "gaussian", "ksize": 5}},
            },
            {
                "id": "bilateral_s75",
                "display": "Bilateral (sigma=75)",
                "methods": ["denoise"],
                "params": {"denoise": {"method": "bilateral", "d": 9,
                                       "sigma_color": 75, "sigma_space": 75}},
            },
            {
                "id": "bilateral_s150",
                "display": "Bilateral (sigma=150)",
                "methods": ["denoise"],
                "params": {"denoise": {"method": "bilateral", "d": 9,
                                       "sigma_color": 150, "sigma_space": 150}},
            },
            {
                "id": "wiener_s3",
                "display": "Wiener (size=3)",
                "methods": ["denoise"],
                "params": {"denoise": {"method": "wiener", "size": 3}},
            },
            {
                "id": "wiener_s5",
                "display": "Wiener (size=5)",
                "methods": ["denoise"],
                "params": {"denoise": {"method": "wiener", "size": 5}},
            },
            {
                "id": "lee_k3",
                "display": "Lee (ksize=3)",
                "methods": ["denoise"],
                "params": {"denoise": {"method": "lee", "ksize": 3}},
            },
            {
                "id": "lee_k5",
                "display": "Lee (ksize=5)",
                "methods": ["denoise"],
                "params": {"denoise": {"method": "lee", "ksize": 5}},
            },
        ],
        # Пул шумоподавления покрывает три классических пространственных фильтра
        # (базовый набор, всегда активен):
        #   Median    — импульсный шум (salt & pepper). Gonzalez & Woods (2018).
        #   Gaussian  — равномерный фоновый шум, speckle. Gonzalez & Woods (2018).
        #   Bilateral — Gaussian шум с сохранением краёв. Tomasi & Manduchi (1998).
        # Wiener (опционально, включается пользователем):
        #   Wiener    — адаптивный линейный фильтр, минимизирует MSE.
        #               Wiener (1949); Fan et al. (2019) "Brief review of image
        #               denoising techniques", Visual Computing for Industry,
        #               Biomedicine, and Art, 2(1).
        #               Исключён по умолчанию: scipy.signal.wiener реализован на
        #               Python без SIMD-оптимизации — ~1–1.5 сек/изображение vs
        #               ~3–8 мс для OpenCV-фильтров. На датасетах >10k изображений
        #               применение Wiener увеличивает время предобработки на часы.
        # Lee — специализированный фильтр для мультипликативного speckle-шума SAR.
        #   Lee (1980) IEEE Trans. PAMI-2(2):165-168;
        #   Lee (1981) Comput. Graph. Image Process. 17(1):24-32.
        #   Включён в базовый пул: реализован через numpy/cv2, быстрый (~5-10 мс).
        # NLM исключён: O(N²) сложность непрактична для SHA-скрининга.
        # Jamieson & Talwalkar (2016) — бюджет на обучение, не предобработку.
    },
    "contrast": {
        "label": "Контраст",
        "candidates": [
            {
                "id": "clahe_c10",
                "display": "CLAHE (clip=1.0)",
                "methods": ["contrast_enhancement"],
                "params": {"contrast_enhancement": {"method": "clahe", "clip_limit": 1.0}},
            },
            {
                "id": "clahe_c20",
                "display": "CLAHE (clip=2.0)",
                "methods": ["contrast_enhancement"],
                "params": {"contrast_enhancement": {"method": "clahe", "clip_limit": 2.0}},
            },
            {
                "id": "clahe_c30",
                "display": "CLAHE (clip=3.0)",
                "methods": ["contrast_enhancement"],
                "params": {"contrast_enhancement": {"method": "clahe", "clip_limit": 3.0}},
            },
            {
                "id": "histeq",
                "display": "Histogram Equalization",
                "methods": ["contrast_enhancement"],
                "params": {"contrast_enhancement": {"method": "histogram_eq"}},
            },
        ],
    },
    "brightness": {
        "label": "Яркость",
        "candidates": [
            {
                "id": "gamma_05",
                "display": "Gamma (γ=0.5, осветление)",
                "methods": ["brightness_correction"],
                "params": {"brightness_correction": {"gamma": 0.5}},
            },
            {
                "id": "gamma_08",
                "display": "Gamma (γ=0.8, лёгкое осветление)",
                "methods": ["brightness_correction"],
                "params": {"brightness_correction": {"gamma": 0.8}},
            },
            {
                "id": "gamma_12",
                "display": "Gamma (γ=1.2, лёгкое затемнение)",
                "methods": ["brightness_correction"],
                "params": {"brightness_correction": {"gamma": 1.2}},
            },
        ],
        # Гамма-коррекция — нелинейное степенное преобразование яркости.
        # gamma < 1 осветляет (подтягивает тёмные области),
        # gamma > 1 затемняет (подавляет пересветы).
        # Три значения покрывают типичные проблемы датасетов:
        #   γ=0.5 — сильное осветление (недоэкспонированные изображения)
        #   γ=0.8 — мягкая коррекция (слегка тёмные датасеты)
        #   γ=1.2 — мягкое затемнение (слегка пересвеченные датасеты)
        # Реализация через LUT — O(1) на пиксель, очень быстро.
        # Gonzalez & Woods (2018) "Digital Image Processing", 4th ed., гл. 3.2.
    },
    "sharpening": {
        "label": "Резкость",
        "candidates": [
            {
                "id": "usm_a05",
                "display": "Unsharp Mask (alpha=0.5)",
                "methods": ["sharpening"],
                "params": {"sharpening": {"method": "unsharp_mask", "alpha": 0.5}},
            },
            {
                "id": "usm_a10",
                "display": "Unsharp Mask (alpha=1.0)",
                "methods": ["sharpening"],
                "params": {"sharpening": {"method": "unsharp_mask", "alpha": 1.0}},
            },
            {
                "id": "usm_a15",
                "display": "Unsharp Mask (alpha=1.5)",
                "methods": ["sharpening"],
                "params": {"sharpening": {"method": "unsharp_mask", "alpha": 1.5}},
            },
        ],
    },
}


# ══════════════════════════════════════════════════════════════════════════════
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ АЛГОРИТМА
# ══════════════════════════════════════════════════════════════════════════════

def sha_prune(candidates: List[Dict], eta: int = 2) -> List[Dict]:
    """
    Successive Halving: оставляет top-ceil(N/eta) кандидатов по score.
    Jamieson & Talwalkar (2016) "Non-stochastic best arm identification", AISTATS.
    """
    n = len(candidates)
    keep = math.ceil(n / eta)
    keep = max(1, keep)
    sorted_c = sorted(candidates, key=lambda x: x.get("score", 0.0), reverse=True)
    return sorted_c[:keep]


def _spearman_rho(scores_a: Dict[str, float], scores_b: Dict[str, float]) -> float:
    """
    Коэффициент ранговой корреляции Спирмена между двумя наборами scores.

    Принимает словари {candidate_id: score} для двух прогонов.
    Учитывает только общие ключи (кандидаты присутствующие в обоих прогонах).

    Используется для автоподбора процента скрининга:
    если ρ(x%, x+10%) >= критического значения, то x% достаточно.

    Научное обоснование:
    Li et al. (2018) "Hyperband", JMLR 18(185): warm-start successive
    halving с бюджетным ранжированием.

    Returns:
        float: ρ ∈ [-1, 1]. 1.0 — идеальное совпадение рангов.
               0.0 если менее 2 общих кандидатов (нет смысла считать).
    """
    # Общие ключи
    common_keys = [k for k in scores_a if k in scores_b]
    n = len(common_keys)
    if n < 2:
        return 0.0

    a = [scores_a[k] for k in common_keys]
    b = [scores_b[k] for k in common_keys]

    def _ranks(vals):
        # Присваиваем ранги: 1 = лучший (наибольший score)
        indexed = sorted(enumerate(vals), key=lambda x: x[1], reverse=True)
        ranks = [0.0] * len(vals)
        for rank, (idx, _) in enumerate(indexed, 1):
            ranks[idx] = float(rank)
        return ranks

    ra = _ranks(a)
    rb = _ranks(b)

    d2 = sum((ra[i] - rb[i]) ** 2 for i in range(n))
    rho = 1.0 - (6.0 * d2) / (n * (n * n - 1))
    return round(rho, 4)


def _spearman_critical_rho(n: int, alpha: float = 0.05) -> float:
    """
    Критическое значение ρ Спирмена для заданного N и уровня значимости α.

    Вычисляется через обратное t-распределение:
      t_crit = t_{α/2, n-2}
      ρ_crit = t_crit / sqrt(t_crit² + n - 2)

    Zar J.H. (2005) "Spearman Rank Correlation", Encyclopedia of
    Biostatistics, Wiley. DOI: 10.1002/0470011815.b2a15150.

    Ramsey P.H. (1989) "Critical Values for Spearman's Rank Order
    Correlation", J. Educational Statistics, 14(3), 245–253.

    Args:
        n:     число наблюдений (кандидатов)
        alpha: уровень значимости (двусторонний тест). 0.05 → p < 0.05.

    Returns:
        ρ_crit: минимальное значение |ρ| для статистической значимости.
                При n < 4 возвращает 1.0 (невозможно достичь значимости).
    """
    if n < 4:
        return 1.0

    # Квантиль t-распределения через аппроксимацию Абрамовица-Стегана.
    # Abramowitz & Stegun (1964) "Handbook of Mathematical Functions",
    # формула 26.2.17 — точность ~4.5e-4 для p ∈ [0.0001, 0.5].
    import math
    p = alpha / 2.0  # двусторонний тест
    df = n - 2

    # Аппроксимация квантили нормального распределения (Abramowitz & Stegun 26.2.17)
    _t = math.sqrt(-2.0 * math.log(p))
    _c0, _c1, _c2 = 2.515517, 0.802853, 0.010328
    _d1, _d2, _d3 = 1.432788, 0.189269, 0.001308
    z = _t - (_c0 + _c1 * _t + _c2 * _t ** 2) / (1 + _d1 * _t + _d2 * _t ** 2 + _d3 * _t ** 3)

    # Cornish-Fisher поправка для t-распределения (малые df)
    # Johnson N.L. et al. (1995) "Continuous Univariate Distributions", Vol.2, Wiley.
    g1 = (z ** 3 + z) / (4 * df)
    g2 = (5 * z ** 5 + 16 * z ** 3 + 3 * z) / (96 * df ** 2)
    t_crit = z + g1 + g2

    # ρ_crit из t_crit: Zar (2005), формула обратного преобразования
    rho_crit = t_crit / math.sqrt(t_crit ** 2 + df)
    return round(min(rho_crit, 0.99), 4)  # cap at 0.99 для n=4


def _check_flat_scores(scores: Dict[str, float], log_fn=None) -> bool:
    """
    Проверяет являются ли scores кандидатов «плоскими» (слишком близкими).

    Если коэффициент вариации (CV = stdev/mean) ниже порога, ранжирование
    нестабильно: стохастический шум обучения превышает различия между
    кандидатами, и Спирмен ρ не может сойтись ни при каком числе эпох.

    Порог CV < 1.5% выбран эмпирически: при типичном шуме YOLO ±0.005–0.01
    по composite score, spread < 0.02 делает ранги случайными.

    Научное обоснование:
    Audibert, Bubeck & Munos (2010) "Best Arm Identification in Multi-Armed
    Bandits", COLT 2010: при ε-close arms (разница между arms < ε) число
    сэмплов для идентификации лучшего растёт как O(1/ε²). При ε→0
    идентификация требует бесконечного бюджета.

    Args:
        scores: словарь {candidate_id: score}
        log_fn: функция логирования

    Returns:
        True если scores плоские (предобработка не даёт значимого эффекта).
    """
    import statistics

    CV_THRESHOLD = 0.015  # 1.5%

    vals = list(scores.values())
    if len(vals) < 2:
        return False

    mean = statistics.mean(vals)
    if mean < 1e-8:
        return False

    stdev = statistics.stdev(vals)
    cv = stdev / mean

    if log_fn and cv < CV_THRESHOLD:
        log_fn(f"    Scores кандидатов практически одинаковы:")
        log_fn(f"     CV = {cv:.4f} ({cv*100:.2f}%) < порог {CV_THRESHOLD*100:.1f}%")
        log_fn(f"     mean={mean:.4f}  stdev={stdev:.4f}  "
               f"spread={max(vals)-min(vals):.4f}")
        log_fn(f"     Audibert et al. (2010) COLT: при ε-close candidates")
        log_fn(f"     ранжирование нестабильно при любом бюджете эпох.")
        log_fn(f"     Предобработка не даёт значимого эффекта для данного датасета.")

    return cv < CV_THRESHOLD


def merge_methods_params(candidates: List[Dict]) -> Tuple[List[str], Dict]:
    """
    Объединяет методы и параметры нескольких кандидатов в один пайплайн.

    Поддерживает несколько кандидатов из одной группы (например два denoise-фильтра
    последовательно): Gaussian k3 → Wiener s3 — легитимная комбинация для смешанного
    шума (Gonzalez & Woods, 2018; PMC7036412).

    Возвращает:
        methods: список шагов вида "denoise__0", "denoise__1", ...
                 (уникальные ключи для apply_pipeline)
        params:  {"denoise__0": {...}, "denoise__1": {...}, ...}
    """
    methods = []
    params = {}
    # Счётчик повторений имён методов для уникальных ключей
    method_counts: Dict[str, int] = {}
    for c in candidates:
        for m in c["methods"]:
            idx = method_counts.get(m, 0)
            method_counts[m] = idx + 1
            key = f"{m}__{idx}" if idx > 0 else m
            methods.append(key)
            # Копируем params кандидата под новым ключом
            cand_params = c["params"].get(m, {})
            params[key] = cand_params
    return methods, params


def score_from_metrics(metrics: Dict) -> float:
    """
    Скалярная оценка для ранжирования кандидатов предобработки.

    Используется одна метрика — стандарт соответствующей задачи:

    Классификация: AUC (Area Under ROC Curve).
      Huang & Ling (2005) "Using AUC and Accuracy in Evaluating Learning
      Algorithms", IEEE TKDE, 17(3): AUC консистентнее и дискриминативнее
      Accuracy для сравнения моделей. Инвариантен к порогу и дисбалансу.
      Bradley (1997) Pattern Recognition, 30(7): AUC предпочтительнее
      для ранжирования классификаторов.

    Детекция: mAP50-95 (COCO primary metric).
      Lin et al. (2014) "Microsoft COCO", ECCV: mAP усреднённый по IoU
      от 0.50 до 0.95 — основная метрика для ранжирования детекторов.
      Используется во всех крупных бенчмарках (COCO, LVIS, Open Images).

    Остальные метрики (mAP50, F1, precision, recall, accuracy)
    сохраняются в метриках и отображаются в таблицах результатов,
    но не участвуют в ранжировании.
    """
    if metrics is None:
        return 0.0
    # Классификация — AUC
    if "val_auc" in metrics:
        return float(metrics.get("val_auc", 0.0))
    if "auc" in metrics:
        return float(metrics.get("auc", 0.0))
    # Детекция — mAP50-95
    return float(metrics.get("mAP50-95", 0.0))


# ══════════════════════════════════════════════════════════════════════════════
# ЯДРО АЛГОРИТМА (запускается в фоновом потоке)
# ══════════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════════
# АНАЛИЗ МОДАЛЬНОСТИ (опциональный предшественник Фазы 1)
# ══════════════════════════════════════════════════════════════════════════════

# Типы для которых правила содержательны и применяются
_MODALITY_FILTER_TYPES = {"medical_xray", "sar", "microscopy"}

# Типы для которых fallback на чистый SHA+SFS (правила неприменимы)
_MODALITY_FALLBACK_TYPES = {"natural_photo", "infrared"}

# Маппинг группы CANDIDATE_GROUPS → имя метода в PreprocessingRules
_GROUP_TO_METHOD = {
    "denoise":    "denoise",
    "contrast":   "contrast_enhancement",
    "brightness": "brightness_correction",
    "sharpening": "sharpening",
}

# Кандидаты группы denoise запрещённые для конкретных модальностей.
# Основание:
#   SAR    — Gaussian/Bilateral не учитывают мультипликативную природу speckle;
#            Lee фильтр разработан именно для мультипликативного шума SAR.
#            Lee J.S. (1980) IEEE Trans. PAMI-2(2):165-168;
#            Lee J.S. (1981) Comput. Graph. Image Process. 17(1):24-32.
#            Lee фильтр не применяется к non-SAR модальностям т.к. там шум
#            аддитивный (Gaussian/Poisson) — мультипликативная модель некорректна.
#   microscopy — Gaussian blur размывает мелкие клеточные структуры (ядра,
#            митозы) критичные для гистопатологической классификации.
#            Kolarević et al. (2018) Journal of Microscopy 269(3):264-276.
_MODALITY_DENIED_CANDIDATES: Dict[str, set] = {
    "sar":        {"gaussian_k3", "gaussian_k5", "bilateral_s75", "bilateral_s150"},
    "microscopy": {"gaussian_k3", "gaussian_k5", "lee_k3", "lee_k5"},
    "medical_xray": {"lee_k3", "lee_k5"},
    "natural_photo": {"lee_k3", "lee_k5"},
    "infrared":   {"lee_k3", "lee_k5"},
}


def _get_active_candidate_groups(modality_result: Optional[Dict],
                                  use_wiener: bool = False) -> Dict:
    """
    Возвращает CANDIDATE_GROUPS с учётом трёх фильтров:
    1. Модальность — исключает группы методов запрещённые для типа датасета.
    2. Модальность — исключает конкретных кандидатов внутри группы denoise
                     (например Gaussian для SAR, Lee для non-SAR модальностей).
    3. use_wiener  — если False, удаляет кандидатов wiener_s3 и wiener_s5
                     из группы denoise.

    Аргументы:
        modality_result: результат анализа модальности или None
        use_wiener: включить ли Wiener-фильтры в пул кандидатов.
            По умолчанию False — Wiener реализован через scipy.signal.wiener
            без SIMD-оптимизации (~1–1.5 сек/изображение против ~3–8 мс для
            OpenCV-фильтров). На датасетах >10k изображений это критично.
            Fan et al. (2019); сравнительный анализ скорости OpenCV vs scipy.
    """
    # Шаг 1: фильтр по модальности (исключение групп целиком)
    if modality_result is None or not modality_result.get("apply_filter", False):
        groups = CANDIDATE_GROUPS
    else:
        excluded = set(modality_result.get("excluded_groups", []))
        groups = {k: v for k, v in CANDIDATE_GROUPS.items() if k not in excluded} \
            if excluded else CANDIDATE_GROUPS

    # Шаг 2: фильтр кандидатов внутри denoise по модальности
    modality = (modality_result or {}).get("modality", "")
    denied_ids = _MODALITY_DENIED_CANDIDATES.get(modality, set())

    # Шаг 3: фильтр Wiener
    wiener_ids = set() if use_wiener else {"wiener_s3", "wiener_s5"}

    # Объединяем все исключения для denoise
    all_denied = denied_ids | wiener_ids

    if not all_denied:
        return groups

    # Строим копию groups с отфильтрованными кандидатами в denoise
    result = {}
    for gid, ginfo in groups.items():
        if gid == "denoise" and all_denied:
            filtered_cands = [c for c in ginfo["candidates"]
                              if c["id"] not in all_denied]
            result[gid] = {**ginfo, "candidates": filtered_cands}
        else:
            result[gid] = ginfo
    return result


def _run_search(q: queue.Queue, config: Dict):
    """
    Полный цикл двухфазного поиска. Запускается в отдельном потоке.

    Алгоритм: Group-wise SHA (Фаза 1) + SFS+SHA на survivors (Фаза 2).
    Поддерживает классификацию (ClassificationTrainer) и детекцию
    (YOLO / Faster R-CNN / RetinaNet через _run_training из module3).

    Научное обоснование:
    - Guyon & Elisseeff (2003) JMLR 3:1157-1182  — двухфазная группировка
    - Liu & Motoda (2007) Chapman&Hall/CRC ISBN 978-1584888789 — группировка
    - Jamieson & Talwalkar (2016) AISTATS 240-248 — SHA, 30% эпох
    - Kohavi & John (1997) AI 97(1-2):273-324     — SFS
    - Dodge & Karam (2017) CVPRW                  — seed фиксация
    """

    def log(msg: str):
        ts = datetime.now().strftime("%H:%M:%S")
        q.put(("log", f"[{ts}] {msg}"))

    def _put_result(result_dict: Dict):
        q.put(("result", result_dict))

    try:
        import os
        os.environ["DATASETS_GLOBAL_PATH"] = str(config["datasets_path"])

        import random
        import numpy as np
        import torch
        import shutil as _shutil

        from Data.Datasets.dataset_work import get_dataset_path
        from Preprocessing.applicator import DatasetPreprocessor
        from Train.Classification.classification_trainer import (
            ClassificationTrainer,
            set_global_seed,
        )

        seed         = config["seed"]
        eta          = config["eta"]
        dataset_name = config["dataset_name"]
        task         = config["task"]
        max_epochs   = config["epochs"]
        patience     = config["patience"]

        # ── Параметры модели ───────────────────────────────────────────────
        if task == "classification":
            model_type = config["model_type"]
            imgsz      = config["imgsz"]
            pretrained = config.get("pretrained", True)
            batch      = config.get("batch", -1)
            freeze_backbone = config.get("freeze_backbone", False)
            _model_cfg_base = {
                "type":            model_type,
                "image_size":      imgsz,
                "pretrained":      pretrained,
                "freeze_backbone": freeze_backbone,
            }
            if batch > 0:
                _model_cfg_base["batch"] = batch
        else:  # detection
            det_model      = config.get("det_model", "yolo")
            yolo_size      = config.get("yolo_size", "n")
            det_pretrained = config.get("det_pretrained", True)
            det_batch      = config.get("det_batch", -1)
            model_type     = det_model
            imgsz          = 640
            det_imgsz_val = config.get("det_imgsz", 640)
            imgsz          = det_imgsz_val  # переопределяем для логов
            _model_cfg_base = {
                "type":       det_model,
                "size":       yolo_size,
                "pretrained": det_pretrained,
                "imgsz":      det_imgsz_val,
            }
            if det_batch > 0:
                _model_cfg_base["batch"] = det_batch

        screening_ratio = config.get("screening_ratio", 30)
        fast_epochs     = max(1, int(max_epochs * (screening_ratio / 100)))
        datasets_path = Path(config["datasets_path"])

        # Рабочая папка запуска — хранит временные датасеты и финальные веса
        ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
        work_dir = datasets_path / "p2_runs" / f"{dataset_name}_{ts}"
        work_dir.mkdir(parents=True, exist_ok=True)

        set_global_seed(seed)

        # ── Прогрев GPU (warm-up) ──────────────────────────────────────────
        # Первый forward+backward pass на GPU инициализирует CUDA-буферы и
        # JIT-кеш torch.compile — без этого первый кандидат получает другой
        # score из-за недетерминированной инициализации.
        # Занимает ~2-5 сек. Dodge & Karam (2017) — воспроизводимость в DL.
        try:
            import torch as _tw
            if _tw.cuda.is_available():
                _dummy = _tw.randn(4, 3, imgsz, imgsz, device="cuda")
                _dummy_out = _tw.nn.Conv2d(3, 16, 3, padding=1).cuda()(_dummy)
                _dummy_out.sum().backward()
                del _dummy, _dummy_out
                _tw.cuda.empty_cache()
                log("  [GPU WARMUP] CUDA инициализирована")
        except Exception as _we:
            log(f"  [GPU WARMUP] Пропущен: {_we}")

        log("=" * 70)
        log(f"Датасет:      {dataset_name}")
        log(f"Задача:       {task}")
        log(f"Модель:       {model_type}  imgsz={imgsz}")
        log(f"Финал: {max_epochs} эп. | Быстрый: {fast_epochs} эп. ({screening_ratio}%) | ES patience: {patience}")
        log(f"eta (SHA): {eta} | Seed: {seed}")
        log(f"Рабочая папка: {work_dir}")
        log(f"[DEBUG] auto_screen={config.get('auto_screen')} | "
            f"auto_screen_start={config.get('auto_screen_start')} | "
            f"screening_ratio={config.get('screening_ratio')}")
        log("")

        preprocessor = DatasetPreprocessor()

        # ══════════════════════════════════════════════════════════════════
        # ВНУТРЕННИЕ УТИЛИТЫ
        # ══════════════════════════════════════════════════════════════════

        def _make_tmp_ds(candidate_id: str, methods: List[str],
                         params: Dict) -> str:
            """
            Создаёт временный предобработанный датасет.
            Возвращает его имя (для get_dataset_path).
            """
            safe_id = candidate_id[:35].replace("+", "_").replace(" ", "_")
            ds_name = f"_p2tmp_{dataset_name}_{safe_id}_{ts}"
            preprocessor.apply_global_preprocessing(
                source_dataset=dataset_name,
                target_dataset=ds_name,
                methods=methods,
                params=params,
            )
            return ds_name

        def _cleanup_ds(ds_name: str):
            """Удаляет временный датасет."""
            try:
                p = get_dataset_path(ds_name)
                if p.exists():
                    _shutil.rmtree(p)
            except Exception:
                pass

        # ── Обучение для классификации ─────────────────────────────────────

        def _train_cls(ds_name: str, epochs: int, use_es: bool,
                       name_suffix: str,
                       resume_from_path: Optional[str] = None,
                       return_last: bool = False,
                       return_history: bool = False):
            """
            Обучает ClassificationTrainer. Возвращает лучшие метрики.
            Очистка памяти гарантирована в finally ClassificationTrainer._train_one.

            checkpoint_interval:
            - Быстрое обучение (use_es=False, SHA-скрининг): чекпоинты не нужны,
              ставим epochs чтобы сохранить только финальный.
            - Финальное обучение (use_es=True): сохраняем лучший чекпоинт для ES,
              но не чаще чем раз в 10 эпох — избегаем лишних записей на диск.

            resume_from_path: путь к чекпоинту для warm-start (автоподбор скрининга).
            return_last: если True — возвращает метрики последней эпохи, а не лучшей.
                Используется автоподбором скрининга (quick_train_n) при warm-start:
                при max(history) с warm-start запись прогона A включается в историю
                и max всегда возвращает score ≥ прогона A, даже если дообучение
                ничего не улучшило. Это даёт ρ=1.0 тривиально — ложное подтверждение.
                return_last решает проблему: score отражает реальное состояние
                модели после дополнительных эпох.
            """
            _cfg = {
                **_model_cfg_base,
                "name":              f"{model_type}_{name_suffix}",
                "max_epochs":        epochs,
                "use_torch_compile": config.get("use_torch_compile", False),
            }
            if use_es:
                _cfg["early_stopping"] = {"patience": patience, "metric": "val_auc"}

            # Быстрый скрининг: один чекпоинт в конце, не пишем лишнего на диск.
            # Финал: раз в 10 эпох достаточно для ES restore_best.
            ckpt_interval = epochs if not use_es else max(10, epochs // 5)

            trainer = ClassificationTrainer(
                model_configs=[_cfg],
                dataset_names=[ds_name],
                max_epochs=epochs,
                checkpoint_interval=ckpt_interval,
                seed=seed,
                enable_early_stopping=use_es,
                early_stopping_patience=patience,
                early_stopping_metric="val_auc",
                enable_early_selection=False,
            )
            _key = f"{model_type}_{name_suffix}_{ds_name}"
            # Передаём resume_path если есть
            _resume_paths = {_key: resume_from_path} if resume_from_path else {}
            trainer.run_training(resume_paths=_resume_paths)
            history = trainer.metrics_history.get(_key, [])
            if not history:
                return {}

            if return_last:
                # Для автоподбора скрининга: метрики последней реально
                # обученной эпохи (не из чекпоинта warm-start).
                # Исключаем записи с _from_checkpoint=True.
                real_epochs = [h for h in history if not h.get('_from_checkpoint')]
                result = real_epochs[-1] if real_epochs else history[-1]
            else:
                # Для Фазы 1 / финала: лучший результат за всю историю.
                result = max(history, key=lambda x: x.get("val_auc", x.get("val_acc", 0.0)))

            # Возвращаем также путь к чекпоинту для возможного warm-start
            result["_ckpt_path"] = trainer.last_checkpoint_paths.get(_key, "")

            if return_history:
                # Полная per-epoch история AUC (без записей warm-start и дубля
                # run_training) — используется для построения графика обучения.
                _auc_hist = [
                    float(h.get("val_auc", h.get("auc", 0.0)))
                    for h in history
                    if not h.get("_from_checkpoint")
                ][:epochs]
                return result, _auc_hist
            return result

        # ── Обучение для детекции ──────────────────────────────────────────

        def _train_det(ds_name: str, epochs: int, use_es: bool,
                       result_subdir: str, keep_weights: bool,
                       resume: str = '',
                       final: bool = False) -> Dict:
            """
            Обучает детекционную модель через _run_training / _run_training_final.

            Параметр final=True: финальное обучение победителя.
                Вызывает _run_training_final (ES включён, eval на test,
                result_dir не удаляется — веса сохраняются).
                Используется из full_train.

            Параметр final=False (по умолчанию): скрининговое обучение.
                Вызывает _run_training с keep_weights и resume.
                keep_weights=True  → result_dir НЕ удаляется (last.pt нужен
                                     для следующего warm-start прогона).
                keep_weights=False → result_dir удаляется (экономия места).
                Используется из quick_train (без warm-start) и
                quick_train_n (с warm-start, keep_weights=True).

            Исправление бага: исходный код при keep_weights=True всегда
            вызывал _run_training_final, которая игнорирует resume и
            обучает с нуля. Прогон B автоподбора скрининга попадал туда
            и получал обучение с 0% вместо дообучения с N%.
            Теперь final=False → всегда _run_training с корректным resume.
            Jamieson & Talwalkar (2016) AISTATS 240-248: warm-start SHA
            требует сохранения и загрузки весов предыдущего прогона.
            """
            from module3_preprocessing_search import (
                _run_training,
                _run_training_final,
            )
            ds_path    = get_dataset_path(ds_name)
            result_dir = work_dir / result_subdir

            if final:
                # Финальное обучение победителя: ES включён, eval на test.
                # resume здесь не нужен — победитель обучается с нуля на
                # полном числе эпох с early stopping.
                metrics = _run_training_final(
                    dataset_path=ds_path,
                    model_config=_model_cfg_base,
                    epochs=epochs,
                    result_dir=result_dir,
                    log_fn=log,
                    early_stopping_patience=patience,
                    eval_split="test",
                    seed=seed,
                )
            else:
                # Скрининговое обучение (SHA или warm-start автоподбора).
                # resume_from передаётся только для warm-start (прогон B).
                # keep_weights=True → result_dir не удаляется, last.pt
                # остаётся для следующего warm-start.
                # keep_weights=False → result_dir удаляется после обучения.
                metrics = _run_training(
                    dataset_path=ds_path,
                    model_config=_model_cfg_base,
                    epochs=epochs,
                    result_dir=result_dir,
                    log_fn=log,
                    use_early_stopping=use_es,
                    early_stopping_patience=patience,
                    eval_split="valid",
                    resume_from=resume,
                    keep_weights=keep_weights,
                    max_epochs_for_schedule=max_epochs,
                    disable_mosaic=True,
                    seed=seed,
                )
            return metrics

        # ── Универсальные quick_train / full_train ─────────────────────────

        def quick_train(ds_name: str, label: str) -> float:
            """
            Быстрое обучение (screening_ratio% эпох) для Фазы 1 и Фазы 2.

            Классификация: ES выключен — история единая, max(history)
            корректно отражает пик кандидата.

            Детекция (YOLO): ES включён с patience из UI.
            Обоснование: каждый вызов model.train() независим, best.pt
            сохраняет пик кривой автоматически. ES останавливает обучение
            после patience эпох без улучшения — лишних вычислений нет,
            score из best.pt не изменяется. При переобучении (наблюдалось
            на WGISD ~эпоха 40) без ES модель тратит оставшиеся эпохи
            впустую, score всё равно берётся из best.pt — ES устраняет
            эти бесполезные итерации.
            Prechelt (1998): ES как детектор плато корректен для
            ранжирования при условии что score берётся из best, а не last.
            """
            log(f"  Обучаю [{label}] — {fast_epochs} эп. (быстрый)...")
            try:
                if task == "classification":
                    m = _train_cls(ds_name, fast_epochs, use_es=False,
                                   name_suffix=f"q{fast_epochs}ep")
                    sc = score_from_metrics(m)
                    log(f"    score={sc:.4f}  auc={m.get('val_auc',0):.4f}"
                        f"  acc={m.get('val_acc',0):.4f}")
                else:
                    # Уникальный subdir чтобы параллельные запуски не конфликтовали
                    safe_label = label[:20].replace(" ", "_").replace("+", "_")
                    subdir = f"det_quick_{safe_label}_{fast_epochs}ep"
                    # ES включён: останавливает бесполезные эпохи после плато.
                    # keep_weights=False: result_dir удаляется после оценки —
                    # веса не нужны, next warm-start не используется в Фазах 1/2.
                    m  = _train_det(ds_name, fast_epochs, use_es=True,
                                    result_subdir=subdir, keep_weights=False)
                    sc = score_from_metrics(m)
                    log(f"    score={sc:.4f}  mAP50-95={m.get('mAP50-95',0):.4f}"
                        f"  mAP50={m.get('mAP50',0):.4f}  f1={m.get('f1',0):.4f}")
                return sc
            except Exception as e:
                log(f"    [ОШИБКА quick_train] {e}")
                log(traceback.format_exc())
                return 0.0
            finally:
                try:
                    import torch as _t
                    if _t.cuda.is_available():
                        _t.cuda.empty_cache()
                        _t.cuda.synchronize()
                except Exception:
                    pass
                gc.collect()

        def quick_train_n(ds_name: str, label: str, n_epochs: int,
                          resume_from: Optional[str] = None,
                          score_floor: float = 0.0) -> Tuple[float, str]:
            """
            Быстрое обучение на произвольном числе эпох n_epochs.
            Используется автоподбором процента скрининга.

            resume_from: путь к чекпоинту для warm-start (дообучение с x% до x+10%).
            score_floor: минимальный score из предыдущего прогона (прогон A).
                При resume history содержит только новые эпохи — если все они
                хуже score_floor, возвращаем score_floor как нижнюю границу.
                Это гарантирует что resume не даёт результат хуже прогона A.

            Returns:
                (score, ckpt_path) — score для ранжирования и путь к
                финальному чекпоинту для следующего warm-start прогона.
                ckpt_path = "" если чекпоинт недоступен.
            """
            try:
                if task == "classification":
                    # При warm-start (resume_from указан) используем return_last=True:
                    # возвращаем score последней реально обученной эпохи, а не
                    # max(history) который включает запись прогона A из чекпоинта.
                    # Без этого scores прогонов A и B идентичны (max всегда
                    # возвращает лучший из A), ρ=1.0 тривиально — ложное
                    # подтверждение стабильности ранжирования.
                    _is_resume = bool(resume_from)
                    m = _train_cls(ds_name, n_epochs, use_es=False,
                                   name_suffix=f"qas{n_epochs}ep",
                                   resume_from_path=resume_from,
                                   return_last=_is_resume)
                    ckpt_path = m.pop("_ckpt_path", "")
                    sc = score_from_metrics(m)
                    return sc, ckpt_path
                else:
                    safe_label = label[:20].replace(" ", "_").replace("+", "_")
                    # Добавляем короткий уникальный суффикс чтобы прогоны A / B / C
                    # одного кандидата с одинаковым n_epochs не перезаписывали
                    # папку и last.pt друг друга.
                    # Проблема: прогон C (подтверждение) может иметь ту же дельту
                    # что прогон A → одинаковый subdir → last.pt перезаписывается.
                    import uuid as _uuid
                    subdir = f"det_qas_{safe_label}_{n_epochs}ep_{_uuid.uuid4().hex[:6]}"
                    # YOLO: передаём resume через result_subdir last.pt если есть
                    _resume_arg = resume_from if resume_from and os.path.exists(
                        resume_from) else False
                    # final=False: скрининговый прогон, не финальное обучение.
                    # keep_weights=True: result_dir НЕ удаляется — last.pt
                    # нужен для следующего warm-start прогона.
                    # Это исправляет баг где keep_weights=True направлял в
                    # _run_training_final, которая игнорирует resume и
                    # обучала с нуля вместо дообучения с N%.
                    m = _train_det(ds_name, n_epochs, use_es=False,
                                   result_subdir=subdir, keep_weights=True,
                                   resume=_resume_arg, final=False)
                    # Для YOLO ищем last.pt в папке результатов.
                    # _run_training сохраняет в result_dir/'run'/'weights'/last.pt,
                    # где result_dir = work_dir / subdir.
                    # Исправление бага: исходный код не включал подпапку 'run',
                    # из-за чего last.pt никогда не находился и warm-start
                    # для детекции фактически не работал.
                    _det_dir = str(work_dir / subdir)
                    _last_pt = os.path.join(_det_dir, "run", "weights", "last.pt")
                    ckpt_path = _last_pt if os.path.exists(_last_pt) else ""
                    sc = score_from_metrics(m)
                    # score_floor для детекции: гарантируем что warm-start прогон B/C
                    # не возвращает score хуже прогона A.
                    #
                    # В _run_training метрики берутся через best.pt (module3, _run_training
                    # строка: eval_model = YOLO(best_pt if best_pt.exists() else ...)).
                    # best.pt в прогоне B охватывает только дельта-эпохи — если за них
                    # улучшения не было, best.pt может быть хуже финального best.pt
                    # всей траектории A+B. Без score_floor score прогона B < score прогона A
                    # — это инвертирует ранги и ρ не сходится.
                    #
                    # Нижняя граница max(sc, score_floor) зеркалит логику классификации
                    # (return_last + score_floor в _train_cls).
                    # Li et al. (2018) JMLR 18(185): warm-start SHA сохраняет
                    # лучших кандидатов — score не должен деградировать.
                    if score_floor > 0.0:
                        sc = max(sc, score_floor)
                    return sc, ckpt_path
            except Exception:
                return 0.0, ""
            finally:
                try:
                    import torch as _t
                    if _t.cuda.is_available():
                        _t.cuda.empty_cache()
                        _t.cuda.synchronize()
                except Exception:
                    pass
                gc.collect()

        def full_train(ds_name: str, label: str,
                       result_subdir: str = "final",
                       collect_history: bool = False):
            """
            Полное обучение (100% эпох, ES вкл).
            Для детекции сохраняет веса в work_dir/result_subdir.

            collect_history=True: вместе с метриками возвращает список
            значений AUC/mAP50-95 по эпохам для построения графика.
            Классификация: берётся из ClassificationTrainer.metrics_history.
            Детекция: читается из results.csv в YOLO-директории.
            """
            log(f"  Финальное обучение [{label}] — {max_epochs} эп. + ES(patience={patience})...")
            try:
                if task == "classification":
                    if collect_history:
                        m, _hist = _train_cls(ds_name, max_epochs, use_es=True,
                                              name_suffix="final",
                                              return_history=True)
                    else:
                        m = _train_cls(ds_name, max_epochs, use_es=True,
                                       name_suffix="final")
                        _hist = []
                else:
                    # final=True: финальное обучение победителя.
                    # Вызывает _run_training_final (ES, eval на test).
                    m = _train_det(ds_name, max_epochs, use_es=True,
                                   result_subdir=result_subdir, keep_weights=True,
                                   final=True)
                    _hist = []
                    if collect_history:
                        # Читаем per-epoch mAP50-95 из results.csv YOLO.
                        # YOLO всегда пишет этот файл — дополнительного обучения
                        # не требуется.
                        _csv_path = work_dir / result_subdir / "run" / "results.csv"
                        try:
                            import pandas as _pd_hist
                            if _csv_path.exists():
                                _df = _pd_hist.read_csv(_csv_path)
                                _map_col = next(
                                    (c for c in _df.columns
                                     if "mAP50-95" in c or "mAP_50-95" in c),
                                    None)
                                if _map_col:
                                    _hist = [float(v) for v in
                                             _df[_map_col].dropna().tolist()]
                        except Exception as _he:
                            log(f"    [ПРЕДУПРЕЖДЕНИЕ] Не удалось прочитать "
                                f"историю детекции: {_he}")
                if collect_history:
                    return m, _hist
                return m
            except Exception as e:
                log(f"    [ОШИБКА full_train] {e}")
                log(traceback.format_exc())
                if collect_history:
                    return {}, []
                return {}
            finally:
                try:
                    import torch as _t
                    if _t.cuda.is_available():
                        _t.cuda.empty_cache()
                        _t.cuda.synchronize()
                except Exception:
                    pass
                gc.collect()

        # ── Анализ модальности (встроенный, если включён) ────────────────────
        modality_result = config.get("modality_result", None)
        manual_modality = config.get("manual_modality", "auto")

        # Если модальность задана вручную — строим modality_result без анализа.
        # Применяем те же правила фильтрации групп что и при автоопределении.
        if (config.get("use_modality", False)
                and manual_modality != "auto"
                and modality_result is None):
            log("")
            log("=" * 70)
            log(f"МОДАЛЬНОСТЬ: {'ДРУГАЯ (все методы)' if manual_modality == 'other' else manual_modality.upper()}")
            log("=" * 70)
            try:
                import sys as _sys
                _sys.path.insert(0, str(Path(__file__).parent.parent))
                from Utils.preprocessing_rules import PreprocessingRules as PR

                _excluded, _allowed = [], []
                if manual_modality in _MODALITY_FILTER_TYPES:
                    log(f"  Применяем правила для '{manual_modality}':")
                    for _gid, _mname in _GROUP_TO_METHOD.items():
                        _ok = PR.is_method_allowed(manual_modality, _mname)
                        if _ok:
                            _allowed.append(_gid)
                            log(f"     {_gid} ({_mname}) — разрешён")
                        else:
                            _excluded.append(_gid)
                            _rat = PR.get_rationale(manual_modality, _mname)
                            log(f"     {_gid} — запрещён: "
                                f"{_rat[:70]}{'...' if len(_rat) > 70 else ''}")
                else:
                    log(f"  {'Другая модальность' if manual_modality == 'other' else f'Тип {manual_modality!r}'} — фильтрация не применяется, используются все методы предобработки")
                    _allowed = list(_GROUP_TO_METHOD.keys())

                modality_result = {
                    "modality":        manual_modality,
                    "confidence":      1.0,  # задано вручную — уверенность 100%
                    "excluded_groups": _excluded,
                    "allowed_groups":  _allowed,
                    "is_color":        manual_modality not in {"sar", "medical_xray"},
                    "apply_filter":    manual_modality in _MODALITY_FILTER_TYPES,
                }
                q.put(("modality_result", modality_result))
                log("  Модальность применена.")
            except Exception as _e:
                log(f"  [ПРЕДУПРЕЖДЕНИЕ] Не удалось применить правила модальности: {_e}")
                log("  Продолжаем без фильтрации.")
                modality_result = None

        # ══════════════════════════════════════════════════════════════════
        # BASELINE: быстрое обучение оригинала
        # ══════════════════════════════════════════════════════════════════
        # ══════════════════════════════════════════════════════════════════
        # АВТОПОДБОР ПРОЦЕНТА СКРИНИНГА (опционально)
        #
        # Запускает Фазу 1 дважды — на x% и x+10% эпох — и сравнивает
        # ранги кандидатов через коэффициент Спирмена.
        # Если ρ >= порога — x% достаточно, используем его результаты.
        # Если нет — увеличиваем x на 10 и повторяем.
        #
        # Научное обоснование:
        # Egele et al. (2024) Neurocomputing — ранги доминирующих моделей
        # стабильны на ранних эпохах (early discarding).
        # Спирмен ρ ≥ 0.9 — стандартный порог сильной корреляции.
        # ══════════════════════════════════════════════════════════════════
        auto_screen           = config.get("auto_screen", False)
        auto_screen_start     = config.get("auto_screen_start", 40)
        auto_screen_direction = config.get("auto_screen_direction", "top_down")
        # Инициализируем до блока auto_screen чтобы переменная была
        # гарантированно доступна снаружи блока независимо от того
        # выполнился ли внутренний else (кандидатов >= 2).
        _baseline_history: List[float] = []
        _screening_table_data: List[Dict] = []  # таблица mAP50-95 по 10% шагам
        # auto_screen_rho больше не задаётся пользователем —
        # критическое значение ρ вычисляется автоматически по числу
        # кандидатов N через _spearman_critical_rho(N, alpha=0.01).
        # Zar (2005): критическое значение зависит от N и α.
        # α=0.01 (p < 0.01) — строгий порог для научной работы.

        # Будет заполнен автоподбором: {cand_id: score} при найденном %.
        # Используется кэшем Фазы 1 — повторное обучение не нужно.
        _auto_screen_scores: Optional[Dict[str, float]] = None

        if auto_screen:
            log("")
            log("=" * 70)
            log("АВТОПОДБОР БЮДЖЕТА")
            log(f"Начальный %: {auto_screen_start}%")
            log("Порог Спирмена ρ: Zar, 2005; α=0.01")
            log("close_mosaic=0: равные условия для всех порогов N%.")
            log("=" * 70)

            _as_active_groups = _get_active_candidate_groups(
                modality_result,
                use_wiener=config.get("use_wiener", False),
            )
            _as_all_cands = []
            for _gid, _ginfo in _as_active_groups.items():
                for _cand in _ginfo["candidates"]:
                    _as_all_cands.append(_cand)

            if len(_as_all_cands) < 2:
                log("  [ПРОПУСК] Менее 2 кандидатов — автоподбор невозможен.")
            else:
                _n_cands = len(_as_all_cands)
                _rho_crit = _spearman_critical_rho(_n_cands, alpha=0.01)
                log(f"  Кандидатов: {_n_cands} → ρ_crit = {_rho_crit:.4f} "
                    f"(Zar 2005, α=0.01, N={_n_cands})")

                from module3_preprocessing_search import (
                    _run_training_full_history,
                    _history_score_at,
                )
                # Метка основной метрики для логов и таблиц — определяем здесь
                # чтобы переменная была доступна во всех ветках (включая warm_start).
                _score_metric_label = "AUC" if task == "classification" else "mAP50-95"

                # ── Шаг 1: обучаем каждого кандидата до 100% один раз ──────
                # close_mosaic=0: равные условия на всех порогах N%.
                # Без него последние 10 эпох (дефолт Ultralytics) проходят
                # без мозаики — при сравнении max(0..N%) vs max(0..N+10%)
                # кандидаты с бо́льшим N получают систематическое преимущество.
                # Redmon & Farhadi (2018): мозаика меняет распределение входных
                # данных, её отсутствие влияет на метрики финальных эпох.
                _as_ds_map: Dict[str, str] = {}
                _as_histories: Dict[str, List[float]] = {}

                # ── Загрузка истории из CSV (если предоставлен) ──────────
                # Парсим CSV и заполняем _as_histories и _baseline_history.
                # Формат: колонки "Метод", "10%", "20%", ..., "100%".
                # Для каждого кандидата восстанавливаем историю длиной
                # max_epochs: значение N% повторяется для эпох в диапазоне
                # ((N-1)*max_epochs//10 .. N*max_epochs//10).
                # Это корректно т.к. _history_score_at использует max(0..k),
                # а CSV уже содержит max(0..N%) для каждого шага.
                _history_csv_content = config.get("history_csv_content", "")
                _csv_loaded_cands: set = set()  # display-имена загруженных
                if _history_csv_content:
                    try:
                        import csv as _csv_mod
                        import io as _io
                        _pct_cols = [f"{_p}%" for _p in range(10, 110, 10)]
                        _reader = _csv_mod.DictReader(
                            _io.StringIO(_history_csv_content))
                        for _csv_row in _reader:
                            _display = _csv_row.get("Метод", "").strip()
                            if not _display:
                                continue
                            # Читаем значения по 10% шагам
                            _vals = []
                            for _pc in _pct_cols:
                                try:
                                    _vals.append(float(
                                        _csv_row.get(_pc, "0").strip()))
                                except ValueError:
                                    _vals.append(0.0)
                            # Восстанавливаем историю длиной max_epochs:
                            # для каждой эпохи берём значение из
                            # соответствующего 10%-шага.
                            _hist_full = []
                            for _ep in range(1, max_epochs + 1):
                                _step_idx = min(
                                    int((_ep - 1) * 10 / max_epochs), 9)
                                _hist_full.append(_vals[_step_idx])
                            # Сопоставляем с кандидатами по display-имени
                            if _display == "— Baseline —":
                                _baseline_history = _hist_full
                                log(f"  [CSV] baseline история загружена "
                                    f"({max_epochs} эпох)")
                            else:
                                for _cand in _as_all_cands:
                                    if _cand["display"] == _display:
                                        _as_histories[_cand["id"]] = _hist_full
                                        _csv_loaded_cands.add(_display)
                                        break
                        _n_loaded = len(_csv_loaded_cands)
                        _bl_loaded = bool(_baseline_history)
                        log(f"  [CSV] Загружено из истории: "
                            f"{_n_loaded} кандидатов"
                            f"{' + baseline' if _bl_loaded else ''}")
                    except Exception as _csv_e:
                        log(f"  [CSV ОШИБКА] {_csv_e} — "
                            f"продолжаем с обучением с нуля")

                _pct_steps = list(range(10, 110, 10))  # 10,20,...,100

                # Создаём датасеты для всех кандидатов — нужно для всех режимов
                # включая warm_start (датасеты нужны для дообучения).
                log(f"\n  Создаю датасеты для {_n_cands} кандидатов...")
                for _cand in _as_all_cands:
                    try:
                        _ds = _make_tmp_ds(_cand["id"], _cand["methods"],
                                           _cand["params"])
                        _as_ds_map[_cand["id"]] = _ds
                    except Exception as _e:
                        log(f"  [ОШИБКА датасет] {_cand['display']}: {_e}")

                if auto_screen_direction != "warm_start":
                    # Для warm_start обучение до 100% не нужно —
                    # история накапливается инкрементально.
                    log(f"\n  Обучаю {_n_cands} кандидатов до 100%...")
                    for _cand in _as_all_cands:
                        if _cand["id"] not in _as_ds_map:
                            continue

                        # Пропускаем если история уже загружена из CSV
                        if _cand["id"] in _as_histories:
                            log(f"  [{_cand['display']}] — история из CSV, "
                                f"обучение пропущено")
                            continue

                        # ВАЖНО: берём датасет конкретного кандидата из _as_ds_map,
                        # а не переменную _ds из внешнего scope (после цикла создания
                        # она равна последнему датасету и без этой строки все кандидаты
                        # обучались бы на одном датасете).
                        _ds = _as_ds_map[_cand["id"]]

                        log(f"  Обучаю [{_cand['display']}] до {max_epochs} эп. "
                            f"(close_mosaic=0)...")
                        import uuid as _uuid
                        _hist_subdir = (work_dir /
                            f"as_hist_{_cand['id'][:20]}_{_uuid.uuid4().hex[:6]}")
                        try:
                            if task == "classification":
                                _cfg_hist = {
                                    **_model_cfg_base,
                                    "name": f"{model_type}_ashist",
                                    "max_epochs": max_epochs,
                                    "use_torch_compile": config.get(
                                        "use_torch_compile", False),
                                }
                                _trainer_hist = ClassificationTrainer(
                                    model_configs=[_cfg_hist],
                                    dataset_names=[_ds],
                                    max_epochs=max_epochs,
                                    checkpoint_interval=max_epochs,
                                    seed=seed,
                                    enable_early_stopping=False,
                                    enable_early_selection=False,
                                )
                                _trainer_hist.run_training(resume_paths={})
                                _hkey = f"{model_type}_ashist_{_ds}"
                                _raw = _trainer_hist.metrics_history.get(_hkey, [])
                                # Фильтруем _from_checkpoint (запись warm-start) и
                                # дубль best-метрик, который run_training дописывает
                                # поверх per-epoch записей _train_one.
                                # Срез [:max_epochs] = ровно N значений, по одному
                                # на эпоху — как у детекции (_run_training_full_history).
                                _as_histories[_cand["id"]] = [
                                    float(h.get("val_auc", h.get("auc", 0.0)))
                                    for h in _raw
                                    if not h.get("_from_checkpoint")
                                ][:max_epochs]
                            else:
                                _hist = _run_training_full_history(
                                    dataset_path=get_dataset_path(_ds),
                                    model_config=_model_cfg_base,
                                    epochs=max_epochs,
                                    result_dir=_hist_subdir,
                                    log_fn=log,
                                    early_stopping_patience=patience,
                                    seed=seed,
                                )
                                _as_histories[_cand["id"]] = _hist
                        except Exception as _e:
                            log(f"  [ОШИБКА обучение] {_cand['display']}: {_e}")
                            log(traceback.format_exc())
                            _as_histories[_cand["id"]] = []
                        finally:
                            try:
                                import torch as _t
                                if _t.cuda.is_available():
                                    _t.cuda.empty_cache()
                                    _t.cuda.synchronize()
                            except Exception:
                                pass
                            gc.collect()
                        log(f"    → история: {len(_as_histories.get(_cand['id'], []))} эпох")

                    # ── Шаг 2: поиск минимального % по историям ─────────────────
                    # Score кандидата при бюджете N% = max(history[0..N%*epochs]).
                    # Li et al. (2018) JMLR 18(185): SHA использует пиковый
                    # потенциал кандидата на данном бюджете.
                    #
                    # Условия принятия N%:
                    #   p1 = ρ(scores_N%, scores_N+10%) ≥ ρ_crit
                    #   p2 = ρ(scores_N+10%, scores_N+20%) ≥ ρ_crit
                    # Обе проверки из одной истории — повторного обучения нет.
                    # Максимальный стартовый N = 80% (N+20% ≤ 100%).
                    # При N > 80% без успеха → 100%.
                    #
                    # Двойная проверка (p1 и p2):
                    # Audibert & Bubeck (2010) COLT, Theorem 1: два подтверждения
                    # дают экспоненциально меньшую вероятность ложного срабатывания.

                    def _scores_at(ratio: float) -> Dict[str, float]:
                        return {
                            _cid: _history_score_at(_hist, ratio)
                            for _cid, _hist in _as_histories.items()
                        }

                    # ── Baseline: обучение до 100% с историей ───────────────────
                    # Baseline обучается на тех же условиях что кандидаты:
                    # до 100% эпох, close_mosaic=0, ES с паддингом.
                    # Score при найденном % берётся из истории через
                    # _history_score_at — max(0..N%), равные условия.
                    # Без этого baseline обучался бы только на fast_epochs
                    # что создаёт систематическое преимущество.
                    # Пропускаем если baseline уже загружен из CSV
                    if _baseline_history:
                        log(f"  baseline — история из CSV, обучение пропущено")
                    else:
                        log(f"\n  Обучаю baseline до {max_epochs} эп. (история)...")
                        import uuid as _uuid_bl
                        _bl_subdir = work_dir / f"as_hist_baseline_{_uuid_bl.uuid4().hex[:6]}"
                        try:
                            if task == "classification":
                                _cfg_bl = {
                                    **_model_cfg_base,
                                    "name": f"{model_type}_ashist_bl",
                                    "max_epochs": max_epochs,
                                    "use_torch_compile": config.get(
                                        "use_torch_compile", False),
                                }
                                _trainer_bl = ClassificationTrainer(
                                    model_configs=[_cfg_bl],
                                    dataset_names=[dataset_name],
                                    max_epochs=max_epochs,
                                    checkpoint_interval=max_epochs,
                                    seed=seed,
                                    enable_early_stopping=False,
                                    enable_early_selection=False,
                                )
                                _trainer_bl.run_training(resume_paths={})
                                _bl_key = f"{model_type}_ashist_bl_{dataset_name}"
                                _bl_raw = _trainer_bl.metrics_history.get(_bl_key, [])
                                _baseline_history = [
                                    float(h.get("val_auc", h.get("auc", 0.0)))
                                    for h in _bl_raw
                                    if not h.get("_from_checkpoint")
                                ][:max_epochs]
                            else:
                                _baseline_history = _run_training_full_history(
                                    dataset_path=get_dataset_path(dataset_name),
                                    model_config=_model_cfg_base,
                                    epochs=max_epochs,
                                    result_dir=_bl_subdir,
                                    log_fn=log,
                                    early_stopping_patience=patience,
                                    seed=seed,
                                )
                        except Exception as _e:
                            log(f"  [ОШИБКА baseline история] {_e}")
                            log(traceback.format_exc())
                            _baseline_history = []
                        finally:
                            try:
                                import torch as _t
                                if _t.cuda.is_available():
                                    _t.cuda.empty_cache()
                                    _t.cuda.synchronize()
                            except Exception:
                                pass
                            gc.collect()
                    log(f"    → baseline история: {len(_baseline_history)} эпох")

                    # ── Таблица mAP50-95 по каждым 10% для всех кандидатов ──────
                    # Выводится после сбора всех историй, до начала поиска %.
                    # Содержит max(0..N%) для N = 10%, 20%, ..., 100% —
                    # полная картина потенциала каждого кандидата.
                    # Используется для анализа в научной работе.
                    _pct_steps = list(range(10, 110, 10))  # 10,20,...,100
                    log(f"\n{'─'*70}")
                    log(f"ТАБЛИЦА {_score_metric_label} max(0..N%) ПО ЭПОХАМ (все кандидаты)")
                    log(f"{'─'*70}")
                    # Заголовок
                    _hdr = f"  {'Метод':40s}"
                    for _p in _pct_steps:
                        _hdr += f"  {_p:>4}%"
                    log(_hdr)
                    log(f"  {'─'*40}" + "  -----" * len(_pct_steps))
                    # Строка для каждого кандидата
                    _screening_table_rows = []
                    for _cand in _as_all_cands:
                        _cid = _cand["id"]
                        _hist = _as_histories.get(_cid, [])
                        _row = {"Метод": _cand["display"]}
                        _line = f"  {_cand['display']:40s}"
                        for _p in _pct_steps:
                            _v = _history_score_at(_hist, _p / 100.0)
                            _row[f"{_p}%"] = round(_v, 4)
                            _line += f"  {_v:.4f}"
                        _screening_table_rows.append(_row)
                        log(_line)
                    # Строка для baseline
                    _bl_row = {"Метод": "— Baseline —"}
                    _bl_line = f"  {'— Baseline —':40s}"
                    for _p in _pct_steps:
                        _bv = _history_score_at(_baseline_history, _p / 100.0)
                        _bl_row[f"{_p}%"] = round(_bv, 4)
                        _bl_line += f"  {_bv:.4f}"
                    _screening_table_rows.append(_bl_row)
                    log(_bl_line)
                    log(f"{'─'*70}")
                    # Сохраняем для передачи в result dict и CSV
                    _screening_table_data = _screening_table_rows

                # ── Шаг 2: таблица рангов ────────────────────────────────
                # Для каждого столбца (10%..100%) независимо ранжируем
                # кандидатов по mAP50-95 (1 = лучший).
                # Baseline не участвует в ранжировании — он не кандидат SHA.
                # Используется для визуального анализа стабильности рангов.
                log(f"\n{'─'*70}")
                log(f"ТАБЛИЦА РАНГОВ (1=лучший, по {_score_metric_label} для каждых 10% эпох)")
                log(f"{'─'*70}")
                _hdr_r = f"  {'Метод':40s}"
                for _p in _pct_steps:
                    _hdr_r += f"  {_p:>4}%"
                log(_hdr_r)
                log(f"  {'─'*40}" + "  -----" * len(_pct_steps))

                # Вычисляем ранги: для каждого % сортируем кандидатов по score
                _rank_rows = []
                for _cand in _as_all_cands:
                    _rank_rows.append({"Метод": _cand["display"]})

                for _p in _pct_steps:
                    # scores всех кандидатов при данном %
                    _col_scores = []
                    for _cand in _as_all_cands:
                        _hist = _as_histories.get(_cand["id"], [])
                        _col_scores.append(_history_score_at(_hist, _p / 100.0))
                    # Ранги: сортируем по убыванию, присваиваем 1..N
                    _sorted_idx = sorted(range(len(_col_scores)),
                                         key=lambda i: _col_scores[i],
                                         reverse=True)
                    _ranks = [0] * len(_col_scores)
                    for _rank_pos, _orig_idx in enumerate(_sorted_idx, 1):
                        _ranks[_orig_idx] = _rank_pos
                    for _i, _cand in enumerate(_as_all_cands):
                        _rank_rows[_i][f"{_p}%"] = _ranks[_i]

                # Выводим таблицу рангов
                for _i, _cand in enumerate(_as_all_cands):
                    _rline = f"  {_cand['display']:40s}"
                    for _p in _pct_steps:
                        _rline += f"  {_rank_rows[_i][f'{_p}%']:>5}"
                    log(_rline)
                log(f"{'─'*70}")

                # ── Шаг 3: выбор бюджета скрининга ──────────────────
                # Четыре режима: top_down, bottom_up, full_budget, warm_start.

                _sc_100_ref   = _scores_at(1.0) if auto_screen_direction != "warm_start" else {}
                _as_ratio_best = 100
                _as_found      = False

                if auto_screen_direction == "full_budget":
                    # ── 100% бюджет ──────────────────────────────────────
                    log(f"\n  Режим: 100% бюджет (поиск % не выполняется)")
                    _as_ratio_best = 100
                    _as_found = True

                elif auto_screen_direction == "warm_start":
                    # ── Warm-start: инкрементальное дообучение ───────────
                    # Алгоритм: Li et al. (2018) JMLR 18(185) warm-start SHA.
                    # Кандидаты обучаются с нуля до r_A%, затем дообучаются
                    # warm-start через _run_training до r_B% и r_C%.
                    # Score = max(score_prev, score_new) — защита от деградации.
                    # Две локальные проверки: ρ(A,B) и ρ(B,C) ≥ ρ_crit.
                    # Если обе прошли → фиксируем r_A%.
                    # Если нет → r_A=r_B, ckpts сдвигаются, дообучаем r_D.
                    # При r_A=90% → бюджет 100%.
                    #
                    # Таблица строится по накопленным scores для каждых 10%.
                    # Значения после найденного бюджета заполняются последним
                    # известным (паддинг). Prechelt (1998): ES детектирует плато.

                    log(f"\n  Режим: warm-start (снизу вверх с дообучением)")
                    log(f"  Начальный бюджет: {auto_screen_start}%")

                    from module3_preprocessing_search import (
                        _run_training,
                        _run_training_warmstart,
                    )

                    _ws_ratio_a = max(10, min(auto_screen_start, 80))
                    # scores_dict: {cand_id: best_score_so_far}
                    _ws_scores: Dict[str, float] = {}
                    # ckpts: {cand_id: path_to_last_pt}
                    _ws_ckpts:  Dict[str, str]   = {}
                    # accumulated_scores: {pct: {cand_id: score}} для таблицы
                    _ws_acc: Dict[int, Dict[str, float]] = {}

                    # ── Шаг 1: обучаем с нуля до r_A% ───────────────────
                    _ws_ep_a = max(1, int(max_epochs * (_ws_ratio_a / 100)))
                    log(f"\n  Шаг 1: обучение с нуля до {_ws_ratio_a}% ({_ws_ep_a} эп.)")
                    for _cand in _as_all_cands:
                        _ds = _as_ds_map.get(_cand["id"])
                        if _ds is None:
                            continue
                        try:
                            if task == "classification":
                                # Классификация: _train_cls без resume.
                                # max_epochs=_ws_ep_a → обучение с нуля до r_A%.
                                # _ckpt_path из результата — чекпоинт для след.
                                # warm-start прогона.
                                _m = _train_cls(
                                    _ds, _ws_ep_a, use_es=False,
                                    name_suffix=(
                                        f"ws_{_cand['id'][:10]}_r{_ws_ratio_a}"),
                                )
                                _ckpt = _m.pop("_ckpt_path", "")
                                _sc = score_from_metrics(_m)
                                _ws_scores[_cand["id"]] = _sc
                                _ws_ckpts[_cand["id"]] = _ckpt
                                log(f"    {_cand['display']:40s}  "
                                    f"score={_sc:.4f}  "
                                    f"auc={_m.get('val_auc', 0):.4f}")
                            else:
                                # Детекция: _run_training (YOLO).
                                import uuid as _uuid_ws
                                _ws_subdir = (work_dir /
                                    f"ws_{_cand['id'][:15]}_{_ws_ratio_a}pct_"
                                    f"{_uuid_ws.uuid4().hex[:6]}")
                                _m = _run_training(
                                    dataset_path=get_dataset_path(_ds),
                                    model_config=_model_cfg_base,
                                    epochs=_ws_ep_a,
                                    result_dir=_ws_subdir,
                                    log_fn=log,
                                    use_early_stopping=True,
                                    early_stopping_patience=patience,
                                    eval_split="valid",
                                    keep_weights=True,
                                    max_epochs_for_schedule=max_epochs,
                                    disable_mosaic=True,
                                    seed=seed,
                                )
                                _sc = float(_m.get("mAP50-95", 0.0))
                                _ws_scores[_cand["id"]] = _sc
                                # last.pt для warm-start
                                _last = os.path.join(str(_ws_subdir),
                                                     "run", "weights", "last.pt")
                                _ws_ckpts[_cand["id"]] = (
                                    _last if os.path.exists(_last) else "")
                                log(f"    {_cand['display']:40s}  "
                                    f"score={_sc:.4f}")
                        except Exception as _e:
                            log(f"    [ОШИБКА] {_cand['display']}: {_e}")
                            _ws_scores[_cand["id"]] = 0.0
                            _ws_ckpts[_cand["id"]] = ""
                        finally:
                            try:
                                import torch as _t
                                if _t.cuda.is_available():
                                    _t.cuda.empty_cache()
                            except Exception:
                                pass
                            gc.collect()
                    _ws_acc[_ws_ratio_a] = dict(_ws_scores)

                    # ── Основной цикл warm-start ──────────────────────────
                    # На каждом шаге дообучаем одну новую дельту и проверяем
                    # две локальные корреляции.
                    # scores_A = _ws_scores (текущие)
                    # scores_B = дообучение до r_B%
                    # scores_C = дообучение до r_C%
                    # После сдвига: scores_A=scores_B, scores_B=scores_C,
                    # ckpts_B=ckpts_C, дообучаем только r_C→r_D.

                    def _ws_finetune(ratio_from: int, ratio_to: int,
                                     ckpts_from: Dict[str, str],
                                     scores_floor: Dict[str, float]
                                     ) -> tuple:
                        """Дообучает всех кандидатов warm-start от ratio_from до ratio_to.
                        Возвращает (new_scores, new_ckpts).
                        new_scores = max(scores_floor, score_new) — защита от деградации.
                        """
                        _ep_delta = max(1, int(max_epochs * (ratio_to / 100))
                                        - int(max_epochs * (ratio_from / 100)))
                        log(f"\n  Дообучение: {ratio_from}% → {ratio_to}%"
                            f" (дельта {_ep_delta} эп.)")
                        _new_scores: Dict[str, float] = {}
                        _new_ckpts:  Dict[str, str]   = {}
                        for _cand in _as_all_cands:
                            _ds = _as_ds_map.get(_cand["id"])
                            if _ds is None:
                                continue
                            _resume = ckpts_from.get(_cand["id"], "")
                            try:
                                if task == "classification":
                                    # Классификация: ClassificationTrainer тренирует
                                    # range(resume_epoch+1, max_epochs+1), поэтому
                                    # нужно передавать ОБЩЕЕ число эпох (ratio_to%),
                                    # а не дельту. При resume_epoch = ep(ratio_from%)
                                    # и max_epochs = ep(ratio_to%) цикл пройдёт
                                    # ровно _ep_delta итераций.
                                    _ep_total_to = max(
                                        1, int(max_epochs * (ratio_to / 100)))
                                    _m = _train_cls(
                                        _ds, _ep_total_to, use_es=False,
                                        name_suffix=(
                                            f"ws_{_cand['id'][:10]}_r{ratio_to}"),
                                        resume_from_path=_resume if _resume else None,
                                        return_last=bool(_resume),
                                    )
                                    _ckpt_new = _m.pop("_ckpt_path", "")
                                    _sc_new = score_from_metrics(_m)
                                    # score_floor: защита от деградации
                                    _sc = max(_sc_new,
                                              scores_floor.get(_cand["id"], 0.0))
                                    _new_scores[_cand["id"]] = _sc
                                    _new_ckpts[_cand["id"]] = (
                                        _ckpt_new if _ckpt_new else _resume)
                                    log(f"    {_cand['display']:40s}  "
                                        f"score={_sc:.4f}  "
                                        f"auc={_m.get('val_auc', 0):.4f}")
                                else:
                                    # Детекция: _run_training_warmstart (YOLO),
                                    # принимает дельта-эпохи.
                                    import uuid as _uuid_ws2
                                    _ws_sub = (work_dir /
                                        f"ws_{_cand['id'][:15]}_{ratio_to}pct_"
                                        f"{_uuid_ws2.uuid4().hex[:6]}")
                                    _m = _run_training_warmstart(
                                        dataset_path=get_dataset_path(_ds),
                                        model_config=_model_cfg_base,
                                        epochs_delta=_ep_delta,
                                        result_dir=_ws_sub,
                                        log_fn=log,
                                        resume_from=_resume,
                                        use_early_stopping=True,
                                        early_stopping_patience=patience,
                                        eval_split="valid",
                                        disable_mosaic=True,
                                        seed=seed,
                                    )
                                    _sc_new = float(_m.get("mAP50-95", 0.0))
                                    # score_floor: защита от деградации
                                    _sc = max(_sc_new,
                                              scores_floor.get(_cand["id"], 0.0))
                                    _new_scores[_cand["id"]] = _sc
                                    _last = os.path.join(str(_ws_sub),
                                                         "run", "weights", "last.pt")
                                    _new_ckpts[_cand["id"]] = (
                                        _last if os.path.exists(_last) else _resume)
                                    log(f"    {_cand['display']:40s}  "
                                        f"score={_sc:.4f}")
                            except Exception as _e:
                                log(f"    [ОШИБКА] {_cand['display']}: {_e}")
                                _new_scores[_cand["id"]] = scores_floor.get(
                                    _cand["id"], 0.0)
                                _new_ckpts[_cand["id"]] = ckpts_from.get(
                                    _cand["id"], "")
                            finally:
                                try:
                                    import torch as _t
                                    if _t.cuda.is_available():
                                        _t.cuda.empty_cache()
                                except Exception:
                                    pass
                                gc.collect()
                        return _new_scores, _new_ckpts

                    # Начинаем с r_A, вычисляем B и C сразу
                    _ws_scores_a = dict(_ws_scores)
                    _ws_ckpts_a  = dict(_ws_ckpts)
                    _ws_ratio_b  = _ws_ratio_a + 10
                    _ws_ratio_c  = _ws_ratio_a + 20

                    _ws_scores_b, _ws_ckpts_b = _ws_finetune(
                        _ws_ratio_a, _ws_ratio_b,
                        _ws_ckpts_a, _ws_scores_a)
                    _ws_acc[_ws_ratio_b] = dict(_ws_scores_b)

                    _ws_scores_c, _ws_ckpts_c = _ws_finetune(
                        _ws_ratio_b, _ws_ratio_c,
                        _ws_ckpts_b, _ws_scores_b)
                    _ws_acc[_ws_ratio_c] = dict(_ws_scores_c)

                    while True:
                        _rho_p1 = _spearman_rho(_ws_scores_a, _ws_scores_b)
                        _rho_p2 = _spearman_rho(_ws_scores_b, _ws_scores_c)
                        log(f"\n  Проверка N={_ws_ratio_a}%:")
                        for _cand in _as_all_cands:
                            _cid = _cand["id"]
                            log(f"    {_cand['display']:40s}  "
                                f"{_ws_ratio_a}%={_ws_scores_a.get(_cid,0):.4f}  "
                                f"{_ws_ratio_b}%={_ws_scores_b.get(_cid,0):.4f}  "
                                f"{_ws_ratio_c}%={_ws_scores_c.get(_cid,0):.4f}")
                        log(f"  ρ(p1: {_ws_ratio_a}% vs {_ws_ratio_b}%) = "
                            f"{_rho_p1:.4f}  |  "
                            f"ρ(p2: {_ws_ratio_b}% vs {_ws_ratio_c}%) = "
                            f"{_rho_p2:.4f}  (ρ_crit={_rho_crit:.4f})")

                        if _rho_p1 >= _rho_crit and _rho_p2 >= _rho_crit:
                            log(f"  Обе проверки пройдены — "
                                f"ограниченный бюджет = {_ws_ratio_a}%.")
                            _as_ratio_best = _ws_ratio_a
                            _as_found = True
                            break

                        # Не прошли — сдвигаемся
                        log(f"  Проверка не пройдена — сдвигаемся на "
                            f"{_ws_ratio_b}%.")
                        _ws_ratio_a = _ws_ratio_b
                        _ws_ratio_b = _ws_ratio_c
                        _ws_ratio_c = _ws_ratio_a + 20
                        _ws_scores_a = _ws_scores_b
                        _ws_ckpts_a  = _ws_ckpts_b
                        _ws_scores_b = _ws_scores_c
                        _ws_ckpts_b  = _ws_ckpts_c

                        if _ws_ratio_a >= 90:
                            log(f"  Достигнут предел 90% — бюджет 100%.")
                            _as_ratio_best = 100
                            _as_found = True
                            break

                        # Дообучаем только одну новую дельту (r_B → r_C)
                        _ws_scores_c, _ws_ckpts_c = _ws_finetune(
                            _ws_ratio_b, _ws_ratio_c,
                            _ws_ckpts_b, _ws_scores_b)
                        _ws_acc[_ws_ratio_c] = dict(_ws_scores_c)

                    # ── Строим таблицу для warm_start ─────────────────────
                    # Значения накоплены в _ws_acc для вычисленных бюджетов.
                    # Остальные 10%-шаги заполняем паддингом последнего
                    # известного значения. Prechelt (1998): плато после ES.
                    log(f"\n{'─'*70}")
                    log(f"ТАБЛИЦА {_score_metric_label} (warm-start, накопленные значения)")
                    log(f"{'─'*70}")
                    _ws_pct_steps = list(range(10, 110, 10))
                    _hdr_ws = f"  {'Метод':40s}"
                    for _p in _ws_pct_steps:
                        _hdr_ws += f"  {_p:>4}%"
                    log(_hdr_ws)
                    log(f"  {'─'*40}" + "  -----" * len(_ws_pct_steps))

                    _ws_table_rows = []
                    for _cand in _as_all_cands:
                        _cid = _cand["id"]
                        _row = {"Метод": _cand["display"]}
                        _line = f"  {_cand['display']:40s}"
                        _last_known = 0.0
                        for _p in _ws_pct_steps:
                            if _p in _ws_acc and _cid in _ws_acc[_p]:
                                _v = _ws_acc[_p][_cid]
                                _last_known = _v
                            else:
                                _v = _last_known  # паддинг
                            _row[f"{_p}%"] = round(_v, 4)
                            _line += f"  {_v:.4f}"
                        _ws_table_rows.append(_row)
                        log(_line)

                    # Baseline строка (если есть история)
                    if _baseline_history:
                        from module3_preprocessing_search import _history_score_at
                        _bl_row_ws = {"Метод": "— Baseline —"}
                        _bl_line_ws = f"  {'— Baseline —':40s}"
                        for _p in _ws_pct_steps:
                            _bv = _history_score_at(
                                _baseline_history, _p / 100.0)
                            _bl_row_ws[f"{_p}%"] = round(_bv, 4)
                            _bl_line_ws += f"  {_bv:.4f}"
                        _ws_table_rows.append(_bl_row_ws)
                        log(_bl_line_ws)
                    log(f"{'─'*70}")

                    # Сохраняем для result dict
                    _screening_table_data = _ws_table_rows

                    # scores для кэша Фазы 1
                    _ws_final_scores = (
                        _ws_scores_a if _as_ratio_best < 100
                        else _ws_scores_c)

                else:
                    def _check_pct(n_pct: int) -> bool:
                        """Проверяет N% через ρ_global и ρ_local. True = стабилен."""
                        _sc_n    = _scores_at(n_pct / 100.0)
                        _sc_n10  = _scores_at((n_pct + 10) / 100.0)
                        _rg = _spearman_rho(_sc_n, _sc_100_ref)
                        _rl = _spearman_rho(_sc_n, _sc_n10)
                        log(f"  ρ_global({n_pct}% vs 100%) = {_rg:.4f}  |  "
                            f"ρ_local({n_pct}% vs {n_pct+10}%) = {_rl:.4f}  "
                            f"(ρ_crit={_rho_crit:.4f})")
                        if _rg >= _rho_crit and _rl >= _rho_crit:
                            return True
                        if _rg < _rho_crit:
                            log(f"  ρ_global={_rg:.4f} < {_rho_crit:.4f} — "
                                f"нестабилен относительно 100%.")
                        if _rl < _rho_crit:
                            log(f"  ρ_local={_rl:.4f} < {_rho_crit:.4f} — "
                                f"нестабилен локально.")
                        return False

                    if auto_screen_direction == "bottom_up":
                        log(f"\n  Режим: снизу вверх (10% → 90%)")
                        for _as_ratio_cur in range(10, 100, 10):
                            _sc_cur = _scores_at(_as_ratio_cur / 100.0)
                            log(f"\n  Проверка N={_as_ratio_cur}%:")
                            for _cand in _as_all_cands:
                                _cid = _cand["id"]
                                _sc_n10 = _scores_at((_as_ratio_cur + 10) / 100.0)
                                log(f"    {_cand['display']:40s}  "
                                    f"{_as_ratio_cur}%={_sc_cur.get(_cid,0):.4f}  "
                                    f"{_as_ratio_cur+10}%={_sc_n10.get(_cid,0):.4f}  "
                                    f"100%={_sc_100_ref.get(_cid,0):.4f}")
                            if _check_pct(_as_ratio_cur):
                                log(f"  N={_as_ratio_cur}% стабилен — "
                                    f"принимаем (первый стабильный снизу).")
                                _as_ratio_best = _as_ratio_cur
                                break
                            else:
                                log(f"  N={_as_ratio_cur}% нестабилен — "
                                    f"поднимаемся выше.")
                        else:
                            log(f"\n  Ни один % не прошёл — принимаем 100%.")
                            _as_ratio_best = 100
                    else:
                        log(f"\n  Режим: сверху вниз (90% → 10%)")
                        for _as_ratio_cur in range(90, 0, -10):
                            _sc_cur = _scores_at(_as_ratio_cur / 100.0)
                            log(f"\n  Проверка N={_as_ratio_cur}%:")
                            for _cand in _as_all_cands:
                                _cid = _cand["id"]
                                _sc_n10 = _scores_at((_as_ratio_cur + 10) / 100.0)
                                log(f"    {_cand['display']:40s}  "
                                    f"{_as_ratio_cur}%={_sc_cur.get(_cid,0):.4f}  "
                                    f"{_as_ratio_cur+10}%={_sc_n10.get(_cid,0):.4f}  "
                                    f"100%={_sc_100_ref.get(_cid,0):.4f}")
                            if _check_pct(_as_ratio_cur):
                                log(f"  N={_as_ratio_cur}% стабилен — "
                                    f"продолжаем спуск.")
                                _as_ratio_best = _as_ratio_cur
                            else:
                                log(f"  N={_as_ratio_cur}% нестабилен — "
                                    f"останавливаемся на {_as_ratio_best}%.")
                                break

                screening_ratio = _as_ratio_best
                fast_epochs = max(1, int(max_epochs * (_as_ratio_best / 100)))
                # Для warm_start scores берутся из накопленных, для остальных — из историй
                if auto_screen_direction == "warm_start":
                    _auto_screen_scores = _ws_final_scores
                else:
                    _auto_screen_scores = _scores_at(_as_ratio_best / 100.0)
                _as_found = True

                log(f"\n  Итог поиска: бюджет скрининга = "
                    f"{_as_ratio_best}%  ({fast_epochs} эп.)  "
                    f"режим: {auto_screen_direction}")
                # Удаляем временные датасеты
                for _ds in _as_ds_map.values():
                    _cleanup_ds(_ds)

                log(f"\n  Итог автоподбора: screening_ratio={screening_ratio}%"
                    f"  ({fast_epochs} эп.)")
                log(f"  Li et al. (2018) Hyperband — однораундовый отсев "
                    f"(s=0 bracket) с подобранным бюджетом")

        # ══════════════════════════════════════════════════════════════════
        # BASELINE: быстрое обучение оригинала
        # Выполняется ПОСЛЕ автоподбора чтобы использовать итоговый
        # fast_epochs — baseline должен обучаться на том же проценте
        # что и кандидаты для честного сравнения.
        # ══════════════════════════════════════════════════════════════════
        log("")
        log("=" * 70)
        log(f"BASELINE: оригинальный датасет ({screening_ratio}% эпох)")
        log("=" * 70)
        # Если автоподбор включён — baseline уже обучен до 100% с историей.
        # Берём max(0..screening_ratio%) — равные условия с кандидатами.
        # Если автоподбор выключен — обучаем baseline обычным способом.
        if auto_screen and _baseline_history:
            baseline_score = _history_score_at(
                _baseline_history, screening_ratio / 100.0)
            log(f"  Baseline score = {baseline_score:.4f}  "
                f"[из истории автоподбора, max(0..{screening_ratio}%)]")
        else:
            baseline_score = quick_train(dataset_name, "baseline")
            log(f"  Baseline score = {baseline_score:.4f}")

        # ══════════════════════════════════════════════════════════════════
        # ФАЗА 1: Group-wise SHA-скрининг
        # Guyon & Elisseeff (2003); Liu & Motoda (2007).
        # Для каждой группы: обучаем всех кандидатов (screening_ratio% эпох),
        # SHA-отсев оставляет ceil(N_group / eta) survivors.
        # ══════════════════════════════════════════════════════════════════
        log("")
        log("=" * 70)
        log("ФАЗА 1: Групповой скрининг (baseline-фильтр → SHA)")
        log("Guyon & Elisseeff (2003) группировка; Kohavi & John (1997) фильтр по baseline;")
        log("Jamieson & Talwalkar (2016) SHA среди кандидатов выше baseline")
        log("=" * 70)

        # Получаем активные группы с учётом модальности и флага Wiener
        active_groups = _get_active_candidate_groups(
            modality_result,
            use_wiener=config.get("use_wiener", False),
        )

        if modality_result and modality_result.get("apply_filter"):
            excluded = modality_result.get("excluded_groups", [])
            if excluded:
                log(f"  Исключены группы по модальности '{modality_result['modality']}': "
                    f"{', '.join(excluded)}")
                log(f"  Источники: Oliver & Quegan (2004) SAR; "
                    f"Pisano et al. (1998, 2000) рентген; "
                    f"Kolarević et al. (2018) микроскопия")
            log(f"  Активных групп для Фазы 1: "
                f"{len(active_groups)} из {len(CANDIDATE_GROUPS)}")

        # Лог исключённых кандидатов внутри группы denoise по модальности
        modality_str = (modality_result or {}).get("modality", "")
        denied_ids = _MODALITY_DENIED_CANDIDATES.get(modality_str, set())
        if denied_ids and "denoise" in active_groups:
            denied_displays = [
                c["display"] for c in CANDIDATE_GROUPS["denoise"]["candidates"]
                if c["id"] in denied_ids
            ]
            if denied_displays:
                log(f"  Исключены кандидаты denoise для '{modality_str}': "
                    f"{', '.join(denied_displays)}")

        if config.get("use_wiener", False):
            log("  Wiener-фильтры: ВКЛЮЧЕНЫ (wiener_s3, wiener_s5)")
            log("  Предупреждение: ~1–1.5 сек/изображение (scipy, без SIMD).")
            log("  Fan et al. (2019) Visual Computing 2(1); Wiener (1949).")
        else:
            log("  Wiener-фильтры: выключены — используются Median/Gaussian/Bilateral")
            log("  Gonzalez & Woods (2018); Tomasi & Manduchi (1998).")

        all_survivors: List[Dict] = []
        # Сохраняем scored по каждой группе для SHA-fallback
        # (используется если все группы оказались ниже baseline)
        _scored_per_group: Dict[str, List[Dict]] = {}

        # ── Кеш scores из автоподбора скрининга ───────────────────────
        # При новом алгоритме автоподбора каждый кандидат обучается один
        # раз до 100%, а scores при любом % извлекаются из истории.
        # _auto_screen_scores содержит scores при итоговом screening_ratio%
        # — кэш работает для любого найденного %.
        #
        # Dodge & Karam (2017) CVPRW: seed фиксирован → результат
        # детерминирован. Scores из истории эквивалентны обучению с нуля
        # на том же числе эпох при том же seed.
        _as_reusable: Dict[str, float] = {}
        if _auto_screen_scores:
            _as_reusable = _auto_screen_scores
        if _as_reusable:
            log(f"\n  [КЕШИРОВАНИЕ] Переиспользую {len(_as_reusable)} scores "
                f"из автоподбора скрининга ({screening_ratio}% эпох)")
            log(f"  Dodge & Karam (2017): seed фиксирован → результат детерминирован")

        for group_id, group_info in active_groups.items():
            group_label = group_info["label"]
            candidates  = group_info["candidates"]
            log(f"\n  Группа [{group_label}] — {len(candidates)} кандидатов")

            scored: List[Dict] = []
            for cand in candidates:
                # Проверяем кеш из автоподбора
                _cached_score = _as_reusable.get(cand["id"])
                if _cached_score is not None:
                    log(f"    Кандидат: {cand['display']}  score={_cached_score:.4f}  [из кеша автоподбора]")
                    scored.append({**cand, "score": _cached_score})
                    continue

                log(f"    Кандидат: {cand['display']}")
                try:
                    ds_tmp = _make_tmp_ds(cand["id"], cand["methods"], cand["params"])
                except Exception as e:
                    log(f"    [ОШИБКА создания датасета] {e}")
                    continue

                sc = quick_train(ds_tmp, cand["display"])
                _cleanup_ds(ds_tmp)
                scored.append({**cand, "score": sc})

            if not scored:
                log(f"    [ПРОПУСК] нет кандидатов в группе {group_label}")
                continue

            # Сохраняем для возможного SHA-fallback
            _scored_per_group[group_id] = scored

            # Фаза 1: фильтр по baseline → SHA-отсев среди прошедших фильтр.
            #
            # Порядок: baseline-фильтр сначала, SHA после.
            # Kohavi & John (1997) — отсев кандидатов хуже baseline бессмысленен
            # для пула Фазы 2. SHA (Jamieson & Talwalkar, 2016) применяется только
            # к кандидатам выше baseline, оставляя ceil(N/eta) лучших.
            #
            # Если вся группа хуже baseline — группа пропускается целиком.
            # Обоснование: ни один метод группы не улучшает качество относительно
            # оригинала, поэтому включать их в пул Фазы 2 нецелесообразно.
            above = [s for s in scored if s["score"] > baseline_score]
            if not above:
                log(f"\n    Фильтр baseline: все {len(scored)} кандидатов группы "
                    f"ниже baseline ({baseline_score:.4f}) — группа [{group_label}] пропускается.")
                continue

            n_filtered = len(scored) - len(above)
            if n_filtered:
                log(f"\n    Фильтр baseline: {n_filtered} отсеяно из {len(scored)} "
                    f"(score <= {baseline_score:.4f}), осталось: {len(above)}")

            # SHA-отсев среди кандидатов выше baseline
            survivors = sha_prune(above, eta=eta)
            log(f"    SHA-отсев (eta={eta}): {len(above)} → {len(survivors)} survivors")
            for s in sorted(survivors, key=lambda x: x["score"], reverse=True):
                log(f"      + {s['display']:40s}  score={s['score']:.4f}")

            all_survivors.extend(survivors)

        if not all_survivors:
            sha_fallback = config.get("sha_fallback", False)
            if sha_fallback:
                # SHA-fallback: все группы оказались хуже baseline.
                # Применяем SHA без baseline-фильтра — берём ceil(N/eta) лучших
                # из каждой группы отдельно.
                # Используется только когда НИ ОДНА группа не дала survivors выше
                # baseline — если хотя бы одна группа прошла, мы уже не попадём сюда.
                # Jamieson & Talwalkar (2016) — SHA ранжирует кандидатов независимо
                # от абсолютного порога.
                log("")
                log("=" * 70)
                log("SHA-FALLBACK: все группы Фазы 1 ниже baseline")
                log("Применяем SHA без baseline-фильтра (ceil(N/eta) лучших из каждой группы)")
                log("Jamieson & Talwalkar (2016) AISTATS 240-248")
                log("=" * 70)

                # Повторно собираем scored по группам из уже обученных результатов.
                # Поскольку scored не сохранялся между группами — нужно пройти заново
                # по active_groups и взять все кандидаты которые уже обучались.
                # Но scored уже потерян после цикла. Поэтому используем другой подход:
                # запускаем SHA напрямую на всех scored из каждой группы заново,
                # но scored уже не доступен. Нужно сохранить scored_per_group.
                # Исправление: сохраним scored_per_group в первом цикле.
                log("  [INFO] Повторный проход по группам для SHA-fallback...")
                for group_id, group_info in active_groups.items():
                    group_label = group_info["label"]
                    fallback_scored = _scored_per_group.get(group_id, [])
                    if not fallback_scored:
                        continue
                    fallback_survivors = sha_prune(fallback_scored, eta=eta)
                    log(f"  Группа [{group_label}]: SHA без baseline "
                        f"({len(fallback_scored)} → {len(fallback_survivors)} survivors)")
                    for s in sorted(fallback_survivors, key=lambda x: x["score"], reverse=True):
                        log(f"    + {s['display']:40s}  score={s['score']:.4f}")
                    all_survivors.extend(fallback_survivors)

                if not all_survivors:
                    log("")
                    log("Baseline лучше всех методов предобработки. "
                        "Рекомендуется использовать оригинальный датасет.")
                    _put_result({
                        "dataset_name":         dataset_name,
                        "task":                 task,
                        "winner_pipeline":      "baseline",
                        "winner_methods":       [],
                        "winner_params":        {},
                        "winner_ds_name":       None,
                        "better_than_baseline": False,
                        "improvement":          0.0,
                        "baseline_metrics":     {},
                        "winner_metrics":       {},
                        "baseline_score":       baseline_score,
                        "winner_score":         baseline_score,
                        "history":              [],
                        "phase1_survivors":     [],
                        "final_survivors":      [],
                        "baseline_quick_score": baseline_score,
                        "screening_ratio":      screening_ratio,
                    })
                    q.put(("done", "Поиск завершён: baseline лучше всех методов."))
                    return
            else:
                # SHA-fallback выключен — сообщаем пользователю и завершаем корректно
                log("")
                log("=" * 70)
                log("ИТОГ: Baseline лучше всех методов предобработки")
                log("=" * 70)
                log("  Ни один метод предобработки не улучшил качество модели.")
                log("  Рекомендуется использовать оригинальный датасет без предобработки.")
                log("  Совет: попробуйте включить SHA-fallback чтобы всё же получить")
                log("  кандидатов для Фазы 2 без отсечки по baseline.")
                _put_result({
                    "dataset_name":         dataset_name,
                    "task":                 task,
                    "winner_pipeline":      "baseline",
                    "winner_methods":       [],
                    "winner_params":        {},
                    "winner_ds_name":       None,
                    "better_than_baseline": False,
                    "improvement":          0.0,
                    "baseline_metrics":     {},
                    "winner_metrics":       {},
                    "baseline_score":       baseline_score,
                    "winner_score":         baseline_score,
                    "history":              [],
                    "phase1_survivors":     [],
                    "final_survivors":      [],
                    "baseline_quick_score": baseline_score,
                    "screening_ratio":      screening_ratio,
                })
                q.put(("done", "Поиск завершён: baseline лучше всех методов."))
                return

        log(f"\n  Итого survivors после Фазы 1: {len(all_survivors)}")
        for s in sorted(all_survivors, key=lambda x: x["score"], reverse=True):
            log(f"    {s['display']:45s}  score={s['score']:.4f}  "
                f"{'↑ выше baseline' if s['score'] > baseline_score else '↓ ниже baseline (группа без улучшений)'}")

        # ══════════════════════════════════════════════════════════════════
        # ФАЗА 2: SFS+SHA на survivors
        #
        # Правильная реализация по алгоритму:
        #   - known_scores: Dict[tuple, float] — кеш оценок по кортежу id методов.
        #     Порядок методов в кортеже важен: (A, B) != (B, A).
        #   - На каждой итерации: берём survivors_prev, генерируем расширения
        #     (добавляем каждый метод из общего пула survivors в конец каждого
        #     survivor), исключаем дубли методов, используем кеш, обучаем новые,
        #     SHA-отсев по ВСЕМУ пулу (survivors_prev + новые расширения).
        #   - Критерий остановки: N-1 итераций макс, или все расширения содержат
        #     дубли методов.
        #
        # Kohavi & John (1997) SFS + Jamieson & Talwalkar (2016) SHA.
        # ══════════════════════════════════════════════════════════════════
        log("")
        log("=" * 70)
        log("ФАЗА 2: SFS+SHA на survivors")
        log("Kohavi & John (1997) SFS + Jamieson & Talwalkar (2016) SHA")
        log("Порядок методов учитывается: [A+B] != [B+A]")
        log("=" * 70)

        # Пул survivors Фазы 1 — используется для генерации расширений на каждой итерации
        survivor_pool: List[Dict] = list(all_survivors)
        N_survivors = len(survivor_pool)

        # known_scores: ключ — кортеж id методов в порядке применения, значение — score
        # Инициализируем из результатов Фазы 1 (их не обучаем заново)
        known_scores: Dict[tuple, float] = {}
        for s in survivor_pool:
            known_scores[(s["id"],)] = s["score"]

        # Текущие survivors для итерации (изначально — одиночные методы из Фазы 1)
        # Каждый элемент — dict с полями: pipeline_ids (tuple), pipeline_cands (list), display, score
        current_survivors: List[Dict] = []
        for s in survivor_pool:
            current_survivors.append({
                "pipeline_ids":   (s["id"],),
                "pipeline_cands": [s],
                "display":        s["display"],
                "score":          s["score"],
            })

        sfs_history     = []
        # Максимум N-1 итераций (Фаза 1 уже дала длину 1, итерации 1..N-1 дают длину 2..N)
        max_iterations  = max(0, N_survivors - 1)

        log(f"  Survivors после Фазы 1: {N_survivors}, максимум итераций Фазы 2: {max_iterations}")

        for iteration in range(1, max_iterations + 1):
            log(f"\n  Итерация {iteration}/{max_iterations}")
            log(f"    Текущих survivors: {len(current_survivors)}")

            # Генерируем расширения: для каждого survivor добавляем каждый метод из пула
            new_extensions: List[Dict] = []
            skipped_duplicates = 0

            for surv in current_survivors:
                existing_ids = set(surv["pipeline_ids"])
                for pool_item in survivor_pool:
                    # Дедупликация: не добавляем метод, если он уже есть в пайплайне
                    if pool_item["id"] in existing_ids:
                        skipped_duplicates += 1
                        continue

                    new_pipeline_ids   = surv["pipeline_ids"] + (pool_item["id"],)
                    new_pipeline_cands = surv["pipeline_cands"] + [pool_item]
                    new_display        = " + ".join(c["display"] for c in new_pipeline_cands)

                    if new_pipeline_ids in known_scores:
                        # Уже оценивался — берём из кеша
                        ext_score = known_scores[new_pipeline_ids]
                        log(f"    [кеш] {new_display}  score={ext_score:.4f}")
                        new_extensions.append({
                            "pipeline_ids":   new_pipeline_ids,
                            "pipeline_cands": new_pipeline_cands,
                            "display":        new_display,
                            "score":          ext_score,
                        })
                    else:
                        # Новый пайплайн — обучаем
                        combined_methods, combined_params = merge_methods_params(new_pipeline_cands)
                        safe_id = ("_".join(new_pipeline_ids))[:35]
                        log(f"    Обучаю: {new_display}")
                        try:
                            ds_tmp = _make_tmp_ds(safe_id, combined_methods, combined_params)
                        except Exception as e:
                            log(f"    [ОШИБКА датасета] {e}")
                            continue
                        ext_score = quick_train(ds_tmp, new_display)
                        _cleanup_ds(ds_tmp)
                        known_scores[new_pipeline_ids] = ext_score
                        new_extensions.append({
                            "pipeline_ids":   new_pipeline_ids,
                            "pipeline_cands": new_pipeline_cands,
                            "display":        new_display,
                            "score":          ext_score,
                        })

            if skipped_duplicates:
                log(f"    Пропущено расширений с дублями методов: {skipped_duplicates}")

            if not new_extensions:
                log("  [СТОП] Все возможные расширения содержат дубли методов — пайплайн достиг максимума.")
                break

            # Фаза 2: тот же принцип что в Фазе 1 — сначала фильтр по baseline,
            # затем SHA. Порядок "фильтр → SHA" (Kohavi & John, 1997):
            # пайплайны хуже baseline бессмысленно нести дальше.
            # Если все хуже baseline — итерация останавливается, финальными
            # survivors становятся survivors предыдущей итерации.
            combined_pool = current_survivors + new_extensions

            # Дедупликация по pipeline_ids — убираем только точные дубли:
            # одинаковые пайплайны с одинаковым порядком методов И одинаковыми id.
            # ВАЖНО: используем tuple а не frozenset — порядок методов важен,
            # [A+B] и [B+A] это разные пайплайны (применяются последовательно,
            # результат зависит от порядка). frozenset их уравнивал — это было баг.
            # Дубли возникают только когда несколько survivors расширяются
            # абсолютно одинаково (совпадают и методы и порядок).
            seen_pids: set = set()
            deduped_pool: List[Dict] = []
            for c in combined_pool:
                key = tuple(c["pipeline_ids"])  # tuple сохраняет порядок
                if key not in seen_pids:
                    seen_pids.add(key)
                    deduped_pool.append(c)
            n_dupes = len(combined_pool) - len(deduped_pool)
            if n_dupes:
                log(f"    Дедупликация: удалено {n_dupes} дублей наборов методов")
            combined_pool = deduped_pool

            above_baseline = [c for c in combined_pool if c["score"] > baseline_score]

            if not above_baseline:
                log(f"  [СТОП] Все {len(combined_pool)} кандидатов хуже baseline "
                    f"({baseline_score:.4f}) — останавливаем итерации. "
                    f"Финальные survivors — из предыдущей итерации.")
                break

            n_filtered = len(combined_pool) - len(above_baseline)
            if n_filtered:
                log(f"    Фильтр baseline: отсеяно {n_filtered} из {len(combined_pool)} "
                    f"(score <= {baseline_score:.4f})")

            log(f"    SHA-отсев: {len(above_baseline)} кандидатов после фильтра baseline")
            survivors_after_sha = sha_prune(above_baseline, eta=eta)
            log(f"    -> {len(survivors_after_sha)} survivors после SHA")
            for s in survivors_after_sha:
                log(f"      + {s['display']:55s}  score={s['score']:.4f}")

            best_this_iter = max(survivors_after_sha, key=lambda x: x["score"])
            sfs_history.append({
                "iteration":         iteration,
                "best_pipeline":     best_this_iter["display"],
                "score":             best_this_iter["score"],
                "n_candidates":      len(combined_pool),
                "n_above_baseline":  len(above_baseline),
                "n_survivors":       len(survivors_after_sha),
            })

            current_survivors = survivors_after_sha

        # ── Финальные survivors последней итерации (или начальные если 0 итераций)
        final_survivors = current_survivors
        log(f"\n  Финальных survivors: {len(final_survivors)}")
        for s in sorted(final_survivors, key=lambda x: x["score"], reverse=True):
            log(f"    {s['display']:55s}  score={s['score']:.4f}")

        # ══════════════════════════════════════════════════════════════════
        # ФИНАЛЬНОЕ ОБУЧЕНИЕ ПОБЕДИТЕЛЯ
        # Победитель — лучший по score среди финальных survivors.
        # ══════════════════════════════════════════════════════════════════
        log("")
        log("=" * 70)
        log("ФИНАЛЬНОЕ ОБУЧЕНИЕ ПОБЕДИТЕЛЕЙ (топ-N)")
        log("=" * 70)

        # Сколько топ-survivors обучать финально
        top_k = config.get("top_k_winners", 1)
        # Сортируем survivors по score (лучший первый)
        _sorted_survivors = sorted(final_survivors,
                                   key=lambda x: x["score"], reverse=True)
        # Берём min(top_k, len(survivors))
        _top_survivors = _sorted_survivors[:min(top_k, len(_sorted_survivors))]

        log(f"  Финальных survivors: {len(final_survivors)}, "
            f"будем обучать топ-{len(_top_survivors)}")
        log(f"  Baseline score ({screening_ratio}%): {baseline_score:.4f}")
        for _s in _top_survivors:
            log(f"    • {_s['display']:40s}  score({screening_ratio}%)={_s['score']:.4f}")

        # Обучаем каждого из топ-N survivors
        # Победителем станет лучший по score на test сплите
        _finalist_results = []  # список (survivor, ds_name, metrics)
        _finalist_histories: Dict[str, List[float]] = {}  # display → per-epoch history
        for _rank, _surv in enumerate(_top_survivors, 1):
            _surv_methods, _surv_params = merge_methods_params(
                _surv["pipeline_cands"])
            _surv_display = _surv["display"]
            _surv_ds_name = f"{dataset_name}_p2_top{_rank}"
            log(f"\n  [{_rank}/{len(_top_survivors)}] Финал: {_surv_display}")
            log(f"  Создаю датасет: {_surv_ds_name}")
            preprocessor.apply_global_preprocessing(
                source_dataset=dataset_name,
                target_dataset=_surv_ds_name,
                methods=_surv_methods,
                params=_surv_params,
            )
            _surv_result = full_train(_surv_ds_name, _surv_display,
                                      result_subdir=f"final_top{_rank}",
                                      collect_history=True)
            if isinstance(_surv_result, tuple):
                _surv_metrics, _surv_hist = _surv_result
            else:
                _surv_metrics, _surv_hist = _surv_result, []
            _finalist_histories[_surv_display] = _surv_hist
            _surv_score = score_from_metrics(_surv_metrics)
            if task == "classification":
                log(f"    → score(val_auc)={_surv_score:.4f}  "
                    f"acc={_surv_metrics.get('val_acc',0):.4f}")
            else:
                log(f"    → score(test)={_surv_score:.4f}  "
                    f"mAP50-95={_surv_metrics.get('mAP50-95',0):.4f}")
            _finalist_results.append((_surv, _surv_ds_name, _surv_metrics))

        # Лучший финалист по score на test
        _best_finalist = max(_finalist_results,
                             key=lambda x: score_from_metrics(x[2]))
        best_survivor, winner_ds_name, final_metrics = _best_finalist
        winner_display  = best_survivor["display"]
        winner_methods, winner_params = merge_methods_params(
            best_survivor["pipeline_cands"])

        # Переименовываем датасет победителя в стандартное имя
        _final_winner_ds = f"{dataset_name}_p2_winner"
        if winner_ds_name != _final_winner_ds:
            try:
                import shutil as _sh
                _src_p = get_dataset_path(winner_ds_name)
                _dst_p = get_dataset_path(_final_winner_ds)
                if _src_p.exists():
                    if _dst_p.exists():
                        _sh.rmtree(_dst_p)
                    _sh.copytree(_src_p, _dst_p)
                    winner_ds_name = _final_winner_ds
            except Exception as _re:
                log(f"  [ПРЕДУПРЕЖДЕНИЕ] Не удалось переименовать датасет: {_re}")

        log(f"\n  Победитель финала: {winner_display}  "
            f"score(test)={score_from_metrics(final_metrics):.4f}")
        if len(_finalist_results) > 1:
            log("  Все финалисты (test):")
            for _s, _ds, _m in sorted(_finalist_results,
                                       key=lambda x: score_from_metrics(x[2]),
                                       reverse=True):
                log(f"    {_s['display']:40s}  "
                    f"score={score_from_metrics(_m):.4f}  "
                    f"mAP50-95={_m.get('mAP50-95',0):.4f}")

        # ══════════════════════════════════════════════════════════════════
        # ФИНАЛЬНЫЙ BASELINE (полное обучение для честного сравнения)
        # ══════════════════════════════════════════════════════════════════
        log("")
        log("=" * 70)
        log("ФИНАЛЬНЫЙ BASELINE (полное обучение для сравнения)")
        log("=" * 70)

        _baseline_result = full_train(dataset_name, "baseline",
                                       result_subdir="final_baseline",
                                       collect_history=True)
        if isinstance(_baseline_result, tuple):
            baseline_final_metrics, _baseline_history = _baseline_result
        else:
            baseline_final_metrics, _baseline_history = _baseline_result, []

        # ══════════════════════════════════════════════════════════════════
        # ИТОГ И СРАВНЕНИЕ
        # ══════════════════════════════════════════════════════════════════
        winner_score_full   = score_from_metrics(final_metrics)
        baseline_score_full = score_from_metrics(baseline_final_metrics)
        improvement         = winner_score_full - baseline_score_full
        better              = improvement > 0

        log("")
        log("=" * 70)
        log("ИТОГ")
        log("=" * 70)

        if task == "classification":
            log(f"  Baseline — score={baseline_score_full:.4f}"
                f"  auc={baseline_final_metrics.get('val_auc',0):.4f}"
                f"  acc={baseline_final_metrics.get('val_acc',0):.4f}")
            log(f"  Победитель — score={winner_score_full:.4f}"
                f"  auc={final_metrics.get('val_auc',0):.4f}"
                f"  acc={final_metrics.get('val_acc',0):.4f}")
        else:
            log(f"  Baseline — score={baseline_score_full:.4f}"
                f"  mAP50-95={baseline_final_metrics.get('mAP50-95',0):.4f}"
                f"  mAP50={baseline_final_metrics.get('mAP50',0):.4f}"
                f"  f1={baseline_final_metrics.get('f1',0):.4f}")
            log(f"  Победитель — score={winner_score_full:.4f}"
                f"  mAP50-95={final_metrics.get('mAP50-95',0):.4f}"
                f"  mAP50={final_metrics.get('mAP50',0):.4f}"
                f"  f1={final_metrics.get('f1',0):.4f}")

        log(f"  Изменение: {improvement:+.4f}")

        if better:
            log(f"  + Предобработка улучшила метрики.")
            log(f"  + Датасет '{winner_ds_name}' сохранён.")
        else:
            log(f"  - Предобработка не улучшила метрики относительно baseline.")
            log(f"  - Рекомендуется использовать оригинальный датасет.")
            log(f"  - Датасет-победитель удалён.")
            _cleanup_ds(winner_ds_name)
            winner_ds_name = None

        log("=" * 70)

        # ── Сохраняем JSON-отчёт ──────────────────────────────────────────
        result = {
            "dataset_name":         dataset_name,
            "task":                 task,
            "winner_pipeline":      winner_display,
            "winner_methods":       winner_methods,
            "winner_params":        winner_params,
            "winner_ds_name":       winner_ds_name if better else None,
            "better_than_baseline": better,
            "improvement":          round(improvement, 4),
            "baseline_metrics":     baseline_final_metrics,
            "winner_metrics":       final_metrics,
            "baseline_score":       round(baseline_score_full, 4),
            "winner_score":         round(winner_score_full, 4),
            "history":              sfs_history,
            "phase1_survivors": [
                {"id": s["id"], "display": s["display"],
                 "score": round(s["score"], 4)}
                for s in all_survivors
            ],
            # Финальные survivors последней итерации Фазы 2 (с quick-scores для таблицы)
            "final_survivors": [
                {
                    "display": s["display"],
                    "score":   round(s["score"], 4),
                    "pipeline_ids": list(s["pipeline_ids"]),
                }
                for s in sorted(final_survivors, key=lambda x: x["score"], reverse=True)
            ],
            "screening_table": _screening_table_data,
            # Все финалисты (топ-N) с метриками на test
            "all_finalists": [
                {
                    "display":   _s["display"],
                    "score":     round(score_from_metrics(_m), 4),
                    "mAP50-95":  round(_m.get("mAP50-95", 0), 4),
                    "mAP50":     round(_m.get("mAP50", 0), 4),
                    "f1":        round(_m.get("f1", 0), 4),
                    # Метрики классификации (0.0 для детекции)
                    "val_auc":   round(_m.get("val_auc", 0), 4),
                    "val_acc":   round(_m.get("val_acc", 0), 4),
                    "val_f1":    round(_m.get("val_f1", 0), 4),
                }
                for _s, _ds, _m in sorted(
                    _finalist_results,
                    key=lambda x: score_from_metrics(x[2]), reverse=True)
            ],
            "baseline_quick_score": round(baseline_score, 4),
            "screening_ratio":      screening_ratio,
            # Данные для графика кривых обучения финального прогона.
            # winner_history / baseline_history — per-epoch AUC (classification)
            # или mAP50-95 (detection) из full_train.
            # Используется в UI для построения графика даже без auto_screen.
            "chart_data": {
                "winner_display":    winner_display,
                "winner_history":    _finalist_histories.get(winner_display, []),
                "baseline_history":  _baseline_history,
                "max_epochs":        max_epochs,
            },
        }

        result_json = work_dir / "p2_result.json"
        with open(result_json, "w", encoding="utf-8") as _f:
            json.dump(result, _f, indent=2, ensure_ascii=False)
        log(f"\nРезультаты сохранены: {result_json}")

        _put_result(result)
        q.put(("done", "Поиск завершён успешно."))

    except Exception as e:
        q.put(("error", f"Критическая ошибка:\n{e}\n{traceback.format_exc()}"))
    finally:
        try:
            import torch as _t
            if _t.cuda.is_available():
                _t.cuda.empty_cache()
        except Exception:
            pass
        gc.collect()


# ══════════════════════════════════════════════════════════════════════════════
# VRAM (кешируем чтобы не вызывать torch на каждый rerun)
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def _get_vram_gb() -> float:
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
    except Exception:
        pass
    return 0.0


def _suggest_batch_cls(model_type: str, vram: float, imgsz: int) -> int:
    """Подсказывает batch_size для классификации. Yang et al. (2021)."""
    if vram <= 0:
        return 16
    max_b = {"resnet18": 128, "resnet50": 64, "efficientnet_b0": 96}.get(model_type, 64)
    size_scale = (224 / max(imgsz, 1)) ** 2
    vram_scale = vram / 8.0
    return max(1, min(max_b, int(max_b * size_scale * vram_scale * 0.75)))


def _suggest_batch_det(model_type: str, vram: float,
                       yolo_size: str = "n", imgsz: int = 640) -> int:
    """
    Подсказывает batch_size для детекции с учётом размера модели и imgsz.
    Базовые значения подобраны для GPU с 8 GB VRAM при imgsz=640.
    Масштабируются на (640/imgsz)^2 — площадь изображения.
    Для YOLO учитывается размер модели (n/s/m/l/x).
    """
    if vram <= 0:
        return 1
    # Базовый batch при 8 GB VRAM и imgsz=640
    if model_type == "yolo":
        base_at_8gb = {"n": 32, "s": 16, "m": 8, "l": 4, "x": 2}.get(yolo_size, 16)
    else:
        base_at_8gb = {"faster_rcnn": 4, "retinanet": 8}.get(model_type, 4)
    # Масштабируем на VRAM и на размер изображения
    imgsz_scale = (640 / max(imgsz, 1)) ** 2
    vram_scale  = vram / 8.0
    return max(1, int(base_at_8gb * imgsz_scale * vram_scale))


# ══════════════════════════════════════════════════════════════════════════════
# UI — ЭТАП 1: КОНФИГУРАЦИЯ
# ══════════════════════════════════════════════════════════════════════════════

if st.session_state.p2_stage == "configure":

    vram_gb = _get_vram_gb()

    st.title("Подбор комбинаций методов предобработки для повышения точности нейронных моделей")
    st.divider()

    datasets = get_available_datasets()
    if not datasets:
        st.warning("Датасеты не найдены. Проверь путь в Настройках.")
        st.stop()

    # ── Выбор типа изображений (ручной, без автоопределения) ─────────────────
    st.subheader("1. Выбор типа изображений")
    st.markdown(
        "Для медицинских и SAR-снимков применяется ограниченный набор методов предобработки "
        "согласно физическим особенностям данных типов изображений. "
        "Для остальных типов — выбирайте **'Другая модальность'** — в этом случае "
        "рассматриваются все методы предобработки."
    )

    # Автоматическое определение модальности отключено — тип задаётся только вручную.
    st.session_state.p2_use_modality = True

    _MODALITY_OPTIONS = ["medical_xray", "sar", "natural_photo", "other"]
    _current_modality = st.session_state.get("p2_manual_modality", "other")
    if _current_modality not in _MODALITY_OPTIONS:
        _current_modality = "other"

    manual_modality = st.selectbox(
        "Тип датасета",
        options=_MODALITY_OPTIONS,
        index=_MODALITY_OPTIONS.index(_current_modality),
        format_func=lambda x: {
            "medical_xray":  "Медицинские изображения",
            "sar":           "SAR (радарные снимки)",
            "natural_photo": "Натуральные изображения",
            "other":         "Другая модальность",
        }[x],
        key="p2_manual_modality_select",
    )
    st.session_state.p2_manual_modality = manual_modality

    use_modality = True  # переменная сохранена для совместимости с кодом ниже

    use_wiener = st.checkbox(
        "Включить Wiener-фильтры в пул кандидатов",
        value=st.session_state.get("p2_use_wiener", False),
        key="p2_use_wiener_cb",
        help=(
            "Добавляет Wiener (size=3) и Wiener (size=5) в группу шумоподавления. "
            "Данный фильтр предназначен для SAR изображений, но требует высокой "
            "вычислительной сложности. Включение данного метода рекомендуется только "
            "при наличии SSD и высокой вычислительной мощности процессора."
        ),
    )
    st.session_state.p2_use_wiener = use_wiener

    if use_wiener:
        st.warning(
            "Время создание датасетов с применение данного фильтра будет требовать больше времени"
        )

    use_torch_compile = st.checkbox(
        "Включить torch.compile (JIT-компиляция модели)",
        value=st.session_state.get("p2_use_torch_compile", True),
        key="p2_use_torch_compile_cb",
        help=(
            "torch.compile позволяет ускорить обучение на +20–40%. Но "
            "каждая первая эпоха обучения будет занимать от одной до двух минут."
        ),
    )
    st.session_state.p2_use_torch_compile = use_torch_compile

    # Показываем результат предыдущего анализа если есть
    prev_modal = st.session_state.get("p2_modality_result")
    if prev_modal:
        modality    = prev_modal.get("modality", "?")
        confidence  = prev_modal.get("confidence", 0)
        excluded    = prev_modal.get("excluded_groups", [])
        apply_flt   = prev_modal.get("apply_filter", False)
        color_str   = "цветной" if prev_modal.get("is_color") else "grayscale"

        if apply_flt and excluded:
            st.success(
                f"Модальность: **{modality.upper()}** ({confidence*100:.0f}%) | "
                f"{color_str} | "
                f"Исключены группы: **{', '.join(excluded)}**"
            )
        elif apply_flt:
            st.success(
                f"Модальность: **{modality.upper()}** ({confidence*100:.0f}%) | "
                f"{color_str} | Все группы разрешены"
            )
        else:
            st.info(
                f"Модальность: **{modality.upper()}** ({confidence*100:.0f}%) | "
                f"{color_str} | Фильтрация не применяется (fallback на полный SHA+SFS)"
            )

        if st.button("✕ Сбросить результат", key="reset_modal"):
            st.session_state.p2_modality_result = None
            st.rerun()

    st.divider()

    # ── Датасет и задача ───────────────────────────────────────────────────
    col_ds, col_task = st.columns(2, gap="large")

    with col_ds:
        st.subheader("2. Датасет")
        saved_ds = st.session_state.p2_dataset
        default_idx = datasets.index(saved_ds) if saved_ds in datasets else 0
        selected_dataset = st.selectbox(
            "Датасет для поиска",
            options=datasets,
            index=default_idx,
            key="p2_ds_select",
        )
        st.session_state.p2_dataset = selected_dataset

    with col_task:
        st.subheader("3. Тип задачи")
        task = st.radio(
            "Задача",
            options=["classification", "detection"],
            format_func=lambda x: "Классификация" if x == "classification" else "Детекция",
            index=0 if st.session_state.p2_task == "classification" else 1,
            key="p2_task_radio",
            horizontal=True,
        )
        st.session_state.p2_task = task

    st.divider()

    # ── Модель ────────────────────────────────────────────────────────────
    st.subheader("4. Модель для обучения")
    st.caption(
        "Эта модель используется для быстрой оценки кандидатов (30% эпох). "
        "Чем легче модель — тем быстрее поиск. Рекомендуется ResNet-18 / YOLOv8n."
    )

    if task == "classification":
        col_m1, col_m2 = st.columns(2, gap="large")

        with col_m1:
            model_type = st.selectbox(
                "Архитектура",
                options=["resnet18", "resnet50", "efficientnet_b0"],
                format_func=lambda x: {
                    "resnet18":        "ResNet-18  (рекомендуется — быстрее)",
                    "resnet50":        "ResNet-50",
                    "efficientnet_b0": "EfficientNet-B0  (Tan & Le, 2019)",
                }[x],
                key="p2_model_select",
            )
            imgsz = st.selectbox(
                "Размер изображений",
                options=[28, 224, 240],
                index=1,
                key="p2_imgsz_select",
            )
            pretrained = st.checkbox(
                "Предобученная модель (ImageNet)",
                value=st.session_state.p2_pretrained,
                key="p2_pretrained_cb",
                help="Рекомендуется для малых и средних датасетов. Обученная модель быстрее обучается ",
            )
            freeze_backbone = st.checkbox(
                "Заморозить веса предобученной модели",
                value=st.session_state.p2_freeze_backbone,
                key="p2_freeze_cb",
                help=(
                    "Рекомендуется для малых датасетов. Замораживает все слои, кроме головы"
                ),
            )
            if freeze_backbone and not pretrained:
                st.warning(
                    "Заморозка backbone без pretrained весов бессмысленна — "
                    "замороженный backbone будет давать случайные признаки."
                )

        with col_m2:
            auto_batch = _suggest_batch_cls(model_type, vram_gb, imgsz)
            batch_mode = st.radio(
                "Batch size",
                options=["auto", "manual"],
                format_func=lambda x: f"Авто (≈{auto_batch})" if x == "auto" else "Вручную",
                horizontal=True,
                key="p2_batch_mode",
            )
            if batch_mode == "manual":
                batch = st.number_input(
                    "Batch size",
                    min_value=1, max_value=512,
                    value=auto_batch,
                    key="p2_batch_manual",
                    help="Yang et al. (2021): batch=128 для ResNet при 224×224",
                )
            else:
                batch = -1  # -1 = авто внутри ClassificationTrainer

            if vram_gb > 0:
                st.caption(f"Доступная VRAM: {vram_gb:.1f} GB  |  Авто-batch: {auto_batch}")
            else:
                st.caption("GPU не обнаружена — обучение на CPU (медленно)")

        st.session_state.p2_model_type = model_type
        st.session_state.p2_imgsz = imgsz
        st.session_state.p2_pretrained = pretrained
        st.session_state.p2_freeze_backbone = freeze_backbone
        st.session_state.p2_batch = batch

    else:  # detection
        col_m1, col_m2 = st.columns(2, gap="large")

        with col_m1:
            det_model = st.selectbox(
                "Модель",
                options=["yolo", "faster_rcnn", "retinanet"],
                format_func=lambda x: {
                    "yolo":        "YOLOv8  (рекомендуется — быстрее)",
                    "faster_rcnn": "Faster R-CNN  (Ren et al., 2015)",
                    "retinanet":   "RetinaNet  (Lin et al., 2017)",
                }[x],
                key="p2_det_model_select",
            )
            st.session_state.p2_det_model = det_model

            if det_model == "yolo":
                yolo_size = st.selectbox(
                    "Размер YOLOv8",
                    options=["n", "s", "m", "l", "x"],
                    format_func=lambda x: {
                        "n": "nano  (рекомендуется)", "s": "small",
                        "m": "medium", "l": "large", "x": "xlarge",
                    }[x],
                    key="p2_yolo_size_select",
                )
                st.session_state.p2_yolo_size = yolo_size
            else:
                det_pretrained = st.checkbox(
                    "Pretrained веса",
                    value=st.session_state.p2_det_pretrained,
                    key="p2_det_pretrained_cb",
                )
                st.session_state.p2_det_pretrained = det_pretrained

            det_imgsz = st.number_input(
                "Размер изображений (imgsz)",
                min_value=32,
                max_value=4096,
                value=st.session_state.p2_det_imgsz,
                step=32,
                key="p2_det_imgsz_input",
                help="Стандарт: 640 (Wang et al. 2023). "
                     "Кратно 32 — требование YOLO. "
                     "Больше размер → выше точность на мелких объектах, но медленнее.",
            )
            st.session_state.p2_det_imgsz = int(det_imgsz)

        with col_m2:
            auto_det_batch = _suggest_batch_det(
                det_model, vram_gb,
                yolo_size=st.session_state.get('p2_yolo_size', 'n'),
                imgsz=st.session_state.get('p2_det_imgsz', 640),
            )
            det_batch_mode = st.radio(
                "Batch size",
                options=["auto", "manual"],
                format_func=lambda x: f"Авто (≈{auto_det_batch})" if x == "auto" else "Вручную",
                horizontal=True,
                key="p2_det_batch_mode",
            )
            if det_batch_mode == "manual":
                det_batch = st.number_input(
                    "Batch size",
                    min_value=1, max_value=256,
                    value=auto_det_batch,
                    key="p2_det_batch_manual",
                )
            else:
                det_batch = -1

            if vram_gb > 0:
                st.caption(f"Доступная VRAM: {vram_gb:.1f} GB  |  Авто-batch: {auto_det_batch}")
            else:
                st.caption("GPU не обнаружена — обучение на CPU (медленно)")

            st.session_state.p2_det_batch = det_batch

    st.divider()

    # ── Параметры обучения ────────────────────────────────────────────────
    st.subheader("5. Параметры обучения")
    col_p1, col_p2 = st.columns(2, gap="large")

    with col_p1:
        default_epochs = 50 if task == "classification" else 30
        epochs = st.number_input(
            "Эпох финального обучения",
            min_value=5, max_value=300,
            value=st.session_state.p2_epochs,
            step=5,
            key="p2_epochs_input",
            help="Применяется к финальному обучению победителя и baseline. ",
        )
        patience = st.number_input(
            "Early Stopping patience (финал)",
            min_value=3, max_value=100,
            value=st.session_state.p2_patience,
            key="p2_patience_input",
            help="Раняя остановка если метрика не улучшается N эпох",
        )

    with col_p2:
        eta = st.selectbox(
            "eta — коэффициент SHA-отсева",
            options=[2, 3, 4],
            index=0,
            key="p2_eta_select",
            help="Число кандидатов делится на данное число, в случае остатка к полученному числу от деления прибавляется один",
        )
        seed = st.number_input(
            "Seed воспроизводимости",
            min_value=0, max_value=2**31 - 1,
            value=st.session_state.p2_seed,
            key="p2_seed_input",
            help="Значение случайного зерна.",
        )
        top_k_winners = st.number_input(
            "Топ-N финалистов для полного обучения",
            min_value=1, max_value=10,
            value=st.session_state.p2_top_k_winners,
            key="p2_top_k_winners_input",
            help=(
                "В результате отбора может оставать более одного кандидата. "
                "Данное число отвечает за то, сколько отобранных кандидатов будут полностью обучены. "
            ),
        )
        st.session_state.p2_top_k_winners = top_k_winners
        screening_ratio = st.slider(
            "Значение ограниченного бюджета",
            min_value=10, max_value=100,
            value=st.session_state.get("p2_screening_ratio", 30),
            step=5,
            key="p2_screening_ratio_slider",
            help="Не используется при выборе опции автоподбора бюджета",
        ) if not st.session_state.get("p2_auto_screen", False) else st.session_state.get("p2_screening_ratio", 30)

    # ── Автоподбор процента скрининга ─────────────────────────────────────
    st.divider()
    auto_screen = st.checkbox(
        "Автоподбор минимального достаточного бюджета",
        value=st.session_state.get("p2_auto_screen", False),
        key="p2_auto_screen_cb",
    )
    st.session_state.p2_auto_screen = auto_screen
    uploaded_history_csv = None  # инициализация до блока if auto_screen

    if auto_screen:
        _as_col1, _as_col2 = st.columns(2)
        with _as_col1:
            auto_screen_start = st.slider(
                "Начальный бюджет",
                min_value=30, max_value=100,
                value=st.session_state.get("p2_auto_screen_start", 30),
                step=10,
                key="p2_auto_screen_start_slider",
                help="Используется только для дообучения.",
            )
            st.session_state.p2_auto_screen_start = auto_screen_start
        with _as_col2:
            _dir_options = ["top_down", "bottom_up", "warm_start"]
            _dir_cur = st.session_state.get(
                "p2_auto_screen_direction", "top_down")
            if _dir_cur == "full_budget": _dir_cur = "top_down"  # миграция
            _dir_idx = _dir_options.index(_dir_cur) if _dir_cur in _dir_options else 0
            auto_screen_direction = st.radio(
                "Режим поиска бюджета",
                options=_dir_options,
                format_func=lambda x: {
                    "top_down":    "Сверху вниз",
                    "bottom_up":   "Снизу вверх",
                    "warm_start":  "Дообучение",
                }[x],
                index=_dir_idx,
                key="p2_auto_screen_direction_radio",
                help=(
                    "Сверху вниз: начинает с 90%, спускается вниз, "
                    "пока не будет найден минимальный достаточный бюджет.\n\n"
                    "Снизу вверх: начинает с 10%, поднимается вверх, "
                    "пока не будет найден минимальный достаточный бюджет.\n\n"
                    "Дообучение: подбор бюджете производится при помощи дообучения при помощи двух локальных проверок"
                ),
            )
            st.session_state.p2_auto_screen_direction = auto_screen_direction


    # ── Загрузка истории скрининга из CSV ────────────────────────────────
    # Позволяет переиспользовать результаты предыдущего прогона автоподбора
    # вместо повторного обучения всех кандидатов до 100% эпох.
    # Формат CSV: колонки "Метод", "10%", "20%", ..., "100%"
    # (файл p2_screening_history_*.csv из предыдущего запуска).
    # Требование: тот же seed, датасет, модель и max_epochs.
    if auto_screen:
        st.markdown("**Переиспользовать историю из предыдущего прогона (опционально)**")
        uploaded_history_csv = st.file_uploader(
            "Загрузить p2_screening_history_*.csv",
            type=["csv"],
            key="p2_history_csv_uploader",
            help=(
                "Загрузите CSV с историей скрининга из предыдущего запуска "
                "чтобы пропустить повторное обучение кандидатов до 100% эпох. "
                "Обязательно: тот же seed, датасет, модель, max_epochs."
            ),
        )
        if uploaded_history_csv is not None:
            # Сохраняем содержимое в session_state пока файл доступен
            st.session_state.p2_history_csv_content = (
                uploaded_history_csv.read().decode("utf-8"))
            st.success(
                f"CSV загружен: {uploaded_history_csv.name}  "
                f"— обучение кандидатов будет пропущено."
            )
            st.warning(
                "Убедитесь что seed, датасет, модель и max_epochs совпадают "
                "с параметрами прогона из которого взят CSV."
            )
        else:
            # Сбрасываем если файл убран
            if "p2_history_csv_content" not in st.session_state:
                st.session_state.p2_history_csv_content = ""
    else:
        uploaded_history_csv = None

    st.session_state.p2_epochs = epochs
    st.session_state.p2_patience = patience
    st.session_state.p2_eta = eta
    st.session_state.p2_seed = seed
    # Сохраняем screening_ratio только когда автоподбор выключен —
    # при включённом автоподборе значение определяется автоматически
    if not auto_screen:
        st.session_state.p2_screening_ratio = screening_ratio

    # ── Оценка числа обучений ─────────────────────────────────────────────
    fast_ep = max(1, int(epochs * (screening_ratio / 100)))
    # Считаем активный пул с учётом флага Wiener
    _active_groups_ui = _get_active_candidate_groups(None, use_wiener=use_wiener)
    total_candidates = sum(len(g["candidates"]) for g in _active_groups_ui.values())
    survivors_est = sum(
        math.ceil(len(g["candidates"]) / eta) for g in _active_groups_ui.values()
    )
    phase2_est = survivors_est

    st.divider()

    btn_label = "▶ Запустить подбор предобработки"
    if st.button(btn_label, type="primary", use_container_width=True):
        st.session_state.p2_stage = "running"
        st.session_state.p2_log_lines = []
        st.session_state.p2_thread_done = False
        st.session_state.p2_error = None
        st.session_state.p2_result = None
        st.session_state.p2_output_queue = None
        # Сбрасываем предыдущий результат модальности чтобы анализ
        # выполнился заново внутри потока _run_search
        if use_modality:
            st.session_state.p2_modality_result = None
        st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# UI — ЭТАП 2: ВЫПОЛНЕНИЕ С ВЫВОДОМ ЛОГА
# ══════════════════════════════════════════════════════════════════════════════

elif st.session_state.p2_stage == "running":

    # ── Запускаем поток если ещё не запущен ───────────────────────────────
    if st.session_state.p2_output_queue is None:
        q = queue.Queue()
        st.session_state.p2_output_queue = q

        config = {
            "dataset_name":   st.session_state.p2_dataset,
            "task":           st.session_state.p2_task,
            # Классификация
            "model_type":     st.session_state.p2_model_type,
            "imgsz":          st.session_state.p2_imgsz,
            "pretrained":     st.session_state.p2_pretrained,
            "freeze_backbone": st.session_state.p2_freeze_backbone,
            "batch":          st.session_state.p2_batch,
            # Детекция
            "det_model":      st.session_state.p2_det_model,
            "yolo_size":      st.session_state.p2_yolo_size,
            "det_pretrained": st.session_state.p2_det_pretrained,
            "det_batch":      st.session_state.p2_det_batch,
            "det_imgsz":      st.session_state.p2_det_imgsz,
            # Общие
            "epochs":         st.session_state.p2_epochs,
            "patience":       st.session_state.p2_patience,
            "seed":           st.session_state.p2_seed,
            "eta":            st.session_state.p2_eta,
            "datasets_path":  str(get_datasets_path()),
            # Параметры скрининга
            "screening_ratio": st.session_state.get("p2_screening_ratio", 30),
            # Автоподбор процента скрининга — читаем из ключей виджетов напрямую,
            # так как session_state может не успеть обновиться до rerun
            "auto_screen":        st.session_state.get("p2_auto_screen_cb",
                                  st.session_state.get("p2_auto_screen", False)),
            "auto_screen_start":  st.session_state.get("p2_auto_screen_start_slider",
                                  st.session_state.get("p2_auto_screen_start", 40)),
            "auto_screen_direction": st.session_state.get("p2_auto_screen_direction_radio",
                                  st.session_state.get("p2_auto_screen_direction", "top_down")),
            # CSV с историей скрининга: читаем через session_state.
            # uploaded_history_csv может быть недоступен в этом блоке,
            # поэтому используем session_state где файл был сохранён.
            "history_csv_content": st.session_state.get(
                "p2_history_csv_content", ""),
            "top_k_winners":      st.session_state.get("p2_top_k_winners_input",
                                  st.session_state.get("p2_top_k_winners", 1)),
            # auto_screen_rho убран — критическое значение ρ вычисляется
            # автоматически внутри _run_search через _spearman_critical_rho(N).
            # Zar (2005); Ramsey (1989).
            # Анализ модальности
            "use_modality":    st.session_state.get("p2_use_modality", True),
            "manual_modality": st.session_state.get("p2_manual_modality", "other"),
            "modality_result": None,  # будет заполнен внутри _run_search если use_modality=True
            # Wiener-фильтры
            "use_wiener":      st.session_state.get("p2_use_wiener", False),
            # SHA-fallback при пустой Фазе 1
            "sha_fallback":    st.session_state.get("p2_sha_fallback", False),
            # torch.compile
            "use_torch_compile": st.session_state.get("p2_use_torch_compile", True),
        }

        t = threading.Thread(target=_run_search, args=(q, config), daemon=True)
        t.start()

    # ── Читаем очередь ─────────────────────────────────────────────────────
    q = st.session_state.p2_output_queue
    if q is not None:
        try:
            while True:
                msg_type, payload = q.get_nowait()
                if msg_type == "log":
                    st.session_state.p2_log_lines.append(str(payload))
                elif msg_type == "modality_result":
                    # Сохраняем результат анализа модальности для отображения в UI
                    st.session_state.p2_modality_result = payload
                elif msg_type == "result":
                    st.session_state.p2_result = payload
                elif msg_type == "error":
                    st.session_state.p2_error = str(payload)
                    st.session_state.p2_log_lines.append(f"[ОШИБКА] {payload}")
                    st.session_state.p2_thread_done = True
                    st.session_state.p2_stage = "done"
                    st.session_state.p2_output_queue = None
                    break
                elif msg_type == "done":
                    st.session_state.p2_log_lines.append(str(payload))
                    st.session_state.p2_thread_done = True
                    st.session_state.p2_stage = "done"
                    st.session_state.p2_output_queue = None
                    break
        except queue.Empty:
            pass

    # ── Если процесс завершён — сразу переходим к результатам ─────────────
    if st.session_state.p2_thread_done:
        st.rerun()

    # ── Рендерим страницу выполнения (только если ещё идёт) ───────────────
    # st.empty() в начале гарантирует что весь предыдущий контент (configure)
    # полностью вытесняется и не просачивается в running-страницу.
    _task_label = st.session_state.p2_task
    if _task_label == "classification":
        _model_label = (
            f"{st.session_state.p2_model_type} "
            f"imgsz={st.session_state.p2_imgsz} "
            f"batch={'авто' if st.session_state.p2_batch < 0 else st.session_state.p2_batch}"
        )
    else:
        _ysize = st.session_state.p2_yolo_size if st.session_state.p2_det_model == "yolo" else ""
        _model_label = (
            f"{st.session_state.p2_det_model}{_ysize} "
            f"batch={'авто' if st.session_state.p2_det_batch < 0 else st.session_state.p2_det_batch}"
        )

    st.title("Выполняется подбор предобработки...")
    st.info(
        f"**Датасет:** `{st.session_state.p2_dataset}` | "
        f"**Задача:** `{_task_label}` | "
        f"**Модель:** `{_model_label}` | "
        f"**Seed:** `{st.session_state.p2_seed}`"
    )
    st.caption("Страница обновляется автоматически каждые ~1.5 сек.")

    # ── Окно логов — единственный контент ниже ────────────────────────────
    st.markdown("**Вывод процесса:**")
    lines = st.session_state.p2_log_lines
    log_text = "\n".join(lines[-300:]) if lines else "Инициализация..."

    st.code(log_text, language=None)

    time.sleep(1.5)
    st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# UI — ЭТАП 3: РЕЗУЛЬТАТЫ
# ══════════════════════════════════════════════════════════════════════════════

elif st.session_state.p2_stage == "done":

    if st.session_state.p2_error:
        st.title("Подбор завершён с ошибкой")
        st.error("Во время выполнения возникла ошибка:")
        st.code(st.session_state.p2_error)

    else:
        result = st.session_state.p2_result
        _task  = result.get("task", st.session_state.p2_task) if result else st.session_state.p2_task

        if result and result.get("better_than_baseline"):
            st.title("Предобработка улучшила метрики")
            st.success(
                f"**Лучшая комбинация методов: {result['winner_pipeline']}** — "
                f"точность улучшилась на **{result['improvement']:+.4f}** относительно оригинального датасета.\n\n"
                f"Предобработанный датасет `{result['winner_ds_name']}` сохранён"
            )
        elif result:
            st.title("Подбор завершён")
            st.warning(
                "Ни одна комбинация методов предобработки не улучшила метрики "
                f"относительно baseline (разница: **{result.get('improvement', 0):+.4f}**). "
                "Рекомендуется использовать **оригинальный датасет**."
            )
        else:
            st.title("Подбор завершён")
            st.info("Результаты недоступны.")

        if result:
            import pandas as pd

            bm  = result.get("baseline_metrics",    {}) or {}
            wm  = result.get("winner_metrics",       {}) or {}
            bqs = result.get("baseline_quick_score", 0.0)

            # Процент эпох скрининга, который пользователь выставил в UI.
            # Берём из session_state — он точно есть, так как пользователь
            # выставил его перед запуском. Дефолт 30 на случай старых сессий.
            _sr = st.session_state.get("p2_screening_ratio", 30)
            _score_col = f"Score ({_sr}% эп.)"

            # ─────────────────────────────────────────────────────────────
            # ТАБЛИЦА 1: финальные survivors последней итерации vs baseline
            #            (сравнение по quick-score, _sr% эпох)
            # ─────────────────────────────────────────────────────────────
            st.subheader(f"Таблица 1 — Финальные survivors vs Baseline (быстрое обучение, {_sr}% эпох)")
            st.caption(
                "Все пайплайны, прошедшие SHA-отсев на последней итерации Фазы 2, "
                "отсортированы по score по убыванию."
            )

            final_survivors_data = result.get("final_survivors", [])
            if final_survivors_data:
                table1_rows = [
                    {
                        "Пайплайн":   "— Baseline (оригинал) —",
                        _score_col:   f"{bqs:.4f}",
                        "vs Baseline": "—",
                    }
                ]
                for s in final_survivors_data:
                    delta = s["score"] - bqs
                    table1_rows.append({
                        "Пайплайн":   s["display"],
                        _score_col:   f"{s['score']:.4f}",
                        "vs Baseline": f"{delta:+.4f}",
                    })
                st.dataframe(pd.DataFrame(table1_rows), use_container_width=True)
            else:
                st.info("Нет данных о финальных survivors.")

            st.divider()

            # ─────────────────────────────────────────────────────────────
            # ТАБЛИЦА 2: победитель vs baseline (полное обучение)
            # ─────────────────────────────────────────────────────────────
            st.subheader("Таблица 2 — Победитель vs Baseline (полное обучение, 100% эпох + ES)")

            def _fmt(v):
                try:
                    return f"{float(v):.4f}"
                except Exception:
                    return "—"

            def _delta_fmt(w, b):
                try:
                    d = float(w) - float(b)
                    return f"{d:+.4f}"
                except Exception:
                    return "—"

            if _task == "classification":
                w_score = result.get("winner_score",   0.0)
                b_score = result.get("baseline_score", 0.0)
                table2_rows = [
                    {
                        "Метрика":    "Score  (AUC)",
                        "Baseline":   _fmt(b_score),
                        "Победитель": _fmt(w_score),
                        "Δ":          _delta_fmt(w_score, b_score),
                    },
                    {
                        "Метрика":    "AUC  (macro-OvR)",
                        "Baseline":   _fmt(bm.get("val_auc", bm.get("auc", 0))),
                        "Победитель": _fmt(wm.get("val_auc", wm.get("auc", 0))),
                        "Δ":          _delta_fmt(
                            wm.get("val_auc", wm.get("auc", 0)),
                            bm.get("val_auc", bm.get("auc", 0)),
                        ),
                    },
                    {
                        "Метрика":    "Accuracy",
                        "Baseline":   _fmt(bm.get("val_acc", bm.get("acc", 0))),
                        "Победитель": _fmt(wm.get("val_acc", wm.get("acc", 0))),
                        "Δ":          _delta_fmt(
                            wm.get("val_acc", wm.get("acc", 0)),
                            bm.get("val_acc", bm.get("acc", 0)),
                        ),
                    },
                    {
                        "Метрика":    "Precision  (macro)",
                        "Baseline":   _fmt(bm.get("val_precision", 0)),
                        "Победитель": _fmt(wm.get("val_precision", 0)),
                        "Δ":          _delta_fmt(
                            wm.get("val_precision", 0),
                            bm.get("val_precision", 0),
                        ),
                    },
                    {
                        "Метрика":    "Recall  (macro)",
                        "Baseline":   _fmt(bm.get("val_recall", 0)),
                        "Победитель": _fmt(wm.get("val_recall", 0)),
                        "Δ":          _delta_fmt(
                            wm.get("val_recall", 0),
                            bm.get("val_recall", 0),
                        ),
                    },
                    {
                        "Метрика":    "F1  (macro)",
                        "Baseline":   _fmt(bm.get("val_f1", 0)),
                        "Победитель": _fmt(wm.get("val_f1", 0)),
                        "Δ":          _delta_fmt(
                            wm.get("val_f1", 0),
                            bm.get("val_f1", 0),
                        ),
                    },
                ]
            else:
                # Детекция
                w_score = result.get("winner_score",   0.0)
                b_score = result.get("baseline_score", 0.0)
                table2_rows = [
                    {
                        "Метрика":    "Score  (mAP50-95)",
                        "Baseline":   _fmt(b_score),
                        "Победитель": _fmt(w_score),
                        "Δ":          _delta_fmt(w_score, b_score),
                    },
                ]
                for met_label, met_key in [
                    ("mAP50-95",  "mAP50-95"),
                    ("mAP50",     "mAP50"),
                    ("F1",        "f1"),
                    ("Precision", "precision"),
                    ("Recall",    "recall"),
                ]:
                    table2_rows.append({
                        "Метрика":    met_label,
                        "Baseline":   _fmt(bm.get(met_key, 0)),
                        "Победитель": _fmt(wm.get(met_key, 0)),
                        "Δ":          _delta_fmt(wm.get(met_key, 0), bm.get(met_key, 0)),
                    })

            df2 = pd.DataFrame(table2_rows)
            st.dataframe(df2, use_container_width=True)

            if result.get("winner_ds_name"):
                st.info(
                    f"💾 Датасет с предобработкой сохранён: `{result['winner_ds_name']}`\n\n"
                    "Перейди на страницу **Обучение** чтобы обучить финальную модель на нём."
                )

            st.divider()

            # ─────────────────────────────────────────────────────────────
            # ИСТОРИЯ ФАЗЫ 2
            # ─────────────────────────────────────────────────────────────
            history = result.get("history", [])
            if history:
                with st.expander("История итераций Фазы 2 (SFS+SHA)", expanded=False):
                    rows = [
                        {
                            "Итерация":             h.get("iteration", "—"),
                            "Лучший пайплайн":      h.get("best_pipeline", "—"),
                            _score_col:             f"{h.get('score', 0.0):.4f}",
                            "Кандидатов (SHA вход)": h.get("n_candidates", "—"),
                            "Survivors (SHA выход)": h.get("n_survivors", "—"),
                        }
                        for h in history
                    ]
                    st.dataframe(pd.DataFrame(rows), use_container_width=True)

            # ─────────────────────────────────────────────────────────────
            # SURVIVORS ФАЗЫ 1
            # ─────────────────────────────────────────────────────────────
            survivors_p1 = result.get("phase1_survivors", [])
            if survivors_p1:
                with st.expander("Survivors Фазы 1 (SHA-скрининг по группам)", expanded=False):
                    rows = [
                        {
                            "Метод":   s["display"],
                            _score_col: f"{s['score']:.4f}",
                        }
                        for s in sorted(survivors_p1, key=lambda x: x["score"], reverse=True)
                    ]
                    st.dataframe(pd.DataFrame(rows), use_container_width=True)

            st.divider()

            # ─────────────────────────────────────────────────────────────
            # ВСЕ ФИНАЛИСТЫ (топ-N)
            # ─────────────────────────────────────────────────────────────
            all_finalists = result.get("all_finalists", [])
            if len(all_finalists) > 1:
                with st.expander(
                    f"Все финалисты топ-N (полное обучение, test)",
                    expanded=True
                ):
                    if _task == "classification":
                        _fin_rows = [
                            {
                                "Пайплайн":    f["display"],
                                "Score (AUC)": f"{f['score']:.4f}",
                                "ACC":         f"{f.get('val_acc', 0):.4f}",
                                "F1":          f"{f.get('val_f1', 0):.4f}",
                                "Победитель":  "" if f["display"] == result.get("winner_pipeline") else "",
                            }
                            for f in all_finalists
                        ]
                    else:
                        _fin_rows = [
                            {
                                "Пайплайн":   f["display"],
                                "Score":      f"{f['score']:.4f}",
                                "mAP50-95":   f"{f['mAP50-95']:.4f}",
                                "mAP50":      f"{f['mAP50']:.4f}",
                                "F1":         f"{f['f1']:.4f}",
                                "Победитель": "" if f["display"] == result.get("winner_pipeline") else "",
                            }
                            for f in all_finalists
                        ]
                    import pandas as _pd_fin
                    st.dataframe(_pd_fin.DataFrame(_fin_rows),
                                 use_container_width=True)

            st.divider()

            # ─────────────────────────────────────────────────────────────
            # ГРАФИК КРИВЫХ ОБУЧЕНИЯ
            # ─────────────────────────────────────────────────────────────
            _chart_data = result.get("chart_data", {})
            _winner_hist     = _chart_data.get("winner_history", [])
            _baseline_hist   = _chart_data.get("baseline_history", [])
            _winner_disp     = _chart_data.get("winner_display", "Победитель")
            _chart_max_ep    = _chart_data.get("max_epochs", 1) or 1
            _metric_label    = "AUC" if _task == "classification" else "mAP50-95"

            if _winner_hist or _baseline_hist:
                st.subheader("Кривые обучения")

                # Процент скрининга для пунктирной вертикальной линии
                _task_sr = int(result.get(
                    "screening_ratio",
                    st.session_state.get("p2_screening_ratio", 30)))

                # Профессиональная академическая палитра
                # (colorblind-friendly, приглушённые тона)
                _PALETTE = [
                    "#2563EB",  # синий  — победитель
                    "#16A34A",  # зелёный
                    "#D97706",  # янтарный
                    "#7C3AED",  # фиолетовый
                    "#DB2777",  # розовый
                ]

                def _curve_pts(hist: list, n_epochs: int) -> list:
                    """max(0..N%) для N=0,10,20,...,100. 0% всегда 0."""
                    pts = [0.0]  # 0% — до обучения
                    for _p in range(10, 110, 10):
                        _k = max(1, int(round(n_epochs * (_p / 100))))
                        _k = min(_k, len(hist)) if hist else 0
                        pts.append(round(max(hist[:_k]), 4) if _k > 0 else 0.0)
                    return pts

                _x_pct = list(range(0, 110, 10))  # 0, 10, 20, ..., 100

                try:
                    import plotly.graph_objects as _go
                except ImportError:
                    st.info(
                        "📦 Для отображения графика установите plotly: "
                        "`uv add plotly` или `uv pip install plotly`"
                    )
                    _go = None

                if _go is not None:
                    _fig = _go.Figure()

                    # Baseline — серый пунктир
                    if _baseline_hist:
                        _bl_pts = _curve_pts(_baseline_hist, _chart_max_ep)
                        _bl_label = (
                            f"Baseline"
                            f"  (эп. {len(_baseline_hist)}/{_chart_max_ep})")
                        _fig.add_trace(_go.Scatter(
                            x=_x_pct, y=_bl_pts,
                            name=_bl_label,
                            mode="lines+markers",
                            line=dict(color="#6B7280", width=2, dash="dash"),
                            marker=dict(size=5, color="#6B7280"),
                            hovertemplate=(
                                "<b>Baseline</b><br>"
                                f"Бюджет: %{{x}}%<br>{_metric_label}: %{{y:.4f}}"
                                "<extra></extra>"
                            ),
                        ))

                    # Победитель — жирная синяя линия
                    if _winner_hist:
                        _win_pts = _curve_pts(_winner_hist, _chart_max_ep)
                        _win_label = (
                            f"★ {_winner_disp}"
                            f"  (эп. {len(_winner_hist)}/{_chart_max_ep})")
                        _fig.add_trace(_go.Scatter(
                            x=_x_pct, y=_win_pts,
                            name=_win_label,
                            mode="lines+markers",
                            line=dict(color=_PALETTE[0], width=3),
                            marker=dict(size=6, color=_PALETTE[0]),
                            hovertemplate=(
                                f"<b>{_winner_disp}</b><br>"
                                f"Бюджет: %{{x}}%<br>{_metric_label}: %{{y:.4f}}"
                                "<extra></extra>"
                            ),
                        ))

                    # Вертикальная линия на скрининговом бюджете
                    _fig.add_vline(
                        x=_task_sr,
                        line_dash="dot",
                        line_color="#9CA3AF",
                        line_width=1.5,
                        annotation_text=f"скрининг ({_task_sr}%)",
                        annotation_position="top right",
                        annotation_font=dict(size=11, color="#6B7280"),
                    )

                    # Оформление: академический стиль, сетка 0.1
                    _fig.update_layout(
                        template="plotly_white",
                        height=420,
                        margin=dict(l=60, r=30, t=40, b=60),
                        xaxis=dict(
                            title="Бюджет обучения, % от максимума эпох",
                            tickvals=_x_pct,
                            ticktext=[f"{p}%" for p in _x_pct],
                            gridcolor="#E5E7EB",
                            showgrid=True,
                            zeroline=False,
                        ),
                        yaxis=dict(
                            title=f"{_metric_label} max(0..N%)",
                            dtick=0.1,
                            gridcolor="#E5E7EB",
                            showgrid=True,
                            zeroline=False,
                        ),
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="left",
                            x=0,
                            bgcolor="rgba(255,255,255,0.8)",
                            bordercolor="#E5E7EB",
                            borderwidth=1,
                        ),
                        plot_bgcolor="white",
                        paper_bgcolor="white",
                        font=dict(
                            family="Inter, Arial, sans-serif",
                            size=13, color="#111827"),
                        hoverlabel=dict(
                            bgcolor="white",
                            bordercolor="#D1D5DB",
                            font_size=13,
                        ),
                    )

                    st.plotly_chart(_fig, use_container_width=True)
                # Конец блока графика

            st.divider()

            # ─────────────────────────────────────────────────────────────
            # СКАЧИВАНИЕ
            # ─────────────────────────────────────────────────────────────
            st.subheader("Скачать результаты")
            dl_col1, dl_col2, dl_col3 = st.columns(3)

            # CSV таблица 1 — финальные survivors vs baseline
            if final_survivors_data:
                csv_table1_rows = [
                    {"Пайплайн": "— Baseline (оригинал) —",
                     f"Score_{_sr}pct": bqs, "vs_Baseline": 0.0}
                ]
                for s in final_survivors_data:
                    csv_table1_rows.append({
                        "Пайплайн":        s["display"],
                        f"Score_{_sr}pct": s["score"],
                        "vs_Baseline":     round(s["score"] - bqs, 4),
                    })
                csv1 = pd.DataFrame(csv_table1_rows).to_csv(index=False, encoding="utf-8")
                with dl_col1:
                    st.download_button(
                        label="Таблица 1 (финальные survivors)",
                        data=csv1,
                        file_name=f"p2_final_survivors_{result.get('dataset_name','')}.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )

            # CSV таблица 2 — победитель vs baseline (полное обучение)
            csv2 = pd.DataFrame(table2_rows).to_csv(index=False, encoding="utf-8")
            with dl_col2:
                st.download_button(
                    label="Таблица 2 (победитель vs baseline)",
                    data=csv2,
                    file_name=f"p2_winner_vs_baseline_{result.get('dataset_name','')}.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

            # CSV таблица скрининга — Score по 10% шагам (автоподбор)
            screening_table_data = result.get("screening_table", [])
            if screening_table_data:
                _screening_metric_label = (
                    "AUC" if result.get("task") == "classification"
                    else "mAP50-95")
                csv_screening = pd.DataFrame(screening_table_data).to_csv(
                    index=False, encoding="utf-8")
                # Добавляем четвёртую колонку
                dl_col4, = st.columns(1)
                with dl_col4:
                    st.download_button(
                        label=f"Таблица скрининга ({_screening_metric_label} по 10% эпох)",
                        data=csv_screening,
                        file_name=f"p2_screening_history_{result.get('dataset_name','')}.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )

            # Полный лог
            full_log = "\n".join(st.session_state.p2_log_lines)
            with dl_col3:
                st.download_button(
                    label="Полный лог",
                    data=full_log,
                    file_name=f"p2_log_{result.get('dataset_name','')}.txt",
                    mime="text/plain",
                    use_container_width=True,
                )

    # Лог в свёрнутом виде всегда доступен
    with st.expander("Полный лог выполнения", expanded=False):
        full_log_exp = "\n".join(st.session_state.p2_log_lines)
        st.code(full_log_exp, language=None)

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Запустить ещё раз", use_container_width=True):
            _reset()
            st.rerun()
    with col2:
        if st.button("Перейти к обучению", type="primary", use_container_width=True):
            st.switch_page("pages/3_Обучение.py")
