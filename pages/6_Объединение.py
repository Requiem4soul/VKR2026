"""
pages/6_Объединение.py — Объединённый подбор пайплайна предобработки

Алгоритм: Анализ модальности (опц.) + Group-wise SHA (Фаза 1) + SFS+SHA на survivors (Фаза 2)

Расширение 2_Предобработка.py:
    Перед Фазой 1 опционально запускается анализ метрик изображений
    (UniversalImageAnalyzer + ImageModalityClassifier из 5_Метрика_предобработка.py).
    Для модальностей medical_xray / sar / microscopy применяются правила
    PreprocessingRules — группы методов запрещённые для данной модальности
    исключаются из CANDIDATE_GROUPS до начала Фазы 1.
    Для natural_photo / infrared используется чистый SHA+SFS без фильтрации.

    Научное обоснование фильтрации по модальности:
    - SAR: Oliver & Quegan (2004) — brightness/sharpening искажают физическую информацию
    - Medical: Pisano et al. (1998) J.Digital Imaging 11(4):193-200; Pisano et al. (2000) RadioGraphics 20:1479-1491
    - Microscopy: Kolarević et al. (2018) Journal of Microscopy 269(3):264-276; Sternberg (1983) Computer 16(1):22-34
    Выборка для анализа: min(N_train × 0.3, 300), минимум 30
    (CLT: Kim, 2017, PMC5370305 — n ≥ 30 достаточно для оценки среднего)

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
    "p2_manual_modality": "auto",   # "auto" или одна из модальностей
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
    Вычисляет скалярную оценку из метрик классификации или детекции.
    Для классификации: AUC (приоритет) или ACC.
    Для детекции: взвешенная сумма mAP метрик.
    """
    if metrics is None:
        return 0.0
    # Классификация
    if "val_auc" in metrics:
        return 0.6 * metrics.get("val_auc", 0.0) + 0.4 * metrics.get("val_acc", 0.0)
    if "auc" in metrics:
        return 0.6 * metrics.get("auc", 0.0) + 0.4 * metrics.get("acc", 0.0)
    # Детекция (аналог composite_score из module3)
    return (
        0.45 * metrics.get("mAP50-95", 0.0)
        + 0.35 * metrics.get("mAP50", 0.0)
        + 0.20 * metrics.get("f1", 0.0)
    )


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
        log("ДВУХФАЗНЫЙ ПОДБОР ПАЙПЛАЙНА ПРЕДОБРАБОТКИ")
        log("Алгоритм: Group-wise SHA (Ф.1) + SFS+SHA на survivors (Ф.2)")
        log("Источники: Guyon & Elisseeff (2003) JMLR 3:1157-1182;")
        log("           Liu & Motoda (2007) Chapman&Hall/CRC ISBN 978-1584888789;")
        log("           Jamieson & Talwalkar (2016) AISTATS 240-248;")
        log("           Kohavi & John (1997) AI 97(1-2):273-324")
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
                       resume_from_path: Optional[str] = None) -> Dict:
            """
            Обучает ClassificationTrainer. Возвращает лучшие метрики.
            Очистка памяти гарантирована в finally ClassificationTrainer._train_one.

            checkpoint_interval:
            - Быстрое обучение (use_es=False, SHA-скрининг): чекпоинты не нужны,
              ставим epochs чтобы сохранить только финальный.
            - Финальное обучение (use_es=True): сохраняем лучший чекпоинт для ES,
              но не чаще чем раз в 10 эпох — избегаем лишних записей на диск.

            resume_from_path: путь к чекпоинту для warm-start (автоподбор скрининга).
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
            best = max(history, key=lambda x: x.get("val_auc", x.get("val_acc", 0.0)))
            # Возвращаем также путь к чекпоинту для возможного warm-start
            best["_ckpt_path"] = trainer.last_checkpoint_paths.get(_key, "")
            return best

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
                    use_early_stopping=False,
                    early_stopping_patience=patience,
                    eval_split="valid",
                    resume_from=resume,
                    keep_weights=keep_weights,
                )
            return metrics

        # ── Универсальные quick_train / full_train ─────────────────────────

        def quick_train(ds_name: str, label: str) -> float:
            """
            Быстрое обучение (30% эпох, ES выкл).
            Возвращает scalar score для ранжирования SHA.
            Jamieson & Talwalkar (2016) — % эпох для скрининга задаётся пользователем.
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
                    m  = _train_det(ds_name, fast_epochs, use_es=False,
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
                    m = _train_cls(ds_name, n_epochs, use_es=False,
                                   name_suffix=f"qas{n_epochs}ep",
                                   resume_from_path=resume_from)
                    ckpt_path = m.pop("_ckpt_path", "")
                    sc = score_from_metrics(m)
                    # score_floor здесь не применяется: _train_one уже возвращает
                    # best по всей истории включая запись прогона A (добавляется
                    # в metrics_history из чекпоинта перед циклом обучения).
                    # Применение score_floor поверх этого давало эффект что все
                    # кандидаты прогона B получали ровно те же scores что прогон A
                    # (если прогон B не улучшил результат), что приводило к ρ=1.0
                    # и ложному выводу о достаточности % скрининга.
                    return sc, ckpt_path
                else:
                    safe_label = label[:20].replace(" ", "_").replace("+", "_")
                    subdir = f"det_qas_{safe_label}_{n_epochs}ep"
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
                    return score_from_metrics(m), ckpt_path
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
                       result_subdir: str = "final") -> Dict:
            """
            Полное обучение (100% эпох, ES вкл).
            Для детекции сохраняет веса в work_dir/result_subdir.
            """
            log(f"  Финальное обучение [{label}] — {max_epochs} эп. + ES(patience={patience})...")
            try:
                if task == "classification":
                    m = _train_cls(ds_name, max_epochs, use_es=True,
                                   name_suffix="final")
                else:
                    # final=True: финальное обучение победителя.
                    # Вызывает _run_training_final (ES, eval на test).
                    m = _train_det(ds_name, max_epochs, use_es=True,
                                   result_subdir=result_subdir, keep_weights=True,
                                   final=True)
                return m
            except Exception as e:
                log(f"    [ОШИБКА full_train] {e}")
                log(traceback.format_exc())
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
            log(f"МОДАЛЬНОСТЬ ЗАДАНА ВРУЧНУЮ: {manual_modality.upper()}")
            log("Автоматический анализ изображений пропущен.")
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
                            log(f"    ✓ {_gid} ({_mname}) — разрешён")
                        else:
                            _excluded.append(_gid)
                            _rat = PR.get_rationale(manual_modality, _mname)
                            log(f"    ✗ {_gid} — запрещён: "
                                f"{_rat[:70]}{'...' if len(_rat) > 70 else ''}")
                else:
                    log(f"  Тип '{manual_modality}' — фильтрация не применяется (fallback)")
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

        if config.get("use_modality", False) and modality_result is None:
            log("")
            log("=" * 70)
            log("АНАЛИЗ МОДАЛЬНОСТИ ДАТАСЕТА")
            log("Gonzalez & Woods (2018); Oliver & Quegan (2004); Pisano et al. (1998); Kolarević et al. (2018)")
            log("Выборка: min(N_train×0.3, 300), мин. 30 — CLT (Kim 2017, PMC5370305)")
            log("=" * 70)
            try:
                import sys as _sys
                _sys.path.insert(0, str(Path(__file__).parent.parent))
                from Utils.image_analyzer      import UniversalImageAnalyzer
                from Utils.modality_classifier import ImageModalityClassifier
                from Utils.preprocessing_rules import PreprocessingRules as PR

                try:
                    _train_dir = datasets_path / dataset_name / "train"
                    if (_train_dir / "images").exists():
                        _n = len(list((_train_dir / "images").glob("*.*")))
                    else:
                        _n = sum(1 for _f in _train_dir.rglob("*")
                                 if _f.suffix.lower() in {".jpg",".jpeg",".png",".bmp"})
                    _sample = max(30, min(int(_n * 0.3), 300))
                except Exception:
                    _sample = 100
                log(f"  Датасет: {dataset_name} | выборка: {_sample} изображений")

                _analyzer      = UniversalImageAnalyzer(verbose=False)
                _ds_metrics, _ = _analyzer.analyze_dataset(
                    datasets_path / dataset_name, sample_size=_sample, split="train"
                )
                _color_str = "цветной" if _ds_metrics.is_color_dataset else "grayscale"
                log(f"  SNR: {_ds_metrics.avg_snr:.1f} dB | "
                    f"Контраст: {_ds_metrics.avg_contrast:.3f} | "
                    f"Яркость: {_ds_metrics.avg_brightness:.3f} | {_color_str}")

                _classifier = ImageModalityClassifier()
                _modal_info = _classifier.classify(_ds_metrics)
                _modality   = _modal_info["modality"]
                _confidence = _modal_info["confidence"]
                log(f"  Тип: {_modality.upper()} (уверенность {_confidence*100:.1f}%)")

                _excluded, _allowed = [], []
                if _modality in _MODALITY_FILTER_TYPES:
                    log(f"  Применяем правила для '{_modality}':")
                    for _gid, _mname in _GROUP_TO_METHOD.items():
                        _ok = PR.is_method_allowed(_modality, _mname)
                        if _ok:
                            _allowed.append(_gid)
                            log(f"    ✓ {_gid} ({_mname}) — разрешён")
                        else:
                            _excluded.append(_gid)
                            _rat = PR.get_rationale(_modality, _mname)
                            log(f"    ✗ {_gid} — запрещён: "
                                f"{_rat[:70]}{'...' if len(_rat) > 70 else ''}")
                else:
                    log(f"  Тип '{_modality}' — фильтрация не применяется (fallback)")
                    _allowed = list(_GROUP_TO_METHOD.keys())

                modality_result = {
                    "modality":        _modality,
                    "confidence":      _confidence,
                    "excluded_groups": _excluded,
                    "allowed_groups":  _allowed,
                    "is_color":        _ds_metrics.is_color_dataset,
                    "apply_filter":    _modality in _MODALITY_FILTER_TYPES,
                }
                q.put(("modality_result", modality_result))
                log("  Анализ модальности завершён.")

            except Exception as _e:
                log(f"  [ПРЕДУПРЕЖДЕНИЕ] Анализ модальности не удался: {_e}")
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
        auto_screen       = config.get("auto_screen", False)
        auto_screen_start = config.get("auto_screen_start", 40)
        # auto_screen_rho больше не задаётся пользователем —
        # критическое значение ρ вычисляется автоматически по числу
        # кандидатов N через _spearman_critical_rho(N, alpha=0.01).
        # Zar (2005): критическое значение зависит от N и α.
        # α=0.01 (p < 0.01) — строгий порог для научной работы.

        # Будет заполнен либо автоподбором, либо использует текущий fast_epochs
        _auto_screen_scores: Optional[Dict[str, float]] = None  # scores при найденном %

        if auto_screen:
            log("")
            log("=" * 70)
            log("АВТОПОДБОР ПРОЦЕНТА СКРИНИНГА")
            log(f"Начальный %: {auto_screen_start}%")
            log("Порог Спирмена ρ: автоматический (Zar, 2005; α=0.01)")
            log("Li et al. (2018) Hyperband — warm-start successive halving")
            log("=" * 70)

            # Получаем пул кандидатов (тот же что будет в Фазе 1)
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
                # Критическое значение ρ для данного N кандидатов.
                # Zar (2005): ρ_crit зависит от N; при N=16, α=0.01 → ρ_crit≈0.665.
                # Ramsey (1989): табулированные значения подтверждают расчёт.
                _n_cands = len(_as_all_cands)
                _rho_crit = _spearman_critical_rho(_n_cands, alpha=0.01)
                log(f"  Кандидатов: {_n_cands} → ρ_crit = {_rho_crit:.4f} "
                    f"(Zar 2005, α=0.01, N={_n_cands})")

                # Предварительно создаём датасеты кандидатов один раз
                _as_ds_map: Dict[str, str] = {}  # cand_id → ds_name
                log(f"  Создаю датасеты для {_n_cands} кандидатов...")
                for _cand in _as_all_cands:
                    try:
                        _ds = _make_tmp_ds(_cand["id"], _cand["methods"], _cand["params"])
                        _as_ds_map[_cand["id"]] = _ds
                    except Exception as _e:
                        log(f"  [ОШИБКА] {_cand['display']}: {_e}")

                _as_ratio_a = auto_screen_start
                _as_found   = False
                _as_scores_a: Dict[str, float] = {}
                # Словарь чекпоинтов после прогона A: {cand_id: ckpt_path}
                _as_ckpts_a: Dict[str, str] = {}

                while _as_ratio_a <= 100:
                    _as_ep_a = max(1, int(max_epochs * (_as_ratio_a / 100)))
                    _as_ratio_b = _as_ratio_a + 10
                    _as_ep_b = max(1, int(max_epochs * (_as_ratio_b / 100)))

                    log(f"\n  Прогон A: {_as_ratio_a}% ({_as_ep_a} эп.)")
                    # Обучаем прогон A только если scores ещё не вычислены.
                    # При цепочечном переходе (ρ < порога на предыдущей итерации)
                    # _as_scores_a и _as_ckpts_a уже заполнены из прогона B —
                    # в этом случае пропускаем обучение и идём сразу к прогону B.
                    # Li et al. (2018) JMLR 18(185): экономия N_cands обучений
                    # на каждой итерации кроме первой.
                    if not _as_scores_a:
                        for _cand in _as_all_cands:
                            _ds = _as_ds_map.get(_cand["id"])
                            if _ds is None:
                                continue
                            _sc, _ckpt = quick_train_n(_ds, _cand["display"], _as_ep_a)
                            _as_scores_a[_cand["id"]] = _sc
                            _as_ckpts_a[_cand["id"]] = _ckpt
                            log(f"    {_cand['display']:40s}  score={_sc:.4f}")
                    else:
                        # Scores уже есть из предыдущего прогона B — выводим их
                        for _cand in _as_all_cands:
                            _sc = _as_scores_a.get(_cand["id"], 0.0)
                            log(f"    {_cand['display']:40s}  score={_sc:.4f}  [перенесено из пред. прогона B]")

                    # ── Flat-scores guard ──────────────────────────────────
                    # Audibert et al. (2010) COLT: при ε-close arms бюджет
                    # идентификации → ∞. Если CV scores < 1.5%, ранжирование
                    # невозможно — предобработка не помогает этому датасету.
                    # Досрочно прекращаем автоподбор, экономя GPU-время.
                    if _check_flat_scores(_as_scores_a, log_fn=log):
                        log(f"\n  ДОСРОЧНАЯ ОСТАНОВКА: scores плоские (CV < 1.5%)")
                        log(f"     Ранжирование предобработок невозможно при любом % эпох.")
                        log(f"     Рекомендация: использовать оригинальный датасет.")
                        log(f"     Используем {_as_ratio_a}% как fallback.")
                        screening_ratio = _as_ratio_a
                        fast_epochs = _as_ep_a
                        _auto_screen_scores = _as_scores_a
                        _as_found = True
                        break

                    if _as_ratio_b > 100:
                        # Достигли потолка — используем 100%
                        log(f"  Достигнут потолок 100% — используем {_as_ratio_a}%")
                        screening_ratio = _as_ratio_a
                        fast_epochs = _as_ep_a
                        _auto_screen_scores = _as_scores_a
                        _as_found = True
                        break

                    log(f"\n  Прогон B: {_as_ratio_b}% ({_as_ep_b} эп.) "
                        f"[warm-start с {_as_ratio_a}%]")
                    _as_scores_b: Dict[str, float] = {}
                    # Чекпоинты прогона B — нужны если ρ < порога,
                    # чтобы следующая итерация могла дообучать с них.
                    # Li et al. (2018) JMLR 18(185): цепочечное дообучение
                    # корректно для successive halving — ранжирование сохраняется.
                    _as_ckpts_b: Dict[str, str] = {}
                    for _cand in _as_all_cands:
                        _ds = _as_ds_map.get(_cand["id"])
                        if _ds is None:
                            continue
                        # Warm-start: дообучаем с чекпоинта прогона A
                        _resume = _as_ckpts_a.get(_cand["id"], "")
                        # score_floor — лучший score прогона A для этого кандидата
                        # гарантирует что resume не даст результат хуже прогона A
                        _floor = _as_scores_a.get(_cand["id"], 0.0)
                        # Передаём дельту эпох (ep_b - ep_a), а не абсолютное
                        # число. Ultralytics при warm-start сбрасывает счётчик
                        # эпох с нуля — train_results[-1] после прогона B
                        # хранит число эпох текущего запуска, а не суммарное.
                        # Передавая дельту явно, мы обходим эту проблему:
                        # функция обучает ровно _as_ep_b - _as_ep_a эпох.
                        # Li et al. (2018) JMLR 18(185): warm-start SHA
                        # использует бюджет дополнительных эпох.
                        _as_ep_delta = _as_ep_b - _as_ep_a
                        _sc, _ckpt_b = quick_train_n(_ds, _cand["display"], _as_ep_delta,
                                                     resume_from=_resume,
                                                     score_floor=_floor)
                        _as_scores_b[_cand["id"]] = _sc
                        _as_ckpts_b[_cand["id"]] = _ckpt_b
                        log(f"    {_cand['display']:40s}  score={_sc:.4f}"
                            f"{'  [resume]' if _resume else ''}")

                    _rho = _spearman_rho(_as_scores_a, _as_scores_b)
                    log(f"\n  Спирмен ρ({_as_ratio_a}% vs {_as_ratio_b}%) = {_rho:.4f}"
                        f"  (ρ_crit={_rho_crit:.4f}, α=0.01, N={_n_cands})")

                    if _rho >= _rho_crit:
                        log(f"  ρ ≥ ρ_crit — корреляция статистически значима. "
                            f"Используем {_as_ratio_a}% скрининга.")
                        screening_ratio = _as_ratio_a
                        fast_epochs = _as_ep_a
                        _auto_screen_scores = _as_scores_a
                        _as_found = True
                        break
                    else:
                        log(f"  ρ={_rho:.4f} < ρ_crit={_rho_crit:.4f} — "
                            f"увеличиваем до {_as_ratio_b}%")
                        # Следующая итерация: прогон B становится новым прогоном A.
                        # Scores и чекпоинты прогона B переносятся напрямую —
                        # нет необходимости обучать A заново с нуля.
                        #
                        # Научное обоснование цепочечного дообучения:
                        # Li et al. (2018) "Hyperband", JMLR 18(185), 1-52:
                        # каждый следующий бюджет строится поверх предыдущего;
                        # все кандидаты проходят одинаковую траекторию обучения,
                        # поэтому ранжирование остаётся корректным.
                        #
                        # Цепочка: 30%→40%→50%→... вместо повторного обучения с нуля
                        # на каждом новом A экономит (N_iter - 1) × N_cands прогонов.
                        _as_scores_a = _as_scores_b          # scores прогона B → новый A
                        _as_ckpts_a  = _as_ckpts_b            # ckpts прогона B → новый A
                        _as_ratio_a  = _as_ratio_b            # сдвигаем %

                # Удаляем временные датасеты созданные для автоподбора
                for _ds in _as_ds_map.values():
                    _cleanup_ds(_ds)

                if _as_found:
                    log(f"\n  Итог автоподбора: screening_ratio={screening_ratio}%"
                        f"  ({fast_epochs} эп.)")
                    log(f"  Li et al. (2018) Hyperband — "
                        f"однораундовый отсев (s=0 bracket) с подобранным бюджетом")

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

        for group_id, group_info in active_groups.items():
            group_label = group_info["label"]
            candidates  = group_info["candidates"]
            log(f"\n  Группа [{group_label}] — {len(candidates)} кандидатов")

            scored: List[Dict] = []
            for cand in candidates:
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
        log("ФИНАЛЬНОЕ ОБУЧЕНИЕ ПОБЕДИТЕЛЯ")
        log("=" * 70)

        # Выбираем лучшего среди финальных survivors
        best_survivor = max(final_survivors, key=lambda x: x["score"])
        best_pipeline_cands = best_survivor["pipeline_cands"]

        # Если победитель — одиночный метод из Фазы 1 (len==1) и его score
        # не превышает baseline, честно сообщаем пользователю, но всё равно
        # обучаем и сравниваем полным обучением.
        log(f"  Лучший survivor: {best_survivor['display']}  score({screening_ratio}%)={best_survivor['score']:.4f}")
        log(f"  Baseline score ({screening_ratio}%): {baseline_score:.4f}")

        # pipeline_cands — список одиночных survivor-dict из Фазы 1
        winner_methods, winner_params = merge_methods_params(best_pipeline_cands)
        winner_display = best_survivor["display"]
        log(f"  Победитель для полного обучения: {winner_display}")

        # Создаём постоянный датасет-победитель (не временный)
        winner_ds_name = f"{dataset_name}_p2_winner"
        log(f"  Создаю постоянный датасет: {winner_ds_name}")
        preprocessor.apply_global_preprocessing(
            source_dataset=dataset_name,
            target_dataset=winner_ds_name,
            methods=winner_methods,
            params=winner_params,
        )
        final_metrics = full_train(winner_ds_name, winner_display,
                                   result_subdir="final_winner")

        # ══════════════════════════════════════════════════════════════════
        # ФИНАЛЬНЫЙ BASELINE (полное обучение для честного сравнения)
        # ══════════════════════════════════════════════════════════════════
        log("")
        log("=" * 70)
        log("ФИНАЛЬНЫЙ BASELINE (полное обучение для сравнения)")
        log("=" * 70)

        baseline_final_metrics = full_train(dataset_name, "baseline",
                                            result_subdir="final_baseline")

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
            "baseline_quick_score": round(baseline_score, 4),
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

    st.title("Объединённый подбор пайплайна предобработки")
    st.markdown(
        "**Двухфазный алгоритм** с опциональным анализом модальности датасета. "
        "Перед Фазой 1 система определяет тип изображений и исключает методы "
        "запрещённые для данной модальности."
    )

    with st.expander("Как работает алгоритм", expanded=False):
        st.markdown("""
**Фаза 1 — Групповой скрининг** *(Guyon & Elisseeff, 2003; Kohavi & John, 1997; Jamieson & Talwalkar, 2016)*

Методы предобработки разбиты на тематические группы (шум, контраст, яркость, резкость).
Внутри каждой группы все кандидаты обучаются **30% эпох** без Early Stopping.
Затем: фильтр по baseline (отсекает кандидатов хуже оригинала) → SHA-отсев среди прошедших фильтр (`ceil(N/eta)` лучших).
Если вся группа хуже baseline — группа пропускается целиком.

**Фаза 2 — SFS+SHA на survivors** *(Kohavi & John, 1997; Jamieson & Talwalkar, 2016)*

Итеративно строится пайплайн из survivors: на каждом шаге добавляется метод,
который даёт наибольший прирост. SHA отсекает слабые расширения.

**Baseline** обучается до 30% эпох для ранжирования и до 100% для финального сравнения.

**Сохранение датасета** — только если победитель лучше baseline.
        """)
        st.markdown("""
| Шаг | Что происходит | Эпох |
|-----|----------------|------|
| Baseline (быстрый) | Оценка оригинала | 30% |
| Фаза 1 | Baseline-фильтр → SHA по каждой группе | 30% × N кандидатов |
| Фаза 2 | SFS+SHA на survivors | 30% × итерации |
| Финал победителя | Полное обучение | 100% + ES |
| Финал baseline | Полное обучение | 100% + ES |
        """)

    st.divider()

    datasets = get_available_datasets()
    if not datasets:
        st.warning("Датасеты не найдены. Проверь путь в Настройках.")
        st.stop()

    # ── Анализ модальности ─────────────────────────────────────────────────
    st.subheader("0. Анализ модальности (опционально)")
    st.markdown(
        "Если включено — перед Фазой 1 система анализирует метрики изображений "
        "и определяет тип датасета (medical, SAR, microscopy и др.). "
        "Группы методов запрещённые для данной модальности будут исключены из Фазы 1. "
        "Для natural_photo и infrared фильтрация не применяется."
    )

    use_modality = st.checkbox(
        "Использовать анализ модальности",
        value=st.session_state.get("p2_use_modality", True),
        key="p2_use_modality_cb",
        help=(
            "Oliver & Quegan (2004) SAR; Pisano et al. (1998, 2000) рентген; "
            "Kolarević et al. (2018) микроскопия. "
            "Выборка: min(N_train×0.3, 300), мин. 30 изображений "
            "(CLT: Kim 2017, Korean J Anesthesiol, PMC5370305)."
        ),
    )
    st.session_state.p2_use_modality = use_modality

    if use_modality:
        manual_modality = st.selectbox(
            "Модальность датасета",
            options=["auto", "medical_xray", "microscopy", "sar", "natural_photo", "infrared"],
            index=["auto", "medical_xray", "microscopy", "sar", "natural_photo", "infrared"].index(
                st.session_state.get("p2_manual_modality", "auto")
            ),
            format_func=lambda x: {
                "auto":          "Авто (определить автоматически)",
                "medical_xray":  "Medical X-ray (рентген, маммография)",
                "microscopy":    "Microscopy (микроскопия, гистопатология)",
                "sar":           "SAR (радарные снимки)",
                "natural_photo": "Natural Photo (натуральные фото)",
                "infrared":      "Infrared (тепловизор, ИК-снимки)",
            }[x],
            key="p2_manual_modality_select",
            help=(
                "Авто — автоматическое определение по метрикам изображений. "
                "Если автоопределение ошибается (например микроскопия определяется "
                "как натуральное фото) — укажите модальность вручную. "
                "Это применит соответствующие правила фильтрации групп методов."
            ),
        )
        st.session_state.p2_manual_modality = manual_modality

        if manual_modality != "auto":
            st.info(
                f"Модальность задана вручную: **{manual_modality.upper()}**. "
                "Автоматический анализ изображений будет пропущен."
            )
    else:
        manual_modality = "auto"
        st.session_state.p2_manual_modality = "auto"

    use_wiener = st.checkbox(
        "Включить Wiener-фильтры в пул кандидатов",
        value=st.session_state.get("p2_use_wiener", False),
        key="p2_use_wiener_cb",
        help=(
            "Добавляет Wiener (size=3) и Wiener (size=5) в группу шумоподавления. "
            "Wiener — адаптивный линейный фильтр, минимизирует MSE относительно "
            "оригинала (Wiener, 1949; Fan et al., 2019). "
            "⚠ Отключён по умолчанию: scipy.signal.wiener работает без SIMD-оптимизации "
            "и занимает ~1–1.5 сек/изображение против ~3–8 мс для OpenCV-фильтров. "
            "Рекомендуется включать только на датасетах до ~5k изображений."
        ),
    )
    st.session_state.p2_use_wiener = use_wiener

    if use_wiener:
        st.warning(
            "⚠ Wiener включён. На датасете >5k изображений предобработка может занять "
            "несколько часов. Убедитесь что это оправдано размером датасета."
        )

    use_sha_fallback = st.checkbox(
        "SHA-fallback: если Фаза 1 пуста — отобрать лучших без baseline-фильтра",
        value=st.session_state.get("p2_sha_fallback", False),
        key="p2_sha_fallback_cb",
        help=(
            "Применяется только если ВСЕ группы Фазы 1 оказались хуже baseline. "
            "Если хотя бы одна группа дала survivors — этот режим не используется. "
            "При включении: SHA-отсев (ceil(N/eta) лучших) применяется к каждой группе "
            "без отсечки по baseline, давая Фазе 2 хоть каких-то кандидатов. "
            "При выключении: алгоритм завершается с сообщением что baseline лучше всех методов. "
            "Jamieson & Talwalkar (2016) — SHA ранжирует кандидатов без порогового отсева."
        ),
    )
    st.session_state.p2_sha_fallback = use_sha_fallback

    use_torch_compile = st.checkbox(
        "Включить torch.compile (JIT-компиляция модели)",
        value=st.session_state.get("p2_use_torch_compile", True),
        key="p2_use_torch_compile_cb",
        help=(
            "torch.compile компилирует граф модели через Triton/CUDA, давая +20–40% "
            "скорости обучения. Требует triton-windows. "
            "⚠ Отключён по умолчанию: первая эпоха каждого обучения занимает "
            "~1–2 мин на компиляцию. При SHA-скрининге (~27 обучений) накладные "
            "расходы превышают выигрыш. Включать только при финальном обучении "
            "одной модели на большом датасете (>10k изображений, >50 эпох)."
        ),
    )
    st.session_state.p2_use_torch_compile = use_torch_compile

    if use_torch_compile:
        st.warning(
            "⚠ torch.compile включён. Первая эпоха каждого обучения будет медленнее "
            "(~1–2 мин компиляция). На малых датасетах или коротких прогонах "
            "это может замедлить подбор."
        )

    # Показываем результат предыдущего анализа если есть
    prev_modal = st.session_state.get("p2_modality_result")
    if use_modality and prev_modal:
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

        col_rerun, col_reset = st.columns([2, 1])
        with col_rerun:
            if st.button("🔄 Перезапустить анализ модальности", key="rerun_modal"):
                st.session_state.p2_modality_result = None
                st.rerun()
        with col_reset:
            if st.button("✕ Сбросить", key="reset_modal"):
                st.session_state.p2_modality_result = None
                st.session_state.p2_use_modality = False
                st.rerun()

    st.divider()

    # ── Датасет и задача ───────────────────────────────────────────────────
    col_ds, col_task = st.columns(2, gap="large")

    with col_ds:
        st.subheader("1. Датасет")
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
        st.subheader("2. Тип задачи")
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
    st.subheader("3. Proxy-модель")
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
                help="28 — MedMNIST; 224 — ImageNet-стандарт; 240 — нативный EfficientNet-B0",
                key="p2_imgsz_select",
            )
            pretrained = st.checkbox(
                "Pretrained (ImageNet)",
                value=st.session_state.p2_pretrained,
                key="p2_pretrained_cb",
                help="He et al. (2016) — предобученные веса ускоряют сходимость",
            )
            freeze_backbone = st.checkbox(
                "Заморозить backbone (freeze)",
                value=st.session_state.p2_freeze_backbone,
                key="p2_freeze_cb",
                help=(
                    "Замораживает все слои кроме головы классификатора. "
                    "Рекомендуется при малом датасете — предотвращает переобучение backbone. "
                    "Yosinski et al. (2014) NeurIPS; Pan & Yang (2010) IEEE TKDE."
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
    st.subheader("4. Параметры обучения")
    col_p1, col_p2 = st.columns(2, gap="large")

    with col_p1:
        default_epochs = 50 if task == "classification" else 30
        epochs = st.number_input(
            "Эпох финального обучения",
            min_value=5, max_value=300,
            value=st.session_state.p2_epochs,
            step=5,
            key="p2_epochs_input",
            help="Применяется к финальному обучению победителя и baseline. "
                 "SHA-скрининг использует 30% от этого числа.",
        )
        patience = st.number_input(
            "Early Stopping patience (финал)",
            min_value=3, max_value=100,
            value=st.session_state.p2_patience,
            key="p2_patience_input",
            help="Prechelt (1998) — остановка если метрика не улучшается N эпох",
        )

    with col_p2:
        eta = st.selectbox(
            "eta — коэффициент SHA-отсева",
            options=[2, 3, 4],
            index=0,
            key="p2_eta_select",
            help="eta=2: оставляем ½ кандидатов; eta=3: ⅓. Jamieson & Talwalkar (2016).",
        )
        seed = st.number_input(
            "Seed воспроизводимости",
            min_value=0, max_value=2**31 - 1,
            value=st.session_state.p2_seed,
            key="p2_seed_input",
            help="Dodge & Karam (2017) — фиксация seed устраняет случайный разброс метрик",
        )
        screening_ratio = st.slider(
            "% эпох для скрининга",
            min_value=10, max_value=60,
            value=st.session_state.get("p2_screening_ratio", 30),
            step=5,
            key="p2_screening_ratio_slider",
            help=(
                "Доля от полных эпох для быстрой оценки кандидатов (SHA-скрининг). "
                "30% — значение по умолчанию (Jamieson & Talwalkar, 2016). "
                "При нестабильных кривых обучения или малом числе эпох рекомендуется "
                "40–50% для снижения rank instability (Li et al., 2018 Hyperband)."
            ),
        ) if not st.session_state.get("p2_auto_screen", False) else st.session_state.get("p2_screening_ratio", 30)

    # ── Автоподбор процента скрининга ─────────────────────────────────────
    st.divider()
    auto_screen = st.checkbox(
        "Автоподбор процента скрининга",
        value=st.session_state.get("p2_auto_screen", False),
        key="p2_auto_screen_cb",
        help=(
            "Автоматически подбирает минимальный процент скрининга при котором "
            "ранжирование кандидатов стабильно. Запускает Фазу 1 дважды "
            "(на x% и x+10%) и проверяет корреляцию Спирмена. "
            "Egele et al. (2024) Neurocomputing — early discarding stability. "
            "Увеличивает время подбора: каждая пара прогонов = 2x кандидатов."
        ),
    )
    st.session_state.p2_auto_screen = auto_screen

    if auto_screen:
        _as_col1, _as_col2 = st.columns(2)
        with _as_col1:
            auto_screen_start = st.slider(
                "Начальный % для проверки",
                min_value=20, max_value=60,
                value=st.session_state.get("p2_auto_screen_start", 30),
                step=10,
                key="p2_auto_screen_start_slider",
                help="С какого процента начинать поиск. 30% рекомендуется как минимум.",
            )
            st.session_state.p2_auto_screen_start = auto_screen_start
        with _as_col2:
            st.info(
                "**Порог ρ: автоматический**\n\n"
                "Критическое значение Спирмена ρ рассчитывается по числу "
                "кандидатов N и уровню значимости α=0.01 "
                "(Zar, 2005; Ramsey, 1989).\n\n"
                "Также: если scores всех кандидатов практически одинаковы "
                "(CV < 1.5%), подбор прекращается досрочно — предобработка "
                "не даёт значимого эффекта (Audibert et al., 2010 COLT)."
            )

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

    st.info(
        f"**Оценка числа обучений при eta={eta}:** "
        f"Baseline быстрый: 1 x {fast_ep} эп. | "
        f"Фаза 1: {total_candidates} x {fast_ep} эп. | "
        f"Survivors Ф.1: ~{survivors_est} из {total_candidates} | "
        f"Фаза 2: ~{phase2_est} x {fast_ep} эп. | "
        f"Финал: 2 x до {epochs} эп."
    )

    st.divider()

    btn_label = (
        "▶ Запустить подбор предобработки (с анализом модальности)"
        if use_modality
        else "▶ Запустить подбор предобработки"
    )
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
            # auto_screen_rho убран — критическое значение ρ вычисляется
            # автоматически внутри _run_search через _spearman_critical_rho(N).
            # Zar (2005); Ramsey (1989).
            # Анализ модальности
            "use_modality":    st.session_state.get("p2_use_modality", True),
            "manual_modality": st.session_state.get("p2_manual_modality", "auto"),
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
            st.title("✓ Предобработка улучшила метрики!")
            st.success(
                f"**Лучшая комбинация методов: {result['winner_pipeline']}** — "
                f"score улучшился на **{result['improvement']:+.4f}** относительно baseline.\n\n"
                f"Предобработанный датасет `{result['winner_ds_name']}` сохранён и готов к обучению."
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
                        "Метрика":    "Score  (0.6×AUC + 0.4×ACC)",
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
                        "Метрика":    "Score  (0.45×mAP50-95 + 0.35×mAP50 + 0.2×F1)",
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
                        label="📥 Таблица 1 (финальные survivors)",
                        data=csv1,
                        file_name=f"p2_final_survivors_{result.get('dataset_name','')}.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )

            # CSV таблица 2 — победитель vs baseline (полное обучение)
            csv2 = pd.DataFrame(table2_rows).to_csv(index=False, encoding="utf-8")
            with dl_col2:
                st.download_button(
                    label="📥 Таблица 2 (победитель vs baseline)",
                    data=csv2,
                    file_name=f"p2_winner_vs_baseline_{result.get('dataset_name','')}.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

            # Полный лог
            full_log = "\n".join(st.session_state.p2_log_lines)
            with dl_col3:
                st.download_button(
                    label="📥 Полный лог",
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
