"""
pages/2_Предобработка.py — Двухфазный автоматический подбор пайплайна предобработки

Алгоритм: Group-wise SHA (Фаза 1) + SFS+SHA на survivors (Фаза 2)

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
    page_title="Подбор предобработки — VKR2026",
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
                "id": "nlm_h10",
                "display": "NLM (h=10)",
                "methods": ["denoise"],
                "params": {"denoise": {"method": "nlm", "h": 10,
                                       "template_window_size": 7,
                                       "search_window_size": 21}},
            },
        ],
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
                "id": "bright_04",
                "display": "Яркость → 0.4",
                "methods": ["brightness_correction"],
                "params": {"brightness_correction": {"target_brightness": 0.4}},
            },
            {
                "id": "bright_05",
                "display": "Яркость → 0.5",
                "methods": ["brightness_correction"],
                "params": {"brightness_correction": {"target_brightness": 0.5}},
            },
            {
                "id": "bright_06",
                "display": "Яркость → 0.6",
                "methods": ["brightness_correction"],
                "params": {"brightness_correction": {"target_brightness": 0.6}},
            },
        ],
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


def merge_methods_params(candidates: List[Dict]) -> Tuple[List[str], Dict]:
    """
    Объединяет методы и параметры нескольких кандидатов в один пайплайн.
    Используется при комбинировании survivors из разных групп.
    """
    methods = []
    params = {}
    for c in candidates:
        for m in c["methods"]:
            if m not in methods:
                methods.append(m)
        params.update(c["params"])
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
        0.40 * metrics.get("mAP50-95", 0.0)
        + 0.30 * metrics.get("mAP50", 0.0)
        + 0.20 * metrics.get("f1", 0.0)
        + 0.10 * (1.0 / (1.0 + metrics.get("val_loss", 1.0)))
    )


# ══════════════════════════════════════════════════════════════════════════════
# ЯДРО АЛГОРИТМА (запускается в фоновом потоке)
# ══════════════════════════════════════════════════════════════════════════════

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
            _model_cfg_base = {
                "type":       model_type,
                "image_size": imgsz,
                "pretrained": pretrained,
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

        fast_epochs   = max(1, int(max_epochs * 0.30))
        datasets_path = Path(config["datasets_path"])

        # Рабочая папка запуска — хранит временные датасеты и финальные веса
        ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
        work_dir = datasets_path / "p2_runs" / f"{dataset_name}_{ts}"
        work_dir.mkdir(parents=True, exist_ok=True)

        set_global_seed(seed)

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
        log(f"Финал: {max_epochs} эп. | Быстрый: {fast_epochs} эп. (30%) | ES patience: {patience}")
        log(f"eta (SHA): {eta} | Seed: {seed}")
        log(f"Рабочая папка: {work_dir}")
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
                       name_suffix: str) -> Dict:
            """
            Обучает ClassificationTrainer. Возвращает лучшие метрики.
            Очистка памяти гарантирована в finally ClassificationTrainer._train_one.
            """
            _cfg = {
                **_model_cfg_base,
                "name":       f"{model_type}_{name_suffix}",
                "max_epochs": epochs,
            }
            if use_es:
                _cfg["early_stopping"] = {"patience": patience, "metric": "val_auc"}
            trainer = ClassificationTrainer(
                model_configs=[_cfg],
                dataset_names=[ds_name],
                max_epochs=epochs,
                checkpoint_interval=epochs if not use_es else max(1, epochs // 10),
                seed=seed,
                enable_early_stopping=use_es,
                early_stopping_patience=patience,
                early_stopping_metric="val_auc",
                enable_early_selection=False,
            )
            trainer.run_training()
            key     = f"{model_type}_{name_suffix}_{ds_name}"
            history = trainer.metrics_history.get(key, [])
            if not history:
                return {}
            best = max(history, key=lambda x: x.get("val_auc", x.get("val_acc", 0.0)))
            return best

        # ── Обучение для детекции ──────────────────────────────────────────

        def _train_det(ds_name: str, epochs: int, use_es: bool,
                       result_subdir: str, keep_weights: bool) -> Dict:
            """
            Обучает детекционную модель через _run_training / _run_training_final
            из module3_preprocessing_search.
            keep_weights=True — не удаляет result_dir (финальное обучение).
            keep_weights=False — удаляет result_dir (SHA-скрининг).
            Очистка GPU-памяти внутри вызываемых функций.
            """
            from module3_preprocessing_search import (
                _run_training,
                _run_training_final,
            )
            ds_path    = get_dataset_path(ds_name)
            result_dir = work_dir / result_subdir

            if keep_weights:
                # Финальное обучение — ES всегда вкл, eval на test, веса сохраняются
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
                # SHA-скрининг — ES выкл для честного сравнения, eval на valid
                metrics = _run_training(
                    dataset_path=ds_path,
                    model_config=_model_cfg_base,
                    epochs=epochs,
                    result_dir=result_dir,
                    log_fn=log,
                    use_early_stopping=False,
                    early_stopping_patience=patience,
                    eval_split="valid",
                )
            return metrics

        # ── Универсальные quick_train / full_train ─────────────────────────

        def quick_train(ds_name: str, label: str) -> float:
            """
            Быстрое обучение (30% эпох, ES выкл).
            Возвращает scalar score для ранжирования SHA.
            Jamieson & Talwalkar (2016) — 30% эпох достаточно для ранжирования.
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
                    m = _train_det(ds_name, max_epochs, use_es=True,
                                   result_subdir=result_subdir, keep_weights=True)
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

        # ══════════════════════════════════════════════════════════════════
        # BASELINE: быстрое обучение оригинала
        # ══════════════════════════════════════════════════════════════════
        log("")
        log("=" * 70)
        log("BASELINE: оригинальный датасет (30% эпох)")
        log("=" * 70)
        baseline_score = quick_train(dataset_name, "baseline")
        log(f"  Baseline score = {baseline_score:.4f}")

        # ══════════════════════════════════════════════════════════════════
        # ФАЗА 1: Group-wise SHA-скрининг
        # Guyon & Elisseeff (2003); Liu & Motoda (2007).
        # Для каждой группы: обучаем всех кандидатов (30% эпох),
        # SHA-отсев оставляет ceil(N_group / eta) survivors.
        # ══════════════════════════════════════════════════════════════════
        log("")
        log("=" * 70)
        log("ФАЗА 1: Групповой SHA-скрининг")
        log("Jamieson & Talwalkar (2016) SHA + Guyon & Elisseeff (2003) группировка")
        log("=" * 70)

        all_survivors: List[Dict] = []

        for group_id, group_info in CANDIDATE_GROUPS.items():
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

            # SHA-отсев внутри группы
            survivors = sha_prune(scored, eta=eta)
            log(f"\n    SHA: {len(scored)} -> {len(survivors)} survivors")
            for s in survivors:
                log(f"      + {s['display']:40s}  score={s['score']:.4f}")

            all_survivors.extend(survivors)

        if not all_survivors:
            raise RuntimeError(
                "Фаза 1 не дала survivors — проверьте датасет и модель."
            )

        log(f"\n  Итого survivors после Фазы 1: {len(all_survivors)}")
        for s in sorted(all_survivors, key=lambda x: x["score"], reverse=True):
            log(f"    {s['display']:45s}  score={s['score']:.4f}")

        # ══════════════════════════════════════════════════════════════════
        # ФАЗА 2: SFS+SHA на survivors
        # Kohavi & John (1997) SFS + Jamieson & Talwalkar (2016) SHA.
        # Итеративно строим пайплайн: добавляем метод только если он
        # улучшает текущий score. SHA режет слабые расширения.
        # ══════════════════════════════════════════════════════════════════
        log("")
        log("=" * 70)
        log("ФАЗА 2: SFS+SHA на survivors")
        log("Kohavi & John (1997) SFS + Jamieson & Talwalkar (2016) SHA")
        log("=" * 70)

        current_pipeline: Optional[List[Dict]] = None  # survivors в пайплайне
        current_score   = baseline_score
        remaining       = list(all_survivors)
        sfs_history     = []
        iteration       = 0

        while remaining:
            iteration += 1
            log(f"\n  SFS итерация {iteration}: {len(remaining)} кандидатов в пуле")

            # Расширяем — добавляем каждый из remaining к текущему пайплайну
            candidates_iter = []
            for cand in remaining:
                pipeline_cands = (current_pipeline or []) + [cand]
                combined_methods, combined_params = merge_methods_params(pipeline_cands)
                pipeline_id = "+".join(c["id"] for c in pipeline_cands)
                display     = " + ".join(c["display"] for c in pipeline_cands)
                candidates_iter.append({
                    "id":               pipeline_id,
                    "display":          display,
                    "pipeline_cands":   pipeline_cands,
                    "methods":          combined_methods,
                    "params":           combined_params,
                    "score":            0.0,
                })

            # Оцениваем быстрым обучением
            scored_iter = []
            for ci in candidates_iter:
                log(f"    Пайплайн: {ci['display']}")
                try:
                    ds_tmp = _make_tmp_ds(ci["id"], ci["methods"], ci["params"])
                except Exception as e:
                    log(f"    [ОШИБКА датасета] {e}")
                    continue
                sc = quick_train(ds_tmp, ci["display"])
                _cleanup_ds(ds_tmp)
                scored_iter.append({**ci, "score": sc})

            if not scored_iter:
                log("  [СТОП] Нет кандидатов в итерации")
                break

            # SHA-отсев и выбор лучшего
            survivors_iter = sha_prune(scored_iter, eta=eta)
            best_iter      = survivors_iter[0]

            sfs_history.append({
                "iteration":      iteration,
                "best_pipeline":  best_iter["display"],
                "score":          best_iter["score"],
                "n_candidates":   len(scored_iter),
                "n_survivors":    len(survivors_iter),
            })

            log(f"    Лучший: {best_iter['display']}  score={best_iter['score']:.4f}")

            if best_iter["score"] > current_score:
                current_score    = best_iter["score"]
                current_pipeline = best_iter["pipeline_cands"]
                last_added_id    = current_pipeline[-1]["id"]
                remaining        = [r for r in remaining if r["id"] != last_added_id]
                log(f"    + Улучшение принято. Score={current_score:.4f}")
            else:
                log(f"    - Улучшения нет ({best_iter['score']:.4f} <= {current_score:.4f}). SFS стоп.")
                break

        # ══════════════════════════════════════════════════════════════════
        # ФИНАЛЬНОЕ ОБУЧЕНИЕ ПОБЕДИТЕЛЯ
        # ══════════════════════════════════════════════════════════════════
        log("")
        log("=" * 70)
        log("ФИНАЛЬНОЕ ОБУЧЕНИЕ ПОБЕДИТЕЛЯ")
        log("=" * 70)

        if current_pipeline is None:
            # Предобработка не улучшила baseline — обучаем оригинал
            log("  Пайплайн не улучшил baseline. Финальное обучение на оригинале.")
            winner_display  = "Оригинал (baseline)"
            winner_methods  = []
            winner_params   = {}
            winner_ds_name  = dataset_name
            final_metrics   = full_train(winner_ds_name, winner_display,
                                         result_subdir="final_baseline")
        else:
            winner_methods, winner_params = merge_methods_params(current_pipeline)
            winner_display = " + ".join(c["display"] for c in current_pipeline)
            log(f"  Победитель: {winner_display}")

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

        if current_pipeline is None:
            # winner = baseline, не дублируем обучение
            baseline_final_metrics = final_metrics
            log("  Winner = baseline, используем уже обученные метрики.")
        else:
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
                f"  mAP50={baseline_final_metrics.get('mAP50',0):.4f}")
            log(f"  Победитель — score={winner_score_full:.4f}"
                f"  mAP50-95={final_metrics.get('mAP50-95',0):.4f}"
                f"  mAP50={final_metrics.get('mAP50',0):.4f}")

        log(f"  Изменение: {improvement:+.4f}")

        if better:
            log(f"  + Предобработка улучшила метрики.")
            log(f"  + Датасет '{winner_ds_name}' сохранён.")
        else:
            log(f"  - Предобработка не улучшила метрики.")
            log(f"  - Датасет-победитель удалён (нет смысла применять).")
            if winner_ds_name != dataset_name:
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


def _suggest_batch_det(model_type: str, vram: float) -> int:
    """Подсказывает batch_size для детекции."""
    if vram <= 0:
        return 1
    base = {"yolo": 16, "faster_rcnn": 4, "retinanet": 8}.get(model_type, 4)
    return max(1, int(base * vram / 8.0))


# ══════════════════════════════════════════════════════════════════════════════
# UI — ЭТАП 1: КОНФИГУРАЦИЯ
# ══════════════════════════════════════════════════════════════════════════════

if st.session_state.p2_stage == "configure":

    vram_gb = _get_vram_gb()

    st.title("Подбор пайплайна предобработки")
    st.markdown(
        "**Двухфазный алгоритм** автоматически подбирает оптимальную комбинацию "
        "методов предобработки и сравнивает результат с baseline."
    )

    with st.expander("Как работает алгоритм", expanded=False):
        st.markdown("""
**Фаза 1 — Групповой SHA-скрининг** *(Guyon & Elisseeff, 2003; Liu & Motoda, 2007)*

Методы предобработки разбиты на тематические группы (шум, контраст, яркость, резкость).
Внутри каждой группы все кандидаты обучаются **30% эпох** без Early Stopping,
затем SHA-отсев оставляет `ceil(N/eta)` лучших survivors.

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
| Фаза 1 | SHA по каждой группе | 30% × N кандидатов |
| Фаза 2 | SFS+SHA на survivors | 30% × итерации |
| Финал победителя | Полное обучение | 100% + ES |
| Финал baseline | Полное обучение | 100% + ES |
        """)

    st.divider()

    datasets = get_available_datasets()
    if not datasets:
        st.warning("Датасеты не найдены. Проверь путь в Настройках.")
        st.stop()

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
            auto_det_batch = _suggest_batch_det(det_model, vram_gb)
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

    st.session_state.p2_epochs = epochs
    st.session_state.p2_patience = patience
    st.session_state.p2_eta = eta
    st.session_state.p2_seed = seed

    # ── Оценка числа обучений ─────────────────────────────────────────────
    fast_ep = max(1, int(epochs * 0.30))
    total_candidates = sum(len(g["candidates"]) for g in CANDIDATE_GROUPS.values())
    survivors_est = sum(
        math.ceil(len(g["candidates"]) / eta) for g in CANDIDATE_GROUPS.values()
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

    if st.button(
        "Запустить подбор предобработки",
        type="primary",
        use_container_width=True,
    ):
        st.session_state.p2_stage = "running"
        st.session_state.p2_log_lines = []
        st.session_state.p2_thread_done = False
        st.session_state.p2_error = None
        st.session_state.p2_result = None
        st.session_state.p2_output_queue = None
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

        if result and result.get("better_than_baseline"):
            st.title("✓ Предобработка улучшила метрики")
            st.success(
                f"Пайплайн **{result['winner_pipeline']}** "
                f"лучше baseline на **{result['improvement']:+.4f}** по score."
            )
        else:
            st.title("Подбор завершён")
            if result:
                st.warning(
                    f"Предобработка не улучшила метрики относительно baseline "
                    f"(разница: {result.get('improvement', 0):+.4f}). "
                    f"Рекомендуется использовать оригинальный датасет."
                )

        if result:
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Победитель")
                st.code(result.get("winner_pipeline", "Оригинал"), language=None)
                if result.get("winner_methods"):
                    st.caption(f"Методы: {', '.join(result['winner_methods'])}")
                if result.get("winner_ds_name"):
                    st.caption(f"Датасет: `{result['winner_ds_name']}`")
                else:
                    st.caption("Датасет победителя не сохранён (не лучше baseline).")

            with col2:
                st.subheader("Сравнение метрик (полное обучение)")
                bm = result.get("baseline_metrics", {})
                wm = result.get("winner_metrics", {})
                _task = result.get("task", st.session_state.p2_task)

                if _task == "classification":
                    metrics_to_show = [
                        ("AUC",   "val_auc"),
                        ("ACC",   "val_acc"),
                        ("Score", None),
                    ]
                else:
                    # Детекция — метрики упорядочены по весу в score_from_metrics:
                    # score = 0.4*mAP50-95 + 0.3*mAP50 + 0.2*f1 + 0.1*(1/(1+val_loss))
                    metrics_to_show = [
                        ("Score (0.4×mAP50-95 + 0.3×mAP50 + 0.2×F1 + 0.1×loss)", None),
                        ("mAP50-95", "mAP50-95"),
                        ("mAP50",    "mAP50"),
                        ("F1",       "f1"),
                        ("Precision","precision"),
                        ("Recall",   "recall"),
                    ]

                for label, key in metrics_to_show:
                    if key:
                        b_val = float(bm.get(key, 0.0) or 0.0)
                        w_val = float(wm.get(key, 0.0) or 0.0)
                    else:
                        b_val = result.get("baseline_score", 0.0)
                        w_val = result.get("winner_score", 0.0)
                    delta = w_val - b_val
                    st.metric(
                        label=f"{label} (baseline → победитель)",
                        value=f"{w_val:.4f}",
                        delta=f"{delta:+.4f}",
                        delta_color="normal",
                    )

            # История SFS
            history = result.get("history", [])
            if history:
                st.subheader("История Фазы 2 (SFS)")
                import pandas as pd
                rows = [
                    {
                        "Итерация": h.get("iteration", "—"),
                        "Лучший пайплайн": h.get("best_pipeline", "—"),
                        "Score (30% эп.)": f"{h.get('score', 0.0):.4f}",
                        "Кандидатов": h.get("candidates_count", h.get("n_candidates", "—")),
                        "Survivors": h.get("survivors_count", h.get("n_survivors", "—")),
                    }
                    for h in history
                ]
                st.dataframe(pd.DataFrame(rows), use_container_width=True)

            # Survivors Фазы 1
            survivors = result.get("phase1_survivors", [])
            if survivors:
                with st.expander("Survivors Фазы 1 (SHA-скрининг по группам)"):
                    import pandas as pd
                    rows = [
                        {
                            "Метод": s["display"],
                            "Score (30% эп.)": f"{s['score']:.4f}",
                        }
                        for s in sorted(survivors, key=lambda x: x["score"], reverse=True)
                    ]
                    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    # Лог
    with st.expander("Полный лог выполнения", expanded=False):
        full_log = "\n".join(st.session_state.p2_log_lines)
        st.code(full_log, language=None)
        st.download_button(
            label="Скачать лог",
            data=full_log,
            file_name=f"p2_log_{st.session_state.p2_dataset}.txt",
            mime="text/plain",
        )

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Запустить ещё раз", use_container_width=True):
            _reset()
            st.rerun()
    with col2:
        if st.button("Перейти к обучению", type="primary", use_container_width=True):
            st.switch_page("pages/3_Обучение.py")
