"""
module3_preprocessing_search.py - Автоматический подбор пайплайна предобработки

Алгоритм: SFS (Sequential Forward Selection) + SHA (Successive Halving)

Научное обоснование:
- SFS: Kohavi & John (1997) "Wrappers for feature subset selection", AI, 97, 273-324
- SHA: Jamieson & Talwalkar (2016) "Non-stochastic best arm identification", AISTATS, 240-248
- CASH: Thornton et al. (2013) "Auto-WEKA", KDD, 847-855

Параметры методов:
- Gonzalez & Woods (2018) Digital Image Processing
- Tomasi & Manduchi (1998) ICCV, 839-846
- Pisano et al. (1998) J. Digital Imaging, 11(4), 193-200
"""

import os
import gc
import json
import math
import shutil
import logging
import traceback
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

import cv2
import numpy as np
import torch

# -- Константы алгоритма ----------------------------------------------------

ETA = 2          # коэффициент отсева SHA (Jamieson & Talwalkar, 2016)
EPSILON = 0.005  # порог улучшения mAP (стандарт в object detection)
FAST_EPOCHS_RATIO = 0.30  # 30% эпох для быстрого обучения


def composite_score(metrics: Dict[str, float]) -> float:
    """
    Скалярная оценка для ранжирования кандидатов предобработки (детекция).

    Используется одна метрика: mAP50-95 (COCO primary metric).
    Lin et al. (2014) "Microsoft COCO: Common Objects in Context", ECCV:
    mAP усреднённый по IoU от 0.50 до 0.95 - основная метрика ранжирования.

    Формула унифицирована с score_from_metrics (6_Объединение.py).
    """
    return float(metrics.get('mAP50-95', 0.0))


# ==============================================================================
# ОПИСАНИЕ ПУЛА МЕТОДОВ
# ==============================================================================

def build_candidate_pool() -> List[Dict[str, Any]]:
    """
    Формирует пул методов-кандидатов согласно ТЗ.

    Итого 13 кандидатов:
      1  x оригинал
      2  x median (ksize=3, 5)
      2  x gaussian (ksize=(3,3), (5,5))
      2  x bilateral (sigma=75, 150)
      2  x CLAHE (clip=1.0, 2.0)
      2  x unsharp_mask (amount=0.5, 1.0)
      2  x normalization (z-score, min-max)
    """
    pool = [
        # -- Оригинал ------------------------------------------------------
        {
            'id': 'original',
            'display': 'Оригинал (baseline)',
            'methods': [],
            'params': {},
        },
        # -- Median filter -------------------------------------------------
        {
            'id': 'median_k3',
            'display': 'Median (ksize=3)',
            'methods': ['denoise'],
            'params': {'denoise': {'method': 'median', 'ksize': 3}},
        },
        {
            'id': 'median_k5',
            'display': 'Median (ksize=5)',
            'methods': ['denoise'],
            'params': {'denoise': {'method': 'median', 'ksize': 5}},
        },
        # -- Gaussian blur --------------------------------------------------
        {
            'id': 'gaussian_3x3',
            'display': 'Gaussian blur (3x3)',
            'methods': ['denoise'],
            'params': {'denoise': {'method': 'gaussian', 'ksize': 3}},
        },
        {
            'id': 'gaussian_5x5',
            'display': 'Gaussian blur (5x5)',
            'methods': ['denoise'],
            'params': {'denoise': {'method': 'gaussian', 'ksize': 5}},
        },
        # -- Bilateral filter -----------------------------------------------
        {
            'id': 'bilateral_s75',
            'display': 'Bilateral (d=9, s=75)',
            'methods': ['denoise'],
            'params': {'denoise': {'method': 'bilateral', 'd': 9,
                                   'sigma_color': 75, 'sigma_space': 75}},
        },
        {
            'id': 'bilateral_s150',
            'display': 'Bilateral (d=9, s=150)',
            'methods': ['denoise'],
            'params': {'denoise': {'method': 'bilateral', 'd': 9,
                                   'sigma_color': 150, 'sigma_space': 150}},
        },
        # -- CLAHE ----------------------------------------------------------
        {
            'id': 'clahe_c10',
            'display': 'CLAHE (clip=1.0)',
            'methods': ['contrast_enhancement'],
            'params': {'contrast_enhancement': {'method': 'clahe', 'clip_limit': 1.0,
                                                'tile_grid_size': (8, 8)}},
        },
        {
            'id': 'clahe_c20',
            'display': 'CLAHE (clip=2.0)',
            'methods': ['contrast_enhancement'],
            'params': {'contrast_enhancement': {'method': 'clahe', 'clip_limit': 2.0,
                                                'tile_grid_size': (8, 8)}},
        },
        # -- Unsharp mask ---------------------------------------------------
        {
            'id': 'unsharp_a05',
            'display': 'Unsharp mask (amount=0.5)',
            'methods': ['sharpening'],
            'params': {'sharpening': {'method': 'unsharp_mask', 'alpha': 0.5}},
        },
        {
            'id': 'unsharp_a10',
            'display': 'Unsharp mask (amount=1.0)',
            'methods': ['sharpening'],
            'params': {'sharpening': {'method': 'unsharp_mask', 'alpha': 1.0}},
        },
        # -- Нормализация ---------------------------------------------------
        {
            'id': 'norm_zscore',
            'display': 'Нормализация Z-score',
            'methods': ['normalize'],
            'params': {'normalize': {'method': 'zscore'}},
        },
        {
            'id': 'norm_minmax',
            'display': 'Нормализация Min-Max',
            'methods': ['normalize'],
            'params': {'normalize': {'method': 'minmax'}},
        },
    ]

    return pool


# ==============================================================================
# СТРУКТУРЫ ДАННЫХ
# ==============================================================================

@dataclass
class Pipeline:
    """Пайплайн предобработки - последовательность шагов."""
    steps: List[str] = field(default_factory=list)
    methods: List[str] = field(default_factory=list)
    params: Dict[str, Any] = field(default_factory=dict)
    score:   float            = 0.0
    metrics: Dict[str, float] = field(default_factory=dict)
    display_name: str = ''

    def clone(self) -> 'Pipeline':
        import copy
        return Pipeline(
            steps=list(self.steps),
            methods=list(self.methods),
            params=copy.deepcopy(self.params),
            score=self.score,
            metrics=copy.deepcopy(self.metrics),
            display_name=self.display_name,
        )


def _make_pipeline_from_base_and_candidate(
    base: Pipeline,
    candidate: Dict[str, Any]
) -> Pipeline:
    """Создаёт новый пайплайн = base + один метод-кандидат."""
    import copy
    p = base.clone()
    p.steps.append(candidate['id'])

    for method in candidate['methods']:
        if method not in p.methods:
            p.methods.append(method)

    for k, v in candidate['params'].items():
        p.params[k] = copy.deepcopy(v)

    p.display_name = ' -> '.join(p.steps) if p.steps else 'original'
    return p


# ==============================================================================
# ПРИМЕНЕНИЕ ПРЕДОБРАБОТКИ К ВРЕМЕННОМУ ДАТАСЕТУ
# ==============================================================================

def _apply_normalize(image: np.ndarray, method: str) -> np.ndarray:
    """Применяет нормализацию к изображению."""
    img_f = image.astype(np.float32)
    if method == 'zscore':
        mean, std = img_f.mean(), img_f.std()
        if std < 1e-8:
            return image
        normalized = (img_f - mean) / std
        normalized = ((normalized - normalized.min()) /
                      (normalized.max() - normalized.min() + 1e-8) * 255)
        return normalized.astype(np.uint8)
    elif method == 'minmax':
        mn, mx = img_f.min(), img_f.max()
        if mx - mn < 1e-8:
            return image
        normalized = (img_f - mn) / (mx - mn) * 255
        return normalized.astype(np.uint8)
    return image


def apply_pipeline_to_image(image: np.ndarray, pipeline: Pipeline) -> np.ndarray:
    """Применяет пайплайн к одному изображению."""
    if not pipeline.methods:
        return image

    result = image.copy()
    try:
        from Preprocessing.methods import PreprocessingMethods

        standard_methods = [m for m in pipeline.methods if m != 'normalize']
        standard_params = {k: v for k, v in pipeline.params.items()
                          if k != 'normalize'}

        if standard_methods:
            result = PreprocessingMethods.apply_pipeline(
                result, standard_methods, standard_params
            )

        if 'normalize' in pipeline.methods:
            norm_params = pipeline.params.get('normalize', {})
            norm_method = norm_params.get('method', 'minmax')
            result = _apply_normalize(result, norm_method)

    except Exception as e:
        print(f"[WARNING] Ошибка применения пайплайна: {e}")
        return image

    return result


def create_preprocessed_dataset(
    source_dataset_path: Path,
    pipeline: Pipeline,
    target_path: Path,
    splits: List[str] = None,
    progress_callback=None,
) -> Path:
    """
    Создаёт временный датасет с применённым пайплайном предобработки.
    """
    if splits is None:
        splits = ['train', 'valid', 'test']

    if target_path.exists():
        shutil.rmtree(target_path)

    for split in splits:
        (target_path / split / 'images').mkdir(parents=True, exist_ok=True)
        (target_path / split / 'labels').mkdir(parents=True, exist_ok=True)

    yaml_src = source_dataset_path / 'data.yaml'
    if yaml_src.exists():
        # Переписываем пути на абсолютные - относительные пути вида ../train/images
        # не работают когда YOLO запускает val() из своей внутренней директории.
        # Папки train/valid/test создаются всегда (mkdir выше), поэтому
        # пишем абсолютные пути безусловно, не проверяя exists().
        try:
            import yaml as _yaml
            with open(yaml_src, 'r', encoding='utf-8') as _f:
                _yaml_data = _yaml.safe_load(_f)
            # train -> target/train/images
            # Относительные пути - работают при любом перемещении датасета
            _yaml_data.pop('path', None)
            _yaml_data['train'] = '../train/images'
            _yaml_data['val']   = '../valid/images'
            _yaml_data['test']  = '../test/images' 
            with open(target_path / 'data.yaml', 'w', encoding='utf-8') as _f:
                _yaml.dump(_yaml_data, _f, allow_unicode=True)
            # Лог для диагностики - печатаем итоговый yaml
            print(f"[YAML] Записан data.yaml в {target_path}:")
            print(f"  train: {_yaml_data.get('train')}")
            print(f"  val:   {_yaml_data.get('val')}")
            print(f"  test:  {_yaml_data.get('test')}")
        except Exception:
            shutil.copy(yaml_src, target_path / 'data.yaml')

    for split in splits:
        images_dir = source_dataset_path / split / 'images'
        labels_dir = source_dataset_path / split / 'labels'

        if not images_dir.exists():
            continue

        image_files = sorted(
            list(images_dir.glob('*.jpg')) +
            list(images_dir.glob('*.png')) +
            list(images_dir.glob('*.jpeg'))
        )

        for i, img_path in enumerate(image_files):
            if progress_callback:
                progress_callback(i, len(image_files), split)

            # Читаем в оригинальном формате - цветные датасеты сохраняют RGB.
            # IMREAD_GRAYSCALE терял цветовую информацию у цветных датасетов,
            # что приводило к некорректному сравнению с baseline.
            img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
            if img is None:
                continue

            processed = apply_pipeline_to_image(img, pipeline)
            out_path = target_path / split / 'images' / img_path.name
            cv2.imwrite(str(out_path), processed)

            label_src = labels_dir / (img_path.stem + '.txt')
            if label_src.exists():
                shutil.copy(label_src, target_path / split / 'labels' / label_src.name)

    return target_path


# ==============================================================================
# БЫСТРОЕ ОБУЧЕНИЕ (PROXY TRAINING)
# ==============================================================================

def train_candidate_fast(
    pipeline: Pipeline,
    source_dataset_path: Path,
    model_config: Dict[str, Any],
    fast_epochs: int,
    work_dir: Path,
    candidate_idx: int,
    log_fn=None,
    use_early_stopping: bool = False,
    early_stopping_patience: int = 10,
    eval_split: str = 'val',
) -> Dict[str, float]:
    """
    Обучает proxy-модель на предобработанном датасете.
    Возвращает словарь метрик {'mAP50-95', 'mAP50', 'f1'}.
    Использует механизм очистки памяти из Модуля 2.

    ВАЖНО: tmp_dataset удаляется только ПОСЛЕ того как _run_training полностью
    завершил val() - иначе YOLO не найдёт data.yaml при eval_split='test'.
    """
    if log_fn is None:
        log_fn = print

    candidate_name = pipeline.display_name.replace(' ', '_').replace('->', 'to')[:40]
    tmp_dataset = work_dir / f'tmp_ds_{candidate_idx}'
    result_dir = work_dir / f'train_{candidate_idx}'

    metrics: Dict[str, float] = {'mAP50-95': 0.0, 'mAP50': 0.0, 'f1': 0.0}

    try:
        log_fn(f"    Подготовка датасета: {pipeline.display_name}")
        create_preprocessed_dataset(
            source_dataset_path=source_dataset_path,
            pipeline=pipeline,
            target_path=tmp_dataset,
        )

        # _run_training сам не удаляет tmp_dataset - только result_dir (веса).
        # tmp_dataset живёт до выхода из try-блока, чтобы val() мог
        # прочитать data.yaml с абсолютными путями на split='test'.
        metrics = _run_training(
            dataset_path=tmp_dataset,
            model_config=model_config,
            epochs=fast_epochs,
            result_dir=result_dir,
            log_fn=log_fn,
            use_early_stopping=use_early_stopping,
            early_stopping_patience=early_stopping_patience,
            eval_split=eval_split,
        )

    except Exception as e:
        log_fn(f"    [ERROR] Ошибка обучения кандидата {candidate_name}: {e}")
        log_fn(traceback.format_exc())
        metrics = {'mAP50-95': 0.0, 'mAP50': 0.0, 'f1': 0.0}

    finally:
        # Очищаем GPU-память
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()

        # Теперь безопасно удаляем временный датасет
        if tmp_dataset.exists():
            try:
                shutil.rmtree(tmp_dataset)
            except Exception:
                pass

    return metrics


def _run_training(
    dataset_path: Path,
    model_config: Dict[str, Any],
    epochs: int,
    result_dir: Path,
    log_fn,
    use_early_stopping: bool = False,
    early_stopping_patience: int = 10,
    eval_split: str = 'val',
    resume_from: str = '',
    keep_weights: bool = False,
    max_epochs_for_schedule: int = 0,
    disable_mosaic: bool = False,
    seed: int = 0,
) -> Dict[str, float]:
    """
    Запускает обучение через wrapper'ы из Модуля 2.
    Возвращает словарь метрик {'mAP50-95', 'mAP50', 'f1'}.
    """
    result_dir.mkdir(parents=True, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_type = model_config.get('type', 'yolo')
    metrics: Dict[str, float] = {'mAP50-95': 0.0, 'mAP50': 0.0, 'f1': 0.0}

    try:
        if model_type == 'yolo':
            from ultralytics import YOLO
            model_size = model_config.get('size', 'n')

            # Определяем число уже обученных эпох из метаданных last.pt
            if resume_from and os.path.exists(resume_from):
                model = YOLO(resume_from)
                log_fn(
                    f"  [RESUME] YOLO загружен с весов: {os.path.basename(resume_from)}"
                    f", будет обучено: {epochs} эп., lr0=0.001 (fine-tune)"
                )
            else:
                model = YOLO(f'yolov8{model_size}.pt')

            yaml_path = dataset_path / 'data.yaml'

            # imgsz: приоритет model_config, fallback на автодетект из датасета
            if model_config.get('imgsz'):
                imgsz = int(model_config['imgsz'])
            else:
                try:
                    from Train.Universal_train.universal_model_trainer import get_image_size_from_dataset
                    imgsz = get_image_size_from_dataset(dataset_path)
                except Exception:
                    imgsz = 640

            # epochs - это уже дельта (при warm-start) или полное число (с нуля).
            # Минимум 1 - защита от нулевого запуска.
            epochs_delta = max(1, epochs)

            # Определяем lr0 для warm-start
            _is_warmstart = bool(resume_from and os.path.exists(resume_from))
            if _is_warmstart:
                _lr0 = 0.001       # 1/10 дефолтного 0.01
                _lrf = 0.5         # final lr = lr0 * lrf = 0.0005
                _warmup_ep = 1.0   # 1 эпоха warmup для плавного старта
            else:
                _lr0 = 0.01        # дефолт Ultralytics
                _lrf = 0.01        # дефолт Ultralytics
                _warmup_ep = 3.0   # дефолт Ultralytics

            # Логика lr-schedule и прерывания
            _use_schedule_override = (
                max_epochs_for_schedule > 0
                and max_epochs_for_schedule > epochs_delta
            )
            _actual_train_epochs = (
                max_epochs_for_schedule if _use_schedule_override
                else epochs_delta
            )
            _stop_at_epoch = epochs_delta  # реально нужное число эпох

            if _use_schedule_override:
                # Callback прерывает обучение после _stop_at_epoch эпох.
                # trainer.epoch - 0-based счётчик внутри текущего запуска.
                def _early_stop_cb(trainer):
                    if (trainer.epoch + 1) >= _stop_at_epoch:
                        trainer.stop = True
                model.add_callback('on_fit_epoch_end', _early_stop_cb)
                log_fn(
                    f"  [LR-SCHEDULE] Запуск на {_actual_train_epochs} эп., "
                    f"прерывание после {_stop_at_epoch} эп. "
                    f"(lr-schedule согласован с автоподбором)"
                )

            train_kwargs = dict(
                data=str(yaml_path),
                epochs=_actual_train_epochs,
                imgsz=imgsz,
                batch=model_config.get('batch', -1),
                device=device,
                project=str(result_dir),
                name='run',
                exist_ok=True,
                workers=0,
                cache=False,
                verbose=False,
                save=True,
                # resume=False явно - управляем эпохами через дельту сами.
                # resume=True потребовало бы что last.pt содержит ровно те
                # total_epochs что были при исходном запуске, что не выполняется
                # при сравнении разных процентов скрининга.
                resume=False,
                # LR параметры: сниженные при warm-start, дефолтные при обучении с нуля.
                lr0=_lr0,
                lrf=_lrf,
                warmup_epochs=_warmup_ep,
                # close_mosaic: если disable_mosaic=True - отключаем полностью.
                close_mosaic=0 if disable_mosaic else 10,
                # seed: фиксирует случайность внутри Ultralytics
                seed=seed,
            )
            if use_early_stopping:
                train_kwargs['patience'] = early_stopping_patience
            else:
                # patience=0 отключает встроенный ES Ultralytics для честного сравнения
                train_kwargs['patience'] = 0

            results = model.train(**train_kwargs)
            train_metrics_dict = results.results_dict

            # Отдельная валидация на нужном сплите (valid для SHA, test для финала)
            best_pt = result_dir / 'run' / 'weights' / 'best.pt'
            eval_model = YOLO(str(best_pt) if best_pt.exists() else f'yolov8{model_size}.pt')
            # ultralytics принимает только 'val'/'train'/'test' - нормализуем 'valid' -> 'val'
            _yolo_split = 'val' if eval_split in ('val', 'valid') else eval_split
            val_res = eval_model.val(
                data=str(dataset_path / 'data.yaml'),
                split=_yolo_split,
                device=device,
                verbose=False,
            )
            val_dict = val_res.results_dict

            map5095 = float(val_dict.get('metrics/mAP50-95(B)', 0.0))
            map50   = float(val_dict.get('metrics/mAP50(B)',    0.0))
            # F1 не отдаётся напрямую через results_dict - считаем вручную
            # как в YOLOWrapper.extract_metrics() из Модуля 2
            _pre = float(val_dict.get('metrics/precision(B)', 0.0))
            _rec = float(val_dict.get('metrics/recall(B)', 0.0))
            f1 = 2 * _pre * _rec / (_pre + _rec) if (_pre + _rec) > 0 else 0.0

            metrics = {
                'mAP50-95':  map5095,
                'mAP50':     map50,
                'f1':        f1,
                'precision': _pre,
                'recall':    _rec,
                '_ckpt_path': str(result_dir / 'run' / 'weights' / 'last.pt'),
            }

            del model, results, eval_model, val_res

        elif model_type in ('faster_rcnn', 'retinanet'):
            from Train.Universal_train.universal_model_trainer import (
                FasterRCNNWrapper, RetinaNetWrapper,
                YOLODatasetInfo, YOLOToFasterRCNNDataset,
                calculate_optimal_batch_size, get_available_vram_gb,
            )
            from torch.utils.data import DataLoader

            vram = get_available_vram_gb()
            batch_size = calculate_optimal_batch_size(model_type, vram)

            if model_type == 'faster_rcnn':
                wrapper = FasterRCNNWrapper(pretrained=True)
            else:
                wrapper = RetinaNetWrapper(pretrained=True)

            num_classes = YOLODatasetInfo.get_num_classes(dataset_path)
            wrapper.initialize(num_classes=num_classes)
            wrapper.model.to(device)

            # Базовый lr для SGD - используется и при обычном обучении
            _base_lr = 0.005

            optimizer = torch.optim.SGD(
                wrapper.model.parameters(),
                lr=_base_lr,
                momentum=0.9,
                weight_decay=0.0005,
            )

            # Warm-start для Faster R-CNN / RetinaNet.
            # В отличие от YOLO, здесь нет метаданных эпох в чекпоинте
            resume_start_epoch_det = 0
            if resume_from and os.path.exists(resume_from):
                try:
                    ckpt_det = torch.load(
                        resume_from, map_location=device, weights_only=False
                    )
                    # (a) Веса модели
                    wrapper.model.load_state_dict(ckpt_det['model_state_dict'])
                    # (b) Optimizer state (momentum buffers)
                    if 'optimizer_state_dict' in ckpt_det:
                        optimizer.load_state_dict(ckpt_det['optimizer_state_dict'])
                        # (c) Сброс lr - ПОСЛЕ load_state_dict, иначе будет перезаписан
                        for _pg in optimizer.param_groups:
                            _pg['lr'] = _base_lr
                    resume_start_epoch_det = int(ckpt_det.get('epoch', 0))
                    log_fn(
                        f"  [RESUME] {model_type} загружен с весов: "
                        f"{os.path.basename(resume_from)} "
                        f"(эпоха {resume_start_epoch_det}, lr={_base_lr})"
                    )
                except Exception as _re:
                    log_fn(
                        f"  [RESUME] Не удалось загрузить чекпоинт {model_type}: "
                        f"{_re} - обучаем с нуля"
                    )
                    resume_start_epoch_det = 0

            def collate_fn(batch):
                return tuple(zip(*batch))

            train_ds = YOLOToFasterRCNNDataset(dataset_path, split='train')
            val_ds   = YOLOToFasterRCNNDataset(dataset_path, split=eval_split)
            train_loader = DataLoader(train_ds, batch_size=batch_size,
                                      shuffle=True, collate_fn=collate_fn, num_workers=0)
            val_loader = DataLoader(val_ds, batch_size=batch_size,
                                    shuffle=False, collate_fn=collate_fn, num_workers=0)

            for epoch in range(epochs):
                wrapper.train_epoch(train_loader, device, optimizer, epoch)

            val_metrics = wrapper.validate(val_loader, device)

            # Сохраняем чекпоинт для следующего warm-start прогона если нужно
            if keep_weights:
                _ckpt_det_path = result_dir / 'checkpoint_det.pt'
                result_dir.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        'model_state_dict':     wrapper.model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        # resume_start_epoch_det + epochs = абсолютная эпоха.
                        # При следующем warm-start этот счётчик не используется
                        # напрямую (в отличие от YOLO), но полезен для лога.
                        'epoch': resume_start_epoch_det + epochs,
                        'num_classes': wrapper.num_classes,
                    },
                    _ckpt_det_path,
                )
                log_fn(f"  [CKPT] Сохранён чекпоинт: {_ckpt_det_path}")

            metrics = {
                'mAP50-95': float(val_metrics.get('mAP50-95', 0.0)),
                'mAP50':    float(val_metrics.get('mAP50',    0.0)),
                'f1':       float(val_metrics.get('f1',       0.0)),
                # Путь к чекпоинту для warm-start следующего прогона.
                # При keep_weights=False файл не создаётся - пустая строка.
                '_ckpt_path': str(result_dir / 'checkpoint_det.pt') if keep_weights else '',
            }

            del wrapper, train_loader, val_loader

        log_fn(f"    mAP50-95={metrics['mAP50-95']:.4f}  mAP50={metrics['mAP50']:.4f}  "
               f"f1={metrics['f1']:.4f}  "
               f"[split={eval_split}]")

    except Exception as e:
        log_fn(f"    [ERROR] Ошибка при обучении: {e}")
        log_fn(traceback.format_exc())
        metrics = {'mAP50-95': 0.0, 'mAP50': 0.0, 'f1': 0.0}

    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        # result_dir удаляем только если keep_weights=False (обычный SHA-скрининг)
        if not keep_weights and result_dir.exists():
            try:
                shutil.rmtree(result_dir)
            except Exception:
                pass

    return metrics

# Обучение до 100% с сохранением историей

def _run_training_warmstart(
    dataset_path: Path,
    model_config: Dict[str, Any],
    epochs_delta: int,
    result_dir: Path,
    log_fn,
    resume_from: str,
    use_early_stopping: bool = True,
    early_stopping_patience: int = 20,
    eval_split: str = 'val',
    disable_mosaic: bool = True,
    seed: int = 0,
) -> Dict[str, float]:
    """
    Дообучение YOLO warm-start с дельта-эпохами.
    """
    result_dir.mkdir(parents=True, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_type = model_config.get('type', 'yolo')
    metrics: Dict[str, float] = {'mAP50-95': 0.0, 'mAP50': 0.0, 'f1': 0.0}

    try:
        if model_type == 'yolo':
            from ultralytics import YOLO
            import yaml as _yaml

            model_size = model_config.get('size', 'n')
            imgsz     = model_config.get('imgsz', 640)

            # Ищем data.yaml
            yaml_path = dataset_path / 'data.yaml'
            if not yaml_path.exists():
                for _p in dataset_path.rglob('data.yaml'):
                    yaml_path = _p
                    break

            # Загружаем модель из checkpoint для resume
            if resume_from and Path(resume_from).exists():
                model = YOLO(resume_from)
                log_fn(f"  [WARMSTART] Загружен checkpoint: {resume_from}")
            else:
                model = YOLO(f'yolov8{model_size}.pt')
                log_fn(f"  [WARMSTART] checkpoint не найден - обучение с нуля")

            train_kwargs = dict(
                data=str(yaml_path),
                epochs=epochs_delta,
                imgsz=imgsz,
                batch=model_config.get('batch', -1),
                device=device,
                project=str(result_dir),
                name='run',
                exist_ok=True,
                workers=0,
                cache=False,
                verbose=False,
                save=True,
                # resume=False: передаём дельту эпох явно.
                # resume=True в Ultralytics имеет баг - иногда обучает
                # неверное число эпох. Дельта + resume=False гарантирует
                # ровно epochs_delta дополнительных эпох.
                resume=False,
                # lr0 снижен для fine-tune: веса загружены из checkpoint,
                # высокий lr0 дестабилизирует уже обученные веса.
                lr0=0.001,
                lrf=0.01,
                warmup_epochs=0.0,
                close_mosaic=0 if disable_mosaic else 10,
                seed=seed,
            )

            if use_early_stopping:
                train_kwargs['patience'] = early_stopping_patience
            else:
                train_kwargs['patience'] = 0

            results = model.train(**train_kwargs)

            # Оценка на нужном сплите из best.pt
            best_pt = result_dir / 'run' / 'weights' / 'best.pt'
            eval_model = YOLO(
                str(best_pt) if best_pt.exists() else f'yolov8{model_size}.pt')
            _yolo_split = 'val' if eval_split in ('val', 'valid') else eval_split
            val_res = eval_model.val(
                data=str(yaml_path),
                split=_yolo_split,
                device=device,
                verbose=False,
            )
            box = val_res.results_dict
            metrics['mAP50-95'] = float(box.get('metrics/mAP50-95(B)', 0.0))
            metrics['mAP50']    = float(box.get('metrics/mAP50(B)',    0.0))
            # F1 из best.pt train metrics
            train_dict = results.results_dict
            metrics['f1'] = float(train_dict.get('metrics/F1(B)', 0.0))

            log_fn(f"    [WARMSTART] mAP50-95={metrics['mAP50-95']:.4f}  "
                   f"mAP50={metrics['mAP50']:.4f}  [split={eval_split}]")

        else:
            log_fn(f"  [WARMSTART] модель {model_type} не поддерживает warm-start")

    except Exception as e:
        log_fn(f"  [WARMSTART ОШИБКА] {e}")
        import traceback as _tb
        log_fn(_tb.format_exc())
    finally:
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except Exception:
            pass

    return metrics


def _run_training_full_history(
    dataset_path: Path,
    model_config: Dict[str, Any],
    epochs: int,
    result_dir: Path,
    log_fn,
    early_stopping_patience: int = 20,
    seed: int = 0,
) -> List[float]:
    """
    Обучает YOLO до 100% эпох с close_mosaic=0 и возвращает историю
    mAP50-95 по каждой эпохе в виде списка [ep1, ep2, ..., ep_N].
    """
    result_dir.mkdir(parents=True, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_type = model_config.get('type', 'yolo')
    history: List[float] = []

    try:
        if model_type == 'yolo':
            from ultralytics import YOLO
            model_size = model_config.get('size', 'n')
            model = YOLO(f'yolov8{model_size}.pt')

            yaml_path = dataset_path / 'data.yaml'

            if model_config.get('imgsz'):
                imgsz = int(model_config['imgsz'])
            else:
                try:
                    from Train.Universal_train.universal_model_trainer import get_image_size_from_dataset
                    imgsz = get_image_size_from_dataset(dataset_path)
                except Exception:
                    imgsz = 640

            model.train(
                data=str(yaml_path),
                epochs=epochs,
                imgsz=imgsz,
                batch=model_config.get('batch', -1),
                device=device,
                project=str(result_dir),
                name='run',
                exist_ok=True,
                workers=0,
                cache=False,
                verbose=False,
                save=True,
                resume=False,
                patience=early_stopping_patience,
                close_mosaic=0,      # мозаика одинакова на всех эпохах
                lr0=0.01,            # дефолт Ultralytics
                lrf=0.01,
                warmup_epochs=3.0,
                seed=seed,
            )

            # Читаем историю mAP50-95 из results.csv
            # Ultralytics пишет CSV с заголовком; колонка называется
            # '                  metrics/mAP50-95(B)' (с пробелами - strip нужен)
            csv_path = result_dir / 'run' / 'results.csv'
            if csv_path.exists():
                import csv as _csv
                with open(csv_path, newline='', encoding='utf-8') as f:
                    reader = _csv.DictReader(f)
                    for row in reader:
                        for k, v in row.items():
                            if 'mAP50-95' in k and 'mAP50(B)' not in k:
                                try:
                                    history.append(float(v.strip()))
                                except ValueError:
                                    history.append(0.0)
                                break
                log_fn(f"  [HISTORY] Прочитано {len(history)} эпох из results.csv")
            else:
                log_fn(f"  [HISTORY] results.csv не найден: {csv_path}")

            # Паддинг после ES: если ES остановил обучение раньше epochs
            if len(history) < epochs:
                last_val = history[-1] if history else 0.0
                padded = epochs - len(history)
                history.extend([last_val] * padded)
                log_fn(f"  [HISTORY] ES сработал на эпохе {epochs - padded} "
                       f"из {epochs} - паддинг {padded} эпох "
                       f"значением {last_val:.4f} (Prechelt, 1998)")

            del model

        else:
            # Faster R-CNN / RetinaNet: история собирается вручную по эпохам
            from Train.Universal_train.universal_model_trainer import (
                FasterRCNNWrapper, RetinaNetWrapper,
                YOLODatasetInfo, YOLOToFasterRCNNDataset,
                calculate_optimal_batch_size, get_available_vram_gb,
            )
            from torch.utils.data import DataLoader

            vram = get_available_vram_gb()
            batch_size = calculate_optimal_batch_size(model_type, vram)

            if model_type == 'faster_rcnn':
                wrapper = FasterRCNNWrapper(pretrained=True)
            else:
                wrapper = RetinaNetWrapper(pretrained=True)

            num_classes = YOLODatasetInfo.get_num_classes(dataset_path)
            wrapper.initialize(num_classes=num_classes)
            wrapper.model.to(device)

            _base_lr = 0.005
            optimizer = torch.optim.SGD(
                wrapper.model.parameters(),
                lr=_base_lr, momentum=0.9, weight_decay=0.0005,
            )

            def collate_fn(batch):
                return tuple(zip(*batch))

            train_ds = YOLOToFasterRCNNDataset(dataset_path, split='train')
            val_ds   = YOLOToFasterRCNNDataset(dataset_path, split='val')
            train_loader = DataLoader(train_ds, batch_size=batch_size,
                                      shuffle=True, collate_fn=collate_fn, num_workers=0)
            val_loader   = DataLoader(val_ds, batch_size=batch_size,
                                      shuffle=False, collate_fn=collate_fn, num_workers=0)

            for epoch in range(epochs):
                wrapper.train_epoch(train_loader, device, optimizer, epoch)
                val_metrics = wrapper.validate(val_loader, device)
                history.append(float(val_metrics.get('mAP50-95', 0.0)))

            log_fn(f"  [HISTORY] Собрано {len(history)} эпох (Faster R-CNN/RetinaNet)")

    except Exception as e:
        log_fn(f"  [HISTORY ERROR] {e}")
        log_fn(traceback.format_exc())

    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        # result_dir НЕ удаляем - best.pt может понадобиться для финального

    return history


def _history_score_at(history: List[float], ratio: float) -> float:
    """
    Возвращает max(mAP50-95) по первым ratio*len(history) эпохам.
    """
    if not history:
        return 0.0
    n = len(history)
    k = max(1, int(round(n * ratio)))
    k = min(k, n)
    return max(history[:k])


# Финальное полное обучение (не удаляет веса, eval на test)

def _run_training_final(
    dataset_path: Path,
    model_config: Dict[str, Any],
    epochs: int,
    result_dir: Path,
    log_fn,
    early_stopping_patience: int = 10,
    eval_split: str = 'test',
    seed: int = 0,
) -> Dict[str, float]:
    """
    Финальное обучение победителя на постоянном датасете.
    """
    result_dir.mkdir(parents=True, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_type = model_config.get('type', 'yolo')
    metrics: Dict[str, float] = {'mAP50-95': 0.0, 'mAP50': 0.0, 'f1': 0.0}

    try:
        if model_type == 'yolo':
            from ultralytics import YOLO
            model_size = model_config.get('size', 'n')
            model = YOLO(f'yolov8{model_size}.pt')

            yaml_path = dataset_path / 'data.yaml'

            # imgsz: приоритет model_config, fallback на автодетект из датасета
            if model_config.get('imgsz'):
                imgsz = int(model_config['imgsz'])
            else:
                try:
                    from Train.Universal_train.universal_model_trainer import get_image_size_from_dataset
                    imgsz = get_image_size_from_dataset(dataset_path)
                except Exception:
                    imgsz = 640

            results = model.train(
                data=str(yaml_path),
                epochs=epochs,
                imgsz=imgsz,
                batch=model_config.get('batch', -1),
                device=device,
                project=str(result_dir),
                name='run',
                exist_ok=True,
                workers=0,
                cache=False,
                verbose=False,
                save=True,
                patience=early_stopping_patience,
                seed=seed,
            )
            train_metrics_dict = results.results_dict

            # Валидация на test split с лучшими весами
            best_pt = result_dir / 'run' / 'weights' / 'best.pt'
            eval_model = YOLO(str(best_pt) if best_pt.exists() else f'yolov8{model_size}.pt')
            _yolo_split = 'val' if eval_split in ('val', 'valid') else eval_split
            val_res = eval_model.val(
                data=str(yaml_path),
                split=_yolo_split,
                device=device,
                verbose=False,
            )
            val_dict = val_res.results_dict

            map5095 = float(val_dict.get('metrics/mAP50-95(B)', 0.0))
            map50   = float(val_dict.get('metrics/mAP50(B)',    0.0))
            _pre = float(val_dict.get('metrics/precision(B)', 0.0))
            _rec = float(val_dict.get('metrics/recall(B)', 0.0))
            f1 = 2 * _pre * _rec / (_pre + _rec) if (_pre + _rec) > 0 else 0.0

            metrics = {
                'mAP50-95':  map5095,
                'mAP50':     map50,
                'f1':        f1,
                'precision': _pre,
                'recall':    _rec,
            }

            del model, results, eval_model, val_res

        elif model_type in ('faster_rcnn', 'retinanet'):
            from Train.Universal_train.universal_model_trainer import (
                FasterRCNNWrapper, RetinaNetWrapper,
                YOLODatasetInfo, YOLOToFasterRCNNDataset,
                calculate_optimal_batch_size, get_available_vram_gb,
            )
            from torch.utils.data import DataLoader

            vram = get_available_vram_gb()
            batch_size = calculate_optimal_batch_size(model_type, vram)

            if model_type == 'faster_rcnn':
                wrapper = FasterRCNNWrapper(pretrained=True)
            else:
                wrapper = RetinaNetWrapper(pretrained=True)

            num_classes = YOLODatasetInfo.get_num_classes(dataset_path)
            wrapper.initialize(num_classes=num_classes)
            wrapper.model.to(device)

            def collate_fn(batch):
                return tuple(zip(*batch))

            train_ds = YOLOToFasterRCNNDataset(dataset_path, split='train')
            val_ds   = YOLOToFasterRCNNDataset(dataset_path, split=eval_split)
            train_loader = DataLoader(train_ds, batch_size=batch_size,
                                      shuffle=True, collate_fn=collate_fn, num_workers=0)
            val_loader = DataLoader(val_ds, batch_size=batch_size,
                                    shuffle=False, collate_fn=collate_fn, num_workers=0)

            optimizer = torch.optim.SGD(
                wrapper.model.parameters(), lr=0.005,
                momentum=0.9, weight_decay=0.0005
            )

            for epoch in range(epochs):
                wrapper.train_epoch(train_loader, device, optimizer, epoch)

            val_metrics = wrapper.validate(val_loader, device)
            metrics = {
                'mAP50-95': float(val_metrics.get('mAP50-95', 0.0)),
                'mAP50':    float(val_metrics.get('mAP50',    0.0)),
                'f1':       float(val_metrics.get('f1',       0.0)),
            }

            del wrapper, train_loader, val_loader

        log_fn(f"    [ФИНАЛ] mAP50-95={metrics['mAP50-95']:.4f}  mAP50={metrics['mAP50']:.4f}  "
               f"f1={metrics['f1']:.4f}  "
               f"[split={eval_split}]")

    except Exception as e:
        log_fn(f"    [ERROR] Ошибка при финальном обучении: {e}")
        log_fn(traceback.format_exc())
        metrics = {'mAP50-95': 0.0, 'mAP50': 0.0, 'f1': 0.0}

    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        # result_dir НЕ удаляем - веса победителя остаются

    return metrics


# Утилита SHA

def sha_prune(
    candidates: List[Pipeline],
    eta: int = ETA,
) -> List[Pipeline]:
    """
    Successive Halving: оставляет top-ceil(N/eta) кандидатов по composite score.
    """
    n = len(candidates)
    keep_count = math.ceil(n / eta)
    keep_count = max(1, keep_count)

    sorted_candidates = sorted(candidates, key=lambda p: p.score, reverse=True)
    return sorted_candidates[:keep_count]


# SFS + SHA

@dataclass
class SearchResult:
    """Результат поиска пайплайна."""
    best_pipeline: Pipeline
    best_map: float
    final_map: float
    final_metrics_test: Dict[str, float]
    history: List[Dict]
    total_iterations: int
    stop_reason: str
    winner_dataset_path: Path = field(default=None)


def run_sfs_sha_search(
    source_dataset_path: Path,
    model_config: Dict[str, Any],
    max_epochs: int = 50,
    early_stopping_patience: int = 10,
    datasets_global_path: Path = None,
    log_fn=None,
    progress_callback=None,
) -> SearchResult:
    """
    Запускает SFS+SHA поиск пайплайна предобработки.
    """
    if log_fn is None:
        log_fn = print

    # Определяем корневую папку датасетов
    if datasets_global_path is None:
        _env_path = os.environ.get('DATASETS_GLOBAL_PATH', '')
        if _env_path:
            datasets_global_path = Path(_env_path)
        else:
            # Крайний случай - рядом с исходным датасетом
            datasets_global_path = source_dataset_path.parent

    # Формируем уникальную рабочую папку:
    # <datasets_global_path>/m3_runs/<dataset_name>_<YYYYMMDD_HHMMSS>/
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = f"{source_dataset_path.name}_{timestamp}"
    work_dir = datasets_global_path / 'm3_runs' / run_name
    work_dir.mkdir(parents=True, exist_ok=True)
    log_fn(f"Рабочая папка запуска: {work_dir}")

    # Открываем файл лога в рабочей папке
    log_file_path = work_dir / 'run.log'
    _log_file = open(log_file_path, 'w', encoding='utf-8')

    _original_log_fn = log_fn

    def log_fn(msg: str):
        _original_log_fn(msg)
        try:
            _log_file.write(str(msg) + '\n')
            _log_file.flush()
        except Exception:
            pass

    log_fn(f"Лог сохраняется в: {log_file_path}")

    fast_epochs = max(1, int(max_epochs * FAST_EPOCHS_RATIO))
    log_fn("=" * 70)
    log_fn("МОДУЛЬ 3: АВТОМАТИЧЕСКИЙ ПОДБОР ПАЙПЛАЙНА ПРЕДОБРАБОТКИ")
    log_fn("Алгоритм: SFS + SHA (Kohavi&John 1997; Jamieson&Talwalkar 2016)")
    log_fn("=" * 70)
    log_fn(f"Датасет:         {source_dataset_path.name}")
    log_fn(f"Модель:          {model_config.get('type', '?')} {model_config.get('size', '')}")
    log_fn(f"Max epochs:      {max_epochs}")
    log_fn(f"Fast epochs:     {fast_epochs} (30%, без ES - честное сравнение SHA)")
    log_fn(f"ES patience:     {early_stopping_patience} (только финал)")
    log_fn(f"eta:             {ETA}")
    log_fn(f"epsilon:         {EPSILON}")
    log_fn("")

    # Инициализация
    pool = build_candidate_pool()
    pool_by_id = {c['id']: c for c in pool}

    log_fn(f"Пул методов ({len(pool)} кандидатов):")
    for c in pool:
        log_fn(f"  * {c['id']:20s} -- {c['display']}")
    log_fn("")

    root_pipeline = Pipeline(
        steps=[],
        methods=[],
        params={},
        display_name='original',
    )

    survivors = [root_pipeline]
    history = []
    best_pipeline = root_pipeline
    best_map = 0.0
    stop_reason = ''
    N = len(pool)

    # Итерации SFS
    for iteration in range(1, N + 1):
        log_fn("-" * 70)
        log_fn(f"ИТЕРАЦИЯ {iteration}/{N}")
        log_fn("-" * 70)

        if progress_callback:
            progress_callback(iteration, N, 'expand',
                              f'Итерация {iteration}: расширение кандидатов')

        # Расширение территории
        candidates = []
        for survivor in survivors:
            candidates.append(survivor.clone())

            for candidate_def in pool:
                cid = candidate_def['id']
                if cid == 'original':
                    continue
                if cid in survivor.steps:
                    continue

                new_pipeline = _make_pipeline_from_base_and_candidate(
                    survivor, candidate_def
                )
                candidates.append(new_pipeline)

        # Дедупликация
        seen = set()
        unique_candidates = []
        for c in candidates:
            key = tuple(sorted(c.steps))
            if key not in seen:
                seen.add(key)
                unique_candidates.append(c)
        candidates = unique_candidates

        log_fn(f"Кандидатов после расширения: {len(candidates)}")

        # Быстрое обучение
        log_fn(f"\nБыстрое обучение ({fast_epochs} эп., eval=valid, без ES):")
        for i, candidate in enumerate(candidates):
            log_fn(f"\n  [{i+1}/{len(candidates)}] {candidate.display_name}")
            if progress_callback:
                progress_callback(
                    iteration, N, 'train',
                    f'Итерация {iteration}: обучение [{i+1}/{len(candidates)}] '
                    f'{candidate.display_name}'
                )

            score_metrics = train_candidate_fast(
                pipeline=candidate,
                source_dataset_path=source_dataset_path,
                model_config=model_config,
                fast_epochs=fast_epochs,
                work_dir=work_dir / f'iter_{iteration}',
                candidate_idx=i,
                log_fn=log_fn,
                use_early_stopping=False,
                eval_split='val',
            )
            candidate.metrics = score_metrics
            candidate.score   = composite_score(candidate.metrics)

        # Сортировка
        candidates_sorted = sorted(candidates, key=lambda p: p.score, reverse=True)
        log_fn(f"\nРезультаты (composite score / mAP50-95):")
        for i, c in enumerate(candidates_sorted):
            marker = " +" if i == 0 else ""
            log_fn(f"  {i+1:2d}. {c.display_name:45s} "
                   f"score={c.score:.4f}  mAP={c.metrics.get('mAP50-95', 0.0):.4f}{marker}")

        # SHA отсев
        survivors_new = sha_prune(candidates, eta=ETA)
        eliminated = [c for c in candidates if c not in survivors_new]

        log_fn(f"\nSHA-отсев: оставляем top-{len(survivors_new)} из {len(candidates)}")
        log_fn(f"Выжившие:  {[c.display_name for c in survivors_new]}")
        log_fn(f"Отсеяны:   {[c.display_name for c in eliminated]}")

        current_best = max(candidates, key=lambda p: p.score)
        current_map = current_best.metrics.get('mAP50-95', 0.0)
        iter_data = {
            'iteration': iteration,
            'n_candidates': len(candidates),
            'n_survivors': len(survivors_new),
            'best_pipeline': current_best.display_name,
            'best_map': current_map,
            'candidates': [
                {'name': c.display_name, 'map': round(c.metrics.get('mAP50-95', 0.0), 4)}
                for c in candidates_sorted
            ],
        }
        history.append(iter_data)

        # Критерий остановки
        if len(survivors_new) == 1:
            best_pipeline = survivors_new[0]
            best_map = best_pipeline.metrics.get('mAP50-95', 0.0)
            stop_reason = 'Остался 1 кандидат'
            log_fn(f"\n>>> СТОП: {stop_reason}")
            break

        delta = current_map - best_map
        if iteration > 1 and delta < EPSILON:
            best_pipeline = current_best
            best_map = best_pipeline.metrics.get('mAP50-95', 0.0)
            stop_reason = f'Delta mAP = {delta:.4f} < epsilon = {EPSILON}'
            log_fn(f"\n>>> СТОП: {stop_reason}")
            break

        best_map = current_map
        best_pipeline = current_best
        survivors = survivors_new

        if iteration == N:
            stop_reason = f'Достигнут предел итераций (N={N})'
            log_fn(f"\n>>> СТОП: {stop_reason}")
            break

    # Финальное обучение
    log_fn("")
    log_fn("=" * 70)
    log_fn("ФИНАЛЬНОЕ ОБУЧЕНИЕ (100% эпох, eval=test, с early stopping)")
    log_fn(f"Победитель: {best_pipeline.display_name}")
    log_fn("=" * 70)

    if progress_callback:
        progress_callback(N, N, 'final',
                          f'Финальное обучение: {best_pipeline.display_name}')

    # Создаём постоянный датасет-победитель в рабочей папке запуска
    winner_safe_name = best_pipeline.display_name.replace(' ', '_').replace('->', 'to')[:50]
    winner_dataset_path = work_dir / f'winner_{winner_safe_name}'
    log_fn(f"Создаём датасет-победитель: {winner_dataset_path}")
    create_preprocessed_dataset(
        source_dataset_path=source_dataset_path,
        pipeline=best_pipeline,
        target_path=winner_dataset_path,
    )

    # Финальное обучение на постоянном датасете (не удаляется в finally)
    final_result_dir = work_dir / 'final_train'
    final_metrics = _run_training_final(
        dataset_path=winner_dataset_path,
        model_config={**model_config, 'full_training': True},
        epochs=max_epochs,
        result_dir=final_result_dir,
        log_fn=log_fn,
        early_stopping_patience=early_stopping_patience,
        eval_split='test',
    )
    best_pipeline.metrics = final_metrics
    best_pipeline.score   = composite_score(final_metrics)
    final_map = final_metrics.get('mAP50-95', 0.0)

    # Удаляем все временные папки кандидатов - оставляем только winner
    log_fn("")
    log_fn("Очистка временных папок кандидатов...")
    for item in work_dir.iterdir():
        if item.is_dir() and not item.name.startswith('winner_'):
            try:
                shutil.rmtree(item)
                log_fn(f"  Удалено: {item.name}")
            except Exception as _e:
                log_fn(f"  [WARNING] Не удалось удалить {item.name}: {_e}")

    log_fn("")
    log_fn("=" * 70)
    log_fn("РЕЗУЛЬТАТ")
    log_fn("=" * 70)
    log_fn(f"Лучший пайплайн: {best_pipeline.display_name}")
    log_fn(f"Методы:          {best_pipeline.methods}")
    log_fn(f"Параметры:       {best_pipeline.params}")
    log_fn(f"mAP50-95 (test): {final_map:.4f}")
    log_fn(f"mAP50    (test): {final_metrics.get('mAP50', 0.0):.4f}")
    log_fn(f"f1       (test): {final_metrics.get('f1', 0.0):.4f}")
    log_fn(f"Причина стопа:   {stop_reason}")
    log_fn(f"Датасет-победитель: {winner_dataset_path}")
    log_fn("=" * 70)

    try:
        _log_file.close()
    except Exception:
        pass

    return SearchResult(
        best_pipeline=best_pipeline,
        best_map=best_map,
        final_map=final_map,
        final_metrics_test=final_metrics,
        history=history,
        total_iterations=len(history),
        stop_reason=stop_reason,
        winner_dataset_path=winner_dataset_path,
    )
