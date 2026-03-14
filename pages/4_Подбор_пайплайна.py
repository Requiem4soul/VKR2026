"""
pages/4_Подбор_пайплайна.py — Модуль 3: Автоматический подбор пайплайна предобработки

Алгоритм SFS+SHA:
- SFS: Kohavi & John (1997) Artificial Intelligence, 97, 273-324
- SHA: Jamieson & Talwalkar (2016) AISTATS, 240-248

Поддерживает задачи:
- Детекция:       YOLOv8, Faster R-CNN, RetinaNet
- Классификация:  ResNet-18, ResNet-50, EfficientNet-B0
"""

import sys
import gc
import json
import time
import queue
import threading
import traceback
import streamlit as st
from pathlib import Path

from ui.sidebar import render_sidebar
from ui.state import (
    init_session_state,
    is_path_configured,
    get_available_datasets,
    get_datasets_path,
)

st.set_page_config(
    page_title="Подбор пайплайна — VKR2026",
    page_icon=None,
    layout="wide",
)
init_session_state()
render_sidebar()

if not is_path_configured():
    st.error("Сначала настрой путь к датасетам в разделе **Настройки**.")
    st.stop()

# ── Состояние страницы ─────────────────────────────────────────────────────
_KEYS = {
    'm3_stage': 'configure',
    'm3_task': 'detection',
    'm3_log_lines': [],
    'm3_output_queue': None,
    'm3_thread_done': False,
    'm3_error': None,
    'm3_result': None,
    'm3_dataset': None,
    'm3_model_type': 'yolo',
    'm3_yolo_size': 'n',
    'm3_cls_model': 'resnet18',
    'm3_cls_imgsz': 224,
    'm3_epochs': 30,
    'm3_patience': 10,
    'm3_seed': 42,
}
for k, v in _KEYS.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ── Вспомогательные функции ───────────────────────────────────────────────

def _reset():
    for k, v in _KEYS.items():
        st.session_state[k] = v


def _render_log_box():
    lines = st.session_state.m3_log_lines
    if lines:
        st.code('\n'.join(lines[-300:]), language=None)


def _render_result(result_dict: dict):
    best = result_dict.get('best_pipeline', {})
    final_metrics_test = result_dict.get('final_metrics_test', {})
    stop_reason = result_dict.get('stop_reason', '—')
    iters = result_dict.get('total_iterations', 0)
    history = result_dict.get('history', [])
    task = result_dict.get('task', 'detection')

    st.success("Поиск завершён!")
    st.info("Финальные метрики получены на test split — независимая оценка.")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Лучший пайплайн:**")
        st.code(
            best.get('display_name') or best.get('name') or str(best),
            language=None,
        )
        st.caption(f"Итераций SFS: {iters} | Причина остановки: {stop_reason}")

    with col2:
        st.markdown("**Финальные метрики (test):**")
        if task == 'classification':
            for metric in ['val_auc', 'val_acc', 'val_loss']:
                val = final_metrics_test.get(metric, final_metrics_test.get(f'test_{metric}', None))
                if val is not None:
                    label = {'val_auc': 'AUC', 'val_acc': 'ACC', 'val_loss': 'Loss'}.get(metric, metric)
                    st.metric(label, f"{val:.4f}")
        else:
            for metric in ['mAP50-95', 'mAP50', 'f1']:
                val = final_metrics_test.get(metric)
                if val is not None:
                    st.metric(metric, f"{val:.4f}")

    if history:
        import pandas as pd
        st.markdown("**История итераций SFS:**")
        rows = []
        for h in history:
            if isinstance(h, dict):
                rows.append({
                    'Итерация': h.get('iteration', '—'),
                    'Пайплайн': h.get('pipeline', '—'),
                    'mAP/AUC (быстрый)': f"{h.get('map_fast', 0):.4f}",
                })
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True)


# ── Фоновые функции поиска ────────────────────────────────────────────────

def _run_detection_search_thread(dataset_path, model_config, epochs, patience, q, datasets_path, seed):
    import io, os

    class QWriter(io.TextIOBase):
        def write(self, text):
            if text.strip():
                q.put(('log', text.rstrip()))
            return len(text)
        def flush(self): pass

    old = sys.stdout
    sys.stdout = QWriter()

    try:
        project_root = str(Path(__file__).parent.parent)
        if project_root not in sys.path:
            sys.path.insert(0, project_root)

        os.environ['DATASETS_GLOBAL_PATH'] = str(datasets_path)

        import random, numpy as np, torch
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ['PYTHONHASHSEED'] = str(seed)

        from module3_preprocessing_search import run_sfs_sha_search

        def progress_cb(iteration, total, stage, message):
            q.put(('progress', {
                'iteration': iteration, 'total': total,
                'stage': stage, 'message': message,
            }))

        result = run_sfs_sha_search(
            source_dataset_path=dataset_path,
            model_config=model_config,
            max_epochs=epochs,
            early_stopping_patience=patience,
            datasets_global_path=Path(datasets_path),
            log_fn=lambda msg: q.put(('log', str(msg))),
            progress_callback=progress_cb,
        )

        result_dict = {
            'task': 'detection',
            'best_pipeline': {
                'display_name': result.best_pipeline.display_name
                if hasattr(result.best_pipeline, 'display_name')
                else str(result.best_pipeline),
            },
            'best_map_fast': result.best_map,
            'final_map_100pct': result.final_map,
            'final_metrics_test': result.final_metrics_test,
            'history': result.history,
            'total_iterations': result.total_iterations,
            'stop_reason': result.stop_reason,
            'winner_dataset_path': str(result.winner_dataset_path)
            if result.winner_dataset_path else None,
        }
        q.put(('result', result_dict))
        q.put(('done', 'Поиск завершён'))

    except Exception as e:
        q.put(('error', f"{type(e).__name__}: {e}\n{traceback.format_exc()}"))
        q.put(('done', 'Ошибка!'))
    finally:
        sys.stdout = old


def _run_classification_search_thread(
    dataset_path, cls_model_type, cls_imgsz, epochs, patience, q, datasets_path, seed
):
    """
    Поиск оптимального пайплайна предобработки с proxy-моделью классификации.
    Использует ClassificationTrainer как proxy вместо детекции.
    """
    import io, os

    class QWriter(io.TextIOBase):
        def write(self, text):
            if text.strip():
                q.put(('log', text.rstrip()))
            return len(text)
        def flush(self): pass

    old = sys.stdout
    sys.stdout = QWriter()

    try:
        project_root = str(Path(__file__).parent.parent)
        if project_root not in sys.path:
            sys.path.insert(0, project_root)

        os.environ['DATASETS_GLOBAL_PATH'] = str(datasets_path)

        import random, numpy as np, torch
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ['PYTHONHASHSEED'] = str(seed)

        from module3_preprocessing_search import run_sfs_sha_search

        model_config = {
            'type': cls_model_type,
            'image_size': cls_imgsz,
            'pretrained': True,
        }

        def progress_cb(iteration, total, stage, message):
            q.put(('progress', {
                'iteration': iteration, 'total': total,
                'stage': stage, 'message': message,
            }))

        result = run_sfs_sha_search(
            source_dataset_path=dataset_path,
            model_config=model_config,
            max_epochs=epochs,
            early_stopping_patience=patience,
            datasets_global_path=Path(datasets_path),
            log_fn=lambda msg: q.put(('log', str(msg))),
            progress_callback=progress_cb,
        )

        result_dict = {
            'task': 'classification',
            'best_pipeline': {
                'display_name': result.best_pipeline.display_name
                if hasattr(result.best_pipeline, 'display_name')
                else str(result.best_pipeline),
            },
            'best_map_fast': result.best_map,
            'final_map_100pct': result.final_map,
            'final_metrics_test': result.final_metrics_test,
            'history': result.history,
            'total_iterations': result.total_iterations,
            'stop_reason': result.stop_reason,
            'winner_dataset_path': str(result.winner_dataset_path)
            if result.winner_dataset_path else None,
        }
        q.put(('result', result_dict))
        q.put(('done', 'Поиск завершён'))

    except Exception as e:
        q.put(('error', f"{type(e).__name__}: {e}\n{traceback.format_exc()}"))
        q.put(('done', 'Ошибка!'))
    finally:
        sys.stdout = old


# ══════════════════════════════════════════════════════════════════════════════
# ЭТАП 1: Конфигурация
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.m3_stage == 'configure':

    st.title("Подбор пайплайна предобработки")
    st.markdown(
        "**Модуль 3** автоматически подбирает оптимальный пайплайн предобработки "
        "с помощью алгоритма **SFS+SHA**."
    )

    with st.expander("Как работает алгоритм", expanded=False):
        st.markdown("""
**Sequential Forward Selection (SFS)** итеративно строит пайплайн,
добавляя по одному методу предобработки.

**Successive Halving (SHA)** на каждом шаге отсевает слабых кандидатов
через быстрое частичное обучение (30% эпох).

**Пул методов (13 кандидатов):** Оригинал, Median filter (x2), Gaussian blur (x2),
Bilateral filter (x2), CLAHE (x2), Unsharp mask (x2), Z-score, Min-Max.

*Источники: Kohavi & John (1997); Jamieson & Talwalkar (2016)*
        """)

    st.divider()

    datasets = get_available_datasets()
    if not datasets:
        st.warning("Датасеты не найдены. Проверь путь в Настройках.")
        st.stop()

    # ── Выбор задачи ──────────────────────────────────────────────────────
    st.subheader("0. Тип задачи")
    task = st.radio(
        "Задача",
        options=["detection", "classification"],
        format_func=lambda x: "Детекция" if x == "detection" else "Классификация",
        index=0 if st.session_state.m3_task == "detection" else 1,
        key="m3_task_radio",
        horizontal=True,
    )
    st.session_state.m3_task = task
    st.divider()

    col_left, col_right = st.columns(2, gap="large")

    with col_left:
        st.subheader("1. Датасет")
        selected_dataset = st.selectbox(
            "Датасет для поиска", options=datasets, index=0, key='m3_ds_select',
        )

    with col_right:
        st.subheader("2. Proxy-модель")

        if task == "detection":
            model_type = st.selectbox(
                "Тип модели",
                options=['yolo', 'faster_rcnn', 'retinanet'],
                format_func=lambda x: {
                    'yolo': 'YOLOv8', 'faster_rcnn': 'Faster R-CNN', 'retinanet': 'RetinaNet',
                }[x],
                key='m3_model_select',
            )
            yolo_size = 'n'
            if model_type == 'yolo':
                yolo_size = st.selectbox(
                    "Размер YOLOv8",
                    options=['n', 's', 'm', 'l', 'x'],
                    format_func=lambda x: {'n': 'nano', 's': 'small', 'm': 'medium',
                                            'l': 'large', 'x': 'xlarge'}[x],
                    index=0,
                    key='m3_yolo_size_select',
                )
            cls_model = 'resnet18'
            cls_imgsz = 224
        else:
            model_type = 'cls'
            yolo_size = 'n'
            cls_model = st.selectbox(
                "Архитектура",
                options=['resnet18', 'resnet50', 'efficientnet_b0'],
                format_func=lambda x: {
                    'resnet18': 'ResNet-18 (быстрее)',
                    'resnet50': 'ResNet-50',
                    'efficientnet_b0': 'EfficientNet-B0',
                }[x],
                key='m3_cls_model_select',
            )
            # Виджет записывает значение в session_state[key] автоматически.
            # Не присваивать session_state.m3_cls_imgsz вручную после этого —
            # это вызывает StreamlitAPIException в Streamlit 1.5+.
            cls_imgsz = st.selectbox(
                "Размер изображений", [28, 224], index=1,
                help="28 = как в MedMNIST; 224 = ImageNet-стандарт",
                key='m3_cls_imgsz',
            )
            st.caption(
                "Для SHA-сравнения пайплайнов используется 30% эпох без Early Stopping, "
                "финальное обучение победителя — 100% с ES."
            )

        st.subheader("3. Параметры обучения")
        default_epochs = 30 if task == "detection" else 50
        epochs = st.number_input(
            "Эпох финального обучения", 5, 300, default_epochs, step=5, key='m3_epochs_input',
        )
        patience = st.number_input(
            "Early stopping patience (финал)", 3, 50, 10, key='m3_patience_input',
        )
        fast_ep = max(1, int(epochs * 0.30))
        st.caption(
            f"Быстрое SHA: {fast_ep} эп. без ES | Финал: {epochs} эп. + ES (patience={patience})"
        )

        try:
            import torch
            if torch.cuda.is_available():
                vram = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
                st.info(f"GPU: {vram:.1f} GB VRAM")
            else:
                st.warning("GPU не обнаружен — CPU (медленно)")
        except Exception:
            pass

    st.divider()

    # ── Воспроизводимость ─────────────────────────────────────────────────
    with st.expander("Воспроизводимость (Seed)", expanded=False):
        st.markdown(
            "Фиксирует все источники случайности для идентичных результатов "
            "при повторном запуске с тем же seed."
        )
        seed = st.number_input(
            "Random seed", 0, 2 ** 31 - 1, value=42, key='m3_seed_input',
        )

    # ── Оценка времени ─────────────────────────────────────────────────────
    n_candidates = 13
    n_iters_est = 4
    if task == "detection":
        time_per = {'yolo': 2, 'faster_rcnn': 5, 'retinanet': 4}.get(model_type, 3)
    else:
        time_per = {'resnet18': 3, 'resnet50': 5, 'efficientnet_b0': 3}.get(cls_model, 4)
    total_est = n_candidates * n_iters_est * time_per

    st.info(
        f"Примерное время: {total_est}–{total_est * 2} мин. | "
        f"Всего обучений: ~{n_candidates * n_iters_est}"
    )

    if st.button("Запустить поиск пайплайна", type='primary', use_container_width=True):
        st.session_state.m3_dataset = selected_dataset
        st.session_state.m3_model_type = model_type
        st.session_state.m3_yolo_size = yolo_size
        st.session_state.m3_cls_model = cls_model
        # m3_cls_imgsz уже записан в session_state виджетом через key='m3_cls_imgsz'.
        # Повторное присвоение вызывает StreamlitAPIException — не трогаем.
        st.session_state.m3_epochs = epochs
        st.session_state.m3_patience = patience
        st.session_state.m3_seed = seed if 'm3_seed_input' in st.session_state else 42
        st.session_state.m3_log_lines = []
        st.session_state.m3_error = None
        st.session_state.m3_result = None
        st.session_state.m3_thread_done = False
        st.session_state.m3_stage = 'running'
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# ЭТАП 2: Запуск и мониторинг
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.m3_stage == 'running':

    task = st.session_state.m3_task
    task_label = "классификации" if task == "classification" else "детекции"
    st.title(f"Поиск пайплайна ({task_label}) — выполняется...")

    dataset_name = st.session_state.m3_dataset
    model_type = st.session_state.m3_model_type
    yolo_size = st.session_state.m3_yolo_size
    cls_model = st.session_state.m3_cls_model
    cls_imgsz = st.session_state.m3_cls_imgsz
    epochs = st.session_state.m3_epochs
    patience = st.session_state.m3_patience
    seed = st.session_state.m3_seed
    datasets_path = str(get_datasets_path())

    try:
        project_root = str(Path(__file__).parent.parent)
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        from Data.Datasets.dataset_work import get_dataset_path
        dataset_path = get_dataset_path(dataset_name)
    except Exception:
        dataset_path = get_datasets_path() / dataset_name

    if st.session_state.m3_output_queue is None:
        q = queue.Queue()
        st.session_state.m3_output_queue = q

        if task == "detection":
            model_config = {'type': model_type, 'size': yolo_size}
            thread = threading.Thread(
                target=_run_detection_search_thread,
                args=(dataset_path, model_config, epochs, patience, q, datasets_path, seed),
                daemon=True,
            )
        else:
            thread = threading.Thread(
                target=_run_classification_search_thread,
                args=(dataset_path, cls_model, cls_imgsz, epochs, patience,
                      q, datasets_path, seed),
                daemon=True,
            )
        thread.start()

    if st.button("Остановить поиск"):
        st.session_state.m3_stage = 'configure'
        st.session_state.m3_output_queue = None
        st.warning("Поиск прерван пользователем.")

    status_placeholder = st.empty()
    progress_bar = st.progress(0)
    log_placeholder = st.empty()

    q = st.session_state.m3_output_queue
    if q is not None:
        while True:
            try:
                msg_type, payload = q.get_nowait()
                if msg_type == 'log':
                    st.session_state.m3_log_lines.append(str(payload))
                elif msg_type == 'progress':
                    pct = int(payload['iteration'] / max(payload['total'], 1) * 100)
                    progress_bar.progress(pct)
                    status_placeholder.info(
                        f"Итерация {payload['iteration']}/{payload['total']}: {payload['message']}"
                    )
                elif msg_type == 'result':
                    st.session_state.m3_result = payload
                elif msg_type == 'error':
                    st.session_state.m3_error = str(payload)
                elif msg_type == 'done':
                    st.session_state.m3_thread_done = True
            except queue.Empty:
                break

    lines = st.session_state.m3_log_lines
    if lines:
        with log_placeholder.container():
            st.code('\n'.join(lines[-200:]), language=None)

    if st.session_state.m3_thread_done:
        st.session_state.m3_stage = 'done'
        st.rerun()
    else:
        time.sleep(1.5)
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# ЭТАП 3: Результаты
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.m3_stage == 'done':

    task = st.session_state.m3_task
    task_label = "классификации" if task == "classification" else "детекции"
    st.title(f"Подбор пайплайна ({task_label}) — завершён")

    if st.session_state.m3_error:
        st.error(f"Ошибка:\n```\n{st.session_state.m3_error}\n```")
        with st.expander("Лог выполнения"):
            _render_log_box()
        if st.button("Назад к настройкам", use_container_width=True):
            _reset()
            st.rerun()
    else:
        if st.session_state.m3_result:
            _render_result(st.session_state.m3_result)
        else:
            st.warning("Результат не получен. Поиск мог быть прерван.")

        with st.expander("Лог выполнения"):
            _render_log_box()

        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Новый поиск", use_container_width=True):
                _reset()
                st.rerun()
        with col2:
            if st.button("Перейти к обучению", type='primary', use_container_width=True):
                st.switch_page("pages/3_Обучение.py")
