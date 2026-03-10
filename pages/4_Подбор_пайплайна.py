"""
pages/4_Подбор_пайплайна.py -- Модуль 3: Автоматический подбор пайплайна предобработки

Алгоритм SFS+SHA:
- SFS: Kohavi & John (1997) Artificial Intelligence, 97, 273-324
- SHA: Jamieson & Talwalkar (2016) AISTATS, 240-248
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
    page_title="Подбор пайплайна -- VKR2026",
    page_icon=None,
    layout="wide",
)
init_session_state()
render_sidebar()

if not is_path_configured():
    st.error("Сначала настрой путь к датасетам в разделе **Настройки**.")
    st.stop()

# ── Состояние страницы ────────────────────────────────────────────────────
_KEYS = {
    'm3_stage': 'configure',
    'm3_log_lines': [],
    'm3_output_queue': None,
    'm3_thread_done': False,
    'm3_error': None,
    'm3_result': None,
    'm3_dataset': None,
    'm3_model_type': 'yolo',
    'm3_yolo_size': 'n',
    'm3_epochs': 30,
    'm3_patience': 10,
}
for k, v in _KEYS.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ══════════════════════════════════════════════════════════════════════════════
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ══════════════════════════════════════════════════════════════════════════════

def _log(msg: str):
    st.session_state.m3_log_lines.append(str(msg))


def _reset():
    for k, v in _KEYS.items():
        st.session_state[k] = v


def _render_log_box():
    lines = st.session_state.m3_log_lines
    if lines:
        log_text = '\n'.join(lines[-300:])
        st.code(log_text, language=None)


def _render_result(result_dict: dict):
    """Отображает результаты поиска."""
    best = result_dict.get('best_pipeline', {})
    final_metrics_test = result_dict.get('final_metrics_test', {})
    final_map = result_dict.get('final_map_100pct', 0.0)
    fast_map = result_dict.get('best_map_fast', 0.0)
    stop_reason = result_dict.get('stop_reason', '--')
    iters = result_dict.get('total_iterations', 0)
    history = result_dict.get('history', [])

    st.success("Поиск завершён!")
    st.info("Финальные метрики получены на test split -- независимая оценка.")

    winner_path = result_dict.get('winner_dataset_path')
    if winner_path:
        st.success(f"Датасет-победитель сохранён: `{winner_path}`")

    st.divider()

    # Финальные метрики (все, из test split)
    st.subheader("Финальные метрики (test split)")
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("mAP50-95", f"{final_metrics_test.get('mAP50-95', final_map):.4f}")
    with col2:
        st.metric("mAP50", f"{final_metrics_test.get('mAP50', 0.0):.4f}")
    with col3:
        st.metric("F1", f"{final_metrics_test.get('f1', 0.0):.4f}")
    with col4:
        st.metric("val_loss", f"{final_metrics_test.get('val_loss', 0.0):.4f}")
    with col5:
        st.metric("mAP (поиск SHA)", f"{fast_map:.4f}")

    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("Итераций SFS", iters)
    with col_b:
        st.metric("Причина стопа",
                  stop_reason[:25] + '...' if len(stop_reason) > 25 else stop_reason)

    st.divider()

    # Лучший пайплайн
    st.subheader("Лучший пайплайн")
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        st.markdown(f"**Название:** `{best.get('display_name', '--')}`")
        steps = best.get('steps', [])
        if steps:
            steps_str = ' -> '.join(steps)
            st.markdown(f"**Шаги:** {steps_str}")
        else:
            st.markdown("**Шаги:** Оригинал (без предобработки)")

        methods = best.get('methods', [])
        if methods:
            st.markdown(f"**Методы:** `{', '.join(methods)}`")

    with col_p2:
        params = best.get('params', {})
        if params:
            st.markdown("**Параметры:**")
            st.json(params)
        else:
            st.markdown("**Параметры:** нет (оригинал)")

    # История итераций
    if history:
        st.divider()
        st.subheader("История итераций")
        import pandas as pd

        rows = []
        for item in history:
            rows.append({
                'Итерация': item['iteration'],
                'Кандидатов': item['n_candidates'],
                'Выживших': item['n_survivors'],
                'Лучший': item['best_pipeline'],
                'mAP50-95': round(item['best_map'], 4),
            })

        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)

        try:
            import plotly.graph_objects as go
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=[r['Итерация'] for r in rows],
                y=[r['mAP50-95'] for r in rows],
                mode='lines+markers',
                name='mAP50-95',
                line=dict(color='#2196F3', width=2),
                marker=dict(size=8),
            ))
            fig.update_layout(
                title='Прогресс поиска: лучший mAP по итерациям',
                xaxis_title='Итерация',
                yaxis_title='mAP50-95',
                height=350,
            )
            st.plotly_chart(fig, use_container_width=True)
        except ImportError:
            pass

        with st.expander("Детали по каждой итерации"):
            for item in history:
                st.markdown(f"**Итерация {item['iteration']}**")
                cands = item.get('candidates', [])
                if cands:
                    c_df = pd.DataFrame(cands)
                    c_df.columns = ['Пайплайн', 'mAP50-95']
                    st.dataframe(c_df, use_container_width=True, hide_index=True)

    st.divider()
    with st.expander("Полный JSON результата"):
        st.json(result_dict)


# ══════════════════════════════════════════════════════════════════════════════
# ФОНОВЫЙ ПОИСК
# ══════════════════════════════════════════════════════════════════════════════

def _run_search_thread(
    dataset_path: Path,
    model_config: dict,
    epochs: int,
    patience: int,
    q: queue.Queue,
    datasets_global_path: str,
):
    """Фоновый поток для запуска поиска."""
    import io, os

    class QueueWriter(io.TextIOBase):
        def write(self, text):
            if text.strip():
                q.put(('log', text.rstrip()))
            return len(text)
        def flush(self):
            pass

    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = QueueWriter()
    sys.stderr = QueueWriter()

    try:
        os.environ['DATASETS_GLOBAL_PATH'] = datasets_global_path

        project_root = str(Path(__file__).parent.parent)
        if project_root not in sys.path:
            sys.path.insert(0, project_root)

        from module3_preprocessing_search import run_sfs_sha_search

        def progress_cb(iteration, total, stage, message):
            q.put(('progress', {
                'iteration': iteration,
                'total': total,
                'stage': stage,
                'message': message,
            }))

        result = run_sfs_sha_search(
            source_dataset_path=dataset_path,
            model_config=model_config,
            max_epochs=epochs,
            early_stopping_patience=patience,
            datasets_global_path=Path(datasets_global_path),
            log_fn=lambda msg: q.put(('log', msg)),
            progress_callback=progress_cb,
        )

        result_dict = {
            'best_pipeline': {
                'display_name': result.best_pipeline.display_name,
                'steps': result.best_pipeline.steps,
                'methods': result.best_pipeline.methods,
                'params': result.best_pipeline.params,
            },
            'best_map_fast': round(result.best_map, 4),
            'final_map_100pct': round(result.final_map, 4),
            'final_metrics_test': {k: round(v, 4) for k, v in result.final_metrics_test.items()},
            'stop_reason': result.stop_reason,
            'total_iterations': result.total_iterations,
            'history': result.history,
            'winner_dataset_path': str(result.winner_dataset_path) if result.winner_dataset_path else None,
        }

        q.put(('result', result_dict))
        q.put(('done', 'Поиск завершён успешно.'))

    except Exception as e:
        q.put(('error', f"{type(e).__name__}: {e}\n{traceback.format_exc()}"))
        q.put(('done', 'Ошибка при поиске.'))
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr


# ══════════════════════════════════════════════════════════════════════════════
# ЭТАПЫ UI
        sys.stdout = old_stdout
        sys.stderr = old_stderr


# ══════════════════════════════════════════════════════════════════════════════
# ЭТАПЫ UI
# ══════════════════════════════════════════════════════════════════════════════

# ── ЭТАП 1: Конфигурация ──────────────────────────────────────────────────
if st.session_state.m3_stage == 'configure':

    st.title("🔬 Подбор пайплайна предобработки")
    st.markdown(
        "**Модуль 3** автоматически подбирает оптимальный пайплайн предобработки "
        "с помощью алгоритма **SFS+SHA** -- гибрида последовательного отбора признаков "
        "и метода successive halving."
    )

    with st.expander("ℹ️ Как работает алгоритм", expanded=False):
        st.markdown("""
**Sequential Forward Selection (SFS)** итеративно строит пайплайн,
добавляя по одному методу предобработки.

**Successive Halving (SHA)** на каждом шаге отсевает слабых кандидатов
через быстрое частичное обучение (30% эпох).

**Пул методов (13 кандидатов):**
- Оригинал (baseline)
- Median filter: ksize=3, 5
- Gaussian blur: 3x3, 5x5
- Bilateral filter: sigma=75, sigma=150
- CLAHE: clip=1.0, clip=2.0
- Unsharp mask: amount=0.5, 1.0
- Нормализация: Z-score, Min-Max

*Источники: Kohavi & John (1997); Jamieson & Talwalkar (2016)*
        """)

    st.divider()

    datasets = get_available_datasets()
    if not datasets:
        st.warning("Датасеты не найдены. Проверь путь в Настройках.")
        st.stop()

    col_left, col_right = st.columns(2, gap="large")

    with col_left:
        st.subheader("1. Датасет")
        selected_dataset = st.selectbox(
            "Датасет для поиска",
            options=datasets,
            index=0,
            key='m3_ds_select',
        )

    with col_right:
        st.subheader("2. Proxy-модель")

        model_type = st.selectbox(
            "Тип модели",
            options=['yolo', 'faster_rcnn', 'retinanet'],
            format_func=lambda x: {
                'yolo': 'YOLOv8',
                'faster_rcnn': 'Faster R-CNN',
                'retinanet': 'RetinaNet',
            }[x],
            key='m3_model_select',
        )

        yolo_size = 'n'
        if model_type == 'yolo':
            yolo_size = st.selectbox(
                "Размер YOLOv8",
                options=['n', 's', 'm', 'l', 'x'],
                format_func=lambda x: {
                    'n': 'nano (быстрее всего)',
                    's': 'small',
                    'm': 'medium',
                    'l': 'large',
                    'x': 'xlarge (медленнее всего)',
                }[x],
                index=0,
                key='m3_yolo_size_select',
            )

        st.subheader("3. Параметры обучения")
        epochs = st.number_input(
            "Эпох финального обучения",
            min_value=5,
            max_value=300,
            value=30,
            step=5,
            help="Столько эпох займёт финальное обучение победителя. Для SHA-сравнения кандидатов используется 30% от этого числа.",
            key='m3_epochs_input',
        )

        patience = st.number_input(
            "Early stopping patience (финал)",
            min_value=3,
            max_value=50,
            value=10,
            help="Только для финального обучения победителя. "
                 "Быстрые 30% эпох -- всегда без early stopping.",
            key='m3_patience_input',
        )

        fast_ep = max(1, int(epochs * 0.30))
        st.caption(
            f"Быстрое: {fast_ep} эп. без ES (честное сравнение SHA) | "
            f"Финал: {epochs} эп. + ES (patience={patience})"
        )

        # VRAM info
        try:
            import torch
            if torch.cuda.is_available():
                vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
                st.info(f"GPU доступен: {vram:.1f} GB VRAM")
            else:
                st.warning("GPU не обнаружен -- обучение на CPU (медленно)")
        except Exception:
            pass

    st.divider()

    n_candidates = 13
    n_iters_est = 4
    time_per_candidate_min = {'yolo': 2, 'faster_rcnn': 5, 'retinanet': 4}.get(model_type, 3)
    total_est = n_candidates * n_iters_est * time_per_candidate_min

    st.info(
        f"Примерное время: {total_est}--{total_est*2} мин "
        f"(зависит от датасета и GPU). "
        f"Всего обучений: ~{n_candidates * n_iters_est}"
    )

    if st.button("Запустить поиск пайплайна", type='primary', use_container_width=True):
        st.session_state.m3_dataset = selected_dataset
        st.session_state.m3_model_type = model_type
        st.session_state.m3_yolo_size = yolo_size
        st.session_state.m3_epochs = epochs
        st.session_state.m3_patience = patience
        st.session_state.m3_log_lines = []
        st.session_state.m3_error = None
        st.session_state.m3_result = None
        st.session_state.m3_thread_done = False
        st.session_state.m3_stage = 'running'
        st.rerun()


# ── ЭТАП 2: Запуск и мониторинг ──────────────────────────────────────────
elif st.session_state.m3_stage == 'running':

    st.title("🔬 Поиск пайплайна -- выполняется...")

    dataset_name = st.session_state.m3_dataset
    model_type = st.session_state.m3_model_type
    yolo_size = st.session_state.m3_yolo_size
    epochs = st.session_state.m3_epochs
    patience = st.session_state.m3_patience
    datasets_path = str(get_datasets_path())

    try:
        project_root = str(Path(__file__).parent.parent)
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        from Data.Datasets.dataset_work import get_dataset_path
        dataset_path = get_dataset_path(dataset_name)
    except Exception:
        dataset_path = get_datasets_path() / dataset_name

    model_config = {
        'type': model_type,
        'size': yolo_size,
    }

    if st.session_state.m3_output_queue is None:
        q = queue.Queue()
        st.session_state.m3_output_queue = q

        thread = threading.Thread(
            target=_run_search_thread,
            args=(dataset_path, model_config, epochs, patience, q, datasets_path),
            daemon=True,
        )
        thread.start()

    if st.button("Остановить поиск"):
        st.session_state.m3_stage = 'configure'
        st.session_state.m3_output_queue = None
        st.warning("Поиск прерван пользователем.")
        st.rerun()

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
                        f"Итерация {payload['iteration']}/{payload['total']}: "
                        f"{payload['message']}"
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

    if not st.session_state.m3_thread_done:
        time.sleep(1.5)
        st.rerun()


# ── ЭТАП 3: Результаты ────────────────────────────────────────────────────
elif st.session_state.m3_stage == 'done':

    st.title("🔬 Подбор пайплайна -- завершён")

    if st.session_state.m3_error:
        st.error(f"Ошибка:\n```\n{st.session_state.m3_error}\n```")
        st.info("Проверь лог ниже для деталей.")

        with st.expander("Лог выполнения"):
            _render_log_box()

        if st.button("Назад к настройкам", use_container_width=True):
            _reset()
            st.rerun()

    else:
        if st.session_state.m3_result:
            _render_result(st.session_state.m3_result)
        else:
            st.warning("Результат не получен. Возможно, поиск был прерван.")

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
