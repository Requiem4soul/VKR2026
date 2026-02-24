"""
pages/3_Training.py — Обучение моделей детекции

Выбор датасетов, моделей, настройка гиперпараметров, запуск обучения
с отображением прогресса в реальном времени, финальные метрики.
"""

import sys
import time
import threading
import queue
import json
import streamlit as st
import pandas as pd
from pathlib import Path

from ui.sidebar import render_sidebar
from ui.state import (
    init_session_state,
    is_path_configured,
    get_available_datasets,
    get_datasets_path,
)

st.set_page_config(page_title="Обучение — VKR2026", page_icon="🚀", layout="wide")
init_session_state()

render_sidebar()

if not is_path_configured():
    st.error("⚠️ Сначала настрой путь к датасетам в разделе **⚙️ Настройки**.")
    st.stop()

st.title("🚀 Обучение моделей детекции")
st.markdown(
    "Выбери датасеты, модели и настрой гиперпараметры. "
    "Значения по умолчанию подобраны автоматически с учётом доступной VRAM."
)
st.divider()

# ── Состояние страницы ─────────────────────────────────────────────────────
if "train_stage" not in st.session_state:
    st.session_state.train_stage = "configure"

if "train_log_lines" not in st.session_state:
    st.session_state.train_log_lines = []

if "train_output_queue" not in st.session_state:
    st.session_state.train_output_queue = None

if "train_thread_done" not in st.session_state:
    st.session_state.train_thread_done = False

if "train_error" not in st.session_state:
    st.session_state.train_error = None

if "train_metrics_data" not in st.session_state:
    st.session_state.train_metrics_data = {}

if "train_results_dir" not in st.session_state:
    st.session_state.train_results_dir = None

# ── Определяем VRAM заранее ────────────────────────────────────────────────
@st.cache_resource
def get_vram_gb():
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_memory / 1024**3
    except Exception:
        pass
    return 0.0

vram_gb = get_vram_gb()

def suggest_batch(model_type: str, vram: float) -> int:
    """Подсказывает batch_size на основе VRAM."""
    if vram <= 0:
        return 1
    if 14 <= vram <= 18:
        return {"yolo": 16, "faster_rcnn": 8, "retinanet": 12}.get(model_type, 8)
    scale = vram / 8.0
    base = {"yolo": 16, "faster_rcnn": 4, "retinanet": 8}.get(model_type, 4)
    return max(1, int(base * scale))

# ══════════════════════════════════════════════════════════════════════════════
# ЭТАП 1: Конфигурация
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.train_stage == "configure":

    datasets = get_available_datasets()
    if not datasets:
        st.warning("Датасеты не найдены. Добавь датасеты или проверь путь в Настройках.")
        st.stop()

    # ── Выбор датасетов ────────────────────────────────────────────────────
    st.subheader("Шаг 1: Выберите датасеты для обучения")
    selected_datasets = st.multiselect(
        "Датасеты",
        options=datasets,
        help="Можно выбрать несколько. Каждая модель будет обучена на каждом датасете.",
    )

    st.divider()

    # ── Выбор моделей ──────────────────────────────────────────────────────
    st.subheader("Шаг 2: Выберите модели для обучения")
    st.markdown("Отметь модели которые хочешь обучить:")

    col1, col2, col3 = st.columns(3)
    with col1:
        use_yolo = st.checkbox("🟦 YOLOv8", value=True, help="Быстрая модель одноэтапной детекции")
    with col2:
        use_frcnn = st.checkbox("🟧 Faster R-CNN", value=False, help="Двухэтапная детекция, высокая точность")
    with col3:
        use_retina = st.checkbox("🟩 RetinaNet", value=False, help="Одноэтапная детекция с focal loss")

    # ── Гиперпараметры ─────────────────────────────────────────────────────
    st.divider()
    st.subheader("Шаг 3: Гиперпараметры")

    if vram_gb > 0:
        st.caption(f"🖥️ Обнаружена VRAM: **{vram_gb:.1f} GB** — значения batch_size подобраны автоматически.")
    else:
        st.caption("⚠️ GPU не обнаружен, обучение будет на CPU.")

    model_configs = []

    # ─ YOLOv8 ────────────────────────────────────────────────────────────
    if use_yolo:
        with st.expander("🟦 Настройки YOLOv8", expanded=True):
            col_a, col_b = st.columns(2)
            with col_a:
                yolo_size = st.selectbox(
                    "Размер модели",
                    ["n", "s", "m", "l", "x"],
                    index=1,
                    format_func=lambda x: {
                        "n": "nano (fastest)", "s": "small", "m": "medium",
                        "l": "large", "x": "xlarge (best)"
                    }[x],
                    key="yolo_size"
                )
                yolo_epochs = st.number_input("Макс. эпох", min_value=1, max_value=500, value=80, key="yolo_epochs")
            with col_b:
                yolo_batch = st.number_input(
                    "Batch size (-1 = авто)",
                    min_value=-1, max_value=256,
                    value=suggest_batch("yolo", vram_gb),
                    key="yolo_batch"
                )
            st.caption("ℹ️ Размер изображений определяется автоматически из датасета.")

            st.markdown("**Early Stopping:**")
            col_c, col_d = st.columns(2)
            with col_c:
                yolo_es = st.checkbox("Включить Early Stopping", value=True, key="yolo_es")
            with col_d:
                yolo_patience = st.number_input(
                    "Patience (эпох без улучшения)",
                    min_value=1, max_value=100, value=15,
                    disabled=not yolo_es,
                    key="yolo_patience"
                )

            model_configs.append({
                "type": "yolo",
                "size": yolo_size,
                "name": f"yolo_{yolo_size}",
                "max_epochs": yolo_epochs,
                "batch": yolo_batch,
                "early_stopping": {"patience": yolo_patience, "metric": "mAP50-95"} if yolo_es else None,
            })

    # ─ Faster R-CNN ───────────────────────────────────────────────────────
    if use_frcnn:
        with st.expander("🟧 Настройки Faster R-CNN", expanded=True):
            col_a, col_b = st.columns(2)
            with col_a:
                frcnn_epochs = st.number_input("Макс. эпох", min_value=1, max_value=200, value=25, key="frcnn_epochs")
                frcnn_pretrained = st.checkbox("Использовать предобученные веса", value=True, key="frcnn_pre")
            with col_b:
                frcnn_batch = st.number_input(
                    "Batch size",
                    min_value=1, max_value=64,
                    value=suggest_batch("faster_rcnn", vram_gb),
                    key="frcnn_batch"
                )

            col_c, col_d = st.columns(2)
            with col_c:
                frcnn_es = st.checkbox("Включить Early Stopping", value=True, key="frcnn_es")
            with col_d:
                frcnn_patience = st.number_input(
                    "Patience",
                    min_value=1, max_value=50, value=7,
                    disabled=not frcnn_es,
                    key="frcnn_patience"
                )

            model_configs.append({
                "type": "faster_rcnn",
                "pretrained": frcnn_pretrained,
                "name": "faster_rcnn",
                "max_epochs": frcnn_epochs,
                "batch": frcnn_batch,
                "early_stopping": {"patience": frcnn_patience, "metric": "mAP50-95"} if frcnn_es else None,
            })

    # ─ RetinaNet ──────────────────────────────────────────────────────────
    if use_retina:
        with st.expander("🟩 Настройки RetinaNet", expanded=True):
            col_a, col_b = st.columns(2)
            with col_a:
                ret_epochs = st.number_input("Макс. эпох", min_value=1, max_value=200, value=35, key="ret_epochs")
                ret_pretrained = st.checkbox("Использовать предобученные веса", value=True, key="ret_pre")
            with col_b:
                ret_batch = st.number_input(
                    "Batch size",
                    min_value=1, max_value=64,
                    value=suggest_batch("retinanet", vram_gb),
                    key="ret_batch"
                )

            col_c, col_d = st.columns(2)
            with col_c:
                ret_es = st.checkbox("Включить Early Stopping", value=True, key="ret_es")
            with col_d:
                ret_patience = st.number_input(
                    "Patience",
                    min_value=1, max_value=50, value=10,
                    disabled=not ret_es,
                    key="ret_patience"
                )

            model_configs.append({
                "type": "retinanet",
                "pretrained": ret_pretrained,
                "name": "retinanet",
                "max_epochs": ret_epochs,
                "batch": ret_batch,
                "early_stopping": {"patience": ret_patience, "metric": "mAP50-95"} if ret_es else None,
            })

    # ── Ранний отбор моделей ───────────────────────────────────────────────
    st.divider()
    st.subheader("Шаг 4: Дополнительные параметры")

    with st.expander("💾 Чекпоинты", expanded=True):
        st.markdown("Сохранение весов модели каждые N эпох.")
        checkpoint_interval = st.number_input(
            "Интервал чекпоинтов (эпох)",
            min_value=1, max_value=50, value=5,
            help="Чекпоинт сохраняется каждые N эпох. Рекомендуется значение кратное max_epochs каждой модели."
        )
        # Предупреждение если интервал больше минимального max_epochs
        if model_configs:
            min_epochs = min(m.get("max_epochs", 40) for m in model_configs)
            if checkpoint_interval > min_epochs:
                st.warning(
                    f"⚠️ Интервал {checkpoint_interval} больше чем max_epochs одной из моделей ({min_epochs}). "
                    f"Для неё чекпоинт сохранится только в конце."
                )

    with st.expander("⚡ Ранний отбор моделей", expanded=True):
        st.markdown(
            "После первых N% эпох система предсказывает итоговое качество каждой модели. "
            "Слабые модели останавливаются раньше, экономя время. *(Jamieson & Talwalkar, 2016)*"
        )
        col_a, col_b = st.columns(2)
        with col_a:
            enable_selection = st.checkbox("Включить ранний отбор", value=True)
        with col_b:
            selection_ratio = st.slider(
                "Отбор после % эпох", 10, 50, 30,
                disabled=not enable_selection
            )
            top_k = st.slider(
                "Оставить % лучших", 25, 75, 50,
                disabled=not enable_selection
            )

    st.divider()

    # ── Итоговый список комбинаций ─────────────────────────────────────────
    if selected_datasets and model_configs:
        total_runs = len(selected_datasets) * len(model_configs)
        st.info(
            f"**Будет выполнено:** {total_runs} комбинаций "
            f"({len(model_configs)} модел{'ь' if len(model_configs)==1 else 'и' if len(model_configs)<5 else 'ей'} × "
            f"{len(selected_datasets)} датасет{'а' if len(selected_datasets)<5 else 'ов'})"
        )

    # ── Кнопка запуска ────────────────────────────────────────────────────
    launch_disabled = not selected_datasets or not model_configs
    if launch_disabled:
        st.warning("Выбери хотя бы один датасет и хотя бы одну модель.")

    if st.button(
        "🚀 Начать обучение",
        type="primary",
        use_container_width=True,
        disabled=launch_disabled,
    ):
        st.session_state.train_selected_datasets = selected_datasets
        st.session_state.train_model_configs_data = model_configs
        st.session_state.train_enable_selection = enable_selection
        st.session_state.train_selection_ratio = selection_ratio / 100
        st.session_state.train_top_k = top_k / 100
        st.session_state.train_checkpoint_interval = checkpoint_interval
        st.session_state.train_stage = "running"
        st.session_state.train_log_lines = []
        st.session_state.train_output_queue = None
        st.session_state.train_thread_done = False
        st.session_state.train_error = None
        st.session_state.train_metrics_data = {}
        st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# ЭТАП 2: Обучение с выводом лога
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.train_stage == "running":

    datasets_sel = st.session_state.train_selected_datasets
    model_cfgs = st.session_state.train_model_configs_data

    st.subheader("⏳ Выполняется обучение...")
    st.info(
        f"**Датасеты:** {', '.join(datasets_sel)}  \n"
        f"**Модели:** {', '.join(m['name'] for m in model_cfgs)}"
    )

    log_placeholder = st.empty()
    progress_placeholder = st.empty()

    # Запускаем обучение в фоновом потоке
    if st.session_state.train_output_queue is None:
        q = queue.Queue()
        st.session_state.train_output_queue = q

        def run_training(q, datasets, model_configs, datasets_path,
                         enable_selection, selection_ratio, top_k, checkpoint_interval):
            import io, os

            class QueueWriter(io.TextIOBase):
                def write(self, text):
                    if text.strip():
                        q.put(("log", text.rstrip()))
                    return len(text)
                def flush(self):
                    pass

            old_stdout = sys.stdout
            sys.stdout = QueueWriter()

            try:
                project_root = str(Path(__file__).parent.parent)
                if project_root not in sys.path:
                    sys.path.insert(0, project_root)

                os.environ["DATASETS_GLOBAL_PATH"] = str(datasets_path)

                from Train.Universal_train.universal_model_trainer import UniversalModelTrainer

                # Извлекаем параметры early stopping из конфигов
                global_patience = 10
                global_metric = 'mAP50-95'
                enable_es = any(m.get("early_stopping") for m in model_configs)

                if enable_es:
                    for mc in model_configs:
                        if mc.get("early_stopping"):
                            global_patience = mc["early_stopping"].get("patience", 10)
                            global_metric = mc["early_stopping"].get("metric", "mAP50-95")
                            break

                trainer = UniversalModelTrainer(
                    model_configs=model_configs,
                    dataset_names=datasets,
                    max_epochs=max(m.get("max_epochs", 40) for m in model_configs),
                    checkpoint_interval=checkpoint_interval,
                    enable_early_stopping=enable_es,
                    early_stopping_patience=global_patience,
                    early_stopping_metric=global_metric,
                    enable_early_selection=enable_selection,
                    early_selection_ratio=selection_ratio,
                    early_selection_top_k=top_k,
                    clean_old_results=False,
                )

                trainer.run_training()

                # Отправляем путь к результатам
                q.put(("results_dir", trainer.results_dir))
                q.put(("metrics", json.dumps(trainer.metrics_history)))
                q.put(("done", "✅ Обучение завершено!"))

            except Exception as e:
                import traceback
                q.put(("error", f"{e}\n{traceback.format_exc()}"))
            finally:
                sys.stdout = old_stdout

        thread = threading.Thread(
            target=run_training,
            args=(
                q,
                datasets_sel,
                model_cfgs,
                get_datasets_path(),
                st.session_state.train_enable_selection,
                st.session_state.train_selection_ratio,
                st.session_state.train_top_k,
                st.session_state.get("train_checkpoint_interval", 5),
            ),
            daemon=True,
        )
        thread.start()

    # Читаем очередь
    q = st.session_state.train_output_queue
    if q is not None:
        try:
            while True:
                msg_type, msg = q.get_nowait()
                if msg_type == "log":
                    st.session_state.train_log_lines.append(msg)
                elif msg_type == "results_dir":
                    st.session_state.train_results_dir = msg
                elif msg_type == "metrics":
                    st.session_state.train_metrics_data = json.loads(msg)
                elif msg_type == "done":
                    st.session_state.train_log_lines.append(msg)
                    st.session_state.train_thread_done = True
                    st.session_state.train_stage = "done"
                    st.session_state.train_output_queue = None
                    break
                elif msg_type == "error":
                    st.session_state.train_error = msg
                    st.session_state.train_log_lines.append(f"❌ {msg}")
                    st.session_state.train_thread_done = True
                    st.session_state.train_stage = "done"
                    st.session_state.train_output_queue = None
                    break
        except queue.Empty:
            pass

    with log_placeholder.container():
        lines = st.session_state.train_log_lines
        log_text = "\n".join(lines[-300:]) if lines else "Ожидание вывода..."
        st.markdown("**Лог обучения:**")
        st.code(log_text, language=None)

    if not st.session_state.train_thread_done:
        with progress_placeholder:
            st.info("🔄 Обучение выполняется... Страница обновляется автоматически.")
        time.sleep(2.0)
        st.rerun()
    else:
        st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# ЭТАП 3: Результаты
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.train_stage == "done":

    if st.session_state.train_error:
        st.error("❌ Во время обучения возникла ошибка:")
        st.code(st.session_state.train_error)
    else:
        st.success("✅ Обучение завершено успешно!")

        # ── Финальные метрики в таблице ────────────────────────────────────
        st.subheader("📊 Финальные метрики")

        st.warning(
            "⚠️ **Важное замечание:** `train_loss` и `val_loss` вычисляются по-разному "
            "для YOLO, Faster R-CNN и RetinaNet, поэтому **сравнивать модели по loss некорректно**. "
            "Для объективного сравнения используй **mAP50** и **mAP50-95**."
        )

        metrics_data = st.session_state.train_metrics_data
        if metrics_data:
            # Строим сводную таблицу последних метрик каждой комбинации
            rows = []
            for key, history in metrics_data.items():
                if not history:
                    continue
                last = history[-1] if isinstance(history, list) else {}
                if isinstance(last, dict):
                    row = {"Модель / Датасет": key}
                    for metric in ["mAP50", "mAP50-95", "precision", "recall", "f1"]:
                        row[metric] = f"{last.get(metric, 0):.4f}" if metric in last else "—"
                    rows.append(row)

            if rows:
                df = pd.DataFrame(rows).set_index("Модель / Датасет")
                st.dataframe(df, use_container_width=True)
            else:
                st.info("Метрики ещё не записаны (возможно, обучение завершилось слишком быстро).")

        # ── Полный лог ─────────────────────────────────────────────────────
        with st.expander("📋 Полный лог обучения", expanded=False):
            full_log = "\n".join(st.session_state.train_log_lines)
            st.code(full_log, language=None)

        # ── Сохранение логов ───────────────────────────────────────────────
        st.subheader("💾 Сохранение результатов")

        full_log_text = "\n".join(st.session_state.train_log_lines)
        st.download_button(
            label="💾 Скачать лог обучения (.txt)",
            data=full_log_text,
            file_name="training_log.txt",
            mime="text/plain",
        )

        if st.session_state.train_metrics_data:
            metrics_json = json.dumps(st.session_state.train_metrics_data, indent=2, ensure_ascii=False)
            st.download_button(
                label="📈 Скачать метрики (.json)",
                data=metrics_json,
                file_name="metrics.json",
                mime="application/json",
            )

        if st.session_state.train_results_dir:
            st.info(f"📁 Папка с результатами: `{st.session_state.train_results_dir}`\n\n"
                    f"Там находятся чекпоинты моделей и полные логи.")

    st.divider()
    st.subheader("Что дальше?")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🧪 Подобрать предобработку", use_container_width=True):
            st.switch_page("pages/2_Preprocessing.py")
    with col2:
        if st.button("🔁 Обучить модели ещё раз", use_container_width=True):
            st.session_state.train_stage = "configure"
            st.session_state.train_log_lines = []
            st.session_state.train_output_queue = None
            st.session_state.train_error = None
            st.session_state.train_metrics_data = {}
            st.rerun()
