"""
pages/3_Обучение.py — Обучение моделей (детекция и классификация)

Поддерживает две задачи:
- Детекция: YOLOv8, Faster R-CNN, RetinaNet
- Классификация: ResNet-18, ResNet-50, EfficientNet-B0
  (Yang et al., 2021 MedMNIST; He et al., 2016; Tan & Le, 2019)
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

st.set_page_config(page_title="Обучение — VKR2026", page_icon=None, layout="wide")
init_session_state()
render_sidebar()

if not is_path_configured():
    st.error("⚠️ Сначала настрой путь к датасетам в разделе **⚙️ Настройки**.")
    st.stop()

# ── Состояние страницы ─────────────────────────────────────────────────────
_state_defaults = {
    "train_stage": "configure",
    "train_task": "detection",          # "detection" | "classification"
    "train_log_lines": [],
    "train_output_queue": None,
    "train_thread_done": False,
    "train_error": None,
    "train_metrics_data": {},
    "train_results_dir": None,
    # Сохраняем параметры для фонового потока
    "train_selected_datasets": [],
    "train_model_configs_data": [],
    "train_enable_selection": True,
    "train_selection_ratio": 0.3,
    "train_top_k": 0.5,
    "train_checkpoint_interval": 10,
    "train_seed": 42,
}
for k, v in _state_defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ── VRAM ──────────────────────────────────────────────────────────────────
@st.cache_resource
def get_vram_gb():
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
    except Exception:
        pass
    return 0.0


vram_gb = get_vram_gb()


def suggest_batch_detection(model_type: str, vram: float) -> int:
    if vram <= 0:
        return 1
    if 14 <= vram <= 18:
        return {"yolo": 16, "faster_rcnn": 8, "retinanet": 12}.get(model_type, 8)
    scale = vram / 8.0
    base = {"yolo": 16, "faster_rcnn": 4, "retinanet": 8}.get(model_type, 4)
    return max(1, int(base * scale))


def suggest_batch_classification(model_type: str, vram: float, image_size: int = 224) -> int:
    """Подсказывает batch_size для классификации на основе VRAM."""
    if vram <= 0:
        return 16
    base = {"resnet18": 128, "resnet50": 64, "efficientnet_b0": 96}.get(model_type, 64)
    size_scale = (224 / max(image_size, 1)) ** 2
    vram_scale = vram / 8.0
    return max(1, int(base * size_scale * vram_scale * 0.75))


# ══════════════════════════════════════════════════════════════════════════════
# ФОНОВЫЕ ФУНКЦИИ ОБУЧЕНИЯ
# ══════════════════════════════════════════════════════════════════════════════

def _run_detection_thread(q, datasets, model_configs, datasets_path,
                          enable_selection, selection_ratio, top_k,
                          checkpoint_interval, seed):
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

        # Фиксируем seed для воспроизводимости
        import random, numpy as np, torch
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["PYTHONHASHSEED"] = str(seed)

        from Train.Universal_train.universal_model_trainer import UniversalModelTrainer

        global_patience = 10
        global_metric = "mAP50-95"
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

        q.put(("results_dir", trainer.results_dir))
        q.put(("metrics", json.dumps(trainer.metrics_history)))
        q.put(("done", "Обучение завершено!"))

    except Exception as e:
        import traceback
        q.put(("error", f"{type(e).__name__}: {e}\n{traceback.format_exc()}"))
        q.put(("done", "Ошибка!"))
    finally:
        sys.stdout = old_stdout


def _run_classification_thread(q, datasets, model_configs, datasets_path,
                                checkpoint_interval, seed,
                                enable_selection, selection_ratio, top_k):
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

        from Train.Classification.classification_trainer import ClassificationTrainer

        global_patience = 15
        global_metric = "val_auc"
        enable_es = any(m.get("early_stopping") for m in model_configs)
        if enable_es:
            for mc in model_configs:
                if mc.get("early_stopping"):
                    global_patience = mc["early_stopping"].get("patience", 15)
                    global_metric = mc["early_stopping"].get("metric", "val_auc")
                    break

        trainer = ClassificationTrainer(
            model_configs=model_configs,
            dataset_names=datasets,
            max_epochs=max(m.get("max_epochs", 100) for m in model_configs),
            checkpoint_interval=checkpoint_interval,
            seed=seed,
            enable_early_stopping=enable_es,
            early_stopping_patience=global_patience,
            early_stopping_metric=global_metric,
            enable_early_selection=enable_selection,
            early_selection_ratio=selection_ratio,
            early_selection_top_k=top_k,
            clean_old_results=False,
        )
        trainer.run_training()

        q.put(("results_dir", trainer.results_dir))
        q.put(("metrics", json.dumps(trainer.metrics_history)))
        q.put(("done", "Обучение завершено!"))

    except Exception as e:
        import traceback
        q.put(("error", f"{type(e).__name__}: {e}\n{traceback.format_exc()}"))
        q.put(("done", "Ошибка!"))
    finally:
        sys.stdout = old_stdout


# ══════════════════════════════════════════════════════════════════════════════
# ЭТАП 1: Конфигурация
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.train_stage == "configure":

    st.title("Обучение моделей")
    st.markdown(
        "Выбери задачу, датасеты, модели и настрой гиперпараметры. "
        "Значения по умолчанию подобраны автоматически с учётом доступной VRAM."
    )

    if vram_gb > 0:
        st.caption(f"🖥️ Обнаружена VRAM: **{vram_gb:.1f} GB**")
    else:
        st.caption("⚠️ GPU не обнаружен — обучение на CPU.")

    st.divider()

    # ── Выбор задачи ──────────────────────────────────────────────────────
    st.subheader("Шаг 0: Тип задачи")
    task = st.radio(
        "Выберите задачу",
        options=["detection", "classification"],
        format_func=lambda x: "🔍 Детекция объектов (YOLOv8, Faster R-CNN, RetinaNet)"
                               if x == "detection"
                               else "🏷️ Классификация изображений (ResNet-18, ResNet-50, EfficientNet-B0)",
        index=0 if st.session_state.train_task == "detection" else 1,
        key="task_radio",
        horizontal=True,
    )
    st.session_state.train_task = task
    st.divider()

    datasets = get_available_datasets()
    if not datasets:
        st.warning("Датасеты не найдены. Добавь датасеты или проверь путь в Настройках.")
        st.stop()

    selected_datasets = []
    model_configs = []
    enable_selection = True
    selection_ratio = 30
    top_k = 50
    checkpoint_interval = 10
    seed = 42

    with st.expander("Настройки", expanded=True):

        # ── Выбор датасетов ──────────────────────────────────────────────
        st.subheader("Шаг 1: Выберите датасеты")
        selected_datasets = st.multiselect(
            "Датасеты",
            options=datasets,
            help="Каждая модель будет обучена на каждом датасете.",
        )
        st.divider()

        # ══════════════════════════════════════════════════════════════════
        # ВЕТКА: ДЕТЕКЦИЯ
        # ══════════════════════════════════════════════════════════════════
        if task == "detection":
            st.subheader("Шаг 2: Модели детекции")
            col1, col2, col3 = st.columns(3)
            with col1:
                use_yolo = st.checkbox("🟦 YOLOv8", value=True)
            with col2:
                use_frcnn = st.checkbox("🟧 Faster R-CNN", value=False)
            with col3:
                use_retina = st.checkbox("🟩 RetinaNet", value=False)

            st.divider()
            st.subheader("Шаг 3: Гиперпараметры")

            if use_yolo:
                with st.expander("🟦 YOLOv8", expanded=True):
                    col_a, col_b = st.columns(2)
                    with col_a:
                        yolo_size = st.selectbox(
                            "Размер модели",
                            ["n", "s", "m", "l", "x"],
                            index=1,
                            format_func=lambda x: {"n": "nano", "s": "small", "m": "medium",
                                                    "l": "large", "x": "xlarge"}[x],
                            key="yolo_size",
                        )
                        yolo_epochs = st.number_input("Макс. эпох", 1, 500, 80, key="yolo_epochs")
                    with col_b:
                        yolo_batch = st.number_input(
                            "Batch size (-1 = авто)", -1, 256,
                            value=suggest_batch_detection("yolo", vram_gb), key="yolo_batch",
                        )
                    col_c, col_d = st.columns(2)
                    with col_c:
                        yolo_es = st.checkbox("Early Stopping", value=True, key="yolo_es")
                    with col_d:
                        yolo_patience = st.number_input("Patience", 1, 100, 15,
                                                         disabled=not yolo_es, key="yolo_patience")
                    model_configs.append({
                        "type": "yolo", "size": yolo_size, "name": f"yolo_{yolo_size}",
                        "max_epochs": yolo_epochs, "batch": yolo_batch,
                        "early_stopping": {"patience": yolo_patience, "metric": "mAP50-95"} if yolo_es else None,
                    })

            if use_frcnn:
                with st.expander("🟧 Faster R-CNN", expanded=True):
                    col_a, col_b = st.columns(2)
                    with col_a:
                        frcnn_epochs = st.number_input("Макс. эпох", 1, 200, 25, key="frcnn_epochs")
                        frcnn_pretrained = st.checkbox("Pretrained веса", value=True, key="frcnn_pre")
                    with col_b:
                        frcnn_batch = st.number_input(
                            "Batch size", 1, 64,
                            value=suggest_batch_detection("faster_rcnn", vram_gb), key="frcnn_batch",
                        )
                    col_c, col_d = st.columns(2)
                    with col_c:
                        frcnn_es = st.checkbox("Early Stopping", value=True, key="frcnn_es")
                    with col_d:
                        frcnn_patience = st.number_input("Patience", 1, 50, 10,
                                                          disabled=not frcnn_es, key="frcnn_patience")
                    model_configs.append({
                        "type": "faster_rcnn", "pretrained": frcnn_pretrained,
                        "name": "faster_rcnn", "max_epochs": frcnn_epochs, "batch": frcnn_batch,
                        "early_stopping": {"patience": frcnn_patience, "metric": "mAP50-95"} if frcnn_es else None,
                    })

            if use_retina:
                with st.expander("🟩 RetinaNet", expanded=True):
                    col_a, col_b = st.columns(2)
                    with col_a:
                        ret_epochs = st.number_input("Макс. эпох", 1, 200, 35, key="ret_epochs")
                        ret_pretrained = st.checkbox("Pretrained веса", value=True, key="ret_pre")
                    with col_b:
                        ret_batch = st.number_input(
                            "Batch size", 1, 64,
                            value=suggest_batch_detection("retinanet", vram_gb), key="ret_batch",
                        )
                    col_c, col_d = st.columns(2)
                    with col_c:
                        ret_es = st.checkbox("Early Stopping", value=True, key="ret_es")
                    with col_d:
                        ret_patience = st.number_input("Patience", 1, 50, 10,
                                                        disabled=not ret_es, key="ret_patience")
                    model_configs.append({
                        "type": "retinanet", "pretrained": ret_pretrained,
                        "name": "retinanet", "max_epochs": ret_epochs, "batch": ret_batch,
                        "early_stopping": {"patience": ret_patience, "metric": "mAP50-95"} if ret_es else None,
                    })

        # ══════════════════════════════════════════════════════════════════
        # ВЕТКА: КЛАССИФИКАЦИЯ
        # ══════════════════════════════════════════════════════════════════
        else:
            st.subheader("Шаг 2: Модели классификации")
            st.caption(
                "Архитектуры из Yang et al. (2021) MedMNIST + EfficientNet-B0 (Tan & Le, 2019). "
                "Гиперпараметры по умолчанию взяты из статьи: 100 эпох, SGD lr=1e-3."
            )

            col1, col2, col3 = st.columns(3)
            with col1:
                use_rn18 = st.checkbox("🔵 ResNet-18", value=True,
                                        help="He et al. (2016), CVPR. ~11M параметров.")
            with col2:
                use_rn50 = st.checkbox("🟠 ResNet-50", value=False,
                                        help="He et al. (2016), CVPR. ~25M параметров.")
            with col3:
                use_eff = st.checkbox("🟢 EfficientNet-B0", value=False,
                                       help="Tan & Le (2019), ICML. ~5M параметров.")

            st.divider()
            st.subheader("Шаг 3: Гиперпараметры")

            if use_rn18:
                with st.expander("🔵 ResNet-18", expanded=True):
                    col_a, col_b = st.columns(2)
                    with col_a:
                        rn18_imgsz = st.selectbox(
                            "Размер изображений", [28, 224], index=1,
                            help="28 = как в оригинальной статье MedMNIST; 224 = стандарт ImageNet",
                            key="rn18_imgsz",
                        )
                        rn18_epochs = st.number_input(
                            "Макс. эпох", 1, 300, 100,
                            help="100 эпох — значение из Yang et al. (2021)",
                            key="rn18_epochs",
                        )
                        rn18_pre = st.checkbox("Pretrained (ImageNet)", value=True, key="rn18_pre")
                    with col_b:
                        rn18_batch = st.number_input(
                            "Batch size", 1, 512,
                            value=suggest_batch_classification("resnet18", vram_gb, rn18_imgsz),
                            help="128 — значение из Yang et al. (2021), масштабируется под VRAM",
                            key="rn18_batch",
                        )
                    col_c, col_d = st.columns(2)
                    with col_c:
                        rn18_es = st.checkbox("Early Stopping", value=True, key="rn18_es")
                    with col_d:
                        rn18_patience = st.number_input("Patience", 1, 100, 15,
                                                          disabled=not rn18_es, key="rn18_patience")
                    model_configs.append({
                        "type": "resnet18", "name": f"resnet18_{rn18_imgsz}",
                        "image_size": rn18_imgsz, "max_epochs": rn18_epochs,
                        "pretrained": rn18_pre, "batch": rn18_batch,
                        "early_stopping": {"patience": rn18_patience, "metric": "val_auc"} if rn18_es else None,
                    })

            if use_rn50:
                with st.expander("🟠 ResNet-50", expanded=True):
                    col_a, col_b = st.columns(2)
                    with col_a:
                        rn50_imgsz = st.selectbox(
                            "Размер изображений", [28, 224], index=1, key="rn50_imgsz",
                        )
                        rn50_epochs = st.number_input("Макс. эпох", 1, 300, 100, key="rn50_epochs")
                        rn50_pre = st.checkbox("Pretrained (ImageNet)", value=True, key="rn50_pre")
                    with col_b:
                        rn50_batch = st.number_input(
                            "Batch size", 1, 512,
                            value=suggest_batch_classification("resnet50", vram_gb),
                            key="rn50_batch",
                        )
                    col_c, col_d = st.columns(2)
                    with col_c:
                        rn50_es = st.checkbox("Early Stopping", value=True, key="rn50_es")
                    with col_d:
                        rn50_patience = st.number_input("Patience", 1, 100, 15,
                                                          disabled=not rn50_es, key="rn50_patience")
                    model_configs.append({
                        "type": "resnet50", "name": f"resnet50_{rn50_imgsz}",
                        "image_size": rn50_imgsz, "max_epochs": rn50_epochs,
                        "pretrained": rn50_pre, "batch": rn50_batch,
                        "early_stopping": {"patience": rn50_patience, "metric": "val_auc"} if rn50_es else None,
                    })

            if use_eff:
                with st.expander("🟢 EfficientNet-B0", expanded=True):
                    col_a, col_b = st.columns(2)
                    with col_a:
                        eff_imgsz = st.selectbox(
                            "Размер изображений", [224, 240], index=0,
                            help="240 — нативный размер EfficientNet-B0",
                            key="eff_imgsz",
                        )
                        eff_epochs = st.number_input("Макс. эпох", 1, 300, 100, key="eff_epochs")
                        eff_pre = st.checkbox("Pretrained (ImageNet)", value=True, key="eff_pre")
                    with col_b:
                        eff_batch = st.number_input(
                            "Batch size", 1, 512,
                            value=suggest_batch_classification("efficientnet_b0", vram_gb),
                            key="eff_batch",
                        )
                    col_c, col_d = st.columns(2)
                    with col_c:
                        eff_es = st.checkbox("Early Stopping", value=True, key="eff_es")
                    with col_d:
                        eff_patience = st.number_input("Patience", 1, 100, 15,
                                                         disabled=not eff_es, key="eff_patience")
                    model_configs.append({
                        "type": "efficientnet_b0", "name": f"efficientnet_b0_{eff_imgsz}",
                        "image_size": eff_imgsz, "max_epochs": eff_epochs,
                        "pretrained": eff_pre, "batch": eff_batch,
                        "early_stopping": {"patience": eff_patience, "metric": "val_auc"} if eff_es else None,
                    })

        # ── Дополнительные параметры (общие для обеих задач) ─────────────
        st.divider()
        st.subheader("Шаг 4: Дополнительные параметры")

        with st.expander("Чекпоинты", expanded=True):
            checkpoint_interval = st.number_input(
                "Интервал чекпоинтов (эпох)", 1, 50,
                value=10 if task == "classification" else 5,
                help="Сохранение весов и очистка памяти GPU каждые N эпох.",
            )
            if model_configs:
                min_ep = min(m.get("max_epochs", 40) for m in model_configs)
                if checkpoint_interval > min_ep:
                    st.warning(
                        f"⚠️ Интервал {checkpoint_interval} > max_epochs одной модели ({min_ep}). "
                        "Чекпоинт сохранится только в конце."
                    )

        with st.expander("🎲 Воспроизводимость (Seed)", expanded=True):
            st.markdown(
                "Фиксация seed гарантирует одинаковые результаты при повторных запусках. "
                "Передай значение другому пользователю — он получит идентичные метрики "
                "на том же оборудовании."
            )
            seed = st.number_input(
                "Random seed", min_value=0, max_value=2 ** 31 - 1, value=42,
                help="Фиксирует: torch, numpy, random, cudnn.deterministic=True",
            )

        with st.expander("⚡ Ранний отбор моделей", expanded=True):
            st.markdown(
                "После первых N% эпох слабые модели останавливаются раньше. "
                "*(Jamieson & Talwalkar, 2016)*"
            )
            col_a, col_b = st.columns(2)
            with col_a:
                enable_selection = st.checkbox("Включить ранний отбор", value=False)
            with col_b:
                selection_ratio = st.slider("Отбор после % эпох", 10, 50, 30,
                                             disabled=not enable_selection)
                top_k = st.slider("Оставить % лучших", 25, 75, 50,
                                   disabled=not enable_selection)

    st.divider()

    if selected_datasets and model_configs:
        total_runs = len(selected_datasets) * len(model_configs)
        st.info(
            f"**Будет выполнено:** {total_runs} комбинаций "
            f"({len(model_configs)} мод. × {len(selected_datasets)} датасет.) | "
            f"Seed: {seed}"
        )

    launch_disabled = not selected_datasets or not model_configs
    if launch_disabled:
        st.warning("Выбери хотя бы один датасет и хотя бы одну модель.")

    if st.button("Начать обучение", type="primary", use_container_width=True,
                  disabled=launch_disabled):
        st.session_state.train_selected_datasets = selected_datasets
        st.session_state.train_model_configs_data = model_configs
        st.session_state.train_enable_selection = enable_selection
        st.session_state.train_selection_ratio = selection_ratio / 100
        st.session_state.train_top_k = top_k / 100
        st.session_state.train_checkpoint_interval = checkpoint_interval
        st.session_state.train_seed = seed
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

    task = st.session_state.train_task
    datasets_sel = st.session_state.train_selected_datasets
    model_cfgs = st.session_state.train_model_configs_data
    task_label = "классификации" if task == "classification" else "детекции"

    st.title(f"Выполняется обучение ({task_label})...")
    st.info(
        f"**Датасеты:** {', '.join(datasets_sel)}  \n"
        f"**Модели:** {', '.join(m['name'] for m in model_cfgs)}  \n"
        f"**Seed:** {st.session_state.train_seed}"
    )

    progress_placeholder = st.empty()

    if st.session_state.train_output_queue is None:
        q = queue.Queue()
        st.session_state.train_output_queue = q

        datasets_path = get_datasets_path()
        seed = st.session_state.train_seed

        if task == "detection":
            t = threading.Thread(
                target=_run_detection_thread,
                args=(
                    q, datasets_sel, model_cfgs, datasets_path,
                    st.session_state.train_enable_selection,
                    st.session_state.train_selection_ratio,
                    st.session_state.train_top_k,
                    st.session_state.train_checkpoint_interval,
                    seed,
                ),
                daemon=True,
            )
        else:
            t = threading.Thread(
                target=_run_classification_thread,
                args=(
                    q, datasets_sel, model_cfgs, datasets_path,
                    st.session_state.train_checkpoint_interval,
                    seed,
                    st.session_state.train_enable_selection,
                    st.session_state.train_selection_ratio,
                    st.session_state.train_top_k,
                ),
                daemon=True,
            )
        t.start()

    q = st.session_state.train_output_queue
    while not q.empty():
        msg_type, msg_data = q.get_nowait()
        if msg_type == "log":
            st.session_state.train_log_lines.append(msg_data)
        elif msg_type == "results_dir":
            st.session_state.train_results_dir = msg_data
        elif msg_type == "metrics":
            try:
                st.session_state.train_metrics_data = json.loads(msg_data)
            except Exception:
                pass
        elif msg_type == "error":
            st.session_state.train_error = msg_data
        elif msg_type == "done":
            st.session_state.train_thread_done = True

    with progress_placeholder.container():
        log_lines = st.session_state.train_log_lines
        display = "\n".join(log_lines[-80:]) if log_lines else "Инициализация..."
        st.code(display, language=None)

    if st.session_state.train_thread_done:
        st.session_state.train_stage = "done"
        time.sleep(0.5)
        st.rerun()
    else:
        time.sleep(2.0)
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# ЭТАП 3: Результаты
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.train_stage == "done":

    task = st.session_state.train_task
    task_label = "классификации" if task == "classification" else "детекции"

    if st.session_state.train_error:
        st.title("Обучение завершено с ошибкой")
        st.error("Во время обучения возникла ошибка:")
        st.code(st.session_state.train_error)
    else:
        st.title(f"Обучение завершено ({task_label})")

        st.subheader("Финальные метрики")

        if task == "detection":
            st.warning(
                "⚠️ `train_loss` и `val_loss` вычисляются по-разному для YOLO, "
                "Faster R-CNN и RetinaNet — сравнивать модели по loss некорректно. "
                "Используй **mAP50** и **mAP50-95**."
            )
        else:
            st.info(
                "Метрики AUC и ACC соответствуют Yang et al. (2021) MedMNIST. "
                f"Seed обучения: **{st.session_state.train_seed}**"
            )

        metrics_data = st.session_state.train_metrics_data
        if metrics_data:
            rows = []
            for key, history in metrics_data.items():
                if not history:
                    continue
                last = history[-1] if isinstance(history, list) else {}
                if isinstance(last, dict):
                    row = {"Модель / Датасет": key}
                    if task == "detection":
                        for metric in ["mAP50", "mAP50-95", "precision", "recall", "f1"]:
                            row[metric] = f"{last.get(metric, 0):.4f}" if metric in last else "—"
                    else:
                        # Ищем лучший по val_auc среди всей истории
                        best = max(history, key=lambda x: x.get("val_auc", x.get("val_acc", 0)))
                        for metric in ["val_auc", "val_acc", "val_loss"]:
                            row[metric + " (best)"] = f"{best.get(metric, 0):.4f}"
                        for metric in ["test_val_auc", "test_val_acc"]:
                            if metric in last:
                                row[metric.replace("test_val_", "test_")] = f"{last[metric]:.4f}"
                    rows.append(row)

            if rows:
                df = pd.DataFrame(rows).set_index("Модель / Датасет")
                st.dataframe(df, use_container_width=True)

        with st.expander("Полный лог обучения", expanded=False):
            st.code("\n".join(st.session_state.train_log_lines), language=None)

        st.subheader("Сохранение результатов")
        full_log_text = "\n".join(st.session_state.train_log_lines)
        st.download_button("Скачать лог (.txt)", data=full_log_text,
                            file_name="training_log.txt", mime="text/plain")

        if st.session_state.train_metrics_data:
            metrics_json = json.dumps(st.session_state.train_metrics_data, indent=2, ensure_ascii=False)
            st.download_button("Скачать метрики (.json)", data=metrics_json,
                                file_name="metrics.json", mime="application/json")

        if st.session_state.train_results_dir:
            st.info(f"📁 Папка с результатами: `{st.session_state.train_results_dir}`")

    st.divider()
    st.subheader("Что дальше?")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Подобрать предобработку", use_container_width=True):
            st.switch_page("pages/2_Предобработка.py")
    with col2:
        if st.button("Обучить снова", use_container_width=True):
            for k in ["train_stage", "train_log_lines", "train_output_queue",
                      "train_error", "train_metrics_data", "train_thread_done"]:
                st.session_state[k] = _state_defaults.get(k, None)
            st.session_state.train_stage = "configure"
            st.rerun()
