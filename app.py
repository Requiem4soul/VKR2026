"""
VKR2026 — Streamlit интерфейс
Главная страница: навигация и проверка конфигурации
"""

import streamlit as st
from pathlib import Path
from ui.state import init_session_state, get_datasets_path, is_path_configured

# ── Конфигурация страницы ──────────────────────────────────────────────────
st.set_page_config(
    page_title="VKR2026 — Детекция объектов",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

init_session_state()

# ── Сайдбар ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🔬 VKR2026")
    st.caption("Система детекции объектов")
    st.divider()

    if is_path_configured():
        path = get_datasets_path()
        st.success(f"📁 Датасеты подключены")
        st.caption(str(path))
    else:
        st.warning("⚠️ Путь к датасетам не задан")
        st.caption("Перейди в **Настройки** для настройки")

    st.divider()
    st.caption("Навигация через меню слева ↑")

# ── Главная страница ───────────────────────────────────────────────────────
st.title("🔬 Система анализа и обучения моделей детекции")
st.markdown("**Дипломная работа (ВКР) 2026**")

st.divider()

if not is_path_configured():
    st.warning(
        "### ⚠️ Требуется первоначальная настройка\n\n"
        "Перейди в раздел **⚙️ Настройки** в левом меню и укажи путь к папке с датасетами. "
        "Это нужно сделать только один раз."
    )
    st.stop()

# ── Карточки выбора режима ─────────────────────────────────────────────────
st.subheader("Выберите режим работы")
st.markdown("Оба режима можно использовать независимо друг от друга.")

col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown("### 🧪 Подбор предобработки")
    st.markdown(
        "Автоматический анализ датасета, определение типа изображений "
        "и подбор оптимальных методов предобработки. Создаёт один или "
        "несколько вариантов предобработанного датасета (слабый / базовый / сильный)."
    )
    st.markdown("**Что произойдёт:**")
    st.markdown(
        "- Анализ шума, контраста, яркости, резкости\n"
        "- Определение типа датасета (SAR, медицинский, натуральный...)\n"
        "- Подбор стратегии (глобальная или адаптивная)\n"
        "- Создание предобработанных датасетов"
    )
    if st.button("🧪 Начать подбор предобработки", type="primary", use_container_width=True):
        st.switch_page("pages/2_Preprocessing.py")

with col2:
    st.markdown("### 🚀 Обучение моделей")
    st.markdown(
        "Обучение нескольких моделей детекции (YOLOv8, Faster R-CNN, RetinaNet) "
        "на выбранных датасетах. Поддерживает Early Stopping и ранний отбор моделей."
    )
    st.markdown("**Что произойдёт:**")
    st.markdown(
        "- Выбор датасетов и моделей для обучения\n"
        "- Настройка гиперпараметров (с умными значениями по умолчанию)\n"
        "- Обучение с отображением прогресса в реальном времени\n"
        "- Сравнение финальных метрик моделей"
    )
    if st.button("🚀 Перейти к обучению", type="primary", use_container_width=True):
        st.switch_page("pages/3_Training.py")

st.divider()

# ── Статус системы ─────────────────────────────────────────────────────────
st.subheader("📊 Статус системы")

col_a, col_b, col_c = st.columns(3)

with col_a:
    try:
        import torch
        cuda_ok = torch.cuda.is_available()
        if cuda_ok:
            gpu_name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
            st.metric("🖥️ GPU", gpu_name[:25] + "..." if len(gpu_name) > 25 else gpu_name)
            st.caption(f"VRAM: {vram:.1f} GB")
        else:
            st.metric("🖥️ GPU", "Не обнаружен")
            st.caption("Обучение будет на CPU (медленно)")
    except ImportError:
        st.metric("🖥️ GPU", "PyTorch не установлен")

with col_b:
    datasets_path = get_datasets_path()
    if datasets_path and datasets_path.exists():
        datasets = [d for d in datasets_path.iterdir() if d.is_dir()]
        st.metric("📁 Датасетов найдено", len(datasets))
        st.caption(str(datasets_path))
    else:
        st.metric("📁 Датасетов найдено", "—")
        st.caption("Путь не настроен")

with col_c:
    try:
        import ultralytics
        st.metric("🤖 Ultralytics", ultralytics.__version__)
        st.caption("YOLOv8 готов к работе")
    except ImportError:
        st.metric("🤖 Ultralytics", "Не установлен")
