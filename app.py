"""
VKR2026 — Streamlit интерфейс
Главная страница: навигация и статус системы
"""

import streamlit as st
from ui.state import init_session_state, get_datasets_path, is_path_configured
from ui.sidebar import render_sidebar

st.set_page_config(
    page_title="Информация",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

init_session_state()
render_sidebar()

st.title("Подбор методов предобработки для повышения точности нейронных моделей")
st.markdown("**Дипломная работа (ВКР) 2026**")
st.divider()

if not is_path_configured():
    st.warning(
        "### Требуется первоначальная настройка\n\n"
        "Перейди в раздел **Настройки** в левом меню и укажи путь к папке с датасетами. "
        "Это нужно сделать только один раз."
    )
    st.stop()

col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown("### Подбор пайплайна предобработки")
    st.markdown(
        "Автоматический поиск оптимальной комбинации методов предобработки "
        "для повышения точности нейронной модели."
    )
    st.markdown("**Как выполняется:**")
    st.markdown(
        "1. Необходимо выбрать тип изображений\n"
        "2. При применении методов автоподбора бюджета определяется минимальный достаточный бюджет\n"
        "3. Происходит первоначальный отбор признаков\n"
        "4. Производится построение различных комбинаций, их оценка и отсев\n"
        "5. Полученный набор и оригинальный датасет обучаются и сравниваются на полном бюджете"
    )
    if st.button(
        "Перейти к подбору комбинаций методов",
        type="primary",
        use_container_width=True,
        key="btn_pipeline",
    ):
        st.switch_page("pages/2_Подбор_предобработки.py")

with col2:
    st.markdown("### Настройка")
    st.markdown(
        "Первоначальная настройка приложения: "
        "необходимо указать путь к папке с датасетами."
    )
    st.markdown("**Важно!**")
    st.markdown(
        "Свободное пространство на диске, в котором находится данная папка, \n"
        "должно быть достаточным не только для хранения оригинальных датасетов, \n"
        "но и для дополнительных датасетов, которые будут создаваться по ходу выполнения \n"
        "подбора предобработки. Строго рекомендуется наличие на диске как минимум 100ГБ \n"
        "свободного пространства"
    )
    st.markdown("")
    if st.button(
        "Перейти к настройкам",
        type="primary",
        use_container_width=True,
        key="btn_settings",
    ):
        st.switch_page("pages/1_Настройки.py")

st.divider()
st.subheader("Статус системы")

col_a, col_b, col_c = st.columns(3)

with col_a:
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
            label = gpu_name[:25] + "..." if len(gpu_name) > 25 else gpu_name
            st.metric("GPU", label)
            st.caption(f"VRAM: {vram:.1f} GB")
        else:
            st.metric("GPU", "Не обнаружен")
            st.caption("Обучение будет на CPU (значительно медленнее)")
    except ImportError:
        st.metric("GPU", "PyTorch не установлен")

with col_b:
    datasets_path = get_datasets_path()
    if datasets_path and datasets_path.exists():
        datasets = [d for d in datasets_path.iterdir() if d.is_dir()]
        st.metric("Датасетов найдено", len(datasets))
        st.caption(str(datasets_path))
    else:
        st.metric("Датасетов найдено", "—")
        st.caption("Путь не настроен")

with col_c:
    try:
        import ultralytics
        st.metric("Ultralytics YOLO", ultralytics.__version__)
        st.caption("YOLOv8 готов к работе")
    except ImportError:
        st.metric("Ultralytics YOLO", "Не установлен")
        st.caption("pip install ultralytics")
