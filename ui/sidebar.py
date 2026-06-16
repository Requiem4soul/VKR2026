import os
import signal
import streamlit as st
from ui.state import is_path_configured, get_datasets_path


def render_sidebar():
    """Отрисовывает общий сайдбар. Вызывать на каждой странице."""
    with st.sidebar:
        st.title("Система подбора комбинаций методов предобработки для повышения точности нейронной модели")
        st.caption("Выпускная квалификационная работа")
        st.divider()

        if is_path_configured():
            path = get_datasets_path()
            st.success("Датасеты подключены")
            st.caption(str(path))
        else:
            st.warning("Путь к датасетам не задан")
            st.caption("Перейди в **Настройки** для настройки")

        if st.button("Завершить работу", use_container_width=True):
            st.info("Приложение завершается...")
            os.kill(os.getpid(), signal.SIGTERM)
