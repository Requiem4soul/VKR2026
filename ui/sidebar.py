"""
ui/sidebar.py — Общий сайдбар для всех страниц
"""

import os
import signal
import streamlit as st
from ui.state import is_path_configured, get_datasets_path


def render_sidebar():
    """Отрисовывает общий сайдбар. Вызывать на каждой странице."""
    with st.sidebar:
        st.title("🔬 VKR2026")
        st.caption("Система детекции объектов")
        st.divider()

        if is_path_configured():
            path = get_datasets_path()
            st.success("📁 Датасеты подключены")
            st.caption(str(path))
        else:
            st.warning("⚠️ Путь к датасетам не задан")
            st.caption("Перейди в **Настройки** для настройки")

        st.divider()
        st.caption("Навигация через меню слева ↑")
        st.divider()

        if st.button("🔴 Завершить работу", use_container_width=True):
            st.info("Приложение завершается...")
            os.kill(os.getpid(), signal.SIGTERM)
