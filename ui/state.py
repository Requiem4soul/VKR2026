"""
ui/state.py — Управление состоянием сессии и конфигурацией

Хранит путь к датасетам в .env (один раз), остальное — в st.session_state.
"""

import os
import streamlit as st
from pathlib import Path
from dotenv import load_dotenv, set_key, find_dotenv

# Путь до .env в корне проекта (рядом с app.py)
ENV_FILE = Path(__file__).parent.parent / ".env"
ENV_KEY = "DATASETS_GLOBAL_PATH"
_DEFAULT_SENTINEL = ""


def init_session_state():
    """Инициализация всех ключей session_state при первом запуске."""
    defaults = {
        # Конфигурация
        "datasets_path": None,          # Path | None

        # Предобработка
        "prep_dataset": None,           # str — выбранный датасет
        "prep_running": False,          # bool — идёт ли процесс
        "prep_log": [],                 # list[str] — строки лога
        "prep_done": False,             # bool — завершено ли
        "prep_result_datasets": [],     # list[str] — созданные датасеты

        # Обучение
        "train_datasets": [],           # list[str]
        "train_model_configs": [],      # list[dict]
        "train_running": False,
        "train_log_file": None,         # str — путь к файлу лога
        "train_results_dir": None,      # str — папка результатов
        "train_done": False,
        "train_metrics": {},            # dict финальных метрик
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

    # Подгружаем сохранённый путь из .env если session_state пуст
    if st.session_state["datasets_path"] is None:
        saved = _load_path_from_env()
        if saved:
            st.session_state["datasets_path"] = saved


# ── Работа с путём ─────────────────────────────────────────────────────────

def _load_path_from_env() -> Path | None:
    """Загружает путь из .env файла. Возвращает Path или None."""
    if ENV_FILE.exists():
        load_dotenv(ENV_FILE, override=True)
    raw = os.getenv(ENV_KEY, _DEFAULT_SENTINEL).strip()
    if not raw:
        return None
    p = Path(raw)
    return p if p.exists() else None


def save_path_to_env(path: Path):
    """Сохраняет путь в .env и в session_state."""
    # Создаём .env если не существует
    if not ENV_FILE.exists():
        ENV_FILE.touch()
    set_key(str(ENV_FILE), ENV_KEY, str(path))
    st.session_state["datasets_path"] = path


def get_datasets_path() -> Path | None:
    """Возвращает текущий путь к папке датасетов или None."""
    return st.session_state.get("datasets_path")


def is_path_configured() -> bool:
    """Проверяет что путь задан и существует."""
    p = get_datasets_path()
    return p is not None and p.exists()


def get_available_datasets() -> list[str]:
    """Возвращает список доступных датасетов (названия папок)."""
    p = get_datasets_path()
    if not p or not p.exists():
        return []
    return sorted([d.name for d in p.iterdir() if d.is_dir()])


# ── Вспомогательные функции для лога ──────────────────────────────────────

def append_log(key: str, line: str):
    """Добавляет строку в лог в session_state."""
    if key not in st.session_state:
        st.session_state[key] = []
    st.session_state[key].append(line)


def get_log(key: str) -> list[str]:
    return st.session_state.get(key, [])


def clear_log(key: str):
    st.session_state[key] = []
