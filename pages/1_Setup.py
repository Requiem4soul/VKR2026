"""
pages/1_Setup.py — Настройка пути к датасетам

Показывается при первом запуске или при необходимости изменить путь.
Путь сохраняется в .env и не нужно вводить повторно.
"""

import streamlit as st
from pathlib import Path
from ui.state import (
    init_session_state,
    save_path_to_env,
    get_datasets_path,
    is_path_configured,
    get_available_datasets,
)

st.set_page_config(page_title="Настройки — VKR2026", page_icon="⚙️", layout="wide")
init_session_state()

# ── Сайдбар ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🔬 VKR2026")
    if is_path_configured():
        st.success("📁 Путь настроен")
    else:
        st.warning("⚠️ Требуется настройка")

# ── Заголовок ─────────────────────────────────────────────────────────────
st.title("⚙️ Настройки")
st.markdown("Укажи путь к папке с датасетами. Это нужно сделать **только один раз** — путь сохранится в `.env` файл.")

st.divider()

# ── Текущий статус ─────────────────────────────────────────────────────────
current_path = get_datasets_path()

if current_path and current_path.exists():
    st.success(f"✅ Текущий путь: `{current_path}`")
    datasets = get_available_datasets()
    if datasets:
        st.info(f"Найдено датасетов: **{len(datasets)}**\n\n" + ", ".join(f"`{d}`" for d in datasets))
    else:
        st.warning("Папка существует, но датасетов внутри не найдено (нет подпапок).")
elif current_path:
    st.error(f"❌ Сохранённый путь не существует: `{current_path}`")
else:
    st.info("Путь ещё не задан.")

st.divider()

# ── Форма ввода нового пути ────────────────────────────────────────────────
st.subheader("Задать / изменить путь к датасетам")
st.markdown(
    "Укажи **полный путь** до папки, внутри которой лежат подпапки с датасетами. "
    "Например: `N:\\VKR_Datasets` или `/home/user/datasets`"
)

# Предзаполняем текущим значением если есть
default_val = str(current_path) if current_path else ""
new_path_str = st.text_input(
    "Путь к папке с датасетами",
    value=default_val,
    placeholder="Например: N:\\VKR_Datasets",
)

col1, col2 = st.columns([1, 3])

with col1:
    save_clicked = st.button("💾 Сохранить", type="primary", use_container_width=True)

with col2:
    if current_path and is_path_configured():
        if st.button("🔄 Сбросить путь", use_container_width=True):
            save_path_to_env(Path(""))
            st.session_state["datasets_path"] = None
            st.rerun()

# ── Обработка сохранения ───────────────────────────────────────────────────
if save_clicked:
    if not new_path_str.strip():
        st.error("Путь не может быть пустым.")
    else:
        p = Path(new_path_str.strip())
        if not p.exists():
            st.error(f"❌ Папка не существует: `{p}`\n\nПроверь правильность пути.")
        elif not p.is_dir():
            st.error(f"❌ Это не папка: `{p}`")
        else:
            save_path_to_env(p)
            datasets = [d.name for d in p.iterdir() if d.is_dir()]
            st.success(f"✅ Путь сохранён! Найдено датасетов: **{len(datasets)}**")
            if datasets:
                st.markdown("**Найденные датасеты:**")
                for d in sorted(datasets):
                    st.markdown(f"  - `{d}`")
            st.rerun()

st.divider()

# ── Подсказки ──────────────────────────────────────────────────────────────
with st.expander("💡 Подсказки по структуре датасетов"):
    st.markdown("""
Каждый датасет должен быть папкой внутри указанной директории и содержать структуру YOLO:

```
VKR_Datasets/
├── MyDataset/
│   ├── train/
│   │   ├── images/
│   │   └── labels/
│   ├── valid/
│   │   ├── images/
│   │   └── labels/
│   ├── test/
│   │   ├── images/
│   │   └── labels/
│   └── data.yaml
└── AnotherDataset/
    └── ...
```

Файл `data.yaml` обязателен для обучения моделей.
    """)
