"""
pages/2_Preprocessing.py — Подбор и применение предобработки

Полный цикл: выбор датасета → анализ → стратегия → варианты → применение → результаты.
Запускает apply_preprocessing_with_modality_detection() в фоновом процессе,
перехватывает вывод и показывает его в реальном времени.
"""

import sys
import time
import threading
import queue
import streamlit as st
from pathlib import Path

from ui.sidebar import render_sidebar
from ui.state import (
    init_session_state,
    is_path_configured,
    get_available_datasets,
    get_datasets_path,
)

st.set_page_config(page_title="Предобработка — VKR2026", page_icon="🧪", layout="wide")
init_session_state()

# ── Сайдбар ───────────────────────────────────────────────────────────────
render_sidebar()

# ── Проверка конфигурации ──────────────────────────────────────────────────
if not is_path_configured():
    st.error("⚠️ Сначала настрой путь к датасетам в разделе **⚙️ Настройки**.")
    st.stop()

st.title("🧪 Подбор и применение предобработки")
st.markdown(
    "Система автоматически определит тип датасета и подберёт оптимальную стратегию предобработки. "
    "При необходимости можно создать несколько вариантов интенсивности для последующего сравнения."
)
st.divider()

# ── Состояние страницы ─────────────────────────────────────────────────────
# Используем ключи с префиксом 'prep_' в session_state

if "prep_stage" not in st.session_state:
    st.session_state.prep_stage = "configure"  # configure | running | done

if "prep_log_lines" not in st.session_state:
    st.session_state.prep_log_lines = []

if "prep_result_names" not in st.session_state:
    st.session_state.prep_result_names = []

if "prep_output_queue" not in st.session_state:
    st.session_state.prep_output_queue = None

if "prep_thread_done" not in st.session_state:
    st.session_state.prep_thread_done = False

if "prep_error" not in st.session_state:
    st.session_state.prep_error = None

# ── ЭТАП 1: Конфигурация ───────────────────────────────────────────────────
if st.session_state.prep_stage == "configure":

    st.subheader("Шаг 1: Выберите датасет")
    datasets = get_available_datasets()

    if not datasets:
        st.warning("В указанной папке не найдено датасетов. Проверь путь в Настройках.")
        st.stop()

    selected_dataset = st.selectbox(
        "Датасет для анализа и предобработки",
        options=datasets,
        help="Выбери датасет, для которого нужно подобрать предобработку"
    )

    st.divider()
    st.subheader("Шаг 2: Варианты интенсивности")
    st.markdown(
        "Помимо базового варианта можно создать дополнительные с другой интенсивностью. "
        "Это позволит эмпирически выбрать лучший после обучения модели *(Montaha et al., 2022)*."
    )

    col1, col2 = st.columns(2)
    with col1:
        create_weak = st.checkbox(
            "🟡 Слабый вариант (×0.5)",
            help="Параметры предобработки уменьшены в 2 раза от базового"
        )
    with col2:
        create_strong = st.checkbox(
            "🔴 Сильный вариант (×2.0)",
            help="Параметры предобработки увеличены в 2 раза от базового"
        )

    # Базовый вариант всегда создаётся
    selected_variants = []
    if create_weak:
        selected_variants.append("weak")
    selected_variants.append("base")  # базовый всегда
    if create_strong:
        selected_variants.append("strong")

    st.info(f"Будет создано вариантов: **{len(selected_variants)}** — {', '.join(selected_variants)}")

    st.divider()
    st.subheader("Шаг 3: Названия новых датасетов")

    default_base = f"{selected_dataset}_preprocessed"
    dataset_names_input = {}

    if len(selected_variants) == 1:
        # Только один вариант — одно простое поле
        custom = st.text_input(
            "Название нового датасета",
            value=default_base,
            key="name_only",
            help="Под этим именем будет сохранён предобработанный датасет"
        )
        dataset_names_input["base"] = custom
    else:
        # Несколько вариантов — шаблон + отдельные поля
        st.markdown(
            "Задай базовое название — к нему автоматически добавятся суффиксы. "
            "При желании можно изменить каждое название отдельно."
        )
        base_name = st.text_input("Базовое название (шаблон)", value=default_base, disabled=True)
        st.caption("Итоговые названия датасетов:")
        for level in selected_variants:
            default = f"{base_name}_{level}"
            custom = st.text_input(
                f"{'🟡 Слабый' if level == 'weak' else '🟢 Базовый' if level == 'base' else '🔴 Сильный'}",
                value=default,
                key=f"name_{level}"
            )
            dataset_names_input[level] = custom

    st.divider()

    if st.button("🚀 Запустить подбор предобработки", type="primary", use_container_width=True):
        # Сохраняем параметры в session_state и переходим к запуску
        st.session_state.prep_selected_dataset = selected_dataset
        st.session_state.prep_selected_variants = selected_variants
        st.session_state.prep_dataset_names = dataset_names_input
        st.session_state.prep_stage = "running"
        st.session_state.prep_log_lines = []
        st.session_state.prep_thread_done = False
        st.session_state.prep_error = None
        st.session_state.prep_result_names = list(dataset_names_input.values())
        st.rerun()

# ── ЭТАП 2: Выполнение с выводом лога ─────────────────────────────────────
elif st.session_state.prep_stage == "running":

    st.subheader("⏳ Выполняется подбор предобработки...")
    dataset_nm = st.session_state.prep_selected_dataset
    variants = st.session_state.prep_selected_variants
    ds_names = st.session_state.prep_dataset_names

    st.info(
        f"**Датасет:** `{dataset_nm}` | "
        f"**Варианты:** {', '.join(variants)} | "
        f"**Новые датасеты:** {', '.join(ds_names.values())}"
    )

    log_placeholder = st.empty()
    status_placeholder = st.empty()

    # Запускаем фоновый поток если ещё не запущен
    if st.session_state.prep_output_queue is None:
        q = queue.Queue()
        st.session_state.prep_output_queue = q

        def run_preprocessing(q, dataset_name, selected_variants, dataset_names, datasets_path):
            """Запускает предобработку в отдельном потоке, перехватывает вывод."""
            import io

            class QueueWriter(io.TextIOBase):
                def write(self, text):
                    if text.strip():
                        q.put(("log", text.rstrip()))
                    return len(text)
                def flush(self):
                    pass

            old_stdout = sys.stdout
            old_stderr = sys.stderr
            sys.stdout = QueueWriter()
            sys.stderr = QueueWriter()

            try:
                # Добавляем корень проекта в путь для импортов
                import os
                project_root = str(Path(__file__).parent.parent)
                if project_root not in sys.path:
                    sys.path.insert(0, project_root)

                # Устанавливаем переменную окружения для config.py
                os.environ["DATASETS_GLOBAL_PATH"] = str(datasets_path)

                from Data.Datasets.dataset_work import get_dataset_path
                from Utils.image_analyzer import UniversalImageAnalyzer
                from Utils.modality_classifier import ImageModalityClassifier
                from Utils.preprocessing_rules import PreprocessingRules
                from Utils.preprocessing_selector import AdaptivePreprocessingSelector
                from Utils.intensity_variants import generate_intensity_variants, print_variants_comparison
                from Preprocessing.applicator import DatasetPreprocessor

                dataset_path = get_dataset_path(dataset_name)

                # Шаг 2: Анализ
                q.put(("log", "=" * 70))
                q.put(("log", "ШАГ 2: АНАЛИЗ ДАТАСЕТА"))
                q.put(("log", "=" * 70))
                analyzer = UniversalImageAnalyzer(verbose=False)
                dataset_metrics, image_metrics = analyzer.analyze_dataset(dataset_path, split='train')

                # Шаг 3: Классификация
                q.put(("log", "=" * 70))
                q.put(("log", "ШАГ 3: КЛАССИФИКАЦИЯ ТИПА ДАТАСЕТА"))
                q.put(("log", "=" * 70))
                classifier = ImageModalityClassifier()
                modality_info = classifier.classify(dataset_metrics)
                classifier.print_classification_report(modality_info)

                # Шаг 4: Правила
                q.put(("log", "=" * 70))
                q.put(("log", f"ШАГ 4: ПРАВИЛА ДЛЯ ТИПА '{modality_info['modality'].upper()}'"))
                q.put(("log", "=" * 70))
                PreprocessingRules.print_rules_summary(modality_info['modality'])

                # Шаг 5: Стратегия
                q.put(("log", "=" * 70))
                q.put(("log", "ШАГ 5: ВЫБОР СТРАТЕГИИ ПРЕДОБРАБОТКИ"))
                q.put(("log", "=" * 70))
                selector = AdaptivePreprocessingSelector(analyzer=analyzer, modality_info=modality_info)
                strategy = selector.select_strategy(dataset_path, split='train')
                selector.print_strategy_info(strategy)

                # Шаг 6-7: Параметры и варианты
                from apply_preprocessing import _build_params_for_modality
                base_params = _build_params_for_modality(modality_info, selector)
                methods = strategy.get('methods', [])

                variants_params = generate_intensity_variants(
                    base_params=base_params,
                    methods=methods,
                    variants=selected_variants
                )

                if len(selected_variants) > 1:
                    print_variants_comparison(variants_params, methods)

                # Шаг 8: Применение
                q.put(("log", "=" * 70))
                q.put(("log", "ШАГ 8: ПРИМЕНЕНИЕ ПРЕДОБРАБОТКИ"))
                q.put(("log", "=" * 70))
                preprocessor = DatasetPreprocessor()

                for level in selected_variants:
                    new_name = dataset_names[level]
                    level_params = variants_params[level]
                    q.put(("log", f"\n  → Создаю '{new_name}' (вариант: {level})"))

                    if strategy['strategy'] == 'adaptive':
                        preprocessor.apply_adaptive_preprocessing(
                            source_dataset=dataset_name,
                            target_dataset=new_name,
                            clusters=strategy['clusters'],
                            image_metrics=image_metrics,
                            params=level_params
                        )
                    else:
                        preprocessor.apply_global_preprocessing(
                            source_dataset=dataset_name,
                            target_dataset=new_name,
                            methods=methods,
                            params=level_params
                        )

                # Шаг 9-10: Анализ результатов
                q.put(("log", "=" * 70))
                q.put(("log", "ШАГ 9-10: АНАЛИЗ И СРАВНЕНИЕ РЕЗУЛЬТАТОВ"))
                q.put(("log", "=" * 70))

                all_preprocessed_metrics = {}
                for level in selected_variants:
                    ds_name = dataset_names[level]
                    ds_path = get_dataset_path(ds_name)
                    q.put(("log", f"\n  Анализирую '{ds_name}'..."))
                    metrics, _ = analyzer.analyze_dataset(ds_path, split='train')
                    all_preprocessed_metrics[level] = metrics

                from apply_preprocessing import _print_comparison_all
                _print_comparison_all(
                    original_metrics=dataset_metrics,
                    variants_metrics=all_preprocessed_metrics,
                    selected_variants=selected_variants,
                    modality_info=modality_info
                )

                q.put(("done", "✅ Предобработка завершена успешно!"))

            except Exception as e:
                import traceback
                q.put(("error", f"Ошибка: {e}\n{traceback.format_exc()}"))
            finally:
                sys.stdout = old_stdout
                sys.stderr = old_stderr

        thread = threading.Thread(
            target=run_preprocessing,
            args=(q, dataset_nm, variants, ds_names, get_datasets_path()),
            daemon=True
        )
        thread.start()

    # Читаем очередь и обновляем лог
    q = st.session_state.prep_output_queue
    if q is not None:
        try:
            while True:
                msg_type, msg = q.get_nowait()
                if msg_type == "log":
                    st.session_state.prep_log_lines.append(msg)
                elif msg_type == "done":
                    st.session_state.prep_log_lines.append(msg)
                    st.session_state.prep_thread_done = True
                    st.session_state.prep_stage = "done"
                    st.session_state.prep_output_queue = None
                    break
                elif msg_type == "error":
                    st.session_state.prep_error = msg
                    st.session_state.prep_log_lines.append(f"❌ {msg}")
                    st.session_state.prep_thread_done = True
                    st.session_state.prep_stage = "done"
                    st.session_state.prep_output_queue = None
                    break
        except queue.Empty:
            pass

    # Отображаем лог — используем st.code для надёжного обновления
    with log_placeholder.container():
        lines = st.session_state.prep_log_lines
        if lines:
            log_text = "\n".join(lines[-200:])
        else:
            log_text = "Ожидание вывода..."
        st.markdown("**Вывод процесса:**")
        st.code(log_text, language=None)

    if not st.session_state.prep_thread_done:
        with status_placeholder:
            st.info("🔄 Выполняется... страница обновляется автоматически.")
        time.sleep(1.5)
        st.rerun()
    else:
        st.rerun()

# ── ЭТАП 3: Результаты ─────────────────────────────────────────────────────
elif st.session_state.prep_stage == "done":

    if st.session_state.prep_error:
        st.error("❌ Во время предобработки возникла ошибка:")
        st.code(st.session_state.prep_error)
    else:
        st.success("✅ Предобработка завершена успешно!")
        result_names = st.session_state.prep_result_names
        st.markdown(f"**Создано датасетов: {len(result_names)}**")
        for name in result_names:
            st.markdown(f"  - `{name}`")

    # Полный лог
    with st.expander("📋 Полный лог выполнения", expanded=False):
        full_log = "\n".join(st.session_state.prep_log_lines)
        st.code(full_log, language=None)

        # Кнопка сохранения лога
        st.download_button(
            label="💾 Скачать лог предобработки",
            data=full_log,
            file_name=f"preprocessing_log_{st.session_state.prep_selected_dataset}.txt",
            mime="text/plain",
        )

    st.divider()
    st.subheader("Что дальше?")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🧪 Подобрать предобработку ещё раз", use_container_width=True):
            # Сброс состояния
            st.session_state.prep_stage = "configure"
            st.session_state.prep_log_lines = []
            st.session_state.prep_output_queue = None
            st.session_state.prep_error = None
            st.rerun()
    with col2:
        if st.button("🚀 Перейти к обучению моделей", type="primary", use_container_width=True):
            st.switch_page("pages/3_Training.py")
