"""
pages/old_metrics_2_Предобработка.py — Метрика-based подбор предобработки

Альтернативный подход к подбору предобработки:
1. Анализ метрик изображений (SNR, контраст, яркость, резкость)
2. Классификация модальности датасета (SAR, рентген, натуральное фото и др.)
3. Выбор стратегии (глобальная / адаптивная с кластеризацией KMeans)
4. Генерация вариантов интенсивности (weak / base / strong)
5. Применение предобработки и сравнение метрик изображений

Научное обоснование:
- Анализ метрик: Gonzalez & Woods (2018) "Digital Image Processing", 4th ed., Pearson
- SAR модальность: Oliver & Quegan (2004) "Understanding SAR Images", SciTech
- Медицинские снимки: Pham et al. (2000) Annual Review of Biomedical Engineering 2:315-337
- Инфракрасные: Vollmer & Möllmann (2017) "Infrared Thermal Imaging", Wiley
- Микроскопия: Sternberg (1983) Computer Methods in Anatomy 3:1-22
- CLAHE: Pisano et al. (1998) J. Digital Imaging 11(4):193-200
- Bilateral filter: Tomasi & Manduchi (1998) ICCV
- Интенсивности: Montaha et al. (2022) Front. Med. doi:10.3389/fmed.2022.924979

Отличие от SHA+SFS подхода (2_Предобработка.py):
- Не требует обучения модели для выбора методов (только анализ пикселей)
- Работает быстро, без GPU
- Параметры выбираются по научным правилам модальности, а не эмпирически
- Позволяет сравнить два подхода на одном датасете
"""

import sys
import json
import queue
import threading
import traceback
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional

import streamlit as st

from ui.sidebar import render_sidebar
from ui.state import (
    init_session_state,
    is_path_configured,
    get_available_datasets,
    get_datasets_path,
)

st.set_page_config(
    page_title="Метрика-based предобработка — VKR2026",
    page_icon=None,
    layout="wide",
)
init_session_state()
render_sidebar()

if not is_path_configured():
    st.error("Сначала настрой путь к датасетам в разделе **Настройки**.")
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
# СОСТОЯНИЕ СТРАНИЦЫ
# ══════════════════════════════════════════════════════════════════════════════

_STATE_DEFAULTS: Dict = {
    "om_stage":        "configure",  # configure | running | done
    "om_log_lines":    [],
    "om_queue":        None,
    "om_thread_done":  False,
    "om_error":        None,
    "om_result":       None,
}

for k, v in _STATE_DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


def _reset():
    for k, v in _STATE_DEFAULTS.items():
        st.session_state[k] = v
    st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# ФОНОВЫЙ ПОТОК
# ══════════════════════════════════════════════════════════════════════════════

def _run(q: queue.Queue, config: Dict):
    """Полный цикл: анализ → классификация → стратегия → применение."""

    def log(msg: str):
        ts = datetime.now().strftime("%H:%M:%S")
        q.put(("log", f"[{ts}] {msg}"))

    try:
        sys.path.insert(0, str(Path(__file__).parent.parent))

        from Utils.image_analyzer          import UniversalImageAnalyzer
        from Utils.modality_classifier     import ImageModalityClassifier
        from Utils.preprocessing_selector  import AdaptivePreprocessingSelector
        from Utils.preprocessing_rules     import PreprocessingRules
        from Utils.intensity_variants      import generate_intensity_variants
        from Preprocessing.applicator      import DatasetPreprocessor

        datasets_path = Path(config["datasets_path"])
        dataset_name  = config["dataset_name"]
        variants      = config["variants"]    # ['weak','base','strong']
        sample_size   = config["sample_size"]
        dataset_path  = datasets_path / dataset_name

        # ── ШАГ 1: Анализ изображений ─────────────────────────────────────
        log("=" * 70)
        log("ШАГ 1: АНАЛИЗ ИЗОБРАЖЕНИЙ")
        log("Gonzalez & Woods (2018) — SNR, контраст, яркость, резкость")
        log("=" * 70)

        analyzer = UniversalImageAnalyzer(verbose=False)
        log(f"  Датасет: {dataset_name} | выборка: {sample_size} изображений")
        dataset_metrics, image_metrics = analyzer.analyze_dataset(
            dataset_path, sample_size=sample_size, split="train"
        )

        log(f"  Проанализировано: {dataset_metrics.num_images} изображений")
        log(f"  Средний SNR:      {dataset_metrics.avg_snr:.1f} dB")
        log(f"  Средний контраст: {dataset_metrics.avg_contrast:.3f}")
        log(f"  Средняя яркость:  {dataset_metrics.avg_brightness:.3f}")
        log(f"  Средняя резкость: {dataset_metrics.avg_sharpness:.1f}")
        log(f"  Однородность:     {dataset_metrics.dataset_homogeneity:.2%}")
        is_color_str = "да" if dataset_metrics.is_color_dataset else "нет"
        log(f"  Цветной датасет:  {is_color_str} (MICD={dataset_metrics.avg_color_diversity:.1f})")
        issues = ', '.join(dataset_metrics.dominant_issues) or "не обнаружены"
        log(f"  Доминирующие проблемы: {issues}")

        # ── ШАГ 2: Классификация модальности ──────────────────────────────
        log("")
        log("=" * 70)
        log("ШАГ 2: КЛАССИФИКАЦИЯ ТИПА ДАТАСЕТА")
        log("Oliver & Quegan (2004), Pham et al. (2000), Gonzalez & Woods (2018)")
        log("=" * 70)

        classifier    = ImageModalityClassifier()
        modality_info = classifier.classify(dataset_metrics)

        log(f"  Тип:         {modality_info['modality'].upper()}")
        log(f"  Уверенность: {modality_info['confidence']*100:.1f}%")
        log(f"  Описание:    {modality_info['description']}")
        log(f"  Источник:    {modality_info['source']}")
        if dataset_metrics.is_color_dataset:
            log("  (SAR и рентген исключены — физически grayscale-модальности)")
        log("  Оценки по типам:")
        for mod, score in sorted(modality_info["all_scores"].items(),
                                  key=lambda x: x[1], reverse=True):
            bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
            log(f"    {mod:15s} {bar} {score*100:5.1f}%")

        # ── ШАГ 3: Правила предобработки ──────────────────────────────────
        log("")
        log("=" * 70)
        log(f"ШАГ 3: ПРАВИЛА ДЛЯ ТИПА '{modality_info['modality'].upper()}'")
        log("=" * 70)

        modality = modality_info["modality"]
        rules    = PreprocessingRules.get_rules(modality)
        all_methods = ["denoise", "contrast_enhancement",
                       "brightness_correction", "sharpening"]
        for method in all_methods:
            rule = rules.get(method, {})
            enabled = rule.get("enabled", True)
            status  = "✓ разрешён" if enabled else "✗ запрещён"
            rationale = rule.get("rationale", "")
            log(f"  {method:<25} {status}")
            if rationale:
                # Обрезаем длинные обоснования для лога
                short = rationale[:100] + "..." if len(rationale) > 100 else rationale
                log(f"    → {short}")

        # ── ШАГ 4: Выбор стратегии ────────────────────────────────────────
        log("")
        log("=" * 70)
        log("ШАГ 4: ВЫБОР СТРАТЕГИИ ПРЕДОБРАБОТКИ")
        log("=" * 70)

        selector = AdaptivePreprocessingSelector(
            analyzer=analyzer,
            modality_info=modality_info,
        )
        strategy = selector.select_strategy(dataset_path, split="train")

        log(f"  Стратегия:  {strategy['strategy'].upper()}")
        log(f"  Модальность: {strategy['modality'].upper()} "
            f"(уверенность {strategy['modality_confidence']*100:.1f}%)")

        if strategy["strategy"] == "global":
            methods = strategy.get("methods", [])
            log(f"  Методы: {', '.join(methods) if methods else 'не требуются'}")
        else:
            log(f"  Кластеров: {strategy['n_clusters']}")
            for cid, cinfo in strategy["clusters"].items():
                prep = cinfo.get("preprocessing", [])
                log(f"    Кластер {cid}: {cinfo['size']} изображений | "
                    f"SNR={cinfo['characteristics']['avg_snr']:.1f}dB | "
                    f"методы: {', '.join(prep) if prep else 'нет'}")

        # ── ШАГ 5: Параметры и варианты интенсивности ─────────────────────
        log("")
        log("=" * 70)
        log("ШАГ 5: ВАРИАНТЫ ИНТЕНСИВНОСТИ")
        log("Montaha et al. (2022) Front. Med. — оптимальная интенсивность")
        log("определяется эмпирически через метрики обученной модели")
        log("=" * 70)

        # Базовые параметры из правил модальности
        base_params: Dict = {}
        if strategy["strategy"] == "global":
            methods = strategy.get("methods", [])
        else:
            # Для адаптивной — берём объединение методов всех кластеров
            methods_set = set()
            for cinfo in strategy["clusters"].values():
                methods_set.update(cinfo.get("preprocessing", []))
            methods = list(methods_set)

        # Собираем параметры из правил модальности.
        # get_method_params возвращает только вложенный 'params',
        # но некоторые поля (например target_brightness) лежат на уровень выше.
        # Поэтому собираем параметры вручную из полного правила.
        rules = PreprocessingRules.get_rules(modality)
        for method in methods:
            rule = rules.get(method, {})
            if not rule.get("enabled", True):
                continue
            collected = {}
            # Берём вложенный 'params' если есть
            collected.update(rule.get("params", {}))
            # Добавляем верхнеуровневые числовые/строковые поля
            # (target_brightness, method и т.д.) — кроме служебных
            _skip = {"enabled", "params", "rationale", "source", "description"}
            for k, v in rule.items():
                if k not in _skip and isinstance(v, (int, float, str)):
                    collected[k] = v
            if collected:
                base_params[method] = collected

        variants_params = generate_intensity_variants(
            base_params=base_params,
            methods=methods,
            variants=variants,
        )

        for level in variants:
            vp = variants_params.get(level, {})
            log(f"  Вариант [{level}]:")
            for method, params in vp.items():
                nums = {k: v for k, v in params.items()
                        if isinstance(v, (int, float))}
                log(f"    {method}: {nums}")

        # ── ШАГ 6: Применение предобработки ───────────────────────────────
        log("")
        log("=" * 70)
        log("ШАГ 6: ПРИМЕНЕНИЕ ПРЕДОБРАБОТКИ")
        log("applicator.py — IMREAD_UNCHANGED, корректная поддержка RGB/grayscale")
        log("=" * 70)

        preprocessor   = DatasetPreprocessor()
        applied_names  = {}   # level → dataset_name
        applied_metrics: Dict = {}  # level → DatasetMetrics

        for level in variants:
            new_name    = f"{dataset_name}_om_{modality}_{level}"
            level_params = variants_params[level]
            applied_names[level] = new_name
            log(f"\n  → Создаю '{new_name}' (вариант: {level})")

            if strategy["strategy"] == "adaptive":
                preprocessor.apply_adaptive_preprocessing(
                    source_dataset=dataset_name,
                    target_dataset=new_name,
                    clusters=strategy["clusters"],
                    image_metrics=image_metrics,
                    params=level_params,
                )
            else:
                preprocessor.apply_global_preprocessing(
                    source_dataset=dataset_name,
                    target_dataset=new_name,
                    methods=methods,
                    params=level_params,
                )
            log(f"    ✓ '{new_name}' создан")

        # ── ШАГ 7: Анализ результатов ──────────────────────────────────────
        log("")
        log("=" * 70)
        log("ШАГ 7: СРАВНЕНИЕ МЕТРИК ИЗОБРАЖЕНИЙ")
        log("=" * 70)

        for level in variants:
            new_name  = applied_names[level]
            new_path  = datasets_path / new_name
            log(f"\n  Анализирую '{new_name}'...")
            try:
                new_metrics, _ = analyzer.analyze_dataset(
                    new_path, sample_size=sample_size, split="train"
                )
                applied_metrics[level] = new_metrics
                log(f"    SNR:      {dataset_metrics.avg_snr:.1f} → "
                    f"{new_metrics.avg_snr:.1f} dB "
                    f"({'▲' if new_metrics.avg_snr > dataset_metrics.avg_snr else '▼'})")
                log(f"    Контраст: {dataset_metrics.avg_contrast:.3f} → "
                    f"{new_metrics.avg_contrast:.3f} "
                    f"({'▲' if new_metrics.avg_contrast > dataset_metrics.avg_contrast else '▼'})")
                log(f"    Резкость: {dataset_metrics.avg_sharpness:.1f} → "
                    f"{new_metrics.avg_sharpness:.1f} "
                    f"({'▲' if new_metrics.avg_sharpness > dataset_metrics.avg_sharpness else '▼'})")
                log(f"    Яркость:  {dataset_metrics.avg_brightness:.3f} → "
                    f"{new_metrics.avg_brightness:.3f}")
            except Exception as e:
                log(f"    [ОШИБКА анализа] {e}")

        # ── Финал ─────────────────────────────────────────────────────────
        log("")
        log("=" * 70)
        log("ГОТОВО")
        log("=" * 70)
        log(f"  Датасет-источник:  {dataset_name}")
        for level, name in applied_names.items():
            log(f"  [{level}] → {name}")
        log("")
        log("Для оценки влияния на качество модели запусти обучение на")
        log("созданных датасетах в разделе 'Обучение' или в 2_Предобработка.py")

        q.put(("result", {
            "dataset_name":      dataset_name,
            "modality":          modality_info["modality"],
            "modality_conf":     modality_info["confidence"],
            "strategy":          strategy["strategy"],
            "variants":          variants,
            "applied_names":     applied_names,
            "orig_metrics":      dataset_metrics,
            "applied_metrics":   applied_metrics,
            "methods":           methods,
        }))

    except Exception as e:
        log(f"[ОШИБКА] {e}")
        log(traceback.format_exc())
        q.put(("error", str(e)))
    finally:
        q.put(("done", None))


# ══════════════════════════════════════════════════════════════════════════════
# UI — ЗАГОЛОВОК
# ══════════════════════════════════════════════════════════════════════════════

st.title("Метрика-based подбор предобработки")
st.markdown(
    "Система анализирует метрики изображений, определяет тип датасета "
    "(SAR, рентген, натуральное фото и др.) и подбирает предобработку "
    "по научным правилам для каждой модальности. "
    "**Не требует GPU и обучения модели.**"
)
st.info(
    "💡 **Сравнение подходов:** этот метод работает на основе статистики пикселей. "
    "Для сравнения с SHA+SFS подходом запусти тот же датасет в **2_Предобработка.py**.",
    icon=None,
)
st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# UI — КОНФИГУРАЦИЯ
# ══════════════════════════════════════════════════════════════════════════════

if st.session_state.om_stage == "configure":

    datasets = get_available_datasets()
    if not datasets:
        st.error("Нет доступных датасетов. Проверь путь в Настройках.")
        st.stop()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Датасет")
        dataset = st.selectbox(
            "Выберите датасет",
            options=datasets,
            key="om_dataset_select",
        )

        sample_size = st.number_input(
            "Изображений для анализа (выборка)",
            min_value=20, max_value=2000, value=100, step=20,
            help="Анализируются случайные N изображений из train-split. "
                 "Больше → точнее, медленнее. 100–200 обычно достаточно.",
            key="om_sample_input",
        )

    with col2:
        st.subheader("Варианты интенсивности")
        st.markdown(
            "Montaha et al. (2022) показали что оптимальная интенсивность "
            "предобработки определяется эмпирически. Можно создать несколько "
            "датасетов с разными параметрами."
        )
        var_weak   = st.checkbox("Слабый (weak)  — параметры ×0.5–0.7",  value=False, key="om_var_weak")
        var_base   = st.checkbox("Базовый (base) — параметры по правилам", value=True,  key="om_var_base")
        var_strong = st.checkbox("Сильный (strong) — параметры ×1.5–2.0", value=False, key="om_var_strong")

        selected_variants = []
        if var_weak:   selected_variants.append("weak")
        if var_base:   selected_variants.append("base")
        if var_strong: selected_variants.append("strong")

        if not selected_variants:
            st.warning("Выберите хотя бы один вариант интенсивности.")

    st.divider()

    st.subheader("Что будет создано")
    if selected_variants and dataset:
        modality_placeholder = "<тип>"
        for v in selected_variants:
            st.code(f"{dataset}_om_{modality_placeholder}_{v}")
        st.caption(
            "Тип датасета определяется автоматически после анализа. "
            "Созданные датасеты появятся в папке датасетов."
        )

    can_run = bool(selected_variants)
    if st.button(
        "▶ Запустить анализ и применить предобработку",
        disabled=not can_run,
        type="primary",
    ):
        config = {
            "datasets_path": get_datasets_path(),
            "dataset_name":  dataset,
            "variants":      selected_variants,
            "sample_size":   int(sample_size),
        }
        q = queue.Queue()
        st.session_state.om_queue       = q
        st.session_state.om_thread_done = False
        st.session_state.om_log_lines   = []
        st.session_state.om_error       = None
        st.session_state.om_result      = None
        st.session_state.om_stage       = "running"

        t = threading.Thread(target=_run, args=(q, config), daemon=True)
        t.start()
        st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# UI — ВЫПОЛНЕНИЕ
# ══════════════════════════════════════════════════════════════════════════════

elif st.session_state.om_stage == "running":

    st.subheader("Выполняется анализ...")
    log_container = st.empty()
    q = st.session_state.om_queue

    # Читаем ВСЕ доступные сообщения за одну итерацию страницы.
    # Раньше break после queue.Empty выходил после первого сообщения,
    # из-за чего при быстром завершении потока лог не успевал отобразиться.
    import time
    deadline = time.time() + 0.5  # читаем не дольше 0.5 сек за один rerun
    while time.time() < deadline:
        try:
            msg_type, payload = q.get_nowait()
        except queue.Empty:
            break

        if msg_type == "log":
            st.session_state.om_log_lines.append(payload)
        elif msg_type == "result":
            st.session_state.om_result = payload
        elif msg_type == "error":
            st.session_state.om_error = payload
        elif msg_type == "done":
            st.session_state.om_thread_done = True
            # Дочитываем остаток очереди после done
            while True:
                try:
                    mt2, pl2 = q.get_nowait()
                    if mt2 == "log":
                        st.session_state.om_log_lines.append(pl2)
                    elif mt2 == "result":
                        st.session_state.om_result = pl2
                    elif mt2 == "error":
                        st.session_state.om_error = pl2
                except queue.Empty:
                    break
            if st.session_state.om_error is None:
                st.session_state.om_stage = "done"
            break

    log_container.text("\n".join(st.session_state.om_log_lines[-60:]))

    if st.session_state.om_thread_done:
        if st.session_state.om_error:
            st.error(f"Ошибка: {st.session_state.om_error}")
            if st.button("← Назад"):
                _reset()
        else:
            # Небольшая пауза чтобы лог успел отобразиться перед переходом
            time.sleep(0.1)
            st.rerun()
    else:
        st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# UI — РЕЗУЛЬТАТЫ
# ══════════════════════════════════════════════════════════════════════════════

elif st.session_state.om_stage == "done":

    result = st.session_state.om_result

    st.success("✓ Предобработка завершена")

    # ── Лог ───────────────────────────────────────────────────────────────
    with st.expander("Полный лог выполнения", expanded=False):
        st.text("\n".join(st.session_state.om_log_lines))
        log_text = "\n".join(st.session_state.om_log_lines)
        st.download_button(
            "⬇ Скачать лог",
            data=log_text,
            file_name=f"om_log_{result['dataset_name']}.txt",
            mime="text/plain",
        )

    if result:
        st.divider()
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Тип датасета", result["modality"].upper())
        with col2:
            st.metric("Уверенность", f"{result['modality_conf']*100:.1f}%")
        with col3:
            st.metric("Стратегия", result["strategy"].upper())

        # ── Сравнение метрик изображений ──────────────────────────────────
        st.divider()
        st.subheader("Сравнение метрик изображений")
        st.caption(
            "Метрики измерены на пикселях — не на результатах обучения. "
            "Для оценки влияния на качество модели запусти обучение на созданных датасетах."
        )

        orig = result.get("orig_metrics")
        applied = result.get("applied_metrics", {})

        if orig and applied:
            # Формируем таблицу
            rows = []
            metric_names = [
                ("SNR, dB",      "avg_snr",       "+"),
                ("Контраст",     "avg_contrast",   "+"),
                ("Резкость",     "avg_sharpness",  "+"),
                ("Яркость",      "avg_brightness", "~"),
                ("Размытых, шт", "blur_count",     "-"),
            ]

            for label, attr, good in metric_names:
                orig_val = getattr(orig, attr, None)
                row = {"Метрика": label, "Оригинал": f"{orig_val:.3f}" if isinstance(orig_val, float) else str(orig_val)}
                for level in result["variants"]:
                    m = applied.get(level)
                    if m:
                        val = getattr(m, attr, None)
                        delta = val - orig_val if isinstance(val, (int, float)) and isinstance(orig_val, (int, float)) else None
                        if delta is not None:
                            sign = "▲" if delta > 0 else ("▼" if delta < 0 else "—")
                            row[f"{level}"] = f"{val:.3f} ({sign}{abs(delta):.3f})"
                        else:
                            row[f"{level}"] = str(val)
                    else:
                        row[f"{level}"] = "—"
                rows.append(row)

            import pandas as pd
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True, hide_index=True)

            # Кнопка скачать CSV
            csv_data = df.to_csv(index=False, encoding="utf-8-sig")
            st.download_button(
                "⬇ Скачать таблицу (CSV)",
                data=csv_data,
                file_name=f"om_metrics_{result['dataset_name']}.csv",
                mime="text/csv",
            )

        # ── Созданные датасеты ─────────────────────────────────────────────
        st.divider()
        st.subheader("Созданные датасеты")
        for level, name in result.get("applied_names", {}).items():
            st.code(name)
        st.info(
            "Запусти обучение на этих датасетах в разделе **Обучение** "
            "или используй **2_Предобработка.py** для SHA+SFS сравнения.",
            icon=None,
        )

    st.divider()
    if st.button("← Новый запуск"):
        _reset()
