"""
apply_preprocessing.py
Применяет пайплайн предобработки к датасету — один в один как в основном приложении.

Использование:
    python apply_preprocessing.py --src <исходный_датасет> \
                                   --dst <целевой_датасет> \
                                   --pipeline "Lee (ksize=3) + CLAHE (clip=1.0)"

Пример:
    python apply_preprocessing.py \
        --src 0_SSDD_yolo \
        --dst 0_SSDD_yolo_winner \
        --pipeline "Lee (ksize=3) + CLAHE (clip=1.0)"

Флаг --list: показывает все доступные названия методов (как в логах).
    python apply_preprocessing.py --list

Запускать из корня проекта (рядом с папками Preprocessing/, Data/ и .env).
"""

import argparse
import os
import sys
import shutil
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional


# ── Путь к датасетам из .env ────────────────────────────────────────────────
def _get_datasets_path() -> Path:
    """Читает DATASETS_GLOBAL_PATH из .env файла."""
    env_path = Path(".env")
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            if key.strip() == "DATASETS_GLOBAL_PATH":
                return Path(value.strip().strip('"').strip("'"))
    # Fallback: попробовать переменную окружения
    env_val = os.environ.get("DATASETS_GLOBAL_PATH")
    if env_val:
        return Path(env_val)
    raise EnvironmentError(
        "DATASETS_GLOBAL_PATH не найден. "
        "Проверьте .env файл в корне проекта."
    )


def get_dataset_path(dataset_name: str) -> Path:
    return _get_datasets_path() / dataset_name


# ── Полная таблица кандидатов: display → (methods, params) ──────────────────
# Скопировано ДОСЛОВНО из 6_Объединение.py — гарантирует идентичность параметров.
_CANDIDATES = [
    # ── Шумоподавление ────────────────────────────────────────────────────────
    {"display": "Median (ksize=3)",
     "methods": ["denoise"],
     "params":  {"denoise": {"method": "median", "ksize": 3}}},
    {"display": "Median (ksize=5)",
     "methods": ["denoise"],
     "params":  {"denoise": {"method": "median", "ksize": 5}}},
    {"display": "Gaussian blur (ksize=3)",
     "methods": ["denoise"],
     "params":  {"denoise": {"method": "gaussian", "ksize": 3}}},
    {"display": "Gaussian blur (ksize=5)",
     "methods": ["denoise"],
     "params":  {"denoise": {"method": "gaussian", "ksize": 5}}},
    {"display": "Bilateral (sigma=75)",
     "methods": ["denoise"],
     "params":  {"denoise": {"method": "bilateral", "d": 9,
                             "sigma_color": 75, "sigma_space": 75}}},
    {"display": "Bilateral (sigma=150)",
     "methods": ["denoise"],
     "params":  {"denoise": {"method": "bilateral", "d": 9,
                             "sigma_color": 150, "sigma_space": 150}}},
    {"display": "Wiener (size=3)",
     "methods": ["denoise"],
     "params":  {"denoise": {"method": "wiener", "size": 3}}},
    {"display": "Wiener (size=5)",
     "methods": ["denoise"],
     "params":  {"denoise": {"method": "wiener", "size": 5}}},
    {"display": "Lee (ksize=3)",
     "methods": ["denoise"],
     "params":  {"denoise": {"method": "lee", "ksize": 3}}},
    {"display": "Lee (ksize=5)",
     "methods": ["denoise"],
     "params":  {"denoise": {"method": "lee", "ksize": 5}}},
    # ── Контраст ──────────────────────────────────────────────────────────────
    {"display": "CLAHE (clip=1.0)",
     "methods": ["contrast_enhancement"],
     "params":  {"contrast_enhancement": {"method": "clahe", "clip_limit": 1.0}}},
    {"display": "CLAHE (clip=2.0)",
     "methods": ["contrast_enhancement"],
     "params":  {"contrast_enhancement": {"method": "clahe", "clip_limit": 2.0}}},
    {"display": "CLAHE (clip=3.0)",
     "methods": ["contrast_enhancement"],
     "params":  {"contrast_enhancement": {"method": "clahe", "clip_limit": 3.0}}},
    {"display": "Histogram Equalization",
     "methods": ["contrast_enhancement"],
     "params":  {"contrast_enhancement": {"method": "histogram_eq"}}},
    # ── Яркость ───────────────────────────────────────────────────────────────
    {"display": "Gamma (γ=0.5, осветление)",
     "methods": ["brightness_correction"],
     "params":  {"brightness_correction": {"gamma": 0.5}}},
    {"display": "Gamma (γ=0.8, лёгкое осветление)",
     "methods": ["brightness_correction"],
     "params":  {"brightness_correction": {"gamma": 0.8}}},
    {"display": "Gamma (γ=1.2, лёгкое затемнение)",
     "methods": ["brightness_correction"],
     "params":  {"brightness_correction": {"gamma": 1.2}}},
    # ── Резкость ──────────────────────────────────────────────────────────────
    {"display": "Unsharp Mask (alpha=0.5)",
     "methods": ["sharpening"],
     "params":  {"sharpening": {"method": "unsharp_mask", "alpha": 0.5}}},
    {"display": "Unsharp Mask (alpha=1.0)",
     "methods": ["sharpening"],
     "params":  {"sharpening": {"method": "unsharp_mask", "alpha": 1.0}}},
    {"display": "Unsharp Mask (alpha=1.5)",
     "methods": ["sharpening"],
     "params":  {"sharpening": {"method": "unsharp_mask", "alpha": 1.5}}},
]

# Быстрый поиск по display-имени
_DISPLAY_LOOKUP: Dict[str, Dict] = {c["display"]: c for c in _CANDIDATES}


def parse_pipeline(pipeline_str: str) -> Tuple[List[str], Dict[str, Any]]:
    """
    Разбирает строку пайплайна из лога в (methods, params).

    Реализует merge_methods_params из 6_Объединение.py:
    при повторении одного типа метода (напр. два "denoise") второй получает
    суффикс __1, третий — __2 и т.д. Это обеспечивает идентичность
    с apply_pipeline в methods.py.

    Пример:
        "Lee (ksize=3) + CLAHE (clip=1.0)"
        →  methods = ["denoise", "contrast_enhancement"]
           params  = {"denoise": {"method": "lee", "ksize": 3},
                      "contrast_enhancement": {"method": "clahe", "clip_limit": 1.0}}
    """
    parts = [p.strip() for p in pipeline_str.split("+")]
    merged_methods: List[str] = []
    merged_params:  Dict[str, Any] = {}
    method_counts:  Dict[str, int] = {}

    for part in parts:
        if part not in _DISPLAY_LOOKUP:
            raise ValueError(
                f"Метод '{part}' не найден в таблице кандидатов.\n"
                f"Запустите с флагом --list чтобы увидеть все доступные названия."
            )
        cand = _DISPLAY_LOOKUP[part]
        for base_method in cand["methods"]:
            cnt = method_counts.get(base_method, 0)
            key = base_method if cnt == 0 else f"{base_method}__{cnt}"
            method_counts[base_method] = cnt + 1
            merged_methods.append(key)
            if base_method in cand["params"]:
                merged_params[key] = cand["params"][base_method].copy()

    return merged_methods, merged_params


# ── Вспомогательные функции для работы с датасетом ──────────────────────────
# Скопированы из applicator.py без изменений.

def _detect_dataset_type(dataset_path: Path) -> str:
    if (dataset_path / "data.yaml").exists():
        return "yolo"
    train_path = dataset_path / "train"
    if train_path.exists():
        if (train_path / "images").exists():
            return "yolo"
        subdirs = [d for d in train_path.iterdir() if d.is_dir()]
        if subdirs:
            return "classification"
    return "yolo"


def _collect_image_files(split_dir: Path) -> List[Path]:
    images_dir = split_dir / "images"
    if images_dir.exists():
        return sorted(
            list(images_dir.glob("*.jpg"))
            + list(images_dir.glob("*.jpeg"))
            + list(images_dir.glob("*.png"))
        )
    files = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff"):
        files.extend(split_dir.rglob(ext))
    return sorted(files)


def _dst_image_path(src_img: Path, src_split_dir: Path,
                    dst_split_dir: Path) -> Path:
    rel = src_img.relative_to(src_split_dir)
    return dst_split_dir / rel


def _create_dst_structure(src_path: Path, dst_path: Path,
                           splits: List[str], dataset_type: str):
    """Зеркалирует структуру папок источника — из applicator.py."""
    if dataset_type == "yolo":
        for split in splits:
            (dst_path / split / "images").mkdir(parents=True, exist_ok=True)
            (dst_path / split / "labels").mkdir(parents=True, exist_ok=True)
        if (src_path / "data.yaml").exists():
            try:
                import yaml as _yaml
                with open(src_path / "data.yaml", "r", encoding="utf-8") as _f:
                    _yaml_data = _yaml.safe_load(_f)
                _yaml_data.pop("path", None)
                _yaml_data["train"] = "../train/images"
                _yaml_data["val"]   = "../valid/images"
                _yaml_data["test"]  = "../test/images"
                with open(dst_path / "data.yaml", "w", encoding="utf-8") as _f:
                    _yaml.dump(_yaml_data, _f,
                               default_flow_style=False, allow_unicode=True)
            except Exception:
                shutil.copy(src_path / "data.yaml", dst_path / "data.yaml")
    else:
        for split in splits:
            src_split = src_path / split
            if not src_split.exists():
                continue
            subdirs = [d for d in src_split.iterdir() if d.is_dir()]
            if subdirs:
                for cls_dir in subdirs:
                    (dst_path / split / cls_dir.name).mkdir(
                        parents=True, exist_ok=True)
            else:
                (dst_path / split).mkdir(parents=True, exist_ok=True)
        if (src_path / "dataset_info.json").exists():
            shutil.copy(src_path / "dataset_info.json",
                        dst_path / "dataset_info.json")


# ── Основная функция применения предобработки ───────────────────────────────

def apply_preprocessing(
    source_dataset: str,
    target_dataset: str,
    methods: List[str],
    params: Dict[str, Any],
    splits: List[str] = ("train", "valid", "test"),
):
    """
    Применяет пайплайн предобработки к датасету — идентично
    DatasetPreprocessor.apply_global_preprocessing из applicator.py.
    Разница: импортирует PreprocessingMethods напрямую из проекта.
    """
    import cv2
    import numpy as np
    from tqdm import tqdm

    # Импорт PreprocessingMethods из проекта (гарантирует один в один)
    try:
        from Preprocessing.methods import PreprocessingMethods
    except ImportError:
        print("ОШИБКА: не удалось импортировать Preprocessing.methods.\n"
              "Запустите скрипт из корня проекта (рядом с папкой Preprocessing/).")
        sys.exit(1)

    src_path = get_dataset_path(source_dataset)
    dst_path = get_dataset_path(target_dataset)

    if not src_path.exists():
        print(f"ОШИБКА: Исходный датасет не найден: {src_path}")
        sys.exit(1)

    dataset_type = _detect_dataset_type(src_path)

    print(f"\nПрименяем глобальную предобработку:")
    print(f"  Тип датасета: {dataset_type}")
    print(f"  Методы: {', '.join(methods)}")
    print(f"  {source_dataset} -> {target_dataset}")
    if params:
        print(f"  Параметры:")
        for m_key, m_params in params.items():
            print(f"    {m_key}: {m_params}")

    if dst_path.exists():
        shutil.rmtree(dst_path)

    _create_dst_structure(src_path, dst_path, list(splits), dataset_type)

    # ── RAM-кеш (из applicator.py) ────────────────────────────────────────────
    _img_cache: Dict[str, np.ndarray] = {}
    try:
        import psutil
        _avail_gb = psutil.virtual_memory().available / (1024 ** 3)
        _ram_limit_gb = max(1.0, _avail_gb * 0.70)
    except Exception:
        _ram_limit_gb = 2.0

    _all_files: List[Path] = []
    for _split in splits:
        _sd = src_path / _split
        if _sd.exists():
            _all_files.extend(_collect_image_files(_sd))

    if _all_files:
        _sample = cv2.imread(str(_all_files[0]), cv2.IMREAD_UNCHANGED)
        if _sample is not None:
            _bytes_per_img = _sample.nbytes
            _total_gb = (_bytes_per_img * len(_all_files)) / (1024 ** 3)
            if _total_gb <= _ram_limit_gb:
                print(f"  [RAM-кеш] Загружаем {len(_all_files)} изображений"
                      f" (~{_total_gb:.2f} GB, лимит {_ram_limit_gb:.1f} GB)...")
                for _fp in tqdm(_all_files, desc="Кеширование в RAM"):
                    _img = cv2.imread(str(_fp), cv2.IMREAD_UNCHANGED)
                    if _img is not None:
                        if _img.ndim == 3 and _img.shape[2] == 4:
                            _img = cv2.cvtColor(_img, cv2.COLOR_BGRA2BGR)
                        if _img.dtype == np.uint16:
                            _img = cv2.convertScaleAbs(_img,
                                                        alpha=255.0 / 65535.0)
                        _img_cache[str(_fp)] = _img

    # ── Обработка по сплитам ──────────────────────────────────────────────────
    preprocessor = PreprocessingMethods()

    for split in splits:
        src_split_dir = src_path / split
        dst_split_dir = dst_path / split
        if not src_split_dir.exists():
            continue

        image_files = _collect_image_files(src_split_dir)
        if not image_files:
            continue

        print(f"  Обрабатываем {split}: {len(image_files)} изображений")

        # Параметры по умолчанию (из applicator.py строки 211-212)
        _params = {k: v.copy() for k, v in params.items()}
        if "denoise" in methods and "denoise" not in _params:
            _params["denoise"] = {"method": "median"}

        for img_path in tqdm(image_files, desc=f"Processing {split}"):
            _cached = _img_cache.get(str(img_path))
            if _cached is not None:
                image = _cached.copy()
            else:
                image = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
                if image is None:
                    continue
                if image.ndim == 3 and image.shape[2] == 4:
                    image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
                if image.dtype == np.uint16:
                    image = cv2.convertScaleAbs(image, alpha=255.0 / 65535.0)

            processed = preprocessor.apply_pipeline(image, methods, _params)

            out_path = _dst_image_path(img_path, src_split_dir, dst_split_dir)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_path), processed)

            # Копируем labels для YOLO
            if dataset_type == "yolo":
                label_src = (src_split_dir / "labels"
                             / (img_path.stem + ".txt"))
                if label_src.exists():
                    label_dst = dst_split_dir / "labels" / label_src.name
                    shutil.copy(label_src, label_dst)

    print(f"\nГотово! Датасет сохранён в {dst_path}")

    if _img_cache:
        _img_cache.clear()


# ── Точка входа ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Применяет пайплайн предобработки к датасету (один в один с основным приложением).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  python apply_preprocessing.py \\
      --src 0_SSDD_yolo \\
      --dst 0_SSDD_yolo_winner \\
      --pipeline "Lee (ksize=3) + CLAHE (clip=1.0)"

  python apply_preprocessing.py \\
      --src 0_Brain_Tumor \\
      --dst 0_Brain_Tumor_gauss_clahe \\
      --pipeline "Gaussian blur (ksize=3) + CLAHE (clip=2.0)"

  # Одиночный метод:
  python apply_preprocessing.py \\
      --src 0_LUNA16 \\
      --dst 0_LUNA16_lee \\
      --pipeline "Lee (ksize=3)"

  # Просмотр всех доступных методов:
  python apply_preprocessing.py --list
        """,
    )
    parser.add_argument("--src",  type=str,
                        help="Название исходного датасета (как в DATASETS_GLOBAL_PATH)")
    parser.add_argument("--dst",  type=str,
                        help="Название целевого датасета (будет создан)")
    parser.add_argument("--pipeline", type=str,
                        help='Строка пайплайна, например "Lee (ksize=3) + CLAHE (clip=1.0)"')
    parser.add_argument("--splits", type=str, default="train,valid,test",
                        help='Сплиты через запятую (по умолчанию: train,valid,test)')
    parser.add_argument("--list",  action="store_true",
                        help="Показать все доступные названия методов и выйти")
    args = parser.parse_args()

    if args.list:
        print("\nДоступные методы предобработки (используйте точно такие же строки):\n")
        groups = {
            "Шумоподавление":    [c for c in _CANDIDATES if "denoise" in c["methods"]],
            "Контраст":          [c for c in _CANDIDATES if "contrast_enhancement" in c["methods"]],
            "Яркость":           [c for c in _CANDIDATES if "brightness_correction" in c["methods"]],
            "Резкость":          [c for c in _CANDIDATES if "sharpening" in c["methods"]],
        }
        for group_name, candidates in groups.items():
            print(f"  {group_name}:")
            for c in candidates:
                print(f"    \"{c['display']}\"")
                print(f"      → {c['methods']}  params={c['params']}")
        print()
        return

    if not args.src or not args.dst or not args.pipeline:
        parser.print_help()
        sys.exit(1)

    # Парсим строку пайплайна
    print(f"\nПайплайн: {args.pipeline}")
    try:
        methods, params = parse_pipeline(args.pipeline)
    except ValueError as e:
        print(f"\nОШИБКА при парсинге пайплайна: {e}")
        sys.exit(1)

    print(f"  Преобразовано в:")
    print(f"    methods = {methods}")
    print(f"    params  = {params}")

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]

    apply_preprocessing(
        source_dataset=args.src,
        target_dataset=args.dst,
        methods=methods,
        params=params,
        splits=splits,
    )


if __name__ == "__main__":
    main()
