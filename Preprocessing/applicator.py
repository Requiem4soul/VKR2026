import shutil
from pathlib import Path
from tqdm import tqdm
import cv2
import numpy as np
from typing import Dict, List, Any, Optional

from Preprocessing.methods import PreprocessingMethods
from Data.Datasets.dataset_work import get_dataset_path


def _detect_dataset_type(dataset_path: Path) -> str:
    """
    Определяет тип датасета по структуре папок.
    Возвращает 'yolo' или 'classification'.
    """
    if (dataset_path / 'data.yaml').exists():
        return 'yolo'
    train_path = dataset_path / 'train'
    if train_path.exists():
        if (train_path / 'images').exists():
            return 'yolo'
        subdirs = [d for d in train_path.iterdir() if d.is_dir()]
        if subdirs:
            return 'classification'
    return 'yolo'


def _collect_image_files(split_dir: Path) -> List[Path]:
    """Собирает все файлы изображений из split-папки (YOLO или классификация)."""
    images_dir = split_dir / 'images'
    if images_dir.exists():
        return sorted(
            list(images_dir.glob('*.jpg')) +
            list(images_dir.glob('*.jpeg')) +
            list(images_dir.glob('*.png'))
        )
    # Классификация: рекурсивно из подпапок и самой папки
    files = []
    for ext in ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff'):
        files.extend(split_dir.rglob(ext))
    return sorted(files)


def _create_dst_structure(src_path: Path, dst_path: Path,
                           splits: List[str], dataset_type: str):
    """Создаёт структуру целевой папки, зеркалируя структуру источника."""
    if dataset_type == 'yolo':
        for split in splits:
            (dst_path / split / 'images').mkdir(parents=True, exist_ok=True)
            (dst_path / split / 'labels').mkdir(parents=True, exist_ok=True)
        if (src_path / 'data.yaml').exists():
            try:
                import yaml as _yaml
                with open(src_path / 'data.yaml', 'r', encoding='utf-8') as _f:
                    _yaml_data = _yaml.safe_load(_f)
                # Относительные пути - работают при любом перемещении датасета
                # (схема аналогична LUNA16: пути относительно data.yaml)
                _yaml_data.pop('path', None)
                _yaml_data['train'] = '../train/images'
                _yaml_data['val']   = '../valid/images'
                _yaml_data['test']  = '../test/images'
                with open(dst_path / 'data.yaml', 'w', encoding='utf-8') as _f:
                    _yaml.dump(_yaml_data, _f, default_flow_style=False, allow_unicode=True)
            except Exception:
                shutil.copy(src_path / 'data.yaml', dst_path / 'data.yaml')
    else:
        # Классификация: копируем дерево подпапок (классы)
        for split in splits:
            src_split = src_path / split
            if not src_split.exists():
                continue
            subdirs = [d for d in src_split.iterdir() if d.is_dir()]
            if subdirs:
                for cls_dir in subdirs:
                    (dst_path / split / cls_dir.name).mkdir(parents=True, exist_ok=True)
            else:
                (dst_path / split).mkdir(parents=True, exist_ok=True)
        if (src_path / 'dataset_info.json').exists():
            shutil.copy(src_path / 'dataset_info.json',
                        dst_path / 'dataset_info.json')


def _dst_image_path(src_img: Path, src_split_dir: Path,
                     dst_split_dir: Path) -> Path:
    """
    Вычисляет путь назначения для изображения, сохраняя относительную структуру
    """
    rel = src_img.relative_to(src_split_dir)
    return dst_split_dir / rel


class DatasetPreprocessor:
    """Применяет предобработку к целому датасету."""

    def __init__(self):
        self.methods = PreprocessingMethods()

    def apply_global_preprocessing(
            self,
            source_dataset: str,
            target_dataset: str,
            methods: List[str],
            params: Optional[Dict[str, Dict[str, Any]]] = None,
            splits: List[str] = ['train', 'valid', 'test'],
            target_path: Optional[Path] = None,
    ):
        """
        Применяет одну и ту же предобработку ко всем изображениям.
        """
        src_path = get_dataset_path(source_dataset)
        dst_path = target_path if target_path is not None else get_dataset_path(target_dataset)
        dataset_type = _detect_dataset_type(src_path)

        print(f"\nПрименяем глобальную предобработку:")
        print(f"  Тип датасета: {dataset_type}")
        print(f"  Методы: {', '.join(methods)}")
        print(f"  {source_dataset} -> {target_dataset}")

        if params:
            print(f"  Параметры:")
            for method, method_params in params.items():
                if method in methods:
                    print(f"    {method}: {method_params}")

        if dst_path.exists():
            shutil.rmtree(dst_path)

        _create_dst_structure(src_path, dst_path, splits, dataset_type)

        # Загрузка в VRAM датасетов
        _img_cache: Dict[str, np.ndarray] = {}
        try:
            import psutil
            _avail_gb = psutil.virtual_memory().available / (1024 ** 3)
            _ram_limit_gb = max(1.0, _avail_gb * 0.70)
        except Exception:
            _ram_limit_gb = 2.0

        # Собираем все файлы по всем сплитам и оцениваем размер
        _all_files: List[Path] = []
        for _split in splits:
            _sd = src_path / _split
            if _sd.exists():
                _all_files.extend(_collect_image_files(_sd))

        if _all_files:
            # Оцениваем по первому изображению
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
                                _img = cv2.convertScaleAbs(_img, alpha=255.0 / 65535.0)
                            _img_cache[str(_fp)] = _img
                    print(f"  [RAM-кеш] Закешировано {len(_img_cache)} изображений")
                else:
                    print(f"  [RAM-кеш] Пропуск - датасет {_total_gb:.2f} GB"
                          f" > лимит {_ram_limit_gb:.1f} GB, читаем с диска")

        for split in splits:
            src_split_dir = src_path / split
            dst_split_dir = dst_path / split
            if not src_split_dir.exists():
                print(f"  Пропускаем {split} (не найден)")
                continue

            image_files = _collect_image_files(src_split_dir)
            if not image_files:
                print(f"  Пропускаем {split} (нет изображений)")
                continue

            print(f"  Обрабатываем {split}: {len(image_files)} изображений")

            _params = params.copy() if params else {}
            if 'denoise' in methods and 'denoise' not in _params:
                _params['denoise'] = {'method': 'median'}

            for img_path in tqdm(image_files, desc=f"Processing {split}"):
                # Берём из RAM-кеша если доступен, иначе читаем с диска
                _cached = _img_cache.get(str(img_path))
                if _cached is not None:
                    image = _cached.copy()  # copy чтобы не мутировать кеш
                else:
                    # Читаем в оригинальном формате: цветные датасеты остаются RGB,
                    # grayscale остаются grayscale. IMREAD_GRAYSCALE теряет
                    # цветовую информацию что критично для цветных датасетов.
                    image = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
                    if image is None:
                        continue
                    # Убираем alpha-канал если есть (BGRA -> BGR).
                    if image.ndim == 3 and image.shape[2] == 4:
                        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
                    # Конвертируем 16-битные изображения в 8-битные.
                    if image.dtype == np.uint16:
                        image = cv2.convertScaleAbs(image, alpha=255.0 / 65535.0)
                processed = self.methods.apply_pipeline(image, methods, _params)

                out_path = _dst_image_path(img_path, src_split_dir, dst_split_dir)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(out_path), processed)

                # Копируем label только для YOLO
                if dataset_type == 'yolo':
                    label_src = src_split_dir / 'labels' / (img_path.stem + '.txt')
                    if label_src.exists():
                        label_dst = dst_split_dir / 'labels' / label_src.name
                        shutil.copy(label_src, label_dst)

        print(f"\nГотово! Датасет сохранён в {dst_path}")

        # Освобождаем RAM-кеш
        if _img_cache:
            _img_cache.clear()
