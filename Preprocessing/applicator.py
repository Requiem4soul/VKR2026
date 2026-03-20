"""
Применение предобработки к датасету.
Поддерживает глобальную и адаптивную (кластерную) стратегии обработки.
Поддерживает два формата датасетов:
- YOLO: split/images/*.jpg + split/labels/*.txt + data.yaml
- Классификация: split/class_name/*.jpg (ImageFolder)
"""

import shutil
from pathlib import Path
from tqdm import tqdm
import cv2
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
            # Обновляем пути в data.yaml на абсолютные пути назначения.
            # Относительные пути из оригинала не работают когда датасет
            # скопирован в другую папку (YOLO не найдёт split='val'/'test').
            try:
                import yaml as _yaml
                with open(src_path / 'data.yaml', 'r', encoding='utf-8') as _f:
                    _yaml_data = _yaml.safe_load(_f)
                # Относительные пути — работают при любом перемещении датасета
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
    Вычисляет путь назначения для изображения, сохраняя относительную структуру.
    Для YOLO: dst/split/images/file.jpg
    Для классификации: dst/split/class_name/file.jpg
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
            splits: List[str] = ['train', 'valid', 'test']
    ):
        """
        Применяет одну и ту же предобработку ко всем изображениям.

        Args:
            source_dataset: Название исходного датасета
            target_dataset: Название нового датасета
            methods: Список методов ['denoise', 'contrast_enhancement', ...]
            params: Параметры для методов, например:
                    {
                        'denoise': {'method': 'median', 'ksize': 5},
                        'contrast_enhancement': {'clip_limit': 1.0}
                    }
            splits: Какие splits обрабатывать
        """
        src_path = get_dataset_path(source_dataset)
        dst_path = get_dataset_path(target_dataset)
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
                image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                if image is None:
                    continue
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

    def apply_adaptive_preprocessing(
            self,
            source_dataset: str,
            target_dataset: str,
            clusters: Dict[int, Dict],
            image_metrics: List,
            params: Optional[Dict[str, Dict[str, Any]]] = None,
            splits: List[str] = ['train', 'valid', 'test']
    ):
        """
        Применяет разную предобработку к разным кластерам.

        Args:
            source_dataset: Название исходного датасета
            target_dataset: Название нового датасета
            clusters: Словарь с кластерами и их методами обработки
            image_metrics: Метрики изображений (для автоопределения шума)
            params: Параметры для методов (общие для всех кластеров)
            splits: Какие splits обрабатывать
        """
        src_path = get_dataset_path(source_dataset)
        dst_path = get_dataset_path(target_dataset)
        dataset_type = _detect_dataset_type(src_path)

        print(f"\nПрименяем адаптивную предобработку:")
        print(f"  Тип датасета: {dataset_type}")
        print(f"  Кластеров: {len(clusters)}")

        if dst_path.exists():
            shutil.rmtree(dst_path)

        _create_dst_structure(src_path, dst_path, splits, dataset_type)

        # Обрабатываем train split по кластерам
        src_train_dir = src_path / 'train'
        dst_train_dir = dst_path / 'train'
        image_files = _collect_image_files(src_train_dir)

        for cluster_id, cluster_info in clusters.items():
            cluster_methods = cluster_info['preprocessing']
            indices = cluster_info['image_indices']

            print(f"\nКластер {cluster_id}: {len(indices)} изображений")
            print(f"  Методы: {', '.join(cluster_methods) if cluster_methods else 'нет обработки'}")

            for idx in tqdm(indices, desc=f"Cluster {cluster_id}"):
                img_path = image_files[idx]
                img_metrics = image_metrics[idx]

                image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                if image is None:
                    continue

                if cluster_methods:
                    combined_params = self._build_params_for_image(
                        cluster_methods,
                        img_metrics,
                        params
                    )
                    processed = self.methods.apply_pipeline(image, cluster_methods, combined_params)
                else:
                    processed = image

                out_path = _dst_image_path(img_path, src_train_dir, dst_train_dir)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(out_path), processed)

                # Копируем label только для YOLO
                if dataset_type == 'yolo':
                    label_src = src_train_dir / 'labels' / (img_path.stem + '.txt')
                    if label_src.exists():
                        label_dst = dst_train_dir / 'labels' / label_src.name
                        shutil.copy(label_src, label_dst)

        # Копируем valid и test без изменений
        for split in ['valid', 'test']:
            src_split = src_path / split
            if src_split.exists():
                self._copy_split_as_is(src_path, dst_path, split, dataset_type)

        print(f"\nГотово! Датасет сохранён в {dst_path}")

    def _copy_split_as_is(self, src_path: Path, dst_path: Path,
                           split: str, dataset_type: str = 'yolo'):
        """Копирует split без изменений."""
        if dataset_type == 'yolo':
            for subfolder in ['images', 'labels']:
                src_dir = src_path / split / subfolder
                dst_dir = dst_path / split / subfolder
                if src_dir.exists():
                    dst_dir.mkdir(parents=True, exist_ok=True)
                    for file in src_dir.iterdir():
                        shutil.copy(file, dst_dir / file.name)
        else:
            # Классификация: копируем всё дерево рекурсивно
            src_split_dir = src_path / split
            dst_split_dir = dst_path / split
            if src_split_dir.exists():
                if dst_split_dir.exists():
                    shutil.rmtree(dst_split_dir)
                shutil.copytree(src_split_dir, dst_split_dir)

    def _build_params_for_image(
        self,
        methods: List[str],
        img_metrics,
        global_params: Optional[Dict]
    ) -> Dict:
        """
        Строит параметры с учётом характеристик конкретного изображения.
        Объединяет глобальные параметры из правил с поправками на метрики изображения.
        """
        combined_params = global_params.copy() if global_params else {}

        if 'denoise' in methods:
            if 'denoise' not in combined_params:
                combined_params['denoise'] = {}

            combined_params['denoise']['noise_type'] = img_metrics.noise_type
            combined_params['denoise']['noise_level'] = img_metrics.noise_level

            if 'method' not in combined_params['denoise']:
                noise_to_method = {
                    'gaussian': 'bilateral',
                    'salt_pepper': 'median',
                    'poisson': 'nlm',
                    'speckle': 'median'
                }
                combined_params['denoise']['method'] = noise_to_method.get(
                    img_metrics.noise_type,
                    'median'
                )

        return combined_params
