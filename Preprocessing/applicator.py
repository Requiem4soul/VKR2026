"""
Применение предобработки к датасету.
Поддерживает глобальную и адаптивную (кластерную) стратегии обработки.
"""

import shutil
from pathlib import Path
from tqdm import tqdm
import cv2
from typing import Dict, List, Any, Optional

from Preprocessing.methods import PreprocessingMethods
from Data.Datasets.dataset_work import get_dataset_path


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
        Применяет одну и ту же предобработку ко всем изображениям
        
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

        print(f"\nПрименяем глобальную предобработку:")
        print(f"  Методы: {', '.join(methods)}")
        print(f"  {source_dataset} → {target_dataset}")
        
        if params:
            print(f"  Параметры:")
            for method, method_params in params.items():
                if method in methods:
                    print(f"     {method}: {method_params}")

        # Удаляем старую версию если есть
        if dst_path.exists():
            shutil.rmtree(dst_path)

        # Создаём структуру папок
        for split in splits:
            for subfolder in ['images', 'labels']:
                (dst_path / split / subfolder).mkdir(parents=True, exist_ok=True)

        # Копируем data.yaml
        if (src_path / 'data.yaml').exists():
            shutil.copy(src_path / 'data.yaml', dst_path / 'data.yaml')

        # Обрабатываем каждый split
        for split in splits:
            images_dir = src_path / split / 'images'
            if not images_dir.exists():
                print(f"  Пропускаем {split} (не найден)")
                continue

            self._process_split_global(
                src_path, dst_path, split, methods, params
            )

        print(f"\n Готово! Датасет сохранён в {dst_path}")

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
        Применяет разную предобработку к разным кластерам
        
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

        print(f"\nПрименяем адаптивную предобработку:")
        print(f"  Кластеров: {len(clusters)}")

        # Удаляем старую версию
        if dst_path.exists():
            shutil.rmtree(dst_path)

        # Создаём структуру
        for split in splits:
            for subfolder in ['images', 'labels']:
                (dst_path / split / subfolder).mkdir(parents=True, exist_ok=True)

        # Копируем data.yaml
        if (src_path / 'data.yaml').exists():
            shutil.copy(src_path / 'data.yaml', dst_path / 'data.yaml')

        # Обрабатываем train split по кластерам
        images_dir = src_path / 'train' / 'images'
        image_files = sorted(list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png")))

        # Обрабатываем по кластерам
        for cluster_id, cluster_info in clusters.items():
            methods = cluster_info['preprocessing']
            indices = cluster_info['image_indices']

            print(f"\nКластер {cluster_id}: {len(indices)} изображений")
            print(f"  Методы: {', '.join(methods) if methods else 'нет обработки'}")

            for idx in tqdm(indices, desc=f"Cluster {cluster_id}"):
                img_path = image_files[idx]
                img_metrics = image_metrics[idx]

                # Загружаем и обрабатываем
                image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)

                if methods:
                    combined_params = self._build_params_for_image(
                        methods, 
                        img_metrics, 
                        params
                    )
                    
                    processed = self.methods.apply_pipeline(image, methods, combined_params)
                else:
                    processed = image

                # Сохраняем
                target_img = dst_path / 'train' / 'images' / img_path.name
                cv2.imwrite(str(target_img), processed)

                # Копируем label
                label_src = src_path / 'train' / 'labels' / (img_path.stem + '.txt')
                label_dst = dst_path / 'train' / 'labels' / (img_path.stem + '.txt')
                if label_src.exists():
                    shutil.copy(label_src, label_dst)

        # Копируем valid и test без изменений
        for split in ['valid', 'test']:
            if split == 'train':
                continue

            src_split = src_path / split
            if src_split.exists():
                self._copy_split_as_is(src_path, dst_path, split)

        print(f"\n Готово! Датасет сохранён в {dst_path}")

    def _process_split_global(
            self,
            src_path: Path,
            dst_path: Path,
            split: str,
            methods: List[str],
            params: Optional[Dict]
    ):
        """Обрабатывает один split с глобальными параметрами"""
        images_dir = src_path / split / 'images'
        labels_dir = src_path / split / 'labels'

        image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))

        for img_path in tqdm(image_files, desc=f"Processing {split}"):
            # Загружаем
            image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)

            if params is None:
                params = {}

            if 'denoise' in methods and 'denoise' not in params:
                params['denoise'] = {'method': 'median'}

            processed = self.methods.apply_pipeline(image, methods, params)

            # Сохраняем
            target_img = dst_path / split / 'images' / img_path.name
            cv2.imwrite(str(target_img), processed)

            # Копируем label
            label_path = labels_dir / (img_path.stem + '.txt')
            target_label = dst_path / split / 'labels' / (img_path.stem + '.txt')

            if label_path.exists():
                shutil.copy(label_path, target_label)

    def _copy_split_as_is(self, src_path: Path, dst_path: Path, split: str):
        """Копирует split без изменений"""
        for subfolder in ['images', 'labels']:
            src_dir = src_path / split / subfolder
            dst_dir = dst_path / split / subfolder

            if src_dir.exists():
                for file in src_dir.iterdir():
                    shutil.copy(file, dst_dir / file.name)
    
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
        
        # Для denoise объединяем параметры правильно
        if 'denoise' in methods:
            if 'denoise' not in combined_params:
                combined_params['denoise'] = {}
            
            # Добавляем тип шума из метрик изображения
            combined_params['denoise']['noise_type'] = img_metrics.noise_type
            combined_params['denoise']['noise_level'] = img_metrics.noise_level
            
            # ВАЖНО: Если в global_params уже есть 'method', НЕ перезаписываем
            # Иначе выбираем автоматически на основе типа шума
            if 'method' not in combined_params['denoise']:
                # Автоматический выбор метода на основе типа шума
                noise_to_method = {
                    'gaussian': 'bilateral',
                    'salt_pepper': 'median',
                    'poisson': 'nlm',
                    'speckle': 'median'
                }
                combined_params['denoise']['method'] = noise_to_method.get(
                    img_metrics.noise_type, 
                    'median'  # Безопасный дефолт
                )
            # Если метод уже указан в global_params - используем его
        
        return combined_params
