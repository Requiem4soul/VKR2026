import os
import json
import gc
import torch
import time
from datetime import datetime
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import yaml

# Импорты для разных моделей
from ultralytics import YOLO
import torchvision
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    FasterRCNN_ResNet50_FPN_Weights,
    retinanet_resnet50_fpn,
    RetinaNet_ResNet50_FPN_Weights
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.retinanet import RetinaNetHead
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import cv2
import numpy as np

from Data.Datasets.dataset_work import get_dataset_path


# ===================== YOLO DATASET INFO =====================

class YOLODatasetInfo:
    """Класс для извлечения информации из YOLO датасета"""

    @staticmethod
    def get_num_classes(dataset_path: Path) -> int:
        """Автоматически определяет количество классов из data.yaml"""
        yaml_path = dataset_path / "data.yaml"

        if not yaml_path.exists():
            raise FileNotFoundError(f"data.yaml не найден в {dataset_path}")

        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)

        if 'names' in data:
            if isinstance(data['names'], dict):
                num_classes = len(data['names'])
            elif isinstance(data['names'], list):
                num_classes = len(data['names'])
            else:
                raise ValueError("Неверный формат 'names' в data.yaml")
        elif 'nc' in data:
            num_classes = data['nc']
        else:
            raise ValueError("data.yaml должен содержать 'names' или 'nc'")

        print(f"[INFO] Обнаружено классов в датасете: {num_classes}")
        return num_classes

    @staticmethod
    def get_class_names(dataset_path: Path) -> List[str]:
        """Получает названия классов из data.yaml"""
        yaml_path = dataset_path / "data.yaml"

        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)

        if 'names' in data:
            if isinstance(data['names'], dict):
                return [data['names'][i] for i in sorted(data['names'].keys())]
            elif isinstance(data['names'], list):
                return data['names']

        return [f"class_{i}" for i in range(YOLODatasetInfo.get_num_classes(dataset_path))]


# ===================== DATASET CONVERTER =====================

class YOLOToFasterRCNNDataset(Dataset):
    """Конвертер YOLO датасета для Faster R-CNN и RetinaNet"""

    def __init__(self, dataset_path: Path, split: str = 'train', transforms=None):
        self.dataset_path = dataset_path
        self.split = split
        self.transforms = transforms

        self.images_dir = dataset_path / split / "images"
        self.labels_dir = dataset_path / split / "labels"

        self.image_files = sorted(list(self.images_dir.glob("*.jpg")))

        print(f"[INFO] Загружено {len(self.image_files)} изображений из {split}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        height, width = image.shape[:2]

        label_path = self.labels_dir / f"{img_path.stem}.txt"

        boxes = []
        labels = []

        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        x_center = float(parts[1]) * width
                        y_center = float(parts[2]) * height
                        w = float(parts[3]) * width
                        h = float(parts[4]) * height

                        x_min = x_center - w / 2
                        y_min = y_center - h / 2
                        x_max = x_center + w / 2
                        y_max = y_center + h / 2

                        if x_max > x_min and y_max > y_min:
                            boxes.append([x_min, y_min, x_max, y_max])
                            labels.append(class_id + 1)

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx])
        }

        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        if self.transforms:
            image, target = self.transforms(image, target)

        return image, target


def compute_detection_metrics(model, dataloader, device, iou_threshold=0.5):
    """
    Универсальная функция расчета метрик детекции для PyTorch моделей
    Возвращает mAP, precision, recall
    """
    from collections import defaultdict

    model.to(device)
    model.eval()

    all_predictions = []
    all_targets = []

    print(f"[METRICS] Расчет метрик детекции...")

    with torch.no_grad():
        for images, targets in dataloader:
            images = [img.to(device) for img in images]

            # Получаем предсказания
            try:
                predictions = model(images)
            except Exception as e:
                print(f"[METRICS] Ошибка при inference: {e}")
                continue

            # Переносим обратно на CPU
            predictions = [{k: v.cpu() for k, v in pred.items()} for pred in predictions]
            targets = [{k: v.cpu() for k, v in t.items()} for t in targets]

            all_predictions.extend(predictions)
            all_targets.extend(targets)

    if len(all_predictions) == 0 or len(all_targets) == 0:
        print(f"[METRICS] Нет предсказаний или targets!")
        return {
            'precision': 0.0,
            'recall': 0.0,
            'mAP50': 0.0,
            'f1': 0.0,
            'avg_iou': 0.0
        }

    # Расчет метрик
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    total_iou = 0
    iou_count = 0

    for pred, target in zip(all_predictions, all_targets):
        pred_boxes = pred.get('boxes', torch.tensor([]))
        pred_labels = pred.get('labels', torch.tensor([]))
        pred_scores = pred.get('scores', torch.tensor([]))

        target_boxes = target.get('boxes', torch.tensor([]))
        target_labels = target.get('labels', torch.tensor([]))

        # Проверка на пустые тензоры
        if len(pred_boxes) == 0 and len(target_boxes) == 0:
            continue

        if len(target_boxes) > 0 and len(pred_boxes) == 0:
            false_negatives += len(target_boxes)
            continue

        # Фильтруем предсказания по score threshold
        score_threshold = 0.5
        if len(pred_scores) > 0:
            keep = pred_scores > score_threshold
            pred_boxes = pred_boxes[keep]
            pred_labels = pred_labels[keep]
            pred_scores = pred_scores[keep]

        if len(pred_boxes) == 0:
            if len(target_boxes) > 0:
                false_negatives += len(target_boxes)
            continue

        matched_targets = set()

        # Для каждого предсказания ищем соответствующий GT
        for pred_box, pred_label in zip(pred_boxes, pred_labels):
            best_iou = 0
            best_target_idx = -1

            for target_idx, (target_box, target_label) in enumerate(zip(target_boxes, target_labels)):
                if target_idx in matched_targets:
                    continue

                if pred_label != target_label:
                    continue

                # Расчет IoU
                iou = compute_iou(pred_box, target_box)

                if iou > best_iou:
                    best_iou = iou
                    best_target_idx = target_idx

            if best_iou >= iou_threshold and best_target_idx != -1:
                true_positives += 1
                matched_targets.add(best_target_idx)
                total_iou += best_iou
                iou_count += 1
            else:
                false_positives += 1

        # Непокрытые ground truth - это false negatives
        if len(target_boxes) > 0:
            false_negatives += len(target_boxes) - len(matched_targets)

    # Расчет метрик
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    avg_iou = total_iou / iou_count if iou_count > 0 else 0.0

    # mAP50 = средний IoU при пороге 0.5 (упрощенная версия)
    mAP50 = avg_iou if avg_iou > 0 else 0.0

    print(f"[METRICS] Готово! TP={true_positives}, FP={false_positives}, FN={false_negatives}")
    print(f"[METRICS] Precision: {precision:.4f}, Recall: {recall:.4f}, mAP50: {mAP50:.4f}")

    return {
        'precision': float(precision),
        'recall': float(recall),
        'mAP50': float(mAP50),
        'f1': float(f1),
        'avg_iou': float(avg_iou)
    }


def compute_iou(box1, box2):
    """Расчет IoU между двумя bbox"""
    # Конвертируем в numpy если это тензоры
    if torch.is_tensor(box1):
        box1 = box1.numpy()
    if torch.is_tensor(box2):
        box2 = box2.numpy()

    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    # Площадь пересечения
    inter_xmin = max(x1_min, x2_min)
    inter_ymin = max(y1_min, y2_min)
    inter_xmax = min(x1_max, x2_max)
    inter_ymax = min(y1_max, y2_max)

    if inter_xmax < inter_xmin or inter_ymax < inter_ymin:
        return 0.0

    inter_area = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)

    # Площадь объединения
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)

    union_area = box1_area + box2_area - inter_area

    iou = inter_area / union_area if union_area > 0 else 0.0

    return float(iou)


# ===================== BASE MODEL WRAPPER =====================

class BaseModelWrapper(ABC):
    """Базовый класс для обертки над моделями"""

    def __init__(self, task_type: str):
        self.task_type = task_type
        self.model = None

    @abstractmethod
    def initialize(self, num_classes: int = None, **kwargs):
        pass

    @abstractmethod
    def train_epoch(self, dataloader, device, optimizer, epoch, **kwargs):
        pass

    @abstractmethod
    def validate(self, dataloader, device, **kwargs):
        pass

    @abstractmethod
    def save(self, path: str):
        pass

    @abstractmethod
    def load(self, path: str):
        pass


# ===================== YOLO WRAPPER =====================

class YOLOWrapper(BaseModelWrapper):
    """Обертка для YOLO моделей"""

    def __init__(self, model_size: str = 'n'):
        super().__init__(task_type='detection')
        self.model_size = model_size
        self.results = None

    def initialize(self, num_classes: int = None, **kwargs):
        self.model = YOLO(f'yolov8{self.model_size}.pt')
        print(f"[✓] Инициализирована YOLOv8{self.model_size}")

    def train_epoch(self, dataset_path: Path, epochs: int, device, **kwargs):
        yaml_path = dataset_path / "data.yaml"

        self.results = self.model.train(
            data=str(yaml_path),
            epochs=epochs,
            imgsz=kwargs.get('imgsz', 640),
            batch=kwargs.get('batch', 16),
            device=device,
            save=False,
            project=kwargs.get('project', 'runs'),
            name=kwargs.get('name', 'exp'),
            exist_ok=True,
            workers=1,
            cache=False,
            verbose=False
        )

        return self.extract_metrics()

    def validate(self, dataloader, device, **kwargs):
        return self.extract_metrics()

    def extract_metrics(self) -> Dict[str, float]:
        try:
            return {
                'precision': float(self.results.results_dict.get('metrics/precision(B)', 0)),
                'recall': float(self.results.results_dict.get('metrics/recall(B)', 0)),
                'mAP50': float(self.results.results_dict.get('metrics/mAP50(B)', 0)),
                'mAP50-95': float(self.results.results_dict.get('metrics/mAP50-95(B)', 0)),
                'train_loss': float(self.results.results_dict.get('train/box_loss', 0)),
                'val_loss': float(self.results.results_dict.get('val/box_loss', 0))
            }
        except:
            return {
                'precision': 0.0, 'recall': 0.0, 'mAP50': 0.0,
                'mAP50-95': 0.0, 'train_loss': 0.0, 'val_loss': 0.0
            }

    def save(self, path: str):
        self.model.save(path)

    def load(self, path: str):
        self.model = YOLO(path)


# ===================== FASTER R-CNN WRAPPER =====================

class FasterRCNNWrapper(BaseModelWrapper):
    """Обертка для Faster R-CNN"""

    def __init__(self, pretrained: bool = True):
        super().__init__(task_type='detection')
        self.pretrained = pretrained
        self.num_classes = None

    def initialize(self, num_classes: int, **kwargs):
        self.num_classes = num_classes + 1

        if self.pretrained:
            weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
            self.model = fasterrcnn_resnet50_fpn(weights=weights)
        else:
            self.model = fasterrcnn_resnet50_fpn(weights=None)

        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.num_classes)

        print(f"[✓] Инициализирован Faster R-CNN (classes={self.num_classes}, включая фон)")

    def train_epoch(self, dataloader, device, optimizer, epoch, **kwargs):
        self.model.to(device)
        self.model.train()

        total_loss = 0
        num_batches = 0

        for images, targets in dataloader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = self.model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            total_loss += losses.item()
            num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        return {'train_loss': avg_loss}

    def validate(self, dataloader, device, **kwargs):
        """Валидация с расчетом метрик"""
        # Расчет loss
        self.model.to(device)
        self.model.eval()

        total_loss = 0
        num_batches = 0

        with torch.no_grad():
            for images, targets in dataloader:
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                self.model.train()
                loss_dict = self.model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                self.model.eval()

                total_loss += losses.item()
                num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0

        # Расчет метрик детекции
        metrics = compute_detection_metrics(self.model, dataloader, device)

        return {
            'val_loss': avg_loss,
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'mAP50': metrics['mAP50'],
            'f1': metrics['f1']
        }

    def save(self, path: str):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'num_classes': self.num_classes
        }, path)

    def load(self, path: str):
        checkpoint = torch.load(path)
        self.num_classes = checkpoint['num_classes']

        if self.pretrained:
            weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
            self.model = fasterrcnn_resnet50_fpn(weights=weights)
        else:
            self.model = fasterrcnn_resnet50_fpn(weights=None)

        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.num_classes)

        self.model.load_state_dict(checkpoint['model_state_dict'])


# ===================== RETINANET WRAPPER =====================

class RetinaNetWrapper(BaseModelWrapper):
    """Обертка для RetinaNet с Focal Loss"""

    def __init__(self, pretrained: bool = True):
        super().__init__(task_type='detection')
        self.pretrained = pretrained
        self.num_classes = None

    def initialize(self, num_classes: int, **kwargs):
        self.num_classes = num_classes + 1

        if self.pretrained:
            weights = RetinaNet_ResNet50_FPN_Weights.DEFAULT
            self.model = retinanet_resnet50_fpn(weights=weights)
        else:
            self.model = retinanet_resnet50_fpn(weights=None)

        num_anchors = self.model.head.classification_head.num_anchors

        self.model.head = RetinaNetHead(
            in_channels=self.model.backbone.out_channels,
            num_anchors=num_anchors,
            num_classes=self.num_classes
        )

        print(f"Инициализирован RetinaNet (classes={self.num_classes}, включая фон)")
        print(f"    Использует Focal Loss для борьбы с дисбалансом классов")

    def train_epoch(self, dataloader, device, optimizer, epoch, **kwargs):
        self.model.to(device)
        self.model.train()

        total_loss = 0
        num_batches = 0
        nan_count = 0

        for batch_idx, (images, targets) in enumerate(dataloader):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            try:
                loss_dict = self.model(images, targets)

                cls_loss = loss_dict.get('classification', torch.tensor(0.0, device=device))
                box_loss = loss_dict.get('bbox_regression', torch.tensor(0.0, device=device))
                losses = cls_loss + box_loss

                # Проверка на NaN ДО backward
                if torch.isnan(losses) or torch.isinf(losses):
                    print(f"[WARNING] NaN/Inf в loss на батче {batch_idx}, пропускаем")
                    nan_count += 1
                    if nan_count > 10:
                        print(f"[ERROR] Слишком много NaN ({nan_count}), останавливаем эпоху")
                        break
                    continue

                optimizer.zero_grad()
                losses.backward()

                # КРИТИЧЕСКИ ВАЖНО: Gradient Clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                optimizer.step()

                total_loss += losses.item()
                num_batches += 1

            except RuntimeError as e:
                print(f"[WARNING] RuntimeError на батче {batch_idx}: {e}")
                nan_count += 1
                continue

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

        if nan_count > 0:
            print(f"[INFO] Пропущено {nan_count} батчей из-за NaN/Inf")

        return {'train_loss': avg_loss}

    def validate(self, dataloader, device, **kwargs):
        """Валидация с расчетом метрик"""
        self.model.to(device)
        self.model.eval()

        total_loss = 0
        num_batches = 0

        # Расчет loss БЕЗ градиентов
        with torch.no_grad():
            for images, targets in dataloader:
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                # ВАЖНО: НЕ переключаем в train mode для валидации!
                # RetinaNet может вернуть loss в eval mode
                try:
                    outputs = self.model(images, targets)

                    # Если модель в eval mode, она возвращает predictions, а не loss
                    # Поэтому временно переключаем в train
                    self.model.train()
                    loss_dict = self.model(images, targets)
                    self.model.eval()

                    cls_loss = loss_dict.get('classification', torch.tensor(0.0))
                    box_loss = loss_dict.get('bbox_regression', torch.tensor(0.0))
                    losses = cls_loss + box_loss

                    # Проверка на NaN
                    if torch.isnan(losses) or torch.isinf(losses):
                        print(f"[WARNING] NaN или Inf в loss, пропускаем батч")
                        continue

                    total_loss += losses.item()
                    num_batches += 1

                except Exception as e:
                    print(f"[WARNING] Ошибка при расчете loss: {e}")
                    continue

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

        # Расчет метрик детекции
        try:
            metrics = compute_detection_metrics(self.model, dataloader, device)
        except Exception as e:
            print(f"[WARNING] Ошибка при расчете метрик: {e}")
            metrics = {
                'precision': 0.0,
                'recall': 0.0,
                'mAP50': 0.0,
                'f1': 0.0
            }

        return {
            'val_loss': avg_loss,
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'mAP50': metrics['mAP50'],
            'f1': metrics['f1']
        }

    def save(self, path: str):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'num_classes': self.num_classes
        }, path)

    def load(self, path: str):
        checkpoint = torch.load(path)
        self.num_classes = checkpoint['num_classes']

        if self.pretrained:
            weights = RetinaNet_ResNet50_FPN_Weights.DEFAULT
            self.model = retinanet_resnet50_fpn(weights=weights)
        else:
            self.model = retinanet_resnet50_fpn(weights=None)

        num_anchors = self.model.head.classification_head.num_anchors
        self.model.head = RetinaNetHead(
            in_channels=self.model.backbone.out_channels,
            num_anchors=num_anchors,
            num_classes=self.num_classes
        )

        self.model.load_state_dict(checkpoint['model_state_dict'])


# ===================== UNIVERSAL TRAINER =====================

class UniversalModelTrainer:
    """Поэтапное обучение моделей с поиском оптимальной предобработки"""

    def __init__(
            self,
            model_configs: List[Dict[str, Any]],
            dataset_names: List[str],
            max_epochs: int = 40,
            checkpoint_interval: int = 10,
            clean_old_results: bool = False
    ):
        self.model_configs = model_configs
        self.dataset_names = dataset_names
        self.max_epochs = max_epochs
        self.checkpoint_interval = checkpoint_interval

        self.models = {}
        self.metrics_history = {}
        self.current_epochs = {}
        self.dataloaders = {}

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        models_str = "_".join([m['name'] for m in model_configs])
        self.results_dir = f"results_{models_str}_{timestamp}"
        os.makedirs(self.results_dir, exist_ok=True)

        self.results_file = os.path.join(self.results_dir, f"training_results_{timestamp}.txt")
        self.checkpoint_dir = os.path.join(self.results_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        for model_config in model_configs:
            model_name = model_config['name']
            for dataset_name in dataset_names:
                key = f"{model_name}_{dataset_name}"
                self.models[key] = None
                self.metrics_history[key] = []
                self.current_epochs[key] = 0

        if clean_old_results:
            self.clean_old_runs()

        self.log_message("=== НАЧАЛО ОБУЧЕНИЯ ===")
        self.log_message(f"Модели: {[m['name'] for m in model_configs]}")
        self.log_message(f"Датасеты: {', '.join(dataset_names)}")
        self.log_message(f"Максимальное количество эпох: {max_epochs}")
        self.log_message(f"Интервал проверки: {checkpoint_interval} эпох")
        self.log_message("=" * 50)

    def clean_old_runs(self):
        import shutil

        if Path("runs").exists():
            shutil.rmtree("runs")
            self.log_message("[CLEAN] Удалена папка runs/")

        for old_results in Path(".").glob("results_*"):
            if old_results.is_dir() and old_results != Path(self.results_dir):
                shutil.rmtree(old_results)
                self.log_message(f"[CLEAN] Удалена папка {old_results}")

    def log_message(self, message):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        full_message = f"[{timestamp}] {message}"
        print(full_message)

        with open(self.results_file, 'a', encoding='utf-8') as f:
            f.write(full_message + '\n')

    def collate_fn(self, batch):
        return tuple(zip(*batch))

    def create_dataloaders(self, dataset_path: Path, model_type: str, batch_size: int = 16):
        if model_type == 'yolo':
            return None

        elif model_type in ['faster_rcnn', 'retinanet']:
            train_dataset = YOLOToFasterRCNNDataset(dataset_path, split='train')
            val_dataset = YOLOToFasterRCNNDataset(dataset_path, split='valid')

            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=0,
                collate_fn=self.collate_fn
            )

            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,
                collate_fn=self.collate_fn
            )

            return {'train': train_loader, 'val': val_loader}

        return None

    def initialize_model(self, model_config: Dict[str, Any], dataset_path: Path) -> BaseModelWrapper:
        model_type = model_config['type']

        num_classes = YOLODatasetInfo.get_num_classes(dataset_path)
        class_names = YOLODatasetInfo.get_class_names(dataset_path)

        self.log_message(f"Классы датасета: {class_names}")

        if model_type == 'yolo':
            wrapper = YOLOWrapper(model_size=model_config.get('size', 'n'))
            wrapper.initialize(num_classes=num_classes)

        elif model_type == 'faster_rcnn':
            wrapper = FasterRCNNWrapper(pretrained=model_config.get('pretrained', True))
            wrapper.initialize(num_classes=num_classes)

        elif model_type == 'retinanet':
            wrapper = RetinaNetWrapper(pretrained=model_config.get('pretrained', True))
            wrapper.initialize(num_classes=num_classes)

        else:
            raise ValueError(f"Неподдерживаемый тип модели: {model_type}")

        return wrapper

    def train_model_segment(
            self,
            model_config: Dict[str, Any],
            dataset_name: str,
            start_epoch: int,
            end_epoch: int
    ):
        model_name = model_config['name']
        model_type = model_config['type']
        key = f"{model_name}_{dataset_name}"

        self.log_message(
            f"\n--- Обучение {model_name} на {dataset_name}: "
            f"эпохи {start_epoch + 1}-{end_epoch} ---"
        )

        dataset_path = get_dataset_path(dataset_name)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        try:
            if start_epoch == 0:
                self.models[key] = self.initialize_model(model_config, dataset_path)
            else:
                checkpoint_path = os.path.join(
                    self.checkpoint_dir,
                    f"{key}_epoch_{start_epoch}.pt"
                )
                if os.path.exists(checkpoint_path):
                    self.models[key] = self.initialize_model(model_config, dataset_path)
                    self.models[key].load(checkpoint_path)
                    self.log_message(f"Загружен чекпоинт с эпохи {start_epoch}")
                else:
                    self.log_message(f"ОШИБКА: Чекпоинт не найден: {checkpoint_path}")
                    return None

            epochs_to_train = end_epoch - start_epoch

            if model_type == 'yolo':
                metrics = self.models[key].train_epoch(
                    dataset_path=dataset_path,
                    epochs=epochs_to_train,
                    device=device,
                    batch=16,
                    imgsz=640
                )

            elif model_type in ['faster_rcnn', 'retinanet']:
                if key not in self.dataloaders:
                    self.dataloaders[key] = self.create_dataloaders(
                        dataset_path,
                        model_type,
                        batch_size=4
                    )

                train_loader = self.dataloaders[key]['train']
                val_loader = self.dataloaders[key]['val']

                optimizer = optim.SGD(
                    self.models[key].model.parameters(),
                    lr=0.005,
                    momentum=0.9,
                    weight_decay=0.0005
                )

                all_metrics = {'train_loss': 0, 'val_loss': 0}

                for epoch in range(epochs_to_train):
                    current_epoch = start_epoch + epoch + 1
                    self.log_message(f"  Эпоха {current_epoch}/{end_epoch}")

                    train_metrics = self.models[key].train_epoch(
                        train_loader, device, optimizer, epoch
                    )

                    val_metrics = self.models[key].validate(val_loader, device)

                    self.log_message(
                        f"    Train Loss: {train_metrics['train_loss']:.4f}, "
                        f"Val Loss: {val_metrics['val_loss']:.4f}"
                    )

                    all_metrics.update(val_metrics)
                    all_metrics['train_loss'] = train_metrics['train_loss']

                metrics = all_metrics

            metrics['epoch'] = end_epoch
            self.metrics_history[key].append(metrics)
            self.current_epochs[key] = end_epoch

            self.log_message(f"Завершено обучение {key} до эпохи {end_epoch}")
            self.log_metrics(key, metrics)

            if end_epoch < self.max_epochs:
                checkpoint_path = os.path.join(
                    self.checkpoint_dir,
                    f"{key}_epoch_{end_epoch}.pt"
                )
                self.models[key].save(checkpoint_path)
                self.log_message(f"Сохранен чекпоинт: {checkpoint_path}")

                del self.models[key]
                self.models[key] = None

                if model_type in ['faster_rcnn', 'retinanet'] and key in self.dataloaders:
                    del self.dataloaders[key]

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()

                gc.collect()

                self.log_message(f"[MEM] Память очищена после {key}")

            return metrics

        except Exception as e:
            self.log_message(f"ОШИБКА при обучении {key}: {str(e)}")
            import traceback
            self.log_message(traceback.format_exc())
            return None

    def log_metrics(self, key: str, metrics: Dict[str, float]):
        self.log_message(f"Метрики для {key} (эпоха {metrics.get('epoch', 0)}):")
        for metric_name, value in metrics.items():
            if metric_name != 'epoch':
                self.log_message(f"  {metric_name}: {value:.4f}")

    def compare_models(self, epoch: int):
        self.log_message(f"\n=== СРАВНЕНИЕ МОДЕЛЕЙ НА ЭПОХЕ {epoch} ===")

        current_metrics = {}
        for key in self.metrics_history:
            if self.metrics_history[key]:
                current_metrics[key] = self.metrics_history[key][-1]

        if len(current_metrics) < 2:
            self.log_message("Недостаточно данных для сравнения")
            return

        sample_metrics = next(iter(current_metrics.values()))
        available_metrics = [k for k in sample_metrics.keys() if k != 'epoch']

        for metric in available_metrics:
            self.log_message(f"\n{metric.upper()}:")

            reverse = 'loss' not in metric.lower()

            sorted_items = sorted(
                current_metrics.items(),
                key=lambda x: x[1].get(metric, 0),
                reverse=reverse
            )

            for i, (key, metrics) in enumerate(sorted_items):
                rank = i + 1
                value = metrics.get(metric, 0)
                self.log_message(f"  {rank}. {key}: {value:.4f}")

        self.log_message("=" * 50)

    def run_training(self):
        self.log_message("Начинаем поэтапное обучение моделей...")

        for epoch in range(
                self.checkpoint_interval,
                self.max_epochs + 1,
                self.checkpoint_interval
        ):
            self.log_message(f"\n{'=' * 60}")
            self.log_message(f"ЭТАП: Обучение до эпохи {epoch}")
            self.log_message(f"{'=' * 60}")

            for model_config in self.model_configs:
                for dataset_name in self.dataset_names:
                    key = f"{model_config['name']}_{dataset_name}"
                    start_epoch = self.current_epochs[key]

                    metrics = self.train_model_segment(
                        model_config,
                        dataset_name,
                        start_epoch,
                        epoch
                    )

                    if metrics is None:
                        self.log_message(f"Пропускаем {key} из-за ошибки обучения")
                        continue

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()

                    time.sleep(2)

            self.compare_models(epoch)

        self.save_final_results()
        self.log_message("\n=== ОБУЧЕНИЕ ЗАВЕРШЕНО ===")

    def save_final_results(self):
        results_data = {
            'model_configs': self.model_configs,
            'dataset_names': self.dataset_names,
            'max_epochs': self.max_epochs,
            'metrics_history': self.metrics_history
        }

        json_file = os.path.join(
            self.results_dir,
            f"final_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )

        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)

        self.log_message(f"Финальные результаты сохранены в {json_file}")


# ===================== ПРИМЕР ИСПОЛЬЗОВАНИЯ =====================

if __name__ == "__main__":
    from Data.Datasets.dataset_work import get_dataset_path

    # ============ КОНФИГУРАЦИИ МОДЕЛЕЙ ============
    model_configs = [
        {
            'type': 'yolo',
            'size': 's',  # small
            'name': 'yolo_small'
        },
        {
            'type': 'retinanet',
            'pretrained': True,
            'name': 'retinanet'
        }
    ]

    # ============ ДАТАСЕТЫ С РАЗНОЙ ПРЕДОБРАБОТКОЙ ============
    dataset_names = [
        "SAR_low",
        "SAR_LP_med3_CLACHE1_16",
    ]

    # ============ ПРОВЕРКА ДАТАСЕТОВ ============
    print("=" * 60)
    print("ПРОВЕРКА ДАТАСЕТОВ")
    print("=" * 60)
    missing = []
    for name in dataset_names:
        path = get_dataset_path(name)
        if not path.exists():
            missing.append(name)
            print(f"  ✗ {name} - НЕ НАЙДЕН по пути: {path}")
        else:
            print(f"  ✓ {name}")

    if missing:
        print(f"\nОШИБКА! Не найдены датасеты: {missing}")
        print("Проверьте названия в списке dataset_names")
        exit(1)

    print(f"\nВсе {len(dataset_names)} датасетов найдены!")
    print("=" * 60)
    print()

    # ============ СОЗДАНИЕ И ЗАПУСК ТРЕНЕРА ============
    trainer = UniversalModelTrainer(
        model_configs=model_configs,
        dataset_names=dataset_names,
        max_epochs=80,
        checkpoint_interval=5
    )

    trainer.run_training()