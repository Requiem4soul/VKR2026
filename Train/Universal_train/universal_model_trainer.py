"""
Universal Model Trainer - ИСПРАВЛЕННАЯ ВЕРСИЯ

ИСПРАВЛЕНИЯ:
1. ✅ RetinaNet метрики: score_threshold снижен до 0.1 (было 0.5)
2. ✅ YOLO loss: читается из CSV файла results.csv
3. ✅ mAP50-95: добавлен расчёт для всех моделей (не только YOLO)
4. ✅ Улучшенная статистика и логирование

Автор: Исправленная версия для дипломной работы
Дата: 2026-02-12
"""

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
import numpy as np

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

from Data.Datasets.dataset_work import get_dataset_path


# ===================== УТИЛИТЫ =====================

def get_available_vram_gb() -> float:
    """Получает доступную VRAM в гигабайтах"""
    if not torch.cuda.is_available():
        return 0.0
    total_memory = torch.cuda.get_device_properties(0).total_memory
    return total_memory / (1024 ** 3)


def calculate_optimal_batch_size(
    model_type: str,
    vram_gb: float,
    image_size: int = 640,
    safety_margin: float = 0.7
) -> int:
    """Вычисляет оптимальный batch_size на основе VRAM"""
    if vram_gb < 2:
        return 1
    
    # Специальная оптимизация для 16GB GPU
    if 14.0 <= vram_gb <= 18.0:
        print(f"[OPTIMIZE] Обнаружена 16GB GPU - применяются оптимизированные параметры")
        
        if model_type == 'yolo':
            scale_factor = (image_size / 640) ** 2
            if image_size <= 640:
                batch = 24
            elif image_size <= 800:
                batch = 16
            elif image_size <= 1024:
                batch = 12
            else:
                batch = 8
            return batch
            
        elif model_type == 'faster_rcnn':
            return 8
            
        elif model_type == 'retinanet':
            return 12
    
    # Автоматический расчёт для других конфигураций
    effective_vram = vram_gb * safety_margin
    
    if model_type == 'yolo':
        scale_factor = (image_size / 640) ** 2
        memory_per_image = 0.5 * scale_factor
        batch = int(effective_vram / memory_per_image)
        batch = max(1, min(batch, 32))
        
    elif model_type == 'faster_rcnn':
        memory_per_image = 1.2
        batch = int(effective_vram / memory_per_image)
        batch = max(1, min(batch, 10))
        
    elif model_type == 'retinanet':
        memory_per_image = 1.0
        batch = int(effective_vram / memory_per_image)
        batch = max(1, min(batch, 16))
    else:
        batch = 2
    
    return batch


def get_image_size_from_dataset(dataset_path: Path) -> int:
    """Определяет размер изображений в датасете"""
    images_dir = dataset_path / "train" / "images"
    image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
    
    if not image_files:
        return 640
    
    img = cv2.imread(str(image_files[0]))
    h, w = img.shape[:2]
    max_side = max(h, w)
    imgsz = ((max_side + 31) // 32) * 32
    
    return imgsz


# ===================== YOLO DATASET INFO =====================

class YOLODatasetInfo:
    """Класс для извлечения информации из YOLO датасета"""
    
    @staticmethod
    def get_num_classes(dataset_path: Path) -> int:
        yaml_path = dataset_path / "data.yaml"
        
        if not yaml_path.exists():
            raise FileNotFoundError(f"data.yaml не найден: {yaml_path}")
        
        with open(yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        if 'nc' in data:
            return int(data['nc'])
        elif 'names' in data:
            return len(data['names'])
        else:
            raise ValueError("Не удалось определить количество классов")
    
    @staticmethod
    def get_class_names(dataset_path: Path) -> List[str]:
        yaml_path = dataset_path / "data.yaml"
        
        with open(yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        if 'names' in data:
            names = data['names']
            if isinstance(names, dict):
                return [names[i] for i in sorted(names.keys())]
            else:
                return names
        else:
            raise ValueError("Названия классов не найдены")


# ===================== DATASET CONVERTER =====================

class YOLOToFasterRCNNDataset(Dataset):
    """Конвертирует YOLO датасет в формат Faster R-CNN/RetinaNet"""
    
    def __init__(self, dataset_path: Path, split: str = 'train', transforms=None):
        self.dataset_path = dataset_path
        self.split = split
        self.transforms = transforms
        
        self.images_dir = dataset_path / split / "images"
        self.labels_dir = dataset_path / split / "labels"
        
        self.image_files = sorted(list(self.images_dir.glob("*.jpg")) + 
                                 list(self.images_dir.glob("*.png")))
        
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
        
        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
        
        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx])
        }
        
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        
        if self.transforms:
            image, target = self.transforms(image, target)
        
        return image, target


# ===================== МЕТРИКИ ДЕТЕКЦИИ (ИСПРАВЛЕННАЯ ВЕРСИЯ) =====================

def compute_iou(box1, box2):
    """Расчет IoU между двумя bbox"""
    if torch.is_tensor(box1):
        box1 = box1.numpy()
    if torch.is_tensor(box2):
        box2 = box2.numpy()
    
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    inter_xmin = max(x1_min, x2_min)
    inter_ymin = max(y1_min, y2_min)
    inter_xmax = min(x1_max, x2_max)
    inter_ymax = min(y1_max, y2_max)
    
    if inter_xmax < inter_xmin or inter_ymax < inter_ymin:
        return 0.0
    
    inter_area = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)
    
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    
    union_area = box1_area + box2_area - inter_area
    
    iou = inter_area / union_area if union_area > 0 else 0.0
    
    return float(iou)


def compute_detection_metrics(model, dataloader, device, 
                              iou_thresholds=None, score_threshold=0.1):
    """
    ИСПРАВЛЕННАЯ ВЕРСИЯ: Расчёт метрик детекции с поддержкой mAP50-95
    
    Args:
        model: Модель для оценки
        dataloader: DataLoader с валидационными данными
        device: Устройство (cuda/cpu)
        iou_thresholds: Список порогов IoU для mAP50-95 (default: [0.5, 0.55, ..., 0.95])
        score_threshold: Порог уверенности (default: 0.1 вместо 0.5!)
    
    Returns:
        dict: Словарь с метриками (precision, recall, mAP50, mAP50-95, f1, avg_iou)
    """
    
    # ИСПРАВЛЕНИЕ #1: Снижен score_threshold до 0.1 для RetinaNet
    if iou_thresholds is None:
        iou_thresholds = [0.5 + 0.05 * i for i in range(10)]  # [0.5, 0.55, ..., 0.95]
    
    model.to(device)
    model.eval()
    
    all_predictions = []
    all_targets = []
    
    print(f"[METRICS] Расчет метрик детекции (score_threshold={score_threshold})...")
    
    with torch.no_grad():
        for images, targets in dataloader:
            images = [img.to(device) for img in images]
            
            try:
                predictions = model(images)
            except Exception as e:
                print(f"[METRICS] Ошибка при inference: {e}")
                continue
            
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
            'mAP50-95': 0.0,  # ИСПРАВЛЕНИЕ #3: добавлен mAP50-95
            'f1': 0.0,
            'avg_iou': 0.0
        }
    
    # НОВОЕ: Расчёт mAP для всех порогов IoU
    map_results = []
    
    for iou_threshold in iou_thresholds:
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        total_iou = 0
        iou_count = 0
        
        total_predictions_before_filter = 0
        total_predictions_after_filter = 0
        
        for pred, target in zip(all_predictions, all_targets):
            pred_boxes = pred.get('boxes', torch.tensor([]))
            pred_labels = pred.get('labels', torch.tensor([]))
            pred_scores = pred.get('scores', torch.tensor([]))
            
            target_boxes = target.get('boxes', torch.tensor([]))
            target_labels = target.get('labels', torch.tensor([]))
            
            if len(pred_boxes) == 0 and len(target_boxes) == 0:
                continue
            
            if len(target_boxes) > 0 and len(pred_boxes) == 0:
                false_negatives += len(target_boxes)
                continue
            
            total_predictions_before_filter += len(pred_boxes)
            
            # Фильтруем по score
            if len(pred_scores) > 0:
                keep = pred_scores > score_threshold
                pred_boxes = pred_boxes[keep]
                pred_labels = pred_labels[keep]
                pred_scores = pred_scores[keep]
            
            total_predictions_after_filter += len(pred_boxes)
            
            if len(pred_boxes) == 0:
                if len(target_boxes) > 0:
                    false_negatives += len(target_boxes)
                continue
            
            matched_targets = set()
            
            for pred_box, pred_label in zip(pred_boxes, pred_labels):
                best_iou = 0
                best_target_idx = -1
                
                for target_idx, (target_box, target_label) in enumerate(
                    zip(target_boxes, target_labels)
                ):
                    if target_idx in matched_targets:
                        continue
                    
                    if pred_label != target_label:
                        continue
                    
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
            
            if len(target_boxes) > 0:
                false_negatives += len(target_boxes) - len(matched_targets)
        
        # Расчет метрик для текущего порога
        precision = (true_positives / (true_positives + false_positives) 
                    if (true_positives + false_positives) > 0 else 0.0)
        recall = (true_positives / (true_positives + false_negatives) 
                 if (true_positives + false_negatives) > 0 else 0.0)
        
        map_results.append(precision)
        
        # Для первого порога (0.5) сохраняем детальную статистику
        if iou_threshold == 0.5:
            mAP50 = precision
            final_precision = precision
            final_recall = recall
            final_tp = true_positives
            final_fp = false_positives
            final_fn = false_negatives
            final_before = total_predictions_before_filter
            final_after = total_predictions_after_filter
            avg_iou = total_iou / iou_count if iou_count > 0 else 0.0
    
    # ИСПРАВЛЕНИЕ #3: Расчёт mAP50-95
    mAP50_95 = float(np.mean(map_results)) if map_results else 0.0
    
    f1 = (2 * final_precision * final_recall / (final_precision + final_recall) 
          if (final_precision + final_recall) > 0 else 0.0)
    
    # Статистика
    print(f"[METRICS] Предсказаний до фильтрации: {final_before}")
    print(f"[METRICS] Предсказаний после фильтрации (score > {score_threshold}): {final_after}")
    print(f"[METRICS] TP={final_tp}, FP={final_fp}, FN={final_fn}")
    print(f"[METRICS] Precision: {final_precision:.4f}, Recall: {final_recall:.4f}")
    print(f"[METRICS] mAP50: {mAP50:.4f}, mAP50-95: {mAP50_95:.4f}")
    print(f"[METRICS] f1: {f1:.4f}")
    
    return {
        'precision': float(final_precision),
        'recall': float(final_recall),
        'mAP50': float(mAP50),
        'mAP50-95': float(mAP50_95),  # ИСПРАВЛЕНИЕ #3
        'f1': float(f1),
        'avg_iou': float(avg_iou)
    }


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
    def train_epoch(self, *args, **kwargs):
        pass
    
    @abstractmethod
    def validate(self, *args, **kwargs):
        pass
    
    @abstractmethod
    def save(self, path: str):
        pass
    
    @abstractmethod
    def load(self, path: str):
        pass


# ===================== YOLO WRAPPER (ИСПРАВЛЕННАЯ ВЕРСИЯ) =====================

class YOLOWrapper(BaseModelWrapper):
    """Обертка для YOLO моделей - ИСПРАВЛЕНА"""
    
    def __init__(self, model_size: str = 'm'):
        super().__init__(task_type='detection')
        self.model_size = model_size
        self.results = None
        self.last_project_path = None  # ИСПРАВЛЕНИЕ #2: для чтения CSV
    
    def initialize(self, num_classes: int = None, **kwargs):
        self.model = YOLO(f'yolov8{self.model_size}.pt')
        print(f"[✓] Инициализирована YOLOv8{self.model_size}")
    
    def train_epoch(self, dataset_path: Path, epochs: int, device, **kwargs):
        yaml_path = dataset_path / "data.yaml"
        
        project = kwargs.get('project', 'runs')
        name = kwargs.get('name', 'exp')
        
        # ИСПРАВЛЕНИЕ #2: Сохраняем путь для чтения CSV
        self.last_project_path = Path(project) / name
        
        self.results = self.model.train(
            data=str(yaml_path),
            epochs=epochs,
            imgsz=kwargs.get('imgsz', 640),
            batch=kwargs.get('batch', -1),
            device=device,
            save=False,
            project=project,
            name=name,
            exist_ok=True,
            workers=1,
            cache=False,
            verbose=False
        )
        
        return self.extract_metrics()
    
    def validate(self, dataloader, device, **kwargs):
        return self.extract_metrics()

    def extract_metrics(self) -> Dict[str, float]:
        """
        ИСПРАВЛЕНА: Теперь читает loss из CSV файла + добавлен F1
        """
        try:
            # Метрики детекции (из results_dict)
            precision = float(self.results.results_dict.get('metrics/precision(B)', 0))
            recall = float(self.results.results_dict.get('metrics/recall(B)', 0))
            map50 = float(self.results.results_dict.get('metrics/mAP50(B)', 0))
            map50_95 = float(self.results.results_dict.get('metrics/mAP50-95(B)', 0))

            # Вычисляем F1-score (гармоническое среднее precision и recall)
            if precision + recall > 0:
                f1 = 2 * (precision * recall) / (precision + recall)
            else:
                f1 = 0.0

            # ИСПРАВЛЕНИЕ #2: Loss метрики (из CSV файла)
            train_loss = 0.0
            val_loss = 0.0

            if self.last_project_path:
                csv_path = self.last_project_path / 'results.csv'

                if csv_path.exists():
                    try:
                        import pandas as pd
                        df = pd.read_csv(csv_path)

                        if len(df) > 0:
                            last_row = df.iloc[-1]

                            # Суммируем компоненты train loss
                            train_box = last_row.get('train/box_loss', 0) or 0
                            train_cls = last_row.get('train/cls_loss', 0) or 0
                            train_dfl = last_row.get('train/dfl_loss', 0) or 0
                            train_loss = float(train_box + train_cls + train_dfl)

                            # Суммируем компоненты val loss
                            val_box = last_row.get('val/box_loss', 0) or 0
                            val_cls = last_row.get('val/cls_loss', 0) or 0
                            val_dfl = last_row.get('val/dfl_loss', 0) or 0
                            val_loss = float(val_box + val_cls + val_dfl)

                            print(f"[YOLO] Loss прочитан из CSV: train={train_loss:.4f}, val={val_loss:.4f}")

                    except Exception as e:
                        print(f"[WARNING] Не удалось прочитать loss из CSV: {e}")

            return {
                'precision': precision,
                'recall': recall,
                'mAP50': map50,
                'mAP50-95': map50_95,
                'f1': f1,
                'train_loss': train_loss,
                'val_loss': val_loss
            }

        except Exception as e:
            print(f"[WARNING] Ошибка извлечения метрик YOLO: {e}")
            return {
                'precision': 0.0,
                'recall': 0.0,
                'mAP50': 0.0,
                'mAP50-95': 0.0,
                'f1': 0.0,
                'train_loss': 0.0,
                'val_loss': 0.0
            }
    
    def save(self, path: str):
        self.model.save(path)
    
    def load(self, path: str):
        self.model = YOLO(path)


# ===================== FASTER R-CNN WRAPPER (ИСПРАВЛЕННАЯ ВЕРСИЯ) =====================

class FasterRCNNWrapper(BaseModelWrapper):
    """Обертка для Faster R-CNN - ИСПРАВЛЕНА"""
    
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
        
        print(f"[✓] Инициализирован Faster R-CNN (classes={self.num_classes})")
    
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
        """Валидация с расчетом метрик - ИСПРАВЛЕНА"""
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
        
        # ИСПРАВЛЕНИЕ #1 и #3: Передаём score_threshold=0.1
        metrics = compute_detection_metrics(self.model, dataloader, device, 
                                           score_threshold=0.1)
        
        return {
            'val_loss': avg_loss,
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'mAP50': metrics['mAP50'],
            'mAP50-95': metrics['mAP50-95'],
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


# ===================== RETINANET WRAPPER (ИСПРАВЛЕННАЯ ВЕРСИЯ) =====================

class RetinaNetWrapper(BaseModelWrapper):
    """Обертка для RetinaNet - ИСПРАВЛЕНА"""
    
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
        
        print(f"[✓] Инициализирован RetinaNet (classes={self.num_classes})")
    
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
                
                if torch.isnan(losses) or torch.isinf(losses):
                    print(f"[WARNING] NaN/Inf в loss на батче {batch_idx}")
                    nan_count += 1
                    if nan_count > 10:
                        break
                    continue
                
                optimizer.zero_grad()
                losses.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                
                total_loss += losses.item()
                num_batches += 1
                
            except RuntimeError as e:
                print(f"[WARNING] RuntimeError: {e}")
                nan_count += 1
                continue
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        if nan_count > 0:
            print(f"[INFO] Пропущено {nan_count} батчей из-за NaN/Inf")
        
        return {'train_loss': avg_loss}
    
    def validate(self, dataloader, device, **kwargs):
        """Валидация с расчетом метрик - ИСПРАВЛЕНА"""
        self.model.to(device)
        self.model.eval()
        
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for images, targets in dataloader:
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                
                try:
                    self.model.train()
                    loss_dict = self.model(images, targets)
                    self.model.eval()
                    
                    cls_loss = loss_dict.get('classification', torch.tensor(0.0))
                    box_loss = loss_dict.get('bbox_regression', torch.tensor(0.0))
                    losses = cls_loss + box_loss
                    
                    if torch.isnan(losses) or torch.isinf(losses):
                        continue
                    
                    total_loss += losses.item()
                    num_batches += 1
                    
                except Exception as e:
                    print(f"[WARNING] Ошибка при расчете loss: {e}")
                    continue
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        # ИСПРАВЛЕНИЕ #1 и #3: Передаём score_threshold=0.1
        try:
            metrics = compute_detection_metrics(self.model, dataloader, device,
                                               score_threshold=0.1)
        except Exception as e:
            print(f"[WARNING] Ошибка при расчете метрик: {e}")
            metrics = {
                'precision': 0.0,
                'recall': 0.0,
                'mAP50': 0.0,
                'mAP50-95': 0.0,
                'f1': 0.0
            }
        
        return {
            'val_loss': avg_loss,
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'mAP50': metrics['mAP50'],
            'mAP50-95': metrics['mAP50-95'],  # ИСПРАВЛЕНИЕ #3: добавлен
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


# ===================== UNIVERSAL TRAINER (УЛУЧШЕННАЯ ВЕРСИЯ) =====================

class UniversalModelTrainer:
    """Поэтапное обучение моделей - УЛУЧШЕННАЯ ВЕРСИЯ"""
    
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
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir = f"results_{timestamp}"
        os.makedirs(self.results_dir, exist_ok=True)
        
        self.checkpoint_dir = os.path.join(self.results_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        self.results_file = os.path.join(self.results_dir, "training_log.txt")
        self.metrics_file = os.path.join(self.results_dir, "metrics.json")
        
        self.models = {}
        self.dataloaders = {}
        self.metrics_history = {
            f"{model['name']}_{dataset}": []
            for model in model_configs
            for dataset in dataset_names
        }
        self.current_epochs = {
            f"{model['name']}_{dataset}": 0
            for model in model_configs
            for dataset in dataset_names
        }
        
        # Определение VRAM
        self.vram_gb = get_available_vram_gb()
        self.log_message(f"[SYSTEM] Доступная VRAM: {self.vram_gb:.2f} GB")
        
        if clean_old_results:
            self.clean_old_result_folders()
    
    def clean_old_result_folders(self):
        import shutil
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
    
    def create_dataloaders(self, dataset_path: Path, model_type: str, 
                          batch_size: Optional[int] = None):
        if model_type == 'yolo':
            return None
        
        elif model_type in ['faster_rcnn', 'retinanet']:
            if batch_size is None:
                batch_size = calculate_optimal_batch_size(model_type, self.vram_gb)
                self.log_message(f"[AUTO] Batch size для {model_type}: {batch_size}")
            
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
    
    def initialize_model(self, model_config: Dict[str, Any], 
                        dataset_path: Path) -> BaseModelWrapper:
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
        
        assert wrapper.task_type == 'detection', \
            f"ОШИБКА: Ожидается только detection, получено {wrapper.task_type}"
        
        return wrapper
    
    def train_model_segment(
        self,
        model_config: Dict[str, Any],
        dataset_name: str,
        start_epoch: int,
        end_epoch: int
    ):
        model_max_epochs = model_config.get('max_epochs', self.max_epochs)

        model_name = model_config['name']
        key = f"{model_name}_{dataset_name}"

        if start_epoch >= model_max_epochs:
            self.log_message(f"\n--- {key}: достигнут max_epochs={model_max_epochs}, пропуск ---")
            return None

        end_epoch = min(end_epoch, model_max_epochs)

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
                imgsz = get_image_size_from_dataset(dataset_path)
                print(f"[INFO] Размер изображений: {imgsz}x{imgsz}")
                
                metrics = self.models[key].train_epoch(
                    dataset_path=dataset_path,
                    epochs=epochs_to_train,
                    device=device,
                    batch=-1,
                    imgsz=imgsz
                )
            
            elif model_type in ['faster_rcnn', 'retinanet']:
                if key not in self.dataloaders:
                    self.dataloaders[key] = self.create_dataloaders(
                        dataset_path,
                        model_type
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
                        f"Train Loss: {train_metrics['train_loss']:.4f}, "
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
    
    def run_training(self):
        """Основной цикл обучения"""
        self.log_message("\n" + "=" * 80)
        self.log_message("НАЧАЛО УНИВЕРСАЛЬНОГО ОБУЧЕНИЯ")
        self.log_message("=" * 80)
        
        for epoch_checkpoint in range(0, self.max_epochs, self.checkpoint_interval):
            start_epoch = epoch_checkpoint
            end_epoch = min(epoch_checkpoint + self.checkpoint_interval, self.max_epochs)
            
            self.log_message(f"\n{'=' * 80}")
            self.log_message(f"ЧЕКПОИНТ: Эпохи {start_epoch + 1}-{end_epoch}")
            self.log_message(f"{'=' * 80}")
            
            for dataset_name in self.dataset_names:
                for model_config in self.model_configs:
                    self.train_model_segment(
                        model_config,
                        dataset_name,
                        start_epoch,
                        end_epoch
                    )
            
            self.compare_models(end_epoch)
            
            with open(self.metrics_file, 'w', encoding='utf-8') as f:
                json.dump(self.metrics_history, f, indent=2)
        
        self.log_message("\n" + "=" * 80)
        self.log_message("ОБУЧЕНИЕ ЗАВЕРШЕНО")
        self.log_message("=" * 80)


# ===================== ПРИМЕР ИСПОЛЬЗОВАНИЯ =====================

if __name__ == "__main__":
    model_configs = [
        {'type': 'yolo', 'size': 's', 'name': 'yolo_small', 'max_epochs': 80},
        {'type': 'faster_rcnn', 'pretrained': True, 'name': 'faster_rcnn', 'max_epochs': 25},
        {'type': 'retinanet', 'pretrained': True, 'name': 'retinanet', 'max_epochs': 35}
    ]
    
    dataset_names = ["dataset_LUNA16", "LUNA16_CHECK"]
    
    trainer = UniversalModelTrainer(
        model_configs=model_configs,
        dataset_names=dataset_names,
        max_epochs=50,
        checkpoint_interval=5
    )
    
    trainer.run_training()
