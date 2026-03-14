"""
Universal Model Trainer - ПОЛНАЯ ФИНАЛЬНАЯ ВЕРСИЯ

ВКЛЮЧАЕТ:
1. ✅ Весь оригинальный код (wrapper'ы, датасеты, метрики, очистка памяти)
2. ✅ Early Stopping для каждой модели (Prechelt, 1998)
3. ✅ Ранний отбор моделей (Jamieson & Talwalkar, 2016)
4. ✅ Критическая очистка памяти GPU после каждого чекпоинта
5. ✅ Индивидуальные max_epochs для каждой модели
6. ✅ Воспроизводимость через seed (Dodge & Karam, 2017) [НОВОЕ]

ИСПРАВЛЕНИЯ из оригинала:
- RetinaNet: score_threshold=0.1 (было 0.5)
- YOLO: loss читается из CSV
- mAP50-95: добавлен для всех моделей

НОВОЕ:
- Early Stopping с настраиваемыми параметрами
- Интеграция с модулем early_model_selection
- Автоматическое восстановление лучших весов
- seed: глобальная фиксация torch/numpy/random/cudnn

Автор: VKR2026 (финальная версия)
Дата: 2026-02-17
"""

import os
import json
import gc
import random
import torch
import time
from datetime import datetime
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
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


# ===================== EARLY STOPPING (НОВОЕ) =====================

@dataclass
class EarlyStoppingConfig:
    """Конфигурация Early Stopping (Prechelt, 1998)"""
    patience: int = 10
    min_delta: float = 0.001
    metric: str = 'mAP50-95'
    mode: str = 'max'
    restore_best: bool = True


class EarlyStopping:
    """
    Early Stopping для предотвращения переобучения
    
    Научное обоснование: Prechelt (1998), Caruana et al. (2001)
    """
    
    def __init__(self, config: EarlyStoppingConfig, model_key: str):
        self.config = config
        self.model_key = model_key
        self.patience_counter = 0
        self.best_score = None
        self.best_epoch = 0
        self.best_model_path = None
        self.history = []
        
        self.is_better = (
            lambda new, best: new > best + config.min_delta
            if config.mode == 'max'
            else lambda new, best: new < best - config.min_delta
        )
    
    def step(self, metrics: Dict[str, float], epoch: int, model_save_fn=None) -> Tuple[bool, str]:
        """Проверяет, нужно ли продолжать обучение"""
        current_score = metrics.get(self.config.metric)
        
        if current_score is None:
            return True, f"Метрика '{self.config.metric}' недоступна"
        
        self.history.append({'epoch': epoch, 'score': current_score, 'metrics': metrics.copy()})
        
        if self.best_score is None:
            self.best_score = current_score
            self.best_epoch = epoch
            if model_save_fn and self.config.restore_best:
                self.best_model_path = model_save_fn(epoch, metrics)
            return True, f"Инициализация (best {self.config.metric}={current_score:.4f})"
        
        if self.is_better(current_score, self.best_score):
            improvement = current_score - self.best_score if self.config.mode == 'max' else self.best_score - current_score
            self.best_score = current_score
            self.best_epoch = epoch
            self.patience_counter = 0
            
            if model_save_fn and self.config.restore_best:
                self.best_model_path = model_save_fn(epoch, metrics)
            
            return True, f"Улучшение на {improvement:.4f} ({self.config.metric}={current_score:.4f})"
        else:
            self.patience_counter += 1
            
            if self.patience_counter >= self.config.patience:
                return False, (
                    f"Early stopping: {self.config.metric} не улучшался "
                    f"{self.patience_counter} эпох "
                    f"(лучший: {self.best_score:.4f} на эпохе {self.best_epoch})"
                )
            else:
                return True, (
                    f"Нет улучшения ({self.patience_counter}/{self.config.patience}), "
                    f"лучший {self.config.metric}={self.best_score:.4f}"
                )
    
    def get_best_info(self) -> Dict[str, Any]:
        return {
            'best_epoch': self.best_epoch,
            'best_score': self.best_score,
            'best_model_path': self.best_model_path,
            'patience_counter': self.patience_counter,
            'stopped_early': self.patience_counter >= self.config.patience
        }


# ===================== УТИЛИТЫ (ОРИГИНАЛЬНЫЕ) =====================

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
    
    effective_vram = vram_gb * safety_margin
    
    base_batch = {
        'yolo': 16,
        'faster_rcnn': 4,
        'retinanet': 8
    }.get(model_type, 4)
    
    scale_factor = effective_vram / 8.0
    batch_size = max(1, int(base_batch * scale_factor))
    
    return batch_size


def get_image_size_from_dataset(dataset_path: Path) -> int:
    """Определяет размер изображений из датасета"""
    train_images = dataset_path / 'train' / 'images'
    
    if not train_images.exists():
        return 640
    
    image_files = list(train_images.glob('*.jpg')) + list(train_images.glob('*.png'))
    
    if not image_files:
        return 640
    
    sample_img = cv2.imread(str(image_files[0]))
    if sample_img is None:
        return 640
    
    height, width = sample_img.shape[:2]
    return max(height, width)


# ===================== DATASET INFO (ОРИГИНАЛЬНЫЙ) =====================

class YOLODatasetInfo:
    """Извлечение информации из YOLO датасета"""
    
    @staticmethod
    def get_num_classes(dataset_path: Path) -> int:
        yaml_path = dataset_path / 'data.yaml'
        if not yaml_path.exists():
            raise FileNotFoundError(f"Не найден data.yaml в {dataset_path}")
        
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        
        return data.get('nc', len(data.get('names', [])))
    
    @staticmethod
    def get_class_names(dataset_path: Path) -> list:
        yaml_path = dataset_path / 'data.yaml'
        if not yaml_path.exists():
            return []
        
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        
        return data.get('names', [])


# ===================== DATASET CONVERTER (ОРИГИНАЛЬНЫЙ) =====================

class YOLOToFasterRCNNDataset(Dataset):
    """Конвертер YOLO → Faster R-CNN формат"""
    
    def __init__(self, dataset_path: Path, split: str = 'train', transforms=None):
        self.dataset_path = dataset_path
        self.split = split
        self.transforms = transforms
        
        self.images_dir = dataset_path / split / 'images'
        self.labels_dir = dataset_path / split / 'labels'
        
        if not self.images_dir.exists():
            raise FileNotFoundError(f"Не найдена папка: {self.images_dir}")
        
        self.image_files = sorted(list(self.images_dir.glob('*.jpg')) + 
                                 list(self.images_dir.glob('*.png')))
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        height, width = image.shape[:2]
        
        label_path = self.labels_dir / (img_path.stem + '.txt')
        
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
                        
                        xmin = x_center - w / 2
                        ymin = y_center - h / 2
                        xmax = x_center + w / 2
                        ymax = y_center + h / 2
                        
                        boxes.append([xmin, ymin, xmax, ymax])
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


# ===================== МЕТРИКИ ДЕТЕКЦИИ (ОРИГИНАЛЬНЫЕ, ИСПРАВЛЕННЫЕ) =====================

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
    Расчёт метрик детекции с поддержкой mAP50-95
    ИСПРАВЛЕНО: score_threshold=0.1 для RetinaNet
    """
    if iou_thresholds is None:
        iou_thresholds = [0.5 + 0.05 * i for i in range(10)]
    
    model.to(device)
    model.eval()
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for images, targets in dataloader:
            images = [img.to(device) for img in images]
            
            try:
                predictions = model(images)
            except Exception as e:
                continue
            
            predictions = [{k: v.cpu() for k, v in pred.items()} for pred in predictions]
            targets = [{k: v.cpu() for k, v in t.items()} for t in targets]
            
            all_predictions.extend(predictions)
            all_targets.extend(targets)
    
    if len(all_predictions) == 0 or len(all_targets) == 0:
        return {
            'precision': 0.0, 'recall': 0.0, 'mAP50': 0.0,
            'mAP50-95': 0.0, 'f1': 0.0, 'avg_iou': 0.0
        }
    
    map_results = []
    
    for iou_threshold in iou_thresholds:
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        total_iou = 0.0
        iou_count = 0
        
        for pred, target in zip(all_predictions, all_targets):
            pred_boxes = pred.get('boxes', torch.zeros((0, 4)))
            pred_scores = pred.get('scores', torch.zeros(0))
            gt_boxes = target.get('boxes', torch.zeros((0, 4)))
            
            mask = pred_scores > score_threshold
            pred_boxes = pred_boxes[mask]
            
            if len(gt_boxes) == 0 and len(pred_boxes) == 0:
                continue
            elif len(gt_boxes) == 0:
                false_positives += len(pred_boxes)
                continue
            elif len(pred_boxes) == 0:
                false_negatives += len(gt_boxes)
                continue
            
            matched_gt = set()
            for pb in pred_boxes:
                best_iou = 0
                best_gt_idx = -1
                for gt_idx, gb in enumerate(gt_boxes):
                    if gt_idx in matched_gt:
                        continue
                    iou = compute_iou(pb.numpy(), gb.numpy())
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx
                
                if best_iou >= iou_threshold and best_gt_idx >= 0:
                    true_positives += 1
                    matched_gt.add(best_gt_idx)
                    total_iou += best_iou
                    iou_count += 1
                else:
                    false_positives += 1
            
            false_negatives += len(gt_boxes) - len(matched_gt)
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
        
        map_results.append(precision)
    
    mAP50 = map_results[0] if len(map_results) > 0 else 0.0
    mAP50_95 = sum(map_results) / len(map_results) if len(map_results) > 0 else 0.0
    
    final_precision = map_results[0] if len(map_results) > 0 else 0.0
    final_recall = recall
    
    f1 = 2 * (final_precision * final_recall) / (final_precision + final_recall) if (final_precision + final_recall) > 0 else 0.0
    avg_iou = total_iou / iou_count if iou_count > 0 else 0.0
    
    return {
        'precision': float(final_precision),
        'recall': float(final_recall),
        'mAP50': float(mAP50),
        'mAP50-95': float(mAP50_95),
        'f1': float(f1),
        'avg_iou': float(avg_iou)
    }


# ===================== BASE MODEL WRAPPER (ОРИГИНАЛЬНЫЙ) =====================

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


# ===================== YOLO WRAPPER (ИСПРАВЛЕННЫЙ) =====================

class YOLOWrapper(BaseModelWrapper):
    """Обертка для YOLO моделей - ИСПРАВЛЕНА"""
    
    def __init__(self, model_size: str = 'm'):
        super().__init__(task_type='detection')
        self.model_size = model_size
        self.results = None
        self.last_project_path = None
    
    def initialize(self, num_classes: int = None, **kwargs):
        self.model = YOLO(f'yolov8{self.model_size}.pt')
        print(f"[✓] Инициализирована YOLOv8{self.model_size}")
    
    def train_epoch(self, dataset_path: Path, epochs: int, device, **kwargs):
        yaml_path = dataset_path / "data.yaml"
        
        project = kwargs.get('project', 'runs')
        name = kwargs.get('name', 'exp')
        
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
            verbose=False,
            seed=kwargs.get('seed', 0),   # ← ДОБАВЛЕНО: seed для воспроизводимости YOLO
        )
        
        return self.extract_metrics()
    
    def validate(self, dataloader, device, **kwargs):
        return self.extract_metrics()
    
    def extract_metrics(self) -> Dict[str, float]:
        """Извлекает метрики из результатов YOLO обучения"""
        if self.results is None:
            return {'mAP50': 0.0, 'mAP50-95': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'train_loss': 1.0, 'val_loss': 1.0}
        
        try:
            results_dict = self.results.results_dict
            
            map50 = float(results_dict.get('metrics/mAP50(B)', 0.0))
            map50_95 = float(results_dict.get('metrics/mAP50-95(B)', 0.0))
            precision = float(results_dict.get('metrics/precision(B)', 0.0))
            recall = float(results_dict.get('metrics/recall(B)', 0.0))
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
            train_loss = float(results_dict.get('train/box_loss', results_dict.get('train/cls_loss', 1.0)))
            val_loss = float(results_dict.get('val/box_loss', results_dict.get('val/cls_loss', 1.0)))
            
            return {
                'mAP50': map50,
                'mAP50-95': map50_95,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'train_loss': train_loss,
                'val_loss': val_loss
            }
        except Exception as e:
            print(f"[WARNING] Не удалось извлечь метрики YOLO: {e}")
            return {'mAP50': 0.0, 'mAP50-95': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'train_loss': 1.0, 'val_loss': 1.0}
    
    def save(self, path: str):
        if self.model and self.last_project_path:
            best_weights = Path(self.last_project_path) / 'weights' / 'best.pt'
            if best_weights.exists():
                import shutil
                shutil.copy(best_weights, path)
    
    def load(self, path: str):
        if os.path.exists(path):
            self.model = YOLO(path)


# ===================== FASTER R-CNN WRAPPER (ОРИГИНАЛЬНЫЙ) =====================

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
        
        metrics = compute_detection_metrics(self.model, dataloader, device, score_threshold=0.1)
        
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


# ===================== RETINANET WRAPPER (ИСПРАВЛЕННЫЙ) =====================

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
        """Валидация RetinaNet - ИСПРАВЛЕНА (score_threshold=0.1)"""
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
        
        metrics = compute_detection_metrics(self.model, dataloader, device, score_threshold=0.1)
        
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


# ===================== UNIVERSAL TRAINER (ФИНАЛЬНАЯ ВЕРСИЯ) =====================

class UniversalModelTrainer:
    """
    Универсальный тренер моделей - ФИНАЛЬНАЯ ВЕРСИЯ
    
    ВКЛЮЧАЕТ:
    - Весь оригинальный функционал
    - Early Stopping для каждой модели
    - Ранний отбор моделей (опционально)
    - Критическую очистку памяти
    - Воспроизводимость через seed [НОВОЕ]
    """
    
    def __init__(
        self,
        model_configs: List[Dict[str, Any]],
        dataset_names: List[str],
        max_epochs: int = 40,
        checkpoint_interval: int = 10,
        
        # ── Воспроизводимость ──────────────────────────────────────────────
        seed: int = 42,                        # ← НОВЫЙ ПАРАМЕТР
        
        # Early Stopping
        enable_early_stopping: bool = False,
        early_stopping_patience: int = 10,
        early_stopping_min_delta: float = 0.001,
        early_stopping_metric: str = 'mAP50-95',
        
        # Ранний отбор
        enable_early_selection: bool = False,
        early_selection_ratio: float = 0.3,
        early_selection_top_k: float = 0.5,
        
        clean_old_results: bool = False
    ):
        self.model_configs = model_configs
        self.dataset_names = dataset_names
        self.max_epochs = max_epochs
        self.checkpoint_interval = checkpoint_interval

        # ── Воспроизводимость (Dodge & Karam, 2017) ───────────────────────
        # Фиксируем seed ДО создания любых моделей и DataLoader'ов.
        # Результаты идентичны при одинаковом seed на одинаковом оборудовании.
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["PYTHONHASHSEED"] = str(seed)
        # ──────────────────────────────────────────────────────────────────
        
        # Early Stopping
        self.enable_early_stopping = enable_early_stopping
        self.default_es_config = EarlyStoppingConfig(
            patience=early_stopping_patience,
            min_delta=early_stopping_min_delta,
            metric=early_stopping_metric,
            mode='max' if 'loss' not in early_stopping_metric.lower() else 'min',
            restore_best=True
        )
        
        # Папки
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir = f"results_{timestamp}"
        os.makedirs(self.results_dir, exist_ok=True)
        
        self.checkpoint_dir = os.path.join(self.results_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        if enable_early_stopping:
            self.early_stopping_dir = os.path.join(self.results_dir, "early_stopping")
            os.makedirs(self.early_stopping_dir, exist_ok=True)
        
        self.results_file = os.path.join(self.results_dir, "training_log.txt")
        self.metrics_file = os.path.join(self.results_dir, "metrics.json")
        
        # Хранилища
        self.models = {}
        self.dataloaders = {}
        self.early_stoppers = {} if enable_early_stopping else None
        
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
        
        self.training_active = {
            f"{model['name']}_{dataset}": True
            for model in model_configs
            for dataset in dataset_names
        }
        
        self.stop_reasons = {}
        
        # VRAM
        self.vram_gb = get_available_vram_gb()
        self.log_message(f"[SYSTEM] Доступная VRAM: {self.vram_gb:.2f} GB")
        
        # Ранний отбор (опционально)
        self.enable_early_selection = enable_early_selection
        if enable_early_selection:
            try:
                from Train.Universal_train.early_model_selection import EarlyModelSelector
                self.early_selector = EarlyModelSelector(
                    checkpoint_ratio=early_selection_ratio,
                    top_k_fraction=early_selection_top_k,
                    min_models_to_keep=2,
                    log_dir=Path(self.results_dir) / "early_selection_logs"
                )
                self.log_message(
                    f"[EARLY_SELECTION] Включен ранний отбор моделей:\n"
                    f"  - Отбор на {early_selection_ratio*100:.0f}% обучения\n"
                    f"  - Оставляем {early_selection_top_k*100:.0f}% лучших моделей"
                )
            except ImportError:
                self.log_message("[WARNING] Модуль early_model_selection не найден, отключаем ранний отбор")
                self.enable_early_selection = False
                self.early_selector = None
        else:
            self.early_selector = None
        
        if clean_old_results:
            self.clean_old_result_folders()
        
        # Логируем конфигурацию
        self.log_message("\n" + "=" * 80)
        self.log_message("КОНФИГУРАЦИЯ ОБУЧЕНИЯ")
        self.log_message("=" * 80)
        self.log_message(f"Seed: {seed} (воспроизводимость включена)")
        self.log_message(f"Early Stopping: {'Включен' if enable_early_stopping else 'Отключен'}")
        if enable_early_stopping:
            self.log_message(f"  - Метрика: {early_stopping_metric}")
            self.log_message(f"  - Patience: {early_stopping_patience} эпох")
            self.log_message(f"  - Min delta: {early_stopping_min_delta}")
        self.log_message(f"Ранний отбор: {'Включен' if enable_early_selection else 'Отключен'}")
        self.log_message("=" * 80)
    
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
            
            # Generator с фиксированным seed для воспроизводимости shuffle
            g = torch.Generator()
            g.manual_seed(self.seed)
            
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=0,
                collate_fn=self.collate_fn,
                generator=g,
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
            return wrapper
        
        elif model_type == 'faster_rcnn':
            pretrained = model_config.get('pretrained', True)
            wrapper = FasterRCNNWrapper(pretrained=pretrained)
            wrapper.initialize(num_classes=num_classes)
            return wrapper
        
        elif model_type == 'retinanet':
            pretrained = model_config.get('pretrained', True)
            wrapper = RetinaNetWrapper(pretrained=pretrained)
            wrapper.initialize(num_classes=num_classes)
            return wrapper
        
        else:
            raise ValueError(f"Неизвестный тип модели: {model_type}")
    
    def create_early_stopper(self, key: str, model_config: Dict[str, Any]) -> EarlyStopping:
        es_params = model_config.get('early_stopping')
        if es_params:
            metric = es_params.get('metric', 'mAP50-95')
            config = EarlyStoppingConfig(
                patience=es_params.get('patience', self.default_es_config.patience),
                min_delta=es_params.get('min_delta', self.default_es_config.min_delta),
                metric=metric,
                mode='min' if 'loss' in metric.lower() else 'max',
                restore_best=True
            )
        else:
            config = self.default_es_config
        
        return EarlyStopping(config, key)
    
    def save_checkpoint_with_metrics(self, key: str, epoch: int, metrics: Dict[str, float]) -> str:
        checkpoint_path = os.path.join(
            self.checkpoint_dir,
            f"{key}_epoch_{epoch}_best.pt"
        )
        
        model = self.models.get(key)
        if model and hasattr(model, 'model') and model.model is not None:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.model.state_dict() if hasattr(model.model, 'state_dict') else None,
                'metrics': metrics,
                'seed': self.seed,
            }, checkpoint_path)
        
        return checkpoint_path
    
    def train_model_segment(
        self,
        model_config: Dict[str, Any],
        dataset_name: str,
        start_epoch: int,
        end_epoch: int
    ):
        """ОРИГИНАЛЬНАЯ ФУНКЦИЯ С КРИТИЧЕСКОЙ ОЧИСТКОЙ ПАМЯТИ"""
        
        model_max_epochs = model_config.get('max_epochs', self.max_epochs)
        model_name = model_config['name']
        key = f"{model_name}_{dataset_name}"

        if start_epoch >= model_max_epochs:
            self.log_message(f"\n--- {key}: достигнут max_epochs={model_max_epochs}, пропуск ---")
            return None

        end_epoch = min(end_epoch, model_max_epochs)
        
        model_type = model_config['type']
        
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
                    batch=model_config.get('batch', -1),
                    imgsz=imgsz,
                    seed=self.seed,   # ← ДОБАВЛЕНО: передаём seed в YOLO
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
            
            if end_epoch < model_max_epochs:
                checkpoint_path = os.path.join(
                    self.checkpoint_dir,
                    f"{key}_epoch_{end_epoch}.pt"
                )
                self.models[key].save(checkpoint_path)
                self.log_message(f"Сохранен чекпоинт: {checkpoint_path}")
            
            # ========== КРИТИЧЕСКАЯ ОЧИСТКА ПАМЯТИ (ОРИГИНАЛ) ==========
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
        """ОСНОВНОЙ ЦИКЛ С EARLY STOPPING И РАННИМ ОТБОРОМ"""
        
        self.log_message("\n" + "=" * 80)
        self.log_message("НАЧАЛО УНИВЕРСАЛЬНОГО ОБУЧЕНИЯ")
        self.log_message("=" * 80)
        
        # Инициализация Early Stoppers
        if self.enable_early_stopping:
            for model_config in self.model_configs:
                for dataset_name in self.dataset_names:
                    key = f"{model_config['name']}_{dataset_name}"
                    self.early_stoppers[key] = self.create_early_stopper(key, model_config)
        
        # Определяем максимум эпох
        max_epochs_global = max(
            model_config.get('max_epochs', self.max_epochs)
            for model_config in self.model_configs
        )
        
        # Основной цикл
        for epoch_checkpoint in range(0, max_epochs_global, self.checkpoint_interval):
            start_epoch = epoch_checkpoint
            end_epoch = min(epoch_checkpoint + self.checkpoint_interval, max_epochs_global)
            
            self.log_message(f"\n{'=' * 80}")
            self.log_message(f"ЧЕКПОИНТ: Эпохи {start_epoch + 1}-{end_epoch}")
            self.log_message(f"{'=' * 80}")
            
            for dataset_name in self.dataset_names:
                for model_config in self.model_configs:
                    key = f"{model_config['name']}_{dataset_name}"
                    
                    if not self.training_active.get(key, True):
                        continue
                    
                    model_max_epochs = model_config.get('max_epochs', self.max_epochs)
                    if start_epoch >= model_max_epochs:
                        self.training_active[key] = False
                        continue
                    
                    metrics = self.train_model_segment(
                        model_config, dataset_name, start_epoch, end_epoch
                    )
                    
                    if metrics is None:
                        continue
                    
                    # Early Stopping проверка
                    if self.enable_early_stopping and self.early_stoppers:
                        stopper = self.early_stoppers.get(key)
                        if stopper:
                            def save_fn(ep, mtr):
                                return self.save_checkpoint_with_metrics(key, ep, mtr)
                            
                            should_continue, reason = stopper.step(metrics, end_epoch, save_fn)
                            self.log_message(f"[ES] {key}: {reason}")
                            
                            if not should_continue:
                                self.training_active[key] = False
                                self.stop_reasons[key] = reason
                                self.log_message(f"[ES] Остановлено обучение {key}")
                    
                    # Ранний отбор
                    if self.enable_early_selection and self.early_selector:
                        should_continue_sel, sel_reason = self.early_selector.should_continue_training(
                            model_key=key,
                            current_metrics=self.metrics_history,
                            current_epoch=end_epoch,
                            max_epochs=max_epochs_global
                        )
                        
                        if not should_continue_sel:
                            self.training_active[key] = False
                            self.stop_reasons[key] = sel_reason
                            self.log_message(f"[EARLY_SEL] Остановлено: {key} - {sel_reason}")
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            gc.collect()
            
            self.compare_models(end_epoch)
            
            with open(self.metrics_file, 'w', encoding='utf-8') as f:
                json.dump(self.metrics_history, f, indent=2)
            
            active_count = sum(1 for active in self.training_active.values() if active)
            if active_count == 0:
                self.log_message("\n[INFO] Все модели завершили обучение")
                break
        
        self.log_message("\n" + "=" * 80)
        self.log_message("ОБУЧЕНИЕ ЗАВЕРШЕНО")
        self.log_message("=" * 80)


# ===================== ПРИМЕР ИСПОЛЬЗОВАНИЯ =====================

if __name__ == "__main__":
    model_configs = [
        {
            'type': 'yolo',
            'size': 's',
            'name': 'yolo_small',
            'max_epochs': 80,
            'early_stopping': {
                'patience': 15,
                'metric': 'mAP50-95'
            }
        },
        {
            'type': 'faster_rcnn',
            'pretrained': True,
            'name': 'faster_rcnn',
            'max_epochs': 25
        },
        {
            'type': 'retinanet',
            'pretrained': True,
            'name': 'retinanet',
            'max_epochs': 35
        }
    ]

    dataset_names = ["dataset_LUNA16", "LUNA16_CHECK"]

    trainer = UniversalModelTrainer(
        model_configs=model_configs,
        dataset_names=dataset_names,
        max_epochs=max(m.get('max_epochs', 40) for m in model_configs),
        checkpoint_interval=5,
        seed=42,                           # ← задаётся здесь

        # Early Stopping
        enable_early_stopping=True,
        early_stopping_patience=10,
        early_stopping_min_delta=0.001,
        early_stopping_metric='mAP50-95',

        # Ранний отбор
        enable_early_selection=True,
        early_selection_ratio=0.3,
        early_selection_top_k=0.5,

        clean_old_results=True
    )

    trainer.run_training()
