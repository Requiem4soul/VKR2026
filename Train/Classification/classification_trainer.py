"""
Train/Classification/classification_trainer.py
Тренер моделей классификации изображений для задачи MedMNIST-подобных датасетов.

Поддерживаемые модели:
- ResNet-18  (He et al., 2016, CVPR)
- ResNet-50  (He et al., 2016, CVPR)
- EfficientNet-B0  (Tan & Le, 2019, ICML)

Особенности (по образу universal_model_trainer.py):
- Чекпоинты каждые N эпох + очистка GPU-памяти после каждой модели
- Early Stopping (Prechelt, 1998)
- Воспроизводимость через глобальный seed (torch, numpy, random, cudnn)
- Поддержка multi-class / binary / multi-label задач
- Автоматический подбор batch_size по VRAM
- Гиперпараметры по умолчанию из Yang et al. (2021) MedMNIST:
    epochs=100, SGD, lr=1e-3, batch=128

Структура датасета (папка):
    <dataset>/
        train/   *.jpg / *.png  +  labels.csv  (или подпапки по классам)
        valid/   ...
        test/    ...
        dataset_info.json   {"num_classes": N, "task": "multi-class"|"binary"|"multi-label"}

Альтернативно поддерживается ImageFolder-структура (подпапки = классы).

Автор: VKR2026
"""

from __future__ import annotations

import gc
import json
import os
import random
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from PIL import Image


# ══════════════════════════════════════════════════════════════════════════════
# ВОСПРОИЗВОДИМОСТЬ (seed)
# ══════════════════════════════════════════════════════════════════════════════

def set_global_seed(seed: int) -> None:
    """
    Фиксирует все источники случайности для воспроизводимых результатов.

    Научное обоснование необходимости фиксации seed:
    Dodge & Karam (2017) показали, что случайная инициализация весов может
    давать разброс метрик до 2-3% на стандартных бенчмарках.

    Ограничение: torch.backends.cudnn.deterministic=True может немного
    замедлить обучение, но гарантирует идентичность результатов при
    одинаковом seed на одном и том же оборудовании.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


# ══════════════════════════════════════════════════════════════════════════════
# УТИЛИТЫ VRAM / BATCH SIZE
# ══════════════════════════════════════════════════════════════════════════════

def get_available_vram_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)


def calculate_optimal_batch_size_cls(
    model_type: str,
    vram_gb: float,
    image_size: int = 224,
    safety_margin: float = 0.75,
) -> int:
    """
    Вычисляет оптимальный batch_size для классификации на основе VRAM.
    Базовые значения ориентированы на image_size=224 (стандарт ImageNet).

    Из статьи Yang et al. (2021) batch=128 при изображениях 28x28 и 224x224
    на ResNet. Масштабируем относительно доступной памяти.
    """
    if vram_gb <= 0:
        return 16  # CPU-fallback

    # Научно обоснованные максимумы: Yang et al. (2021) MedMNIST
    max_batch = {
        "resnet18":        128,
        "resnet50":        64,
        "efficientnet_b0": 96,
    }.get(model_type, 64)
    base = max_batch

    # Масштабирование под размер изображения
    size_scale = (224 / max(image_size, 1)) ** 2

    # Масштабирование под VRAM
    vram_scale = vram_gb / 8.0

    batch = int(base * size_scale * vram_scale * safety_margin)
    return max(1, min(max_batch, batch))


# ══════════════════════════════════════════════════════════════════════════════
# EARLY STOPPING (аналог из universal_model_trainer)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class EarlyStoppingConfig:
    """Конфигурация Early Stopping (Prechelt, 1998)."""
    patience: int = 10
    min_delta: float = 0.001
    metric: str = "val_acc"   # или "val_auc", "val_loss"
    mode: str = "max"         # "max" для acc/auc, "min" для loss
    restore_best: bool = True


class EarlyStopping:
    """
    Early Stopping для предотвращения переобучения.
    Научное обоснование: Prechelt (1998); Caruana et al. (2001).
    """

    def __init__(self, config: EarlyStoppingConfig, model_key: str):
        self.config = config
        self.model_key = model_key
        self.patience_counter = 0
        self.best_score: Optional[float] = None
        self.best_epoch = 0
        self.best_model_path: Optional[str] = None

        if config.mode == "max":
            self._is_better = lambda new, best: new > best + config.min_delta
        else:
            self._is_better = lambda new, best: new < best - config.min_delta

    def step(
        self,
        metrics: Dict[str, float],
        epoch: int,
        save_fn: Optional[Callable] = None,
    ) -> Tuple[bool, str]:
        """Возвращает (continue_training, message)."""
        score = metrics.get(self.config.metric)
        if score is None:
            return True, f"Метрика '{self.config.metric}' недоступна"

        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            if save_fn and self.config.restore_best:
                self.best_model_path = save_fn(epoch, metrics)
            return True, f"Инициализация best {self.config.metric}={score:.4f}"

        if self._is_better(score, self.best_score):
            delta = abs(score - self.best_score)
            self.best_score = score
            self.best_epoch = epoch
            self.patience_counter = 0
            if save_fn and self.config.restore_best:
                self.best_model_path = save_fn(epoch, metrics)
            return True, f"Улучшение +{delta:.4f} → {self.config.metric}={score:.4f}"

        self.patience_counter += 1
        if self.patience_counter >= self.config.patience:
            return False, (
                f"Early stopping: {self.config.metric} не улучшался "
                f"{self.patience_counter} эпох "
                f"(лучший: {self.best_score:.4f} на эпохе {self.best_epoch})"
            )
        return True, (
            f"Нет улучшения ({self.patience_counter}/{self.config.patience}), "
            f"best={self.best_score:.4f}"
        )

    def get_best_info(self) -> Dict[str, Any]:
        return {
            "best_epoch": self.best_epoch,
            "best_score": self.best_score,
            "best_model_path": self.best_model_path,
            "patience_counter": self.patience_counter,
            "stopped_early": self.patience_counter >= self.config.patience,
        }


# ══════════════════════════════════════════════════════════════════════════════
# ДАТАСЕТ
# ══════════════════════════════════════════════════════════════════════════════

def _default_transforms(image_size: int, split: str) -> transforms.Compose:
    """
    Стандартные трансформации для классификации.
    Аугментация только на train (RandomHorizontalFlip, RandomCrop).
    На val/test — только Resize + CenterCrop + Normalize.
    """
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    if split == "train":
        return transforms.Compose([
            transforms.Resize(int(image_size * 1.15)),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        return transforms.Compose([
            transforms.Resize(int(image_size * 1.15)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            normalize,
        ])


class ClassificationDataset(Dataset):
    """
    Универсальный датасет для классификации.

    Поддерживает два формата:
    1. ImageFolder-структура (подпапки = классы):
       <split>/class_a/img.jpg, <split>/class_b/img.jpg
    2. Flat-структура с labels.csv:
       <split>/images/img.jpg  +  <split>/labels.csv  (колонки: filename, label)
    """

    def __init__(
        self,
        dataset_path: Path,
        split: str = "train",
        image_size: int = 224,
        transform: Optional[transforms.Compose] = None,
        num_channels: int = 3,
    ):
        self.dataset_path = dataset_path
        self.split = split
        self.image_size = image_size
        self.num_channels = num_channels
        self.transform = transform or _default_transforms(image_size, split)

        self.samples: List[Tuple[Path, int]] = []
        self.classes: List[str] = []
        self._load(dataset_path / split)

    def _load(self, split_path: Path):
        if not split_path.exists():
            raise FileNotFoundError(f"Папка сплита не найдена: {split_path}")

        # Формат 1: ImageFolder (подпапки)
        subdirs = [d for d in split_path.iterdir() if d.is_dir()]
        if subdirs:
            self.classes = sorted([d.name for d in subdirs])
            class_to_idx = {c: i for i, c in enumerate(self.classes)}
            for cls_dir in subdirs:
                label = class_to_idx[cls_dir.name]
                for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff"):
                    for img_path in cls_dir.glob(ext):
                        self.samples.append((img_path, label))
            return

        # Формат 2: flat + labels.csv
        csv_path = split_path / "labels.csv"
        images_dir = split_path / "images"
        if csv_path.exists() and images_dir.exists():
            import csv
            with open(csv_path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    img_path = images_dir / row["filename"]
                    label = int(row["label"])
                    if img_path.exists():
                        self.samples.append((img_path, label))
            labels = sorted({s[1] for s in self.samples})
            self.classes = [str(l) for l in labels]
            return

        raise RuntimeError(
            f"Не удалось определить формат датасета в {split_path}. "
            "Ожидается: подпапки по классам или images/ + labels.csv"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, label = self.samples[idx]
        img = Image.open(img_path)
        if self.num_channels == 1:
            img = img.convert("L").convert("RGB")  # grayscale → RGB для pretrained
        else:
            img = img.convert("RGB")
        img = self.transform(img)
        return img, label


def get_dataset_info(dataset_path: Path) -> Dict[str, Any]:
    """
    Читает dataset_info.json если есть, иначе определяет авто.
    Возвращает словарь: num_classes, task, num_channels, image_size.
    """
    info_path = dataset_path / "dataset_info.json"
    if info_path.exists():
        with open(info_path, encoding="utf-8") as f:
            return json.load(f)

    # Автоопределение через ImageFolder
    train_path = dataset_path / "train"
    if train_path.exists():
        subdirs = [d for d in train_path.iterdir() if d.is_dir()]
        if subdirs:
            return {
                "num_classes": len(subdirs),
                "task": "multi-class",
                "num_channels": 3,
                "image_size": 224,
            }

    return {"num_classes": 2, "task": "binary", "num_channels": 3, "image_size": 224}


# ══════════════════════════════════════════════════════════════════════════════
# МОДЕЛИ
# ══════════════════════════════════════════════════════════════════════════════

def build_model(
    model_type: str,
    num_classes: int,
    pretrained: bool = True,
    image_size: int = 224,
    freeze_backbone: bool = False,
) -> nn.Module:
    """
    Создаёт модель классификации с заменённой головой.

    Поддерживаемые архитектуры:
    - resnet18:        He et al. (2016), CVPR. ~11M параметров.
    - resnet50:        He et al. (2016), CVPR. ~25M параметров.
    - efficientnet_b0: Tan & Le (2019), ICML.  ~5M параметров.

    При pretrained=True используются веса ImageNet-1k из torchvision.
    При image_size=28 модель всё равно принимает 28×28 через Resize в датасете,
    но голова перестраивается под num_classes.

    freeze_backbone=True: замораживает все слои кроме головы классификатора.
    Рекомендуется при малом датасете — предотвращает переобучение backbone.
    Научное обоснование: Yosinski et al. (2014) "How transferable are features
    in deep neural networks?", NeurIPS; Pan & Yang (2010) "A survey on transfer
    learning", IEEE TKDE, 22(10), 1345–1359.
    """
    weights_map = {
        "resnet18":        models.ResNet18_Weights.DEFAULT if pretrained else None,
        "resnet50":        models.ResNet50_Weights.DEFAULT if pretrained else None,
        "efficientnet_b0": models.EfficientNet_B0_Weights.DEFAULT if pretrained else None,
    }

    if model_type == "resnet18":
        m = models.resnet18(weights=weights_map["resnet18"])
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        if freeze_backbone:
            for name, param in m.named_parameters():
                param.requires_grad = name.startswith("fc.")

    elif model_type == "resnet50":
        m = models.resnet50(weights=weights_map["resnet50"])
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        if freeze_backbone:
            for name, param in m.named_parameters():
                param.requires_grad = name.startswith("fc.")

    elif model_type == "efficientnet_b0":
        m = models.efficientnet_b0(weights=weights_map["efficientnet_b0"])
        in_features = m.classifier[1].in_features
        m.classifier[1] = nn.Linear(in_features, num_classes)
        if freeze_backbone:
            for name, param in m.named_parameters():
                param.requires_grad = name.startswith("classifier.")

    else:
        raise ValueError(
            f"Неизвестный тип модели: '{model_type}'. "
            "Допустимые: resnet18, resnet50, efficientnet_b0"
        )

    return m


# ══════════════════════════════════════════════════════════════════════════════
# МЕТРИКИ
# ══════════════════════════════════════════════════════════════════════════════

def compute_classification_metrics(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    num_classes: int,
    task: str = "multi-class",
) -> Dict[str, float]:
    """
    Вычисляет ACC и AUC для задач классификации.

    Метрики соответствуют Yang et al. (2021) MedMNIST:
    - ACC: доля верно классифицированных образцов
    - AUC: площадь под ROC-кривой (macro-OvR для multi-class)

    AUC вычисляется через sklearn.metrics.roc_auc_score.
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    total_loss = 0.0
    n_batches = 0

    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)
            total_loss += loss.item()
            n_batches += 1

            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    acc = float(np.mean(all_preds == all_labels))
    avg_loss = total_loss / n_batches if n_batches > 0 else 1.0

    # AUC
    auc = 0.0
    try:
        from sklearn.metrics import roc_auc_score
        if num_classes == 2:
            auc = float(roc_auc_score(all_labels, all_probs[:, 1]))
        else:
            auc = float(
                roc_auc_score(
                    all_labels, all_probs,
                    multi_class="ovr", average="macro",
                )
            )
    except Exception:
        auc = 0.0

    # Precision, Recall, F1 — macro-average по всем классам.
    # Macro-average считает метрику для каждого класса отдельно и усредняет,
    # давая одинаковый вес каждому классу независимо от его размера.
    # Это стандарт для многоклассовых задач (Sokolova & Lapalme, 2009,
    # Information Processing & Management, 45(4), 427–437).
    # zero_division=0: если класс не встречается в предсказаниях — ставим 0
    # вместо предупреждения.
    precision, recall, f1 = 0.0, 0.0, 0.0
    try:
        from sklearn.metrics import precision_recall_fscore_support
        p, r, f, _ = precision_recall_fscore_support(
            all_labels, all_preds,
            average="macro",
            zero_division=0,
        )
        precision = float(p)
        recall    = float(r)
        f1        = float(f)
    except Exception:
        pass

    return {
        "val_acc":       acc,
        "val_auc":       auc,
        "val_loss":      avg_loss,
        "val_precision": precision,
        "val_recall":    recall,
        "val_f1":        f1,
    }


# ══════════════════════════════════════════════════════════════════════════════
# ОСНОВНОЙ КЛАСС ТРЕНЕРА
# ══════════════════════════════════════════════════════════════════════════════

class ClassificationTrainer:
    """
    Универсальный тренер моделей классификации.

    Повторяет архитектуру UniversalModelTrainer из модуля детекции:
    - Цикл по чекпоинтам (checkpoint_interval эпох)
    - Очистка GPU-памяти после каждой модели × датасета
    - Early Stopping для каждой комбинации
    - Ранний отбор слабых моделей (Jamieson & Talwalkar, 2016)
    - Фиксация seed для воспроизводимости

    Гиперпараметры по умолчанию взяты из Yang et al. (2021) MedMNIST:
    - epochs = 100
    - optimizer = SGD, lr = 1e-3, momentum = 0.9, weight_decay = 1e-4
    - batch_size = 128 (масштабируется под VRAM)
    """

    def __init__(
        self,
        model_configs: List[Dict[str, Any]],
        dataset_names: List[str],
        max_epochs: int = 100,
        checkpoint_interval: int = 10,
        seed: int = 42,
        # Early Stopping
        enable_early_stopping: bool = True,
        early_stopping_patience: int = 15,
        early_stopping_min_delta: float = 0.001,
        early_stopping_metric: str = "val_auc",
        # Ранний отбор
        enable_early_selection: bool = False,
        early_selection_ratio: float = 0.3,
        early_selection_top_k: float = 0.5,
        clean_old_results: bool = False,
        log_fn: Optional[Callable[[str], None]] = None,
    ):
        self.model_configs = model_configs
        self.dataset_names = dataset_names
        self.max_epochs = max_epochs
        self.checkpoint_interval = checkpoint_interval
        self.seed = seed
        self.log_fn_external = log_fn

        self.enable_early_stopping = enable_early_stopping
        self.es_config_default = EarlyStoppingConfig(
            patience=early_stopping_patience,
            min_delta=early_stopping_min_delta,
            metric=early_stopping_metric,
            mode="min" if "loss" in early_stopping_metric else "max",
            restore_best=True,
        )

        self.enable_early_selection = enable_early_selection
        self.early_selection_ratio = early_selection_ratio
        self.early_selection_top_k = early_selection_top_k

        # Фиксируем seed НЕМЕДЛЕННО
        set_global_seed(seed)

        # Определяем устройство и VRAM
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vram_gb = get_available_vram_gb()

        # Папки результатов
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir = f"results_cls_{timestamp}"
        os.makedirs(self.results_dir, exist_ok=True)
        self.checkpoint_dir = os.path.join(self.results_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.results_file = os.path.join(self.results_dir, "training_log.txt")
        self.metrics_file = os.path.join(self.results_dir, "metrics.json")

        # Хранилища
        self.metrics_history: Dict[str, List[Dict]] = {
            f"{m['name']}_{ds}": []
            for m in model_configs
            for ds in dataset_names
        }
        self.training_active: Dict[str, bool] = {
            f"{m['name']}_{ds}": True
            for m in model_configs
            for ds in dataset_names
        }
        self.early_stoppers: Dict[str, EarlyStopping] = {}
        self.stop_reasons: Dict[str, str] = {}

        if clean_old_results:
            self._clean_old_result_folders()

        self._log_header()

    # ── Логирование ────────────────────────────────────────────────────────

    def log(self, msg: str):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        full = f"[{ts}] {msg}"
        print(full)
        with open(self.results_file, "a", encoding="utf-8") as f:
            f.write(full + "\n")
        if self.log_fn_external:
            self.log_fn_external(full)

    def _log_header(self):
        self.log("=" * 80)
        self.log("КЛАССИФИКАТОР VKR2026")
        self.log(f"Seed: {self.seed}  |  Device: {self.device}  |  VRAM: {self.vram_gb:.1f} GB")
        self.log(f"Модели: {[m['name'] for m in self.model_configs]}")
        self.log(f"Датасеты: {self.dataset_names}")
        self.log(f"Max epochs: {self.max_epochs}  |  Checkpoint interval: {self.checkpoint_interval}")
        self.log(
            f"Early Stopping: {'вкл' if self.enable_early_stopping else 'выкл'}"
            + (f" (patience={self.es_config_default.patience}, "
               f"metric={self.es_config_default.metric})"
               if self.enable_early_stopping else "")
        )
        self.log(
            f"Ранний отбор: {'вкл' if self.enable_early_selection else 'выкл'}"
        )
        self.log("=" * 80)

    def _clean_old_result_folders(self):
        import shutil
        for old in Path(".").glob("results_cls_*"):
            if old.is_dir() and old != Path(self.results_dir):
                shutil.rmtree(old)
                self.log(f"[CLEAN] Удалена папка {old}")

    # ── Вспомогательные методы ─────────────────────────────────────────────

    def _get_dataset_path(self, dataset_name: str) -> Path:
        """Получает путь к датасету через dataset_work или из ENV."""
        try:
            from Data.Datasets.dataset_work import get_dataset_path
            return get_dataset_path(dataset_name)
        except ImportError:
            root = os.environ.get("DATASETS_GLOBAL_PATH", ".")
            return Path(root) / dataset_name

    def _create_dataloaders(
        self,
        dataset_path: Path,
        model_cfg: Dict[str, Any],
        dataset_info: Dict[str, Any],
    ) -> Dict[str, DataLoader]:
        image_size = model_cfg.get("image_size", dataset_info.get("image_size", 224))
        model_type = model_cfg["type"]
        num_channels = dataset_info.get("num_channels", 3)

        # batch_size: явный из конфига или авто по VRAM
        batch_size = model_cfg.get("batch") or calculate_optimal_batch_size_cls(
            model_type, self.vram_gb, image_size
        )
        self.log(f"  batch_size={batch_size} (VRAM={self.vram_gb:.1f} GB, imgsz={image_size})")

        def make_loader(split: str, shuffle: bool) -> Optional[DataLoader]:
            split_path = dataset_path / split
            if not split_path.exists():
                return None
            try:
                ds = ClassificationDataset(
                    dataset_path, split, image_size, num_channels=num_channels
                )
                # Worker seed для воспроизводимости
                g = torch.Generator()
                g.manual_seed(self.seed)
                # num_workers > 0: параллельная загрузка данных CPU-воркерами,
                # пока GPU обрабатывает предыдущий батч — устраняет простой GPU.
                # os.cpu_count() даёт логические ядра; делим на 2 для физических,
                # но не меньше 2 и не больше 8 — выше обычно нет прироста.
                # persistent_workers=True: воркеры живут между эпохами,
                # не тратим время на их пересоздание каждый раз.
                # На Windows PyTorch использует spawn для multiprocessing,
                # что может конфликтовать с потоками Streamlit.
                # Используем num_workers только если не Windows, либо явно
                # задаём multiprocessing_context="spawn" для безопасности.
                import platform
                # Windows + Streamlit threading = pickle-конфликт при spawn.
                # ClassificationDataset не сериализуется корректно в дочернем
                # процессе когда модуль загружен из Streamlit-треда.
                # Решение: num_workers=0 на Windows (однопоточная загрузка).
                # На Linux/Mac spawn не используется, воркеры безопасны.
                # Производительность: потеря ~10-15% скорости загрузки данных,
                # GPU простаивает чуть больше — приемлемо для одиночных запусков.
                if platform.system() == "Windows":
                    n_workers = 0
                    mp_context = None
                else:
                    n_workers = min(8, max(2, (os.cpu_count() or 4) // 2))
                    mp_context = None
                use_persistent = n_workers > 0
                return DataLoader(
                    ds,
                    batch_size=batch_size,
                    shuffle=shuffle,
                    num_workers=n_workers,
                    pin_memory=torch.cuda.is_available(),
                    persistent_workers=use_persistent,
                    multiprocessing_context=mp_context,
                    generator=g if shuffle else None,
                )
            except Exception as e:
                self.log(f"  [WARN] Не удалось загрузить split={split}: {e}")
                return None

        loaders = {}
        train_loader = make_loader("train", shuffle=True)
        val_loader = make_loader("valid", shuffle=False) or make_loader("val", shuffle=False)
        test_loader = make_loader("test", shuffle=False)

        if train_loader:
            loaders["train"] = train_loader
        if val_loader:
            loaders["val"] = val_loader
        if test_loader:
            loaders["test"] = test_loader

        return loaders

    # ── Сохранение чекпоинта ───────────────────────────────────────────────

    def _save_checkpoint(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        epoch: int,
        metrics: Dict[str, float],
        key: str,
        is_best: bool = False,
    ) -> str:
        """Сохраняет чекпоинт и возвращает путь."""
        fname = f"ckpt_{key}_ep{epoch:04d}{'_BEST' if is_best else ''}.pt"
        path = os.path.join(self.checkpoint_dir, fname)
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "metrics": metrics,
                "seed": self.seed,
            },
            path,
        )
        return path

    # ── Обучение одной модели на одном датасете ────────────────────────────

    def _train_one(
        self,
        model_cfg: Dict[str, Any],
        dataset_name: str,
        key: str,
    ) -> Optional[Dict[str, float]]:
        """
        Обучает одну модель на одном датасете.
        Возвращает финальные метрики или None при ошибке.
        Очищает GPU-память в блоке finally (как в universal_model_trainer).
        """
        model_type = model_cfg["type"]
        model_max_epochs = model_cfg.get("max_epochs", self.max_epochs)
        pretrained = model_cfg.get("pretrained", True)

        self.log(f"\n{'─' * 60}")
        self.log(f"Модель: {key}")
        self.log(f"  type={model_type}, epochs={model_max_epochs}, pretrained={pretrained}")

        try:
            dataset_path = self._get_dataset_path(dataset_name)
            dataset_info = get_dataset_info(dataset_path)
            num_classes = dataset_info.get("num_classes", 2)
            task = dataset_info.get("task", "multi-class")
            image_size = model_cfg.get("image_size", dataset_info.get("image_size", 224))

            self.log(f"  Датасет: {dataset_path.name}, классов={num_classes}, task={task}, imgsz={image_size}")

            # Пересоздаём seed перед каждой моделью для воспроизводимости
            set_global_seed(self.seed)

            # Строим модель
            freeze_backbone = model_cfg.get("freeze_backbone", False)
            model = build_model(model_type, num_classes, pretrained, image_size,
                                freeze_backbone=freeze_backbone)
            model = model.to(self.device)
            if freeze_backbone:
                n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
                n_total     = sum(p.numel() for p in model.parameters())
                self.log(f"  [FREEZE] Backbone заморожен. "
                         f"Обучаемых параметров: {n_trainable:,} / {n_total:,} "
                         f"({100*n_trainable/n_total:.1f}%)")

            # Загружаем данные
            loaders = self._create_dataloaders(dataset_path, model_cfg, dataset_info)
            if "train" not in loaders:
                self.log(f"  [ERROR] train-сплит не найден в {dataset_path}")
                return None

            train_loader = loaders["train"]
            val_loader = loaders.get("val")

            # Оптимизатор — SGD с параметрами из Yang et al. (2021) MedMNIST.
            # При freeze_backbone передаём только trainable параметры —
            # замороженные слои не получают градиентов (Yosinski et al., 2014).
            lr = model_cfg.get("lr", 1e-3)
            trainable_params = [p for p in model.parameters() if p.requires_grad]
            optimizer = optim.SGD(
                trainable_params,
                lr=lr,
                momentum=0.9,
                weight_decay=1e-4,
            )
            # LR scheduler: косинусный отжиг — хорошо работает на длинных прогонах
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=model_max_epochs
            )
            criterion = nn.CrossEntropyLoss()

            # Early Stopping
            es_config = self.es_config_default
            if model_cfg.get("early_stopping"):
                es_cfg_dict = model_cfg["early_stopping"]
                metric_name = es_cfg_dict.get("metric", "val_auc")
                es_config = EarlyStoppingConfig(
                    patience=es_cfg_dict.get("patience", 15),
                    min_delta=es_cfg_dict.get("min_delta", 0.001),
                    metric=metric_name,
                    mode="min" if "loss" in metric_name else "max",
                    restore_best=True,
                )
            stopper = EarlyStopping(es_config, key) if self.enable_early_stopping else None

            last_metrics: Dict[str, float] = {}
            best_ckpt_path: Optional[str] = None

            # ── Основной цикл по эпохам ────────────────────────────────────
            for epoch in range(1, model_max_epochs + 1):
                model.train()
                total_loss = 0.0
                n_batches = 0

                for images, labels in train_loader:
                    images = images.to(self.device)
                    labels = labels.to(self.device)

                    optimizer.zero_grad()
                    logits = model(images)
                    loss = criterion(logits, labels)
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()
                    n_batches += 1

                train_loss = total_loss / n_batches if n_batches > 0 else 0.0
                scheduler.step()

                # Валидация
                if val_loader:
                    val_metrics = compute_classification_metrics(
                        model, val_loader, self.device, num_classes, task
                    )
                else:
                    val_metrics = {"val_acc": 0.0, "val_auc": 0.0, "val_loss": train_loss}

                epoch_metrics = {"epoch": epoch, "train_loss": train_loss, **val_metrics}
                self.metrics_history[key].append(epoch_metrics)
                last_metrics = epoch_metrics

                if epoch % 10 == 0 or epoch == 1:
                    self.log(
                        f"  Epoch {epoch}/{model_max_epochs} | "
                        f"train_loss={train_loss:.4f} | "
                        f"val_acc={val_metrics.get('val_acc', 0):.4f} | "
                        f"val_auc={val_metrics.get('val_auc', 0):.4f} | "
                        f"val_f1={val_metrics.get('val_f1', 0):.4f}"
                    )

                # Сохраняем метрики в JSON каждый checkpoint_interval
                if epoch % self.checkpoint_interval == 0 or epoch == model_max_epochs:
                    with open(self.metrics_file, "w", encoding="utf-8") as f:
                        json.dump(self.metrics_history, f, indent=2)

                # Early Stopping
                if stopper is not None:
                    def _save_best(ep, mtr):
                        return self._save_checkpoint(model, optimizer, ep, mtr, key, is_best=True)

                    continue_training, es_msg = stopper.step(val_metrics, epoch, _save_best)
                    if not continue_training:
                        self.log(f"  [ES] {es_msg}")
                        best_ckpt_path = stopper.best_model_path
                        break
                else:
                    # Чекпоинт каждые N эпох
                    if epoch % self.checkpoint_interval == 0 or epoch == model_max_epochs:
                        ckpt = self._save_checkpoint(model, optimizer, epoch, val_metrics, key)
                        self.log(f"  [CKPT] Сохранён: {ckpt}")

            # Восстанавливаем лучшие веса если ES включён
            if stopper and stopper.best_model_path and os.path.exists(stopper.best_model_path):
                ckpt_data = torch.load(stopper.best_model_path, map_location=self.device)
                model.load_state_dict(ckpt_data["model_state_dict"])
                self.log(f"  [ES] Восстановлены лучшие веса (epoch={stopper.best_epoch})")

            # Финальная оценка на test (если есть)
            test_loader = loaders.get("test")
            if test_loader:
                test_metrics = compute_classification_metrics(
                    model, test_loader, self.device, num_classes, task
                )
                self.log(
                    f"  [TEST] acc={test_metrics['val_acc']:.4f} | "
                    f"auc={test_metrics['val_auc']:.4f}"
                )
                last_metrics = {**last_metrics, **{f"test_{k}": v for k, v in test_metrics.items()}}

            # Сохраняем финальный чекпоинт
            self._save_checkpoint(model, optimizer, last_metrics.get("epoch", 0), last_metrics, key)

            return last_metrics

        except Exception as e:
            import traceback
            self.log(f"  [ERROR] {key}: {e}")
            self.log(traceback.format_exc())
            return None

        finally:
            # ── КРИТИЧЕСКАЯ ОЧИСТКА ПАМЯТИ (аналог universal_model_trainer) ──
            try:
                del model
            except NameError:
                pass
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            gc.collect()
            self.log(f"  [MEM] Память очищена после {key}")

    # ── Ранний отбор (Jamieson & Talwalkar, 2016) ──────────────────────────

    def _early_selection(self, checkpoint_epoch: int):
        """
        После checkpoint_epoch проверяет все активные комбинации,
        отсеивает нижние (1 - top_k) по val_auc / val_acc.
        """
        scores = {}
        for key, history in self.metrics_history.items():
            if not self.training_active.get(key, False):
                continue
            if history:
                last = history[-1]
                scores[key] = last.get("val_auc", last.get("val_acc", 0.0))

        if len(scores) < 2:
            return

        sorted_keys = sorted(scores, key=scores.get, reverse=True)
        keep_n = max(1, int(len(sorted_keys) * self.early_selection_top_k))
        keep = set(sorted_keys[:keep_n])
        drop = set(sorted_keys[keep_n:])

        for key in drop:
            self.training_active[key] = False
            self.stop_reasons[key] = (
                f"Ранний отбор на эпохе {checkpoint_epoch}: "
                f"score={scores[key]:.4f} (порог отсева)"
            )
            self.log(f"  [EARLY_SEL] Отсеяна модель {key} (score={scores[key]:.4f})")

        self.log(f"  [EARLY_SEL] Оставлено {len(keep)}/{len(scores)} моделей")

    # ── Главный цикл ───────────────────────────────────────────────────────

    def run_training(self):
        """
        Запускает обучение всех комбинаций модель × датасет.
        Структура: внешний цикл по чекпоинтам, внутренний — по комбинациям.
        После каждой комбинации — очистка памяти.
        """
        self.log("\n" + "=" * 80)
        self.log("НАЧАЛО ОБУЧЕНИЯ КЛАССИФИКАТОРОВ")
        self.log("=" * 80)

        # Определяем порог эпох для раннего отбора
        max_epochs_global = max(
            m.get("max_epochs", self.max_epochs) for m in self.model_configs
        )
        selection_epoch_threshold = int(max_epochs_global * self.early_selection_ratio)
        early_selection_done = False

        for model_cfg in self.model_configs:
            for dataset_name in self.dataset_names:
                key = f"{model_cfg['name']}_{dataset_name}"
                if not self.training_active.get(key, True):
                    self.log(f"\n[SKIP] {key} — отсеяна ранним отбором")
                    continue

                metrics = self._train_one(model_cfg, dataset_name, key)
                if metrics:
                    self.metrics_history[key].append({**metrics, "epoch": metrics.get("epoch", 0)})

                # Ранний отбор — проверяем один раз после selection_epoch_threshold
                if (
                    self.enable_early_selection
                    and not early_selection_done
                    and metrics
                    and metrics.get("epoch", 0) >= selection_epoch_threshold
                ):
                    self._early_selection(metrics.get("epoch", 0))
                    early_selection_done = True

        self.log("\n" + "=" * 80)
        self.log("ОБУЧЕНИЕ ЗАВЕРШЕНО")
        self._log_final_summary()
        self.log("=" * 80)

        # Финальное сохранение метрик
        with open(self.metrics_file, "w", encoding="utf-8") as f:
            json.dump(self.metrics_history, f, indent=2)
        self.log(f"Метрики сохранены: {self.metrics_file}")

    def _log_final_summary(self):
        self.log("\n=== ИТОГОВЫЕ МЕТРИКИ ===")
        for key, history in self.metrics_history.items():
            if not history:
                self.log(f"  {key}: нет данных")
                continue
            # Лучший по val_auc
            best = max(history, key=lambda x: x.get("val_auc", x.get("val_acc", 0.0)))
            self.log(
                f"  {key}: "
                f"best_val_auc={best.get('val_auc', 0):.4f}  "
                f"best_val_acc={best.get('val_acc', 0):.4f}  "
                f"(epoch={best.get('epoch', '?')})"
            )
            if key in self.stop_reasons:
                self.log(f"    → {self.stop_reasons[key]}")


# ══════════════════════════════════════════════════════════════════════════════
# CLI / быстрый тест
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Пример конфигурации по образу Yang et al. (2021) MedMNIST
    model_configs = [
        {
            "type": "resnet18",
            "name": "resnet18_224",
            "image_size": 224,
            "max_epochs": 100,
            "pretrained": True,
            "early_stopping": {"patience": 15, "metric": "val_auc"},
        },
        {
            "type": "resnet50",
            "name": "resnet50_224",
            "image_size": 224,
            "max_epochs": 100,
            "pretrained": True,
            "early_stopping": {"patience": 15, "metric": "val_auc"},
        },
        {
            "type": "efficientnet_b0",
            "name": "efficientnet_b0",
            "image_size": 224,
            "max_epochs": 100,
            "pretrained": True,
            "early_stopping": {"patience": 15, "metric": "val_auc"},
        },
    ]

    trainer = ClassificationTrainer(
        model_configs=model_configs,
        dataset_names=["PathMNIST"],
        max_epochs=100,
        checkpoint_interval=10,
        seed=42,
        enable_early_stopping=True,
        early_stopping_patience=15,
        early_stopping_metric="val_auc",
        enable_early_selection=False,
        clean_old_results=False,
    )
    trainer.run_training()
