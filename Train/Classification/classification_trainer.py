from __future__ import annotations

import gc
import json
import os
import random
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from PIL import Image

# Воспроизводимость

def set_global_seed(seed: int) -> None:
    """
    Фиксирует все источники случайности для воспроизводимых результатов
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


# Утилиты

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
    Базовые значения ориентированы на image_size=224 (стандарт ImageNet)
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


# Раняя остановка

@dataclass
class EarlyStoppingConfig:
    """Конфигурация Early Stopping"""
    patience: int = 10
    min_delta: float = 0.0001
    metric: str = "val_acc"   # или "val_auc", "val_loss"
    mode: str = "max"         # "max" для acc/auc, "min" для loss
    restore_best: bool = True


class EarlyStopping:
    """
    Early Stopping для предотвращения переобучения.
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

class DataPrefetcher:
    """
    Однопоточный prefetcher для DataLoader.

    Проблема: при num_workers=0 на Windows DataLoader загружает батчи
    синхронно — GPU простаивает пока CPU читает и трансформирует изображения.
    """

    def __init__(self, loader: DataLoader, device: torch.device):
        self.loader   = loader
        self.device   = device
        self._iter    = None
        self._next_data: Optional[tuple] = None
        self._thread: Optional[threading.Thread] = None
        self._error: Optional[BaseException] = None
        self._done    = False

    def __iter__(self):
        self._iter  = iter(self.loader)
        self._done  = False
        self._error = None
        self._next_data = None
        # Загружаем первый батч синхронно чтобы сразу начать
        self._prefetch()
        return self

    def _prefetch(self):
        """Запускает загрузку следующего батча в фоновом потоке."""
        if self._done:
            return
        def _load():
            try:
                self._next_data = next(self._iter)
            except StopIteration:
                self._done = True
                self._next_data = None
            except Exception as e:
                self._error = e
                self._done  = True
                self._next_data = None
        self._thread = threading.Thread(target=_load, daemon=True)
        self._thread.start()

    def __next__(self):
        # Ждём пока фоновый поток закончит загрузку
        if self._thread is not None:
            self._thread.join()
            self._thread = None

        # Пробрасываем исключение из фонового потока если было
        if self._error is not None:
            err = self._error
            self._error = None
            raise RuntimeError(f"DataPrefetcher: ошибка загрузки батча: {err}") from err

        if self._done or self._next_data is None:
            raise StopIteration

        # Берём загруженный батч
        images, labels = self._next_data
        self._next_data = None

        # Запускаем загрузку СЛЕДУЮЩЕГО батча в фоне
        # (пока главный поток будет делать .to(device) и forward/backward)
        self._prefetch()

        # Перекладываем на GPU (это быстро — просто pinned memory transfer)
        images = images.to(self.device, non_blocking=True)
        labels = labels.to(self.device, non_blocking=True)
        return images, labels

    def __len__(self):
        return len(self.loader)


# Работа с датасетами

def _default_transforms(image_size: int, split: str) -> transforms.Compose:
    """
    Трансформации для классификации без аугментации.
    """
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    return transforms.Compose([
        transforms.Resize(int(image_size * 1.15)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        normalize,
    ])


class ClassificationDataset(Dataset):
    """
    Универсальный датасет для классификации.
    """

    def __init__(
        self,
        dataset_path: Path,
        split: str = "train",
        image_size: int = 224,
        transform: Optional[transforms.Compose] = None,
        num_channels: int = 3,
        cache_in_memory: bool = False,
    ):
        self.dataset_path = dataset_path
        self.split = split
        self.image_size = image_size
        self.num_channels = num_channels
        self.transform = transform or _default_transforms(image_size, split)
        self.cache_in_memory = cache_in_memory

        # Кеш: idx → PIL.Image (уже convert-нутый, до transform).
        # Хранится как PIL а не тензор — transform включает аугментации
        # (RandomCrop, RandomHorizontalFlip) которые должны применяться заново
        # на каждом __getitem__, иначе аугментация теряется.
        self._cache: Dict[int, "Image.Image"] = {}

        self.samples: List[Tuple[Path, int]] = []
        self.classes: List[str] = []
        self._load(dataset_path / split)

        if cache_in_memory:
            self._preload_to_cache()

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

    def _preload_to_cache(self):
        """
        Загружает все изображения в RAM (PIL, после предварительного Resize).
        Вызывается один раз при создании датасета если cache_in_memory=True.
        """
        # Размер для предварительного Resize: чуть больше image_size
        # чтобы CenterCrop в transform работал корректно (аналог _default_transforms).
        pre_resize = int(self.image_size * 1.15)
        for idx, (img_path, _) in enumerate(self.samples):
            try:
                img = Image.open(img_path)
                # Конвертируем канальность
                if self.num_channels == 1:
                    img = img.convert("L").convert("RGB")
                else:
                    img = img.convert("RGB")
                # Resize ДО кеширования — ключевое отличие от старого бага.
                # Используем BILINEAR (быстрее BICUBIC, качество достаточное
                # для промежуточного кеша — финальный CenterCrop в transform
                # не добавляет дополнительных артефактов).
                img = img.resize((pre_resize, pre_resize), Image.BILINEAR)
                img.load()  # принудительно читает данные с диска в RAM
                self._cache[idx] = img
            except Exception:
                # Если изображение не удалось загрузить — оставляем без кеша,
                # __getitem__ прочитает его с диска как обычно
                pass

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, label = self.samples[idx]
        if idx in self._cache:
            # Берём из кеша — копия чтобы PIL не мутировал в трансформах
            img = self._cache[idx].copy()
        else:
            img = Image.open(img_path)
            if self.num_channels == 1:
                img = img.convert("L").convert("RGB")
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


# Модели

def build_model(
    model_type: str,
    num_classes: int,
    pretrained: bool = True,
    image_size: int = 224,
    freeze_backbone: bool = False,
) -> nn.Module:
    """
    Создаёт модель классификации с заменённой головой
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

# Метрики

def compute_classification_metrics(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    num_classes: int,
    task: str = "multi-class",
) -> Dict[str, float]:
    """
    Вычисляет ACC и AUC для задач классификации
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


def _get_class_weights(
    dataset: "ClassificationDataset",
    num_classes: int,
    device: torch.device,
) -> Optional[torch.Tensor]:
    """
    Вычисляет веса классов для CrossEntropyLoss на основе частот в train-сплите
    """
    try:
        from collections import Counter
        labels = [s[1] for s in dataset.samples]
        counts = Counter(labels)
        total = len(labels)
        # Если все классы представлены одинаково — веса единичные, лосс не меняется
        if len(set(counts.values())) == 1:
            return None
        weights = torch.tensor(
            [total / (num_classes * max(counts.get(i, 1), 1))
             for i in range(num_classes)],
            dtype=torch.float32,
            device=device,
        )
        return weights
    except Exception:
        return None


# Основной класс

class ClassificationTrainer:
    """
    Универсальный тренер моделей классификации.
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
        early_stopping_min_delta: float = 0.0001,
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
        # Пути к последним сохранённым чекпоинтам — для warm-start автоподбора
        self.last_checkpoint_paths: Dict[str, str] = {}

        if clean_old_results:
            self._clean_old_result_folders()

        self._log_header()

    # Логирование

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

    # Вспомогательные методы

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

        # Оцениваем объём датасета для решения о кешировании в RAM
        pre_resize_size = int(image_size * 1.15)

        def _estimate_cache_ram_gb(n_samples: int) -> float:
            return n_samples * (pre_resize_size ** 2) * 3 / (1024 ** 3)

        # Определяем допустимый порог RAM для кеша
        try:
            import psutil
            available_gb = psutil.virtual_memory().available / (1024 ** 3)
            ram_cache_limit_gb = max(1.0, min(16.0, available_gb * 0.70))
        except ImportError:
            # psutil не установлен — используем консервативный порог
            ram_cache_limit_gb = 2.0

        def make_loader(split: str, shuffle: bool) -> Optional[DataLoader]:
            split_path = dataset_path / split
            if not split_path.exists():
                return None
            try:
                # Считаем число образцов без загрузки изображений —
                # просто считаем файлы чтобы оценить RAM.
                n_approx = sum(
                    1 for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff")
                    for _ in split_path.rglob(ext)
                ) if split_path.exists() else 0

                # Кешируем только train и только если хватает RAM.
                # val/test маленькие — там DataPrefetcher достаточен.
                ram_gb_needed = _estimate_cache_ram_gb(n_approx)
                use_cache = (split == "train") and (ram_gb_needed <= ram_cache_limit_gb)

                if split == "train":
                    if use_cache:
                        self.log(
                            f"  [CACHE] train-сплит: {n_approx} изображений, "
                            f"~{ram_gb_needed:.2f} GB → кешируем в RAM "
                            f"(лимит={ram_cache_limit_gb:.1f} GB, ускорение загрузки)"
                        )
                    else:
                        self.log(
                            f"  [CACHE] train-сплит: {n_approx} изображений, "
                            f"~{ram_gb_needed:.2f} GB → превышает лимит RAM "
                            f"({ram_cache_limit_gb:.1f} GB), загружаем с диска"
                        )

                ds = ClassificationDataset(
                    dataset_path, split, image_size, num_channels=num_channels,
                    cache_in_memory=use_cache,
                )
                # Worker seed для воспроизводимости
                g = torch.Generator()
                g.manual_seed(self.seed)
                # num_workers > 0: параллельная загрузка данных CPU-воркерами
                import platform
                # Windows + Streamlit threading = pickle-конфликт при spawn
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

    # Сохранение чекпоинта

    def _save_checkpoint(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        epoch: int,
        metrics: Dict[str, float],
        key: str,
        is_best: bool = False,
        scheduler=None,
    ) -> str:
        """Сохраняет чекпоинт и возвращает путь."""
        fname = f"ckpt_{key}_ep{epoch:04d}{'_BEST' if is_best else ''}.pt"
        path = os.path.join(self.checkpoint_dir, fname)
        payload = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
            "seed": self.seed,
        }
        if scheduler is not None:
            payload["scheduler_state_dict"] = scheduler.state_dict()
        torch.save(payload, path)
        return path

    # Обучение одной модели на одном датасете

    def _train_one(
        self,
        model_cfg: Dict[str, Any],
        dataset_name: str,
        key: str,
        resume_from_path: Optional[str] = None,
    ) -> Optional[Dict[str, float]]:
        """
        Обучает одну модель на одном датасете
        Возвращает финальные метрики или None при ошибке
        Очищает GPU-память в блоке finally (как в universal_model_trainer)
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

            base_model = model
            if model_cfg.get("use_torch_compile", False) and \
               torch.cuda.is_available() and hasattr(torch, "compile") and \
               "efficientnet" not in model_type.lower():
                try:
                    model = torch.compile(model)
                    self.log("  [COMPILE] torch.compile активирован")
                except Exception as _e:
                    self.log(f"  [COMPILE] torch.compile недоступен: {_e} — продолжаем без компиляции")
                    model = base_model

            # Загружаем данные
            loaders = self._create_dataloaders(dataset_path, model_cfg, dataset_info)
            if "train" not in loaders:
                self.log(f"  [ERROR] train-сплит не найден в {dataset_path}")
                return None

            train_loader = loaders["train"]
            val_loader = loaders.get("val")

            # Оптимизатор — SGD с параметрами
            lr = model_cfg.get("lr", 1e-4 if pretrained else 1e-3)
            trainable_params = [p for p in model.parameters() if p.requires_grad]
            optimizer = optim.SGD(
                trainable_params,
                lr=lr,
                momentum=0.9,
                weight_decay=1e-4,
            )

            criterion = nn.CrossEntropyLoss(weight=_get_class_weights(
                loaders["train"].dataset, num_classes, self.device
            ))

            resume_start_epoch = 0

            if resume_from_path and os.path.exists(resume_from_path):
                try:
                    ckpt = torch.load(resume_from_path, map_location=self.device)

                    # (a) Веса модели
                    base_model.load_state_dict(ckpt["model_state_dict"])

                    # (b) Optimizer state (momentum buffers)
                    optimizer.load_state_dict(ckpt["optimizer_state_dict"])

                    # (c) Сброс lr в исходное значение.
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr

                    resume_start_epoch = ckpt.get("epoch", 0)
                    self.log(
                        f"  [RESUME] Загружен: {os.path.basename(resume_from_path)} "
                        f"(эпоха {resume_start_epoch}, lr сброшен в {lr:.2e})"
                    )
                except Exception as _re:
                    self.log(
                        f"  [RESUME] Не удалось загрузить чекпоинт: {_re} "
                        f"— обучаем с нуля"
                    )
                    resume_start_epoch = 0
            elif resume_from_path:
                self.log(
                    f"  [RESUME] Чекпоинт не найден: {resume_from_path} "
                    f"— обучаем с нуля"
                )

            remaining_epochs = max(1, model_max_epochs - resume_start_epoch)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=remaining_epochs,
                # eta_min=0 по умолчанию — стандартное значение из
                # Loshchilov & Hutter (2017).
            )

            # AMP (Automatic Mixed Precision): ускорение за счёт fp16 на GPU
            use_amp = torch.cuda.is_available()
            scaler  = GradScaler(device="cuda", enabled=use_amp)

            # Early Stopping
            es_config = self.es_config_default
            if model_cfg.get("early_stopping"):
                es_cfg_dict = model_cfg["early_stopping"]
                metric_name = es_cfg_dict.get("metric", "val_auc")
                es_config = EarlyStoppingConfig(
                    patience=es_cfg_dict.get("patience", 15),
                    min_delta=es_cfg_dict.get("min_delta", 0.0001),
                    metric=metric_name,
                    mode="min" if "loss" in metric_name else "max",
                    restore_best=True,
                )
            stopper = EarlyStopping(es_config, key) if self.enable_early_stopping else None

            last_metrics: Dict[str, float] = {}

            # При warm-start добавляем в metrics_history запись с метриками
            if resume_start_epoch > 0 and resume_from_path and os.path.exists(resume_from_path):
                try:
                    _ckpt_for_history = torch.load(
                        resume_from_path, map_location=self.device, weights_only=False
                    )
                    _ckpt_metrics = _ckpt_for_history.get('metrics', {})
                    # Добавляем только если метрики есть — иначе не засоряем историю
                    if _ckpt_metrics and _ckpt_metrics.get('val_auc', _ckpt_metrics.get('val_acc')):
                        _history_entry = {**_ckpt_metrics, 'epoch': resume_start_epoch,
                                          '_from_checkpoint': True}
                        self.metrics_history[key].append(_history_entry)
                        self.log(
                            f"  [RESUME] Добавлена запись прогона A в историю: "
                            f"epoch={resume_start_epoch}, "
                            f"val_auc={_ckpt_metrics.get('val_auc', 0):.4f}"
                        )
                except Exception:
                    pass  # Если метрики из чекпоинта недоступны — пропускаем

            prefetcher = DataPrefetcher(train_loader, self.device)

            for epoch in range(resume_start_epoch + 1, model_max_epochs + 1):
                model.train()
                total_loss = 0.0
                n_batches = 0

                for images, labels in prefetcher:
                    # images и labels уже на GPU — DataPrefetcher
                    # выполнил .to(device) во время загрузки следующего батча
                    optimizer.zero_grad()
                    # autocast: forward pass в fp16, backward в fp32 через scaler
                    with autocast(device_type="cuda", enabled=use_amp):
                        try:
                            logits = model(images)
                        except Exception as _compile_err:
                            # torch.compile (Triton) может упасть на первом
                            # forward pass на Windows — особенно для EfficientNet.
                            # Откатываемся на base_model (без компиляции).
                            if model is not base_model:
                                self.log(
                                    f"  [COMPILE] Ошибка Triton при forward pass: "
                                    f"{type(_compile_err).__name__} — "
                                    f"откат на base_model без компиляции"
                                )
                                model = base_model
                                logits = model(images)
                            else:
                                raise
                        loss = criterion(logits, labels)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

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
                        # Сохраняем веса base_model (до компиляции) —
                        # скомпилированная модель может иметь другой формат state_dict
                        return self._save_checkpoint(base_model, optimizer, ep, mtr, key,
                                                     is_best=True, scheduler=scheduler)

                    continue_training, es_msg = stopper.step(val_metrics, epoch, _save_best)
                    if not continue_training:
                        self.log(f"  [ES] {es_msg}")
                        best_ckpt_path = stopper.best_model_path
                        break
                else:
                    # Чекпоинт каждые N эпох
                    if epoch % self.checkpoint_interval == 0 or epoch == model_max_epochs:
                        ckpt = self._save_checkpoint(base_model, optimizer, epoch,
                                                     val_metrics, key, scheduler=scheduler)
                        self.log(f"  [CKPT] Сохранён: {ckpt}")

            # Восстанавливаем лучшие веса если ES включён
            if stopper and stopper.best_model_path and os.path.exists(stopper.best_model_path):
                ckpt_data = torch.load(stopper.best_model_path, map_location=self.device)
                base_model.load_state_dict(ckpt_data["model_state_dict"])
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
            _final_ckpt = self._save_checkpoint(
                base_model, optimizer, last_metrics.get("epoch", 0), last_metrics, key,
                scheduler=scheduler,
            )
            # Запоминаем путь для возможного warm-start следующего прогона
            self.last_checkpoint_paths[key] = _final_ckpt

            # Возвращаем лучшие метрики по всей истории (включая прогон A)
            _history_for_best = self.metrics_history.get(key, [])
            if _history_for_best:
                best_metrics = max(
                    _history_for_best,
                    key=lambda x: x.get("val_auc", x.get("val_acc", 0.0)),
                )
                # Добавляем _ckpt_path из финального чекпоинта, а не из best —
                # именно финальный чекпоинт нужен для следующего warm-start прогона.
                best_metrics = {**best_metrics, "_ckpt_path": _final_ckpt}
                return best_metrics

            return {**last_metrics, "_ckpt_path": _final_ckpt}

        except Exception as e:
            import traceback
            self.log(f"  [ERROR] {key}: {e}")
            self.log(traceback.format_exc())
            return None

        finally:
            # НЕ СЛОМАТЬ! ВАЖНО ЧИСТИТЬ КОРРЕКТНО!
            try:
                del model
            except NameError:
                pass
            try:
                del base_model
            except NameError:
                pass
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            gc.collect()
            self.log(f"  [MEM] Память очищена после {key}")


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


    def run_training(self, resume_paths: Optional[Dict[str, str]] = None):
        """
        Запускает обучение всех комбинаций модель × датасет
        """
        self.log("\n" + "=" * 80)
        self.log("НАЧАЛО ОБУЧЕНИЯ КЛАССИФИКАТОРОВ")
        self.log("=" * 80)

        _resume_paths = resume_paths or {}

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

                metrics = self._train_one(
                    model_cfg, dataset_name, key,
                    resume_from_path=_resume_paths.get(key),
                )
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