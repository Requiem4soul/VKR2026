import os
import json
import gc
import torch
import time
from datetime import datetime
from ultralytics import YOLO
from Data.Datasets.dataset_work import get_dataset_path


class ModelTrainer:
    def __init__(self, dataset_names, max_epochs=40, checkpoint_interval=10):
        self.dataset_names = dataset_names
        self.max_epochs = max_epochs
        self.checkpoint_interval = checkpoint_interval
        self.models = {}
        self.metrics_history = {}
        self.current_epochs = {}

        # Создание папки для результатов
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        datasets_str = "_".join(dataset_names)
        self.results_dir = f"results_{datasets_str}_{timestamp}"
        os.makedirs(self.results_dir, exist_ok=True)

        self.results_file = os.path.join(self.results_dir, f"training_results_{timestamp}.txt")

        # Инициализация для каждого датасета
        for dataset_name in dataset_names:
            self.models[dataset_name] = None
            self.metrics_history[dataset_name] = []
            self.current_epochs[dataset_name] = 0

        # Создание папки для чекпоинтов внутри папки результатов
        self.checkpoint_dir = os.path.join(self.results_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Запись заголовка в файл результатов
        self.log_message("=== НАЧАЛО ОБУЧЕНИЯ ===")
        self.log_message(f"Датасеты: {', '.join(dataset_names)}")
        self.log_message(f"Максимальное количество эпох: {max_epochs}")
        self.log_message(f"Интервал проверки: {checkpoint_interval} эпох")
        self.log_message("=" * 50)

    def log_message(self, message):
        """Запись сообщения в консоль и файл"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        full_message = f"[{timestamp}] {message}"
        print(full_message)

        with open(self.results_file, 'a', encoding='utf-8') as f:
            f.write(full_message + '\n')

    def get_dataset_yaml_path(self, dataset_name):
        """Получение пути к data.yaml файлу датасета"""
        dataset_path = get_dataset_path(dataset_name)
        return os.path.join(dataset_path, "data.yaml")

    def initialize_model(self, dataset_name):
        """Инициализация модели для датасета"""
        if self.models[dataset_name] is None:
            # Принудительная очистка перед инициализацией
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

            self.models[dataset_name] = YOLO('yolov8n.pt')  # Можно изменить на другую модель
            self.log_message(f"Инициализирована модель для датасета: {dataset_name}")

    def save_checkpoint(self, dataset_name, epoch):
        """Сохранение чекпоинта модели"""
        checkpoint_path = os.path.join(self.checkpoint_dir, f"{dataset_name}_epoch_{epoch}.pt")
        self.models[dataset_name].save(checkpoint_path)
        self.log_message(f"Сохранен чекпоинт для {dataset_name} на эпохе {epoch}: {checkpoint_path}")
        return checkpoint_path

    def load_checkpoint(self, dataset_name, epoch):
        """Загрузка чекпоинта модели"""
        checkpoint_path = os.path.join(self.checkpoint_dir, f"{dataset_name}_epoch_{epoch}.pt")
        if os.path.exists(checkpoint_path):
            # Если модель уже загружена, очищаем её перед загрузкой новой
            if self.models[dataset_name] is not None:
                del self.models[dataset_name]
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

            self.models[dataset_name] = YOLO(checkpoint_path)
            self.log_message(f"Загружен чекпоинт для {dataset_name} с эпохи {epoch}")
            return True
        return False

    def train_model_segment(self, dataset_name, start_epoch, end_epoch):
        """Обучение модели на сегменте эпох"""
        self.log_message(f"\n--- Обучение {dataset_name}: эпохи {start_epoch + 1}-{end_epoch} ---")

        dataset_yaml_path = self.get_dataset_yaml_path(dataset_name)

        # Если это не первая эпоха, загружаем чекпоинт
        if start_epoch > 0:
            if not self.load_checkpoint(dataset_name, start_epoch):
                self.log_message(f"ОШИБКА: Не найден чекпоинт для {dataset_name} на эпохе {start_epoch}")
                return None
        else:
            self.initialize_model(dataset_name)

        # Количество эпох для обучения
        epochs_to_train = end_epoch - start_epoch

        try:
            # Очистка памяти перед обучением
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

            # Обучение модели
            results = self.models[dataset_name].train(
                data=dataset_yaml_path,
                epochs=epochs_to_train,
                imgsz=256,
                batch=8,  # Можно уменьшить если мало памяти
                device="cuda",  # Использовать GPU, если доступно
                save=False,  # Не сохранять автоматически
                project=f"runs/{dataset_name}",
                name=f"epoch_{end_epoch}",
                exist_ok=True,
                workers=1,
                cache=False
            )

            # Извлечение метрик
            metrics = self.extract_metrics(results)
            metrics['epoch'] = end_epoch
            self.metrics_history[dataset_name].append(metrics)
            self.current_epochs[dataset_name] = end_epoch

            # Сохранение чекпоинта
            if end_epoch < self.max_epochs:
                self.save_checkpoint(dataset_name, end_epoch)

                # Очищаем модель из памяти
                del self.models[dataset_name]
                self.models[dataset_name] = None

                # Принудительная очистка памяти
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

            self.log_message(f"Завершено обучение {dataset_name} до эпохи {end_epoch}")
            self.log_metrics(dataset_name, metrics)

            # Принудительная очистка после обучения
            del results
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

            return metrics

        except Exception as e:
            self.log_message(f"ОШИБКА при обучении {dataset_name}: {str(e)}")
            return None

    def extract_metrics(self, results):
        """Извлечение метрик из результатов обучения"""
        try:
            # Получаем последние метрики валидации
            metrics = {
                'precision': float(results.results_dict.get('metrics/precision(B)', 0)),
                'recall': float(results.results_dict.get('metrics/recall(B)', 0)),
                'mAP50': float(results.results_dict.get('metrics/mAP50(B)', 0)),
                'mAP50-95': float(results.results_dict.get('metrics/mAP50-95(B)', 0)),
                'train_loss': float(results.results_dict.get('train/box_loss', 0)),
                'val_loss': float(results.results_dict.get('val/box_loss', 0))
            }
        except:
            # Если не удалось извлечь метрики, используем значения по умолчанию
            metrics = {
                'precision': 0.0,
                'recall': 0.0,
                'mAP50': 0.0,
                'mAP50-95': 0.0,
                'train_loss': 0.0,
                'val_loss': 0.0
            }

        return metrics

    def log_metrics(self, dataset_name, metrics):
        """Вывод метрик в лог"""
        self.log_message(f"Метрики для {dataset_name} (эпоха {metrics['epoch']}):")
        self.log_message(f"  Precision: {metrics['precision']:.4f}")
        self.log_message(f"  Recall: {metrics['recall']:.4f}")
        self.log_message(f"  mAP@0.5: {metrics['mAP50']:.4f}")
        self.log_message(f"  mAP@0.5:0.95: {metrics['mAP50-95']:.4f}")
        self.log_message(f"  Train Loss: {metrics['train_loss']:.4f}")
        self.log_message(f"  Val Loss: {metrics['val_loss']:.4f}")

    def compare_models(self, epoch):
        """Сравнение моделей на текущей эпохе"""
        self.log_message(f"\n=== СРАВНЕНИЕ МОДЕЛЕЙ НА ЭПОХЕ {epoch} ===")

        current_metrics = {}
        for dataset_name in self.dataset_names:
            if self.metrics_history[dataset_name]:
                current_metrics[dataset_name] = self.metrics_history[dataset_name][-1]

        if len(current_metrics) < 2:
            self.log_message("Недостаточно данных для сравнения")
            return

        # Сравнение по основным метрикам
        metrics_to_compare = ['precision', 'recall', 'mAP50', 'mAP50-95']

        for metric in metrics_to_compare:
            self.log_message(f"\n{metric.upper()}:")
            sorted_datasets = sorted(current_metrics.items(),
                                     key=lambda x: x[1][metric], reverse=True)

            for i, (dataset_name, metrics) in enumerate(sorted_datasets):
                rank = i + 1
                value = metrics[metric]
                self.log_message(f"  {rank}. {dataset_name}: {value:.4f}")

        # Общий рейтинг (среднее по основным метрикам)
        self.log_message(f"\nОБЩИЙ РЕЙТИНГ (среднее precision, recall, mAP@0.5):")
        overall_scores = {}
        for dataset_name, metrics in current_metrics.items():
            score = (metrics['precision'] + metrics['recall'] + metrics['mAP50']) / 3
            overall_scores[dataset_name] = score

        sorted_overall = sorted(overall_scores.items(), key=lambda x: x[1], reverse=True)
        for i, (dataset_name, score) in enumerate(sorted_overall):
            rank = i + 1
            self.log_message(f"  {rank}. {dataset_name}: {score:.4f}")

        self.log_message("=" * 50)

    def save_final_results(self):
        """Сохранение финальных результатов в JSON"""
        results_data = {
            'dataset_names': self.dataset_names,
            'max_epochs': self.max_epochs,
            'metrics_history': self.metrics_history,
            'final_comparison': self.get_final_comparison()
        }

        json_file = f"final_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)

        self.log_message(f"Финальные результаты сохранены в {json_file}")

    def get_final_comparison(self):
        """Получение финального сравнения моделей"""
        final_metrics = {}
        for dataset_name in self.dataset_names:
            if self.metrics_history[dataset_name]:
                final_metrics[dataset_name] = self.metrics_history[dataset_name][-1]

        return final_metrics

    def run_training(self):
        """Основной цикл обучения"""
        self.log_message("Начинаем поочередное обучение моделей...")

        for epoch in range(self.checkpoint_interval, self.max_epochs + 1, self.checkpoint_interval):
            self.log_message(f"\n{'=' * 60}")
            self.log_message(f"ЭТАП: Обучение до эпохи {epoch}")
            self.log_message(f"{'=' * 60}")

            # Обучаем каждую модель до текущей эпохи
            for dataset_name in self.dataset_names:
                start_epoch = self.current_epochs[dataset_name]
                metrics = self.train_model_segment(dataset_name, start_epoch, epoch)

                if metrics is None:
                    self.log_message(f"Пропускаем {dataset_name} из-за ошибки обучения")
                    continue

                # Дополнительная очистка между датасетами
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

                # Небольшая пауза для завершения фоновых процессов
                time.sleep(2)

            # Сравниваем результаты
            self.compare_models(epoch)

        # Сохраняем финальные результаты
        self.save_final_results()
        self.log_message("\n=== ОБУЧЕНИЕ ЗАВЕРШЕНО ===")


# Использование
if __name__ == "__main__":
    # Названия ваших датасетов
    dataset_names = ["SAR_low", "SAR_LP_med3_CLACHE1_16", "SAR_LP_med3_CLACHE2_8",
                     "SAR_LP_med3_CLACHE2_16", "SAR_LP_med5_CLACHE1_16",
                     "SAR_LP_med5_CLACHE2_8", "SAR_LP_med5_CLACHE2_16"]

    # Создание и запуск тренера
    trainer = ModelTrainer(dataset_names, max_epochs=20, checkpoint_interval=5)
    trainer.run_training()