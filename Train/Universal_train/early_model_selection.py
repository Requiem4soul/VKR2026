"""
Early Model Selection - Ранний отбор моделей на основе промежуточных метрик

НАУЧНОЕ ОБОСНОВАНИЕ:
1. Jamieson & Talwalkar (2016) - Successive Halving Algorithm
2. Domhan et al. (2015) - Extrapolation of Learning Curves
3. Li et al. (2017) - Hyperband Algorithm

КРИТЕРИИ КОРРЕКТНОСТИ:
- Используем validation set (не train!)
- Отбираем после 30-40% обучения (доказано достаточно для корреляции >0.7)
- Всегда дообучиваем финалистов до конца
- Документируем все решения

Автор: Дипломная работа VKR2026
Дата: 2026-02-16
"""

import json
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime


@dataclass
class ModelCheckpointMetrics:
    """Метрики модели на чекпоинте"""
    model_key: str  # Например: "yolo_s_dataset1"
    epoch: int
    val_loss: float
    mAP50: float
    mAP50_95: float
    precision: float
    recall: float
    f1: float
    
    @property
    def composite_score(self) -> float:
        """
        Комплексная метрика для сравнения моделей
        
        Взвешенная сумма:
        - mAP50-95: 40% (главная метрика для COCO)
        - mAP50: 30% (проще детектировать)
        - F1: 20% (баланс precision/recall)
        - 1/val_loss: 10% (стабильность обучения)
        
        Источник: Lin et al. (2014) "Microsoft COCO: Common Objects in Context"
        """
        loss_component = 1.0 / (1.0 + self.val_loss)  # Нормализуем loss
        
        score = (
            0.40 * self.mAP50_95 +
            0.30 * self.mAP50 +
            0.20 * self.f1 +
            0.10 * loss_component
        )
        return score


class EarlyModelSelector:
    """
    Ранний отбор моделей на основе промежуточных метрик
    
    АЛГОРИТМ:
    1. Обучаем N моделей параллельно до checkpoint_epoch
    2. Вычисляем composite_score для каждой модели
    3. Отбираем Top-K моделей (по умолчанию K=50%)
    4. Дообучиваем только Top-K до конца
    
    НАУЧНАЯ КОРРЕКТНОСТЬ:
    - Domhan et al. (2015): корреляция метрик после 30% эпох с финальными > 0.7
    - Используем validation set
    - Документируем все отброшенные модели
    """
    
    def __init__(
        self,
        checkpoint_ratio: float = 0.3,
        top_k_fraction: float = 0.5,
        min_models_to_keep: int = 2,
        log_dir: Path = Path("early_selection_logs")
    ):
        """
        Args:
            checkpoint_ratio: На какой доле обучения делать отбор (0.3 = 30%)
            top_k_fraction: Какую долю моделей оставлять (0.5 = 50%)
            min_models_to_keep: Минимальное количество моделей для продолжения
            log_dir: Папка для логов
        """
        self.checkpoint_ratio = checkpoint_ratio
        self.top_k_fraction = top_k_fraction
        self.min_models_to_keep = min_models_to_keep
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # История решений
        self.selection_history = []
    
    def calculate_checkpoint_epoch(self, max_epochs: int) -> int:
        """Вычисляет эпоху для раннего отбора"""
        checkpoint_epoch = int(max_epochs * self.checkpoint_ratio)
        
        # Минимум 5 эпох для стабильности
        checkpoint_epoch = max(5, checkpoint_epoch)
        
        return checkpoint_epoch
    
    def should_continue_training(
        self,
        model_key: str,
        current_metrics: Dict[str, List[Dict]],
        current_epoch: int,
        max_epochs: int
    ) -> Tuple[bool, Optional[str]]:
        """
        Определяет, продолжать ли обучение модели
        
        Returns:
            (should_continue, reason)
            - should_continue: True/False
            - reason: Объяснение решения (для логов)
        """
        checkpoint_epoch = self.calculate_checkpoint_epoch(max_epochs)
        
        # До чекпоинта - обучаем всех
        if current_epoch < checkpoint_epoch:
            return True, f"До чекпоинта ({checkpoint_epoch} эпох)"
        
        # Ровно на чекпоинте - делаем отбор
        if current_epoch == checkpoint_epoch:
            return self._perform_selection(
                current_metrics,
                checkpoint_epoch,
                max_epochs
            )
        
        # После чекпоинта - проверяем, была ли модель отобрана
        last_decision = self._get_last_decision_for_model(model_key)
        if last_decision is None:
            # Нет решения - значит обучаем (на всякий случай)
            return True, "Нет предыдущего решения"
        
        return last_decision['should_continue'], last_decision['reason']
    
    def _perform_selection(
        self,
        current_metrics: Dict[str, List[Dict]],
        checkpoint_epoch: int,
        max_epochs: int
    ) -> Tuple[bool, str]:
        """
        Выполняет отбор моделей на чекпоинте
        
        АЛГОРИТМ:
        1. Извлекаем метрики всех моделей на текущей эпохе
        2. Вычисляем composite_score
        3. Ранжируем модели
        4. Отбираем Top-K
        5. Логируем решение
        """
        print(f"\n{'='*80}")
        print(f"РАННИЙ ОТБОР МОДЕЛЕЙ НА ЭПОХЕ {checkpoint_epoch}")
        print(f"{'='*80}")
        
        # Собираем метрики всех моделей
        model_scores = []
        
        for model_key, metrics_list in current_metrics.items():
            if not metrics_list:
                continue
            
            # Берём метрики последней эпохи
            latest_metrics = metrics_list[-1]
            
            # Проверяем наличие всех необходимых метрик
            required_keys = ['val_loss', 'mAP50', 'mAP50-95', 'precision', 'recall', 'f1']
            if not all(k in latest_metrics for k in required_keys):
                print(f"[WARNING] У модели {model_key} отсутствуют некоторые метрики, пропускаем")
                continue
            
            checkpoint_metrics = ModelCheckpointMetrics(
                model_key=model_key,
                epoch=latest_metrics.get('epoch', checkpoint_epoch),
                val_loss=latest_metrics['val_loss'],
                mAP50=latest_metrics['mAP50'],
                mAP50_95=latest_metrics['mAP50-95'],
                precision=latest_metrics['precision'],
                recall=latest_metrics['recall'],
                f1=latest_metrics['f1']
            )
            
            model_scores.append({
                'model_key': model_key,
                'metrics': checkpoint_metrics,
                'composite_score': checkpoint_metrics.composite_score
            })
        
        if len(model_scores) == 0:
            print("[WARNING] Нет моделей с полными метриками для отбора")
            return True, "Нет данных для отбора"
        
        # Сортируем по composite_score (от большего к меньшему)
        model_scores.sort(key=lambda x: x['composite_score'], reverse=True)
        
        # Определяем сколько моделей оставить
        num_models = len(model_scores)
        num_to_keep = max(
            self.min_models_to_keep,
            int(num_models * self.top_k_fraction)
        )
        
        # Отбираем Top-K
        selected_models = model_scores[:num_to_keep]
        rejected_models = model_scores[num_to_keep:]
        
        # Выводим результаты
        print(f"\n📊 Результаты ранжирования ({num_models} моделей):")
        print(f"{'Ранг':<6} {'Модель':<30} {'Score':<8} {'mAP50-95':<10} {'mAP50':<8} {'F1':<8}")
        print("-" * 80)
        
        for i, model_data in enumerate(model_scores):
            m = model_data['metrics']
            status = "✅ ПРОДОЛЖАЕМ" if i < num_to_keep else "❌ ОСТАНАВЛИВАЕМ"
            print(
                f"{i+1:<6} {m.model_key:<30} {model_data['composite_score']:.4f}   "
                f"{m.mAP50_95:.4f}     {m.mAP50:.4f}   {m.f1:.4f}   {status}"
            )
        
        # Сохраняем решение
        decision = {
            'timestamp': datetime.now().isoformat(),
            'checkpoint_epoch': checkpoint_epoch,
            'max_epochs': max_epochs,
            'total_models': num_models,
            'models_kept': num_to_keep,
            'models_rejected': len(rejected_models),
            'selected_models': [m['model_key'] for m in selected_models],
            'rejected_models': [m['model_key'] for m in rejected_models],
            'scores': {
                m['model_key']: {
                    'composite_score': m['composite_score'],
                    'mAP50-95': m['metrics'].mAP50_95,
                    'mAP50': m['metrics'].mAP50,
                    'f1': m['metrics'].f1
                }
                for m in model_scores
            }
        }
        
        self.selection_history.append(decision)
        
        # Сохраняем в JSON
        log_file = self.log_dir / f"selection_epoch_{checkpoint_epoch}.json"
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(decision, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ Решение сохранено: {log_file}")
        print(f"📈 Продолжаем обучение {num_to_keep}/{num_models} моделей")
        
        return True, f"Отобрано на эпохе {checkpoint_epoch}"
    
    def _get_last_decision_for_model(self, model_key: str) -> Optional[Dict]:
        """Получает последнее решение для модели"""
        if not self.selection_history:
            return None
        
        last_decision = self.selection_history[-1]
        
        if model_key in last_decision['selected_models']:
            return {
                'should_continue': True,
                'reason': f"Отобрана на эпохе {last_decision['checkpoint_epoch']}"
            }
        elif model_key in last_decision['rejected_models']:
            return {
                'should_continue': False,
                'reason': f"Отклонена на эпохе {last_decision['checkpoint_epoch']}"
            }
        else:
            return None
    
    def get_selection_report(self) -> str:
        """Генерирует отчёт по всем отборам"""
        if not self.selection_history:
            return "Отборы не проводились"
        
        report = []
        report.append("=" * 80)
        report.append("ОТЧЁТ ПО РАННЕМУ ОТБОРУ МОДЕЛЕЙ")
        report.append("=" * 80)
        
        for i, decision in enumerate(self.selection_history, 1):
            report.append(f"\nОтбор #{i} (эпоха {decision['checkpoint_epoch']}):")
            report.append(f"  Всего моделей: {decision['total_models']}")
            report.append(f"  Оставлено: {decision['models_kept']}")
            report.append(f"  Отброшено: {decision['models_rejected']}")
            report.append(f"\n  Продолжают обучение:")
            for model in decision['selected_models']:
                score = decision['scores'][model]['composite_score']
                report.append(f"    - {model} (score: {score:.4f})")
            
            if decision['rejected_models']:
                report.append(f"\n  Остановлены:")
                for model in decision['rejected_models']:
                    score = decision['scores'][model]['composite_score']
                    report.append(f"    - {model} (score: {score:.4f})")
        
        return "\n".join(report)


# ===================== ПРИМЕР ИСПОЛЬЗОВАНИЯ =====================

if __name__ == "__main__":
    # Создаём селектор
    selector = EarlyModelSelector(
        checkpoint_ratio=0.3,  # Отбор на 30% обучения
        top_k_fraction=0.5,    # Оставляем 50% лучших
        min_models_to_keep=2   # Минимум 2 модели
    )
    
    # Симуляция метрик (в реальности берутся из UniversalModelTrainer)
    mock_metrics = {
        'yolo_s_dataset1': [
            {'epoch': 10, 'val_loss': 0.5, 'mAP50': 0.65, 'mAP50-95': 0.45, 
             'precision': 0.7, 'recall': 0.6, 'f1': 0.65}
        ],
        'faster_rcnn_dataset1': [
            {'epoch': 10, 'val_loss': 0.4, 'mAP50': 0.70, 'mAP50-95': 0.50, 
             'precision': 0.75, 'recall': 0.65, 'f1': 0.70}
        ],
        'retinanet_dataset1': [
            {'epoch': 10, 'val_loss': 0.6, 'mAP50': 0.60, 'mAP50-95': 0.40, 
             'precision': 0.65, 'recall': 0.55, 'f1': 0.60}
        ]
    }
    
    # Проверяем отбор на эпохе 10 (из 30)
    should_continue, reason = selector.should_continue_training(
        model_key='yolo_s_dataset1',
        current_metrics=mock_metrics,
        current_epoch=10,
        max_epochs=30
    )
    
    print(f"\nРешение для yolo_s_dataset1: {should_continue}")
    print(f"Причина: {reason}")
    
    # Генерируем отчёт
    print("\n" + selector.get_selection_report())
