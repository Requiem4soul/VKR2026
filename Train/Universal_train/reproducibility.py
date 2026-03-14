"""
Train/Universal_train/reproducibility.py
Утилиты воспроизводимости для модуля детекции.

Добавить в UniversalModelTrainer.__init__:
    from Train.Universal_train.reproducibility import set_global_seed
    self.seed = seed
    set_global_seed(seed)

И в параметры __init__ добавить: seed: int = 42
"""

import os
import random
import numpy as np
import torch


def set_global_seed(seed: int) -> None:
    """
    Фиксирует все источники случайности для воспроизводимых результатов.

    Необходимо вызывать до создания любых моделей и DataLoader'ов.
    При одинаковом seed на одинаковом оборудовании результаты будут идентичны.

    Ограничение: torch.backends.cudnn.deterministic=True может незначительно
    замедлить обучение (Dodge & Karam, 2017).

    Args:
        seed: целое число, значение для фиксации генераторов случайных чисел.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
