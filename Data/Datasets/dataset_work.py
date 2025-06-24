import os
from pathlib import Path
from config import DATASETS_GLOBAL_PATH

# Функция чтобы получить конкретный датасет (название папки) из корневой папки датасетов
def get_dataset_path(name: str) -> Path:
    return DATASETS_GLOBAL_PATH / name


# Функция чтобы посмотреть какие вообще есть датасеты внутри главной папки. Выводит названия папок внутри
def list_available_datasets(verbose: bool = True) -> list[str]:
    """
    Возвращает список всех папок в корневом каталоге датасетов.
    Если verbose=True, выводит список в консоль.
    """
    if not DATASETS_GLOBAL_PATH.exists():
        raise FileNotFoundError(f"Путь {DATASETS_GLOBAL_PATH} не существует")

    datasets = [
        item.name
        for item in DATASETS_GLOBAL_PATH.iterdir()
        if item.is_dir()
    ]

    if verbose:
        if datasets:
            print("\nДоступные датасеты:")
            for name in datasets:
                print(f"  - {name}")
        else:
            print("В корневой папке нет датасетов.")

    return datasets


# Для работы с конкретным датасетом
def choose_dataset_interactively(verbose: bool = True) -> Path:
    """
    Показывает список датасетов, предлагает выбрать один.
    Если всё сделал пользователь верно, возвращается в переменную путь до данного датасета.
    Его потом можно будет уже использовать
    """
    datasets = list_available_datasets(verbose=False)
    if not datasets:
        raise RuntimeError("Нет доступных датасетов")

    print("\nДоступные датасеты:")
    for i, name in enumerate(datasets):
        print(f"[{i}] {name}")

    while True:
        try:
            idx = int(input("\nВведите номер датасета: "))
            if 0 <= idx < len(datasets):
                if verbose:
                    print(f"\nВами был выбран датасет {datasets[idx]}")
                return get_dataset_path(datasets[idx])
            else:
                print("Неверный номер, попробуйте снова.")
        except ValueError:
            print("Введите число.")


# # Пример отображение
# datasets = list_available_datasets()

# # Пример выбора датасета нужного (пока одного)
# SELECTED_DATASET = choose_dataset_interactively()