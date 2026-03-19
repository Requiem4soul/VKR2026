"""
convert_mias.py
===============
Конвертация датасета Mini-MIAS → ImageFolder для классификации.

Исходная структура:
    MIASDBv1.21/
    ├── mdb001.pgm
    ├── mdb002.pgm
    ├── ...
    └── Info.txt   (или аналогичный текстовый файл с разметкой)

Формат разметки Info.txt:
    mdb001 ll  F NORM
    mdb003 ll  G CIRC B 1815 1116 790
    └─ имя   └─ ...   └─ 4й столбец: NORM / B / M

Логика классов:
    - NORM → normal
    - B или M в 4-м поле → abnormal

Целевая структура (ImageFolder):
    MIAS_clf/
    ├── train/
    │   ├── normal/
    │   └── abnormal/
    ├── valid/
    │   ├── normal/
    │   └── abnormal/
    └── test/
        ├── normal/
        └── abnormal/

Конвертация: .pgm → .png через OpenCV
Разбивка: 70% train / 15% valid / 15% test (seed=42)
"""

import shutil
import random
import re
from pathlib import Path

# ─── НАСТРОЙКИ ────────────────────────────────────────────────────────────────
SOURCE_PATH = Path(r"N:\ORIG_Datasets\MIASDBv1.21")  # ← путь к исходному датасету
OUTPUT_PATH = Path(r"N:\ORIG_Datasets\CONVERTED_DATASETS\MIAS_clf")     # ← куда сохранить результат

# Имя текстового файла с разметкой (лежит внутри SOURCE_PATH)
INFO_FILENAME = "Info.txt"

TRAIN_RATIO = 0.70
VALID_RATIO = 0.15
TEST_RATIO  = 0.15
RANDOM_SEED = 42
# ──────────────────────────────────────────────────────────────────────────────

try:
    import cv2
except ImportError:
    raise ImportError("Установите OpenCV: pip install opencv-python")


def parse_info_file(info_path: Path) -> dict[str, str]:
    """
    Парсит Info.txt и возвращает словарь {имя_файла: класс}.

    Формат строки (пробелы могут быть множественными):
        mdb001 ll F NORM
        mdb003 ll G CIRC B 1815 1116 790

    Логика:
        - Ищем токен NORM → класс 'normal'
        - Ищем токен B или M (отдельным словом, не как часть NORM/CIRC/etc) → 'abnormal'

    Для строк с несколькими аномалиями (mdb132, mdb144 и т.д.) берём первую запись.
    """
    labels = {}
    current_stem = None

    with open(info_path, "r", encoding="utf-8", errors="ignore") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            tokens = line.split()
            if not tokens:
                continue

            # Строка начинается с имени файла (mdbXXX...)
            if re.match(r"^mdb\d+", tokens[0], re.IGNORECASE):
                current_stem = tokens[0].lower()

                # Ищем NORM
                if "NORM" in [t.upper() for t in tokens]:
                    labels[current_stem] = "normal"
                else:
                    # Ищем B или M как отдельный токен (4-й индекс или после типа аномалии)
                    # Структура: mdbXXX  side  tissue_type  anomaly_class  ...
                    # anomaly_class стоит на позиции после типа (CIRC/MISC/SPIC/ARCH/ASYM/CALC)
                    anomaly_types = {"CIRC", "MISC", "SPIC", "ARCH", "ASYM", "CALC"}
                    found_class = None
                    for i, t in enumerate(tokens):
                        if t.upper() in anomaly_types:
                            # Следующий токен — B или M
                            if i + 1 < len(tokens) and tokens[i + 1].upper() in ("B", "M"):
                                found_class = "abnormal"
                                break
                    if found_class:
                        labels[current_stem] = found_class
                    else:
                        # Строка с аномалией, но нет явного B/M — считаем abnormal
                        labels[current_stem] = "abnormal"

            else:
                # Продолжение строки для изображения с несколькими аномалиями
                # (mdb132, mdb144, mdb223 и т.д.) — уже обработано через current_stem
                pass

    return labels


def split_by_class(file_cls_pairs: list[tuple], seed: int):
    """
    Стратифицированная разбивка: сохраняет пропорции классов в каждом сплите.
    """
    rng = random.Random(seed)

    by_class: dict[str, list] = {}
    for stem, cls in file_cls_pairs:
        by_class.setdefault(cls, []).append(stem)

    train_list, valid_list, test_list = [], [], []

    for cls, stems in by_class.items():
        stems = stems.copy()
        rng.shuffle(stems)
        n = len(stems)
        n_train = int(n * TRAIN_RATIO)
        n_valid = int(n * VALID_RATIO)

        train_list.extend([(s, cls) for s in stems[:n_train]])
        valid_list.extend([(s, cls) for s in stems[n_train : n_train + n_valid]])
        test_list.extend( [(s, cls) for s in stems[n_train + n_valid:]])

    return train_list, valid_list, test_list


def convert_and_copy(src_pgm: Path, dst_png: Path):
    """Читает .pgm, конвертирует в .png и сохраняет."""
    img = cv2.imread(str(src_pgm), cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"  [WARNING] Не удалось прочитать: {src_pgm}")
        return False
    cv2.imwrite(str(dst_png), img)
    return True


def main():
    # Ищем файл разметки
    info_path = SOURCE_PATH / INFO_FILENAME
    if not info_path.exists():
        # Попробуем найти любой .txt файл
        txt_files = list(SOURCE_PATH.glob("*.txt"))
        if txt_files:
            info_path = txt_files[0]
            print(f"[INFO] Info.txt не найден, использую: {info_path.name}")
        else:
            raise FileNotFoundError(
                f"Файл разметки не найден в {SOURCE_PATH}. "
                "Убедитесь что Info.txt (или аналог) находится в папке датасета."
            )

    print(f"Читаю разметку из: {info_path}")
    labels = parse_info_file(info_path)
    print(f"Найдено записей в разметке: {len(labels)}")

    # Собираем .pgm файлы, для которых есть разметка
    pairs = []
    missing = []
    for stem, cls in labels.items():
        pgm_path = SOURCE_PATH / f"{stem}.pgm"
        if pgm_path.exists():
            pairs.append((stem, cls))
        else:
            missing.append(stem)

    if missing:
        print(f"[WARNING] Не найдены .pgm файлы для {len(missing)} записей: {missing[:5]}...")

    print(f"\nФайлов для конвертации: {len(pairs)}")
    normal_count   = sum(1 for _, c in pairs if c == "normal")
    abnormal_count = sum(1 for _, c in pairs if c == "abnormal")
    print(f"  normal:   {normal_count}")
    print(f"  abnormal: {abnormal_count}")

    # Разбивка (стратифицированная)
    train_list, valid_list, test_list = split_by_class(pairs, RANDOM_SEED)

    # Создаём выходные папки
    for split in ("train", "valid", "test"):
        for cls in ("normal", "abnormal"):
            (OUTPUT_PATH / split / cls).mkdir(parents=True, exist_ok=True)

    split_map = {"train": train_list, "valid": valid_list, "test": test_list}
    total_ok = 0
    total_fail = 0

    for split_name, items in split_map.items():
        ok = 0
        for stem, cls in items:
            src = SOURCE_PATH / f"{stem}.pgm"
            dst = OUTPUT_PATH / split_name / cls / f"{stem}.png"
            if convert_and_copy(src, dst):
                ok += 1
            else:
                total_fail += 1
        print(f"  {split_name}: {ok} файлов")
        total_ok += ok

    print(f"\n✓ Готово! Конвертировано: {total_ok}, ошибок: {total_fail}")
    print(f"  Результат: {OUTPUT_PATH}")

    print("\nСтруктура датасета:")
    for split in ("train", "valid", "test"):
        for cls in ("normal", "abnormal"):
            count = len(list((OUTPUT_PATH / split / cls).glob("*.png")))
            print(f"  {split}/{cls}: {count}")


if __name__ == "__main__":
    main()
