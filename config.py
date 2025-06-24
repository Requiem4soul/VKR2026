import os
from pathlib import Path
from dotenv import load_dotenv

# Тут просто подгружаем env файл
load_dotenv()

# Получаем указанный путь до папки с датасетами
DATASETS_GLOBAL_PATH = Path(os.getenv("DATASETS_GLOBAL_PATH"))



