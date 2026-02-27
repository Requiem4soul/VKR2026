"""
Очистка кэша Python (__pycache__ и .pyc файлы).
Запускать перед стартом Streamlit если менялись .py файлы в проекте.

Использование:
    python clear_cache.py
"""

import shutil
from pathlib import Path


def clear_pycache(root: Path = Path(".")) -> None:
    removed_dirs = 0
    removed_files = 0

    for pycache_dir in root.rglob("__pycache__"):
        if pycache_dir.is_dir():
            shutil.rmtree(pycache_dir)
            print(f"  Удалено: {pycache_dir}")
            removed_dirs += 1

    for pyc_file in root.rglob("*.pyc"):
        if pyc_file.is_file():
            pyc_file.unlink()
            print(f"  Удалено: {pyc_file}")
            removed_files += 1

    print(f"\nГотово: удалено папок __pycache__: {removed_dirs}, файлов .pyc: {removed_files}")


if __name__ == "__main__":
    print("Очистка кэша Python...\n")
    clear_pycache(Path("."))
