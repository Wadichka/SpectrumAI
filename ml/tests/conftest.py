"""Общая настройка тестов ml/.

Добавляет корень ``ml/`` в ``sys.path``, чтобы импорты ``from pipelines ...``
работали без установки пакета. Подавляет шум NumPy 2.x от rdkit-pypi,
скомпилированного под NumPy 1.x — функциональное поведение не страдает.

Импортирует torch до прочих модулей: на Windows совместная загрузка PyTorch и
rdkit-pypi приводит к ошибке c10.dll, если rdkit грузится первым (его
runtime-DLL перекрывают torch-овские). Загрузка torch первым исправляет.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import torch  # noqa: F401 — должен быть импортирован первым на Windows.

_ML_ROOT = Path(__file__).resolve().parents[1]
if str(_ML_ROOT) not in sys.path:
    sys.path.insert(0, str(_ML_ROOT))

warnings.filterwarnings("ignore", message=".*NumPy.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*module compiled against.*")
