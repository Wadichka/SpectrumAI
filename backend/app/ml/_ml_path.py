"""Добавляет корень ``ml/`` в ``sys.path`` для импортов ``pipelines.*``.

Backend не зависит от ``ml/`` как от пакета (``ml/pyproject.toml`` намеренно
не содержит ``[project]``/``[build-system]``). Этот модуль выполняет
side-effect — кладёт абсолютный путь к ``ml/`` в начало ``sys.path`` при
импорте — после чего модули вроде ``pipelines.models.cnn1d`` доступны для
обычных ``from ... import ...``.

Тот же паттерн используется в ``ml/scripts/*`` и ``ml/tests/test_gradcam_with_cnn.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ML_ROOT = Path(__file__).resolve().parents[3] / "ml"
if str(_ML_ROOT) not in sys.path:
    sys.path.insert(0, str(_ML_ROOT))

__all__: list[str] = []
