"""Добавляет корень ``ml/`` в ``sys.path`` для импортов ``pipelines.*``.

Backend не зависит от ``ml/`` как от пакета (``ml/pyproject.toml`` намеренно
не содержит ``[project]``/``[build-system]``). Этот модуль выполняет
side-effect — кладёт абсолютный путь к ``ml/`` в начало ``sys.path`` при
импорте — после чего модули вроде ``pipelines.models.cnn1d`` доступны для
обычных ``from ... import ...``.

Тот же паттерн используется в ``ml/scripts/*`` и ``ml/tests/test_gradcam_with_cnn.py``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# В prod-образе ``app`` устанавливается в site-packages, поэтому вычисление
# через ``parents[3]`` не работает. Переменная ``SPECTRUMAI_ML_ROOT`` явно
# указывает каталог, содержащий пакет ``pipelines``. В dev (venv из корня
# репозитория) переменная не задана — используется fallback на относительный
# путь backend/app/ml → repo_root/ml.
_env_root = os.environ.get("SPECTRUMAI_ML_ROOT")
if _env_root:
    _ML_ROOT = Path(_env_root).resolve()
else:
    _ML_ROOT = Path(__file__).resolve().parents[3] / "ml"

if _ML_ROOT.exists() and str(_ML_ROOT) not in sys.path:
    sys.path.insert(0, str(_ML_ROOT))

__all__: list[str] = []
