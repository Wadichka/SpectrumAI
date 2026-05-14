"""Нормирование интенсивностей спектра.

Поддерживаются два метода: Standard Normal Variate (SNV) и min-max.
Глава 4 §4.4.2 упоминает также векторную L2-нормировку — в этом этапе
не реализуем (нет требования в DEVELOPMENT_PLAN.md этап 3).
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from app.preprocessing.config import NormalizeMethod


def snv_normalize(intensities: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Стандартная нормировка SNV: (y - mean) / std.

    При нулевом стандартном отклонении (константный спектр) возвращает
    массив нулей, чтобы избежать NaN.
    """
    y = np.asarray(intensities, dtype=np.float64)
    std = float(y.std(ddof=0))
    if std == 0.0:
        return np.zeros_like(y)
    return (y - float(y.mean())) / std


def min_max_normalize(intensities: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Min-max нормирование к диапазону [0, 1].

    При вырожденном случае ``max == min`` возвращает массив нулей.
    """
    y = np.asarray(intensities, dtype=np.float64)
    y_min = float(y.min())
    y_max = float(y.max())
    spread = y_max - y_min
    if spread == 0.0:
        return np.zeros_like(y)
    return (y - y_min) / spread


def normalize(
    intensities: npt.NDArray[np.float64],
    *,
    method: NormalizeMethod,
) -> npt.NDArray[np.float64]:
    """Диспетчер по выбранному методу нормирования."""
    if method == "snv":
        return snv_normalize(intensities)
    return min_max_normalize(intensities)
