"""Сглаживание спектра алгоритмом Савицкого–Голея."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy.signal import savgol_filter


def savgol_smooth(
    intensities: npt.NDArray[np.float64],
    *,
    window: int,
    polyorder: int,
) -> npt.NDArray[np.float64]:
    """Применяет Savitzky–Golay-фильтр.

    Args:
        intensities: 1D массив интенсивностей.
        window: длина окна (нечётное, > polyorder).
        polyorder: порядок локальной полиномиальной аппроксимации.

    Returns:
        Сглаженный массив той же длины.
    """
    smoothed = savgol_filter(intensities, window_length=window, polyorder=polyorder)
    return np.asarray(smoothed, dtype=np.float64)
