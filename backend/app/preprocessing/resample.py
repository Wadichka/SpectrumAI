"""Интерполяция спектра на единую сетку волновых чисел.

По умолчанию — 400–4000 см⁻¹ с шагом 1 см⁻¹ (CLAUDE.md §7). Метод — кубический
сплайн ``scipy.interpolate.interp1d``. Если исходный диапазон уже целевого,
краевые значения заполняются ``out_of_range_fill`` (по умолчанию 0.0) и
выводится предупреждение через ``structlog``.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import structlog
from scipy.interpolate import interp1d

from app.preprocessing.config import PreprocessConfig

_logger = structlog.get_logger(__name__)


def build_target_grid(config: PreprocessConfig) -> npt.NDArray[np.float64]:
    """Возвращает фиксированную сетку волновых чисел согласно конфигу."""
    return np.linspace(
        config.target_min,
        config.target_max,
        num=config.target_length,
        dtype=np.float64,
    )


def resample_to_grid(
    wavenumbers: npt.NDArray[np.float64],
    intensities: npt.NDArray[np.float64],
    *,
    config: PreprocessConfig,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Интерполирует спектр на целевую сетку из конфига.

    Args:
        wavenumbers: исходная ось волновых чисел (см⁻¹), 1D.
        intensities: исходные интенсивности, 1D, той же длины.
        config: параметры сетки и метода интерполяции.

    Returns:
        Кортеж ``(target_grid, interpolated_intensities)``, оба длины
        ``config.target_length``.
    """
    # Часть JCAMP-DX источников выдают убывающие волновые числа — поправим.
    if wavenumbers.size > 1 and wavenumbers[0] > wavenumbers[-1]:
        order = np.argsort(wavenumbers)
        wavenumbers = wavenumbers[order]
        intensities = intensities[order]

    target_grid = build_target_grid(config)

    src_min = float(wavenumbers[0])
    src_max = float(wavenumbers[-1])
    if src_min > config.target_min or src_max < config.target_max:
        _logger.warning(
            "spectrum_range_does_not_cover_target",
            source_min=src_min,
            source_max=src_max,
            target_min=config.target_min,
            target_max=config.target_max,
            fill_value=config.out_of_range_fill,
        )

    interpolator = interp1d(
        wavenumbers,
        intensities,
        kind=config.interpolation_kind,
        bounds_error=False,
        fill_value=config.out_of_range_fill,
        assume_sorted=True,
    )
    resampled = np.asarray(interpolator(target_grid), dtype=np.float64)
    return target_grid, resampled
