"""Предобработка спектров: ресемплинг, baseline, сглаживание, нормирование.

Публичный API: :class:`PreprocessConfig`, :class:`ProcessedSpectrum`,
:func:`preprocess`, а также «низкоуровневые» функции этапов для прямого
использования в экспериментах и тестах.
"""

from __future__ import annotations

from app.preprocessing.baseline import asls_baseline, subtract_baseline
from app.preprocessing.config import (
    InterpolationKind,
    NormalizeMethod,
    PreprocessConfig,
    ProcessedSpectrum,
)
from app.preprocessing.normalize import min_max_normalize, normalize, snv_normalize
from app.preprocessing.pipeline import preprocess
from app.preprocessing.resample import build_target_grid, resample_to_grid
from app.preprocessing.smoothing import savgol_smooth

__all__ = [
    "InterpolationKind",
    "NormalizeMethod",
    "PreprocessConfig",
    "ProcessedSpectrum",
    "asls_baseline",
    "build_target_grid",
    "min_max_normalize",
    "normalize",
    "preprocess",
    "resample_to_grid",
    "savgol_smooth",
    "snv_normalize",
    "subtract_baseline",
]
