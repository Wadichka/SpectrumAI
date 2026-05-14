"""Оркестратор пайплайна предобработки.

Цепочка по главе 4 §4.4.2: ресемплинг → коррекция базовой линии →
сглаживание → нормирование. Каждый шаг фиксируется в
``metadata["pipeline_steps"]`` для возможной отладки и формирования
объяснений (см. §4.4.5).
"""

from __future__ import annotations

from typing import Any

from app.parsing import RawSpectrum
from app.preprocessing.baseline import subtract_baseline
from app.preprocessing.config import PreprocessConfig, ProcessedSpectrum
from app.preprocessing.normalize import normalize
from app.preprocessing.resample import resample_to_grid
from app.preprocessing.smoothing import savgol_smooth


def preprocess(
    raw: RawSpectrum,
    config: PreprocessConfig | None = None,
) -> ProcessedSpectrum:
    """Применяет полный пайплайн предобработки к сырому спектру.

    Args:
        raw: результат парсинга (``RawSpectrum``).
        config: параметры пайплайна; если ``None``, используется
            :class:`PreprocessConfig` с дефолтами CLAUDE.md §7.

    Returns:
        ``ProcessedSpectrum`` на целевой сетке с применёнными
        преобразованиями. Журнал шагов в ``metadata["pipeline_steps"]``.
    """
    cfg = config or PreprocessConfig()

    grid, intensities = resample_to_grid(raw.wavenumbers, raw.intensities, config=cfg)
    intensities = subtract_baseline(intensities, config=cfg)
    intensities = savgol_smooth(
        intensities, window=cfg.savgol_window, polyorder=cfg.savgol_polyorder
    )
    intensities = normalize(intensities, method=cfg.normalize_method)

    metadata: dict[str, Any] = {
        **raw.metadata,
        "pipeline_steps": ["resample", "baseline", "smoothing", "normalize"],
        "applied_config": cfg.model_dump(),
    }
    return ProcessedSpectrum(wavenumbers=grid, intensities=intensities, metadata=metadata)
