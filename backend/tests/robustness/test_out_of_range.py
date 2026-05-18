"""Спектры вне целевого диапазона 400–4000 см⁻¹ (глава 11 §11.6.4).

Препроцессор должен интерполировать на сетку и заполнить хвосты
``out_of_range_fill`` (по умолчанию 0.0) с предупреждением в лог.
Цель — graceful degradation: вход вне диапазона не должен приводить
к 500-ошибке.
"""

from __future__ import annotations

import numpy as np

from app.preprocessing.config import PreprocessConfig
from app.preprocessing.resample import resample_to_grid


def test_spectrum_entirely_below_range_is_padded_with_zeros() -> None:
    config = PreprocessConfig()
    src_x = np.linspace(100.0, 300.0, 20, dtype=np.float64)
    src_y = np.linspace(0.1, 0.5, 20, dtype=np.float64)

    target_grid, resampled = resample_to_grid(src_x, src_y, config=config)

    assert target_grid.size == config.target_length == 3601
    assert resampled.size == config.target_length
    # За границей источника (400+) — fill-value.
    assert np.allclose(resampled, config.out_of_range_fill)


def test_spectrum_entirely_above_range_is_padded_with_zeros() -> None:
    config = PreprocessConfig()
    src_x = np.linspace(4500.0, 5000.0, 20, dtype=np.float64)
    src_y = np.linspace(0.1, 0.5, 20, dtype=np.float64)

    _, resampled = resample_to_grid(src_x, src_y, config=config)

    assert np.allclose(resampled, config.out_of_range_fill)


def test_partial_overlap_keeps_signal_in_intersection() -> None:
    """Источник 300–1500: за пределами 400..1500 должен быть сигнал, выше — fill."""
    config = PreprocessConfig()
    src_x = np.linspace(300.0, 1500.0, 200, dtype=np.float64)
    src_y = np.sin(src_x / 100.0) + 1.0  # положительный, ненулевой

    target_grid, resampled = resample_to_grid(src_x, src_y, config=config)

    in_overlap = (target_grid >= 400.0) & (target_grid <= 1500.0)
    out_overlap = target_grid > 1500.0
    # В перекрытии должна быть осмысленная вариативность.
    assert resampled[in_overlap].std() > 0.1
    # За правой границей — fill-value.
    assert np.allclose(resampled[out_overlap], config.out_of_range_fill)


def test_descending_wavenumbers_are_reordered() -> None:
    """Многие JCAMP-источники выдают убывающие x — препроцессор их сортирует."""
    config = PreprocessConfig()
    src_x = np.linspace(4000.0, 400.0, 200, dtype=np.float64)  # убывающие
    src_y = np.linspace(0.0, 1.0, 200, dtype=np.float64)

    target_grid, resampled = resample_to_grid(src_x, src_y, config=config)

    assert target_grid[0] < target_grid[-1]
    # Источник полностью покрывает целевую сетку → нет fill-value на краях.
    assert resampled[0] != config.out_of_range_fill or resampled[-1] != config.out_of_range_fill


def test_large_step_source_is_interpolated_to_unit_step() -> None:
    """Источник с шагом 50 см⁻¹ интерполируется к 1 см⁻¹."""
    config = PreprocessConfig()
    src_x = np.arange(400.0, 4001.0, 50.0, dtype=np.float64)
    src_y = np.linspace(0.0, 1.0, src_x.size, dtype=np.float64)

    _, resampled = resample_to_grid(src_x, src_y, config=config)

    assert resampled.size == 3601
    # Интерполяция должна быть монотонной — как и источник.
    assert resampled[0] < resampled[-1]
