"""Тесты ``resample_to_grid`` и сборки целевой сетки."""

from __future__ import annotations

import numpy as np

from app.preprocessing import PreprocessConfig, build_target_grid, resample_to_grid


def test_target_grid_default_is_3601_points() -> None:
    grid = build_target_grid(PreprocessConfig())
    assert grid.size == 3601
    assert grid[0] == 400.0
    assert grid[-1] == 4000.0


def test_resample_preserves_values_on_source_nodes() -> None:
    cfg = PreprocessConfig()
    x = np.arange(400.0, 4001.0, 1.0, dtype=np.float64)
    y = np.sin(x / 100.0)
    grid, resampled = resample_to_grid(x, y, config=cfg)
    assert grid.size == 3601
    # Кубический сплайн на собственной сетке должен совпадать поточечно.
    np.testing.assert_allclose(resampled, y, atol=1e-9)


def test_resample_fills_zeros_outside_source_range() -> None:
    cfg = PreprocessConfig()
    x = np.arange(500.0, 3501.0, 1.0, dtype=np.float64)
    y = np.ones_like(x)
    _, resampled = resample_to_grid(x, y, config=cfg)
    # Левые точки до 500 см⁻¹ и правые после 3500 см⁻¹ должны быть нулями.
    assert resampled[0] == 0.0
    assert resampled[-1] == 0.0
    # Внутри исходного диапазона значения близки к 1.
    np.testing.assert_allclose(resampled[1500], 1.0, atol=1e-9)


def test_resample_handles_descending_wavenumbers() -> None:
    cfg = PreprocessConfig()
    x = np.arange(4000.0, 399.0, -1.0, dtype=np.float64)
    y = (x / 4000.0).astype(np.float64)
    grid, resampled = resample_to_grid(x, y, config=cfg)
    assert grid.size == 3601
    assert resampled[0] == 400.0 / 4000.0
    np.testing.assert_allclose(resampled[-1], 1.0, atol=1e-9)
