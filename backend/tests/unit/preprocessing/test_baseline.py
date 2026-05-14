"""Тесты AsLS baseline correction."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from app.preprocessing import PreprocessConfig, asls_baseline, subtract_baseline


def test_asls_estimates_linear_background(
    linear_background: tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]],
) -> None:
    _, y = linear_background
    baseline = asls_baseline(y, lam=1.0e5, p=0.01, niter=10)
    # AsLS должна почти точно восстановить линейный фон.
    residual = y - baseline
    assert float(np.std(residual)) < 0.01


def test_subtract_baseline_removes_background_under_peak(
    peak_on_background: tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]],
) -> None:
    x, y = peak_on_background
    cfg = PreprocessConfig()
    flat = subtract_baseline(y, config=cfg)

    # В вне-пиковой области остаточный фон должен быть мал.
    far_from_peak = np.abs(x - 1700.0) > 200.0
    assert float(np.mean(np.abs(flat[far_from_peak]))) < 0.02

    # Пик сохранился по высоте в пределах 10 % (фон сдвигает амплитуду
    # умеренно — итерационный AsLS не идеально занулён под пиком).
    original_peak_height = float(y[np.argmin(np.abs(x - 1700.0))])
    flat_peak_height = float(flat[np.argmin(np.abs(x - 1700.0))])
    expected = original_peak_height - (0.0005 * 1700.0 + 0.1)
    assert abs(flat_peak_height - expected) / expected < 0.10
