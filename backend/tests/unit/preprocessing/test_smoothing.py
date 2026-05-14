"""Тесты сглаживания Savitzky–Golay."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from app.preprocessing import savgol_smooth


def test_savgol_preserves_quadratic_signal() -> None:
    # Savitzky–Golay с polyorder=2 должен точно восстанавливать квадратичный полином.
    x = np.linspace(-10.0, 10.0, 401)
    y = (3.0 * x**2 + 2.0 * x + 1.0).astype(np.float64)
    smoothed = savgol_smooth(y, window=11, polyorder=2)
    np.testing.assert_allclose(smoothed, y, atol=1e-9)


def test_savgol_reduces_white_noise(
    noisy_signal: tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]],
) -> None:
    _, y = noisy_signal
    smoothed = savgol_smooth(y, window=11, polyorder=2)
    # Стандартное отклонение разности до/после — индикатор подавления шума.
    raw_high_freq_std = float(np.std(np.diff(y)))
    smoothed_high_freq_std = float(np.std(np.diff(smoothed)))
    assert smoothed_high_freq_std < raw_high_freq_std / 2.0
