"""Фабрики синтетических спектров для unit-тестов предобработки."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest


def _gaussian(
    x: npt.NDArray[np.float64], center: float, sigma: float, amplitude: float
) -> npt.NDArray[np.float64]:
    return amplitude * np.exp(-(((x - center) / sigma) ** 2))


@pytest.fixture()
def linear_background() -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Чистый линейный фон без пиков на сетке 400–4000 см⁻¹ шагом 1."""
    x = np.arange(400.0, 4001.0, 1.0, dtype=np.float64)
    y = (0.0005 * x + 0.2).astype(np.float64)
    return x, y


@pytest.fixture()
def peak_on_background() -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Гауссов пик на 1700 см⁻¹ поверх линейного фона."""
    x = np.arange(400.0, 4001.0, 1.0, dtype=np.float64)
    peak = _gaussian(x, center=1700.0, sigma=20.0, amplitude=1.0)
    baseline = 0.0005 * x + 0.1
    return x, (peak + baseline).astype(np.float64)


@pytest.fixture()
def noisy_signal() -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Гладкий сигнал с белым шумом — для теста сглаживания."""
    x = np.arange(400.0, 4001.0, 1.0, dtype=np.float64)
    clean = _gaussian(x, center=2000.0, sigma=100.0, amplitude=1.0)
    noise = np.random.default_rng(seed=42).normal(0.0, 0.05, x.size)
    return x, (clean + noise).astype(np.float64)
