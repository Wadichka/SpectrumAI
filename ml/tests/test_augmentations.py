"""Тесты аугментаций спектров (§5.7)."""

from __future__ import annotations

import numpy as np
import pytest

from pipelines.augmentations import (
    BaselineDistortion,
    Compose,
    GaussianNoise,
    IntensityScale,
    SpectralCutout,
    WavenumberShift,
    WavenumberStretch,
    default_train_augmentation,
)


@pytest.fixture()
def rng() -> np.random.Generator:
    return np.random.default_rng(seed=123)


@pytest.fixture()
def peak_spectrum() -> np.ndarray:
    """Гауссов пик на середине сетки длины 3601."""
    grid = np.arange(3601, dtype=np.float64)
    return np.exp(-(((grid - 1800.0) / 50.0) ** 2))


def test_gaussian_noise_changes_signal(rng: np.random.Generator, peak_spectrum: np.ndarray) -> None:
    aug = GaussianNoise(rng=rng)
    out = aug(peak_spectrum)
    assert out.shape == peak_spectrum.shape
    assert not np.array_equal(out, peak_spectrum)
    # Амплитуда шума должна быть в пределах σ_high * peak * 5σ (запас на хвосты).
    residual = out - peak_spectrum
    assert float(np.std(residual)) < 0.05  # peak = 1, sigma_high = 0.01


def test_intensity_scale_multiplies_uniformly(
    rng: np.random.Generator, peak_spectrum: np.ndarray
) -> None:
    aug = IntensityScale(rng=rng, scale_low=1.5, scale_high=1.5)
    out = aug(peak_spectrum)
    np.testing.assert_allclose(out, peak_spectrum * 1.5, atol=1e-6, rtol=0)


def test_wavenumber_shift_preserves_length(
    rng: np.random.Generator, peak_spectrum: np.ndarray
) -> None:
    aug = WavenumberShift(rng=rng, shift_min=3, shift_max=3)
    out = aug(peak_spectrum)
    assert out.shape == peak_spectrum.shape
    # Сдвиг на +3 → пик в позиции 1803, концы заполнены нулями.
    assert out[0] == 0.0
    assert out[1] == 0.0
    assert out[2] == 0.0
    np.testing.assert_allclose(out[3:], peak_spectrum[:-3])


def test_wavenumber_stretch_preserves_length(
    rng: np.random.Generator, peak_spectrum: np.ndarray
) -> None:
    aug = WavenumberStretch(rng=rng)
    out = aug(peak_spectrum)
    assert out.shape == peak_spectrum.shape


def test_baseline_distortion_bounded(rng: np.random.Generator, peak_spectrum: np.ndarray) -> None:
    aug = BaselineDistortion(rng=rng, max_relative=0.05)
    out = aug(peak_spectrum)
    residual = out - peak_spectrum
    assert float(np.max(np.abs(residual))) <= 0.05 + 1e-9


def test_spectral_cutout_zeros_region(rng: np.random.Generator, peak_spectrum: np.ndarray) -> None:
    aug = SpectralCutout(rng=rng, width=100)
    out = aug(peak_spectrum)
    # Где-то 100 подряд идущих нулей должны появиться.
    zeros_runs = np.diff(np.where(np.concatenate(([1], out != 0, [1])))[0]) - 1
    assert int(zeros_runs.max()) >= 100


def test_compose_zero_probability_is_noop(
    rng: np.random.Generator, peak_spectrum: np.ndarray
) -> None:
    aug = GaussianNoise(rng=rng, probability=0.0)
    composed = Compose(transforms=[aug], rng=rng)
    out = composed(peak_spectrum)
    np.testing.assert_array_equal(out, peak_spectrum)


def test_default_train_augmentation_returns_array(
    rng: np.random.Generator, peak_spectrum: np.ndarray
) -> None:
    composed = default_train_augmentation(rng)
    out = composed(peak_spectrum)
    assert out.shape == peak_spectrum.shape
    assert out.dtype == np.float64
