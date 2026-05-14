"""Тесты оркестратора ``preprocess``."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

from app.parsing import RawSpectrum
from app.preprocessing import PreprocessConfig, preprocess


@pytest.fixture()
def raw_spectrum_with_background() -> RawSpectrum:
    """Синтетический сырой спектр с пиком и линейным фоном на разрежённой сетке."""
    x = np.arange(400.0, 4001.0, 5.0, dtype=np.float64)
    peak = np.exp(-(((x - 1700.0) / 25.0) ** 2))
    baseline = 0.0005 * x + 0.1
    rng = np.random.default_rng(seed=7)
    y = (peak + baseline + rng.normal(0.0, 0.01, x.size)).astype(np.float64)
    return RawSpectrum(wavenumbers=x, intensities=y, metadata={"format": "synthetic"})


def test_pipeline_returns_target_length_grid(raw_spectrum_with_background: RawSpectrum) -> None:
    processed = preprocess(raw_spectrum_with_background)
    assert processed.wavenumbers.size == 3601
    assert processed.wavenumbers[0] == 400.0
    assert processed.wavenumbers[-1] == 4000.0


def test_pipeline_records_steps_in_metadata(
    raw_spectrum_with_background: RawSpectrum,
) -> None:
    processed = preprocess(raw_spectrum_with_background)
    assert processed.metadata["pipeline_steps"] == [
        "resample",
        "baseline",
        "smoothing",
        "normalize",
    ]
    # Метаданные исходного спектра сохраняются.
    assert processed.metadata["format"] == "synthetic"
    # Конфиг применённого пайплайна записан целиком.
    assert processed.metadata["applied_config"]["normalize_method"] == "snv"


def test_pipeline_snv_output_has_unit_statistics(
    raw_spectrum_with_background: RawSpectrum,
) -> None:
    processed = preprocess(raw_spectrum_with_background)
    assert abs(float(processed.intensities.mean())) < 1e-9
    np.testing.assert_allclose(float(processed.intensities.std(ddof=0)), 1.0, atol=1e-9)


def test_pipeline_minmax_config_yields_unit_interval(
    raw_spectrum_with_background: RawSpectrum,
) -> None:
    cfg = PreprocessConfig(normalize_method="minmax")
    processed = preprocess(raw_spectrum_with_background, cfg)
    assert float(processed.intensities.min()) == 0.0
    assert float(processed.intensities.max()) == 1.0


def test_pipeline_removes_background_under_peak(
    raw_spectrum_with_background: RawSpectrum,
) -> None:
    # Min-max — линейная нормировка, поэтому относительные положения пика
    # и вне-пиковой области сохраняются и легко проверяются.
    cfg = PreprocessConfig(normalize_method="minmax")
    processed = preprocess(raw_spectrum_with_background, cfg)
    x: npt.NDArray[np.float64] = processed.wavenumbers
    far = np.abs(x - 1700.0) > 200.0
    near = np.abs(x - 1700.0) < 30.0
    # Около пика медиана значимо выше, чем вне.
    assert (
        float(np.median(processed.intensities[near]))
        > float(np.median(processed.intensities[far])) + 0.5
    )


def test_invalid_config_window_even_raises() -> None:
    with pytest.raises(ValueError, match="savgol_window"):
        PreprocessConfig(savgol_window=10)


def test_invalid_config_target_min_max_raises() -> None:
    with pytest.raises(ValueError, match="target_max"):
        PreprocessConfig(target_min=4000.0, target_max=400.0)
