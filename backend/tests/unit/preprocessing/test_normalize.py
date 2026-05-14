"""Тесты функций нормирования."""

from __future__ import annotations

import numpy as np

from app.preprocessing import min_max_normalize, normalize, snv_normalize


def test_snv_yields_zero_mean_unit_std() -> None:
    rng = np.random.default_rng(0)
    y = rng.normal(loc=5.0, scale=2.0, size=500).astype(np.float64)
    out = snv_normalize(y)
    assert abs(float(out.mean())) < 1e-12
    np.testing.assert_allclose(float(out.std(ddof=0)), 1.0, atol=1e-12)


def test_snv_handles_constant_input() -> None:
    out = snv_normalize(np.full(10, 3.5, dtype=np.float64))
    assert np.all(out == 0.0)


def test_min_max_maps_to_unit_interval() -> None:
    y = np.array([2.0, 4.0, 6.0, 8.0], dtype=np.float64)
    out = min_max_normalize(y)
    assert float(out.min()) == 0.0
    assert float(out.max()) == 1.0


def test_min_max_handles_constant_input() -> None:
    out = min_max_normalize(np.full(5, 7.0, dtype=np.float64))
    assert np.all(out == 0.0)


def test_normalize_dispatcher_chooses_method() -> None:
    y = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    np.testing.assert_allclose(normalize(y, method="snv"), snv_normalize(y))
    np.testing.assert_allclose(normalize(y, method="minmax"), min_max_normalize(y))
