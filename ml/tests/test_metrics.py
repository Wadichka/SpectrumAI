"""Тесты метрик multi-label классификации (§6.4 главы 6)."""

from __future__ import annotations

import numpy as np

from pipelines.metrics import compute_metrics, search_thresholds


def test_compute_metrics_on_perfect_predictions() -> None:
    y_true = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 0]], dtype=np.int64)
    y_prob = y_true.astype(np.float64)
    thresholds = np.full(3, 0.5)
    metrics = compute_metrics(y_true, y_prob, thresholds)
    assert metrics["macro_f1"] == 1.0
    assert metrics["hamming_loss"] == 0.0
    assert metrics["subset_accuracy"] == 1.0


def test_compute_metrics_handles_class_without_positives() -> None:
    y_true = np.array([[1, 0], [1, 0], [0, 0]], dtype=np.int64)
    y_prob = np.array([[0.9, 0.1], [0.8, 0.2], [0.1, 0.4]])
    metrics = compute_metrics(y_true, y_prob, 0.5)
    # Класс 1 без положительных — AUC/AP пропускают его, не должно быть NaN в основных метриках.
    assert isinstance(metrics["macro_f1"], float)
    assert metrics["macro_f1"] >= 0.0
    assert metrics["per_class"]["class_1"]["n_positive"] == 0


def test_search_thresholds_picks_optimal_per_class() -> None:
    rng = np.random.default_rng(0)
    n = 200
    y_true = np.zeros((n, 2), dtype=np.int64)
    y_prob = np.zeros((n, 2), dtype=np.float64)

    # Класс 0: оптимум около 0.3 (положительные имеют prob > 0.3).
    y_true[:100, 0] = 1
    y_prob[:100, 0] = rng.uniform(0.35, 0.6, 100)
    y_prob[100:, 0] = rng.uniform(0.0, 0.2, 100)

    # Класс 1: оптимум около 0.7.
    y_true[:50, 1] = 1
    y_prob[:50, 1] = rng.uniform(0.75, 0.95, 50)
    y_prob[50:, 1] = rng.uniform(0.0, 0.6, 150)

    thresholds = search_thresholds(y_true, y_prob)
    assert 0.2 <= thresholds[0] <= 0.5
    assert 0.5 <= thresholds[1] <= 0.85


def test_search_thresholds_defaults_on_degenerate_class() -> None:
    y_true = np.zeros((10, 2), dtype=np.int64)
    y_prob = np.random.RandomState(0).rand(10, 2)
    thresholds = search_thresholds(y_true, y_prob, default=0.5)
    assert (thresholds == 0.5).all()


def test_compute_metrics_rejects_shape_mismatch() -> None:
    y_true = np.zeros((3, 2), dtype=np.int64)
    y_prob = np.zeros((3, 3), dtype=np.float64)
    try:
        compute_metrics(y_true, y_prob, 0.5)
    except ValueError:
        return
    raise AssertionError("ожидался ValueError для несовпадающих форм")
