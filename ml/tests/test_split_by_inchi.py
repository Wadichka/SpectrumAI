"""Тесты split_by_inchi_key — гарантия отсутствия утечки молекул между сплитами."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_ML_ROOT = Path(__file__).resolve().parents[1]
if str(_ML_ROOT) not in sys.path:
    sys.path.insert(0, str(_ML_ROOT))

from pipelines.training.split import split_by_inchi_key  # noqa: E402


def test_no_leakage_between_splits() -> None:
    # 6 уникальных молекул, по 2 спектра на каждую → 12 индексов.
    keys = [f"KEY{i}" for i in range(6) for _ in range(2)]
    train, val, test = split_by_inchi_key(keys, ratios=(0.5, 0.25, 0.25), seed=0)

    train_keys = {keys[i] for i in train}
    val_keys = {keys[i] for i in val}
    test_keys = {keys[i] for i in test}

    assert train_keys.isdisjoint(val_keys)
    assert train_keys.isdisjoint(test_keys)
    assert val_keys.isdisjoint(test_keys)


def test_all_indices_assigned_exactly_once() -> None:
    keys = ["A", "A", "B", "C", "C", "C", "D", "E", "F", "G"]
    train, val, test = split_by_inchi_key(keys, ratios=(0.7, 0.15, 0.15), seed=7)
    union = np.sort(np.concatenate([train, val, test]))
    assert len(union) == len(keys)
    assert np.array_equal(union, np.arange(len(keys), dtype=np.int64))


def test_deterministic_with_seed() -> None:
    keys = [f"K{i // 2}" for i in range(20)]
    a = split_by_inchi_key(keys, seed=42)
    b = split_by_inchi_key(keys, seed=42)
    for left, right in zip(a, b, strict=True):
        assert np.array_equal(left, right)


def test_ratios_validation() -> None:
    with pytest.raises(ValueError, match="ratios"):
        split_by_inchi_key(["A", "B"], ratios=(0.5, 0.4, 0.2), seed=0)


def test_approximate_split_sizes() -> None:
    keys = [f"K{i}" for i in range(100)]  # 100 уникальных молекул, по 1 спектру
    train, val, test = split_by_inchi_key(keys, ratios=(0.7, 0.15, 0.15), seed=42)
    assert 65 <= len(train) <= 75
    assert 10 <= len(val) <= 20
    assert 10 <= len(test) <= 25
