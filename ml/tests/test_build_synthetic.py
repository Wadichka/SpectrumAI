"""Тесты скрипта генерации синтетического датасета."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from pipelines.labeling import GROUP_NAMES
from scripts.build_synthetic_dataset import SMILES_LIST, TARGET_LENGTH, build


@pytest.fixture()
def parquet_path(tmp_path: Path) -> Path:
    out = tmp_path / "synthetic.parquet"
    build(out, seed=42)
    return out


def test_build_writes_parquet_with_expected_columns(parquet_path: Path) -> None:
    frame = pd.read_parquet(parquet_path)
    assert set(frame.columns) >= {"compound_id", "smiles", "spectrum", "labels"}
    assert len(frame) == len(SMILES_LIST)
    assert len(frame) >= 200


def test_build_each_spectrum_has_target_length(parquet_path: Path) -> None:
    frame = pd.read_parquet(parquet_path)
    sample = frame.iloc[0]
    assert len(sample["spectrum"]) == TARGET_LENGTH
    assert len(sample["labels"]) == 25


def test_build_covers_all_25_groups(parquet_path: Path) -> None:
    frame = pd.read_parquet(parquet_path)
    label_matrix = frame["labels"].apply(list).tolist()
    sums = [sum(row[i] for row in label_matrix) for i in range(25)]
    uncovered = [GROUP_NAMES[i] for i, s in enumerate(sums) if s == 0]
    assert not uncovered, f"группы без представителей: {uncovered}"
