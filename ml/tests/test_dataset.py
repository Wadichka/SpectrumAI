"""Тесты ``SpectraDataset``."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from pipelines.dataset import SpectraDataset

_SPECTRUM_LENGTH = 3601
_N_LABELS = 25


@pytest.fixture()
def tmp_parquet(tmp_path: Path) -> Path:
    """Маленький parquet (5 строк) с фейковыми спектрами и метками."""
    rng = np.random.default_rng(seed=0)
    rows = []
    for i in range(5):
        spec = rng.uniform(0.0, 1.0, _SPECTRUM_LENGTH).astype(np.float32).tolist()
        labels = rng.integers(0, 2, _N_LABELS).astype(np.int8).tolist()
        rows.append({"compound_id": i, "smiles": f"C{i}", "spectrum": spec, "labels": labels})
    path = tmp_path / "tiny.parquet"
    pd.DataFrame(rows).to_parquet(path, engine="pyarrow", index=False)
    return path


def test_dataset_len_matches_rows(tmp_parquet: Path) -> None:
    ds = SpectraDataset(tmp_parquet)
    assert len(ds) == 5


def test_dataset_returns_expected_tensor_shapes(tmp_parquet: Path) -> None:
    ds = SpectraDataset(tmp_parquet)
    spectrum, labels = ds[0]
    assert isinstance(spectrum, torch.Tensor)
    assert isinstance(labels, torch.Tensor)
    assert spectrum.shape == (_SPECTRUM_LENGTH,)
    assert labels.shape == (_N_LABELS,)
    assert spectrum.dtype == torch.float32
    assert labels.dtype == torch.float32


def test_dataset_applies_transform(tmp_parquet: Path) -> None:
    def double(spec: np.ndarray) -> np.ndarray:
        return spec * 2.0

    ds_plain = SpectraDataset(tmp_parquet)
    ds_aug = SpectraDataset(tmp_parquet, transform=double)
    plain, _ = ds_plain[0]
    augmented, _ = ds_aug[0]
    np.testing.assert_allclose(augmented.numpy(), plain.numpy() * 2.0, rtol=0, atol=1e-6)


def test_dataset_iterates_through_dataloader(tmp_parquet: Path) -> None:
    ds = SpectraDataset(tmp_parquet)
    loader = DataLoader(ds, batch_size=2, shuffle=False)
    batches = list(loader)
    assert len(batches) == 3  # 5 строк: 2+2+1
    spectra, labels = batches[0]
    assert spectra.shape == (2, _SPECTRUM_LENGTH)
    assert labels.shape == (2, _N_LABELS)


def test_dataset_get_metadata(tmp_parquet: Path) -> None:
    ds = SpectraDataset(tmp_parquet)
    meta = ds.get_metadata(2)
    assert meta == {"smiles": "C2", "compound_id": 2}


def test_dataset_rejects_missing_columns(tmp_path: Path) -> None:
    pd.DataFrame({"only_one": [1, 2, 3]}).to_parquet(
        tmp_path / "broken.parquet", engine="pyarrow", index=False
    )
    with pytest.raises(ValueError, match="не хватает колонок"):
        SpectraDataset(tmp_path / "broken.parquet")
