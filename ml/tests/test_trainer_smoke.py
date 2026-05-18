"""Smoke-тест цикла обучения: одна эпоха на мини-датасете."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from pipelines.dataset import SpectraDataset
from pipelines.losses import make_loss
from pipelines.models.cnn1d import FunctionalGroupsCNN
from pipelines.training import Trainer, set_global_seed


def _make_mini_parquet(path: Path, *, n: int = 12, length: int = 256, n_labels: int = 5) -> None:
    rng = np.random.default_rng(0)
    spectra = rng.random((n, length)).astype(np.float32)
    labels = rng.integers(0, 2, size=(n, n_labels)).astype(np.int64)
    frame = pd.DataFrame(
        {
            "compound_id": np.arange(n, dtype=np.int64),
            "smiles": [f"C{i}" for i in range(n)],
            "spectrum": [row.tolist() for row in spectra],
            "labels": [row.tolist() for row in labels],
        }
    )
    frame.to_parquet(path, engine="pyarrow", index=False)


@pytest.fixture()
def mini_dataset(tmp_path: Path) -> Path:
    path = tmp_path / "mini.parquet"
    _make_mini_parquet(path)
    return path


def test_trainer_runs_one_epoch_and_writes_artifacts(tmp_path: Path, mini_dataset: Path) -> None:
    set_global_seed(42)
    length, n_labels = 256, 5
    dataset = SpectraDataset(mini_dataset, spectrum_length=length, n_labels=n_labels)
    loader = DataLoader(dataset, batch_size=4, shuffle=False)

    blocks = [
        {"in_channels": 1, "out_channels": 8, "kernel_size": 5, "padding": 2, "dropout": 0.0},
        {"in_channels": 8, "out_channels": 16, "kernel_size": 3, "padding": 1, "dropout": 0.0},
    ]
    model = FunctionalGroupsCNN(blocks, embedding_dim=8, n_classes=n_labels)
    loss_fn = make_loss({"type": "bce"}, pos_weight=None)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    output_dir = tmp_path / "checkpoints"
    run_dir = tmp_path / "run"

    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=None,
        device=torch.device("cpu"),
        output_dir=output_dir,
        run_dir=run_dir,
        class_names=[f"g{i}" for i in range(n_labels)],
        threshold_mode="fixed",
        fixed_threshold=0.5,
    )

    state = trainer.fit(loader, loader, epochs=1, patience=5)

    assert state.best_epoch == 1
    assert (output_dir / "last.pt").exists()
    assert (output_dir / "best.pt").exists()
    metrics_path = run_dir / "metrics.json"
    assert metrics_path.exists()
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert len(payload["history"]) == 1
    epoch_record = payload["history"][0]
    assert "macro_f1" in epoch_record["metrics"]
    assert isinstance(epoch_record["metrics"]["macro_f1"], float)
