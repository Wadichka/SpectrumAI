"""Smoke-тест ContrastiveTrainer: warmup + joint, tiny-bert."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from pipelines.dataset import SpectraDataset
from pipelines.losses import make_loss
from pipelines.models.cnn1d import FunctionalGroupsCNN
from pipelines.models.molecule_tower import TINY_MODEL_NAME, MoleculeTower
from pipelines.models.spectrum_tower import SpectrumTower
from pipelines.training import ContrastiveTrainer, set_global_seed
from torch.utils.data import DataLoader


def _make_mini_parquet(path: Path, *, n: int = 8, length: int = 128, n_labels: int = 4) -> None:
    rng = np.random.default_rng(0)
    spectra = rng.random((n, length)).astype(np.float32)
    labels = rng.integers(0, 2, size=(n, n_labels)).astype(np.int64)
    smiles_pool = ["CCO", "CC(=O)O", "c1ccccc1", "CN", "CC", "CCC", "CCN", "CCl"]
    frame = pd.DataFrame(
        {
            "compound_id": np.arange(n, dtype=np.int64),
            "smiles": [smiles_pool[i % len(smiles_pool)] for i in range(n)],
            "spectrum": [row.tolist() for row in spectra],
            "labels": [row.tolist() for row in labels],
        }
    )
    frame.to_parquet(path, engine="pyarrow", index=False)


def _tiny_cnn(n_labels: int) -> FunctionalGroupsCNN:
    blocks = [
        {"in_channels": 1, "out_channels": 8, "kernel_size": 5, "padding": 2, "dropout": 0.0},
        {"in_channels": 8, "out_channels": 16, "kernel_size": 3, "padding": 1, "dropout": 0.0},
    ]
    return FunctionalGroupsCNN(blocks, embedding_dim=24, n_classes=n_labels)


@pytest.fixture()
def mini_parquet(tmp_path: Path) -> Path:
    path = tmp_path / "mini.parquet"
    _make_mini_parquet(path)
    return path


def test_contrastive_trainer_runs_warmup_and_joint(tmp_path: Path, mini_parquet: Path) -> None:
    set_global_seed(42)
    length, n_labels = 128, 4
    dataset = SpectraDataset(
        mini_parquet,
        spectrum_length=length,
        n_labels=n_labels,
        return_smiles=True,
    )
    loader = DataLoader(dataset, batch_size=4, shuffle=False)

    spectrum_tower = SpectrumTower(_tiny_cnn(n_labels), projection_dim=16, hidden_dim=24)
    molecule_tower = MoleculeTower(TINY_MODEL_NAME, projection_dim=16, hidden_dim=24)

    bce_loss = make_loss({"type": "bce"}, pos_weight=None)
    infonce_loss = make_loss(
        {"type": "infonce", "infonce": {"initial_temperature": 0.1, "learnable": True}},
        pos_weight=None,
    )

    params = (
        [p for p in spectrum_tower.parameters() if p.requires_grad]
        + [p for p in molecule_tower.parameters() if p.requires_grad]
        + [p for p in infonce_loss.parameters() if p.requires_grad]
    )
    optimizer = torch.optim.AdamW(params, lr=1e-3)

    output_dir = tmp_path / "ckpt"
    run_dir = tmp_path / "run"

    trainer = ContrastiveTrainer(
        spectrum_tower=spectrum_tower,
        molecule_tower=molecule_tower,
        bce_loss=bce_loss,
        infonce_loss=infonce_loss,
        optimizer=optimizer,
        scheduler=None,
        device=torch.device("cpu"),
        output_dir=output_dir,
        run_dir=run_dir,
        class_names=[f"g{i}" for i in range(n_labels)],
        bce_weight=1.0,
        nce_weight=0.5,
        warmup_epochs=1,
        threshold_mode="fixed",
        fixed_threshold=0.5,
        grad_clip=1.0,
    )

    state = trainer.fit(loader, loader, epochs=2, patience=5)

    assert state.best_epoch >= 1
    assert (output_dir / "best.pt").exists()
    assert (output_dir / "last.pt").exists()
    payload = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert len(payload["history"]) == 2
    phases = [h["phase"] for h in payload["history"]]
    assert phases == ["warmup", "joint"]
    joint_metrics = payload["history"][1]["metrics"]
    assert "retrieval" in joint_metrics
    assert "top1" in joint_metrics["retrieval"]
