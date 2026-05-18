"""Тесты SpectrumTower (§6.6.3)."""

from __future__ import annotations

import torch

from pipelines.models.cnn1d import FunctionalGroupsCNN
from pipelines.models.spectrum_tower import SpectrumTower


def _tiny_cnn() -> FunctionalGroupsCNN:
    blocks = [
        {"in_channels": 1, "out_channels": 8, "kernel_size": 5, "padding": 2, "dropout": 0.0},
        {"in_channels": 8, "out_channels": 16, "kernel_size": 3, "padding": 1, "dropout": 0.0},
    ]
    return FunctionalGroupsCNN(blocks, embedding_dim=32, n_classes=4)


def test_forward_returns_projection_shape() -> None:
    tower = SpectrumTower(_tiny_cnn(), projection_dim=24, hidden_dim=48)
    tower.eval()
    z = tower(torch.zeros(3, 128))
    assert tuple(z.shape) == (3, 24)


def test_forward_is_l2_normalized() -> None:
    tower = SpectrumTower(_tiny_cnn(), projection_dim=16, hidden_dim=32)
    tower.eval()
    z = tower(torch.randn(4, 128))
    norms = z.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_forward_with_embedding_returns_both() -> None:
    cnn = _tiny_cnn()
    tower = SpectrumTower(cnn, projection_dim=16)
    tower.eval()
    embedding, projection = tower.forward_with_embedding(torch.zeros(2, 128))
    assert tuple(embedding.shape) == (2, cnn.embedding_dim)
    assert tuple(projection.shape) == (2, 16)


def test_freeze_encoder_disables_grad() -> None:
    tower = SpectrumTower(_tiny_cnn(), projection_dim=16, freeze_encoder=True)
    for p in tower.encoder.parameters():
        assert not p.requires_grad
    for p in tower.projection.parameters():
        assert p.requires_grad
