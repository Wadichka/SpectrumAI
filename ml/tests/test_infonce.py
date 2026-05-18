"""Тесты SymmetricInfoNCE (§6.6.4)."""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F  # noqa: N812

from pipelines.losses import SymmetricInfoNCE, make_loss


def _random_unit_vectors(n: int, d: int, *, seed: int = 0) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    raw = torch.randn(n, d, generator=gen)
    return F.normalize(raw, p=2, dim=-1)


def test_perfect_alignment_loss_near_zero() -> None:
    z = _random_unit_vectors(8, 16, seed=1)
    loss = SymmetricInfoNCE(initial_temperature=0.07, learnable=False)
    # При очень низкой температуре и парах (z_i, z_i) softmax почти one-hot.
    value = float(loss(z, z))
    assert value < 0.01


def test_orthogonal_pairs_loss_close_to_log_n() -> None:
    # Эмбеддинги-оси: z_a == z_b == identity rows. Тогда S = (1/τ) * I — почти one-hot.
    # Чтобы получить log(N), сделаем z_a и z_b ортогональными по парам:
    # z_a[i] @ z_b[j] = 0 для всех i, j → loss = log(N) при любой τ.
    n = 8
    z_a = torch.zeros(n, 16)
    z_a[:, 0] = 1.0
    z_b = torch.zeros(n, 16)
    z_b[:, 1] = 1.0
    loss = SymmetricInfoNCE(initial_temperature=0.07, learnable=False)
    value = float(loss(z_a, z_b))
    assert value == pytest.approx(math.log(n), abs=0.05)


def test_symmetry() -> None:
    z_a = _random_unit_vectors(6, 8, seed=2)
    z_b = _random_unit_vectors(6, 8, seed=3)
    loss = SymmetricInfoNCE(initial_temperature=0.07, learnable=False)
    assert float(loss(z_a, z_b)) == pytest.approx(float(loss(z_b, z_a)), abs=1e-5)


def test_learnable_temperature_is_parameter() -> None:
    loss = SymmetricInfoNCE(initial_temperature=0.07, learnable=True)
    assert "log_temperature" in dict(loss.named_parameters())
    assert loss.temperature == pytest.approx(0.07, rel=1e-3)


def test_fixed_temperature_is_buffer() -> None:
    loss = SymmetricInfoNCE(initial_temperature=0.07, learnable=False)
    assert "log_temperature" not in dict(loss.named_parameters())
    assert "log_temperature" in dict(loss.named_buffers())


def test_make_loss_infonce_factory() -> None:
    cfg = {"type": "infonce", "infonce": {"initial_temperature": 0.1, "learnable": False}}
    loss = make_loss(cfg, pos_weight=None)
    assert isinstance(loss, SymmetricInfoNCE)
    assert loss.temperature == pytest.approx(0.1, rel=1e-3)


def test_shape_mismatch_raises() -> None:
    z_a = _random_unit_vectors(4, 8, seed=4)
    z_b = _random_unit_vectors(4, 16, seed=5)
    loss = SymmetricInfoNCE(learnable=False)
    with pytest.raises(ValueError):
        loss(z_a, z_b)
