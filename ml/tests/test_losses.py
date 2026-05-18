"""Тесты loss-функций (§6.3 главы 6)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from pipelines.losses import FocalLoss, compute_pos_weight, make_loss


def test_compute_pos_weight_balances_simple_case() -> None:
    labels = np.array([[1, 0], [0, 0], [0, 0]], dtype=np.int64)
    weights = compute_pos_weight(labels)
    assert weights.shape == (2,)
    assert weights[0].item() == pytest.approx(2.0)
    assert weights[1].item() == pytest.approx(1.0)


def test_compute_pos_weight_caps_extreme_imbalance() -> None:
    labels = np.zeros((1001, 1), dtype=np.int64)
    labels[0, 0] = 1
    weights = compute_pos_weight(labels, cap=50.0)
    assert weights[0].item() == pytest.approx(50.0)


def test_make_loss_bce_matches_torch() -> None:
    torch.manual_seed(0)
    logits = torch.randn(4, 5)
    targets = torch.randint(0, 2, (4, 5)).float()
    custom = make_loss({"type": "bce"}, pos_weight=None)
    reference = torch.nn.BCEWithLogitsLoss()
    assert torch.allclose(custom(logits, targets), reference(logits, targets))


def test_make_loss_unknown_type_raises() -> None:
    with pytest.raises(ValueError):
        make_loss({"type": "exotic"}, pos_weight=None)


def test_focal_loss_reduces_easy_example_contribution() -> None:
    logits = torch.tensor([[5.0]])  # очень уверенное и правильное предсказание
    targets = torch.tensor([[1.0]])
    bce = torch.nn.BCEWithLogitsLoss()(logits, targets)
    focal = FocalLoss(gamma=2.0, alpha=0.5)(logits, targets)
    assert focal.item() < bce.item()


def test_focal_loss_rejects_invalid_params() -> None:
    with pytest.raises(ValueError):
        FocalLoss(gamma=-1.0)
    with pytest.raises(ValueError):
        FocalLoss(alpha=2.0)
