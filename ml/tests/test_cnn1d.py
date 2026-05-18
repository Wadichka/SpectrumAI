"""Тесты архитектуры 1D-CNN (§6.2 главы 6)."""

from __future__ import annotations

import torch

from pipelines.models.cnn1d import CNNBlockConfig, FunctionalGroupsCNN, count_parameters


def _default_blocks() -> list[dict[str, int | float]]:
    return [
        {"in_channels": 1, "out_channels": 32, "kernel_size": 11, "padding": 5, "dropout": 0.10},
        {"in_channels": 32, "out_channels": 64, "kernel_size": 9, "padding": 4, "dropout": 0.15},
        {"in_channels": 64, "out_channels": 128, "kernel_size": 7, "padding": 3, "dropout": 0.20},
        {"in_channels": 128, "out_channels": 256, "kernel_size": 5, "padding": 2, "dropout": 0.25},
        {"in_channels": 256, "out_channels": 256, "kernel_size": 3, "padding": 1, "dropout": 0.0},
    ]


def test_forward_returns_expected_logits_shape() -> None:
    model = FunctionalGroupsCNN(_default_blocks(), embedding_dim=128, n_classes=25)
    model.eval()
    x = torch.zeros(4, 3601)
    logits = model(x)
    assert tuple(logits.shape) == (4, 25)


def test_forward_accepts_channel_dim() -> None:
    model = FunctionalGroupsCNN(_default_blocks(), embedding_dim=128, n_classes=25)
    model.eval()
    x = torch.zeros(2, 1, 3601)
    logits = model(x)
    assert tuple(logits.shape) == (2, 25)


def test_forward_embedding_shape_matches_config() -> None:
    model = FunctionalGroupsCNN(_default_blocks(), embedding_dim=128, n_classes=25)
    model.eval()
    embedding = model.forward_embedding(torch.zeros(3, 3601))
    assert tuple(embedding.shape) == (3, 128)


def test_parameter_count_in_expected_range() -> None:
    model = FunctionalGroupsCNN(_default_blocks(), embedding_dim=128, n_classes=25)
    total = count_parameters(model)
    assert 400_000 <= total <= 800_000, f"unexpected parameter count: {total}"


def test_dataclass_blocks_accepted() -> None:
    configs = [
        CNNBlockConfig(1, 8, 5, 2, 0.0),
        CNNBlockConfig(8, 16, 3, 1, 0.0),
    ]
    model = FunctionalGroupsCNN(configs, embedding_dim=8, n_classes=3)
    model.eval()
    out = model(torch.zeros(2, 64))
    assert tuple(out.shape) == (2, 3)


def test_forward_rejects_wrong_rank() -> None:
    model = FunctionalGroupsCNN(_default_blocks(), embedding_dim=16, n_classes=2)
    try:
        model(torch.zeros(2, 1, 1, 3601))
    except ValueError:
        return
    raise AssertionError("ожидался ValueError для тензора неверного ранга")
