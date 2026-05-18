"""Тесты retrieval-метрик (§6.4.2, §6.4.3)."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F  # noqa: N812

from pipelines.retrieval_metrics import (
    mean_reciprocal_rank,
    spearman_embedding_tanimoto,
    tanimoto_matrix,
    topk_accuracy,
)


def _unit(t: torch.Tensor) -> torch.Tensor:
    return F.normalize(t, p=2, dim=-1)


def test_topk_accuracy_perfect_alignment() -> None:
    z = _unit(torch.randn(6, 8, generator=torch.Generator().manual_seed(0)))
    metrics = topk_accuracy(z, z, ks=(1, 5))
    assert metrics["top1"] == 1.0
    assert metrics["top5"] == 1.0


def test_topk_accuracy_random_baseline_low() -> None:
    gen = torch.Generator().manual_seed(0)
    z_a = _unit(torch.randn(50, 16, generator=gen))
    z_b = _unit(torch.randn(50, 16, generator=gen))
    metrics = topk_accuracy(z_a, z_b, ks=(1,))
    # Случайные эмбеддинги: top1 ≈ 1/50 ≈ 0.02. Допускаем разброс.
    assert metrics["top1"] < 0.2


def test_mrr_perfect_alignment_is_one() -> None:
    z = _unit(torch.randn(6, 8, generator=torch.Generator().manual_seed(0)))
    assert mean_reciprocal_rank(z, z) == 1.0


def test_mrr_handles_empty() -> None:
    z = torch.zeros(0, 4)
    assert np.isnan(mean_reciprocal_rank(z, z))


def test_tanimoto_matrix_self_is_one() -> None:
    matrix = tanimoto_matrix(["CCO", "c1ccccc1"])
    assert matrix.shape == (2, 2)
    assert matrix[0, 0] == pytest.approx(1.0)
    assert matrix[1, 1] == pytest.approx(1.0)
    assert matrix[0, 1] == matrix[1, 0]


def test_tanimoto_matrix_invalid_smiles_yields_nan() -> None:
    matrix = tanimoto_matrix(["CCO", "not_smiles"])
    assert np.isnan(matrix[1, 1])
    assert np.isnan(matrix[0, 1])


def test_spearman_correlation_perfect_alignment() -> None:
    z = _unit(torch.randn(8, 4, generator=torch.Generator().manual_seed(0)))
    # Tanimoto-матрица повторяет матрицу cosine — даст корреляцию ~1.0.
    cos = (z @ z.T).numpy().astype(np.float64)
    rho = spearman_embedding_tanimoto(z, z, cos)
    assert rho == pytest.approx(1.0, abs=1e-3)


def test_spearman_correlation_returns_nan_for_singleton() -> None:
    z = _unit(torch.randn(1, 4))
    assert np.isnan(spearman_embedding_tanimoto(z, z, np.zeros((1, 1))))
