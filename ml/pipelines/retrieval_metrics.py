"""Retrieval-метрики для двухбашенной схемы (§6.4.2, §6.4.3 главы 6).

Включает:
- ``topk_accuracy`` (Top-1/5/10) и ``mean_reciprocal_rank`` — стандартные
  retrieval-метрики, где для каждого запроса ровно один правильный «галерейный»
  объект (i-й query соответствует i-му gallery).
- ``tanimoto_matrix`` — попарная Tanimoto-схожесть SMILES по Morgan-fingerprint
  (RDKit). Нужна для §6.4.3 — корреляции эмбеддингов со структурным сходством.
- ``spearman_embedding_tanimoto`` — корреляция Спирмена между cosine
  similarity эмбеддингов и Tanimoto similarity молекул.

Все функции — чистые, без побочных эффектов.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import torch
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from scipy.stats import spearmanr
from torch import Tensor


def topk_accuracy(
    z_query: Tensor,
    z_gallery: Tensor,
    *,
    ks: tuple[int, ...] = (1, 5, 10),
) -> dict[str, float]:
    """Top-k accuracy: для i-го запроса правильный gallery-эмбеддинг — i-й.

    Args:
        z_query: ``(N, D)`` L2-нормированные query-эмбеддинги.
        z_gallery: ``(N, D)`` L2-нормированные gallery-эмбеддинги.
        ks: список значений k.

    Returns:
        Словарь ``{"top1": ..., "top5": ..., ...}``.
    """
    if z_query.shape != z_gallery.shape or z_query.ndim != 2:
        raise ValueError(
            f"ожидались тензоры одинаковой формы (N, D), получено "
            f"z_query={tuple(z_query.shape)} z_gallery={tuple(z_gallery.shape)}"
        )
    n = z_query.shape[0]
    if n == 0:
        return {f"top{k}": float("nan") for k in ks}
    similarities = z_query @ z_gallery.transpose(0, 1)
    # Для каждого i правильный ответ — индекс i. argsort по убыванию.
    sorted_indices = similarities.argsort(dim=1, descending=True)
    targets = torch.arange(n, device=z_query.device).unsqueeze(1)
    rank_hits = sorted_indices == targets  # (N, N)
    out: dict[str, float] = {}
    for k in ks:
        k_eff = min(k, n)
        hits = rank_hits[:, :k_eff].any(dim=1).float()
        out[f"top{k}"] = float(hits.mean().item())
    return out


def mean_reciprocal_rank(z_query: Tensor, z_gallery: Tensor) -> float:
    """MRR: среднее обратного ранга правильного gallery-эмбеддинга."""
    if z_query.shape != z_gallery.shape or z_query.ndim != 2:
        raise ValueError("несовместимые формы для MRR")
    n = z_query.shape[0]
    if n == 0:
        return float("nan")
    similarities = z_query @ z_gallery.transpose(0, 1)
    sorted_indices = similarities.argsort(dim=1, descending=True)
    targets = torch.arange(n, device=z_query.device).unsqueeze(1)
    ranks = (sorted_indices == targets).float().argmax(dim=1) + 1
    return float((1.0 / ranks.float()).mean().item())


def tanimoto_matrix(
    smiles: list[str],
    *,
    radius: int = 2,
    n_bits: int = 2048,
) -> npt.NDArray[np.float64]:
    """Попарная Tanimoto-схожесть для списка SMILES через Morgan-fingerprint.

    Невалидные SMILES игнорируются — соответствующие строки/столбцы заполняются
    NaN, чтобы вызывающий код мог их отфильтровать.
    """
    n = len(smiles)
    if n == 0:
        return np.zeros((0, 0), dtype=np.float64)
    fingerprints: list[object | None] = []
    for s in smiles:
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            fingerprints.append(None)
            continue
        fingerprints.append(
            AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        )
    matrix = np.full((n, n), np.nan, dtype=np.float64)
    for i in range(n):
        if fingerprints[i] is None:
            continue
        matrix[i, i] = 1.0
        for j in range(i + 1, n):
            if fingerprints[j] is None:
                continue
            sim = float(DataStructs.TanimotoSimilarity(fingerprints[i], fingerprints[j]))
            matrix[i, j] = sim
            matrix[j, i] = sim
    return matrix


def spearman_embedding_tanimoto(
    z_a: Tensor,
    z_b: Tensor,
    tanimoto: npt.NDArray[np.float64],
) -> float:
    """Корреляция Спирмена между cos-similarity эмбеддингов и Tanimoto-similarity.

    Используются только пары ``(i, j)`` с ``i < j`` (верхний треугольник),
    где Tanimoto не NaN. Эмбеддинги предполагаются L2-нормированными.
    """
    if z_a.shape != z_b.shape or z_a.ndim != 2:
        raise ValueError("несовместимые формы для Спирмена")
    n = z_a.shape[0]
    if n < 2 or tanimoto.shape != (n, n):
        return float("nan")
    cosine = (z_a @ z_b.transpose(0, 1)).detach().cpu().numpy()
    iu = np.triu_indices(n, k=1)
    cos_flat = cosine[iu]
    tan_flat = tanimoto[iu]
    valid = ~np.isnan(tan_flat)
    if int(valid.sum()) < 2:
        return float("nan")
    result = spearmanr(cos_flat[valid], tan_flat[valid])
    rho = float(result.statistic)  # scipy ≥ 1.10 возвращает SpearmanrResult.statistic
    if np.isnan(rho):
        return float("nan")
    return rho


__all__ = [
    "mean_reciprocal_rank",
    "spearman_embedding_tanimoto",
    "tanimoto_matrix",
    "topk_accuracy",
]
