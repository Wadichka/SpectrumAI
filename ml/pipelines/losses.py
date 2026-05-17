"""Loss-функции SpectrumAI (§6.3, §6.6.4 главы 6).

Реализованы три стратегии:

- **Взвешенная BCE** (§6.3.2) — `nn.BCEWithLogitsLoss` с покласовыми весами
  ``pos_weight = N_neg / N_pos`` для борьбы с дисбалансом редких функциональных
  групп. Значения ограничиваются сверху, чтобы редкие классы не давили loss.
- **Focal Loss** (§6.3.3) — `α · (1 - p_t)^γ · BCE`, по умолчанию γ=2, α=0.5.
  Подходит, если после взвешенной BCE recall редких групп остаётся низким.
- **Symmetric InfoNCE** (§6.6.4) — симметризованный контрастный лосс для
  двухбашенной схемы (Этап 6). Температура τ обучаема через `log_temperature`
  (инициализация log(1/0.07) — как в CLIP).

Выбор стратегии — через секцию ``loss`` в YAML-конфиге.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn


def compute_pos_weight(
    labels: npt.NDArray[np.integer[Any]] | npt.NDArray[np.floating[Any]],
    *,
    cap: float = 100.0,
) -> Tensor:
    """Возвращает ``pos_weight`` для BCEWithLogitsLoss.

    Для каждого класса ``c`` вес равен ``N_negative_c / N_positive_c``,
    но не больше ``cap`` (защита от деления почти на ноль для редких групп).
    Если положительных примеров вообще нет — ставится 1.0.

    Args:
        labels: матрица меток формы ``(N, C)`` со значениями 0/1.
        cap: верхняя граница веса.

    Returns:
        Тензор формы ``(C,)`` со значениями float32.
    """
    arr = np.asarray(labels, dtype=np.int64)
    if arr.ndim != 2:
        raise ValueError(f"ожидалась матрица меток (N, C), получена форма {arr.shape}")
    n_total = arr.shape[0]
    positives = arr.sum(axis=0).astype(np.float64)
    negatives = (n_total - positives).astype(np.float64)

    weights = np.where(
        positives > 0,
        np.minimum(negatives / np.maximum(positives, 1.0), cap),
        1.0,
    )
    return torch.from_numpy(weights.astype(np.float32))


class FocalLoss(nn.Module):
    """Multi-label focal loss (§6.3.3).

    Формула: ``L = α · (1 - p_t)^γ · BCE(logits, targets)``,
    где ``p_t = sigmoid(logits)`` для y=1 и ``1 - sigmoid(logits)`` для y=0.
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.5) -> None:
        super().__init__()
        if gamma < 0:
            raise ValueError("gamma должен быть неотрицательным")
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha должен лежать в [0, 1]")
        self.gamma = float(gamma)
        self.alpha = float(alpha)
        self._bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        bce = self._bce(logits, targets)
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        focal = alpha_t * (1.0 - p_t).pow(self.gamma) * bce
        return focal.mean()


class SymmetricInfoNCE(nn.Module):
    """Симметричный InfoNCE для двухбашенной схемы (§6.6.4).

    Принимает два набора эмбеддингов формы ``(B, D)``. Эмбеддинги должны быть
    предварительно L2-нормированы (это гарантируют ``SpectrumTower`` и
    ``MoleculeTower``). Внутри формула:

        S = z_a @ z_b.T * exp(log_temperature)
        L = 0.5 * (CE(S, arange(B)) + CE(S.T, arange(B)))

    При ``learnable=True`` (по умолчанию) параметр ``log_temperature``
    обучается. Эквивалентная τ доступна как ``self.temperature``.
    """

    def __init__(
        self,
        initial_temperature: float = 0.07,
        learnable: bool = True,
    ) -> None:
        super().__init__()
        if initial_temperature <= 0.0:
            raise ValueError("initial_temperature должна быть положительной")
        # log_temperature := log(1 / τ); τ = exp(-log_temperature).
        init_log = math.log(1.0 / float(initial_temperature))
        log_param = torch.tensor(init_log, dtype=torch.float32)
        if learnable:
            self.log_temperature = nn.Parameter(log_param)
        else:
            self.register_buffer("log_temperature", log_param)
        # Верхняя граница на 1/τ — иначе при обучении τ может скатываться в 0.
        self._max_log_temperature = math.log(100.0)

    @property
    def temperature(self) -> float:
        return float(torch.exp(-self.log_temperature).detach())

    def forward(self, z_a: Tensor, z_b: Tensor) -> Tensor:
        if z_a.shape != z_b.shape:
            raise ValueError(
                f"формы z_a {tuple(z_a.shape)} и z_b {tuple(z_b.shape)} должны совпадать"
            )
        if z_a.ndim != 2:
            raise ValueError(f"ожидался тензор ранга 2 (B, D), получено {z_a.shape}")
        log_t = self.log_temperature.clamp(max=self._max_log_temperature)
        scale = torch.exp(log_t)
        logits = z_a @ z_b.transpose(0, 1) * scale
        targets = torch.arange(z_a.shape[0], device=z_a.device)
        loss_ab = F.cross_entropy(logits, targets)
        loss_ba = F.cross_entropy(logits.transpose(0, 1), targets)
        return 0.5 * (loss_ab + loss_ba)


def make_loss(loss_cfg: dict[str, Any], *, pos_weight: Tensor | None) -> nn.Module:
    """Создаёт loss-модуль по dict-конфигу (секция ``loss`` в YAML)."""
    loss_type = str(loss_cfg.get("type", "bce")).lower()
    if loss_type == "bce":
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    if loss_type == "focal":
        params = loss_cfg.get("focal", {}) or {}
        return FocalLoss(
            gamma=float(params.get("gamma", 2.0)),
            alpha=float(params.get("alpha", 0.5)),
        )
    if loss_type == "infonce":
        params = loss_cfg.get("infonce", {}) or {}
        return SymmetricInfoNCE(
            initial_temperature=float(params.get("initial_temperature", 0.07)),
            learnable=bool(params.get("learnable", True)),
        )
    raise ValueError(f"неизвестный тип loss: {loss_type!r}")


__all__ = ["FocalLoss", "SymmetricInfoNCE", "compute_pos_weight", "make_loss"]
