"""Спектральная башня для контрастного обучения (§6.6.3 главы 6).

Композиция над :class:`FunctionalGroupsCNN` (Этап 5): использует
``forward_embedding`` существующего энкодера, добавляет проекционную голову
(MLP) и L2-нормирует выход для inner-product = cosine similarity в InfoNCE.

Классификационная голова `encoder.classifier` остаётся доступной для multi-task
loss (§6.6.6); поэтому в Этапе 6 трейнер делает по одному forward через
``encoder.forward_embedding`` и вызывает `classifier` (BCE) и проекцию
(InfoNCE) от одного и того же промежуточного представления.
"""

from __future__ import annotations

import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from pipelines.models.cnn1d import FunctionalGroupsCNN


class SpectrumTower(nn.Module):
    """Проекционная башня поверх 1D-CNN.

    Args:
        encoder: обученный (или инициализированный) ``FunctionalGroupsCNN``.
            Энкодер используется через ``forward_embedding`` (без головы
            классификации); сама модель остаётся доступной снаружи через
            атрибут ``encoder`` для подсчёта BCE.
        projection_dim: размерность совместного пространства. По умолчанию
            128 (общая с :class:`MoleculeTower`).
        hidden_dim: размерность скрытого слоя проекции.
        dropout: dropout перед последним линейным слоем.
        freeze_encoder: если True, параметры энкодера не обучаются. По
            умолчанию False — энкодер обучается совместно с проекцией.
    """

    def __init__(
        self,
        encoder: FunctionalGroupsCNN,
        *,
        projection_dim: int = 128,
        hidden_dim: int = 256,
        dropout: float = 0.10,
        freeze_encoder: bool = False,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        in_features = encoder.embedding_dim
        self.projection = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, projection_dim),
        )
        self._projection_dim = projection_dim
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        self._reset_projection()

    @property
    def projection_dim(self) -> int:
        return self._projection_dim

    def _reset_projection(self) -> None:
        for module in self.projection.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def project(self, embedding: Tensor) -> Tensor:
        """Применяет проекцию + L2-нормирование к готовому эмбеддингу CNN."""
        if embedding.shape[-1] != self.encoder.embedding_dim:
            raise ValueError(
                f"эмбеддинг размерности {embedding.shape[-1]} не совпадает с "
                f"encoder.embedding_dim={self.encoder.embedding_dim}"
            )
        # BatchNorm1d требует ≥ 2 элементов в батче в train-режиме;
        # для одиночного forward на CPU (например, в инференсе) переключаемся
        # в eval-режим только для проекции.
        projection_module = self.projection
        if self.training and embedding.shape[0] < 2:
            projection_module = self.projection.eval()
            try:
                z = projection_module(embedding)
            finally:
                self.projection.train()
        else:
            z = projection_module(embedding)
        return F.normalize(z, p=2, dim=-1)

    def forward(self, spectra: Tensor) -> Tensor:
        """Возвращает L2-нормированный эмбеддинг ``(B, projection_dim)``."""
        embedding = self.encoder.forward_embedding(spectra)
        return self.project(embedding)

    def forward_with_embedding(self, spectra: Tensor) -> tuple[Tensor, Tensor]:
        """Возвращает ``(embedding_cnn, projection_normalized)`` за один forward.

        Используется в :class:`ContrastiveTrainer`, чтобы не прогонять CNN
        дважды: BCE считается от ``encoder.classifier(embedding_cnn)``, а
        InfoNCE — от ``projection_normalized``.
        """
        embedding = self.encoder.forward_embedding(spectra)
        return embedding, self.project(embedding)


__all__ = ["SpectrumTower"]
