"""1D-CNN multi-label классификатор функциональных групп (§6.2 главы 6).

Архитектура воспроизводит таблицу 6.1: пять свёрточных блоков
(Conv1d → BatchNorm1d → ReLU → MaxPool1d(2) → Dropout), последний блок без
пулинга/dropout — вместо них AdaptiveAvgPool1d(1) и Flatten. Дальше — голова
эмбеддинга 256→128 (используется на Этапе 6 для контрастного обучения) и
линейный классификатор 128→C, где C = число функциональных групп.

Длина входа — 3601 по CLAUDE.md §7. Глава 6.2.1 указывает 1801; расхождение
зафиксировано в плановом файле этапа 5.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from torch import Tensor, nn


@dataclass(frozen=True, slots=True)
class CNNBlockConfig:
    """Конфигурация одного свёрточного блока (по §6.2.2)."""

    in_channels: int
    out_channels: int
    kernel_size: int
    padding: int
    dropout: float


def _to_block_configs(raw: Sequence[dict[str, Any]]) -> tuple[CNNBlockConfig, ...]:
    """Приводит список словарей из YAML к кортежу датаклассов."""
    return tuple(
        CNNBlockConfig(
            in_channels=int(item["in_channels"]),
            out_channels=int(item["out_channels"]),
            kernel_size=int(item["kernel_size"]),
            padding=int(item["padding"]),
            dropout=float(item["dropout"]),
        )
        for item in raw
    )


class FunctionalGroupsCNN(nn.Module):
    """Multi-label CNN-классификатор по таблице 6.1.

    Args:
        blocks: список конфигураций свёрточных блоков (минимум 1).
        embedding_dim: размерность выхода эмбеддинг-головы (по §6.2.2 — 128).
        head_dropout: dropout перед классификатором (по §6.2.4 — 0.30).
        n_classes: число функциональных групп (по §5.4.1 — 25).
    """

    def __init__(
        self,
        blocks: Sequence[dict[str, Any] | CNNBlockConfig],
        *,
        embedding_dim: int = 128,
        head_dropout: float = 0.30,
        n_classes: int = 25,
    ) -> None:
        super().__init__()
        if len(blocks) < 2:
            raise ValueError("требуется минимум два свёрточных блока")

        configs: tuple[CNNBlockConfig, ...]
        if isinstance(blocks[0], CNNBlockConfig):
            configs = tuple(blocks)  # type: ignore[arg-type]
        else:
            configs = _to_block_configs(blocks)  # type: ignore[arg-type]

        # Последний блок (без MaxPool/Dropout) идёт отдельно.
        feature_blocks: list[nn.Module] = []
        for cfg in configs[:-1]:
            feature_blocks.append(
                nn.Sequential(
                    nn.Conv1d(
                        cfg.in_channels,
                        cfg.out_channels,
                        kernel_size=cfg.kernel_size,
                        padding=cfg.padding,
                    ),
                    nn.BatchNorm1d(cfg.out_channels),
                    nn.ReLU(inplace=True),
                    nn.MaxPool1d(kernel_size=2, stride=2),
                    nn.Dropout(cfg.dropout),
                )
            )
        last = configs[-1]
        feature_blocks.append(
            nn.Sequential(
                nn.Conv1d(
                    last.in_channels,
                    last.out_channels,
                    kernel_size=last.kernel_size,
                    padding=last.padding,
                ),
                nn.BatchNorm1d(last.out_channels),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(start_dim=1),
            )
        )
        self.features = nn.Sequential(*feature_blocks)
        self.embedding_head = nn.Sequential(
            nn.Linear(last.out_channels, embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(head_dropout),
        )
        self.classifier = nn.Linear(embedding_dim, n_classes)

        self._embedding_dim = embedding_dim
        self._n_classes = n_classes
        self._reset_parameters()

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    @property
    def n_classes(self) -> int:
        return self._n_classes

    def _reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    @staticmethod
    def _ensure_channel_dim(x: Tensor) -> Tensor:
        """Принимает тензор формы (B, L) или (B, 1, L), возвращает (B, 1, L)."""
        if x.ndim == 2:
            return x.unsqueeze(1)
        if x.ndim == 3:
            return x
        raise ValueError(f"ожидался тензор размерности 2 или 3, получено {x.ndim}")

    def forward_embedding(self, x: Tensor) -> Tensor:
        """Возвращает эмбеддинг спектра размерности ``embedding_dim``."""
        features = self.features(self._ensure_channel_dim(x))
        return self.embedding_head(features)

    def forward(self, x: Tensor) -> Tensor:
        """Возвращает logits multi-label классификатора (B, n_classes)."""
        return self.classifier(self.forward_embedding(x))


def build_model(model_cfg: dict[str, Any]) -> FunctionalGroupsCNN:
    """Собирает модель из dict-конфига (как в `cnn1d.yaml` секция ``model``)."""
    return FunctionalGroupsCNN(
        blocks=model_cfg["blocks"],
        embedding_dim=int(model_cfg.get("embedding_dim", 128)),
        head_dropout=float(model_cfg.get("head_dropout", 0.30)),
        n_classes=int(model_cfg.get("n_classes", 25)),
    )


def count_parameters(model: nn.Module) -> int:
    """Число обучаемых параметров — для логирования и тестов."""
    return sum(int(p.numel()) for p in model.parameters() if p.requires_grad)


__all__ = [
    "CNNBlockConfig",
    "FunctionalGroupsCNN",
    "build_model",
    "count_parameters",
]
