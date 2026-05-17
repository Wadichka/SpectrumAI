"""Молекулярная башня на ChemBERTa (§6.7 главы 6).

Использует предобученную модель `seyonec/ChemBERTa-zinc-base-v1` (§6.7.1)
с замороженными весами трансформера (§6.7.3) и обучаемой проекционной
головой. Эмбеддинг — CLS-токен последнего слоя (§6.7.4).

Для CI и быстрых тестов поддерживается переменная окружения
``SPECTRUMAI_USE_TINY_BERT=1`` — модель подменяется на
``hf-internal-testing/tiny-random-roberta``, чтобы не качать 150 МБ.
Альтернативно имя модели можно задать через ``SPECTRUMAI_CHEMBERTA_NAME``.
"""

from __future__ import annotations

import os

import torch
import torch.nn.functional as F  # noqa: N812
from rdkit import Chem
from torch import Tensor, nn
from transformers import AutoModel, AutoTokenizer  # type: ignore[import-untyped]

DEFAULT_MODEL_NAME = "seyonec/ChemBERTa-zinc-base-v1"
TINY_MODEL_NAME = "hf-internal-testing/tiny-random-roberta"


def resolve_model_name(explicit: str | None = None) -> str:
    """Возвращает имя HF-модели с учётом env-override.

    Приоритет: явный аргумент > ``SPECTRUMAI_CHEMBERTA_NAME`` >
    ``SPECTRUMAI_USE_TINY_BERT=1`` (включает tiny-mock) > ``DEFAULT_MODEL_NAME``.
    """
    if explicit:
        return explicit
    env_name = os.environ.get("SPECTRUMAI_CHEMBERTA_NAME")
    if env_name:
        return env_name
    if os.environ.get("SPECTRUMAI_USE_TINY_BERT") == "1":
        return TINY_MODEL_NAME
    return DEFAULT_MODEL_NAME


class MoleculeTower(nn.Module):
    """Молекулярный энкодер ChemBERTa + проекционная голова.

    Args:
        model_name: имя HF-модели. Если None — берётся через
            :func:`resolve_model_name` с учётом env-переменных.
        projection_dim: размерность совместного пространства (по умолчанию 128).
        hidden_dim: размерность скрытого слоя проекции.
        dropout: dropout перед последним линейным слоем.
        freeze_encoder: если True (по умолчанию), параметры ChemBERTa
            замораживаются (§6.7.3). Для фазы 2 можно разморозить.
        max_length: максимальная длина токенизированной последовательности.
    """

    def __init__(
        self,
        model_name: str | None = None,
        *,
        projection_dim: int = 128,
        hidden_dim: int = 384,
        dropout: float = 0.10,
        freeze_encoder: bool = True,
        max_length: int = 128,
    ) -> None:
        super().__init__()
        resolved = resolve_model_name(model_name)
        self.model_name = resolved
        self.tokenizer = AutoTokenizer.from_pretrained(resolved)
        self.encoder = AutoModel.from_pretrained(resolved)
        self.max_length = int(max_length)
        self._projection_dim = projection_dim

        encoder_dim = int(self.encoder.config.hidden_size)
        self.projection = nn.Sequential(
            nn.Linear(encoder_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, projection_dim),
        )
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
            self.encoder.eval()
        self._reset_projection()

    @property
    def projection_dim(self) -> int:
        return self._projection_dim

    def train(self, mode: bool = True) -> MoleculeTower:  # type: ignore[override]
        """Переопределение train() — энкодер всегда остаётся в eval, если он заморожен."""
        super().train(mode)
        if not any(p.requires_grad for p in self.encoder.parameters()):
            self.encoder.eval()
        return self

    def _reset_projection(self) -> None:
        for module in self.projection.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    @staticmethod
    def canonicalize(smiles: str) -> str:
        """Канонизирует SMILES через RDKit (§6.7.4).

        Невалидный SMILES → ``ValueError`` с понятным сообщением.
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"invalid SMILES: {smiles!r}")
        return Chem.MolToSmiles(mol, canonical=True)

    def tokenize(self, smiles: list[str]) -> dict[str, Tensor]:
        canonical = [self.canonicalize(s) for s in smiles]
        encoded = self.tokenizer(
            canonical,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {k: v for k, v in encoded.items()}

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def forward(self, smiles: list[str]) -> Tensor:
        """Возвращает L2-нормированный эмбеддинг ``(B, projection_dim)``."""
        if not smiles:
            raise ValueError("smiles не должен быть пустым списком")
        tokens = self.tokenize(smiles)
        device = self.device
        tokens = {k: v.to(device) for k, v in tokens.items()}

        encoder_was_frozen = not any(p.requires_grad for p in self.encoder.parameters())
        if encoder_was_frozen:
            with torch.no_grad():
                outputs = self.encoder(**tokens)
            cls = outputs.last_hidden_state[:, 0, :]
        else:
            outputs = self.encoder(**tokens)
            cls = outputs.last_hidden_state[:, 0, :]

        # LayerNorm-голова работает на любом размере батча, даже 1.
        z = self.projection(cls)
        return F.normalize(z, p=2, dim=-1)


__all__ = ["DEFAULT_MODEL_NAME", "TINY_MODEL_NAME", "MoleculeTower", "resolve_model_name"]
