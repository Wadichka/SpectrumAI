"""Загрузка чекпойнтов ML-моделей (§4.4.3 главы 4).

Поддерживается два формата чекпойнтов:

- **Contrastive** (``ircnn-contrastive-0.2.0/best.pt``, Этап 6) — содержит
  ``spectrum_tower_state_dict`` с весами обёртки ``SpectrumTower``,
  ``molecule_projection_state_dict`` (для ретривала не нужен — он уже
  «свёрнут» в FAISS-индекс), ``thresholds`` per-class, ``class_names``.
- **CNN-only** (``ircnn-multilabel-0.1.0/best.pt``, Этап 5) — содержит
  ``state_dict`` голой ``FunctionalGroupsCNN``, ``thresholds``,
  ``class_names``.

Функции загружают state dicts в уже инициализированные модули и возвращают
произвольные метаданные из чекпойнта (thresholds, версии и т.п.) для
заполнения :class:`MLComponents`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn


def load_contrastive_into_towers(
    checkpoint_path: Path,
    *,
    spectrum_tower: nn.Module,
    device: torch.device,
) -> dict[str, Any]:
    """Загружает чекпойнт ``ContrastiveTrainer`` в ``SpectrumTower``.

    ``spectrum_tower.encoder`` обновляется неявно — он входит в
    ``spectrum_tower.state_dict()``. Возвращает payload без state-dict-полей
    (для удобства логирования и извлечения метаданных).
    """
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"contrastive-чекпойнт не найден: {checkpoint_path}")
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = payload.get("spectrum_tower_state_dict")
    if state_dict is None:
        raise RuntimeError(f"в чекпойнте {checkpoint_path} нет ключа 'spectrum_tower_state_dict'")
    spectrum_tower.load_state_dict(state_dict, strict=True)
    return {k: v for k, v in payload.items() if not k.endswith("_state_dict")}


def load_cnn_into_model(
    checkpoint_path: Path,
    *,
    cnn: nn.Module,
    device: torch.device,
) -> dict[str, Any]:
    """Загружает чекпойнт обычного multi-label CNN (Этап 5)."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"CNN-чекпойнт не найден: {checkpoint_path}")
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = payload.get("state_dict")
    if state_dict is None:
        raise RuntimeError(f"в чекпойнте {checkpoint_path} нет ключа 'state_dict'")
    cnn.load_state_dict(state_dict, strict=True)
    return {k: v for k, v in payload.items() if not k.endswith("_state_dict") and k != "state_dict"}


__all__ = ["load_cnn_into_model", "load_contrastive_into_towers"]
