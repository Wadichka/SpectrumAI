"""Воспроизводимость обучения (§6.5.4 главы 6)."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_global_seed(seed: int = 42) -> None:
    """Фиксирует все источники случайности.

    Включает ``torch.backends.cudnn.deterministic`` и отключает
    ``benchmark``: производительность ниже, зато прогон воспроизводим
    между запусками (требование §6.5.4 пояснительной записки).
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


__all__ = ["set_global_seed"]
