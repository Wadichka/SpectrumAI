"""Стратифицированный сплит multi-label датасета (§6.5.1).

В качестве компромисса используется мягкая стратификация по числу активных
функциональных групп в каждом примере (`labels.sum(axis=1)`). Точный
multi-label стратифицированный сплит требует пакета `iterative-stratification`,
которого в зависимостях нет.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
from sklearn.model_selection import train_test_split


def _stratification_bins(labels: npt.NDArray[np.integer[Any]]) -> npt.NDArray[np.int64]:
    """Дискретизирует число активных групп в бины; -1 для редких комбинаций."""
    counts = labels.sum(axis=1)
    unique, inverse = np.unique(counts, return_inverse=True)
    bin_counts = np.bincount(inverse)
    safe = bin_counts >= 2
    remap = np.where(safe, np.arange(len(unique)), -1)
    return remap[inverse].astype(np.int64)


def stratified_multilabel_split(
    labels: npt.NDArray[np.integer[Any]],
    *,
    ratios: tuple[float, float, float] = (0.70, 0.15, 0.15),
    seed: int = 42,
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """Разбивает индексы на train/val/test с мягкой стратификацией.

    Args:
        labels: матрица меток ``(N, C)``.
        ratios: доли train, val, test (сумма ≈ 1).
        seed: random_state для воспроизводимости.

    Returns:
        Тройка массивов индексов (train, val, test).
    """
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"сумма ratios должна равняться 1.0, получено {sum(ratios)}")
    train_ratio, val_ratio, test_ratio = ratios

    arr = np.asarray(labels, dtype=np.int64)
    n = arr.shape[0]
    indices = np.arange(n, dtype=np.int64)
    bins = _stratification_bins(arr)
    use_first_strat = bool((bins >= 0).all() and len(np.unique(bins)) > 1)

    train_idx, rest_idx = train_test_split(
        indices,
        train_size=train_ratio,
        random_state=seed,
        shuffle=True,
        stratify=bins if use_first_strat else None,
    )
    train_idx = np.asarray(train_idx, dtype=np.int64)
    rest_idx = np.asarray(rest_idx, dtype=np.int64)

    rest_total = val_ratio + test_ratio
    val_share = val_ratio / rest_total if rest_total > 0 else 0.5
    rest_bins = bins[rest_idx]
    use_second_strat = bool((rest_bins >= 0).all() and len(np.unique(rest_bins)) > 1)

    val_idx, test_idx = train_test_split(
        rest_idx,
        train_size=val_share,
        random_state=seed + 1,
        shuffle=True,
        stratify=rest_bins if use_second_strat else None,
    )
    return (
        np.asarray(train_idx, dtype=np.int64),
        np.asarray(val_idx, dtype=np.int64),
        np.asarray(test_idx, dtype=np.int64),
    )


def split_by_inchi_key(
    inchi_keys: list[str] | npt.NDArray[Any],
    *,
    ratios: tuple[float, float, float] = (0.70, 0.15, 0.15),
    seed: int = 42,
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """Сплит индексов по InChI Key — без утечки одной молекулы между сплитами.

    Используется в предзащитном обучении (фаза 2, этап 19): для одной
    молекулы (одинаковый InChI Key) могут быть несколько спектров (разные
    условия регистрации); если они окажутся одновременно в train и val,
    метрики val будут оптимистично смещены.

    Алгоритм:
    1. Уникальные InChI Key перемешиваются с фиксированным seed.
    2. Распределяются по сплитам пропорционально ``ratios``.
    3. Для каждого InChI Key собираются все индексы исходного массива.

    Args:
        inchi_keys: массив InChI Key для каждого спектра (длина N).
        ratios: доли train, val, test (сумма ≈ 1).
        seed: random_state для shuffle.

    Returns:
        Тройка массивов индексов (train, val, test).
    """
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"сумма ratios должна равняться 1.0, получено {sum(ratios)}")

    keys = np.asarray(inchi_keys)
    unique_keys = np.unique(keys)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_keys)

    n_unique = len(unique_keys)
    train_cut = int(n_unique * ratios[0])
    val_cut = train_cut + int(n_unique * ratios[1])

    train_keys = set(unique_keys[:train_cut])
    val_keys = set(unique_keys[train_cut:val_cut])
    # Остальное — test, чтобы не потерять хвост из-за округлений.

    train_idx: list[int] = []
    val_idx: list[int] = []
    test_idx: list[int] = []
    for i, key in enumerate(keys):
        if key in train_keys:
            train_idx.append(i)
        elif key in val_keys:
            val_idx.append(i)
        else:
            test_idx.append(i)

    return (
        np.asarray(train_idx, dtype=np.int64),
        np.asarray(val_idx, dtype=np.int64),
        np.asarray(test_idx, dtype=np.int64),
    )


__all__ = ["split_by_inchi_key", "stratified_multilabel_split"]
