"""Метрики multi-label классификации (§6.4 главы 6).

Реализованы:

- ``macro_f1``, ``micro_f1``, ``weighted_f1`` (sklearn) — §6.4.1.
- ``macro_ap`` (mean Average Precision) — §6.4.1.
- ``macro_auc`` (ROC-AUC) — §6.4.1.
- ``hamming_loss``, ``subset_accuracy`` — §6.4.1.
- ``per_class`` — F1, AP, ROC-AUC и порог по каждой группе.

Retrieval-метрики (Top-k, MRR, Tanimoto) — Этап 6.

`search_thresholds` реализует §6.9.3: грид-поиск порога по каждому классу,
выбирается значение с максимальным per-class F1 на валидации.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    hamming_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _has_positive_and_negative(column: npt.NDArray[np.integer[Any]]) -> bool:
    return bool(column.any()) and bool((~column.astype(bool)).any())


def search_thresholds(
    y_true: npt.NDArray[np.integer[Any]],
    y_prob: npt.NDArray[np.floating[Any]],
    *,
    grid: npt.NDArray[np.floating[Any]] | None = None,
    default: float = 0.5,
) -> npt.NDArray[np.float64]:
    """Подбирает порог классификации для каждого класса.

    Для каждого класса перебирается грид порогов; выбирается тот, при котором
    per-class F1 максимален. Если у класса нет положительных или отрицательных
    примеров в валидации — ставится ``default`` (по умолчанию 0.5).

    Args:
        y_true: матрица истинных меток ``(N, C)``.
        y_prob: матрица вероятностей ``(N, C)`` в диапазоне [0, 1].
        grid: массив порогов; по умолчанию ``np.arange(0.05, 0.96, 0.05)``.
        default: значение для классов с вырожденной валидацией.
    """
    if grid is None:
        grid = np.arange(0.05, 0.96, 0.05, dtype=np.float64)
    y_true_arr = np.asarray(y_true, dtype=np.int64)
    y_prob_arr = np.asarray(y_prob, dtype=np.float64)
    if y_true_arr.shape != y_prob_arr.shape:
        raise ValueError(
            f"формы y_true {y_true_arr.shape} и y_prob {y_prob_arr.shape} должны совпадать"
        )

    n_classes = y_true_arr.shape[1]
    thresholds = np.full(n_classes, default, dtype=np.float64)

    for c in range(n_classes):
        column = y_true_arr[:, c]
        if not _has_positive_and_negative(column):
            continue
        best_f1 = -1.0
        best_th = default
        for threshold in grid:
            preds = (y_prob_arr[:, c] >= threshold).astype(np.int64)
            score = float(f1_score(column, preds, zero_division=0))
            if score > best_f1:
                best_f1 = score
                best_th = float(threshold)
        thresholds[c] = best_th
    return thresholds


def _safe_auc(y_true: npt.NDArray[np.integer[Any]], y_prob: npt.NDArray[np.floating[Any]]) -> float:
    """ROC-AUC по столбцам, усреднённый по классам с обоими исходами."""
    valid_columns = [c for c in range(y_true.shape[1]) if _has_positive_and_negative(y_true[:, c])]
    if not valid_columns:
        return float("nan")
    aucs: list[float] = []
    for c in valid_columns:
        aucs.append(float(roc_auc_score(y_true[:, c], y_prob[:, c])))
    return float(np.mean(aucs))


def _safe_ap(y_true: npt.NDArray[np.integer[Any]], y_prob: npt.NDArray[np.floating[Any]]) -> float:
    """mAP по классам с хотя бы одним положительным примером."""
    valid_columns = [c for c in range(y_true.shape[1]) if y_true[:, c].any()]
    if not valid_columns:
        return float("nan")
    aps = [float(average_precision_score(y_true[:, c], y_prob[:, c])) for c in valid_columns]
    return float(np.mean(aps))


def compute_metrics(
    y_true: npt.NDArray[np.integer[Any]],
    y_prob: npt.NDArray[np.floating[Any]],
    thresholds: npt.NDArray[np.floating[Any]] | float,
    *,
    class_names: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Считает все метрики multi-label классификации.

    Args:
        y_true: истинные метки ``(N, C)``.
        y_prob: вероятности ``(N, C)`` в [0, 1].
        thresholds: массив порогов длины C, либо одно число для всех классов.
        class_names: имена классов для блока ``per_class``.
    """
    y_true_arr = np.asarray(y_true, dtype=np.int64)
    y_prob_arr = np.asarray(y_prob, dtype=np.float64)
    if y_true_arr.shape != y_prob_arr.shape:
        raise ValueError(
            f"формы y_true {y_true_arr.shape} и y_prob {y_prob_arr.shape} должны совпадать"
        )
    n_classes = y_true_arr.shape[1]
    if isinstance(thresholds, int | float):
        th_arr = np.full(n_classes, float(thresholds), dtype=np.float64)
    else:
        th_arr = np.asarray(thresholds, dtype=np.float64)
        if th_arr.shape != (n_classes,):
            raise ValueError(
                f"длина thresholds {th_arr.shape} не совпадает с числом классов {n_classes}"
            )

    y_pred = (y_prob_arr >= th_arr[np.newaxis, :]).astype(np.int64)

    macro_f1 = float(f1_score(y_true_arr, y_pred, average="macro", zero_division=0))
    micro_f1 = float(f1_score(y_true_arr, y_pred, average="micro", zero_division=0))
    weighted_f1 = float(f1_score(y_true_arr, y_pred, average="weighted", zero_division=0))
    macro_precision = float(precision_score(y_true_arr, y_pred, average="macro", zero_division=0))
    macro_recall = float(recall_score(y_true_arr, y_pred, average="macro", zero_division=0))

    metrics: dict[str, Any] = {
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
        "weighted_f1": weighted_f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_ap": _safe_ap(y_true_arr, y_prob_arr),
        "macro_auc": _safe_auc(y_true_arr, y_prob_arr),
        "hamming_loss": float(hamming_loss(y_true_arr, y_pred)),
        "subset_accuracy": float(np.all(y_true_arr == y_pred, axis=1).mean()),
    }

    per_class: dict[str, dict[str, float]] = {}
    f1_per_class = f1_score(y_true_arr, y_pred, average=None, zero_division=0)
    precision_per_class = precision_score(y_true_arr, y_pred, average=None, zero_division=0)
    recall_per_class = recall_score(y_true_arr, y_pred, average=None, zero_division=0)

    for c in range(n_classes):
        name = class_names[c] if class_names is not None else f"class_{c}"
        per_class[name] = {
            "f1": float(f1_per_class[c]),
            "precision": float(precision_per_class[c]),
            "recall": float(recall_per_class[c]),
            "threshold": float(th_arr[c]),
            "n_positive": int(y_true_arr[:, c].sum()),
        }
    metrics["per_class"] = per_class
    return metrics


__all__ = ["compute_metrics", "search_thresholds"]
