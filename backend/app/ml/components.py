"""Контейнер инициализированных ML-компонентов и singleton-загрузчик.

``MLComponents`` собирается ровно один раз через :func:`get_ml_components`
(lazy при первом обращении, кэшируется в модуль-глобале). При первом вызове:

1. Импортируется ``pipelines.*`` через side-effect ``_ml_path``.
2. Собирается ``FunctionalGroupsCNN`` по таблице 6.1 (§6.2.2).
3. Поверх собирается ``SpectrumTower`` (§6.6.3) с проекцией 128.
4. Загружается чекпойнт: предпочтительно contrastive (Этап 6), иначе
   CNN-only (Этап 5).
5. По возможности подключается ``FaissRetriever`` (§6.9).
6. Из чекпойнта забираются ``thresholds`` и ``class_names``; если их нет —
   подменяются дефолтами из ``pipelines.labeling``.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import structlog
import torch
from pipelines.labeling import FUNCTIONAL_GROUPS, GROUP_NAMES
from pipelines.models.cnn1d import FunctionalGroupsCNN
from pipelines.models.spectrum_tower import SpectrumTower
from pipelines.retrieval import FaissRetriever

from app.core.config import Settings, get_settings
from app.ml import _ml_path  # noqa: F401 — side-effect: sys.path += ml/
from app.ml.checkpoint_loader import (
    load_cnn_into_model,
    load_contrastive_into_towers,
)

log = structlog.get_logger(__name__)


# Конфиг по умолчанию совпадает с таблицей 6.1 / `ml/configs/cnn1d.yaml`.
_DEFAULT_BLOCKS: list[dict[str, int | float]] = [
    {"in_channels": 1, "out_channels": 32, "kernel_size": 11, "padding": 5, "dropout": 0.10},
    {"in_channels": 32, "out_channels": 64, "kernel_size": 9, "padding": 4, "dropout": 0.15},
    {"in_channels": 64, "out_channels": 128, "kernel_size": 7, "padding": 3, "dropout": 0.20},
    {"in_channels": 128, "out_channels": 256, "kernel_size": 5, "padding": 2, "dropout": 0.25},
    {"in_channels": 256, "out_channels": 256, "kernel_size": 3, "padding": 1, "dropout": 0.0},
]


@dataclass(frozen=True, slots=True)
class MLComponents:
    """Готовые к инференсу модели и метаданные."""

    cnn: FunctionalGroupsCNN
    spectrum_tower: SpectrumTower | None
    retriever: FaissRetriever | None
    class_names: tuple[str, ...]
    class_codes: tuple[str, ...]
    thresholds: np.ndarray
    default_threshold: float
    device: torch.device
    model_versions: dict[str, str] = field(default_factory=dict)


# Singleton-механика. Используем `threading.Lock`, чтобы первый одновременный
# запрос инициализировал модель ровно один раз.
_LOCK = threading.Lock()
_COMPONENTS: MLComponents | None = None


def reset_ml_components() -> None:
    """Сбрасывает кэшированный singleton (используется в тестах)."""
    global _COMPONENTS
    with _LOCK:
        _COMPONENTS = None


def get_ml_components(settings: Settings | None = None) -> MLComponents:
    """Возвращает закэшированный :class:`MLComponents`.

    При первом вызове загружает модели согласно ``settings``. Последующие
    вызовы возвращают тот же экземпляр.
    """
    global _COMPONENTS
    if _COMPONENTS is not None:
        return _COMPONENTS
    with _LOCK:
        if _COMPONENTS is not None:
            return _COMPONENTS
        _COMPONENTS = _build_components(settings or get_settings())
        return _COMPONENTS


def _build_default_cnn() -> FunctionalGroupsCNN:
    return FunctionalGroupsCNN(
        _DEFAULT_BLOCKS,
        embedding_dim=128,
        head_dropout=0.30,
        n_classes=len(FUNCTIONAL_GROUPS),
    )


def _build_default_spectrum_tower(cnn: FunctionalGroupsCNN) -> SpectrumTower:
    return SpectrumTower(cnn, projection_dim=128, hidden_dim=256, dropout=0.10)


def _normalize_thresholds(payload_thresholds: Any, n_classes: int, default: float) -> np.ndarray:
    if payload_thresholds is None:
        return np.full(n_classes, default, dtype=np.float64)
    arr = np.asarray(payload_thresholds, dtype=np.float64)
    if arr.shape != (n_classes,):
        log.warning(
            "thresholds_shape_mismatch",
            expected=n_classes,
            received=int(arr.size),
            fallback=default,
        )
        return np.full(n_classes, default, dtype=np.float64)
    return arr


def _build_components(settings: Settings) -> MLComponents:
    device = torch.device(settings.ml_device)
    cnn = _build_default_cnn()

    model_versions: dict[str, str] = {}
    spectrum_tower: SpectrumTower | None
    payload_meta: dict[str, Any]

    if settings.ml_contrastive_checkpoint.exists():
        spectrum_tower = _build_default_spectrum_tower(cnn)
        payload_meta = load_contrastive_into_towers(
            settings.ml_contrastive_checkpoint,
            spectrum_tower=spectrum_tower,
            device=device,
        )
        spectrum_tower.to(device).eval()
        model_versions["mode"] = "contrastive"
        model_versions["checkpoint"] = settings.ml_contrastive_checkpoint.name
        log.info("ml_loaded_contrastive", path=str(settings.ml_contrastive_checkpoint))
    elif settings.ml_cnn_checkpoint.exists():
        payload_meta = load_cnn_into_model(settings.ml_cnn_checkpoint, cnn=cnn, device=device)
        spectrum_tower = None
        model_versions["mode"] = "cnn-only"
        model_versions["checkpoint"] = settings.ml_cnn_checkpoint.name
        log.info("ml_loaded_cnn_only", path=str(settings.ml_cnn_checkpoint))
    else:
        raise RuntimeError(
            "не найдено ни одного ML-чекпойнта: проверь settings.ml_contrastive_checkpoint "
            f"({settings.ml_contrastive_checkpoint}) и settings.ml_cnn_checkpoint "
            f"({settings.ml_cnn_checkpoint})"
        )

    cnn.to(device).eval()

    payload_class_names = payload_meta.get("class_names") or GROUP_NAMES
    class_names = tuple(str(n) for n in payload_class_names)
    code_lookup = {g.name: g.code for g in FUNCTIONAL_GROUPS}
    class_codes = tuple(code_lookup.get(name, name) for name in class_names)
    thresholds = _normalize_thresholds(
        payload_meta.get("thresholds"), len(class_names), settings.ml_default_threshold
    )

    retriever: FaissRetriever | None = None
    if spectrum_tower is not None and settings.ml_faiss_root.exists():
        try:
            retriever = FaissRetriever.load(settings.ml_faiss_root)
            model_versions["faiss"] = settings.ml_faiss_root.name
            log.info(
                "ml_faiss_loaded",
                path=str(settings.ml_faiss_root),
                size=retriever.size,
                dim=retriever.dim,
            )
        except (FileNotFoundError, RuntimeError) as exc:
            log.warning("ml_faiss_load_failed", path=str(settings.ml_faiss_root), reason=str(exc))
            retriever = None

    return MLComponents(
        cnn=cnn,
        spectrum_tower=spectrum_tower,
        retriever=retriever,
        class_names=class_names,
        class_codes=class_codes,
        thresholds=thresholds,
        default_threshold=float(settings.ml_default_threshold),
        device=device,
        model_versions=model_versions,
    )


__all__ = [
    "MLComponents",
    "get_ml_components",
    "reset_ml_components",
]
