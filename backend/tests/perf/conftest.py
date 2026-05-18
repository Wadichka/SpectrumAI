"""Фикстуры perf-сюиты (NFR-01, глава 11 §11.6.5).

Поднимает реальную production-архитектуру 1D-CNN из ``pipelines.models``,
оборачивает её в ``MLComponents`` без FAISS-ретривера (на фазе 1 retrieval
часто отсутствует) и предоставляет ``ProcessedSpectrum`` фиксированной длины.
Использовать `pytest tests/perf/ --benchmark-only` для реальных замеров.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from app.ml import _ml_path  # noqa: F401 — кладёт ml/ в sys.path
from app.ml.components import MLComponents
from app.ml.inference_service import InferenceService
from app.preprocessing.config import PreprocessConfig, ProcessedSpectrum
from pipelines.models.cnn1d import FunctionalGroupsCNN

# Конфигурация, идентичная ml/configs/cnn1d.yaml (production-параметры по §6.2).
_PRODUCTION_BLOCKS = [
    {"in_channels": 1, "out_channels": 32, "kernel_size": 11, "padding": 5, "dropout": 0.10},
    {"in_channels": 32, "out_channels": 64, "kernel_size": 9, "padding": 4, "dropout": 0.15},
    {"in_channels": 64, "out_channels": 128, "kernel_size": 7, "padding": 3, "dropout": 0.20},
    {"in_channels": 128, "out_channels": 256, "kernel_size": 5, "padding": 2, "dropout": 0.25},
    {"in_channels": 256, "out_channels": 256, "kernel_size": 3, "padding": 1, "dropout": 0.0},
]

# Стандартные имена 25 функциональных групп — соответствует §5.4.1.
_CLASS_NAMES = tuple(f"FG{i:02d}" for i in range(1, 26))


@pytest.fixture(scope="session")
def production_cnn() -> FunctionalGroupsCNN:
    """Production-размер 1D-CNN из главы 6.2 в eval-режиме."""
    cnn = FunctionalGroupsCNN(blocks=_PRODUCTION_BLOCKS, embedding_dim=128, n_classes=25)
    cnn.eval()
    return cnn


@pytest.fixture(scope="session")
def production_components(production_cnn: FunctionalGroupsCNN) -> MLComponents:
    """MLComponents без FAISS — измеряем только CNN-инференс (NFR-01 для классификации)."""
    return MLComponents(
        cnn=production_cnn,
        spectrum_tower=None,
        retriever=None,
        class_names=_CLASS_NAMES,
        class_codes=_CLASS_NAMES,
        thresholds=np.full(25, 0.5, dtype=np.float32),
        default_threshold=0.5,
        device=torch.device("cpu"),
        model_versions={"mode": "perf-benchmark", "checkpoint": "synthetic"},
    )


@pytest.fixture(scope="session")
def inference_service(production_components: MLComponents) -> InferenceService:
    return InferenceService(production_components)


@pytest.fixture(scope="session")
def processed_spectrum() -> ProcessedSpectrum:
    """Синтетический ProcessedSpectrum длины 3601 с гауссовыми пиками."""
    config = PreprocessConfig()
    wavenumbers = np.linspace(
        config.target_min, config.target_max, config.target_length, dtype=np.float64
    )
    intensities = np.zeros_like(wavenumbers)
    for center in (1700.0, 2900.0, 3300.0):
        intensities += np.exp(-((wavenumbers - center) ** 2) / (2 * 25.0**2))
    return ProcessedSpectrum(
        wavenumbers=wavenumbers,
        intensities=intensities,
        metadata={"source": "perf-fixture"},
    )
