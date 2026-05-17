"""Integration-тест Grad-CAM с реальной FunctionalGroupsCNN (§6.10).

Backend-юнит-тесты (``backend/tests/unit/interpretation/test_gradcam.py``)
работают с мини-моделью. Здесь проверяем, что Grad-CAM корректно
взаимодействует с настоящей архитектурой Этапа 5: 5 свёрточных блоков,
AdaptiveAvgPool, embedding head, classifier.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from pipelines.labeling import GROUP_NAMES, N_GROUPS
from pipelines.models.cnn1d import FunctionalGroupsCNN

# Backend не устанавливается как пакет, поэтому добавляем backend/ в sys.path
# для импорта app.interpretation.gradcam.
_BACKEND_ROOT = Path(__file__).resolve().parents[2] / "backend"
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from app.interpretation.gradcam import (  # noqa: E402
    GradCAM1D,
    compute_gradcam,
    default_target_layer,
)


def _build_cnn() -> FunctionalGroupsCNN:
    blocks = [
        {"in_channels": 1, "out_channels": 32, "kernel_size": 11, "padding": 5, "dropout": 0.10},
        {"in_channels": 32, "out_channels": 64, "kernel_size": 9, "padding": 4, "dropout": 0.15},
        {"in_channels": 64, "out_channels": 128, "kernel_size": 7, "padding": 3, "dropout": 0.20},
        {"in_channels": 128, "out_channels": 256, "kernel_size": 5, "padding": 2, "dropout": 0.25},
        {"in_channels": 256, "out_channels": 256, "kernel_size": 3, "padding": 1, "dropout": 0.0},
    ]
    return FunctionalGroupsCNN(blocks, embedding_dim=128, head_dropout=0.30, n_classes=N_GROUPS)


def test_default_target_layer_is_last_conv1d_of_features() -> None:
    model = _build_cnn()
    expected = model.features[-1][0]
    assert default_target_layer(model) is expected


def test_gradcam_all_groups_shape_and_norm() -> None:
    torch.manual_seed(0)
    model = _build_cnn()
    model.eval()
    spectrum = torch.randn(3601)
    target_layer = default_target_layer(model)
    result = compute_gradcam(
        model,
        target_layer,
        spectrum,
        class_names=GROUP_NAMES,
        target_length=3601,
    )
    assert set(result.keys()) == set(GROUP_NAMES)
    for value in result.values():
        assert value.shape == (3601,)
        assert value.dtype == np.float32
        assert value.min() >= 0.0
        assert value.max() <= 1.0 + 1e-6


def test_gradcam_subset_of_classes() -> None:
    torch.manual_seed(1)
    model = _build_cnn()
    model.eval()
    spectrum = torch.randn(3601)
    with GradCAM1D(
        model, default_target_layer(model), class_names=GROUP_NAMES, target_length=3601
    ) as cam:
        partial = cam.compute(spectrum, class_indices=[0, 5, 14])
    assert list(partial.keys()) == [GROUP_NAMES[0], GROUP_NAMES[5], GROUP_NAMES[14]]


_CHECKPOINT = Path("models/ircnn-multilabel-0.1.0/best.pt")


@pytest.mark.skipif(not _CHECKPOINT.exists(), reason="чекпойнт Этапа 5 не найден — sanity-skip")
def test_gradcam_on_pretrained_checkpoint_runs() -> None:
    """Sanity: с обученным чекпойнтом Grad-CAM выдаёт ненулевые карты.

    На синтетике (фаза 1) сильная проверка диапазонов O-H stretching не имеет
    смысла — модель обучена на гауссах, а не реальных колебательных модах.
    Здесь только проверяем, что код проходит до конца и хотя бы одна карта
    не нулевая (модель что-то «видит»).
    """
    payload = torch.load(_CHECKPOINT, map_location="cpu", weights_only=False)
    model = _build_cnn()
    model.load_state_dict(payload["state_dict"])
    model.eval()
    spectrum = torch.randn(3601)
    result = compute_gradcam(
        model,
        default_target_layer(model),
        spectrum,
        class_names=GROUP_NAMES,
        target_length=3601,
    )
    any_active = any(float(arr.max()) > 1e-6 for arr in result.values())
    assert any_active, "все карты Grad-CAM нулевые — что-то сломано в графе"
