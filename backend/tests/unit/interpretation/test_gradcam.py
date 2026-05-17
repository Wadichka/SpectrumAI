"""Unit-тесты Grad-CAM 1D (§6.10).

Backend не зависит от ``ml/`` (см. Этап 7 плана), поэтому модели для тестов
собираются прямо здесь — без импортов ``FunctionalGroupsCNN``.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import Tensor, nn

from app.interpretation.gradcam import GradCAM1D, compute_gradcam, default_target_layer


class TinyCNN(nn.Module):
    """Минимальный 1D-CNN для тестов: два Conv1d, AdaptiveAvgPool, Linear."""

    def __init__(self, n_classes: int = 3, channels: int = 4) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, channels, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )
        self.classifier = nn.Linear(channels, n_classes)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(1)
        return self.classifier(self.features(x))

    @property
    def target_layer(self) -> nn.Module:
        return self.features[2]  # последний Conv1d


def _make_model(seed: int = 0, n_classes: int = 3) -> TinyCNN:
    torch.manual_seed(seed)
    return TinyCNN(n_classes=n_classes)


# ---------------------------------------------------------------------------
# Базовые проверки формы и нормировки.
# ---------------------------------------------------------------------------
def test_compute_returns_dict_with_expected_keys() -> None:
    model = _make_model(n_classes=3)
    names = ["a", "b", "c"]
    spectrum = torch.randn(64)
    result = compute_gradcam(
        model, model.target_layer, spectrum, class_names=names, target_length=64
    )
    assert list(result.keys()) == names


def test_compute_returns_int_keys_when_no_names() -> None:
    model = _make_model(n_classes=2)
    spectrum = torch.randn(32)
    result = compute_gradcam(model, model.target_layer, spectrum, target_length=32)
    assert set(result.keys()) == {0, 1}


def test_each_map_has_target_length() -> None:
    model = _make_model()
    spectrum = torch.randn(64)
    result = compute_gradcam(model, model.target_layer, spectrum, target_length=200)
    for value in result.values():
        assert value.shape == (200,)
        assert value.dtype == np.float32


def test_each_map_is_in_unit_interval() -> None:
    model = _make_model()
    spectrum = torch.randn(64)
    result = compute_gradcam(model, model.target_layer, spectrum, target_length=64)
    for value in result.values():
        assert value.min() >= 0.0
        assert value.max() <= 1.0 + 1e-6


# ---------------------------------------------------------------------------
# Эдж-кейсы.
# ---------------------------------------------------------------------------
def test_zero_map_when_logit_is_constant() -> None:
    """Если классификатор замёрз на константу, градиенты ~ 0 → карта нулевая."""
    model = _make_model()
    nn.init.zeros_(model.classifier.weight)
    nn.init.zeros_(model.classifier.bias)
    spectrum = torch.randn(64)
    result = compute_gradcam(model, model.target_layer, spectrum, target_length=64)
    for value in result.values():
        assert np.allclose(value, 0.0)


def test_batch_size_greater_than_one_rejected() -> None:
    model = _make_model()
    spectrum = torch.randn(2, 1, 64)
    with pytest.raises(ValueError, match="batch=1"):
        compute_gradcam(model, model.target_layer, spectrum, target_length=64)


def test_compute_outside_context_raises() -> None:
    model = _make_model()
    cam = GradCAM1D(model, model.target_layer, target_length=64)
    with pytest.raises(RuntimeError, match="context manager"):
        cam.compute(torch.randn(64))


def test_invalid_class_index_raises() -> None:
    model = _make_model(n_classes=3)
    spectrum = torch.randn(64)
    with pytest.raises(IndexError):
        compute_gradcam(model, model.target_layer, spectrum, class_indices=[5], target_length=64)


def test_invalid_target_length_raises() -> None:
    model = _make_model()
    with pytest.raises(ValueError):
        GradCAM1D(model, model.target_layer, target_length=0)


# ---------------------------------------------------------------------------
# Lifecycle: hooks остаются только внутри контекста.
# ---------------------------------------------------------------------------
def test_hooks_removed_after_context_exit() -> None:
    model = _make_model()
    layer = model.target_layer
    cam = GradCAM1D(model, layer, target_length=64)
    with cam:
        # Внутри блока два хука: forward и full_backward.
        assert len(layer._forward_hooks) == 1
        assert len(layer._backward_hooks) == 1
        cam.compute(torch.randn(64), class_indices=[0])
    assert len(layer._forward_hooks) == 0
    assert len(layer._backward_hooks) == 0


# ---------------------------------------------------------------------------
# default_target_layer эвристика.
# ---------------------------------------------------------------------------
def test_default_target_layer_picks_last_conv1d() -> None:
    model = _make_model()
    expected = model.features[2]  # второй Conv1d
    assert default_target_layer(model) is expected


def test_default_target_layer_raises_without_conv1d() -> None:
    model = nn.Sequential(nn.Linear(10, 4), nn.ReLU(), nn.Linear(4, 3))
    with pytest.raises(ValueError):
        default_target_layer(model)


# ---------------------------------------------------------------------------
# Sanity-check: после короткого обучения максимум карты сдвигается к пику.
# ---------------------------------------------------------------------------
def test_peak_in_input_drives_peak_in_map() -> None:
    """Учим простую модель отвечать «1» на сигнал с пиком в окрестности позиции X.

    После короткого обучения Grad-CAM по этому классу должен выделять
    окрестность пика, а не противоположный край.
    """
    torch.manual_seed(0)
    length = 64
    peak_pos = 50
    model = TinyCNN(n_classes=1, channels=8)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
    loss_fn = nn.BCEWithLogitsLoss()

    def make_pair() -> tuple[Tensor, Tensor]:
        signal = torch.zeros(length)
        sigma = 2.0
        x_grid = torch.arange(length, dtype=torch.float32)
        signal += torch.exp(-((x_grid - peak_pos) ** 2) / (2 * sigma**2))
        return signal, torch.tensor([1.0])

    # 50 шагов SGD на парах (peak, 1) и (random, 0).
    for _ in range(50):
        positive, target_pos = make_pair()
        negative = torch.randn(length) * 0.1
        target_neg = torch.tensor([0.0])
        batch = torch.stack([positive, negative])
        targets = torch.stack([target_pos, target_neg])
        optimizer.zero_grad()
        logits = model(batch)
        loss = loss_fn(logits, targets)
        loss.backward()
        optimizer.step()

    spectrum, _ = make_pair()
    cam_map = compute_gradcam(
        model, model.target_layer, spectrum, class_indices=[0], target_length=length
    )[0]
    argmax = int(np.argmax(cam_map))
    # Пик должен быть в окрестности peak_pos (5% длины — 3 точки).
    tolerance = max(3, length // 20)
    assert abs(argmax - peak_pos) <= tolerance, (
        f"argmax={argmax}, peak_pos={peak_pos}, tolerance={tolerance}"
    )
