"""1D-вариант Grad-CAM для multi-label CNN-классификатора (§6.10 главы 6).

Реализован «руками» без Captum — реализация на 80 строк, прозрачнее для
объяснимости результатов на защите. Зависимости — только torch и numpy.

Формула (§6.10.2):

    α_k^c = (1/Z) · Σ_i ∂y^c/∂A_i^k
    L^c   = ReLU( Σ_k α_k^c · A^k )

где ``A^k`` — карта активаций k-го канала целевого свёрточного слоя, ``y^c`` —
**логит** класса c (не вероятность). После пост-обработки карта линейно
интерполируется до длины спектра (по умолчанию 3601, см. CLAUDE.md §7) и
min-max-нормируется в [0, 1].

Класс :class:`GradCAM1D` — context manager: внутри ``with`` блока на
целевом слое держатся forward/backward хуки, по выходу из блока они снимаются.

Реализация generic для любой `nn.Module` с указанным `target_layer` —
backend пока не зависит от ``ml/`` (Этап 8). Для удобства добавлена эвристика
:func:`default_target_layer`, ищущая последний ``nn.Conv1d`` в графе модели.
"""

from __future__ import annotations

from collections.abc import Sequence
from types import TracebackType

import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn


def default_target_layer(model: nn.Module) -> nn.Module:
    """Возвращает рекомендуемый целевой слой для Grad-CAM — последний Conv1d.

    Для :class:`ml.pipelines.models.cnn1d.FunctionalGroupsCNN` это Conv1d в
    блоке 5 (``features[-1][0]``); метод применим к любой CNN с слоями
    Conv1d.
    """
    last_conv: nn.Module | None = None
    for module in model.modules():
        if isinstance(module, nn.Conv1d):
            last_conv = module
    if last_conv is None:
        raise ValueError("в модели не найдено ни одного слоя Conv1d")
    return last_conv


def _ensure_batch_one(spectrum: Tensor) -> Tensor:
    """Приводит спектр к форме (1, 1, L). Батчи > 1 отвергаются явно."""
    if spectrum.ndim == 1:
        return spectrum.unsqueeze(0).unsqueeze(0)
    if spectrum.ndim == 2:
        if spectrum.shape[0] != 1:
            raise ValueError(f"Grad-CAM поддерживает только batch=1; получено {spectrum.shape[0]}")
        return spectrum.unsqueeze(1)
    if spectrum.ndim == 3:
        if spectrum.shape[0] != 1:
            raise ValueError(f"Grad-CAM поддерживает только batch=1; получено {spectrum.shape[0]}")
        return spectrum
    raise ValueError(f"ожидался тензор размерности 1, 2 или 3, получено {spectrum.ndim}")


class GradCAM1D:
    """1D-вариант Grad-CAM для multi-label логитов.

    Args:
        model: обучаемая CNN-модель в ``eval``-режиме.
        target_layer: модуль, на чьи активации/градиенты вешаются хуки.
            Обычно — последний Conv1d перед глобальным пулингом.
        class_names: имена классов; если переданы, ключи возвращаемого
            словаря будут строковыми, иначе — int-индексами.
        target_length: длина выходной карты после интерполяции; по умолчанию
            3601 (длина обработанного спектра, CLAUDE.md §7).
    """

    def __init__(
        self,
        model: nn.Module,
        target_layer: nn.Module,
        *,
        class_names: Sequence[str] | None = None,
        target_length: int = 3601,
    ) -> None:
        if target_length <= 0:
            raise ValueError("target_length должен быть положительным")
        self._model = model
        self._target_layer = target_layer
        self._class_names = tuple(class_names) if class_names is not None else None
        self._target_length = int(target_length)
        self._activations: Tensor | None = None
        self._gradients: Tensor | None = None
        self._handles: list[torch.utils.hooks.RemovableHandle] = []

    # ------------------------------------------------------------------
    # Context-manager hook lifecycle.
    # ------------------------------------------------------------------
    def __enter__(self) -> GradCAM1D:
        def forward_hook(_module: nn.Module, _inputs: tuple[Tensor, ...], output: Tensor) -> None:
            # Сохраняем активации без detach — они должны быть в графе.
            self._activations = output

        def backward_hook(
            _module: nn.Module,
            _grad_input: tuple[Tensor | None, ...],
            grad_output: tuple[Tensor | None, ...],
        ) -> None:
            grad = grad_output[0]
            if grad is not None:
                self._gradients = grad.detach()

        self._handles.append(self._target_layer.register_forward_hook(forward_hook))
        self._handles.append(self._target_layer.register_full_backward_hook(backward_hook))
        return self

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc: BaseException | None,
        _tb: TracebackType | None,
    ) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._activations = None
        self._gradients = None

    # ------------------------------------------------------------------
    # Computation.
    # ------------------------------------------------------------------
    def compute(
        self,
        spectrum: Tensor,
        class_indices: Sequence[int] | None = None,
    ) -> dict[str | int, npt.NDArray[np.float32]]:
        """Считает карты Grad-CAM для указанных классов.

        Args:
            spectrum: тензор формы ``(L,)``, ``(1, L)`` или ``(1, 1, L)``.
            class_indices: список классов; по умолчанию — все классы из
                ``class_names`` либо логиты модели, если имён нет.

        Returns:
            Словарь ``{class_key: ndarray[float32, length=target_length]}``,
            где ключ — имя (если задано) или индекс класса.
        """
        if not self._handles:
            raise RuntimeError(
                "GradCAM1D должен использоваться как context manager: with GradCAM1D(...) as cam:"
            )
        x = _ensure_batch_one(spectrum)
        x = x.to(next(self._model.parameters()).device)

        was_training = self._model.training
        self._model.eval()
        try:
            # Forward — захватываем активации через хук.
            self._model.zero_grad(set_to_none=True)
            logits = self._model(x)
            if logits.ndim != 2 or logits.shape[0] != 1:
                raise ValueError(f"ожидались logits формы (1, C); получено {tuple(logits.shape)}")
            n_classes = int(logits.shape[1])

            indices: list[int]
            if class_indices is not None:
                indices = [int(i) for i in class_indices]
            elif self._class_names is not None:
                indices = list(range(len(self._class_names)))
            else:
                indices = list(range(n_classes))

            for idx in indices:
                if idx < 0 or idx >= n_classes:
                    raise IndexError(f"class index {idx} вне диапазона [0, {n_classes})")

            result: dict[str | int, npt.NDArray[np.float32]] = {}
            for position, class_idx in enumerate(indices):
                key: str | int
                if self._class_names is not None and class_idx < len(self._class_names):
                    key = self._class_names[class_idx]
                else:
                    key = class_idx
                # Очищаем градиенты предыдущего класса.
                self._gradients = None
                self._model.zero_grad(set_to_none=True)
                # retain_graph для всех, кроме последнего, чтобы один forward
                # обслужил все классы.
                retain = position < len(indices) - 1
                logits[0, class_idx].backward(retain_graph=retain)
                result[key] = self._cam_from_state()
        finally:
            if was_training:
                self._model.train()
        return result

    def compute_one(self, spectrum: Tensor, class_index: int) -> npt.NDArray[np.float32]:
        """Удобный shortcut для одной карты."""
        out = self.compute(spectrum, class_indices=[class_index])
        return next(iter(out.values()))

    # ------------------------------------------------------------------
    # Internals.
    # ------------------------------------------------------------------
    def _cam_from_state(self) -> npt.NDArray[np.float32]:
        if self._activations is None or self._gradients is None:
            raise RuntimeError(
                "хуки не получили активации/градиенты — проверь, что forward модели "
                "проходит через target_layer"
            )
        activations = self._activations  # (1, C, L')
        gradients = self._gradients  # (1, C, L')
        # α_k = mean по позициям градиентов.
        weights = gradients.mean(dim=2)  # (1, C)
        cam = (weights.unsqueeze(-1) * activations).sum(dim=1)  # (1, L')
        cam = F.relu(cam).unsqueeze(0)  # (1, 1, L')
        cam = F.interpolate(cam, size=self._target_length, mode="linear", align_corners=False)
        cam_array = cam.squeeze(0).squeeze(0).detach().cpu().numpy().astype(np.float32)
        max_value = float(cam_array.max())
        if max_value > 0.0:
            cam_array = cam_array / max_value
        return np.clip(cam_array, 0.0, 1.0).astype(np.float32)


def compute_gradcam(
    model: nn.Module,
    target_layer: nn.Module,
    spectrum: Tensor,
    *,
    class_indices: Sequence[int] | None = None,
    class_names: Sequence[str] | None = None,
    target_length: int = 3601,
) -> dict[str | int, npt.NDArray[np.float32]]:
    """Одноразовая обёртка над :class:`GradCAM1D`.

    Создаёт context manager, считает карты, возвращает словарь. Удобно, когда
    Grad-CAM нужен один раз — например, в Этапе 8 (InferenceService).
    """
    with GradCAM1D(
        model, target_layer, class_names=class_names, target_length=target_length
    ) as cam:
        return cam.compute(spectrum, class_indices=class_indices)


__all__ = ["GradCAM1D", "compute_gradcam", "default_target_layer"]
