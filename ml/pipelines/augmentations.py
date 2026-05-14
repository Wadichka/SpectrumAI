"""Аугментации спектров (§5.7 главы 5).

Реализованы 6 трансформаций без mixup (mixup опционален по §5.7.2 и
добавляется в финальной тонкой настройке). Каждая аугментация хранит
``probability``; ``Compose`` подбрасывает RNG и применяет с этой вероятностью.

Порядок по §5.7.3: сначала структурные (shift, stretch, baseline_distortion,
cutout), затем амплитудные (scale, noise). Все используют общий
``np.random.Generator`` для воспроизводимости.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import numpy.typing as npt
from scipy.interpolate import interp1d


class Augmentation(Protocol):
    """Минимальный интерфейс аугментации."""

    probability: float

    def __call__(self, spectrum: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]: ...


# -----------------------------------------------------------------------------
# Амплитудные аугментации.
# -----------------------------------------------------------------------------


@dataclass
class GaussianNoise:
    """Добавляет гауссов шум с σ ∈ [sigma_low, sigma_high] от max интенсивности."""

    rng: np.random.Generator
    sigma_low: float = 0.001
    sigma_high: float = 0.01
    probability: float = 0.5

    def __call__(self, spectrum: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        sigma_rel = float(self.rng.uniform(self.sigma_low, self.sigma_high))
        peak = float(np.max(np.abs(spectrum))) or 1.0
        return spectrum + self.rng.normal(0.0, sigma_rel * peak, spectrum.shape)


@dataclass
class IntensityScale:
    """Умножает интенсивности на коэффициент из [scale_low, scale_high]."""

    rng: np.random.Generator
    scale_low: float = 0.9
    scale_high: float = 1.1
    probability: float = 0.3

    def __call__(self, spectrum: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        k = float(self.rng.uniform(self.scale_low, self.scale_high))
        return spectrum * k


# -----------------------------------------------------------------------------
# Структурные аугментации.
# -----------------------------------------------------------------------------


@dataclass
class WavenumberShift:
    """Сдвиг оси волновых чисел на Δ ∈ [shift_min, shift_max] см⁻¹.

    Сдвиг реализован сдвигом значений массива (np.roll с интерполяцией концов
    нулями) — это эквивалентно сдвигу оси на Δ при шаге сетки 1 см⁻¹.
    """

    rng: np.random.Generator
    shift_min: int = -5
    shift_max: int = 5
    probability: float = 0.3

    def __call__(self, spectrum: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        delta = int(self.rng.integers(self.shift_min, self.shift_max + 1))
        if delta == 0:
            return spectrum.copy()
        result = np.zeros_like(spectrum)
        if delta > 0:
            result[delta:] = spectrum[:-delta]
        else:
            result[:delta] = spectrum[-delta:]
        return result


@dataclass
class WavenumberStretch:
    """Растяжение/сжатие оси с коэффициентом из [low, high]."""

    rng: np.random.Generator
    low: float = 0.98
    high: float = 1.02
    probability: float = 0.2

    def __call__(self, spectrum: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        k = float(self.rng.uniform(self.low, self.high))
        length = spectrum.size
        original_axis = np.arange(length, dtype=np.float64)
        stretched_axis = original_axis * k
        interpolator = interp1d(
            stretched_axis,
            spectrum,
            kind="linear",
            bounds_error=False,
            fill_value=0.0,
        )
        return np.asarray(interpolator(original_axis), dtype=np.float64)


@dataclass
class BaselineDistortion:
    """Добавляет полиномиальный фон 2-го или 3-го порядка ≤ 5 % от max."""

    rng: np.random.Generator
    max_relative: float = 0.05
    probability: float = 0.3

    def __call__(self, spectrum: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        order = int(self.rng.integers(2, 4))  # 2 или 3
        x = np.linspace(-1.0, 1.0, spectrum.size, dtype=np.float64)
        coeffs = self.rng.uniform(-1.0, 1.0, order + 1)
        baseline = np.polyval(coeffs, x)
        # Нормировка: максимум baseline = max_relative * max(|spectrum|).
        peak = float(np.max(np.abs(spectrum))) or 1.0
        scale = (self.max_relative * peak) / max(float(np.max(np.abs(baseline))), 1e-12)
        return spectrum + baseline * scale


@dataclass
class SpectralCutout:
    """Зануляет случайный участок шириной ``width`` точек."""

    rng: np.random.Generator
    width: int = 50
    probability: float = 0.2

    def __call__(self, spectrum: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        length = spectrum.size
        if self.width >= length:
            return np.zeros_like(spectrum)
        start = int(self.rng.integers(0, length - self.width))
        result = spectrum.copy()
        result[start : start + self.width] = 0.0
        return result


# -----------------------------------------------------------------------------
# Композиция.
# -----------------------------------------------------------------------------


@dataclass
class Compose:
    """Последовательное применение аугментаций с учётом ``probability``."""

    transforms: Sequence[Augmentation]
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng())

    def __call__(self, spectrum: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        out = np.asarray(spectrum, dtype=np.float64)
        for transform in self.transforms:
            if float(self.rng.random()) < transform.probability:
                out = transform(out)
        return out


def default_train_augmentation(rng: np.random.Generator) -> Compose:
    """Стандартный пайплайн обучения по §5.7.

    Порядок (§5.7.3): структурные → амплитудные.
    """
    return Compose(
        transforms=(
            WavenumberShift(rng=rng),
            WavenumberStretch(rng=rng),
            BaselineDistortion(rng=rng),
            SpectralCutout(rng=rng),
            IntensityScale(rng=rng),
            GaussianNoise(rng=rng),
        ),
        rng=rng,
    )
