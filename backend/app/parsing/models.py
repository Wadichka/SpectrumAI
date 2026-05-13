"""Унифицированное внутреннее представление спектра после парсинга.

Соответствует «модели предметной области Спектр» из главы 4 §4.4.1, но
с минимальным контрактом по DEVELOPMENT_PLAN.md: три поля, остальное
располагается в ``metadata``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

from app.domain.errors import SpectrumValidationError


@dataclass(frozen=True, slots=True)
class RawSpectrum:
    """Неизменяемое представление одного спектра.

    Атрибуты:
        wavenumbers: 1D массив волновых чисел (см⁻¹), ``np.float64``.
        intensities: 1D массив интенсивностей, ``np.float64``.
            Природа интенсивности (absorbance/transmittance) фиксируется
            в ``metadata["data_type"]``.
        metadata: произвольные сведения о происхождении спектра
            (формат, единицы измерения, источник, заводские поля JCAMP-DX
            и т. п.). Имена ключей — на английском snake_case.

    Инварианты, проверяемые в ``__post_init__``: массивы одномерные,
    одинаковой длины ≥ 1, тип ``np.float64``. Нарушение → ``SpectrumValidationError``.
    """

    wavenumbers: npt.NDArray[np.float64]
    intensities: npt.NDArray[np.float64]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name, arr in (("wavenumbers", self.wavenumbers), ("intensities", self.intensities)):
            if not isinstance(arr, np.ndarray):
                raise SpectrumValidationError(
                    f"ожидался numpy.ndarray, получено {type(arr).__name__}",
                    field=name,
                )
            if arr.ndim != 1:
                raise SpectrumValidationError(f"ожидался 1D массив, ndim={arr.ndim}", field=name)
            if arr.dtype != np.float64:
                raise SpectrumValidationError(
                    f"ожидался dtype=float64, получено {arr.dtype}", field=name
                )
        if self.wavenumbers.size == 0:
            raise SpectrumValidationError("пустой спектр", field="wavenumbers")
        if self.wavenumbers.size != self.intensities.size:
            raise SpectrumValidationError(
                f"длины массивов различаются: {self.wavenumbers.size} vs {self.intensities.size}",
                field="intensities",
            )
