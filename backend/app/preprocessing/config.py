"""Конфигурация предобработки спектров и тип выхода ``ProcessedSpectrum``.

Параметры по умолчанию согласованы с CLAUDE.md §7 (сетка 400–4000 см⁻¹,
шаг 1 см⁻¹ → 3601 точка) и с главой 4 §4.4.2 (AsLS как baseline-метод
по умолчанию, Savitzky–Golay для сглаживания, SNV для нормирования).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field, model_validator

NormalizeMethod = Literal["snv", "minmax"]
InterpolationKind = Literal["cubic", "linear"]


class PreprocessConfig(BaseModel):
    """Параметры пайплайна предобработки.

    Неизменяемый Pydantic-объект. Подаётся в :func:`app.preprocessing.preprocess`;
    хранит параметры всех этапов: ресемплинга, AsLS baseline correction,
    Savitzky–Golay-сглаживания и нормирования.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    # Целевая сетка волновых чисел (CLAUDE.md §7).
    target_min: float = Field(default=400.0, gt=0.0)
    target_max: float = Field(default=4000.0, gt=0.0)
    target_step: float = Field(default=1.0, gt=0.0)

    interpolation_kind: InterpolationKind = "cubic"
    out_of_range_fill: float = 0.0

    # AsLS — Asymmetric Least Squares (Eilers & Boelens, 2005), глава 4 §4.4.2.
    asls_lam: float = Field(default=1.0e5, gt=0.0)
    asls_p: float = Field(default=0.01, gt=0.0, lt=1.0)
    asls_niter: int = Field(default=10, ge=1)

    # Savitzky–Golay — глава 4 §4.4.2 (рекомендованное окно 11, polyorder 2).
    savgol_window: int = Field(default=11, ge=3)
    savgol_polyorder: int = Field(default=2, ge=0)

    normalize_method: NormalizeMethod = "snv"

    @model_validator(mode="after")
    def _check_consistency(self) -> PreprocessConfig:
        if self.target_max <= self.target_min:
            raise ValueError("target_max должен быть строго больше target_min")
        if self.savgol_window % 2 == 0:
            raise ValueError("savgol_window должно быть нечётным")
        if self.savgol_polyorder >= self.savgol_window:
            raise ValueError("savgol_polyorder должен быть меньше savgol_window")
        return self

    @property
    def target_length(self) -> int:
        """Число точек целевой сетки при текущих границах и шаге."""
        return round((self.target_max - self.target_min) / self.target_step) + 1


@dataclass(frozen=True, slots=True)
class ProcessedSpectrum:
    """Спектр после предобработки: длина и сетка фиксированы конфигом.

    В отличие от :class:`app.parsing.RawSpectrum`, выход предобработки
    всегда привязан к целевой сетке, и ``metadata["pipeline_steps"]``
    фиксирует журнал применённых преобразований (см. главу 4 §4.4.2 —
    оркестратор журналирует промежуточные представления для отладки).
    """

    wavenumbers: npt.NDArray[np.float64]
    intensities: npt.NDArray[np.float64]
    metadata: dict[str, Any] = field(default_factory=dict)
