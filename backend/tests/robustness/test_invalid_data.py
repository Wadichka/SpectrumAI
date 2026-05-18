"""Граничные числовые данные в спектре (глава 11 §11.6.4).

Проверяем поведение CSV/JCAMP-парсеров и инвариантов ``RawSpectrum``
при NaN, Inf, нулевой длине, дубликатах волновых чисел и
непоследовательности — это типовые «плохие» входы из реальных
экспортов спектрометров.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.domain.errors import ParsingError, SpectrumValidationError
from app.parsing.csv_parser import parse_csv
from app.parsing.models import RawSpectrum

from .conftest import build_csv


def test_csv_nan_intensity_raises_parsing_error() -> None:
    """Нечисловое значение «nan» в данных → ParsingError с указанием строки."""
    payload = build_csv(
        rows=[(400.0, 0.1), (401.0, "nan"), (402.0, 0.3)],
        header=None,
    )
    with pytest.raises(ParsingError) as exc_info:
        parse_csv(payload)
    assert exc_info.value.format_name == "CSV"
    assert exc_info.value.position is not None


def test_csv_with_infinity_passes_parser_but_propagates() -> None:
    """``inf`` распознаётся pandas как число; парсер пропустит его дальше.

    Это документирует текущее поведение: фильтрация Inf — задача
    предобработки/валидатора (см. главу 4 §4.4.1), не парсера CSV.
    """
    payload = build_csv(
        rows=[(400.0, 0.1), (401.0, "inf"), (402.0, 0.3)],
        header=None,
    )
    spectrum = parse_csv(payload)
    assert np.isinf(spectrum.intensities).any()


def test_csv_empty_payload_raises_parsing_error() -> None:
    with pytest.raises(ParsingError) as exc_info:
        parse_csv(b"")
    assert "пустой" in str(exc_info.value).lower()


def test_csv_single_point_is_accepted() -> None:
    """Один валидный ряд проходит парсер (size>=1)."""
    payload = build_csv(rows=[(400.0, 0.5)], header=None)
    spectrum = parse_csv(payload)
    assert spectrum.wavenumbers.size == 1
    assert spectrum.intensities.size == 1


def test_raw_spectrum_rejects_mismatched_lengths() -> None:
    with pytest.raises(SpectrumValidationError) as exc_info:
        RawSpectrum(
            wavenumbers=np.array([400.0, 401.0, 402.0], dtype=np.float64),
            intensities=np.array([0.1, 0.2], dtype=np.float64),
        )
    assert exc_info.value.field == "intensities"


def test_raw_spectrum_rejects_empty_arrays() -> None:
    with pytest.raises(SpectrumValidationError) as exc_info:
        RawSpectrum(
            wavenumbers=np.array([], dtype=np.float64),
            intensities=np.array([], dtype=np.float64),
        )
    assert exc_info.value.field == "wavenumbers"


def test_raw_spectrum_rejects_wrong_dtype() -> None:
    with pytest.raises(SpectrumValidationError) as exc_info:
        RawSpectrum(
            wavenumbers=np.array([400, 401, 402], dtype=np.int64),
            intensities=np.array([0.1, 0.2, 0.3], dtype=np.float64),
        )
    assert "float64" in str(exc_info.value)


def test_csv_duplicate_wavenumbers_pass_parser() -> None:
    """Дубликаты волновых чисел — не ошибка на уровне парсинга.

    Предобработка ниже либо использует ``argsort``, либо ``interp1d``
    (последний при дубликатах поднимает ошибку). Парсер сам по себе их
    не запрещает — документируется здесь.
    """
    payload = build_csv(
        rows=[(400.0, 0.1), (400.0, 0.2), (401.0, 0.3)],
        header=None,
    )
    spectrum = parse_csv(payload)
    assert spectrum.wavenumbers.size == 3
    assert spectrum.wavenumbers[0] == spectrum.wavenumbers[1]


def test_csv_negative_intensities_accepted() -> None:
    """Отрицательные интенсивности (например, после baseline-correction) допустимы."""
    payload = build_csv(rows=[(400.0, -0.05), (401.0, 0.1)], header=None)
    spectrum = parse_csv(payload)
    assert (spectrum.intensities < 0).any()
