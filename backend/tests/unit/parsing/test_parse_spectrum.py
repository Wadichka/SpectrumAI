"""Тесты диспетчера ``parse_spectrum`` и инвариантов ``RawSpectrum``."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from app.domain.errors import ParsingError, SpectrumValidationError
from app.parsing import RawSpectrum, parse_spectrum


def test_dispatches_jcamp_by_path_str(fixtures_dir: Path) -> None:
    spectrum = parse_spectrum(str(fixtures_dir / "valid_ethanol.jdx"))
    assert spectrum.metadata["format"] == "JCAMP-DX"
    assert spectrum.wavenumbers.size == 10


def test_dispatches_csv_by_path(fixtures_dir: Path) -> None:
    spectrum = parse_spectrum(fixtures_dir / "valid_acetone.csv")
    assert spectrum.metadata["format"] == "CSV"
    assert spectrum.wavenumbers.size == 10


def test_dispatches_jcamp_by_bytes(fixtures_dir: Path) -> None:
    payload = (fixtures_dir / "valid_ethanol.jdx").read_bytes()
    assert parse_spectrum(payload).metadata["format"] == "JCAMP-DX"


def test_dispatches_csv_by_bytes() -> None:
    spectrum = parse_spectrum(b"400,0.1\n800,0.2\n")
    assert spectrum.metadata["format"] == "CSV"


def test_unknown_binary_garbage_routed_to_csv_and_fails() -> None:
    # Бинарный мусор не похож на JCAMP, попадает в CSV-парсер, тот падает.
    with pytest.raises(ParsingError):
        parse_spectrum(b"\x00\x01\x02\x03")


def test_raw_spectrum_rejects_wrong_dtype() -> None:
    with pytest.raises(SpectrumValidationError) as exc_info:
        RawSpectrum(
            wavenumbers=np.array([1, 2, 3], dtype=np.int64),
            intensities=np.array([0.1, 0.2, 0.3], dtype=np.float64),
        )
    assert exc_info.value.field == "wavenumbers"


def test_raw_spectrum_rejects_length_mismatch() -> None:
    with pytest.raises(SpectrumValidationError) as exc_info:
        RawSpectrum(
            wavenumbers=np.array([1.0, 2.0], dtype=np.float64),
            intensities=np.array([0.1, 0.2, 0.3], dtype=np.float64),
        )
    assert exc_info.value.field == "intensities"


def test_raw_spectrum_rejects_empty() -> None:
    with pytest.raises(SpectrumValidationError):
        RawSpectrum(
            wavenumbers=np.array([], dtype=np.float64),
            intensities=np.array([], dtype=np.float64),
        )


def test_raw_spectrum_rejects_2d_array() -> None:
    with pytest.raises(SpectrumValidationError):
        RawSpectrum(
            wavenumbers=np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64),
            intensities=np.array([0.1, 0.2], dtype=np.float64),
        )
