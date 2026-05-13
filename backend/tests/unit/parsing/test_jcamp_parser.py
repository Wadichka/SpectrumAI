"""Тесты ``parse_jcamp``."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from app.domain.errors import ParsingError, SpectrumValidationError
from app.parsing.jcamp_parser import parse_jcamp


def test_parses_valid_jcamp_from_path(fixtures_dir: Path) -> None:
    spectrum = parse_jcamp(fixtures_dir / "valid_ethanol.jdx")
    assert spectrum.wavenumbers.dtype == np.float64
    assert spectrum.intensities.dtype == np.float64
    assert spectrum.wavenumbers.size == 10
    assert spectrum.metadata["format"] == "JCAMP-DX"
    assert spectrum.metadata["xunits"] == "1/CM"
    assert spectrum.metadata["npoints"] == 10


def test_parses_valid_jcamp_from_bytes(fixtures_dir: Path) -> None:
    payload = (fixtures_dir / "valid_ethanol.jdx").read_bytes()
    spectrum = parse_jcamp(payload)
    assert spectrum.wavenumbers.size == 10


def test_missing_xunits_raises_parsing_error(fixtures_dir: Path) -> None:
    with pytest.raises(ParsingError) as exc_info:
        parse_jcamp(fixtures_dir / "missing_xunits.jdx")
    assert exc_info.value.format_name == "JCAMP-DX"
    assert exc_info.value.position == "##XUNITS"
    assert "XUNITS" in str(exc_info.value)


def test_wrong_xunits_raises_parsing_error(fixtures_dir: Path) -> None:
    with pytest.raises(ParsingError) as exc_info:
        parse_jcamp(fixtures_dir / "wrong_xunits.jdx")
    assert exc_info.value.position == "##XUNITS"
    assert "MICROMETERS" in str(exc_info.value)


def test_npoints_mismatch_raises_validation_error() -> None:
    # NPOINTS объявлен 10, фактически 4 точки в данных.
    content = b"""##TITLE=mismatched
##JCAMP-DX=4.24
##DATA TYPE=INFRARED SPECTRUM
##XUNITS=1/CM
##YUNITS=ABSORBANCE
##FIRSTX=400
##LASTX=4000
##NPOINTS=10
##XFACTOR=1
##YFACTOR=1
##XYDATA=(X++(Y..Y))
400 0.1 0.2 0.3 0.4
##END=
"""
    with pytest.raises(SpectrumValidationError) as exc_info:
        parse_jcamp(content)
    assert exc_info.value.field == "npoints"


def test_garbage_input_raises_parsing_error() -> None:
    # Без ##JCAMP-DX заголовка jcamp_read вернёт пустой dict без полей.
    with pytest.raises(ParsingError):
        parse_jcamp(b"completely not a jcamp file\n")
