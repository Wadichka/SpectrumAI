"""Тесты ``parse_csv``."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from app.domain.errors import ParsingError
from app.parsing.csv_parser import parse_csv


def test_parses_csv_with_header(fixtures_dir: Path) -> None:
    spectrum = parse_csv(fixtures_dir / "valid_acetone.csv")
    assert spectrum.wavenumbers.dtype == np.float64
    assert spectrum.intensities.dtype == np.float64
    assert spectrum.wavenumbers.size == 10
    assert spectrum.metadata["had_header"] is True
    assert spectrum.metadata["delimiter"] == ","


def test_parses_tab_separated_no_header(fixtures_dir: Path) -> None:
    spectrum = parse_csv(fixtures_dir / "valid_no_header.csv")
    assert spectrum.wavenumbers.size == 10
    assert spectrum.metadata["had_header"] is False
    assert spectrum.metadata["delimiter"] == "\t"


def test_parses_csv_from_bytes() -> None:
    payload = b"400,0.5\n800,0.6\n1200,0.7\n"
    spectrum = parse_csv(payload)
    assert spectrum.wavenumbers.tolist() == [400.0, 800.0, 1200.0]
    assert spectrum.metadata["had_header"] is False


def test_nonnumeric_value_raises_parsing_error(fixtures_dir: Path) -> None:
    with pytest.raises(ParsingError) as exc_info:
        parse_csv(fixtures_dir / "nonnumeric.csv")
    assert exc_info.value.format_name == "CSV"
    # Шапка занимает строку 1, NOT_A_NUMBER на строке 3 (вторая строка данных).
    assert exc_info.value.position == "line 3"


def test_three_columns_raises_parsing_error(fixtures_dir: Path) -> None:
    with pytest.raises(ParsingError) as exc_info:
        parse_csv(fixtures_dir / "three_columns.csv")
    assert "2 колонки" in str(exc_info.value)


def test_empty_input_raises_parsing_error() -> None:
    with pytest.raises(ParsingError):
        parse_csv(b"")


def test_csv_with_only_header_and_no_data_raises_parsing_error() -> None:
    with pytest.raises(ParsingError) as exc_info:
        parse_csv(b"wavenumber,absorbance\n")
    assert "нет данных" in str(exc_info.value)


def test_latin1_encoded_csv_falls_back_correctly() -> None:
    # \xb5 (µ) валиден в latin-1, но не utf-8 — путь fallback должен сработать.
    payload = b"wavenumber,intensity\xb5\n400,0.1\n800,0.2\n"
    spectrum = parse_csv(payload)
    assert spectrum.wavenumbers.size == 2


def test_semicolon_delimiter_is_detected() -> None:
    spectrum = parse_csv(b"400;0.1\n800;0.2\n1200;0.3\n")
    assert spectrum.metadata["delimiter"] == ";"
    assert spectrum.wavenumbers.size == 3


def test_single_column_input_raises_parsing_error() -> None:
    # Sniffer не сможет надёжно вывести разделитель из одной колонки.
    with pytest.raises(ParsingError):
        parse_csv(b"only_one_value\nsecond_value\n")
