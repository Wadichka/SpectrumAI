"""Нестандартные кодировки JCAMP-DX и CSV (глава 11 §11.6.4).

Парсер должен принимать UTF-8 с BOM и старые windows-1251/latin-1 файлы
(распространённые экспорты европейских и русскоязычных лабораторий)
без потери данных, либо корректно сообщать о повреждении кодировки.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.parsing.csv_parser import parse_csv
from app.parsing.jcamp_parser import parse_jcamp

from .conftest import build_csv, build_jcamp


def test_csv_with_utf8_bom_is_parsed_correctly() -> None:
    payload = build_csv(rows=[(400.0, 0.1), (401.0, 0.2)], header=("wn", "i"))
    payload_with_bom = b"\xef\xbb\xbf" + payload

    spectrum = parse_csv(payload_with_bom)

    assert spectrum.wavenumbers.size == 2
    # BOM не должен оказаться в заголовке колонки.
    assert spectrum.metadata["had_header"] is True
    column_names = spectrum.metadata["column_names"] or []
    assert not any("﻿" in str(col) for col in column_names)


def test_csv_in_windows_1251_with_russian_header_parses() -> None:
    payload = build_csv(
        rows=[(400.0, 0.1), (401.0, 0.2)],
        header=("Волновое число", "Интенсивность"),
        encoding="cp1251",
    )

    spectrum = parse_csv(payload)

    assert spectrum.wavenumbers.size == 2
    assert np.allclose(spectrum.wavenumbers, [400.0, 401.0])


def test_jcamp_with_utf8_bom_parses() -> None:
    """JCAMP читается через utf-8 → latin-1 fallback; BOM не должен ломать парсинг."""
    payload = b"\xef\xbb\xbf" + build_jcamp()

    spectrum = parse_jcamp(payload)

    assert spectrum.metadata["format"] == "JCAMP-DX"
    assert spectrum.wavenumbers.size == 6


def test_jcamp_with_latin1_title_parses_via_fallback() -> None:
    """Не-utf8 байты в TITLE не должны разрушить парсинг через latin-1 fallback."""
    payload = build_jcamp(replace={"title": "##TITLE=" + "é-spectrum"})
    # Перекодируем в latin-1 целиком — utf-8-декодирование рухнёт, и сработает fallback.
    payload_latin1 = payload.decode("utf-8").encode("latin-1")

    spectrum = parse_jcamp(payload_latin1)

    assert spectrum.metadata["format"] == "JCAMP-DX"
    assert spectrum.wavenumbers.size == 6


def test_jcamp_with_corrupted_bytes_raises_parsing_error() -> None:
    """Случайные бинарные байты без JCAMP-структуры → ParsingError, не 500."""
    payload = b"\x00\x01\x02\x03binary-garbage\x80\x81\x82"
    with pytest.raises(Exception) as exc_info:
        parse_jcamp(payload)
    # Текущая обёртка может бросить ParsingError или SpectrumValidationError,
    # в любом случае — это DomainError, а не голое исключение из jcamp.
    from app.domain.errors import DomainError

    assert isinstance(exc_info.value, DomainError)
