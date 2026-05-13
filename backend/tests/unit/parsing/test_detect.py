"""Тесты эвристики ``detect_format``."""

from __future__ import annotations

from app.parsing.detect import detect_format


def test_jcamp_by_title_marker() -> None:
    assert detect_format(b"##TITLE=ethanol\n##JCAMP-DX=4.24\n") == "jcamp"


def test_jcamp_by_dx_version_marker() -> None:
    assert detect_format(b"##JCAMP-DX=5.01\n##TITLE=foo") == "jcamp"


def test_csv_when_no_jcamp_markers() -> None:
    assert detect_format(b"400,0.1\n800,0.2\n") == "csv"


def test_csv_for_header_text() -> None:
    assert detect_format(b"wavenumber,absorbance\n400,0.1") == "csv"


def test_jcamp_tolerates_bom_and_whitespace() -> None:
    assert detect_format(b"\xef\xbb\xbf  \n##TITLE=foo\n") == "jcamp"


def test_csv_for_binary_garbage() -> None:
    assert detect_format(b"\x00\x01\x02not-jcamp") == "csv"
