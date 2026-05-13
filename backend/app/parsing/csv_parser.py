"""Парсер CSV/TXT для табличных представлений спектра.

Поддерживается автоопределение разделителя (запятая, точка с запятой,
табуляция, пробел) и наличия строки заголовка. Формат — две колонки:
волновое число и интенсивность.
"""

from __future__ import annotations

import csv
import io
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from app.domain.errors import ParsingError, SpectrumValidationError
from app.parsing.models import RawSpectrum

# Кандидаты-разделители для авто-определения. Пробел оставлен последним,
# чтобы Sniffer предпочёл более «строгие» варианты.
_DELIMITERS: str = ",;\t "
# Размер выборки для Sniffer и эвристики шапки.
_SNIFF_SAMPLE_BYTES: int = 4096


def _read_text(source: bytes | str | os.PathLike[str]) -> str:
    """Декодирует bytes/файл в строку (utf-8 → latin-1 fallback)."""
    raw = source if isinstance(source, bytes) else Path(source).read_bytes()
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        return raw.decode("latin-1")


def _detect_delimiter(sample: str) -> str:
    """Эвристически определяет разделитель в первых строках содержимого."""
    sniffer = csv.Sniffer()
    try:
        dialect = sniffer.sniff(sample, delimiters=_DELIMITERS)
    except csv.Error as exc:
        # Fallback: если sniff не сработал, пробуем «whitespace-режим»
        # как для TXT — пандас сам обработает множественные пробелы.
        first_line = sample.splitlines()[0] if sample else ""
        if len(first_line.split()) == 2:
            return r"\s+"
        raise ParsingError(
            f"не удалось определить разделитель CSV: {exc}",
            format_name="CSV",
            position="line 1",
        ) from exc
    return dialect.delimiter


def _looks_like_header(sample: str, delimiter: str) -> bool:
    """Решает, является ли первая строка заголовком (не числовой)."""
    first_line = sample.splitlines()[0] if sample else ""
    if not first_line.strip():
        return False
    tokens = first_line.split() if delimiter == r"\s+" else first_line.split(delimiter)
    if not tokens:
        return False
    # Если хотя бы один токен не парсится как число — это шапка.
    for token in tokens:
        try:
            float(token.replace(",", ".").strip())
        except ValueError:
            return True
    return False


def parse_csv(source: bytes | str | os.PathLike[str]) -> RawSpectrum:
    """Парсит CSV/TXT с двумя числовыми колонками.

    Args:
        source: bytes-содержимое или путь к файлу.

    Returns:
        ``RawSpectrum`` с массивами wavenumbers/intensities и метаданными
        формата (разделитель, наличие шапки, имена колонок).

    Raises:
        ParsingError: при невозможности определить разделитель, числе
            колонок, отличном от двух, нечисловых значениях.
    """
    text = _read_text(source)
    if not text.strip():
        raise ParsingError("пустой файл", format_name="CSV", position="line 1")

    sample = text[:_SNIFF_SAMPLE_BYTES]
    delimiter = _detect_delimiter(sample)
    has_header = _looks_like_header(sample, delimiter)

    read_kwargs: dict[str, Any] = {
        "header": 0 if has_header else None,
        "engine": "python",
        "skip_blank_lines": True,
    }
    if delimiter == r"\s+":
        read_kwargs["sep"] = r"\s+"
    else:
        read_kwargs["sep"] = delimiter

    try:
        frame: pd.DataFrame = pd.read_csv(io.StringIO(text), **read_kwargs)
    except (pd.errors.ParserError, ValueError) as exc:
        raise ParsingError(
            f"ошибка чтения CSV: {exc}",
            format_name="CSV",
        ) from exc

    if frame.shape[1] != 2:
        raise ParsingError(
            f"ожидалось ровно 2 колонки, получено {frame.shape[1]}",
            format_name="CSV",
            position="header" if has_header else "line 1",
        )
    if frame.shape[0] == 0:
        raise ParsingError("нет данных после строки заголовка", format_name="CSV")

    # Принудительное приведение к float; нечисловые → NaN.
    numeric = frame.apply(pd.to_numeric, errors="coerce")
    bad_mask = numeric.isna().any(axis=1)
    if bool(bad_mask.any()):
        first_bad_row = int(bad_mask.idxmax())
        line_number = first_bad_row + (2 if has_header else 1)
        raise ParsingError(
            "нечисловое значение в данных",
            format_name="CSV",
            position=f"line {line_number}",
        )

    x = numeric.iloc[:, 0].to_numpy(dtype=np.float64)
    y = numeric.iloc[:, 1].to_numpy(dtype=np.float64)

    metadata: dict[str, Any] = {
        "format": "CSV",
        "delimiter": delimiter,
        "had_header": has_header,
        "column_names": list(frame.columns) if has_header else None,
    }

    try:
        return RawSpectrum(wavenumbers=x, intensities=y, metadata=metadata)
    except SpectrumValidationError:
        raise
