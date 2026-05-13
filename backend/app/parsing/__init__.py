"""Парсеры спектров: JCAMP-DX и CSV/TXT.

Публичный интерфейс модуля — функция :func:`parse_spectrum`, которая
автоматически определяет формат файла и делегирует разбор
соответствующему специализированному парсеру (глава 4 §4.4.1).
"""

from __future__ import annotations

import os
from pathlib import Path

from app.domain.errors import ParsingError, SpectrumValidationError
from app.parsing.csv_parser import parse_csv
from app.parsing.detect import detect_format
from app.parsing.jcamp_parser import parse_jcamp
from app.parsing.models import RawSpectrum

__all__ = [
    "ParsingError",
    "RawSpectrum",
    "SpectrumValidationError",
    "detect_format",
    "parse_csv",
    "parse_jcamp",
    "parse_spectrum",
]


def parse_spectrum(source: bytes | str | os.PathLike[str]) -> RawSpectrum:
    """Универсальный диспетчер: парсит спектр по содержимому или пути.

    Args:
        source: bytes-содержимое файла или путь к файлу (``str``/``Path``).

    Returns:
        ``RawSpectrum`` с заполненной ``metadata["format"]``.

    Raises:
        ParsingError: если формат не распознан или специализированный
            парсер обнаружил ошибку структуры.
        SpectrumValidationError: при семантической ошибке содержимого
            (NaN, расхождение длин, NPOINTS, и т. п.).
    """
    data = source if isinstance(source, bytes) else Path(source).read_bytes()

    fmt = detect_format(data)
    if fmt == "jcamp":
        return parse_jcamp(data)
    return parse_csv(data)
