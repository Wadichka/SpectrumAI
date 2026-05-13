"""Эвристическое определение формата файла спектра."""

from __future__ import annotations

from typing import Literal

SpectrumFormat = Literal["jcamp", "csv"]

# Маркеры JCAMP-DX — обязательные заголовочные блоки.
_JCAMP_MARKERS: tuple[bytes, ...] = (b"##TITLE=", b"##JCAMP-DX=", b"##JCAMP-DX ")


def detect_format(data: bytes) -> SpectrumFormat:
    """Определяет формат файла по первым 256 байтам.

    Args:
        data: байтовое содержимое файла.

    Returns:
        ``"jcamp"`` если в первых 256 байтах встречается стандартный
        JCAMP-DX-заголовок (``##TITLE=`` или ``##JCAMP-DX=``);
        ``"csv"`` во всех остальных случаях (включая TXT-таблицы).
    """
    sample = data[:256]
    # Толерантность к BOM и пробельным/CRLF-префиксам.
    stripped = sample.lstrip(b"\xef\xbb\xbf").lstrip()
    return "jcamp" if any(stripped.startswith(m) for m in _JCAMP_MARKERS) else "csv"
