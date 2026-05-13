"""Парсер JCAMP-DX (стандарт McDonald—Wilks, версии 4.24 и 5.01).

Обёртка над библиотекой ``jcamp``: добавляет жёсткую валидацию обязательных
полей заголовка (см. CLAUDE.md и DEVELOPMENT_PLAN.md этап 2) и преобразует
результат в ``RawSpectrum``.
"""

from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Any

import jcamp
import numpy as np

from app.domain.errors import ParsingError, SpectrumValidationError
from app.parsing.models import RawSpectrum

# Обязательные поля по DEVELOPMENT_PLAN.md этап 2 и главе 4 §4.4.1.
_REQUIRED_FIELDS: tuple[str, ...] = (
    "xunits",
    "yunits",
    "firstx",
    "lastx",
    "npoints",
    "xydata",
)
# Допустимые значения XUNITS — преобразование длин волн в волновые числа
# выполняется на этапе предобработки (см. главу 4 §4.4.2), но не здесь.
_ACCEPTED_XUNITS: frozenset[str] = frozenset({"1/CM", "1/cm", "CM-1", "cm-1"})


def _read_bytes(source: bytes | str | os.PathLike[str]) -> bytes:
    """Извлекает байтовое содержимое из ``bytes`` или пути."""
    if isinstance(source, bytes):
        return source
    return Path(source).read_bytes()


def _decode(content: bytes) -> str:
    """Декодирует bytes в строку с fallback на latin-1 для не-UTF8 файлов."""
    try:
        return content.decode("utf-8")
    except UnicodeDecodeError:
        return content.decode("latin-1")


def parse_jcamp(source: bytes | str | os.PathLike[str]) -> RawSpectrum:
    """Парсит JCAMP-DX из bytes или файла и валидирует обязательные поля.

    Args:
        source: либо ``bytes`` с содержимым файла, либо путь.

    Returns:
        ``RawSpectrum`` с волновыми числами, интенсивностями и заполненной
        ``metadata`` (формат, единицы, источник, factors).

    Raises:
        ParsingError: при отсутствии обязательного поля, недопустимых
            единицах XUNITS или ошибках самого парсера jcamp.
        SpectrumValidationError: при несоответствии длины массива
            заявленному ``NPOINTS`` или нарушении инвариантов ``RawSpectrum``.
    """
    text = _decode(_read_bytes(source))

    try:
        parsed: dict[str, Any] = jcamp.jcamp_read(io.StringIO(text))
    except Exception as exc:  # библиотека jcamp бросает базовый Exception
        raise ParsingError(
            f"не удалось разобрать JCAMP-DX: {exc}",
            format_name="JCAMP-DX",
        ) from exc

    # Проверка обязательных полей.
    for field_name in _REQUIRED_FIELDS:
        if field_name not in parsed:
            raise ParsingError(
                f"отсутствует обязательное поле ##{field_name.upper()}",
                format_name="JCAMP-DX",
                position=f"##{field_name.upper()}",
            )

    # Проверка единиц измерения оси X.
    xunits = str(parsed["xunits"]).strip()
    if xunits not in _ACCEPTED_XUNITS:
        raise ParsingError(
            f"неподдерживаемые единицы XUNITS={xunits!r}; "
            "ожидается 1/CM (см. §4.4.2 — преобразование вне этого этапа)",
            format_name="JCAMP-DX",
            position="##XUNITS",
        )

    x = np.asarray(parsed["x"], dtype=np.float64)
    y = np.asarray(parsed["y"], dtype=np.float64)

    declared_npoints = int(parsed["npoints"])
    if x.size != declared_npoints:
        raise SpectrumValidationError(
            f"число точек {x.size} не совпадает с ##NPOINTS={declared_npoints}",
            field="npoints",
        )

    metadata: dict[str, Any] = {
        "format": "JCAMP-DX",
        "jcamp_dx_version": parsed.get("jcamp-dx"),
        "title": parsed.get("title"),
        "data_type": parsed.get("data type"),
        "xunits": xunits,
        "yunits": parsed.get("yunits"),
        "firstx": parsed.get("firstx"),
        "lastx": parsed.get("lastx"),
        "npoints": declared_npoints,
        "xfactor": parsed.get("xfactor"),
        "yfactor": parsed.get("yfactor"),
        "origin": parsed.get("origin"),
        "owner": parsed.get("owner"),
    }

    return RawSpectrum(wavenumbers=x, intensities=y, metadata=metadata)
