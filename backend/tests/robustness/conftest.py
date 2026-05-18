"""Общие фикстуры для robustness-сюиты (Этап 16, глава 11 §11.6.4).

Здесь собраны генераторы валидных JCAMP-DX/CSV, которые удобно затем
повредить точечно (удалить поле, заменить значение, поменять кодировку).
Тесты в этой папке проверяют **текущее** поведение системы на нестандартных
входах — не желаемое.
"""

from __future__ import annotations

from collections.abc import Iterable

import pytest

_JCAMP_REQUIRED_LINES: tuple[tuple[str, str], ...] = (
    ("title", "##TITLE=robustness-sample"),
    ("jcamp-dx", "##JCAMP-DX=4.24"),
    ("data type", "##DATA TYPE=INFRARED SPECTRUM"),
    ("xunits", "##XUNITS=1/CM"),
    ("yunits", "##YUNITS=ABSORBANCE"),
    ("firstx", "##FIRSTX=400"),
    ("lastx", "##LASTX=405"),
    ("npoints", "##NPOINTS=6"),
    ("xfactor", "##XFACTOR=1"),
    ("yfactor", "##YFACTOR=1"),
    ("xydata", "##XYDATA=(X++(Y..Y))"),
)
_JCAMP_DATA_LINE = "400 0.10 0.22 0.34 0.41 0.28 0.13"
_JCAMP_END_LINE = "##END="


def build_jcamp(
    *,
    omit_keys: Iterable[str] = (),
    replace: dict[str, str] | None = None,
    data_line: str | None = None,
) -> bytes:
    """Возвращает JCAMP-DX в виде ``bytes`` с возможной точечной поломкой.

    Args:
        omit_keys: перечень ключей (как в ``_JCAMP_REQUIRED_LINES``), строки
            с которыми будут вырезаны из заголовка (моделирует утерю поля).
        replace: словарь key → новая строка (целиком, вместе с ``##KEY=``).
            Позволяет подменить значение конкретного поля.
        data_line: переопределяет строку с данными после ``##XYDATA=``.
            Если ``None`` — берётся дефолтный 6-точечный ряд.
    """
    omit_set = set(omit_keys)
    replace_map = replace or {}
    lines = [
        replace_map.get(key, default_line)
        for key, default_line in _JCAMP_REQUIRED_LINES
        if key not in omit_set
    ]
    lines.append(data_line if data_line is not None else _JCAMP_DATA_LINE)
    lines.append(_JCAMP_END_LINE)
    return ("\n".join(lines) + "\n").encode("utf-8")


@pytest.fixture()
def jcamp_factory():  # type: ignore[no-untyped-def]
    """Фабрика build_jcamp как pytest-фикстура для удобства параметризации."""
    return build_jcamp


def build_csv(
    rows: Iterable[tuple[object, object]],
    *,
    delimiter: str = ",",
    header: tuple[str, str] | None = ("wavenumber", "intensity"),
    encoding: str = "utf-8",
) -> bytes:
    """Собирает CSV из строк и кодирует в указанную кодировку.

    Поддерживает любые строковые формы числовых значений: ``"nan"``, ``"inf"``,
    ``"abc"`` — для проверки поведения парсера на нечисловых данных.
    """
    parts: list[str] = []
    if header is not None:
        parts.append(delimiter.join(header))
    for wn, intensity in rows:
        parts.append(f"{wn}{delimiter}{intensity}")
    return ("\n".join(parts) + "\n").encode(encoding)
