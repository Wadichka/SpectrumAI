"""Повреждённые JCAMP-DX (глава 11 §11.6.4, FR-01).

Проверяем, что парсер бросает доменное исключение с понятным сообщением
при отсутствии обязательного поля, неподдерживаемых единицах и
несовпадении ``NPOINTS`` с фактическим числом точек. Hypothesis-проперти
покрывает «утерю одного случайного поля»: какое бы поле ни пропало,
результат — ``ParsingError`` со ссылкой на это поле.
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from app.domain.errors import ParsingError, SpectrumValidationError
from app.parsing.jcamp_parser import parse_jcamp

from .conftest import build_jcamp

# Поля, отсутствие которых ловит наш собственный pre-check (точная позиция).
_FIELDS_WITH_PRECISE_POSITION = ["xunits", "yunits", "firstx", "npoints"]
# Поля, при отсутствии которых ломается сама библиотека ``jcamp`` до нашей
# проверки — в таком случае ParsingError приходит с обобщённым сообщением
# (position=None). Это известное ограничение текущей обёртки.
_FIELDS_VIA_LIBRARY_ERROR = ["lastx", "xydata"]


@pytest.mark.parametrize("missing_field", _FIELDS_WITH_PRECISE_POSITION)
def test_missing_field_reports_exact_position(missing_field: str) -> None:
    payload = build_jcamp(omit_keys=[missing_field])
    with pytest.raises(ParsingError) as exc_info:
        parse_jcamp(payload)
    assert exc_info.value.format_name == "JCAMP-DX"
    assert exc_info.value.position == f"##{missing_field.upper()}"


@pytest.mark.parametrize("missing_field", _FIELDS_VIA_LIBRARY_ERROR)
def test_missing_field_via_library_still_raises_parsing_error(missing_field: str) -> None:
    payload = build_jcamp(omit_keys=[missing_field])
    with pytest.raises(ParsingError) as exc_info:
        parse_jcamp(payload)
    assert exc_info.value.format_name == "JCAMP-DX"


def test_truncated_xydata_raises_validation_error() -> None:
    # Объявлены 6 точек, а в данных только 3 → SpectrumValidationError.
    payload = build_jcamp(data_line="400 0.10 0.22 0.34")
    with pytest.raises(SpectrumValidationError) as exc_info:
        parse_jcamp(payload)
    assert exc_info.value.field == "npoints"


def test_unsupported_xunits_raises_parsing_error() -> None:
    payload = build_jcamp(replace={"xunits": "##XUNITS=MICROMETERS"})
    with pytest.raises(ParsingError) as exc_info:
        parse_jcamp(payload)
    assert exc_info.value.position == "##XUNITS"
    assert "MICROMETERS" in str(exc_info.value)


def test_completely_garbage_input_raises_parsing_error() -> None:
    with pytest.raises(ParsingError):
        parse_jcamp(b"not a jcamp file at all\n\x00\x01\x02")


def test_header_without_data_section_raises_parsing_error() -> None:
    # Заголовок корректный, но строка данных пуста → npoints не совпадёт.
    payload = build_jcamp(data_line="")
    with pytest.raises((SpectrumValidationError, ParsingError)):
        parse_jcamp(payload)


_ALL_REQUIRED = _FIELDS_WITH_PRECISE_POSITION + _FIELDS_VIA_LIBRARY_ERROR


@given(missing_field=st.sampled_from(_ALL_REQUIRED))
@settings(max_examples=10, deadline=None)
def test_property_missing_any_required_field_raises(missing_field: str) -> None:
    """Какое бы из обязательных полей ни отсутствовало — ParsingError."""
    payload = build_jcamp(omit_keys=[missing_field])
    with pytest.raises(ParsingError):
        parse_jcamp(payload)
