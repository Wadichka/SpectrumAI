"""Общие фикстуры тестов парсинга — пути к статическим файлам-фикстурам."""

from __future__ import annotations

from pathlib import Path

import pytest

_FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture()
def fixtures_dir() -> Path:
    """Корень каталога файлов-образцов спектров для тестов."""
    return _FIXTURES_DIR
