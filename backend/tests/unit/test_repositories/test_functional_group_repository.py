"""Тесты ``FunctionalGroupRepository``."""

from __future__ import annotations

from sqlalchemy.orm import Session

from app.db.models.functional_group import FunctionalGroup
from app.db.repositories.functional_group import FunctionalGroupRepository


def test_get_by_code(
    db_session: Session,
    seeded_functional_groups: list[FunctionalGroup],
) -> None:
    repo = FunctionalGroupRepository(db_session)
    fg = repo.get_by_code("FG01")
    assert fg is not None
    assert fg.name == "alcohol_OH"


def test_list_all_returns_25_in_order(
    db_session: Session,
    seeded_functional_groups: list[FunctionalGroup],
) -> None:
    repo = FunctionalGroupRepository(db_session)
    groups = repo.list_all()
    assert len(groups) == 25
    codes = [g.code for g in groups]
    assert codes == sorted(codes)
    assert codes[0] == "FG01"
    assert codes[-1] == "FG25"


def test_get_by_code_unknown_returns_none(
    db_session: Session,
    seeded_functional_groups: list[FunctionalGroup],
) -> None:
    repo = FunctionalGroupRepository(db_session)
    assert repo.get_by_code("FG99") is None
