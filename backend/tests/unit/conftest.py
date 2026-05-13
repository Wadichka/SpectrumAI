"""Общие фикстуры для unit-тестов слоя данных.

SQLite in-memory используется только для unit-тестов без PostgreSQL-специфики
(CLAUDE.md §3.6). Все наши модели портируемы: ``LargeBinary``, ``DateTime``,
``Text`` и индексы B-tree работают одинаково в обеих СУБД.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from app.db.models import (
    Base,
    FunctionalGroup,
)

# Полный список функциональных групп — повторяет seed из миграции 0001_initial.
_FG_SEED = [
    ("FG01", "alcohol_OH", "[OX2H][CX4]"),
    ("FG02", "phenol_OH", "[OX2H][c]"),
    ("FG03", "carbonyl", "[CX3]=[OX1]"),
    ("FG04", "aldehyde", "[CX3H1](=O)[#6]"),
    ("FG05", "ketone", "[#6][CX3](=O)[#6]"),
    ("FG06", "carboxylic_acid", "[CX3](=O)[OX2H]"),
    ("FG07", "ester", "[#6][CX3](=O)[OX2][#6]"),
    ("FG08", "amide_primary", "[CX3](=O)[NX3H2]"),
    ("FG09", "amide_secondary", "[CX3](=O)[NX3H1]"),
    ("FG10", "amide_tertiary", "[CX3](=O)[NX3H0]"),
    ("FG11", "amine_primary", "[NX3;H2;!$(NC=O)]"),
    ("FG12", "amine_secondary", "[NX3;H1;!$(NC=O)]"),
    ("FG13", "amine_tertiary", "[NX3;H0;!$(NC=O)]"),
    ("FG14", "nitrile", "[CX2]#[NX1]"),
    ("FG15", "nitro", "[$([NX3](=O)=O),$([NX3+](=O)[O-])]"),
    ("FG16", "ether", "[OD2]([#6])[#6]"),
    ("FG17", "alkene", "[CX3]=[CX3]"),
    ("FG18", "alkyne", "[CX2]#[CX2]"),
    ("FG19", "aromatic_ring", "c1ccccc1"),
    ("FG20", "ch2_group", "[CH2]"),
    ("FG21", "ch3_group", "[CH3]"),
    ("FG22", "c_f_bond", "[CX4]F"),
    ("FG23", "c_cl_bond", "[CX4]Cl"),
    ("FG24", "sulfoxide_sulfone", "[#6][SX3](=O)[#6],[#6][SX4](=O)(=O)[#6]"),
    ("FG25", "thiol_thioether", "[SX2H],[SX2]([#6])[#6]"),
]


@pytest.fixture()
def db_session() -> Iterator[Session]:
    """SQLite in-memory сессия c созданной схемой и включёнными внешними ключами."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        future=True,
    )

    # SQLite по умолчанию игнорирует ON DELETE CASCADE / SET NULL; включаем явно.
    @event.listens_for(engine, "connect")
    def _enable_foreign_keys(dbapi_connection: object, _: object) -> None:
        cursor = dbapi_connection.cursor()  # type: ignore[attr-defined]
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
    session = session_factory()
    try:
        yield session
    finally:
        session.close()
        engine.dispose()


@pytest.fixture()
def seeded_functional_groups(db_session: Session) -> list[FunctionalGroup]:
    """Заполняет справочник функциональных групп 25 строками (как делает миграция).

    Используется тестами, которым нужны существующие FG (например, тесты
    PredictedFunctionalGroup или CompoundFunctionalGroup).
    """
    groups = [
        FunctionalGroup(code=code, name=name, smarts_pattern=pattern)
        for code, name, pattern in _FG_SEED
    ]
    db_session.add_all(groups)
    db_session.flush()
    return groups


@pytest.fixture()
def sqlite_engine_with_fk() -> Engine:
    """Альтернативная фикстура: голый engine с включёнными внешними ключами."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        future=True,
    )

    @event.listens_for(engine, "connect")
    def _enable_foreign_keys(dbapi_connection: object, _: object) -> None:
        cursor = dbapi_connection.cursor()  # type: ignore[attr-defined]
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    Base.metadata.create_all(engine)
    return engine
