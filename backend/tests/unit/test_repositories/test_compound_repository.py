"""Тесты ``CompoundRepository``."""

from __future__ import annotations

import pytest
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.db.models.compound import Compound
from app.db.repositories.compound import CompoundRepository
from app.domain.errors import EntityNotFoundError


def _make_compound(**overrides: object) -> Compound:
    defaults: dict[str, object] = {
        "name": "ethanol",
        "iupac_name": "ethanol",
        "cas_number": "64-17-5",
        "smiles_canonical": "CCO",
        "inchi": "InChI=1S/C2H6O/c1-2-3/h3H,2H2,1H3",
        "inchi_key": "LFQSCWFLJHTTHZ-UHFFFAOYSA-N",
        "molecular_formula": "C2H6O",
        "molecular_weight": 46.07,
    }
    defaults.update(overrides)
    return Compound(**defaults)


def test_add_and_get_by_id(db_session: Session) -> None:
    repo = CompoundRepository(db_session)
    compound = repo.add(_make_compound())
    assert compound.id is not None
    assert repo.get(compound.id) is compound


def test_get_by_inchi_key_returns_match(db_session: Session) -> None:
    repo = CompoundRepository(db_session)
    repo.add(_make_compound())
    found = repo.get_by_inchi_key("LFQSCWFLJHTTHZ-UHFFFAOYSA-N")
    assert found is not None
    assert found.name == "ethanol"


def test_get_by_inchi_key_returns_none_when_absent(db_session: Session) -> None:
    repo = CompoundRepository(db_session)
    assert repo.get_by_inchi_key("UNKNOWN") is None


def test_get_by_cas(db_session: Session) -> None:
    repo = CompoundRepository(db_session)
    repo.add(_make_compound())
    assert repo.get_by_cas("64-17-5") is not None


def test_inchi_key_unique_constraint(db_session: Session) -> None:
    repo = CompoundRepository(db_session)
    repo.add(_make_compound())
    with pytest.raises(IntegrityError):
        # Тот же inchi_key — flush внутри add() обязан кинуть IntegrityError.
        repo.add(_make_compound(name="ethanol-dup", cas_number="0-0-0"))


def test_search_by_name_prefix(db_session: Session) -> None:
    repo = CompoundRepository(db_session)
    repo.add(_make_compound(name="ethanol", inchi_key="K" * 25 + "01"))
    repo.add(_make_compound(name="ethane", inchi_key="K" * 25 + "02", cas_number="74-84-0"))
    repo.add(_make_compound(name="methanol", inchi_key="K" * 25 + "03", cas_number="67-56-1"))
    results = repo.search_by_name_prefix("eth")
    names = [c.name for c in results]
    assert names == sorted(["ethane", "ethanol"])


def test_get_or_raise_raises_entity_not_found(db_session: Session) -> None:
    repo = CompoundRepository(db_session)
    with pytest.raises(EntityNotFoundError):
        repo.get_or_raise(999)


def test_count_and_list(db_session: Session) -> None:
    repo = CompoundRepository(db_session)
    for i in range(3):
        repo.add(_make_compound(name=f"c{i}", inchi_key=f"KEY-{i:020d}-UFAYSAN"))
    assert repo.count() == 3
    assert len(repo.list(limit=2)) == 2
