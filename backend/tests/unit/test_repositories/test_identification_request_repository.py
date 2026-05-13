"""Тесты ``IdentificationRequestRepository``."""

from __future__ import annotations

from sqlalchemy.orm import Session

from app.db.models.compound import Compound
from app.db.models.functional_group import FunctionalGroup
from app.db.models.identification_request import IdentificationRequest
from app.db.models.identification_result import IdentificationResult
from app.db.models.predicted_functional_group import PredictedFunctionalGroup
from app.db.repositories.compound import CompoundRepository
from app.db.repositories.identification_request import IdentificationRequestRepository


def _make_compound(inchi_key: str = "K-REQ-0001") -> Compound:
    return Compound(smiles_canonical="C", inchi="InChI=1S/CH4", inchi_key=inchi_key)


def test_add_anonymous_request(db_session: Session) -> None:
    repo = IdentificationRequestRepository(db_session)
    req = repo.add(
        IdentificationRequest(
            user_id=None,
            input_spectrum_path="/uploads/abc.jdx",
            status="pending",
        )
    )
    assert req.id is not None
    assert req.user_id is None


def test_with_results_loads_results_and_predictions(
    db_session: Session,
    seeded_functional_groups: list[FunctionalGroup],
) -> None:
    compounds = CompoundRepository(db_session)
    requests = IdentificationRequestRepository(db_session)
    compound = compounds.add(_make_compound())

    req = requests.add(IdentificationRequest(input_spectrum_path="/uploads/x.jdx", status="done"))
    db_session.add(
        IdentificationResult(
            request_id=req.id,
            compound_id=compound.id,
            rank=1,
            score=0.95,
            method="faiss",
            compound_name_cached=compound.name,
        )
    )
    db_session.add(
        PredictedFunctionalGroup(
            request_id=req.id,
            functional_group_id=seeded_functional_groups[0].id,
            probability=0.87,
        )
    )
    db_session.commit()

    loaded = requests.with_results(req.id)
    assert loaded is not None
    assert len(loaded.results) == 1
    assert loaded.results[0].score == 0.95
    assert len(loaded.predicted_groups) == 1
    assert loaded.predicted_groups[0].probability == 0.87


def test_cascade_delete_request_removes_dependents(
    db_session: Session,
    seeded_functional_groups: list[FunctionalGroup],
) -> None:
    compounds = CompoundRepository(db_session)
    requests = IdentificationRequestRepository(db_session)
    compound = compounds.add(_make_compound())
    req = requests.add(IdentificationRequest(input_spectrum_path="/uploads/x.jdx", status="done"))
    db_session.add(
        IdentificationResult(
            request_id=req.id, compound_id=compound.id, rank=1, score=0.5, method="cnn"
        )
    )
    db_session.add(
        PredictedFunctionalGroup(
            request_id=req.id,
            functional_group_id=seeded_functional_groups[0].id,
            probability=0.5,
        )
    )
    db_session.commit()

    requests.delete(req)
    db_session.commit()

    assert db_session.query(IdentificationResult).count() == 0
    assert db_session.query(PredictedFunctionalGroup).count() == 0


def test_list_recent_orders_by_timestamp_desc(db_session: Session) -> None:
    repo = IdentificationRequestRepository(db_session)
    repo.add(IdentificationRequest(input_spectrum_path="/a", status="done"))
    repo.add(IdentificationRequest(input_spectrum_path="/b", status="done"))
    repo.add(IdentificationRequest(input_spectrum_path="/c", status="done"))
    db_session.commit()
    recent = repo.list_recent(limit=2)
    assert len(recent) == 2
    # timestamp одинаковый (SQLite CURRENT_TIMESTAMP в пределах теста),
    # но запрос должен вернуть осмысленный порядок — сортировка по id desc как tie-breaker.
    assert recent[0].id >= recent[1].id
