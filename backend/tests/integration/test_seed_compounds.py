"""Интеграционные тесты сидинга compound/spectrum/functional-group-links."""

from __future__ import annotations

import sys
from pathlib import Path

_BACKEND_ROOT = Path(__file__).resolve().parents[2]
_REPO_ROOT = _BACKEND_ROOT.parent
_ML_ROOT = _REPO_ROOT / "ml"
for path in (_REPO_ROOT, _ML_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from backend.scripts.seed_compounds import seed_compounds_session  # noqa: E402
from sqlalchemy import func, select  # noqa: E402
from sqlalchemy.orm import Session  # noqa: E402

from app.db.models import (  # noqa: E402
    Compound,
    CompoundFunctionalGroup,
    Spectrum,
)


def _make_dataframe() -> pd.DataFrame:
    # 25-длинный multi-hot вектор: одна положительная позиция за раз.
    labels_ethanol = [0] * 25
    labels_ethanol[0] = 1  # alcohol_OH
    labels_acetone = [0] * 25
    labels_acetone[4] = 1  # ketone
    labels_methane = [0] * 25
    labels_methane[20] = 1  # ch3_group
    return pd.DataFrame(
        [
            {
                "id": 0,
                "inchi_key": "LFQSCWFLJHTTHZ-UHFFFAOYSA-N",
                "smiles": "CCO",
                "name": "ethanol",
                "cas": "64-17-5",
                "labels": labels_ethanol,
            },
            {
                "id": 1,
                "inchi_key": "CSCPPACGZOOCGX-UHFFFAOYSA-N",
                "smiles": "CC(=O)C",
                "name": "acetone",
                "cas": "67-64-1",
                "labels": labels_acetone,
            },
            {
                "id": 2,
                "inchi_key": "VNWKTOKETHGBQD-UHFFFAOYSA-N",
                "smiles": "C",
                "name": "methane",
                "cas": "74-82-8",
                "labels": labels_methane,
            },
        ]
    )


@pytest.mark.asyncio
async def test_seed_creates_expected_records(integration_db_session: Session) -> None:
    df = _make_dataframe()
    stats = seed_compounds_session(integration_db_session, df, source="NIST")

    assert stats["created"] == 3
    assert stats["skipped_existing"] == 0
    assert stats["failed"] == 0
    assert stats["group_links"] >= 3  # хотя бы по одной связи на каждое соединение

    n_compounds = integration_db_session.scalar(select(func.count(Compound.id)))
    n_spectra = integration_db_session.scalar(select(func.count(Spectrum.id)))
    n_links = integration_db_session.scalar(select(func.count(CompoundFunctionalGroup.compound_id)))
    assert n_compounds == 3
    assert n_spectra == 3
    assert n_links == stats["group_links"]


@pytest.mark.asyncio
async def test_seed_is_idempotent_on_inchi_key(integration_db_session: Session) -> None:
    df = _make_dataframe()
    first = seed_compounds_session(integration_db_session, df, source="NIST")
    assert first["created"] == 3
    second = seed_compounds_session(integration_db_session, df, source="NIST")
    assert second["created"] == 0
    assert second["skipped_existing"] == 3


@pytest.mark.asyncio
async def test_seed_skips_invalid_smiles(integration_db_session: Session) -> None:
    df = pd.DataFrame(
        [
            {
                "id": 0,
                "inchi_key": "BOGUS",
                "smiles": "XYZ-not-a-smiles",
                "name": "garbage",
                "cas": "0-0-0",
                "labels": [0] * 25,
            }
        ]
    )
    stats = seed_compounds_session(integration_db_session, df, source="NIST")
    assert stats["created"] == 0
    assert stats["failed"] == 1
