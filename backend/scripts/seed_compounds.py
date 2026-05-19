"""Сидинг таблиц ``compound`` / ``spectrum`` / ``compound_functional_group``
из предзащитного parquet'а (фаза 2, этап 20).

Читает ``ml/data/processed/predefense_labeled.parquet`` (результат
``apply_labeling.py``), создаёт по одной записи ``Compound``, ``Spectrum``
и пачку ``CompoundFunctionalGroup`` для каждой строки. Идемпотентен:
повторный запуск не создаёт дубликатов (uniqueness по
``compound.inchi_key`` обеспечивает БД).

Запуск:
    python backend/scripts/seed_compounds.py \\
        [--parquet ml/data/processed/predefense_labeled.parquet] \\
        [--source NIST]
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd
import structlog
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.inchi import InchiToInchiKey, MolToInchi
from sqlalchemy import select
from sqlalchemy.orm import Session

_REPO_ROOT = Path(__file__).resolve().parents[2]
_BACKEND_ROOT = _REPO_ROOT / "backend"
_ML_ROOT = _REPO_ROOT / "ml"
for path in (_BACKEND_ROOT, _ML_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from app.db.models import (  # noqa: E402
    Compound,
    CompoundFunctionalGroup,
    FunctionalGroup,
    Spectrum,
)
from app.db.session import SessionLocal  # noqa: E402
from pipelines.labeling import GROUP_NAMES  # noqa: E402

log = structlog.get_logger(__name__)

_TARGET_MIN = 400.0
_TARGET_MAX = 4000.0
_TARGET_LENGTH = 3601


def _row_to_compound(row: pd.Series) -> Compound:
    smiles = str(row["smiles"])
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"invalid SMILES: {smiles}")
    inchi = MolToInchi(mol)
    inchi_key = str(row.get("inchi_key") or InchiToInchiKey(inchi))
    canonical_smiles = Chem.MolToSmiles(mol)
    return Compound(
        name=row.get("name"),
        cas_number=row.get("cas"),
        smiles_canonical=canonical_smiles,
        inchi=inchi,
        inchi_key=inchi_key,
        molecular_formula=Chem.rdMolDescriptors.CalcMolFormula(mol),
        molecular_weight=float(Descriptors.MolWt(mol)),
    )


def _row_to_spectrum(compound_id: int, cas: str | None, source: str) -> Spectrum:
    file_path = f"predefense://{cas or 'unknown'}.jdx"
    return Spectrum(
        compound_id=compound_id,
        source=source,
        phase=None,
        technique="FTIR",
        wavenumber_min=_TARGET_MIN,
        wavenumber_max=_TARGET_MAX,
        n_points=_TARGET_LENGTH,
        file_path=file_path,
        quality_score=None,
    )


def _row_to_group_links(
    compound_id: int,
    labels: Iterable[int],
    group_id_by_name: dict[str, int],
) -> list[CompoundFunctionalGroup]:
    links: list[CompoundFunctionalGroup] = []
    for idx, present in enumerate(labels):
        if int(present) == 0:
            continue
        name = GROUP_NAMES[idx]
        group_id = group_id_by_name.get(name)
        if group_id is None:
            continue
        links.append(
            CompoundFunctionalGroup(
                compound_id=compound_id,
                functional_group_id=group_id,
                count=1,
            )
        )
    return links


def _load_group_index(session: Session) -> dict[str, int]:
    rows = session.execute(select(FunctionalGroup.id, FunctionalGroup.name)).all()
    return {name: gid for gid, name in rows}


def seed_compounds_session(
    session: Session,
    df: pd.DataFrame,
    *,
    source: str = "NIST",
) -> dict[str, int]:
    """Идемпотентный сидинг в указанную SQLAlchemy-сессию.

    Возвращает агрегаты для логирования/тестов: created, skipped_existing,
    failed.
    """
    group_id_by_name = _load_group_index(session)
    if not group_id_by_name:
        log.warning("no_functional_groups", hint="run alembic upgrade head first")

    counters = {"created": 0, "skipped_existing": 0, "failed": 0, "group_links": 0}

    for _, row in df.iterrows():
        try:
            compound = _row_to_compound(row)
        except (ValueError, KeyError) as exc:
            log.warning("seed_skip_invalid", smiles=row.get("smiles"), reason=str(exc))
            counters["failed"] += 1
            continue

        existing = session.execute(
            select(Compound).where(Compound.inchi_key == compound.inchi_key)
        ).scalar_one_or_none()
        if existing is not None:
            counters["skipped_existing"] += 1
            continue

        session.add(compound)
        session.flush()  # получаем compound.id

        spectrum = _row_to_spectrum(compound.id, row.get("cas"), source)
        session.add(spectrum)

        labels = row.get("labels") or []
        links = _row_to_group_links(compound.id, labels, group_id_by_name)
        counters["group_links"] += len(links)
        if links:
            session.add_all(links)

        counters["created"] += 1

    session.commit()
    log.info("seed_compounds_done", **counters)
    return counters


def seed_compounds_from_parquet(
    parquet_path: Path,
    *,
    source: str = "NIST",
    session_factory: Any | None = None,
) -> dict[str, int]:
    df = pd.read_parquet(parquet_path)
    log.info("seed_compounds_start", rows=len(df), source=source, parquet=str(parquet_path))
    factory = session_factory or SessionLocal
    session: Session = factory()
    try:
        return seed_compounds_session(session, df, source=source)
    finally:
        session.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--parquet",
        type=Path,
        default=_REPO_ROOT / "ml" / "data" / "processed" / "predefense_labeled.parquet",
    )
    parser.add_argument("--source", default="NIST", help="значение Spectrum.source")
    args = parser.parse_args()
    seed_compounds_from_parquet(args.parquet, source=args.source)


if __name__ == "__main__":
    main()
