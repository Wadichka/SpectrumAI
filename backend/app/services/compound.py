"""Сервис поиска и карточки соединений (§4.4.7, UC-02, UC-04)."""

from __future__ import annotations

from dataclasses import dataclass

from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
from sqlalchemy import func, select
from sqlalchemy.orm import Session, selectinload

from app.db.models.compound import Compound
from app.db.models.compound_functional_group import CompoundFunctionalGroup
from app.db.models.functional_group import FunctionalGroup
from app.db.repositories.compound import CompoundRepository
from app.domain.errors import EntityNotFoundError


@dataclass(frozen=True, slots=True)
class CompoundDetail:
    """Полная карточка соединения (для UC-02)."""

    id: int
    name: str | None
    iupac_name: str | None
    smiles: str
    cas_number: str | None
    formula: str | None
    inchi: str | None
    inchi_key: str | None
    molecular_weight: float | None
    functional_groups: tuple[str, ...]
    spectra_count: int


@dataclass(frozen=True, slots=True)
class PaginatedCompounds:
    items: list[Compound]
    page: int
    size: int
    total: int


class CompoundService:
    """Поиск, карточка и SVG-отрисовка соединений."""

    def __init__(self, session: Session) -> None:
        self._session = session
        self._repo = CompoundRepository(session)

    def search(
        self,
        *,
        query: str | None,
        functional_groups: list[str] | None,
        page: int,
        size: int,
    ) -> PaginatedCompounds:
        base = select(Compound)
        count_stmt = select(func.count()).select_from(Compound)
        if query:
            like = f"%{query}%"
            condition = (
                Compound.name.ilike(like)
                | (Compound.smiles_canonical == query)
                | (Compound.cas_number == query)
                | Compound.inchi_key.ilike(like)
            )
            base = base.where(condition)
            count_stmt = count_stmt.where(condition)
        if functional_groups:
            cfg = CompoundFunctionalGroup
            join_subquery = (
                select(cfg.compound_id)
                .join(FunctionalGroup, FunctionalGroup.id == cfg.functional_group_id)
                .where(FunctionalGroup.code.in_(functional_groups))
                .group_by(cfg.compound_id)
                .having(func.count(cfg.compound_id) == len(functional_groups))
                .subquery()
            )
            base = base.where(Compound.id.in_(select(join_subquery)))
            count_stmt = count_stmt.where(Compound.id.in_(select(join_subquery)))
        offset = max(0, (page - 1) * size)
        stmt = (
            base.order_by(Compound.name.asc().nulls_last(), Compound.id.asc())
            .offset(offset)
            .limit(size)
        )
        items = list(self._session.scalars(stmt).all())
        total = int(self._session.scalar(count_stmt) or 0)
        return PaginatedCompounds(items=items, page=page, size=size, total=total)

    def get_detail(self, compound_id: int) -> CompoundDetail:
        stmt = (
            select(Compound)
            .where(Compound.id == compound_id)
            .options(
                selectinload(Compound.functional_group_links).selectinload(
                    CompoundFunctionalGroup.functional_group
                ),
                selectinload(Compound.spectra),
            )
        )
        compound = self._session.scalars(stmt).one_or_none()
        if compound is None:
            raise EntityNotFoundError("Compound", compound_id)
        codes = tuple(
            sorted(link.functional_group.code for link in compound.functional_group_links)
        )
        return CompoundDetail(
            id=compound.id,
            name=compound.name,
            iupac_name=compound.iupac_name,
            smiles=compound.smiles_canonical,
            cas_number=compound.cas_number,
            formula=compound.molecular_formula,
            inchi=compound.inchi,
            inchi_key=compound.inchi_key,
            molecular_weight=compound.molecular_weight,
            functional_groups=codes,
            spectra_count=len(compound.spectra),
        )

    def render_structure_svg(
        self,
        compound_id: int,
        *,
        width: int = 320,
        height: int = 240,
    ) -> str:
        compound = self._repo.get_or_raise(compound_id)
        mol = Chem.MolFromSmiles(compound.smiles_canonical)
        if mol is None:
            raise EntityNotFoundError("Compound.structure", compound_id)
        drawer = rdMolDraw2D.MolDraw2DSVG(int(width), int(height))
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        svg = drawer.GetDrawingText()
        # MolDraw2DSVG возвращает SVG с XML-декларацией <?xml ...?>; для inline
        # вставки во фронте удобнее обрезать её.
        if svg.startswith("<?xml"):
            svg = svg.split("?>", 1)[1].lstrip()
        return str(svg)


__all__ = ["CompoundDetail", "CompoundService", "PaginatedCompounds"]
