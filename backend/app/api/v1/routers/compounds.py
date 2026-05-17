"""Поиск и карточка соединений (UC-02, UC-04)."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query, Response

from app.api.v1.dependencies import get_compound_service
from app.api.v1.schemas import (
    CompoundDetailResponse,
    CompoundSummary,
    PaginatedCompoundsResponse,
)
from app.services.compound import CompoundService

router = APIRouter(prefix="/compounds", tags=["compounds"])


@router.get(
    "",
    response_model=PaginatedCompoundsResponse,
    summary="Поиск соединений в локальной базе (UC-04)",
)
def list_compounds(
    q: str | None = Query(default=None, description="Поисковая строка"),
    functional_groups: list[str] | None = Query(
        default=None,
        description="Список кодов функциональных групп (FG01..FG25)",
    ),
    page: int = Query(default=1, ge=1),
    size: int = Query(default=20, ge=1, le=100),
    service: CompoundService = Depends(get_compound_service),
) -> PaginatedCompoundsResponse:
    paginated = service.search(query=q, functional_groups=functional_groups, page=page, size=size)
    return PaginatedCompoundsResponse(
        data=[
            CompoundSummary(
                id=item.id,
                name=item.name,
                smiles=item.smiles_canonical,
                formula=item.molecular_formula,
                cas_number=item.cas_number,
            )
            for item in paginated.items
        ],
        page=paginated.page,
        size=paginated.size,
        total=paginated.total,
    )


@router.get(
    "/structure.svg",
    summary="SVG структурной формулы по SMILES (без БД, phase 1)",
    response_class=Response,
    responses={200: {"content": {"image/svg+xml": {}}}},
)
def render_structure_by_smiles(
    smiles: str = Query(..., description="SMILES соединения"),
    width: int = Query(default=320, ge=50, le=2000),
    height: int = Query(default=240, ge=50, le=2000),
    service: CompoundService = Depends(get_compound_service),
) -> Response:
    svg = service.render_svg_from_smiles(smiles, width=width, height=height)
    return Response(content=svg, media_type="image/svg+xml")


@router.get(
    "/{compound_id}",
    response_model=CompoundDetailResponse,
    summary="Карточка соединения (UC-02)",
)
def get_compound(
    compound_id: int,
    service: CompoundService = Depends(get_compound_service),
) -> CompoundDetailResponse:
    detail = service.get_detail(compound_id)
    return CompoundDetailResponse(
        id=detail.id,
        name=detail.name,
        smiles=detail.smiles,
        formula=detail.formula,
        cas_number=detail.cas_number,
        iupac_name=detail.iupac_name,
        inchi=detail.inchi,
        inchi_key=detail.inchi_key,
        molecular_weight=detail.molecular_weight,
        functional_groups=list(detail.functional_groups),
        spectra_count=detail.spectra_count,
    )


@router.get(
    "/{compound_id}/structure.svg",
    summary="SVG структурной формулы (CLAUDE.md §7)",
    response_class=Response,
    responses={200: {"content": {"image/svg+xml": {}}}},
)
def render_structure(
    compound_id: int,
    width: int = Query(default=320, ge=50, le=2000),
    height: int = Query(default=240, ge=50, le=2000),
    service: CompoundService = Depends(get_compound_service),
) -> Response:
    svg = service.render_structure_svg(compound_id, width=width, height=height)
    return Response(content=svg, media_type="image/svg+xml")
