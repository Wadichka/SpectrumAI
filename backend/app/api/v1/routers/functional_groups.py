"""Эндпоинт списка функциональных групп (Этап 13, для фильтра в /compounds)."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.v1.schemas import FunctionalGroupResponse
from app.db.repositories.functional_group import FunctionalGroupRepository
from app.db.session import get_db

router = APIRouter(tags=["functional-groups"])


@router.get(
    "/functional-groups",
    response_model=list[FunctionalGroupResponse],
    summary="Полный список функциональных групп (25 строк, §5.4.1)",
)
def list_functional_groups(
    session: Session = Depends(get_db),
) -> list[FunctionalGroupResponse]:
    repo = FunctionalGroupRepository(session)
    return [
        FunctionalGroupResponse(
            code=group.code,
            name=group.name,
            description=group.description,
            characteristic_bands=group.characteristic_bands,
        )
        for group in repo.list_all()
    ]
