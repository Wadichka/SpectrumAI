"""История запросов идентификации (UC-08)."""

from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Path, Query, status

from app.api.v1.dependencies import get_history_service
from app.api.v1.schemas import (
    ApiError,
    HistoryEntryResponse,
    IdentificationResponse,
    PaginatedHistoryResponse,
)
from app.services.history import HistoryService

router = APIRouter(tags=["history"])


@router.get(
    "/history",
    response_model=PaginatedHistoryResponse,
    summary="История идентификаций (UC-08)",
)
def list_history(
    page: int = Query(default=1, ge=1),
    size: int = Query(default=20, ge=1, le=100),
    date_from: datetime | None = Query(default=None),
    date_to: datetime | None = Query(default=None),
    status_filter: str | None = Query(default=None, alias="status"),
    service: HistoryService = Depends(get_history_service),
) -> PaginatedHistoryResponse:
    paginated = service.list_recent(
        page=page,
        size=size,
        date_from=date_from,
        date_to=date_to,
        status_filter=status_filter,
    )
    return PaginatedHistoryResponse(
        data=[
            HistoryEntryResponse(
                request_id=entry.request_id,
                timestamp=entry.timestamp,
                status=entry.status,
                processing_time_ms=entry.processing_time_ms,
                input_filename=entry.input_filename,
                top_predicted_groups=list(entry.top_predicted_groups),
            )
            for entry in paginated.items
        ],
        page=paginated.page,
        size=paginated.size,
        total=paginated.total,
    )


@router.get(
    "/history/{request_id}",
    response_model=IdentificationResponse,
    summary="Полный ответ /identify по сохранённой записи истории",
)
def get_history_detail(
    request_id: int = Path(..., ge=1),
    service: HistoryService = Depends(get_history_service),
) -> IdentificationResponse:
    """Восстанавливает полный сериализованный ответ /identify из БД.

    Нужно UI: при открытии записи из истории ResultsPage подтягивает
    свежие predictions/candidates/gradcam именно этой записи, а не
    «последний живой» state из in-memory store. См. §20 фазы 2.
    """
    detail = service.get_by_id(request_id)
    if detail is None or detail.result_payload is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=ApiError(
                code="NOT_FOUND",
                message=f"identification request {request_id} не найден или без сохранённого payload",
            ).model_dump(exclude_none=True),
        )
    payload = dict(detail.result_payload)
    payload["request_id"] = detail.request_id
    return IdentificationResponse.model_validate(payload)
