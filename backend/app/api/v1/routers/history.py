"""История запросов идентификации (UC-08)."""

from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Depends, Query

from app.api.v1.dependencies import get_history_service
from app.api.v1.schemas import HistoryEntryResponse, PaginatedHistoryResponse
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
