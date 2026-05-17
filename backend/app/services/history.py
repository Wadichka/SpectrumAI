"""Сервис истории идентификаций (UC-08, §4.4.7)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import func, select
from sqlalchemy.orm import Session, selectinload

from app.db.models.identification_request import IdentificationRequest
from app.db.models.predicted_functional_group import PredictedFunctionalGroup


@dataclass(frozen=True, slots=True)
class HistoryEntry:
    request_id: int
    timestamp: datetime
    status: str
    processing_time_ms: int | None
    input_filename: str | None
    top_predicted_groups: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PaginatedHistory:
    items: list[HistoryEntry]
    page: int
    size: int
    total: int


class HistoryService:
    """Постраничный листинг истории запросов."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def list_recent(
        self,
        *,
        page: int,
        size: int,
        date_from: datetime | None = None,
        date_to: datetime | None = None,
        status_filter: str | None = None,
    ) -> PaginatedHistory:
        base = select(IdentificationRequest).options(
            selectinload(IdentificationRequest.predicted_groups).selectinload(
                PredictedFunctionalGroup.functional_group
            )
        )
        count_stmt = select(func.count()).select_from(IdentificationRequest)
        if date_from is not None:
            base = base.where(IdentificationRequest.timestamp >= date_from)
            count_stmt = count_stmt.where(IdentificationRequest.timestamp >= date_from)
        if date_to is not None:
            base = base.where(IdentificationRequest.timestamp <= date_to)
            count_stmt = count_stmt.where(IdentificationRequest.timestamp <= date_to)
        if status_filter is not None:
            base = base.where(IdentificationRequest.status == status_filter)
            count_stmt = count_stmt.where(IdentificationRequest.status == status_filter)

        offset = max(0, (page - 1) * size)
        stmt = base.order_by(IdentificationRequest.timestamp.desc()).offset(offset).limit(size)
        requests = list(self._session.scalars(stmt).all())
        total = int(self._session.scalar(count_stmt) or 0)

        items = [
            HistoryEntry(
                request_id=req.id,
                timestamp=req.timestamp,
                status=req.status,
                processing_time_ms=req.processing_time_ms,
                input_filename=req.input_spectrum_path,
                top_predicted_groups=tuple(
                    pg.functional_group.code
                    for pg in sorted(req.predicted_groups, key=lambda p: -p.probability)[:5]
                ),
            )
            for req in requests
        ]
        return PaginatedHistory(items=items, page=page, size=size, total=total)


__all__ = ["HistoryEntry", "HistoryService", "PaginatedHistory"]
