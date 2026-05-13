"""Репозиторий сущности ``IdentificationRequest``."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import selectinload

from app.db.models.identification_request import IdentificationRequest
from app.db.repositories.base import BaseRepository


class IdentificationRequestRepository(BaseRepository[IdentificationRequest]):
    """Доступ к истории пользовательских запросов идентификации (§5.10.3)."""

    model = IdentificationRequest

    def list_recent(
        self,
        *,
        user_id: int | None = None,
        limit: int = 50,
    ) -> list[IdentificationRequest]:
        """Последние запросы, опционально фильтруя по пользователю."""
        stmt = (
            select(IdentificationRequest)
            .order_by(IdentificationRequest.timestamp.desc())
            .limit(limit)
        )
        if user_id is not None:
            stmt = stmt.where(IdentificationRequest.user_id == user_id)
        return list(self.session.scalars(stmt).all())

    def with_results(self, request_id: int) -> IdentificationRequest | None:
        """Запрос с предзагруженными результатами и предсказаниями."""
        stmt = (
            select(IdentificationRequest)
            .where(IdentificationRequest.id == request_id)
            .options(
                selectinload(IdentificationRequest.results),
                selectinload(IdentificationRequest.predicted_groups),
            )
        )
        return self.session.scalars(stmt).one_or_none()
