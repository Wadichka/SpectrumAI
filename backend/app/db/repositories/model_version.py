"""Репозиторий сущности ``ModelVersion``."""

from __future__ import annotations

from sqlalchemy import select

from app.db.models.model_version import ModelVersion
from app.db.repositories.base import BaseRepository


class ModelVersionRepository(BaseRepository[ModelVersion]):
    """Доступ к реестру обученных моделей (§5.9.2)."""

    model = ModelVersion

    def latest_by_type(self, model_type: str) -> ModelVersion | None:
        """Последняя по времени обучения версия модели заданного типа."""
        stmt = (
            select(ModelVersion)
            .where(ModelVersion.model_type == model_type)
            .order_by(ModelVersion.trained_at.desc())
            .limit(1)
        )
        return self.session.scalars(stmt).one_or_none()
