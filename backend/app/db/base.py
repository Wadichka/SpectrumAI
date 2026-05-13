"""Декларативная база и общие миксины для SQLAlchemy-моделей."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Корневой класс декларативных моделей.

    Все ORM-сущности приложения наследуются от него и автоматически
    регистрируются в ``Base.metadata`` для Alembic-автогенерации.
    """


class TimestampMixin:
    """Добавляет поле ``created_at`` с серверным дефолтом."""

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False),
        nullable=False,
        server_default=func.current_timestamp(),
    )
