"""ORM-модель ``User`` — учётная запись (только серверный режим, §5.9.2, §5.12).

Соответствует ER-диаграмме главы 5. Auth-эндпоинты, JWT и middleware в системе
не реализуются (CLAUDE.md §8 — авторизация и мультитенантность вне ТЗ);
модель хранится для согласованности схемы и возможности связать запрос
с пользователем при будущем расширении.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin

if TYPE_CHECKING:
    from app.db.models.identification_request import IdentificationRequest


class User(TimestampMixin, Base):
    """Учётная запись пользователя серверного режима."""

    __tablename__ = "user"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    username: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    email: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    role: Mapped[str] = mapped_column(String(16), nullable=False, default="user")
    password_hash: Mapped[str] = mapped_column(String(60), nullable=False)

    requests: Mapped[list[IdentificationRequest]] = relationship(
        back_populates="user",
        passive_deletes=True,
    )
