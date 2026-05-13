"""ORM-модель ``FunctionalGroup`` — справочник функциональных групп (§5.9.2).

Содержание фиксированное: 25 строк из таблицы 5.3 главы 5; заполняется
seed-блоком первой Alembic-миграции.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base

if TYPE_CHECKING:
    from app.db.models.compound_functional_group import CompoundFunctionalGroup


class FunctionalGroup(Base):
    """Описание целевой функциональной группы для multi-label классификации."""

    __tablename__ = "functional_group"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    code: Mapped[str] = mapped_column(String(8), nullable=False, unique=True)
    name: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    smarts_pattern: Mapped[str] = mapped_column(Text, nullable=False)
    description: Mapped[str | None] = mapped_column(Text)
    characteristic_bands: Mapped[str | None] = mapped_column(Text)

    compound_links: Mapped[list[CompoundFunctionalGroup]] = relationship(
        back_populates="functional_group",
        passive_deletes=True,
    )
