"""Связующая таблица M:N между ``Compound`` и ``FunctionalGroup`` (§5.9.2)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import ForeignKey, Integer
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base

if TYPE_CHECKING:
    from app.db.models.compound import Compound
    from app.db.models.functional_group import FunctionalGroup


class CompoundFunctionalGroup(Base):
    """Привязка функциональной группы к соединению с числом вхождений."""

    __tablename__ = "compound_functional_group"

    compound_id: Mapped[int] = mapped_column(
        ForeignKey("compound.id", ondelete="CASCADE"),
        primary_key=True,
    )
    functional_group_id: Mapped[int] = mapped_column(
        ForeignKey("functional_group.id", ondelete="CASCADE"),
        primary_key=True,
    )
    count: Mapped[int] = mapped_column(Integer, nullable=False, default=1)

    compound: Mapped[Compound] = relationship(back_populates="functional_group_links")
    functional_group: Mapped[FunctionalGroup] = relationship(back_populates="compound_links")
