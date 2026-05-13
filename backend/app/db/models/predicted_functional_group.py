"""ORM-модель ``PredictedFunctionalGroup`` — предсказанная классификатором группа (§5.9.2)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import Float, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base

if TYPE_CHECKING:
    from app.db.models.functional_group import FunctionalGroup
    from app.db.models.identification_request import IdentificationRequest


class PredictedFunctionalGroup(Base):
    """Предсказание функциональной группы для конкретного запроса с вероятностью."""

    __tablename__ = "predicted_functional_group"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    request_id: Mapped[int] = mapped_column(
        ForeignKey("identification_request.id", ondelete="CASCADE"),
        nullable=False,
    )
    functional_group_id: Mapped[int] = mapped_column(
        ForeignKey("functional_group.id"),
        nullable=False,
    )
    probability: Mapped[float] = mapped_column(Float, nullable=False)

    request: Mapped[IdentificationRequest] = relationship(back_populates="predicted_groups")
    functional_group: Mapped[FunctionalGroup] = relationship()
