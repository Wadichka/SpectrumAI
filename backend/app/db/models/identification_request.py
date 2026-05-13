"""ORM-модель ``IdentificationRequest`` — пользовательский запрос (§5.9.2, §5.10.3)."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import DateTime, ForeignKey, Index, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base

if TYPE_CHECKING:
    from app.db.models.identification_result import IdentificationResult
    from app.db.models.predicted_functional_group import PredictedFunctionalGroup
    from app.db.models.user import User


class IdentificationRequest(Base):
    """Факт обращения пользователя за идентификацией соединения по спектру."""

    __tablename__ = "identification_request"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    user_id: Mapped[int | None] = mapped_column(
        ForeignKey("user.id", ondelete="SET NULL"),
    )
    input_spectrum_path: Mapped[str] = mapped_column(Text, nullable=False)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=False),
        nullable=False,
        server_default=func.current_timestamp(),
    )
    processing_time_ms: Mapped[int | None] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="pending")

    user: Mapped[User | None] = relationship(back_populates="requests")
    results: Mapped[list[IdentificationResult]] = relationship(
        back_populates="request",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="IdentificationResult.rank",
    )
    predicted_groups: Mapped[list[PredictedFunctionalGroup]] = relationship(
        back_populates="request",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    __table_args__ = (
        Index("ix_request_timestamp", "timestamp"),
        Index("ix_request_user", "user_id"),
    )
