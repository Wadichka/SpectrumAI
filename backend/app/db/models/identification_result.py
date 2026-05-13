"""ORM-модель ``IdentificationResult`` — кандидат-соединение в результате запроса (§5.9.2).

Поле ``compound_name_cached`` — допущенная денормализация по §5.9.4 для быстрого
отображения списка истории без JOIN. Обновляется кодом вызывающей стороны
при сохранении результата (триггеры БД не используем — это не портируемо).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import Float, ForeignKey, Index, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base

if TYPE_CHECKING:
    from app.db.models.compound import Compound
    from app.db.models.identification_request import IdentificationRequest


class IdentificationResult(Base):
    """Одно соединение-кандидат с рангом и значением меры сходства."""

    __tablename__ = "identification_result"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    request_id: Mapped[int] = mapped_column(
        ForeignKey("identification_request.id", ondelete="CASCADE"),
        nullable=False,
    )
    compound_id: Mapped[int] = mapped_column(
        ForeignKey("compound.id"),
        nullable=False,
    )
    rank: Mapped[int] = mapped_column(Integer, nullable=False)
    score: Mapped[float] = mapped_column(Float, nullable=False)
    method: Mapped[str] = mapped_column(String(32), nullable=False)
    compound_name_cached: Mapped[str | None] = mapped_column(String(255))

    request: Mapped[IdentificationRequest] = relationship(back_populates="results")
    compound: Mapped[Compound] = relationship()

    __table_args__ = (Index("ix_result_request", "request_id"),)
