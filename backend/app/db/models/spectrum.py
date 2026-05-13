"""ORM-модель ``Spectrum`` — инфракрасный спектр соединения (§5.9.2, §5.9.3)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import Float, ForeignKey, Index, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin

if TYPE_CHECKING:
    from app.db.models.compound import Compound
    from app.db.models.spectrum_embedding import SpectrumEmbedding


class Spectrum(TimestampMixin, Base):
    """Инфракрасный спектр, привязанный к соединению."""

    __tablename__ = "spectrum"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    compound_id: Mapped[int] = mapped_column(
        ForeignKey("compound.id", ondelete="CASCADE"),
        nullable=False,
    )
    source: Mapped[str] = mapped_column(String(32), nullable=False)
    phase: Mapped[str | None] = mapped_column(String(16))
    technique: Mapped[str | None] = mapped_column(String(32))
    wavenumber_min: Mapped[float] = mapped_column(Float, nullable=False)
    wavenumber_max: Mapped[float] = mapped_column(Float, nullable=False)
    n_points: Mapped[int] = mapped_column(Integer, nullable=False)
    file_path: Mapped[str] = mapped_column(Text, nullable=False)
    quality_score: Mapped[float | None] = mapped_column(Float)

    compound: Mapped[Compound] = relationship(back_populates="spectra")
    embedding: Mapped[SpectrumEmbedding | None] = relationship(
        back_populates="spectrum",
        cascade="all, delete-orphan",
        passive_deletes=True,
        uselist=False,
    )

    __table_args__ = (
        Index("ix_spectrum_compound", "compound_id"),
        Index("ix_spectrum_source", "source"),
    )
