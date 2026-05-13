"""ORM-модель ``Compound`` — органическое соединение (§5.9.2)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import Float, Index, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin

if TYPE_CHECKING:
    from app.db.models.compound_functional_group import CompoundFunctionalGroup
    from app.db.models.spectrum import Spectrum


class Compound(TimestampMixin, Base):
    """Органическое соединение: метаданные структурной идентификации."""

    __tablename__ = "compound"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    name: Mapped[str | None] = mapped_column(String(255))
    iupac_name: Mapped[str | None] = mapped_column(String(512))
    cas_number: Mapped[str | None] = mapped_column(String(15))
    smiles_canonical: Mapped[str] = mapped_column(Text, nullable=False)
    inchi: Mapped[str] = mapped_column(Text, nullable=False)
    inchi_key: Mapped[str] = mapped_column(String(27), nullable=False, unique=True)
    molecular_formula: Mapped[str | None] = mapped_column(String(64))
    molecular_weight: Mapped[float | None] = mapped_column(Float)

    spectra: Mapped[list[Spectrum]] = relationship(
        back_populates="compound",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    functional_group_links: Mapped[list[CompoundFunctionalGroup]] = relationship(
        back_populates="compound",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    __table_args__ = (
        Index("ix_compound_inchi_key", "inchi_key", unique=True),
        Index("ix_compound_cas", "cas_number"),
        Index("ix_compound_name", "name"),
    )
