"""ORM-модель ``SpectrumEmbedding`` — векторное представление спектра (§5.9.2).

Связь 1:1 со ``Spectrum``: ``spectrum_id`` одновременно первичный и внешний
ключ. Сам вектор хранится сериализованным (``LargeBinary``) — это портируемо
между PostgreSQL и SQLite; фактический ANN-поиск идёт через FAISS, см. §5.9.1.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import ForeignKey, LargeBinary, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin

if TYPE_CHECKING:
    from app.db.models.spectrum import Spectrum


class SpectrumEmbedding(TimestampMixin, Base):
    """Эмбеддинг спектра, вычисленный нейросетевым энкодером."""

    __tablename__ = "spectrum_embedding"

    spectrum_id: Mapped[int] = mapped_column(
        ForeignKey("spectrum.id", ondelete="CASCADE"),
        primary_key=True,
    )
    embedding_vector: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    model_version: Mapped[str] = mapped_column(String(32), nullable=False)

    spectrum: Mapped[Spectrum] = relationship(back_populates="embedding")
