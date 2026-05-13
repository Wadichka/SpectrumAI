"""Репозиторий сущности ``Spectrum``."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import selectinload

from app.db.models.spectrum import Spectrum
from app.db.repositories.base import BaseRepository


class SpectrumRepository(BaseRepository[Spectrum]):
    """Доступ к данным инфракрасных спектров."""

    model = Spectrum

    def list_by_compound_id(self, compound_id: int) -> list[Spectrum]:
        """Все спектры, привязанные к соединению."""
        stmt = select(Spectrum).where(Spectrum.compound_id == compound_id).order_by(Spectrum.id)
        return list(self.session.scalars(stmt).all())

    def get_with_embedding(self, spectrum_id: int) -> Spectrum | None:
        """Спектр вместе с предзагруженным эмбеддингом (1:1)."""
        stmt = (
            select(Spectrum)
            .where(Spectrum.id == spectrum_id)
            .options(selectinload(Spectrum.embedding))
        )
        return self.session.scalars(stmt).one_or_none()
