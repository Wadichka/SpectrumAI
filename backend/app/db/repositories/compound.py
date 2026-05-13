"""Репозиторий сущности ``Compound``."""

from __future__ import annotations

from sqlalchemy import select

from app.db.models.compound import Compound
from app.db.repositories.base import BaseRepository


class CompoundRepository(BaseRepository[Compound]):
    """Доступ к данным органических соединений."""

    model = Compound

    def get_by_inchi_key(self, inchi_key: str) -> Compound | None:
        """Поиск соединения по уникальному ``inchi_key`` (см. §5.9.3)."""
        stmt = select(Compound).where(Compound.inchi_key == inchi_key)
        return self.session.scalars(stmt).one_or_none()

    def get_by_cas(self, cas_number: str) -> Compound | None:
        """Поиск соединения по CAS Registry Number."""
        stmt = select(Compound).where(Compound.cas_number == cas_number)
        return self.session.scalars(stmt).one_or_none()

    def search_by_name_prefix(self, prefix: str, *, limit: int = 20) -> list[Compound]:
        """Префиксный поиск по тривиальному имени соединения."""
        stmt = (
            select(Compound)
            .where(Compound.name.is_not(None))
            .where(Compound.name.istartswith(prefix))
            .order_by(Compound.name)
            .limit(limit)
        )
        return list(self.session.scalars(stmt).all())
