"""Репозиторий сущности ``FunctionalGroup``."""

from __future__ import annotations

from sqlalchemy import select

from app.db.models.functional_group import FunctionalGroup
from app.db.repositories.base import BaseRepository


class FunctionalGroupRepository(BaseRepository[FunctionalGroup]):
    """Доступ к справочнику функциональных групп (25 строк, §5.4.1)."""

    model = FunctionalGroup

    def get_by_code(self, code: str) -> FunctionalGroup | None:
        """Поиск по коду вида ``FG01``..``FG25``."""
        stmt = select(FunctionalGroup).where(FunctionalGroup.code == code)
        return self.session.scalars(stmt).one_or_none()

    def list_all(self) -> list[FunctionalGroup]:
        """Полный справочник, отсортированный по коду."""
        stmt = select(FunctionalGroup).order_by(FunctionalGroup.code)
        return list(self.session.scalars(stmt).all())
