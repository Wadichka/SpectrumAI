"""Базовый репозиторий-паттерн для ORM-сущностей."""

from __future__ import annotations

from typing import Generic, TypeVar

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.db.base import Base
from app.domain.errors import EntityNotFoundError

T = TypeVar("T", bound=Base)


class BaseRepository(Generic[T]):
    """Универсальные CRUD-операции над одной ORM-сущностью.

    Репозитории не вызывают ``commit``/``rollback`` — это задача сервисного слоя.
    Метод ``add`` помещает сущность в сессию и делает ``flush`` для получения
    автогенерированных значений (например, первичного ключа).
    """

    model: type[T]

    def __init__(self, session: Session) -> None:
        self.session = session

    def get(self, entity_id: int) -> T | None:
        """Возвращает запись по первичному ключу или ``None``."""
        return self.session.get(self.model, entity_id)

    def get_or_raise(self, entity_id: int) -> T:
        """Возвращает запись или кидает ``EntityNotFoundError``."""
        instance = self.get(entity_id)
        if instance is None:
            raise EntityNotFoundError(self.model.__name__, entity_id)
        return instance

    def list(self, *, limit: int = 100, offset: int = 0) -> list[T]:
        """Возвращает страницу записей в порядке вставки (по первичному ключу)."""
        stmt = select(self.model).offset(offset).limit(limit)
        return list(self.session.scalars(stmt).all())

    def add(self, instance: T) -> T:
        """Добавляет сущность в сессию и выполняет ``flush``."""
        self.session.add(instance)
        self.session.flush()
        return instance

    def delete(self, instance: T) -> None:
        """Помечает сущность к удалению; фактическое удаление при ``commit``/``flush``."""
        self.session.delete(instance)

    def count(self) -> int:
        """Возвращает общее число записей в таблице."""
        stmt = select(func.count()).select_from(self.model)
        return int(self.session.scalar(stmt) or 0)
