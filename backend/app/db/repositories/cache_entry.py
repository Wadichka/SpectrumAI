"""Репозиторий сущности ``CacheEntry``."""

from __future__ import annotations

from datetime import datetime
from typing import Any, cast

from sqlalchemy import delete
from sqlalchemy.engine import CursorResult

from app.db.models.cache_entry import CacheEntry
from app.db.repositories.base import BaseRepository


class CacheEntryRepository(BaseRepository[CacheEntry]):
    """Доступ к кэшу результатов идентификации (§5.10.1)."""

    model = CacheEntry

    def get_by_key(self, key: str) -> CacheEntry | None:
        """Получить запись по SHA-256-ключу (64 hex-символа)."""
        return self.session.get(CacheEntry, key)

    def delete_expired(self, *, now: datetime) -> int:
        """Удалить все записи с истекшим TTL; возвращает число удалённых.

        ``Session.execute`` стабах SA типизирован как ``Result[Any]``, но для DML
        фактически возвращает ``CursorResult`` с полем ``rowcount`` — cast здесь
        отражает рантайм-факт.
        """
        stmt = delete(CacheEntry).where(CacheEntry.expires_at <= now)
        result = cast(CursorResult[Any], self.session.execute(stmt))
        return int(result.rowcount or 0)
