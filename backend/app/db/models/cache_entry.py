"""ORM-модель ``CacheEntry`` — запись кэша результатов идентификации (§5.10.1).

Ключ — SHA-256 hex предобработанного вектора интенсивностей (64 символа).
TTL — 7 суток, фоновое удаление просроченных записей — задача сервиса (§5.10.3).
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, Index, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class CacheEntry(Base):
    """Запись кэша: сериализованный JSON-результат идентификации по хэшу спектра."""

    __tablename__ = "cache_entry"

    key: Mapped[str] = mapped_column(String(64), primary_key=True)
    result_json: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False),
        nullable=False,
        server_default=func.current_timestamp(),
    )
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=False), nullable=False)

    __table_args__ = (Index("ix_cache_entry_expires_at", "expires_at"),)
