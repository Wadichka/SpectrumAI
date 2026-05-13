"""Тесты ``CacheEntryRepository``."""

from __future__ import annotations

from datetime import datetime, timedelta

from sqlalchemy.orm import Session

from app.db.models.cache_entry import CacheEntry
from app.db.repositories.cache_entry import CacheEntryRepository


def _sha256_hex(seed: int) -> str:
    """Возвращает корректный 64-символьный hex-идентификатор (см. §5.10.1)."""
    return f"{seed:064x}"


def test_get_by_key(db_session: Session) -> None:
    repo = CacheEntryRepository(db_session)
    key = _sha256_hex(1)
    repo.add(
        CacheEntry(
            key=key,
            result_json='{"compound_id": 42}',
            expires_at=datetime(2030, 1, 1),
        )
    )
    found = repo.get_by_key(key)
    assert found is not None
    assert found.result_json == '{"compound_id": 42}'


def test_delete_expired_removes_only_expired(db_session: Session) -> None:
    repo = CacheEntryRepository(db_session)
    now = datetime(2026, 5, 13, 12, 0, 0)
    repo.add(
        CacheEntry(
            key=_sha256_hex(2),
            result_json="{}",
            expires_at=now - timedelta(days=1),  # просрочена
        )
    )
    repo.add(
        CacheEntry(
            key=_sha256_hex(3),
            result_json="{}",
            expires_at=now + timedelta(days=1),  # ещё валидна
        )
    )
    db_session.flush()

    removed = repo.delete_expired(now=now)
    db_session.flush()
    assert removed == 1
    assert repo.count() == 1
