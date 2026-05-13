"""Репозиторий сущности ``User``."""

from __future__ import annotations

from sqlalchemy import select

from app.db.models.user import User
from app.db.repositories.base import BaseRepository


class UserRepository(BaseRepository[User]):
    """Доступ к учётным записям пользователей (серверный режим, §5.9.2)."""

    model = User

    def get_by_email(self, email: str) -> User | None:
        """Поиск пользователя по уникальному адресу электронной почты."""
        stmt = select(User).where(User.email == email)
        return self.session.scalars(stmt).one_or_none()

    def get_by_username(self, username: str) -> User | None:
        """Поиск пользователя по уникальному имени."""
        stmt = select(User).where(User.username == username)
        return self.session.scalars(stmt).one_or_none()
