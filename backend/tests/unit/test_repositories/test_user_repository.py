"""Тесты ``UserRepository``."""

from __future__ import annotations

import pytest
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.db.models.user import User
from app.db.repositories.user import UserRepository


def _make_user(**overrides: object) -> User:
    defaults: dict[str, object] = {
        "username": "vadim",
        "email": "v260205@gmail.com",
        "role": "user",
        "password_hash": "$2b$12$" + "x" * 53,
    }
    defaults.update(overrides)
    return User(**defaults)


def test_add_user(db_session: Session) -> None:
    repo = UserRepository(db_session)
    user = repo.add(_make_user())
    assert user.id is not None


def test_get_by_email(db_session: Session) -> None:
    repo = UserRepository(db_session)
    repo.add(_make_user())
    found = repo.get_by_email("v260205@gmail.com")
    assert found is not None
    assert found.username == "vadim"


def test_get_by_username(db_session: Session) -> None:
    repo = UserRepository(db_session)
    repo.add(_make_user())
    assert repo.get_by_username("vadim") is not None
    assert repo.get_by_username("ghost") is None


def test_unique_email_constraint(db_session: Session) -> None:
    repo = UserRepository(db_session)
    repo.add(_make_user())
    with pytest.raises(IntegrityError):
        repo.add(_make_user(username="another"))
