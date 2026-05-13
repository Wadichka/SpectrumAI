"""Фабрика SQLAlchemy-сессий и FastAPI-зависимость ``get_db``."""

from __future__ import annotations

from collections.abc import Iterator

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import get_settings


def _build_engine() -> Engine:
    """Создаёт SQLAlchemy-движок на основании настроек приложения."""
    settings = get_settings()
    connect_args: dict[str, object] = {}
    if settings.database_url.startswith("sqlite"):
        # SQLite требует разрешения многопоточного доступа для тестов и CLI-утилит.
        connect_args["check_same_thread"] = False
    return create_engine(
        settings.database_url,
        pool_pre_ping=True,
        future=True,
        connect_args=connect_args,
    )


engine: Engine = _build_engine()

SessionLocal: sessionmaker[Session] = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
    expire_on_commit=False,
    class_=Session,
)


def get_db() -> Iterator[Session]:
    """FastAPI-зависимость, открывающая и закрывающая SQLAlchemy-сессию."""
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
