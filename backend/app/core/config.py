"""Настройки приложения SpectrumAI.

Читаются через ``pydantic-settings`` из переменных окружения и файла ``.env``
(см. ``backend/.env.example``).
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Конфигурация приложения, неизменяемая на время процесса."""

    database_url: str = Field(
        default="postgresql+psycopg://spectrumai:spectrumai@localhost:5432/spectrumai",
        description="DSN основной реляционной БД (PostgreSQL в dev/prod; SQLite в unit-тестах).",
    )
    redis_url: str = Field(
        default="redis://localhost:6379/0",
        description="DSN кэша Redis.",
    )
    cors_origins: list[str] = Field(
        default_factory=lambda: ["http://localhost:5173"],
        description="Разрешённые origin'ы для CORS (фронтенд Vite).",
    )
    log_level: str = Field(default="INFO", description="Уровень логирования.")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Возвращает закэшированный экземпляр настроек.

    Кэш через ``lru_cache`` гарантирует единственность экземпляра в процессе
    и совместим с FastAPI-зависимостями.
    """
    return Settings()
