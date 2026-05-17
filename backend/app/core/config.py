"""Настройки приложения SpectrumAI.

Читаются через ``pydantic-settings`` из переменных окружения и файла ``.env``
(см. ``backend/.env.example``).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

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

    # Пути к ML-артефактам (Этап 8, §4.4.3). Относительны cwd при старте FastAPI;
    # в Docker задаются через env-переменные ML_*.
    ml_contrastive_checkpoint: Path = Field(
        default=Path("models/ircnn-contrastive-0.2.0/best.pt"),
        description="Чекпойнт двухбашенной модели (CNN + SpectrumTower).",
    )
    ml_cnn_checkpoint: Path = Field(
        default=Path("models/ircnn-multilabel-0.1.0/best.pt"),
        description="Fallback-чекпойнт multi-label CNN (без проекции и FAISS).",
    )
    ml_faiss_root: Path = Field(
        default=Path("models/faiss/ircnn-contrastive-0.2.0"),
        description="Каталог FAISS-индекса (index.faiss + mapping.json).",
    )
    ml_default_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Порог классификации, если чекпойнт не содержит per-class порогов.",
    )
    ml_top_k_default: int = Field(
        default=10,
        ge=1,
        description="Top-K кандидатов retrieval'а по умолчанию.",
    )
    ml_device: str = Field(
        default="cpu",
        description="Устройство инференса (cpu | cuda).",
    )
    ml_consistency_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Порог Jaccard для cross-validation кандидата (§4.4.5).",
    )

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
