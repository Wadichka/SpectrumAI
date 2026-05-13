"""ORM-модель ``ModelVersion`` — реестр обученных моделей (§5.9.2).

Обеспечивает прослеживаемость и откат к предыдущей версии модели.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, String, Text, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class ModelVersion(Base):
    """Запись о конкретной обученной модели: тип, версия, путь, метрики."""

    __tablename__ = "model_version"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    model_type: Mapped[str] = mapped_column(String(32), nullable=False)
    version: Mapped[str] = mapped_column(String(32), nullable=False)
    file_path: Mapped[str] = mapped_column(Text, nullable=False)
    metrics_json: Mapped[str | None] = mapped_column(Text)
    trained_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False),
        nullable=False,
        server_default=func.current_timestamp(),
    )

    __table_args__ = (UniqueConstraint("model_type", "version", name="uq_model_type_version"),)
