"""DTO для возврата ML-инференса в API (§4.4.3, §4.4.5 главы 4).

Pydantic v2 frozen-модели — иммутабельный «контракт» между ``InferenceService``
и REST-роутерами (Этап 9). Не путать с ORM-моделью
``app.db.models.identification.IdentificationResult`` — та хранит историю
запросов в БД и заполняется отдельно после возврата DTO.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class FunctionalGroupPrediction(BaseModel):
    """Предсказание одной функциональной группы (§4.4.3, §4.4.5)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    code: str = Field(description="код группы в стиле FG01..FG25")
    name: str = Field(description="имя группы (alcohol_OH, ...)")
    probability: float = Field(ge=0.0, le=1.0)
    threshold: float = Field(ge=0.0, le=1.0)
    predicted: bool


class CompoundCandidate(BaseModel):
    """Кандидат-соединение из FAISS-ретривала с результатом cross-validation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    rank: int = Field(ge=1, description="ранг в выдаче, 1 — самый похожий")
    compound_id: int
    smiles: str
    name: str | None = None
    formula: str | None = None
    cas_number: str | None = None
    score: float = Field(
        description="cosine similarity (inner product на L2-нормированных эмбеддингах)"
    )
    consistent: bool = Field(
        description="True если cross-validation помечает кандидата как согласованного"
    )
    jaccard: float = Field(ge=0.0, le=1.0)
    matched_groups: tuple[str, ...] = Field(default=())
    missing_groups: tuple[str, ...] = Field(default=())
    extra_groups: tuple[str, ...] = Field(default=())


class GradCamPayload(BaseModel):
    """Активационная карта Grad-CAM для одной функциональной группы (§6.10)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    group_code: str
    group_name: str
    values: list[float]


class IdentificationResult(BaseModel):
    """Полный результат идентификации (DTO API)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    predictions: list[FunctionalGroupPrediction]
    candidates: list[CompoundCandidate]
    spectrum_length: int = Field(ge=1)
    model_versions: dict[str, str]
    threshold_mode: str
    processing_time_ms: int = Field(ge=0)
    timestamp: datetime
    gradcam: GradCamPayload | None = None
    # Обработанные интенсивности (длина = spectrum_length). На phase 1 backend
    # отдаёт её, чтобы ResultsPage мог нарисовать спектр без дополнительного
    # запроса; raw-данные не хранятся.
    spectrum: list[float] | None = None


__all__ = [
    "CompoundCandidate",
    "FunctionalGroupPrediction",
    "GradCamPayload",
    "IdentificationResult",
]
