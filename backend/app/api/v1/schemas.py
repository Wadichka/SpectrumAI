"""Pydantic-схемы запросов и ответов REST API v1 (§4.4.7 главы 4).

Все схемы — Pydantic v2 frozen-модели; описания попадают в OpenAPI и
используются фронтом для автогенерации TypeScript-типов.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.domain.dto import (
    CompoundCandidate,
    FunctionalGroupPrediction,
    GradCamPayload,
)

__all__ = [
    "ApiError",
    "BatchIdentificationItemResponse",
    "BatchIdentificationResponse",
    "CompoundCandidate",
    "CompoundDetailResponse",
    "CompoundSummary",
    "FunctionalGroupPrediction",
    "GradCamPayload",
    "HealthResponse",
    "HistoryEntryResponse",
    "IdentificationResponse",
    "PaginatedCompoundsResponse",
    "PaginatedHistoryResponse",
    "SpectrumCreatedResponse",
]


class HealthResponse(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    status: Literal["ok"]
    version: str


class ApiError(BaseModel):
    """Унифицированный объект ошибки в `detail`."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    code: str
    message: str
    details: dict[str, str] | None = None


class IdentificationResponse(BaseModel):
    """Ответ POST /api/v1/identify."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    request_id: int | None = None
    predictions: list[FunctionalGroupPrediction]
    candidates: list[CompoundCandidate]
    gradcam: GradCamPayload | None = None
    spectrum_length: int = Field(ge=1)
    model_versions: dict[str, str]
    threshold_mode: str
    processing_time_ms: int = Field(ge=0)
    timestamp: datetime


class BatchIdentificationItemResponse(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    filename: str
    status: Literal["success", "error"]
    result: IdentificationResponse | None = None
    error: ApiError | None = None


class BatchIdentificationResponse(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    items: list[BatchIdentificationItemResponse]
    total_processing_time_ms: int = Field(ge=0)


class CompoundSummary(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: int
    name: str | None
    smiles: str
    formula: str | None
    cas_number: str | None


class CompoundDetailResponse(CompoundSummary):
    iupac_name: str | None
    inchi: str | None
    inchi_key: str | None
    molecular_weight: float | None
    functional_groups: list[str]
    spectra_count: int = Field(ge=0)


class PaginatedCompoundsResponse(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    data: list[CompoundSummary]
    page: int = Field(ge=1)
    size: int = Field(ge=1)
    total: int = Field(ge=0)


class HistoryEntryResponse(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    request_id: int
    timestamp: datetime
    status: str
    processing_time_ms: int | None
    input_filename: str | None
    top_predicted_groups: list[str]


class PaginatedHistoryResponse(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    data: list[HistoryEntryResponse]
    page: int = Field(ge=1)
    size: int = Field(ge=1)
    total: int = Field(ge=0)


class SpectrumCreatedResponse(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    spectrum_id: int
    compound_id: int
    status: Literal["created"]
