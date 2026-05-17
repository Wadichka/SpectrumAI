"""Health-check эндпоинт (§4.4.7)."""

from __future__ import annotations

from fastapi import APIRouter

from app.api.v1.schemas import HealthResponse

router = APIRouter(tags=["system"])


@router.get("/health", response_model=HealthResponse, summary="Проверка работоспособности")
async def health() -> HealthResponse:
    return HealthResponse(status="ok", version="0.1.0")
