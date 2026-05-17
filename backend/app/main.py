"""Точка входа FastAPI-приложения SpectrumAI."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1 import api_v1_router
from app.api.v1.exception_handlers import register_exception_handlers
from app.core.config import get_settings

settings = get_settings()

app: FastAPI = FastAPI(
    title="SpectrumAI API",
    version="0.1.0",
    description="API для распознавания органических соединений по ИК-спектрам.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.cors_origins),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

register_exception_handlers(app)

app.include_router(api_v1_router, prefix="/api/v1")


@app.get("/health", tags=["system"], include_in_schema=False)
async def health_alias() -> dict[str, str]:
    """Совместимость с deployment-чеками без префикса ``/api/v1``."""
    return {"status": "ok"}
