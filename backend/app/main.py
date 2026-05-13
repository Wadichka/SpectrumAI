"""Точка входа FastAPI-приложения SpectrumAI.

Минимальная заглушка этапа 0: создаёт экземпляр приложения, подключает CORS
для dev-фронтенда и эндпоинт проверки работоспособности ``GET /health``.
"""

from __future__ import annotations

from typing import Final

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

_DEV_FRONTEND_ORIGIN: Final[str] = "http://localhost:5173"

app: FastAPI = FastAPI(
    title="SpectrumAI API",
    version="0.1.0",
    description="API для распознавания органических соединений по ИК-спектрам.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[_DEV_FRONTEND_ORIGIN],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", tags=["system"])
async def health() -> dict[str, str]:
    """Возвращает статус сервиса для проверок liveness/readiness."""
    return {"status": "ok"}
