"""Агрегатор роутеров REST API v1."""

from __future__ import annotations

from fastapi import APIRouter

from app.api.v1.routers import compounds, health, history, identify, spectra

api_v1_router = APIRouter()
api_v1_router.include_router(health.router)
api_v1_router.include_router(identify.router)
api_v1_router.include_router(compounds.router)
api_v1_router.include_router(spectra.router)
api_v1_router.include_router(history.router)

__all__ = ["api_v1_router"]
