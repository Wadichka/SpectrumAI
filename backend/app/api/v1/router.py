"""Агрегатор роутеров REST API v1."""

from __future__ import annotations

from fastapi import APIRouter

from app.api.v1.routers import (
    compounds,
    functional_groups,
    health,
    history,
    identify,
    reports,
    settings,
    spectra,
)

api_v1_router = APIRouter()
api_v1_router.include_router(health.router)
api_v1_router.include_router(identify.router)
api_v1_router.include_router(compounds.router)
api_v1_router.include_router(spectra.router)
api_v1_router.include_router(history.router)
api_v1_router.include_router(settings.router)
api_v1_router.include_router(functional_groups.router)
api_v1_router.include_router(reports.router)

__all__ = ["api_v1_router"]
