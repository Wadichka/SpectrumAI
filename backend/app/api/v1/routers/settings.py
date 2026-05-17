"""Эндпоинты настроек идентификации (UC-07, §7.5.7).

Phase 1: persistence держится на клиенте (localStorage в Zustand-store).
PATCH делает echo — валидирует и возвращает payload. Реальная запись в БД
(per-user) — phase 2, когда появится auth.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.v1.schemas import SettingsResponse
from app.core.config import Settings, get_settings

router = APIRouter(tags=["settings"])


@router.get(
    "/settings",
    response_model=SettingsResponse,
    summary="Получить дефолтные настройки (UC-07)",
)
def read_settings(settings: Settings = Depends(get_settings)) -> SettingsResponse:
    return SettingsResponse(
        top_k=settings.ml_top_k_default,
        threshold=settings.ml_default_threshold,
    )


@router.patch(
    "/settings",
    response_model=SettingsResponse,
    summary="Сохранить настройки (echo на phase 1)",
)
def update_settings(payload: SettingsResponse) -> SettingsResponse:
    # Phase 1: echo. Реальный persistence будет в phase 2 (user_settings + auth).
    return payload
