"""Интеграционные тесты /api/v1/settings (UC-07)."""

from __future__ import annotations

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_get_settings_returns_defaults(async_client: AsyncClient) -> None:
    response = await async_client.get("/api/v1/settings")
    assert response.status_code == 200
    body = response.json()
    assert body["language"] == "ru"
    assert body["top_k"] == 10
    assert body["threshold"] == pytest.approx(0.5)
    assert body["include_gradcam"] is True


@pytest.mark.asyncio
async def test_patch_settings_echoes_payload(async_client: AsyncClient) -> None:
    payload = {
        "language": "en",
        "include_gradcam": False,
        "top_k": 25,
        "threshold": 0.7,
        "baseline_method": "none",
        "normalize_method": "minmax",
        "savgol_window": 15,
        "savgol_polyorder": 3,
    }
    response = await async_client.patch("/api/v1/settings", json=payload)
    assert response.status_code == 200
    assert response.json() == payload


@pytest.mark.asyncio
async def test_patch_settings_rejects_invalid_top_k(async_client: AsyncClient) -> None:
    response = await async_client.patch(
        "/api/v1/settings",
        json={"top_k": 0},  # min — 1
    )
    assert response.status_code == 422
